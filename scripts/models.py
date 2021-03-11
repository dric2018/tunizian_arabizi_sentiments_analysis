from dataset import DataSet
from config import Config

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import utils
import pandas as pd
import os
import sys
from dataset import DataModule

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-type',
    '-mt',
    type=str,
    default='lstm',
    help='Type of model architechture to use, one of lstm, gru, bert'
)


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = Config.embedding_dim,
        num_layers: int = Config.num_layers,
        bidirectional: bool = True,
        hidden_size: int = Config.hidden_size
    ):
        super(LSTMModel, self).__init__()
        self.save_hyperparameters()
        # print(self.hparams)
        self.best_acc = 0

        # architecture
        print(f'[INFO] Using {Config.base_model} as base model')
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
        # Embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size + 1,
            embedding_dim=self.hparams.embedding_dim
        )
        # self.embedding_layer.requires_grad = False
        # Reccurent layer(s)
        self.lstm = nn.LSTM(
            input_size=self.hparams.embedding_dim,
            num_layers=self.hparams.num_layers,
            hidden_size=self.hparams.hidden_size,
            dropout=Config.drop_out_prob,
            bidirectional=self.hparams.bidirectional,
            batch_first=True
        )
        self.drop_layer = nn.Dropout(p=Config.drop_out_prob)

        # classifier
        if self.hparams.bidirectional:
            self.fc = nn.Linear(
                in_features=2*self.hparams.hidden_size,
                out_features=Config.n_classes
            )
        else:
            self.fc = nn.Linear(
                in_features=self.hparams.hidden_size,
                out_features=Config.n_classes
            )

    def configure_optimizers(self):
        opt = th.optim.AdamW(
            lr=Config.lr,
            params=[p for p in self.parameters() if p.requires_grad],
            eps=Config.eps,
            weight_decay=Config.weight_decay
        )

        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='max',
            factor=0.1,
            patience=Config.early_stopping_patience,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=Config.eps,
            verbose=True,
        )
        return {"optimizer": opt,
                "lr_scheduler": scheduler,
                "monitor": "val_acc"}

    def forward(self, x: th.Tensor):

        bs = x.shape[0]
        if bs > 1:
            # squeeze extra dimension
            x = x.squeeze(1)

        emb = self.embedding_layer(x)
        # print('[INFO] emb shape : ', emb.shape)

        lstm_out, (h_n, c_n) = self.lstm(emb)
        # print('[INFO] lstm_out shape : ', lstm_out.shape)
        # print('[INFO] h_n shape : ', h_n.shape)
        # print('[INFO] c_n shape : ', c_n.shape)

        avg_pool = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        # print('[INFO] avg_pool shape : ', avg_pool.shape)
        # print('[INFO] max_pool shape : ', max_pool.shape)

        # concat
        # features = th.cat((avg_pool, max_pool), dim=1)
        features = avg_pool  # (avg_pool + max_pool) / 2
        # print('[INFO] features shape : ', features.shape)
        # apply classification layer
        out = th.sigmoid(self.fc(self.drop_layer(features)))
        # print('[INFO] out shape : ', out.shape)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['ids'], batch['target']
        # forward pass
        logits = self(x)
        # get predictions
        preds = logits.argmax(dim=1)
        # get loss
        train_loss = self.get_loss(logits=logits, targets=y)
        # get accuracy
        train_acc = self.get_acc(preds=preds, targets=y)

        self.log('train_acc',
                 train_acc,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)

        return {'loss': train_loss,
                'accuracy': train_acc,
                "predictions": preds,
                'targets': y
                }

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch['ids'], batch['target']
        # forward pass
        logits = self(x)
        # get predictions
        preds = logits.argmax(dim=1)
        # get loss
        val_loss = self.get_loss(logits=logits, targets=y)
        # get accuracy
        val_acc = self.get_acc(preds=preds, targets=y)

        self.log('val_acc',
                 val_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        self.log('val_loss',
                 val_loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        return {'loss': val_loss,
                'accuracy': val_acc,
                "predictions": preds,
                'targets': y
                }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch
        )

        self.logger.experiment.add_scalar(
            "Accuracy/Validation",
            avg_acc,
            self.current_epoch
        )
        # monitor acc improvements
        if avg_acc > self.best_acc:
            print("\n")
            print(
                f'[INFO] accuracy improved from {self.best_acc} to {avg_acc}'
            )
            self.best_acc = avg_acc
            print()
        else:
            print("\n")
            print('[INFO] accuracy did not improve')
            print()

    def get_acc(self, preds, targets):
        bs = targets.shape[0]
        if bs > 1:
            # squeeze extra dimension
            targets = targets.squeeze(1)
        return (preds == targets.argmax(dim=1)).float().mean()

    def get_loss(self, logits, targets):
        bs = targets.shape[0]
        if bs > 1:
            # squeeze extra dimension
            targets = targets.squeeze(1)
        return nn.BCELoss(weight=None)(logits.cpu(), targets.cpu())


class GRUModel(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = Config.embedding_dim,
        num_layers: int = Config.num_layers,
        bidirectional: bool = True,
        hidden_size: int = 128
    ):
        super(GRUModel, self).__init__()
        self.save_hyperparameters()
        # print(self.hparams)

        # architecture
        print(f'[INFO] Using {Config.base_model} as base model')
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
        # Embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size + 1,
            embedding_dim=self.hparams.embedding_dim
        )
        # self.embedding_layer.requires_grad = False
        # Reccurent layer(s)
        self.gru = nn.GRU(
            input_size=self.hparams.embedding_dim,
            num_layers=self.hparams.num_layers,
            hidden_size=self.hparams.hidden_size,
            dropout=Config.drop_out_prob,
            bidirectional=self.hparams.bidirectional,
            batch_first=True
        )

        self.drop_layer = nn.Dropout(p=Config.drop_out_prob)
        # classifier
        if self.hparams.bidirectional:
            self.fc = nn.Linear(
                in_features=2*self.hparams.hidden_size,
                out_features=Config.n_classes
            )
        else:
            self.fc = nn.Linear(
                in_features=self.hparams.hidden_size,
                out_features=Config.n_classes
            )

    def configure_optimizers(self):
        opt = th.optim.Adam(
            lr=Config.lr,
            params=[p for p in self.parameters() if p.requires_grad],
            # eps=Config.eps,
            # weight_decay=Config.weight_decay
        )

        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='max',
            factor=0.1,
            patience=Config.early_stopping_patience,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=Config.eps,
            verbose=True,
        )
        return {"optimizer": opt,
                "lr_scheduler": scheduler,
                "monitor": "val_acc"}

    def forward(self, x: th.Tensor):

        bs = x.shape[0]
        if bs > 1:
            # squeeze extra dimension
            x = x.squeeze(1)

        emb = self.embedding_layer(x)
        # print('[INFO] emb shape : ', emb.shape)

        gru_out, h_n = self.gru(emb)
        # print('[INFO] gru_out shape : ', gru_out.shape)
        # print('[INFO] h_n shape : ', h_n.shape)
        # print('[INFO] c_n shape : ', c_n.shape)

        avg_pool = gru_out.mean(dim=1)
        max_pool, _ = gru_out.max(dim=1)
        # print('[INFO] avg_pool shape : ', avg_pool.shape)
        # print('[INFO] max_pool shape : ', max_pool.shape)

        # concat
        # features = th.cat((avg_pool, max_pool), dim=1)
        features = avg_pool  # (avg_pool + max_pool) / 2
        # print('[INFO] features shape : ', features.shape)
        # apply classification layer

        out = th.sigmoid(self.fc(self.drop_layer(features)))
        # print('[INFO] out shape : ', out.shape)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['ids'], batch['target']
        # forward pass
        logits = self(x)
        # get predictions
        preds = logits.argmax(dim=1)
        # get loss
        train_loss = self.get_loss(logits=logits, targets=y)
        # get accuracy
        train_acc = self.get_acc(preds=preds, targets=y)

        self.log('train_acc',
                 train_acc,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)

        return {'loss': train_loss,
                'accuracy': train_acc,
                "predictions": preds,
                'targets': y
                }

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch['ids'], batch['target']
        # forward pass
        logits = self(x)
        # get predictions
        preds = logits.argmax(dim=1)
        # get loss
        val_loss = self.get_loss(logits=logits, targets=y)
        # get accuracy
        val_acc = self.get_acc(preds=preds, targets=y)

        self.log('val_acc',
                 val_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        self.log('val_loss',
                 val_loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        return {'loss': val_loss,
                'accuracy': val_acc,
                "predictions": preds,
                'targets': y
                }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch
        )

        self.logger.experiment.add_scalar(
            "Accuracy/Validation",
            avg_acc,
            self.current_epoch
        )

        # monitor acc improvements
        if avg_acc > self.best_acc:
            print(
                f'[INFO] accuracy improved from {self.best_acc} to {avg_acc}')
            self.best_acc = avg_acc

    def get_acc(self, preds, targets):
        bs = targets.shape[0]
        if bs > 1:
            # squeeze extra dimension
            targets = targets.squeeze(1)
        return (preds == targets.argmax(dim=1)).float().mean()

    def get_loss(self, logits, targets):
        bs = targets.shape[0]
        if bs > 1:
            # squeeze extra dimension
            targets = targets.squeeze(1)
        return nn.BCELoss(weight=None)(logits.cpu(), targets.cpu())


class BertBaseModel(pl.LightningModule):
    def __init__(
            self,):
        super(BertBaseModel, self).__init__()
        self.save_hyperparameters()
        # print(self.hparams)

        # architecture
        print(f'[INFO] Using {Config.base_model} as base model')

        # Encoder
        self.encoder = AutoModel.from_pretrained(Config.base_model)
        # classifier

        self.fc = nn.Linear(
            in_features=768,  # bert output features dim=768 so setting it as default
            out_features=Config.n_classes
        )

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        opt = th.optim.AdamW(
            params=optimizer_grouped_parameters,
            lr=Config.lr
        )

        scheduler = th.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=utils.ramp_scheduler
        )

        return [opt], [scheduler]

    def forward(self, x: th.Tensor, mask=None):

        bs = x.shape[0]
        if bs > 1:
            # squeeze extra dimension
            x = x.squeeze(1)

        # features extraction
        h_n, _ = self.encoder(x)
        features = th.max(h_n[:, 1], axis=1)
        # print('[INFO] features shape : ', features.shape)
        # apply classification layer
        out = th.sigmoid(self.fc(features))
        # print('[INFO] out shape : ', out.shape)

        return out

    def training_step(self, batch, batch_idx):
        x, mask, y = batch['ids'], batch['mask'], batch['target']
        # forward pass
        logits = self(x, mask=mask)
        # get predictions
        preds = logits.argmax(dim=1)
        # get loss
        train_loss = self.get_loss(logits=logits, targets=y)
        # get accuracy
        train_acc = self.get_acc(preds=preds, targets=y)

        self.log('train_acc',
                 train_acc,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)

        return {'loss': train_loss,
                'accuracy': train_acc,
                "predictions": preds,
                'targets': y
                }

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch['ids'], batch['mask'], batch['target']
        # forward pass
        logits = self(x, mask=mask)
        # get predictions
        preds = logits.argmax(dim=1)
        # get loss
        val_loss = self.get_loss(logits=logits, targets=y)
        # get accuracy
        val_acc = self.get_acc(preds=preds, targets=y)

        self.log('val_acc',
                 val_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        return {'loss': val_loss,
                'accuracy': val_acc,
                "predictions": preds,
                'targets': y
                }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch
        )

        self.logger.experiment.add_scalar(
            "Accuracy/Validation",
            avg_acc,
            self.current_epoch
        )

    def get_acc(self, preds, targets):
        bs = targets.shape[0]
        if bs > 1:
            # squeeze extra dimension
            targets = targets.squeeze(1)
        return (preds == targets.argmax(dim=1)).float().mean()

    def get_loss(self, logits, targets):
        bs = targets.shape[0]
        if bs > 1:
            # squeeze extra dimension
            targets = targets.squeeze(1)
        return nn.BCELoss(weight=None)(logits.cpu(), targets.cpu())


if __name__ == "__main__":
    args = parser.parse_args()

    txt = "owel mara n7es eli echa3eb etounsi togrih il o3oud \
    il kadhba eli 7achtou bech i7asen men bladou rahou men \
    9bal elenti5abet b barcha bech ta3mel ki 8irek ta7ki w \
    tfareh la3bed w ki tosel lli t7eb 3lih haka lwa9et 9abloni \
    ken 3mel 7ata 7aja barka meli 9alhoum elkoul taw y8rouf w yrawa7"

    tgt = th.eye(3)[0].view(1, -1)

    print('[INFO] Building model')
    # tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
    try:
        if args.model_type.lower() == 'lstm':
            model = LSTMModel()
        elif args.model_type.lower() == 'gru':
            model = GRUModel()
        else:
            model = BertBaseModel()
        # print('[INFO] Model built')
        # print(model)

        # print('[INFO] Getting tokens')

        # code = tokenizer.encode_plus(
        #     text=txt,
        #     padding='max_length',
        #     max_length=Config.max_len,
        #     truncation=True,
        #     return_tensors='pt'
        # )

        # ids = code['input_ids']
        # mask = code['attention_mask']

        # sys.exit()

        print('[INFO] Loading some data')
        print("[INFO] Reading dataframe")
        train_df = pd.read_csv(os.path.join(
            Config.data_dir, 'Train.csv'), nrows=10000)

        print("[INFO] Building data module")
        dm = DataModule(
            df=train_df,
            frac=1,
            train_batch_size=Config.train_batch_size,
            test_batch_size=Config.test_batch_size,
            test_size=.15
        )

        dm.setup()

        for batch in dm.val_dataloader():
            ids, mask, targets = batch['ids'], batch['mask'], batch['target']
            print('[INFO] input_ids shape :', ids.shape)
            print('[INFO] Attention mask shape :', mask.shape)
            print('[INFO] Targets shape :', targets.shape)

            try:

                print('[INFO] Forward pass')
                try:
                    logits = model(
                        src=ids,
                        mask=mask
                    )
                except Exception as e:
                    logits = model(
                        x=ids
                    )

                    print(e)
                    logits = model(ids)
                    print("[INFO] logits shape : ", logits.shape)
                    # print("[INFO] Logits : ", logits)

                    print("[INFO] Target shape : ", targets.shape)
                    # print("[INFO] Target : ", targets)

                    preds = th.argmax(input=logits, dim=-1)
                    print("[INFO] preds shape : ", preds.shape)
                    print("[INFO] preds : ", preds)

                    print("[INFO] Computing accuracy")
                    acc = model.get_acc(preds=preds, targets=targets)

                    print("[INFO] Computing loss")
                    loss = model.get_loss(logits=logits, targets=targets)

                    print("[INFO] acc : ", acc)
                    print("[INFO] loss : ", loss)

            except Exception as e:
                print(e)
            break

    except Exception as e:
        print(f'[ERROR] {e}')
