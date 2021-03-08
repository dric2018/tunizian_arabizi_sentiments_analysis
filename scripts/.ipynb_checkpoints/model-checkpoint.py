from dataset import DataSet
from config import Config

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from utils import ramp_scheduler

import pandas as pd
import os
import sys
from dataset import DataModule


# * Encoder/Features extractor: bert/roberta/distilbert multilingual
# * Decoder/Classifier : linear layer (bert_out_features, n_classes)
# * loss_fn : CrossEntropyLoss
# * Optimize metrics : accuracy
# * Optimizer : Adam, AdamW, SGD
# * learning rate : (3e-5...1e-1)
# * lr scheduler : linear with warmup, ReduceLROnPlateau
# * pretrained : Always true


class Model(pl.LightningModule):
    def __init__(
            self,
            class_w=None,
            max_len: int = Config.max_len,
            base_model: str = Config.base_model,
            freeze: bool = False):
        super(Model, self).__init__()

        self.save_hyperparameters()
        # load pretrained model from torchvision or anywheare else
        try:
            self.encoder = AutoModel.from_pretrained(
                Config.base_model,
                # Whether the model returns attentions weights.
                output_attentions=False,
                # Whether the model returns all hidden-states.
                output_hidden_states=False,
            )
            # print(self.encoder)
        except Exception as e:
            print(f'{e}')

        if "distil" in self.hparams.base_model:
            try:
                # get num output features from features extractor
                self.num_ftrs = self.encoder.transformer.layer[-1].ffn.lin2.out_features
            except:
                pass

        elif "roberta" in self.hparams.base_model:
            try:
                self.num_ftrs = self.encoder.pooler.dense.out_features
            except Exception as e:
                print(f'{e}')

        else:
            try:
                self.num_ftrs = self.encoder.pooler.dense.out_features
            except Exception as e:
                print(f'{e}')

        if self.hparams.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=.35)
        if "distilbert" in self.hparams.base_model:
            self.pooler = nn.Linear(
                in_features=self.num_ftrs,
                out_features=self.num_ftrs
            )
            self.decoder = nn.Linear(
                in_features=self.num_ftrs,
                out_features=Config.n_classes
            )

        else:
            self.decoder = nn.Linear(
                in_features=self.num_ftrs,
                out_features=Config.n_classes
            )

        #########
        # methods
        #########

    def forward(self, ids, mask, targets=None):
        bs = ids.size(0)
        # extract features
        try:
            outputs = self.encoder(ids.squeeze(1), mask.squeeze(1))
            last_hidden_state = outputs.last_hidden_state
            out = outputs.pooler_output
            # print(out)
        except:
            outputs = self.encoder(ids.squeeze(1), mask.squeeze(1))
            # print(outputs)
            out = outputs.last_hidden_state[:, 0]
            out = self.pooler(out)

        # apply dropout
        out = self.dropout(out)
        # apply classifier
        out = self.decoder(out)

        return out

    def configure_optimizers(self):
        param_optimizer = list(self.encoder.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.001
             },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
             },
        ]

        opt = th.optim.SGD(
            optimizer_parameters,
            lr=Config.lr
        )

        scheduler = th.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=ramp_scheduler,
            verbose=True
        )

        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        ids, mask, targets = batch['ids'], batch['mask'], batch['target']
        # make predictions
        logits = self(ids, mask)

        # compute metrics
        # loss
        loss = self.get_loss(preds=logits, targets=targets)

        # accuracy
        preds = th.argmax(input=logits, dim=-1)
        acc = self.get_acc(preds=preds, targets=targets)

        # logging stuff
        self.log('train_acc', acc, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": logits.argmax(dim=1),
                'targets': targets}

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
        ids, mask, targets = batch['ids'], batch['mask'], batch['target']
        # make predictions
        logits = self(ids, mask)

        # compute metrics
        # loss
        loss = self.get_loss(preds=logits, targets=targets)

        # accuracy
        preds = th.argmax(input=logits, dim=-1)
        acc = self.get_acc(preds=preds, targets=targets)

        # logging stuff
        self.log('val_acc', acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": logits.argmax(dim=1),
                'targets': targets}

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def predict(self, dataloader, batch_size=1):
        if batch_size == 1:
            try:
                preds = self(dataloader.unsqueeze(0))
            except:
                preds = self(dataloader)
        else:
            preds = self(dataloader)

        return preds.detach().cpu().numpy().flatten()

    def get_loss(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return nn.CrossEntropyLoss(weight=self.hparams.class_w)(input=preds, target=targets)

    def get_acc(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return (preds == targets).float().mean()


class Model1(pl.LightningModule):
    def __init__(self,
                 tokenizer,
                 class_w=None,
                 dv: str = 'cuda',
                 input_size=Config.max_len,
                 sequence_len=Config.max_len,
                 hidden_size=Config.hidden_size,
                 dropout_prob=Config.drop_out_prob,
                 embedding_dim=Config.embedding_dim,
                 num_layers=Config.num_layers):
        super(Model1, self).__init__()

        d = dict(Config.__dict__)
        conf_dict = {k: d[k] for k in d.keys() if '__' not in k}
        self.save_hyperparameters(conf_dict)

        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        self.class_w = class_w
        self.num_layers = num_layers
        self.dv = dv

        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size+1,
            embedding_dim=self.sequence_len
        )
        self.encoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.dropout_prob,
            bidirectional=True
        )

        # create classification layer

        self.classifier = nn.Linear(
            in_features=2*self.hidden_size,
            out_features=Config.n_classes
        )
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, seq):
        """
        inspired from : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
        """
        # flatten parameters
        # self.encoder.flatten_parameters()
        # compute embedding
        out = self.embedding(seq.squeeze(1))
        # print('emb shape : ', out.shape)
        # compute init hidden size
        h0 = th.zeros(self.num_layers * 2,  # if bidirectional multiply num_layers by 2
                      seq.size(0),
                      self.hidden_size).to(self.dv)
        # print('h0 shape : ', h0.shape)
        last_hidden_state, _ = self.encoder(out, h0)
        out = last_hidden_state[:, 0]
        # print('encoder out shape : ', out.shape)

        out = out.reshape(out.shape[0], -1)

        # print(out.shape)
        out = self.classifier(out)

        return out

    def training_step(self, batch, batch_idx):
        input_ids, y = batch['ids'], batch['target']
        # forward pass
        logits = self(seq=input_ids)
        preds = th.argmax(input=logits, dim=-1)

        # compute metrics
        loss = self.get_loss(logits=logits, targets=y)
        acc = accuracy(preds.cpu(), y.cpu())

        self.log('train_acc',
                 acc,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)

        return {'loss': loss,
                'accuracy': acc,
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
        input_ids, y = batch['ids'], batch['target']

        # forward pass
        logits = self(input_ids)
        preds = th.argmax(input=logits, dim=-1)

        # compute metrics
        val_loss = self.get_loss(logits=logits, targets=y)
        val_acc = accuracy(preds.cpu(), y.cpu())

        self.log('val_loss',
                 val_loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True
                 )

        self.log('val_acc',
                 val_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True
                 )

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
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def configure_optimizers(self):
        opt = th.optim.Adam(
            params=self.parameters(),
            lr=Config.lr
        )

        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='max',
            factor=0.01,
            patience=10,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=True,
        )
        return {"optimizer": opt,
                "lr_scheduler": scheduler,
                "monitor": "val_acc"}

    def get_acc(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return (preds == targets).float().mean()

    def get_loss(self, logits, targets):
        logits = logits.cpu()
        targets = targets.cpu()
        return F.cross_entropy(weight=self.class_w, input=logits, target=targets)


class TransformerModel(pl.LightningModule):
    '''
        input_size (int): The size of embeddings in the network
        input_vocab (vocab): The input vocab
        num_heads (int): The number of heads in the encoder
        num_layers (int): The number of layers in the transformer encoder
        forward_expansion (int): The factor of expansion in the elementwise feedforward layer
        dropout_prob (float): The amount of dropout
        sequence_len (int): The max sequence length used when a target is not provided
        device (torch.device): The device that the network will run on
    Inputs:
        src (Tensor): The input sequence of shape (src length, batch size)
    Returns:
        output (Tensor): probability distribution of predictions

        '''

    def __init__(self,
                 tokenizer,
                 class_w=None,
                 dv: str = 'cuda',
                 d_model=Config.d_model,
                 n_head=Config.n_head,
                 input_size=Config.max_len,
                 sequence_len=Config.max_len,
                 dropout_prob=Config.drop_out_prob,
                 embedding_dim=Config.embedding_dim,
                 pad_idx=0,
                 num_layers=Config.num_layers):
        super(TransformerModel, self).__init__()

        d = dict(Config.__dict__)
        conf_dict = {k: d[k] for k in d.keys() if '__' not in k}
        self.save_hyperparameters(conf_dict)

        self.num_layers = num_layers
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.src_embedding = nn.Embedding(
            self.tokenizer.get_vocab_size(),
            input_size
        )
        self.src_positional_embedding = nn.Embedding(
            self.sequence_len,
            input_size
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.drop_prob
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.num_layers,
            norm=None
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_out = nn.Linear(
            in_features=self.dim_feedforward,
            out_features=Config.n_classes
        )

    def create_pad_mask(self, idx_seq, pad_idx):
        # idx_seq shape: (seq len, batch size)
        mask = idx_seq.transpose(0, 1) == pad_idx
        # mask shape: (batch size, seq len) <- PyTorch transformer wants this shape for mask
        return mask

    def forward(self, src, mask=None):
        batch_size, src_len = src.squeeze(1).shape
        src = src.squeeze(1)

        print('[INFO] sequence shape : ', src.shape)

        # Get source pad mask
        src_pad_mask = self.create_pad_mask(src, self.pad_idx)

        # Embed src
        src_positions = torch.arange(
            start=0,
            end=src_len,
            device=self.device
        ).unsqueeze(1).expand(src_len, batch_size)

        print('[INFO] src_positions shape : ', src_positions.shape)

        src_embed = self.dropout(
            self.src_embedding(
                src) + self.src_positional_embedding(src_positions)
        )

        out = self.transformer(
            src=src_embed,
            mask=None,
            src_key_padding_mask=src_pad_mask
        )
        out = self.fc_out(out)

        return out

    def training_step(self, batch, batch_idx):
        input_ids, y = batch['ids'], batch['target']
        # forward pass
        logits = self(src=input_ids, mask=masks)
        preds = th.argmax(input=logits, dim=-1)

        # compute metrics
        loss = self.get_loss(logits=logits, targets=y)
        acc = accuracy(preds.cpu(), y.cpu())

        self.log('train_acc',
                 acc,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)

        return {'loss': loss,
                'accuracy': acc,
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
        input_ids, y = batch['ids'], batch['target']
        # forward pass
        logits = self(src=input_ids, mask=masks)
        preds = th.argmax(input=logits, dim=-1)

        # compute metrics
        val_loss = self.get_loss(logits=logits, targets=y)
        val_acc = accuracy(preds.cpu(), y.cpu())

        self.log('val_loss',
                 val_loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True
                 )

        self.log('val_acc',
                 val_acc,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True
                 )

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
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def configure_optimizers(self):
        opt = th.optim.Adam(
            params=self.parameters(),
            lr=Config.lr
        )

        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='max',
            factor=0.01,
            patience=10,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=True,
        )
        return {"optimizer": opt,
                "lr_scheduler": scheduler,
                "monitor": "val_acc"}

    def get_acc(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return (preds == targets).float().mean()

    def get_loss(self, logits, targets):
        logits = logits.cpu()
        targets = targets.cpu()
        return F.cross_entropy(weight=self.class_w, input=logits, target=targets)


if __name__ == "__main__":
    print('[INFO] Building model')
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
    try:
        model = TransformerModel(
            tokenizer=tokenizer
        )

        print('[INFO] Model built')
        print(model)

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
                    # logits = model(
                    #     seq=ids
                    # )

                    print(e)
                print(logits)
                preds = th.argmax(input=logits, dim=-1)
                print("[INFO] Logits : ", logits.shape)
                print("[INFO] Predictions : ", preds)

                acc = model.get_acc(preds=preds, targets=targets)
                loss = model.get_loss(preds=logits, targets=targets)

                print("[INFO] acc : ", acc)
                print("[INFO] loss : ", loss)

            except Exception as e:
                print(e)
            break

    except Exception as e:
        print(f'[ERROR] {e}')
