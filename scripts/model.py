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
from utils import load_vectors

import pandas as pd
import os
import sys
from dataset import DataModule


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = Config.embedding_dim,
        num_layers: int = Config.num_layers,
        bidirectional: bool = True,
        hidden_size: int = 128
    ):
        super(LSTMModel, self).__init__()
        self.save_hyperparameters()
        # print(self.hparams)

        # architecture

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
        # Embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=tokenizer.vocab_size + 1,
            embedding_dim=self.hparams.embedding_dim
        )
        # Reccurent layer(s)
        self.lstm = nn.LSTM(
            input_size=self.hparams.embedding_dim,
            num_layers=self.hparams.num_layers,
            hidden_size=self.hparams.hidden_size,
            dropout=Config.drop_out_prob,
            bidirectional=self.hparams.bidirectional,
            batch_first=True
        )
        # classifier
        if self.hparams.bidirectional:
            self.fc = nn.Linear(
                in_features=2*self.hparams.hidden_size,
                out_features=1
            )
        else:
            self.fc = nn.Linear(
                in_features=self.hparams.hidden_size,
                out_features=1
            )

    def configure_optimizers(self):
        pass

    def forward(self, x: th.Tensor, mask=None):

        emb = self.embedding_layer(x)
        print('[INFO] emb shape : ', emb.shape)

        hidden_states, _ = self.lstm(emb)
        print('[INFO] hidden_states shape : ', hidden_states.shape)

        avg_pool = th.mean(hidden_states, axis=1)
        max_pool, _ = th.max(hidden_states, axis=1)

        print('[INFO] avg_pool shape : ', avg_pool.shape)
        print('[INFO] max_pool shape : ', max_pool.shape)

        # concat
        # features = th.cat((avg_pool, max_pool), dim=1)
        features = (avg_pool + max_pool) / 2
        print('[INFO] features shape : ', features.shape)
        # apply classification layer
        out = self.fc(features)
        print('[INFO] out shape : ', out.shape)

        return out

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass


class GRUModel(pl.LightningModule):

    def __init__(self):
        super(GRUModel, self).__init__()
        pass

    def configure_optimizers(self):
        pass

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass


class TransformerModel(pl.LightningModule):
    def __init__(self):
        super(TransformerModel, self).__init__()
        pass

    def configure_optimizers(self):
        pass

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass


class BertBaseModel(pl.LightningModule):
    def __init__(self):
        super(BertBaseModel, self).__init__()
        pass

    def configure_optimizers(self):
        pass

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass


if __name__ == "__main__":
    txt = "owel mara n7es eli echa3eb etounsi togrih il o3oud \
    il kadhba eli 7achtou bech i7asen men bladou rahou men \
    9bal elenti5abet b barcha bech ta3mel ki 8irek ta7ki w \
    tfareh la3bed w ki tosel lli t7eb 3lih haka lwa9et 9abloni \
    ken 3mel 7ata 7aja barka meli 9alhoum elkoul taw y8rouf w yrawa7"

    print('[INFO] Building model')
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model)
    try:
        model = LSTMModel()

        print('[INFO] Model built')
        print(model)

        print('[INFO] Getting tokens')

        code = tokenizer.encode_plus(
            text=txt,
            padding='max_length',
            max_length=Config.max_len,
            truncation=True,
            return_tensors='pt'
        )

        ids = code['input_ids']
        mask = code['attention_mask']

        logits = model(ids)
        print("[INFO] Logits : ", logits)
        sys.exit()
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
                        seq=ids
                    )

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
