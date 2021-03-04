import torch as th
from torch.utils.data import Dataset, DataLoader

import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from config import Config

# set seed
# seed_everything(Config.seed_value)


class DataSet(Dataset):
    def __init__(self, df,
                 task='train'):
        super(DataSet, self).__init__()

        self.df = df
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(Config.base_model)

        # print(self.images_dir)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.df.iloc[index].text
        code = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            max_length=Config.max_len,
            truncation=True,
            return_tensors='pt'
        )
        # print(code)

        sample = {
            'text': str(text),  # image tensor
            'ids': code['input_ids'].clone().detach(),
            'mask': code['attention_mask'].clone().detach(),

        }

        if self.task == 'train':
            target = self.df.iloc[index].label
            sample.update({
                'target': th.tensor(target, dtype=th.long)
            })

        return sample


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 df: pd.DataFrame,
                 frac: float = 0,
                 train_batch_size: int = 64,
                 test_batch_size: int = 32,
                 test_size: float = .1
                 ):
        super().__init__()
        self.frac = frac
        self.df = df
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.test_size = test_size

    def setup(self, stage=None):
        # datasets
        # if fraction is fed
        train_df, val_df = train_test_split(
            self.df, test_size=self.test_size, random_state=Config.seed_value)

        if self.frac > 0:
            train_df = train_df.sample(frac=self.frac).reset_index(drop=True)
            self.train_ds = DataSet(
                df=train_df,
                task='train'
            )
        else:
            self.train_ds = DataSet(
                df=train_df,
                task='train'
            )

        self.val_ds = DataSet(
            df=val_df,
            task='train'
        )

        training_data_size = len(self.train_ds)
        validation_data_size = len(self.val_ds)

        print(
            f'[INFO] Training on {training_data_size} samples belonging to {Config.n_classes} classes'
        )
        print(
            f'[INFO] Validating on {validation_data_size} samples belonging to {Config.n_classes} classes'
        )

    # data loaders
    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=os.cpu_count())


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(Config.data_dir, 'Train_10_folds.csv'))
    dm = DataModule(
        df=df,
        frac=1,
        train_batch_size=Config.train_batch_size,
        test_batch_size=Config.test_batch_size,
        test_size=.15
    )
    print('[INFO] Setting data module up')
    dm.setup()

    # print(dm.train_ds[4])
