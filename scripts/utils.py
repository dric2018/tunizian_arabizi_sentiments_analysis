import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers, seed_everything

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
import pandas as pd

import argparse

from tqdm.auto import tqdm

try:
    from .config import Config
    from .dataset import DataSet, DataModule
    from .model import Model
except ImportError:
    from config import Config
    from dataset import DataSet, DataModule
    from model import Model


seed_everything(Config.seed_value)


def make_folds(data: pd.DataFrame, args: argparse.Namespace, target_col='label', stratified: bool = True):
    data['fold'] = 0

    if stratified:
        fold = StratifiedKFold(
            n_splits=args.n_folds,
            random_state=Config.seed_value,
            shuffle=True
        )
    else:
        fold = KFold(
            n_splits=args.n_folds,
            random_state=Config.seed_value,
            shuffle=True
        )

    for i, (tr, vr) in enumerate(tqdm(fold.split(data, data[target_col]), desc='Splitting')):
        data.loc[vr, 'fold'] = i

    return data, args.n_folds


def run_fold(fold, train_df, args, size=(224, 224), arch='resnet18', pretrained=True,   path='MODELS/', data_transforms=None):

    torch.cuda.empty_cache()

    fold_train = train_df[train_df.fold != fold].reset_index(drop=True)
    fold_val = train_df[train_df.fold == fold].reset_index(drop=True)

    train_ds = DataSet(images_path=args.specs_images_path,
                       df=fold_train, transforms=data_transforms['train'])
    val_ds = DataSet(images_path=args.specs_images_path,
                     df=fold_val, transforms=data_transforms['train'])

    trainloader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=os.cpu_count())
    validloader = DataLoader(
        val_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=os.cpu_count())

    del train_ds
    del val_ds
    del fold_train
    del fold_val

    model = AudioClassifier(arch_name=arch, lr=args.lr, pretrained=pretrained)

    tb_logger = loggers.TensorBoardLogger(
        save_dir='./runs', name='ZINDI-GIZ-NLP-AGRI-KEYWORDS', version=fold)

    ckpt_callback = pl.callbacks.ModelCheckpoint(filename=f'ZINDI-GIZ-NLP-AGRI-KEYWORDS-{model.hparams.arch_name}-{fold}-based',
                                                 dirpath=path,
                                                 monitor='val_logLoss',
                                                 mode='min')

    trainer = Trainer(max_epochs=args.num_epochs, gpus=args.gpus,
                      logger=tb_logger, callbacks=[ckpt_callback])

    trainer.fit(model, trainloader, validloader)

    gc.collect()  # collect garbage

    return trainer.logged_metrics
