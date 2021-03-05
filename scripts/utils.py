import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
import pandas as pd

import argparse

from tqdm.auto import tqdm

from config import Config
from dataset import DataSet

# learning rate schedule params
LR_START = 1e-5
LR_MAX = 1e-3
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_STEP_DECAY = 0.75


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

    th.cuda.empty_cache()

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


# CUSTOM LEARNING SCHEUDLE
# """
# from https://www.kaggle.com/cdeotte/how-to-compete-with-gpus-workshop#STEP-4:-Training-Schedule
# """

def ramp_scheduler(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MAX * \
            LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//10)
    return lr


def load_models(n_folds: int = None, version: int = 0, arch: str = Config.base_model):
    """
    Load trained models for inference time
    """
    models_list = [m for m in sorted(os.listdir(Config.models_dir)) if m.split(
        '.')[-1] in ['bin', 'pt', 'pth', 'model']]

    matching_models = [m for m in models_list if arch in m]
    print("[INFO] Matching models found : \n", matching_models)
    loaded_models = []
    if n_folds is not None:
        # n_folds models to load for inference
        for m_name in tqdm(matching_models, desc='Loding models'):
            if 'fold' in m_name:
                try:
                    m = th.jit.load(os.path.join(Config.models_dir, m_name))
                    loaded_models.append(m)
                    assert len(loaded_models) == n_folds
                except Exception as e:
                    print(f'[ERROR] while loading model : {e}')
    else:
        # one model to load using the version number
        try:

            m_name = "".join(
                [md for md in matching_models if str(version) in md])
            # print(m_name)
            m = th.jit.load(os.path.join(Config.models_dir, m_name))
            loaded_models.append(m)
            assert len(loaded_models) == 1
        except Exception as e:
            print(f'[ERROR] while loading model : {e}')

    return loaded_models


def predict(dataset: DataSet, model: pl.LightningModule, batch_size=16, n_folds=None):

    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    predictions = []

    if n_folds is None:
        with th.no_grad():
            model.eval()
            model.cuda()
            for data in tqdm(test_dl, desc='Predicting'):
                ids = data['ids']
                preds = model(ids.cuda())
                # as we added 1 to avoid target from being < 0 (Negative sentiment)
                reformat_pred = preds.argmax(dim=1) - 1
                predictions += (reformat_pred.detach().cpu().numpy().tolist())
    else:
        pass

    return predictions
