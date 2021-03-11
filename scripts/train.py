import os
import sys
import pandas as pd
import time

import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config

import utils

from dataset import DataSet, DataModule

import models
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-type',
    '-mt',
    type=str,
    default='lstm',
    help='Type of model architechture to use, one of lstm, gru, transformer'
)

# set seed for repro
_ = seed_everything(seed=Config.seed_value)

if __name__ == '__main__':
    args = parser.parse_args()

    # get datasets
    df = pd.read_csv(os.path.join(Config.data_dir, 'Train_5_folds.csv'))
    # save experiment config
    version = save_experiment_conf()
    if Config.n_folds is not None:
        _ = utils.run_on_folds(df=df, args=args, version=version)
    else:
        dm = DataModule(
            df=df,
            frac=1,
            train_batch_size=Config.train_batch_size,
            test_batch_size=Config.test_batch_size,
            test_size=Config.test_size
        )
        print('[INFO] Setting data module up')
        dm.setup()

        # build model
        print('[INFO] Building model')

        if args.model_type.lower() == 'lstm':
            model = models.LSTMModel()
        elif args.model_type.lower() == 'gru':
            model = models.GRUModel()
        else:
            model = models.BertBaseModel()

        # config training pipeline
        print('[INFO] Callbacks and loggers configuration')
        ckpt_cb = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath=Config.models_dir,
            filename=f'{Config.base_model}-{args.model_type}' +
            '-arabizi-{val_acc:.5f}-{val_loss:.5f}'
        )

        gpu_stats = GPUStatsMonitor(
            memory_utilization=True,
            gpu_utilization=True,
            fan_speed=True,
            temperature=True
        )
        es = EarlyStopping(
            monitor='val_acc',
            patience=Config.early_stopping_patience,
            mode='max'
        )
        # save experiment config
        version = utils.save_experiment_conf()

        Logger = TensorBoardLogger(
            save_dir=Config.logs_dir,
            name='zindi-arabizi',
            version=version
        )

        cbs = [es, ckpt_cb, gpu_stats]

        # build trainer
        print('[INFO] Building trainer')
        trainer = Trainer(
            gpus=1,
            precision=32,
            max_epochs=Config.num_epochs,
            callbacks=cbs,
            logger=Logger,
            deterministic=True,
            # fast_dev_run=True
        )

        print(f'[INFO] Runing experiment N° {version}')
        # train/eval/save model(s)
        print(f'[INFO] Training model for {Config.num_epochs} epochs')
        start = time.time()
        trainer.fit(
            model=model,
            datamodule=dm
        )
        end = time.time()

        duration = (end - start) / 60
        print(f'[INFO] Training time : {duration} mn')
        print("[INFO] Best accuracy = ", model.best_acc.cpu().item())
        sys.exit()
        print(f'[INFO] Saving model for inference')
        try:
            fn = f'arabizi-sentiments-{Config.base_model}-version-{version}.bin'
            th.jit.save(
                model.to_torchscript(),
                os.path.join(
                    Config.models_dir,
                    fn
                )
            )
            print(f'[INFO] Model saved as {fn}')
        except Exception as e:
            print("[ERROR]", e)
