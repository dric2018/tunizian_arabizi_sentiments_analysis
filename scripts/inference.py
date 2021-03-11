import os
import sys
import pandas as pd
import numpy as np

import torch as th
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from config import Config
from dataset import DataSet

from utils import load_models, predict

import argparse

import warnings
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(prog="Inference time")
parser.add_argument(
    '--data_dir',
    "-d",
    type=str,
    default=Config.data_dir,
    help="Data drirectory"
)
parser.add_argument(
    '--models_dir',
    "-md", type=str,
    default=Config.models_dir,
    help="Model Store"
)


parser.add_argument(
    '--n_folds',
    "-nf", type=int,
    default=Config.n_folds,
    help="Number of splits"
)

parser.add_argument(
    '--stratified',
    "-st", type=bool,
    default=Config.stratified,
    help="Stratified or not?"
)

parser.add_argument(
    '--batch_size',
    "-bs", type=int,
    default=16,
    help="Test dataloader batch size"
)

parser.add_argument(
    '--version',
    "-v", type=int,
    required=False,
    help="Version of experiment to use"
)

if __name__ == "__main__":
    args = parser.parse_args()

    print('[INFO] Reading csv file')
    test_df = pd.read_csv(os.path.join(Config.data_dir, "Test.csv"))

    print('[INFO] Building test dataset')
    # build dataset
    test_ds = DataSet(df=test_df, task='test')

    # idx = np.random.randint(low=0, high=len(test_ds))
    # ex = test_ds[idx]
    # print(ex)

    print('[INFO] Loading model(s)')
    try:
        last_version = args.version
    except:
        last_version = int(sorted(os.listdir(os.path.join(
            Config.logs_dir, 'zindi-arabizi')))[-1].split('_')[-1])

    print(f'[INFO] Version : {last_version}')
    loaded_models = load_models(n_folds=args.n_folds, version=last_version)

    # print(loaded_models)
    print('[INFO] Making predictions')
    try:
        for m in loaded_models:
            predictions = predict(
                dataset=test_ds,
                model=m,
                batch_size=args.batch_size
            )

        print('[INFO] Saving submission file')
        submission_df = pd.DataFrame({
            'ID': test_ds.text_ids,
            'label': predictions
        })

        fname = f'version-{last_version}-{Config.base_model}-epochs-{Config.num_epochs}.csv'
        submission_df.to_csv(
            os.path.join(Config.submissions_dir, fname),
            index=False
        )

        print('[INFO] Showing submission file head')
        print(submission_df.head())

        print(f'[INFO] Ssubmission file saved as {fname}')
    except Exception as e:
        print(f'[ERROR] While predicting in inference.py {e}')
