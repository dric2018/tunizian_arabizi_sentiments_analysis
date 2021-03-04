import os
import sys
import pandas as pd

from utils import make_folds
from config import Config
import argparse

parser = argparse.ArgumentParser(prog="Init cross-validation")
parser.add_argument(
    '--data_dir',
    "-d",
    type=str,
    default=Config.data_dir,
    help="Data drirectory"
)
parser.add_argument(
    '--target_dir',
    "-t", type=str,
    default=Config.data_dir,
    help="Destination drirectory"
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

if __name__ == "__main__":
    args = parser.parse_args()
    # sys.exit()
    try:
        print('[INFO] Reading dataframe')
        df = pd.read_csv(os.path.join(Config.data_dir, "Train.csv"))
        print('[INFO] Making folds')
        data, n_folds = make_folds(
            data=df,
            args=args,
            target_col='label',
            stratified=args.stratified
        )

        print(data.head())

        # save kfold dataset

        data.to_csv(
            os.path.join(args.target_dir, f"Train_{n_folds}_folds.csv"),
            index=False
        )
        print('\n[INFO] Finished :!')
    except Exception as e:
        print(f'[ERROR] {e}')
