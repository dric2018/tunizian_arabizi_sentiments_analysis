import os


class Config:
    data_dir = os.path.abspath('../data')
    n_folds = 10
    stratified = True
    seed_value = 2021
