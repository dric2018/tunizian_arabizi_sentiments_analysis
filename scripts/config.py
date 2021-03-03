import os


class Config:
    data_dir = os.path.abspath('../data')
    n_folds = 10
    stratified = True
    seed_value = 2021
    base_model = 'distilbert-base-multilingual-cased'
    n_classes = 2
    max_len = 150
    train_batch_size = 32
    test_batch_size = 32
