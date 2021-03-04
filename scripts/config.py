import os


class Config:
    base_dir = os.path.abspath('../')
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    n_folds = 10
    stratified = True
    seed_value = 2021
    # 'bert-base-multilingual-uncased'  #  xlm-roberta-base # distilbert-base-multilangual-uncased
    base_model = "distilbert-base-multilingual-cased"
    n_classes = 3
    max_len = 100
    train_batch_size = 32
    test_batch_size = 32
    num_epochs = 10
