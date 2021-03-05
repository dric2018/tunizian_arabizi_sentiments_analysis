import os


class Config:
    base_dir = os.path.abspath('../')
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    submissions_dir = os.path.join(base_dir, "submissions")
    n_folds = None
    stratified = None
    seed_value = 2021
    # 'bert-base-multilingual-uncased'  #  xlm-roberta-base # distilbert-base-multilangual-cased
    base_model = "distilbert-base-multilingual-cased"
    n_classes = 3
    max_len = 80
    train_batch_size = 256
    test_batch_size = 256
    num_epochs = 100
    drop_out_prob = .25
    early_stopping_patience = 20
    num_layers = 5
    lr = 4e-3
