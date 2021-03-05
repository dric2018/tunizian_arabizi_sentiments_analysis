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
    base_model = "bert-base-multilingual-uncased"
    n_classes = 3
    max_len = 100
    train_batch_size = 64
    test_batch_size = 64
    num_epochs = 60
    drop_out_prob = .3
    d_model = 512
    n_head = 8
    dim_feedforward = 2048
    early_stopping_patience = 5
    num_layers = 4
    embedding_dim = 512
    hidden_size = None  # 350
    lr = 2e-2
