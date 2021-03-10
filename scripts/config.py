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
    # 'bert-base-multilingual-uncased'  #  xlm-roberta-base # distilbert-base-multilingual-cased
    base_model = "camembert-base"
    n_classes = 1
    max_len = 150
    train_batch_size = 256
    test_batch_size = 256
    num_epochs = 30
    drop_out_prob = .2
    d_model = 256
    n_head = 8
    dim_feedforward = 2048
    early_stopping_patience = 10
    num_layers = 3
    embedding_dim = 256
    hidden_size = 350  # 350
    lr = 3e-4
