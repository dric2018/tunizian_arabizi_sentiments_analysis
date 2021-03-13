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
    # 'bert-base-multilingual-uncased'  #  xlm-roberta-base # distilbert-base-multilingual-cased # camembert-base # distilbert-base-uncased
    base_model = "camembert-base"
    n_classes = 2
    max_len = 200
    train_batch_size = 128
    test_batch_size = 128
    test_size = .2
    num_epochs = 25  # 30 max
    weight_decay = .01
    eps = 1e-08
    drop_out_prob = .2
    early_stopping_patience = 8
    reducing_lr_patience = 3
    num_layers = 1
    embedding_dim = 256
    hidden_size = 64  # 350
    lr = 2e-3
    precision = 32
    optimizer = "adamw"
    bidirectional = True
    accumulate_grad_batches = 1
    nhead = 4
    dim_feedforward = 1024
