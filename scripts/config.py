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
    # camembert-base # distilbert-base-uncased
    base_model = "camembert-base"
    n_classes = 3
    max_len = 100
    train_batch_size = 128
    test_batch_size = 128
    test_size = 0.25
    num_epochs = 25
    weight_decay = 0.001
    eps = 1e-08
    drop_out_prob = 0.25
    early_stopping_patience = 8
    reducing_lr_patience = 6
    num_layers = 1
    embedding_dim = 5000
    hidden_size = 128
    lr = 3e-3
    cooldown = 0
    precision = 32
    bidirectional = False
    accumulate_grad_batches = 1
    nhead = 4
    dim_feedforward = 1024
