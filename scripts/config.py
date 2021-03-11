import os


class Config:
    base_dir = os.path.abspath('../')
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    submissions_dir = os.path.join(base_dir, "submissions")
    n_folds = 5
    stratified = True
    seed_value = 2021
    # 'bert-base-multilingual-uncased'  #  xlm-roberta-base # distilbert-base-multilingual-cased # camembert-base # distilbert-base-uncased
    base_model = "distilbert-base-uncased"
    n_classes = 3
    max_len = 150
    train_batch_size = 512
    test_batch_size = 256
    test_size = .15
    num_epochs = 15  # 30
    weight_decay = .01
    eps = 1e-8
    drop_out_prob = .15
    d_model = 256
    n_head = 8
    dim_feedforward = 2048
    early_stopping_patience = 5
    num_layers = 3
    embedding_dim = 150
    hidden_size = 256  # 350
    lr = 3e-3
