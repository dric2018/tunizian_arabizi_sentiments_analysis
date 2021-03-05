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
    max_len = 70
    train_batch_size = 512
    test_batch_size = 512
    num_epochs = 100
    drop_out_prob = .4
    early_stopping_patience = 5
    num_layers = 6
    embedding_dim = 300
    hidden_size = 350
    lr = 4e-3
