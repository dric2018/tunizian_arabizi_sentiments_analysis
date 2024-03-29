import os
import time
import gc

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
import pandas as pd

import argparse

from tqdm.auto import tqdm

from config import Config
from dataset import DataSet
import models

import matplotlib.pyplot as plt
import seaborn as sb

import io

from typing import Union

import re


# learning rate schedule params
LR_START = Config.lr
LR_MAX = Config.lr/.1
LR_RAMPUP_EPOCHS = 6
LR_SUSTAIN_EPOCHS = 0
LR_STEP_DECAY = 0.8
# CUSTOM LEARNING SCHEUDLE
# """
# from https://www.kaggle.com/cdeotte/how-to-compete-with-gpus-workshop#STEP-4:-Training-Schedule
# """


def ramp_scheduler(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MAX * \
            LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//10)
    return lr


def load_vectors(fname):
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        data = {}
        for line in tqdm(fin, desc='loading vectors'):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
    return data


def remove_repetitions(sequence: str, n_letters_repetitions: int = 2, n_pattern_repetitions: int = 2):
    """
    Clean text by removing most repetitive letters 

    :params sequence (str) : the strig text/sequence to clean
    :params n_repetitions (str) : number of repetitions accepted

    : returns (str) cleaned text

    """
    words = sequence.split()
    text = []
    for word in words:
        try:
            # is the current <word> an integer or a float number
            # (I know a regex woud have been better but I do not understand it at all)
            int(word)
            float(word)
        except ValueError:
            for idx, letter in enumerate(word):
                letter_count = word[idx:].count(letter)
                pattern = word[idx:idx+4]

                if letter_count > n_letters_repetitions:

                    # filter 1 : limit repetitive consecutive letters to n_repetritions
                    word = word.replace(
                        letter*letter_count,
                        letter*n_letters_repetitions
                    )

                # filter 2 : remove repetitive patterns to reduce words like <ahahahahaha> or <hihihihihih>
                n_occur = word.count(pattern)

                if n_occur > n_pattern_repetitions:
                    # print(pattern, n_occur)
                    word = word.replace(pattern, "")

        text.append(word)

    return ' '.join(text)


def replace_accents(text: str):
    """
        Replace accentuated letters with their corresponding non-accentuated letters
        :params text (str): text to clean

        : returns (str) cleaned text
    """

    accents_map = {
        "à": 'a',
        "â": "a",
        "é": "e",
        "è": "e",
        "ë": "e",
        'ê': 'e',
        'ô': 'o',
        "ç": 'c',
        "î": "i",
        "ï": "i",
        "û": "u",
        "ù": "u",
        "ü": "u",

    }
    # convert to lowercase
    text = text.lower()

    # replace accentuated letters
    for letter in accents_map.keys():
        text = text.replace(letter, accents_map[letter])

    return text


def delete_outliers(data: pd.DataFrame):

    max_len_txts = [i for i, txt in enumerate(
        data.text.tolist()) if (len(txt) > 512 or len(txt) < 10)]
    data.drop(max_len_txts, axis=0, inplace=True)

    return data


def show_lengths_distribution(data: pd.DataFrame):
    interval = np.linspace(start=0, stop=len(
        data)-1, num=len(data), dtype=np.int32)
    text_lengths = data.text.map(lambda x: len(x)).tolist()
    plt.figure(figsize=(18, 6))
    plt.plot(interval, text_lengths)
    plt.title('Text lengths distribution', size=16)
    plt.show()


def make_folds(data: pd.DataFrame, n_folds: int = 5, target_col='label', stratified: bool = True, preprocess=False):
    data['fold'] = 0
    if preprocess:
        # apply some cleaning
        data.text = data['text'].apply(
            lambda txt: remove_repetitions(
                sequence=replace_accents(text=txt),
                n_repetitions=3
            )
        )
    if stratified:
        fold = StratifiedKFold(
            n_splits=n_folds,
            random_state=Config.seed_value,
            shuffle=True
        )

    else:

        fold = KFold(
            n_splits=n_folds,
            random_state=Config.seed_value,
            shuffle=True
        )

    for i, (tr, vr) in tqdm(enumerate(fold.split(data, data[target_col].values)), desc='Splitting', total=n_folds):
        data.loc[vr, 'fold'] = i

    return data, n_folds


def run_on_folds(df: pd.DataFrame, args, version, n_folds=Config.n_folds):
    start = time.time()

    accs = []
    for fold_num in range(n_folds):

        # th.cuda.empty_cache()
        # get splits
        train_df = df[df.fold != fold_num].reset_index(drop=True)
        val_df = df[df.fold == fold_num].reset_index(drop=True)

        # get datasets
        print('[INFO] Setting datasets up')
        train_ds = DataSet(df=train_df)
        val_ds = DataSet(df=val_df)
        train_dl = DataLoader(dataset=train_ds,
                              batch_size=Config.train_batch_size,
                              shuffle=True,
                              num_workers=os.cpu_count())

        val_dl = DataLoader(dataset=val_ds,
                            batch_size=Config.test_batch_size,
                            shuffle=False,
                            num_workers=os.cpu_count())

        # build model
        print('[INFO] Building model')

        models_map = {
            'lstm': models.LSTMModel,
            'gru': models.GRUModel,
            'bert': models.BertBaseModel,
            'transformer': models.TransformerModel
        }

        model = models_map[args.model_type]()

        # config training pipeline
        print('[INFO] Callbacks and loggers configuration')
        ckpt_cb = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath=Config.models_dir+'/kfolds',
            filename=f'{Config.base_model}-{args.model_type}-version-{version}' +
            "-arabizi-{val_acc:.5f}-{val_loss:.5f}-{fold_num}"
        )

        gpu_stats = GPUStatsMonitor(
            memory_utilization=True,
            gpu_utilization=True,
            fan_speed=True,
            temperature=True
        )
        es = EarlyStopping(
            monitor='val_acc',
            patience=Config.early_stopping_patience,
            mode='max'
        )

        Logger = TensorBoardLogger(
            save_dir=Config.logs_dir,
            name='zindi-arabizi',
            version=version
        )

        cbs = [es, ckpt_cb, gpu_stats]

        # build trainer
        print('[INFO] Building trainer')
        trainer = Trainer(
            gpus=1,
            precision=32,
            max_epochs=Config.num_epochs,
            callbacks=cbs,
            logger=Logger,
            deterministic=True
            # fast_dev_run=True
        )

        print(f'[INFO] Runing experiment N° {version}')
        # train/eval/save model(s)
        print(
            f'[INFO] (split {fold_num}) Training model for {Config.num_epochs} epochs'
        )
        trainer.fit(
            model=model,
            train_dataloader=train_dl,
            val_dataloaders=val_dl
        )

        # append best acc
        best_acc = model.best_acc.cpu().item()
        accs.append(model.best_acc.cpu().item())

        print(f'[INFO] Saving model for inference')
        try:
            fn = f'arabizi-sentiments-{Config.base_model}-version-{version}-fold-{fold_num}.bin'
            th.jit.save(
                model.to_torchscript(),
                os.path.join(
                    Config.models_dir+'/kfolds',
                    fn
                )
            )
            print(f'[INFO] Model saved as {fn}')
            print(f'[INFO] Split {fold_num} : Best accuracy = {best_acc}')
            print()
        except Exception as e:
            print("[ERROR]", e)

    avg_acc = np.array(accs).mean()
    print(f'[INFO] Mean accuracy = {avg_acc}')
    end = time.time()

    duration = (end - start) / 60
    print(f'[INFO] Training time : {duration} mn')

    gc.collect()  # collect garbage


def load_models(n_folds: int = None, version: int = 0, arch: str = Config.base_model):
    """
    Load trained models for inference time
    """
    models_list = [m for m in sorted(os.listdir(Config.models_dir)) if m.split(
        '.')[-1] in ['bin', 'pt', 'pth', 'model']]

    if n_folds is not None:
        models_list = [m for m in sorted(os.listdir(Config.models_dir+'/kfolds')) if m.split(
            '.')[-1] in ['bin', 'pt', 'pth', 'model']]
        matching_models = [m for m in models_list if (
            arch in m) and (str(version) in m)]
    else:
        matching_models = [m for m in models_list if str(version) in m]
        # sort list to have last version at end
        matching_models.sort(key=natural_keys)

    print(
        f"[INFO] ({len(matching_models)}) Matching models found : \n", matching_models
    )
    loaded_models = []
    if n_folds is not None:
        # n_folds models to load for inference
        for m_name in tqdm(matching_models, desc='Loding models'):
            if 'fold' in m_name:
                try:
                    m = th.jit.load(os.path.join(
                        Config.models_dir+'/kfolds', m_name))
                    loaded_models.append(m)

                except Exception as e:
                    print(f'[ERROR] Model not found : {e}')

        assert len(loaded_models) == n_folds
    else:
        # one model to load using the version number
        try:
            m_name = "".join(
                [md for md in matching_models if str(version) in md]

            )
            # print(m_name)
            m = th.jit.load(os.path.join(Config.models_dir, m_name))
            loaded_models.append(m)
            assert len(loaded_models) == 1
        except Exception as e:
            print(f'[ERROR] while loading model : {e}')

    return loaded_models


def predict(dataset: DataSet, models: list, batch_size=16, n_folds=None):

    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    predictions = []
    all_predictions = []

    if n_folds is None:
        assert len(models) == 1
        model = models[0]
        with th.no_grad():
            model.eval()
            model.cuda()
            for data in tqdm(test_dl, desc='Predicting'):
                ids = data['ids']
                logits = model(ids.cuda())
                pred = logits.argmax(dim=1)

                predictions += (pred.detach().cpu().numpy().tolist())

        # as we added 1 to avoid target from being < 0 (Negative sentiment)
        # we need to reformat the predictions
        for idx, p in enumerate(predictions):
            if p == 0:
                predictions[idx] = -1
            else:
                predictions[idx] = 1
        return predictions
    else:
        assert len(models) == n_folds
        for num in range(n_folds):
            model = models[num]
            print(f'Model from split {num}')
            with th.no_grad():
                model.eval()
                model.cuda()
                for data in tqdm(test_dl, desc='Predicting'):
                    ids = data['ids']
                    logits = model(ids.cuda())
                    # as we added 1 to avoid target from being < 0 (Negative sentiment)
                    # we need to reformat
                    pred = logits.argmax(dim=1) - 1
                    # for idx, p in enumerate(pred):
                    #     if p == 0:
                    #         pred[idx] = -1
                    #     else:
                    #         pred[idx] = 1

                    predictions += (pred.detach().cpu().numpy().tolist())

                all_predictions.append(np.array(predictions))
                predictions = []

                del model
        # perform ensembling task
        print('[INFO] Ensembling results')
        labels = average_predictions(preds=all_predictions, threshold=.5)

        return labels


def average_predictions(preds, threshold: float = .7):
    predictions = np.array(preds).mean(axis=0)
    labels = np.zeros(shape=30000, dtype=int)
    for idx, p in enumerate(predictions):
        if p > threshold:
            labels[idx] = 1
        elif 0 < p < threshold:
            labels[idx] = 0
        else:
            labels[idx] = -1

    return labels


def atoi(text):
    # from  https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside#5967539
    return int(text) if text.isdigit() else text


def natural_keys(text):
    # from  https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside#5967539
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def save_experiment_conf():

    walk = [folder for folder in os.listdir(os.path.join(
        Config.logs_dir, 'zindi-arabizi')) if ("version" in folder) and (len(folder.split('.')) <= 1)]

    # sort the versions list
    walk.sort(key=natural_keys)

    if len(walk) > 0:
        version = int(walk[-1].split('version_')[-1]) + 1
    else:
        version = 0

    # save experiment config

    with open(os.path.join(Config.logs_dir, 'zindi-arabizi', f'conf-exp-{version}.txt'), 'w') as conf:
        conf.write(
            f'================== Config file version {version} ===================\n\n')
        d = dict(Config.__dict__)
        conf_dict = {k: d[k] for k in d.keys() if '__' not in k}

        for k in conf_dict:
            v = conf_dict[k]
            conf.write(f'{k} : {v}\n')

    return version
