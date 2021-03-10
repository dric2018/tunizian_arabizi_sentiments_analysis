import os

import torch as th
import torch.nn as nn

from transformers import AutoTokenizer
from config import Config


if __name__ == '__main__':
    txt = "owel mara n7es eli echa3eb etounsi togrih il o3oud \
        il kadhba eli 7achtou bech i7asen men bladou rahou men \
        9bal elenti5abet b barcha bech ta3mel ki 8irek ta7ki w \
        tfareh la3bed w ki tosel lli t7eb 3lih haka lwa9et 9abloni \
        ken 3mel 7ata 7aja barka meli 9alhoum elkoul taw y8rouf w yrawa7"

    print('[INFO] Loading pretrained weights')
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model)

    vocab_size = tokenizer.vocab_size
    print('[INFO] Getting tokens')

    code = tokenizer.encode_plus(
        text=txt,
        padding='max_length',
        max_length=Config.max_len,
        truncation=True,
        return_tensors='pt'
    )

    ids = code['input_ids']
    mask = code['attention_mask']

    print('vocab size', vocab_size)
    print('ids shape : ', ids.shape)
    print('mask shape : ', mask.shape)

    print('[INFO] Building embedding layer')
    embedding_layer = nn.Embedding(
        num_embeddings=vocab_size + 1,
        embedding_dim=Config.embedding_dim
    )

    print('[INFO] Compute embeddings')
    emb = embedding_layer(ids)

    print('[INFO] Embedding shape : ', emb.shape)
    print('[INFO] Embedding output : ', emb)
