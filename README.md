# chewa-news-classification

<p align="center">
  <img src="poster.png"
  alt="Markdown Monster icon"
      width=800
      height=500/>
</p>

## Description (Copied from zindi)

Sentiment analysis relies on multiple word senses and cultural knowledge, and can be influenced by age, gender and socio-economic status.For this task, we have collected and annotated sentences from different social media platforms. The objective of this challenge is to, given a sentence, classify whether the sentence is of positive, negative, or neutral sentiment. For messages conveying both a positive and negative sentiment, whichever is the stronger sentiment should be chosen. Predict if the text would be considered positive, negative, or neutral (for an average user). This is a binary task.

Such solutions could be used by banking, insurance companies, or social media influencers to better understand and interpret a product’s audience and their reactions.

[Click to see the full challenge info](https://zindi.africa/competitions/ai4d-icompass-social-media-sentiment-analysis-for-tunisian-arabizi)

## Experiments pipeline

- Hardware stack

  - RAM : 16 GB
  - Accelerator type : Nvidia GPU Geforce GTX 1060 Max-Q
  - VRAM : 6.1 GB
  - num workers : 4 (CPU count)

- Software stack
  - Language : Python (version 3.8.6)
  - DL library :
    - Pytorch (version 1.7.1) + Pytorch Lightning (1.2.0)
    - Huggingface transformers/tokenizers

### Exp 1 : Bag of words

HashVectorizer + logistic regression (5-folds)

- local score : 0.8001
- LB score : 0.7726

HashVectorizer + Naive bayes (5-folds)

- local score : 0.78
- LB score : 0.7404

HashVectorizer + Naive bayes (10-folds)

- local score : 0.78
- LB score : 0.749733333333333

HashVectorizer + Passive aggressive (5-folds)

- local score : 0.7833857142857144
- LB score : 0.768133333333333

TfIdfVectorizer + logistic regression (5-folds)

- local score : 0.8005285714285714
- LB score : 0.651266666666667

TfIdfVectorizer + Naive Bayes (5-folds)

- local score : 0.8117285714285714
- LB score : 0.5828

### Exp 2 : Reccurents nets

- Freeze Camembert-base embeddings + 2 layers LSTM

  - local score : 0.80975
  - LB score : 0.793333333333333

- Freeze Camembert-base embeddings + 2 layers GRu

  - local score : 0.80630
  - LB score : 0.784933333333333

- unfreeze Roberta-base embeddings + 2 layers LSTM

  - local score : Nan
  - LB score : NAN

- unfreeze camembert-base embeddings + 2 layers LSTM

  - local score : Nan
  - LB score : NAN

### Exp 4 : BERT-like transformers:

- Distilbert (english)

  - local score : Nan
  - LB score : Nan

- Distilbert (french)

  - local score : Nan
  - LB score : Nan

- Distilbert (multilingual)

  - local score : Nan
  - LB score : Nan

- Roberta (english)

  - local score : Nan
  - LB score : Nan

- Bert-base-uncased (english)

  - local score : Nan
  - LB score : Nan

## Usage

- Training
- Inference

## Results

- What worked:
- What didn't worked:

## Acknowledgements

- The code is based on learning from the shared notebooks on internet
- Some of the snippets code copied from anywhere will liked to their source (original implementation)
