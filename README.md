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

## Approach

## Usage

- Training
- Inference

## Acknowledgements

- The code is based on learning from the shared notebooks on internet
- Some of the snippets code copied from anywhere will liked to their source (original implementation)