# Transformer Sequence Modeling

**Mathematical foundations and a PyTorch encoder–decoder Transformer for English-to-French translation.**

This project explores recurrent neural networks, LSTM gating and gradient flow, and the attention-based Transformer architecture. The original notebook implements a sequence-to-sequence Transformer with positional encodings and greedy decoding, using a subset of WMT14 English–French translation data.

## Repository contents

- [`notebooks/transformer_translation.ipynb`](notebooks/transformer_translation.ipynb): reorganized notebook with experiments, explanations, and results.
- [`src/transformer_model.py`](src/transformer_model.py): embeddings, positional encoding, Transformer, attention masks.
- [`src/data_preprocessing.py`](src/data_preprocessing.py): translation dataset, parsing, tokenization, vocabularies, batching.
- [`src/train.py`](src/train.py): training and validation epochs.
- [`src/evaluate.py`](src/evaluate.py): greedy decoding, translation, sentence BLEU.

## Experimental setup (original notebook)

- Translation direction: **English → French**.
- WMT14 subset: up to **500,001** sentence pairs downloaded by the notebook.
- Encoder/decoder layers: **2 / 2**; embedding dimension: **128**; attention heads: **4**.
- Feedforward dimension: **512**; batch size: **64**; epochs: **25**.
- Optimizer: Adam with learning rate `0.0001`.

These values describe the original notebook; they are not evidence of a successful fresh training run. The original accompanying report records a low BLEU score and discusses limitations. No new result is claimed here.

## Setup and running

1. Create a Python environment compatible with your PyTorch and torchtext versions. The original notebook pinned older versions; the dependency list is **not yet a tested lockfile**.
2. Install packages from `requirements.txt` and spaCy language models `en_core_web_sm` and `fr_core_news_sm`.
3. Put a WMT14 CSV with a `translation` column (mapping `en`/`fr`) in `data/wmt14_en_fr_10k.csv`, or run the optional dataset download cell in the notebook. Despite the filename, the original download requested 500,001 examples.
4. Open `notebooks/transformer_translation.ipynb` **from the repository root** and run its cells in order. The notebook resolves the project root automatically.

**Note:** A full training run has not been verified in this environment. Data downloads can be large and GPU training may be time-consuming. This repository includes the PDF report, dataset, and large model checkpoints.

## Implementation notes

The refactoring removes unsafe CSV `eval()` parsing, eliminates hard-coded Colab paths, and passes dependencies into training and inference helpers. The original notebook is preserved unchanged so you can compare behavior. Changes in code paths mean fresh results should be validated before publishing performance claims.

## Reference

Vaswani et al. (2017), *Attention Is All You Need*.
