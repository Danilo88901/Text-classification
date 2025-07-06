# Transformer Toxic Comment Classification

This project compares the classic TFIDF method with a Transformer model (BERT) for toxic comment classification using the Kaggle dataset:

[https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview)

---

## Overview

- Uses **BERT (bert-base-uncased)** fine-tuned for multi-label classification.
- Evaluation metric: **ROC AUC**.
- Best ROC AUC achieved on validation: **98.74%**.
- TFIDF method is used as a baseline for comparison.

---

## Code summary

```python
# Import libraries, tokenization, Dataset and DataLoader setup

# Define MultiLabelBERT model with BERT backbone + classifier head

# Training loop with optimization and ROC AUC evaluation

# Save best model checkpoint

# Plot training/validation loss and ROC AUC curves


Usage and setup
Requires PyTorch and transformers libraries.

Dataset available from the Kaggle competition link above.

GPU is used if available.

Results
ROC AUC on the validation set reaches 0.9874, significantly outperforming the classic TFIDF method.

