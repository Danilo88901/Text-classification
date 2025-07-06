# TF-IDF Model for Toxic Comment Classification

This project implements a multi-label text classification model using TF-IDF vectorization and a fully connected neural network in PyTorch. It is trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset.

## 🧠 Task Description

The goal is to classify comments into six categories:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

## 🧪 Approach

1. **Text Preprocessing**:
   - Lowercasing
   - Tokenization using regular expressions
   - Stopwords removal (NLTK)
   - Lemmatization (WordNet)

2. **Feature Extraction**:
   - TF-IDF vectorization (up to 20,000 features)

3. **Model Architecture**:
   - Input Layer → Dense (512) → ReLU → Dropout
   - Dense (1024) → ReLU → Dropout
   - Output Layer (6 units with sigmoid activation)

4. **Training**:
   - Binary Cross Entropy with Logits
   - AdamW optimizer with learning rate scheduling
   - Early stopping with patience of 7 epochs

5. **Evaluation**:
   - ROC AUC (macro) on training and validation sets
   - Loss tracking
   - Metrics are visualized at the end of training

## 📊 Performance

- Best Validation ROC AUC: **~0.96**
- Early stopping is applied to avoid overfitting.

## 📁 Structure
tfidf_model/
├── model_tfidf.py # Full training script (to be added)
├── README.md # This file
└── plots/ # ROC AUC & Loss curves (to be added)
## 🔧 Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- matplotlib
- nltk

To install required packages:
```bash
pip install torch scikit-learn pandas matplotlib nltk
