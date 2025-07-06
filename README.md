# Text Classification: Comparing Transformers vs TF-IDF Method 🚀

## Toxic Comment Classification: TF-IDF + Neural Network VS BERT Transformer

### 📌 Project goal:
Compare a classical **TF-IDF vectorization + fully connected neural network** approach against a modern **BERT Transformer** model for multi-label toxic comment classification.

---

### 🧪 What was done?

- Used the **Jigsaw Toxic Comment Classification** dataset  
- Implemented a baseline using **TF-IDF vectorizer + deep neural network** with dropout and batch norm  
- Developed a **BERT Transformer** model fine-tuned on the same task  
- Evaluated all models on **ROC AUC** metric (macro-average over 6 toxic categories)

---

### ⚡ Results

| Method                          | Validation ROC AUC |
|--------------------------------|-------------------:|
| TF-IDF + Neural Network (this project) |           0.9600 |
| **BERT Transformer (this project)**    |       **0.9874** |

> 💥 BERT significantly outperforms the TF-IDF + Neural Network baseline (98.74% vs 96%)!

---

### 📈 Key takeaways:

**TF-IDF + Neural Network:**

- Effective classical NLP pipeline  
- Preprocessing: tokenization, lemmatization, stopwords removal  
- 3-layer fully connected network with batch norm, dropout, ReLU activations  
- Works well but limited by lack of context awareness  

**BERT Transformer:**

- Pretrained contextual embeddings capture nuances and semantics  
- Fine-tuning with simple classification head improves performance drastically  
- Better generalization and higher ROC AUC scores  

Both models trained with learning rate scheduling and early stopping to ensure stability.

---

### 🛠️ How to run:

1. Prepare the dataset (upload and unzip)  
2. Run the **TF-IDF + Neural Network** training script to establish the baseline  
3. Run the **BERT fine-tuning** script for superior results  
4. Monitor losses and ROC AUC metrics  
5. Use the saved `best_model.pt` checkpoint for predictions  

---

### 🎯 Conclusion:

This project clearly demonstrates the power of pretrained Transformers for toxic comment classification compared to traditional vectorization + neural network approaches. While the TF-IDF + NN baseline achieves solid results, BERT’s ability to understand context boosts performance significantly.

---

### 🙌 Thanks for checking out this project!

Feel free to experiment with model architectures, additional preprocessing, or other Transformer variants to push results even further. Happy coding! ✨
