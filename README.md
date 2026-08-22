# AI-Based Sentiment & Emotion Analysis

An end-to-end Deep Learning application that classifies text into **six distinct emotions**: *Joy, Sadness, Anger, Fear, Love, and Surprise*. Built using a Recurrent Neural Network (RNN/BiGRU) architecture, Natural Language Processing (NLP) techniques, and integrated into a web interface.

---

## 📌 Project Overview
- **Objective:** Classify user input text into fine-grained emotional categories using deep learning.
- **Dataset:** Sourced from huggingface for text-based emotion recognition.
- **Model Architecture:** Recurrent Neural Network (RNN / BiGRU model trained with TensorFlow & Keras).
- **Target Emotions:**
  - 😊 Joy
  - 😢 Sadness
  - 😡 Anger
  - 📁 Fear
  - ❤️ Love
  - 😮 Surprise

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Deep Learning & NLP:** TensorFlow / Keras, NLTK
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Backend/Frontend:** FastAPI (Python), HTML, CSS, JavaScript

---

## 📂 Project Structure

```text
AI-based-sentiment-analysis/
│
├── Artifacts/              # Saved model files & tokenizer (e.g., BiGRU_Model.keras, tokenizer.pkl)
├── static/                 # Frontend assets (index.html, script.js, style.css)
├── Main.py                 # Core application backend logic
├── Sentiment_Analysis.ipynb # Data preprocessing, EDA, and Bidrectiional GRU model training notebook
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
