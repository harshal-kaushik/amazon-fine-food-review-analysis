# 🍽️ Amazon Dine Review Intelligence

A simple end‑to‑end NLP project that analyzes Amazon Dine customer reviews to predict sentiment and identify key customer issues in real time.

This project focuses on building a **practical and production‑ready text classification pipeline** using classical NLP techniques and deploying it via Streamlit.

---

## 🎯 What This Project Does

* Predicts whether a review is Positive or Negative
* Handles imbalanced review data effectively
* Detects common issues (delivery, food quality, packaging, etc.)
* Provides real‑time analysis through a Streamlit web app

---

## 📊 Dataset

* ~420K Amazon Dine reviews
* Unstructured text data
* Imbalanced class distribution (~5.4:1)

---

## 🧠 Approach

### 🔹 Text Processing

* Lowercasing and cleaning
* Removal of URLs and special characters
* TF‑IDF vectorization (unigrams + bigrams)

### 🔹 Models Tried

* Logistic Regression ✅ (best)
* Multinomial Naive Bayes

Models were compared using Accuracy, Precision, Recall, and F1‑score.

---

## ⚡ Key Improvements Made

### 1️⃣ Handled Class Imbalance

* Used `class_weight="balanced"`
* Ensured fair learning across classes

### 2️⃣ Threshold Optimization (Major Improvement)

The default 0.5 threshold caused poor minority recall.

**Solution:** Tuned the decision threshold using predicted probabilities.

**Impact:**

| Metric         | Before | After    |
| -------------- | ------ | -------- |
| Class‑1 Recall | 0.38   | **0.81** |
| Class‑1 F1     | 0.53   | **0.80** |
| Weighted F1    | 0.88   | **0.94** |

✅ Result: Much better detection of minority sentiment.

---

## 🌐 Streamlit App (Brief)

The Streamlit app provides real‑time review intelligence:

* User enters a review
* Model predicts sentiment using optimized threshold
* Confidence score is displayed
* Key issues are automatically detected

Run locally:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
amazon-dine-nlp/
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── notebook.ipynb
