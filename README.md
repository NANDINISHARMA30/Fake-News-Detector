
# 📰 Fake News Detection System

## 📌 Project Overview

The **Fake News Detection System** is a web application that classifies news articles as **Fake** or **Real** using **Machine Learning (ML)** and **Natural Language Processing (NLP)**. The system extracts features from text using **TF-IDF vectorization** and supports multiple classifiers:

* **Logistic Regression (LR)**
* **Passive-Aggressive Classifier (PAC)**
* **Multinomial Naive Bayes (NB)**

The model is deployed via **Flask**, enabling real-time predictions through a simple web interface.
This project demonstrates an end-to-end **ML workflow**, including preprocessing, model training, evaluation, and deployment.

---

## 🎯 Objectives

* Automatically detect fake news using multiple ML models
* Apply NLP techniques to preprocess textual data
* Convert text into numerical features via TF-IDF
* Train, evaluate, and deploy classification models
* Provide a user-friendly web interface

---

## 🛠️ Technologies Used

**Programming Language:**

* Python

**Libraries & Frameworks:**

* Flask – Web framework
* Pandas – Data manipulation
* Scikit-learn – Machine learning
* NLTK – NLP preprocessing
* Joblib – Model serialization
* HTML & CSS – Frontend

---

## 📂 Project Structure

```
Fake News Detection/
│
├── app.py                  # Flask backend
├── train_model.py          # Model training and evaluation
├── model_logreg.pkl        # Logistic Regression model
├── model_pac.pkl           # Passive-Aggressive Classifier model
├── model_nb.pkl            # Naive Bayes model
├── tfidf.pkl               # TF-IDF vectorizer
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

## 📊 Dataset Description

Datasets used:

* **Fake.csv** – Fake news articles
* **True.csv** – Real news articles

### Label Encoding

| Label | Description |
| ----- | ----------- |
| 1     | Fake News   |
| 0     | Real News   |

---

## ⚙️ Data Preprocessing

1. Lowercase all text
2. Remove special characters and punctuation
3. Tokenize text
4. Remove stopwords (*the, is, and, etc.*)
5. Apply stemming to reduce words to root form

---

## 🧠 Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts textual data into numerical vectors:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_features = vectorizer.fit_transform(corpus)
```

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression (LR)

* Binary classification algorithm
* Fast and interpretable

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
```

### 2️⃣ Passive-Aggressive Classifier (PAC)

* Online learning algorithm suitable for large datasets
* Adjusts model weights aggressively on misclassified samples, passive on correct predictions

```python
from sklearn.linear_model import PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier(max_iter=1000)
pac.fit(X_train, y_train)
```

### 3️⃣ Multinomial Naive Bayes (NB)

* Probabilistic classifier
* Works well for text classification

```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)
```

---

## 💾 Model Serialization

All models and the TF-IDF vectorizer are saved using **Joblib**:

* `model_logreg.pkl` – Logistic Regression
* `model_pac.pkl` – Passive-Aggressive Classifier
* `model_nb.pkl` – Naive Bayes
* `tfidf.pkl` – TF-IDF vectorizer

---

## 🌐 Web Application (Flask)

### Workflow

1. User inputs news text
2. Flask backend receives input
3. Text is preprocessed + TF-IDF applied
4. Selected ML model predicts **Fake** or **Real**
5. Result displayed on the frontend

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install flask pandas scikit-learn nltk joblib
```

### Train Models

```bash
python train_model.py
```

### Start Flask App

```bash
python app.py
```

### Access in Browser

```
http://127.0.0.1:5000/
```

---

## ✅ Output

* Displays **FAKE NEWS ❌** or **REAL NEWS ✅**
* Supports multiple ML classifiers
* Clean, responsive interface

---

## 📈 Advantages

* Multiple classifier options for flexibility
* Fast, lightweight, and scalable
* Real-time predictions
* Easy deployment

---

## ⚠️ Limitations

* Only supports text-based news
* Model performance depends on dataset quality
* Cannot detect fake images/videos
* English-language only

---

## 🔮 Future Enhancements

* Integrate deep learning models (LSTM, BERT)
* Provide prediction confidence scores
* Add multilingual support
* Enable URL-based news scraping
* Deploy on cloud platforms

---

## 🧾 Conclusion

This project integrates **ML models (Logistic Regression, Passive-Aggressive Classifier, Naive Bayes)** with **TF-IDF NLP features** and **Flask deployment**, providing an effective and scalable fake news detection system. Future enhancements can make it production-ready for real-world applications.

---

## 📚 References

* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [Flask Documentation](https://flask.palletsprojects.com/)
* [NLTK Documentation](https://www.nltk.org/)
* [Kaggle Fake News Dataset](https://www.kaggle.com/datasets)


