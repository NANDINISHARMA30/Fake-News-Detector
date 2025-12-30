# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["category"] = 1
true["category"] = 0

df = pd.concat([fake, true])
X = df['text']  # your text column
y = df['category']

# Create TF-IDF vectorizer and transform data
tfidf = TfidfVectorizer(max_features=5000)
X_features = tfidf.fit_transform(X)

# Train a simple model
model = LogisticRegression(max_iter=1000)
model.fit(X_features, y)

# Save TF-IDF and model
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(model, "model.pkl")
print("Files saved successfully!")
