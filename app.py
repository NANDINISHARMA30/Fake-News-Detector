from flask import Flask, render_template, request
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download stopwords
nltk.download("stopwords")

# Flask app with correct paths
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ---------- PATH HANDLING (Vercel-safe) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "..", "tfidf.pkl"))

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# ---------- TEXT PREPROCESSING ----------
def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

# ---------- ROUTE ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        news = request.form.get("news")

        if news and news.strip():
            processed = preprocess(news)
            vector = tfidf.transform([processed])
            result = model.predict(vector)[0]
            prediction = "FAKE NEWS ❌" if result == 1 else "REAL NEWS ✅"

    return render_template("index.html", prediction=prediction)

# ---------- REQUIRED FOR VERCEL ----------
app = app
