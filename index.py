from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords")

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

stemmer = SnowballStemmer("english")

def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words("english")]
    return " ".join(text)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    model_used = None

    if request.method == "POST":
        news = request.form["news"]
        model_used = request.form["model"]

        processed = preprocess(news)
        vector = tfidf.transform([processed])
        result = model.predict(vector)[0]

        prediction = "FAKE" if result == 1 else "REAL"

    return render_template("index.html", prediction=prediction, model_used=model_used)

# ❌ DO NOT use app.run()
