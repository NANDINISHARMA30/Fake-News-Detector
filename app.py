from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download stopwords once
nltk.download("stopwords")

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

stemmer = SnowballStemmer("english")

def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words("english")]
    return " ".join(text)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        news = request.form["news"]

        if news.strip():
            processed = preprocess(news)
            vector = tfidf.transform([processed])
            result = model.predict(vector)[0]

            prediction = "FAKE NEWS ❌" if result == 1 else "REAL NEWS ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
