# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    features = tfidf.transform([text])
    prediction = model.predict(features)[0]
    result = "Fake" if prediction == 1 else "Real"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
