import os
import pickle
import re
from typing import Tuple

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def _load_pickle(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required model file '{filename}' not found in {MODEL_DIR}. "
            f"Make sure your .pkl files are inside the 'ml_models' folder."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

# Load resources
svm_model = _load_pickle("svm_model.pkl")
tfidf_vectorizer = _load_pickle("tfidf_vectorizer.pkl")

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        if token in stopwords:
            continue
        lemma = lemmatizer.lemmatize(token)
        processed_tokens.append(lemma)
    return " ".join(processed_tokens)

def predict_sentiment(text: str) -> Tuple[str, float, float]:
    processed = preprocess_text(text)
    if not processed:
        return ("Negative", 0.0, 0.0)
    X = tfidf_vectorizer.transform([processed])
    score = svm_model.decision_function(X)[0]
    probability = svm_model.predict_proba(X)[0][1] if hasattr(svm_model, 'predict_proba') else 0.0
    label = "Positive" if score > 0 else "Negative"
    return (label, float(score), float(probability))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    review_text = ""
    if request.method == 'POST':
        review_text = request.form.get('review', '')
        if not review_text.strip():
            error = "Please enter a review."
        else:
            try:
                label, score, probability = predict_sentiment(review_text)
                result = {'label': label, 'score': round(score, 3), 'probability': round(probability, 3)}
            except Exception as e:
                error = f"Error: {str(e)}"
    return render_template('index.html', result=result, error=error, review_text=review_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
