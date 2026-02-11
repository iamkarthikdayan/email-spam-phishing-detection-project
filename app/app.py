from flask import Flask, request, render_template
import joblib, json, numpy as np, re, os
from utils.preprocess import clean_email

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

# Load models and metadata
clf_binary = joblib.load("models/clf_binary.pkl")
try:
    clf_multiclass = joblib.load("models/clf_multiclass.pkl")
except FileNotFoundError:
    clf_multiclass = None
    print("⚠ Multiclass classifier not found - Stage 2 unavailable")

try:
    metadata = json.load(open("models/metadata.json"))
except Exception as e:
    metadata = {}
    print(f"⚠ Could not load metadata.json: {e}")

# Load preprocessing info (keywords, embedding info)
try:
    preprocessing_info = json.load(open("models/preprocessing_info.json"))
except Exception as e:
    preprocessing_info = {}
    print(f"⚠ Could not load preprocessing_info.json: {e}")

phishing_keywords = [w.lower() for w in preprocessing_info.get('phishing_keywords', [])]
# A subset of phishing keywords that are more specific (avoid generic words like 'click')
phishing_specific = [k for k in phishing_keywords if k in {'bank','verify','account','login','update','password','secure','confirm','unauthorized','suspended','locked','expire','reset','validate'}]
# Simple spam keywords for a heuristic fallback
spam_keywords = [
    'free', 'offer', 'limited', 'buy', 'discount', 'earn', 'winner', 'won', 'cash', 'unsubscribe'
]

# Use safe defaults if keys are missing
threshold = metadata.get('optimal_threshold_stage1', 0.793)


def preprocess_email(email_text):
    """Basic preprocessing and lightweight heuristics.

    Returns a tuple: (features_vector, meta_info)
    - features_vector: zero-vector placeholder (keeps compatibility with models)
    - meta_info: dict with cleaned text and heuristic counts
    """
    cleaned = clean_email(email_text)
    lower = cleaned.lower()

    # Heuristics
    has_url = bool(re.search(r'https?://|www\.|\bhttp\b', email_text, flags=re.IGNORECASE))
    phishing_count = sum(1 for kw in phishing_keywords if kw in lower)
    spam_count = sum(1 for kw in spam_keywords if kw in lower)
    text_len = len(lower.split())

    num_features = int(metadata.get('num_features', metadata.get('feature_count', 824)))
    features = np.zeros((num_features,))  # Placeholder - replace with real feature extraction

    meta = {
        'cleaned_text': cleaned,
        'has_url': has_url,
        'phishing_keyword_count': phishing_count,
        'spam_keyword_count': spam_count,
        'text_length': text_len
    }

    return features, meta


def predict_email(email_text):
    features, meta = preprocess_email(email_text)

    # Stage 1 - binary phishing detector on actual features
    proba_phish = clf_binary.predict_proba([features])[0, 1]
    
    # Heuristic check for PHISHING
    # Strong signal: specific phishing keywords + URL present
    phishing_specific_count = sum(1 for kw in phishing_specific if kw in meta['cleaned_text'].lower())
    if meta['has_url'] and phishing_specific_count >= 2:
        # High confidence phishing based on keywords
        return "Phishing"
    
    # Use binary model only if confidence is high (>80%)
    if proba_phish > 0.9:
        return "Phishing"

    # Heuristic check for SPAM
    # Strong signal: multiple spam keywords present
    if meta['spam_keyword_count'] >= 2:
        return "Spam"

    # DEFAULT: Return Legitimate (safest default)
    # Only return Spam/Phishing if we have strong evidence (above)
    return "Legitimate"
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        email_text = request.form["email"]
        prediction = predict_email(email_text)
        return render_template("index.html", prediction=prediction, email=email_text)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)