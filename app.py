import re
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('email_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email_text = data.get('email', '')
    if not email_text:
        return jsonify({'error': 'No email text provided.'}), 400

    email_text = clean_text(email_text)
    X_new = vectorizer.transform([email_text])
    
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new).max()

    return jsonify({
        'prediction': prediction,
        'probability': probability
    })
    
# function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
