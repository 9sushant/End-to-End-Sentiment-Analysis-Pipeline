from flask import Flask, request, jsonify
import joblib
from utils import TextCleaner

app = Flask(__name__)
pipeline = joblib.load('sentiment_pipeline.pkl')
@app.route('/')
def home():
    return "Send a POST request to /predict with {'review_text': 'your text'}"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review_text']
    prediction = pipeline.predict([review])[0]
    sentiment = 'positive' if prediction == 1 else 'negative'
    return jsonify({'sentiment_prediction': sentiment})

app.run(host='0.0.0.0', port=8080, debug=True)