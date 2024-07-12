from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = mlflow.sklearn.load_model("models:/sentiment_model/latest")
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)