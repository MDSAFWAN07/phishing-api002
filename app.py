from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your model
model = load_model("phishing_lstm_model.h5")

# Sample input feature names (you must update according to your model)
input_columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Phishing detection API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        # Check all required fields
        if not all(col in data for col in input_columns):
            return jsonify({"error": f"Missing fields. Required: {input_columns}"}), 400

        # Prepare input for prediction
        input_data = [data[col] for col in input_columns]
        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0][0]
        result = "Phishing" if prediction > 0.5 else "Legitimate"

        return jsonify({"prediction": result, "probability": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
