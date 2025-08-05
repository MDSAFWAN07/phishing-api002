from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model (make sure the file is in the same folder)
model = load_model('phishing_lstm_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data['features']  # Expecting a list of feature values

        # Convert features to numpy array and reshape if needed
        input_features = np.array(features).reshape(1, -1)  

        # Predict using your model
        prediction = model.predict(input_features)

        # Convert prediction to list (for JSON serialization)
        result = prediction.tolist()

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
