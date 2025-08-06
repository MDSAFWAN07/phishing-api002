from flask import Flask, request, jsonify

app = Flask(__name__)

# Optional: Home route for testing
@app.route('/')
def home():
    return "Phishing Prediction API is Live"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Dummy model logic for now
    features = [data.get("feature1"), data.get("feature2"), data.get("feature3"), data.get("feature4"), data.get("feature5")]
    prediction = "Phishing" if sum(features) > 1.5 else "Legitimate"

    return jsonify({"prediction": prediction})
