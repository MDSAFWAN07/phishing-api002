from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("phishing_lstm_model.h5")

@app.route("/", methods=["GET"])
def home():
    return "TensorFlow App Running Successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0][0]
    label = int(prediction > 0.5)
    return jsonify({"prediction": label})

# 👇 DO NOT include app.run() when using gunicorn
# if __name__ == "__main__":
#     app.run(debug=True)
