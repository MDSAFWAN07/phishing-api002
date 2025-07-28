from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "TensorFlow App Running Successfully!"

if __name__ == "__main__":
    app.run(debug=True)