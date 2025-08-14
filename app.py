from flask import Flask, request, jsonify
import os, sys, json, traceback
import pandas as pd
import joblib
import xgboost as xgb

app = Flask(__name__)

# ---- Config ----
MODEL_PATH  = os.getenv("MODEL_PATH", "phishing_xgboost_model.json")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

FEATURE_NAMES = [
    "length_url", "nb_dots", "nb_hyphens", "nb_at",
    "nb_slash", "nb_www", "nb_com"
]

def extract_features(url: str) -> dict:
    url = str(url or "")
    return {
        "length_url": len(url),
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": 1 if "@" in url else 0,
        "nb_slash": url.count("/"),
        "nb_www": 1 if "www" in url else 0,
        "nb_com": 1 if ".com" in url else 0,
    }

# ---- Load model & scaler (with clear logging) ----
model = None
scaler = None

def load_assets():
    global model, scaler
    try:
        m = xgb.XGBClassifier()
        m.load_model(MODEL_PATH)              # expects the JSON you saved from XGBClassifier
        app.logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        app.logger.error(f"Model load failed: {e}\n{traceback.format_exc()}")
        raise

    try:
        s = joblib.load(SCALER_PATH)
        app.logger.info(f"Loaded scaler from {SCALER_PATH}")
    except Exception as e:
        app.logger.error(f"Scaler load failed: {e}\n{traceback.format_exc()}")
        raise

    model, scaler = m, s

load_assets()

# ---- Routes ----
@app.route("/", methods=["GET"])
def home():
    return "Phishing Detection API is Live!", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify(error="Content-Type must be application/json"), 415

        data = request.get_json(silent=True) or {}
        url = data.get("url", "").strip()
        if not url:
            return jsonify(error="URL is required"), 400

        feats = extract_features(url)
        df = pd.DataFrame([feats], columns=FEATURE_NAMES)

        X = scaler.transform(df.values)
        proba = model.predict_proba(X)[0]
        phishing_conf, legit_conf = float(proba[1]), float(proba[0])

        if phishing_conf > 0.85:
            label, conf = "Phishing", phishing_conf
        elif legit_conf > 0.84:
            label, conf = "Legitimate", legit_conf
        else:
            label, conf = "Uncertain", max(phishing_conf, legit_conf)

        return jsonify(
            url=url,
            features=feats,
            prediction=label,
            confidence=round(conf, 4)
        ), 200

    except Exception as e:
        app.logger.error(f"/predict error: {e}\n{traceback.format_exc()}")
        return jsonify(error="Internal Server Error"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
