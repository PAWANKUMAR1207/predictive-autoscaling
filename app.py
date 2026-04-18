from flask import Flask, request, jsonify
import numpy as np
import pickle
import boto3
import os
import tempfile
from tensorflow.keras.models import load_model

app = Flask(__name__)

# S3 config - replace with your bucket name
S3_BUCKET = "pawan-autoscaling-models-2026"
S3_MODEL_KEY = "models/lstm_model.h5"
S3_SCALER_KEY = "models/scaler.pkl"

# These are the exact features used during training
FEATURES = [
    "cpu_utilisation",
    "network_in_bytes",
    "request_count",
    "error_count",
    "p99_latency_ms",
    "error_rate_pct",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos"
]

SEQ_LEN = 60  # same as training

def load_models_from_s3():
    s3 = boto3.client("s3")

    # Download LSTM model
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        s3.download_fileobj(S3_BUCKET, S3_MODEL_KEY, f)
        model_path = f.name

    # Download scaler
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        s3.download_fileobj(S3_BUCKET, S3_SCALER_KEY, f)
        scaler_path = f.name

    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_models_from_s3()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body with 60 rows of data, each row having these 10 features:
    cpu_utilisation, network_in_bytes, request_count, error_count,
    p99_latency_ms, error_rate_pct, hour_sin, hour_cos, dow_sin, dow_cos
    
    Example:
    {
        "data": [
            [13.26, 202204, 191, 0, 65.7, 0.0, 0.065, 0.997, 0.974, -0.222],
            ... (60 rows total)
        ]
    }
    """
    try:
        data = request.json.get("data")

        if not data:
            return jsonify({"error": "No data provided"}), 400

        if len(data) != SEQ_LEN:
            return jsonify({"error": f"Expected {SEQ_LEN} timesteps, got {len(data)}"}), 400

        arr = np.array(data, dtype=np.float32)  # shape: (60, 10)

        if arr.shape[1] != len(FEATURES):
            return jsonify({"error": f"Expected {len(FEATURES)} features per row"}), 400

        # Scale
        arr_scaled = scaler.transform(arr)  # shape: (60, 10)
        arr_scaled = arr_scaled.reshape(1, SEQ_LEN, len(FEATURES))  # shape: (1, 60, 10)

        # Predict
        prediction = model.predict(arr_scaled)

        # Apply scaling decision logic
        cpu_forecast = float(prediction[0][0])
        anomaly_score = float(prediction[0][1])
        error_rate_forecast = float(prediction[0][2])

        # Auto-scaling decision
        if cpu_forecast > 75 or anomaly_score > 0.7:
            scaling_action = "SCALE_UP"
        elif cpu_forecast < 30 and anomaly_score < 0.3:
            scaling_action = "SCALE_DOWN"
        else:
            scaling_action = "NO_CHANGE"

        return jsonify({
            "cpu_forecast": cpu_forecast,
            "anomaly_score": anomaly_score,
            "error_rate_forecast": error_rate_forecast,
            "scaling_action": scaling_action,
            "features_used": FEATURES
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)