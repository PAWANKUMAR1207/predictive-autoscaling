from flask import Flask, request, jsonify
import numpy as np
import joblib
import boto3
import tempfile
from tensorflow.keras.models import load_model
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

S3_BUCKET = "pawan-autoscaling-models-2026"
S3_MODEL_KEY = "lstm_model.keras"
S3_SCALER_KEY = "scaler.pkl"

FEATURES = [
    "cpu_utilisation", "network_in_bytes", "request_count",
    "error_count", "p99_latency_ms", "error_rate_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]
SEQ_LEN = 60

# Prometheus metrics
cpu_forecast_gauge = Gauge('cpu_forecast', 'Predicted CPU utilization')
anomaly_score_gauge = Gauge('anomaly_score', 'Predicted anomaly score')
error_rate_gauge = Gauge('error_rate_forecast', 'Predicted error rate')
scaling_action_counter = Counter('scaling_actions_total', 'Total scaling actions', ['action'])
prediction_counter = Counter('predictions_total', 'Total predictions made')

def load_models_from_s3():
    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
        s3.download_fileobj(S3_BUCKET, S3_MODEL_KEY, f)
        model_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        s3.download_fileobj(S3_BUCKET, S3_SCALER_KEY, f)
        scaler_path = f.name

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    return model, scaler

model, scaler = load_models_from_s3()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route("/predict", methods=["POST"])
def predict():
    # Fix 1: Handle missing/invalid JSON properly
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        body = request.get_json(force=True, silent=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if body is None:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    data = body.get("data")
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if len(data) != SEQ_LEN:
        return jsonify({"error": f"Expected {SEQ_LEN} timesteps, got {len(data)}"}), 400

    # Fix 2: Validate shape properly before accessing shape[1]
    try:
        arr = np.array(data, dtype=np.float32)
    except Exception:
        return jsonify({"error": "Data must be a 2D list of numbers"}), 400

    if arr.ndim != 2 or arr.shape[1] != len(FEATURES):
        return jsonify({"error": f"Each row must have exactly {len(FEATURES)} features"}), 400

    try:
        arr_scaled = scaler.transform(arr)
        arr_scaled = arr_scaled.reshape(1, SEQ_LEN, len(FEATURES))

        prediction = model.predict(arr_scaled)

        # Fix 3: Handle both multi-output and single tensor output
        if isinstance(prediction, list) and len(prediction) == 3:
            # Multi-output: model returns [cpu_array, anomaly_array, error_array]
            cpu_forecast = float(prediction[0][0][0])
            anomaly_score = float(prediction[1][0][0])
            error_rate_forecast = float(prediction[2][0][0])
        elif isinstance(prediction, np.ndarray) and prediction.shape == (1, 3):
            # Single tensor: model returns (1, 3)
            cpu_forecast = float(prediction[0][0])
            anomaly_score = float(prediction[0][1])
            error_rate_forecast = float(prediction[0][2])
        else:
            return jsonify({"error": f"Unexpected model output shape: {type(prediction)}"}), 500

        if cpu_forecast > 0.75 or anomaly_score > 0.7:
            scaling_action = "SCALE_UP"
        elif cpu_forecast < 0.30 and anomaly_score < 0.3:
            scaling_action = "SCALE_DOWN"
        else:
            scaling_action = "NO_CHANGE"

        # Update Prometheus metrics
        cpu_forecast_gauge.set(cpu_forecast)
        anomaly_score_gauge.set(anomaly_score)
        error_rate_gauge.set(error_rate_forecast)
        scaling_action_counter.labels(action=scaling_action).inc()
        prediction_counter.inc()

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