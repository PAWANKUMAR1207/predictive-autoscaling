# Predictive Autoscaling API

This project provides a Flask API for predictive autoscaling. It loads an LSTM model and a scaler, accepts a 60-timestep metrics sequence, forecasts system behavior, and returns a scaling recommendation.

## Features

- Health check endpoint at `/health`
- Prediction endpoint at `/predict`
- Forecasts:
  - CPU utilization
  - Anomaly score
  - Error rate
- Returns a scaling action:
  - `SCALE_UP`
  - `SCALE_DOWN`
  - `NO_CHANGE`

## Project Structure

```text
predictive-autoscaling/
|-- app.py
|-- README.md
|-- Dockerfile
`-- model/
    |-- lstm_model.h5
    `-- scaler.pkl
```

## Requirements

Install the Python dependencies used by `app.py`:

```bash
pip install flask numpy tensorflow boto3 scikit-learn gunicorn pickle5
```

## Run Locally

Start the Flask application:

```bash
python app.py
```

The API runs on:

```text
http://localhost:5000
```

## API Endpoints

### `GET /health`

Returns the service status.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts a JSON body with `60` timesteps. Each timestep must contain these `10` features in order:

1. `cpu_utilisation`
2. `network_in_bytes`
3. `request_count`
4. `error_count`
5. `p99_latency_ms`
6. `error_rate_pct`
7. `hour_sin`
8. `hour_cos`
9. `dow_sin`
10. `dow_cos`

Example request:

```json
{
  "data": [
    [13.26, 202204, 191, 0, 65.7, 0.0, 0.065, 0.997, 0.974, -0.222]
  ]
}
```

Note: the request must contain exactly `60` rows. The example above shows only one row for brevity.

Example response:

```json
{
  "cpu_forecast": 81.4,
  "anomaly_score": 0.76,
  "error_rate_forecast": 1.2,
  "scaling_action": "SCALE_UP",
  "features_used": [
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
}
```

## Scaling Logic

The application chooses the scaling action using this logic:

- `SCALE_UP` if `cpu_forecast > 75` or `anomaly_score > 0.7`
- `SCALE_DOWN` if `cpu_forecast < 30` and `anomaly_score < 0.3`
- `NO_CHANGE` otherwise

## Important Note

`app.py` currently loads the model and scaler from Amazon S3 using these hard-coded values:

- `S3_BUCKET = "your-bucket-name"`
- `S3_MODEL_KEY = "models/lstm_model.h5"`
- `S3_SCALER_KEY = "models/scaler.pkl"`

Although local files exist in the `model/` folder, the current code does not use them. Update the S3 configuration in [app.py](/c:/Users/cheer/OneDrive/Desktop/PROJECT/predictive-autoscaling/app.py) before running this in a real environment, or modify the code to load the local files instead.
