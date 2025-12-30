import numpy as np
import joblib
from flask import Flask, render_template, request

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

app = Flask(__name__)

# =========================
# Load models
# =========================
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
min_soh, max_soh = joblib.load("soh_bounds.pkl")

# =========================
# Build LSTM (same as training)
# =========================
def build_lstm():
    model = Sequential([
        Input(shape=(1, 5)),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

lstm_model = build_lstm()
lstm_model.load_weights("lstm_model.h5")

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = None

    if request.method == "POST":
        try:
            voltage = float(request.form["voltage"])
            current = float(request.form["current"])
            temperature = float(request.form["temperature"])
            time = float(request.form["time"])
            cycle = float(request.form["cycle"])

            X = np.array([[voltage, current, temperature, time, cycle]])

            # ---- XGBoost
            xgb_pred = xgb_model.predict(X)[0]

            # ---- LSTM
            X_scaled = scaler.transform(X)
            X_scaled = np.clip(X_scaled, 0.0, 1.0)
            X_lstm = X_scaled.reshape((1, 1, 5))
            lstm_pred = lstm_model.predict(X_lstm, verbose=0)[0][0]

            # ---- Hybrid
            hybrid_raw = (xgb_pred + lstm_pred) / 2

            # ---- Normalize SoH
            soh = (hybrid_raw - min_soh) / (max_soh - min_soh)
            soh = float(np.clip(soh, 0.0, 1.0))

            prediction = round(soh, 4)

        except Exception as e:
            message = str(e)

    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
