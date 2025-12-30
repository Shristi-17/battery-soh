import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import joblib

# =========================
# Load datasets
# =========================
b5 = pd.read_csv("B0005_discharge.csv")
b6 = pd.read_csv("B0006_discharge.csv")
b7 = pd.read_csv("B0007_discharge.csv")
b18 = pd.read_csv("B0018_discharge.csv")

def add_soh(df):
    df = df.sort_values("cycle").reset_index(drop=True)
    initial_capacity = df.loc[0, "capacity"]
    df["SoH"] = df["capacity"] / initial_capacity
    return df

b5 = add_soh(b5)
b6 = add_soh(b6)
b7 = add_soh(b7)
b18 = add_soh(b18)

# =========================
# Train / Val / Test split
# =========================
train_df = pd.concat([b5, b6], ignore_index=True)
val_df   = b7.copy()    # used for analysis (not training)
test_df  = b18.copy()

# =========================
# Features (WITH cycle)
# =========================
features = [
    "voltage_battery",
    "current_battery",
    "temp_battery",
    "time",
    "cycle"
]

X_train = train_df[features]
y_train = train_df["SoH"]

X_test = test_df[features]
y_test = test_df["SoH"]

# =========================
# Save SoH bounds
# =========================
joblib.dump((y_train.min(), y_train.max()), "soh_bounds.pkl")

# =========================
# XGBoost
# =========================
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)

pred_xgb = xgb_model.predict(X_test)
print("XGB MAE:", mean_absolute_error(y_test, pred_xgb))
print("XGB RMSE:", np.sqrt(mean_squared_error(y_test, pred_xgb)))
print("XGB R2:", r2_score(y_test, pred_xgb))

# =========================
# Scaling for LSTM
# =========================
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_s = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
X_test_s  = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

# =========================
# LSTM
# =========================
lstm = Sequential([
    LSTM(64, input_shape=(1, 5)),
    Dense(1)
])

lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_train_s, y_train, epochs=30, batch_size=32, verbose=1)

pred_lstm = lstm.predict(X_test_s).flatten()
print("LSTM MAE:", mean_absolute_error(y_test, pred_lstm))
print("LSTM RMSE:", np.sqrt(mean_squared_error(y_test, pred_lstm)))
print("LSTM R2:", r2_score(y_test, pred_lstm))

# =========================
# Hybrid
# =========================
hybrid = (pred_xgb + pred_lstm) / 2
print("Hybrid MAE:", mean_absolute_error(y_test, hybrid))
print("Hybrid RMSE:", np.sqrt(mean_squared_error(y_test, hybrid)))
print("Hybrid R2:", r2_score(y_test, hybrid))

# =========================
# Save models
# =========================
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
lstm.save("lstm_model.h5")

print("✅ Models saved successfully")
