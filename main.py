# Installation der benötigten Pakete
!pip install yfinance tensorflow --quiet

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Goldpreis-Daten laden
df = yf.download("GC=F", start="2015-01-01", end="2025-03-31")
df = df[['Close']].dropna()

# 2. Normalisieren
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# 3. Sequenzen für Seq2Seq erzeugen
def create_seq2seq_dataset(data, input_len, output_len):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + output_len])
    return np.array(X), np.array(y)

input_days = 180  # Anzahl der Tage für Eingabesequenzen
output_days = 30  # Anzahl der Tage für die Vorhersage
X, y = create_seq2seq_dataset(scaled, input_days, output_days)

# 4. Trainings-/Testdaten aufteilen
split = int(0.9 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 5. Modell definieren
model = Sequential([
    LSTM(128, input_shape=(input_days, 1)),
    Dropout(0.2),
    Dense(output_days)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# 6. Modell trainieren
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 7. Modell speichern
model_dir = "gold_lstm_model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.keras")
model.save(model_path)

# Modell optional laden
# model = load_model(model_path)

# 8. Vorhersage für die nächsten 30 Tage
last_input = scaled[-input_days:].reshape(1, input_days, 1)
predicted_scaled = model.predict(last_input)[0]

# Korrekte De-Skalierung der Vorhersage
predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

# Zeige den letzten tatsächlichen Goldpreis und den ersten Vorhersagewert
print(f"Letzter realer Preis: {df['Close'].values[-1]}")
print(f"Vorhergesagter Preis (erste Vorhersage): {predicted_prices[0][0]}")

# 9. Visualisierung: 180 Tage Historie + 30 Tage Prognose
last_real = df['Close'].values[-input_days:]
future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=output_days, freq="B")

plt.figure(figsize=(14, 5))
plt.plot(df.index[-input_days:], last_real, label="Letzte 180 Tage (Real)", color='blue')
plt.plot(future_index, predicted_prices.flatten(), label="Vorhersage (30 Tage)", color='orange')
plt.axvline(x=df.index[-1], color='gray', linestyle='--', label='Heute')
plt.title("Goldpreis-Vorhersage mit Seq2Seq LSTM")
plt.xlabel("Datum")
plt.ylabel("Goldpreis in USD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Metriken berechnen auf Testdaten
y_pred_test_scaled = model.predict(X_test)
y_pred_test = scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1))
y_test_flat = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_test))
mape = mean_absolute_percentage_error(y_test_flat, y_pred_test) * 100

print(f"RMSE auf Testdaten: {rmse:.2f}")
print(f"MAPE auf Testdaten: {mape:.2f}%")

# 11. Vorhersage als CSV speichern
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=output_days, freq="B")
forecast_df = pd.DataFrame(predicted_prices, columns=["Forecast_Close"])
forecast_df["Date"] = forecast_dates
forecast_df = forecast_df[["Date", "Forecast_Close"]]
forecast_df.to_csv("gold_forecast.csv", index=False)

print("Vorhersage als 'gold_forecast.csv' gespeichert.")
print("Modell gespeichert unter:", model_path)
