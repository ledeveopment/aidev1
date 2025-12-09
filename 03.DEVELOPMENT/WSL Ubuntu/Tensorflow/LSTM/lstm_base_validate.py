
# Installiere ggf. fehlende Pakete:
# pip install yfinance tensorflow scikit-learn matplotlib pandas

#import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

# 1. Historische daten laden (Tesla)
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.datetime.today() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')

#data = yf.download('TSLA', start=start_date, end=end_date)

# 1. daten abrufen
folderPath_His = "/Users/Shared/ai_work/Trainingdata/ml_data/yh_his/D1/"
folderPath_models = "/Users/Shared/ai_work/Trainingdata/models/rnn/"
folderpath_results = "/Users/Shared/ai_work/Trainingdata/ml_results/rnn/"

#"/Users/workplacelivetv/Library/CloudStorage/GoogleDrive-robo01.rpa@gmail.com/My Drive/ml_data/yh_his/D1/"
#filepath = "/Users/workplacelivetv/Library/CloudStorage/GoogleDrive-robo01.rpa@gmail.com/My Drive/ml_data/yh_his/D1/QBTS.csv"
symbol = '#TSLA'
df = pd.read_csv (folderPath_His + symbol + ".csv")
#df = pd.read_csv (filepath)
# daten laden
#df = yf.download("TSLA", start="2018-01-01", end=None)

lastBars = 60
n_tail = 1000
#n_pred_days = 60
#n_time_steps = 60 # Neuronal Netzwerk
#sequence_length = 60
n_epochs = 100
n_batch_size =16
# 3. Sequenzen vorbereiten
#n_steps = 20
forecast_horizon = 20

df_his = df.tail(lastBars+1)
df_his["date"] = pd.to_datetime(df_his["date"]) 


df = df[['date','open', 'high', 'low', 'close','volume']]
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df_full = df.copy()

#data = df.copy()
if n_tail > 0: df = df.tail(n_tail)
if lastBars > 0: df = df[: len(df)-lastBars]
#df = df[['close']].dropna()
data = df.copy()
prices = data[['close']]

# 2. Normalisierung
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# 3. Sequenzen erstellen
sequence_length = 30  # 60 Tage für Vorhersagebasis
X, y = [], []
for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i-sequence_length:i, 0])
    y.append(scaled_prices[i, 0])
X, y = np.array(X), np.array(y)

# Reshape für LSTM [Samples, Timesteps, Features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Split in Training und Validierung
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# 5. Modell erstellen
model = Sequential([
    LSTM(100, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks für bestes Modell
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 6. Training
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stop], verbose=1)

# 7. Vorhersage für nächste 20 Tage
last_sequence = scaled_prices[-sequence_length:]
future_predictions = []
current_seq = last_sequence.copy()
for _ in range(forecast_horizon):
    pred = model.predict(current_seq.reshape(1, sequence_length, 1))
    future_predictions.append(pred[0, 0])
    current_seq = np.append(current_seq[1:], pred, axis=0)

# Rücktransformation
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
df = df.tail(10)
df_his = df_his.set_index('date')
df_his = df_his.tail(lastBars)
# 8. Visualisierung
plt.figure(figsize=(12, 6))
#plt.plot(prices.index[-200:], prices['close'].values[-200:], label='Historische Preise')
plt.plot(df.index, df['close'], label='Historische Daten')
plt.plot(df_his.index, df_his['close'], label='IS-HIS')

plt.plot(future_dates, future_prices.flatten(), label='Vorhersage (20 Tage)', color='red')
plt.title( symbol + ' Prognose (LSTM)')
plt.xlabel('Datum')
plt.ylabel('Preis (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Ergebnisse ausgeben
print("Letzte bekannte Preise:")
print(prices.tail())
print("\nVorhergesagte Preise für die nächsten 20 Tage:")
print(pd.DataFrame({'Datum': future_dates, 'Prognose': future_prices.flatten()}))
