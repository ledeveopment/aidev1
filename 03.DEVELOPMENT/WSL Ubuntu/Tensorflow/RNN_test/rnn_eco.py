
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model,  load_model
from tensorflow.keras.layers import SimpleRNN, Dense


#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras import mixed_precision
from keras.saving import save_model

#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
#from tensorflow.keras.callbacks import EarlyStopping
import time
#from tensorflow.keras import mixed_precision

import matplotlib.pyplot as plt
import datetime

# 1. Daten abrufen

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365*2)  # 2 Jahre Historie

#df = yf.download(symbol, start=start_date, end=end_date)
#df = df[['close']].dropna()

# 1. Daten abrufen

folderPath_His = "/Users/Shared/ai_work/Trainingdata/economics/eco_symbols/"
#X:\economics\eco_symbols
folderPath_models = "/Users/Shared/ai_work/Trainingdata/models/rnn_ft/"
folderpath_results = "/Users/Shared/ai_work/Trainingdata/ml_results/rnn_ft/"

#"/Users/workplacelivetv/Library/CloudStorage/GoogleDrive-robo01.rpa@gmail.com/My Drive/ml_data/yh_his/D1/"
#filepath = "/Users/workplacelivetv/Library/CloudStorage/GoogleDrive-robo01.rpa@gmail.com/My Drive/ml_data/yh_his/D1/QBTS.csv"
symbol = '#NVDA'
df = pd.read_csv (folderPath_His + symbol + ".csv")
#df = pd.read_csv (filepath)
# Daten laden
#df = yf.download("TSLA", start="2018-01-01", end=None)

lastBars = 20
n_tail = 600
#n_pred_days = 60
#n_time_steps = 60 # Neuronal Netzwerk
#sequence_length = 60
n_epochs = 200 
n_batch_size = 16
# 3. Sequenzen vorbereiten
n_steps = 100
forecast_horizon = 10

df_his = df.tail(lastBars+1)
df_his["date"] = pd.to_datetime(df_his["date"]) 

# date       | open               | high               | low                | close              | volume    | adjclose           
# | sp500_open      | sp500_high      | sp500_low       | sp500_close     | sp500_volume | oil_open           | oil_high           | oil_low          
#   | oil_close          | oil_volume | dj_open         | dj_high         | dj_low          | dj_close        | dj_volume  | nq_open  | nq_high         
#  | nq_low   | nq_close         | nq_volume | tn2y_close   | tn2y_volume | tn5y_close  | tn5y_volume | tn10y_close | tn10y_volume | rus_open          
#  | rus_high           | rus_low            | rus_close          | rus_volume | udx_open           | udx_high           | udx_low            | udx_close        
#   | ty13w_open         | ty13w_high         | ty13w_low          | ty13w_close        | ty5Y_open          | ty5Y_high          | ty5Y_low           | ty5Y_close         | ty10Y_open        
#  | ty10Y_high         | ty10Y_low          | ty10Y_close        | ty30Y_open         | ty30Y_high         | ty30Y_low          | ty30Y_close        | gold_close         | price_bygold          |
#Fed Interest Rate Decision_USD_act,GDP (QoQ)_USD_act,CPI (MoM)_USD_act,CPI (YoY)_USD_act,ISM Non-Manufacturing PMI_USD_act,Unemployment Rate_USD_act
"""
Fed Interest Rate Decision_USD_act,GDP (QoQ)_USD_act,CPI (MoM)_USD_act,CPI (YoY)_USD_act,ISM Non-Manufacturing PMI_USD_act,Unemployment Rate_USD_act,AUDUSD_close,
F_TNote_10Y_close,F_TNote_10Y_volume,cot_noncom_long,cot_com_short,
cot_noncom_short,cot_openinterest,F_TYield_13W_close,F_US_TBond_close,F_US_TBond_volume,F_VIX_close,F_USDX_close,USDJPY_close,USDJPY_ch
"""
df = df[['date','open', 'high', 'low', 'close','volume', 'Fed Interest Rate Decision_USD_act','GDP (QoQ)_USD_act','CPI (MoM)_USD_act','CPI (YoY)_USD_act',
         'ISM Non-Manufacturing PMI_USD_act','Unemployment Rate_USD_act','AUDUSD_close',
         'F_TNote_10Y_close','F_TNote_10Y_volume','cot_noncom_long','cot_com_short']]
#,date,open,high,low,close,volume,AUDUSD_close,F_TNote_10Y_close,F_TNote_10Y_volume,cot_noncom_long,cot_com_short,cot_noncom_short,cot_openinterest,F_TYield_13W_close,F_US_TBond_close,F_US_TBond_volume,F_VIX_close,F_USDX_close,USDJPY_close,USDJPY_ch
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df_full = df.copy()

#data = df.copy()
if n_tail > 0: df = df.tail(n_tail)
if lastBars > 0: df = df[: len(df)-lastBars]
df = df[['close']].dropna()


# 2. Daten skalieren
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
filePath_model =  folderPath_models +symbol+ ".keras"
model_name = filePath_model



X, y = [], []
for i in range(n_steps, len(scaled_data) - forecast_horizon):
    X.append(scaled_data[i - n_steps:i])
    y.append(scaled_data[i:i + forecast_horizon].flatten())

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))
import os 

check_model = os.path.exists(model_name) 
check_model = False
if check_model == False:
    

  # 4. RNN-Modell erstellen
  model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(n_steps, 1)),
    Dense(forecast_horizon)
   ])
  model.compile(optimizer='adam', loss='mean_squared_error')
  # 5. Modell trainieren
  model.fit(X, y, epochs=n_epochs, batch_size=n_batch_size, verbose=1)
  save_model(model, model_name)
  


# --- Modell speichern ---

model_pred = load_model(model_name)

# 6. Vorhersage für die nächsten 7 Wochentage
last_sequence = scaled_data[-n_steps:]
last_sequence = last_sequence.reshape((1, n_steps, 1))
predicted_scaled = model_pred.predict(last_sequence)
predicted_prices = scaler.inverse_transform(predicted_scaled)[0]


# 7. Nur Montag bis Freitag generieren
last_date = df.index[-1]
predicted_dates = []
day_offset = 1
while len(predicted_dates) < forecast_horizon:
    next_date = last_date + datetime.timedelta(days=day_offset)
    if next_date.weekday() < 5:  # 0=Montag, ..., 4=Freitag
        predicted_dates.append(next_date)
    day_offset += 1


# 8. Plot: Historische Daten + Vorhersage
df = df.tail(10)
df_his = df_his.set_index('date')
df_his = df_his.tail(lastBars)
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Historische Daten')
plt.plot(df_his.index, df_his['close'], label='IS-HIS')
plt.plot(predicted_dates, predicted_prices, marker='o', linestyle='--', color='red', label='Vorhersage (nur Wochentage)')
plt.title(symbol + ' Historie und ' + str(forecast_horizon) + ' Tage-Vorhersage (RNN)')
plt.xlabel('Datum')
plt.ylabel('Preis in USD')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


