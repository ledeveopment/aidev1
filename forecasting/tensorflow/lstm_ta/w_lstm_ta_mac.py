# Schritt 1: Bibliotheken importieren
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model,  load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras.saving import save_model
import time
import os

# Parameter
#symbol = 'TSLA'


# ====== SETTINGS ======
# 1. Daten abrufen
folderPath_His = "/Users/Shared/ai_work/Trainingdata/ml_data/yh_his/D1/"
folderPath_models = "/Users/Shared/ai_work/Trainingdata/models/lstm_ta/"
folderpath_results = "/Users/Shared/ai_work/Trainingdata/ml_results/lstm_ta/"
folderpath_result_charts = "/Users/Shared/ai_work/Trainingdata/ml_results/charts/lstm_ta_charts/"
# Schritt 5: Daten skalieren und Sequenzen vorbereiten
features = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_lower']
# ===== END SETTINGS ======

lastBars = 20
n_tail = 600
#n_pred_days = 60
#n_time_steps = 60 # Neuronal Netzwerk
#sequence_length = 60
n_epochs = 200 
n_batch_size = 16
pred_days = 60
look_back = 60
#data = df[['close']].dropna()

def convertDataframeResults (df, symbol, type, future_dates, future_predictions, df_his):
    #df_his = df_his.set_index('date')
    df_result = pd.DataFrame({
        "Symbol": symbol, 
        "Type":  type,
         "Date": future_dates,
         "Close": future_predictions,
         "Source": 0, 
         "Change": 0
    })
    
    lastclose = df['close'].iloc[-1]
    print ("lastclose", lastclose)
    change_percent =  []
    source = []
    prev_source = 0
   
    
    for i in range(len(df_result)):
        if i == 0:
        # Erste Zeile: Vergleich mit Basiswert
          #change = ((df_result.loc[i, 'Close'] - lastclose) / lastclose) 
          change = 0
           
          source_value  = lastclose
          #lastclose + change* lastclose
          prev_source = lastclose
        else:
         # Vergleich mit vorheriger Zeile
          prev_value = df_result.loc[i-1, 'Close']
          #prev_source = df_result.loc[i-1, 'Source']
          change = ((df_result.loc[i, 'Close'] - prev_value) / prev_value) 
          source_value =prev_source + change*prev_source
          prev_source = source_value
          #source_value = prev_value + change*prev_source
        change_percent.append(change)
        source.append(source_value)
    df_result['Change'] = change_percent
    df_result['Source'] = source_value
    

    return df_result

def displayChart (symbol, future_dates,pred_prices,  data, df_his):
    # Schritt 9: Ergebnisse plotten
    fig = go.Figure()
    
    
 
    #fig.figure(figsize=(12, 6))
    df_his = df_his.set_index('date')
    df_his = df_his.tail (lastBars)
    fig.add_trace(go.Scatter(x=data.index[-200:], y=data['close'][-200:], name='Historische Kurse'))
    fig.add_trace(go.Scatter(x=df_his.index, y=df_his['close'],  name='IS HIS'  ))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, name='Prognose (100 Tage)'))
    fig.update_layout(title=   '  Prognose mit LSTM und technischen Indikatoren', xaxis_title='Datum', yaxis_title='Preis (USD)')
    fig.show()
    filepath_chart  = folderpath_result_charts + symbol +".png"
    fig.write_image(filepath_chart)  # Save als PNG

def build_model(X, y, model_name):
    # Schritt 6: LSTM-Modell erstellen
    model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(pred_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    # Schritt 7: Modell trainieren
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch_size, verbose=1)
    
    save_model(model, model_name)
    return model 

def predictBySymbol (symbol):
    
    df = pd.read_csv (folderPath_His + symbol + ".csv")
    df_his = df.tail(lastBars)
    df_his["date"] = pd.to_datetime(df_his["date"]) 
    df = df[['date','open', 'high', 'low', 'close','volume']]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    data = df.copy()
    #data = df.copy()
    if n_tail > 0: data = data.tail(n_tail)
    if lastBars > 0: data = data[: len(data)-lastBars]
    # Schritt 4: Technische Indikatoren berechnen
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    change = data['close'].diff()
    gain = change.where(change > 0, 0).rolling(window=14).mean()
    loss = (-change.where(change < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26

    std = data['close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['SMA_20'] + (std * 2)
    data['Bollinger_lower'] = data['SMA_20'] - (std * 2)

    data.fillna(method='bfill', inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(look_back, len(scaled_data) - pred_days):
       X.append(scaled_data[i - look_back:i])
       y.append(scaled_data[i:i + pred_days, 3])  # close-Preis
    X, y = np.array(X), np.array(y)

    filePath_model =  folderPath_models +symbol+ ".keras"
    model_name = filePath_model
    model_pred =""
    if os.path.exists(model_name):
       model_pred = load_model(model_name)
    else:
       model_pred = build_model(X, y, model_name)
    # Schritt 8: Vorhersage für die nächsten 100 Handelstage
    last_seq = scaled_data[-look_back:]
    last_seq = np.expand_dims(last_seq, axis=0)

    pred_scaled = model_pred.predict(last_seq)[0]

    scaler_close = MinMaxScaler()
    scaler_close.fit(data[['close']])
    pred_prices = scaler_close.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=data.index[-1], periods=pred_days+1, freq='B')[1:]
    
    
    result_df = convertDataframeResults (df,symbol, "LSTM-TA", future_dates, pred_prices,df_his)
    #RESULT 
    result_df.to_csv(folderpath_results+symbol +".csv")


    #displayChart (symbol, future_dates, pred_prices, data, df_his)
    
checksymbol =""

listOfFiles = os.listdir(folderPath_His)
fckey = ".csv"
for entry in listOfFiles:
        s = entry.split (fckey)
        symbol = s[0]
        i_run = 1
       
        #Check Symbol for Check Only
        if checksymbol and symbol != checksymbol:
           i_run = 0

        if i_run == 1: 
         
          
          try: 
            
            print ("Running RNN " , symbol)
            predictBySymbol (symbol)
            
           
            
          except:
             print ("Error RNN  " , symbol)
