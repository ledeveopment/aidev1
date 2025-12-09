import torch
#print("GPU verfügbar:", torch.cuda.is_available())
#print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
symbol  = "#PLTR"

import datetime
import pandas as pd
#from yahooquery import Ticker
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Fetch historical Tesla stock data using yahooquery

#end= "2025-01-01",
#historical_data = ticker.history(  start= "2020-01-01",  interval='1d')

folderpath_models ="/Users/Shared/ai_work/Trainingdata/models/autogluon/"
folderpath_traininglogs =" /Users/Shared/ai_work/Trainingdata/models/neuralprophet/traininglogs/"
#folderpath_charts = "D:\\OneDrive\\AI Workspace\\models\\neuralprophet\\charts\\"
folderpath_historie = "/Users/Shared/ai_work/Trainingdata/ml_data/yh_his/D1/"
folderpath_auto_models ="/Users/Shared/ai_work/Trainingdata/models//autogluon/"
#folderpath_analyse = "D:\\OneDrive\\AI Workspace\\results\\neuralprophet\\"
#folderpath_results = "D:\\OneDrive\\AI Workspace\\results\\nrp_results\\"


# 🟢 Step 1: Download Tesla stock data
#ticker = "TSLA"



#df = yf.download(ticker, period="5y", interval="1d")
data = pd.read_csv (folderpath_historie+symbol+".csv")
df_his = data.copy()
#historical_data = ticker.history(  period="max")
# Clean and prepare the data
#data = historical_data.reset_index()

data = data[['date', 'close']]
data.rename(columns={'date': 'timestamp', 'close': 'item_value'}, inplace=True)
lastBars = 20
n_tail = 400  #600
#data = df.copy()
df = data[:len(data) - lastBars]
if n_tail > 0: df = df.tail(n_tail)
df = df.reset_index()

print (df)
df['item_id'] = symbol
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a TimeSeriesDataFrame for AutoGluon
train_data = TimeSeriesDataFrame(df)

# Define the prediction length (100 days)
prediction_length = 100

# Train the AutoGluon model, explicitly setting the frequency
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    #path="AutogluonPredictor/",
    path=folderpath_auto_models +symbol+"//",
    #"D://OneDrive//AI Workspace//workspaces//dev_ws1//AutoGluon//AutogluonPredictor//models//",
    target="item_value",
    eval_metric="RMSE",
    freq="D" # Set the frequency to daily ('D')
)


predictor.fit(
    train_data,
    presets="best_quality",
    time_limit=300
)




# ... (rest of your prediction and plotting code)

# Generate predictions for the next 100 days
predictions = predictor.predict(train_data)

# Create a future dataframe for the next 100 days
future_timestamps = pd.date_range(
    start=df['timestamp'].max() + pd.Timedelta(days=1),
    periods=prediction_length,
    freq='D'
)

future_df = pd.DataFrame({'timestamp': future_timestamps})
future_df['item_id'] = 'TSLA'
future_df = TimeSeriesDataFrame(future_df)

# Predict the future values
future_predictions = predictor.predict(train_data, known_covariates=future_df)

# Combine the results for easier viewing
combined_results = pd.concat([
    df[['timestamp', 'item_value']],
    future_predictions.reset_index().rename(columns={'mean': 'predicted_value'})[['timestamp', 'predicted_value']]
], ignore_index=True)

# Print the last 100 predicted values
print(combined_results.tail(100))

#Optional: Plot the results
import matplotlib.pyplot as plt
combined_results = combined_results.tail(200)  # Limit to the last 200 entries for better visualization
df_his = df_his.tail(lastBars+3)  # Limit to the last 200 entries for better visualization
df_his['date'] = pd.to_datetime(df_his['date'])
plt.figure(figsize=(12, 6))
plt.plot(combined_results['timestamp'], combined_results['item_value'], label='Historical Close Price')
plt.plot(df_his['date'], df_his['close'], label='ISHIS', color='magenta')
plt.plot(combined_results['timestamp'].tail(100), combined_results['predicted_value'].tail(100), label='Predicted Close Price', color='red')
plt.xlabel('Date')
plt.ylabel( symbol + '  Close Price')
plt.title( symbol + ' Price Prediction')
plt.legend()
plt.grid(True)
plt.show()