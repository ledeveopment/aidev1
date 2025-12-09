# %%
#https://www.kaggle.com/spidy20/stock-market-prediction-with-decision-tree
#https://www.kaggle.com/spidy20/stock-market-prediction-with-decision-tree

from calendar import week
from symtable import Symbol
from typing import no_type_check_decorator
from xmlrpc.client import DateTime
from numpy import full
import pandas as pd #For data related tasks
#import matplotlib.pyplot as plt #for data visualization 
#import quandl #Stock market API for fetching Data
from sklearn.tree import DecisionTreeRegressor # Our DEcision Tree classifier
#import build_markets as markets
#import seaborn as sns
#quandl.ApiConfig.api_key = ''## enter your key 
#stock_data = quandl.get('NSE/INFIBEAM', start_date='2018-12-01', end_date='2018-12-31')
#Let's see the data
#Z:\Projects\ML\data\histories\full
#folderpath_his = "G:\\My Drive\\ml_data\\"


#"Z:\\Projects\\ML\\data\\histories\\full\\"
#folderpath_his = "Z:\\AI Projects - Predication\AI_WORKSPACE\\data\\histories\\full"
#folderpath_results = "D:\\Projects\\ai_system\\data\\decision\\results\\"
from joblib import dump, load
import ta
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
import csv 
import os
import matplotlib.pyplot as plt
#import pyodbc
# DB 
#server = 'tcp:ltcroot' 
server = 'tcp:HAG-11GGDD2' 
server_fc = 'tcp:HAG-11GGDD2' 
database = 'ltcdata' 
username = 'sa' 
password = 'LamKhue?2' 

#cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
from sklearn.model_selection import train_test_split

#-------- CSV HIS -----
#symbol = '#PYPL'
retrain = 0
#BarsLastHIS = 0
#n_preds = 20
#n_train_tail = 100
#n_preds 20 > n_train_tail 100
#hisdays = 200
#ticker_data_filename = folderpath_his+    "full_all_D1.csv"
#df_full = pd.read_csv (ticker_data_filename)
folderpath_his ="G:\\My Drive\\ml_data\\yh_his\\"
filepath_symbols = folderpath_his +  "symbollist.csv"
folderpath_models = "/Users/lehakhanh/Documents/ai_dev/ml_tensorflow-main/decisiontree/testdev/models/"
folderpath_df = "D:\\Projects\\ML\\df\\"

#df_symbol = pd.read_csv( filepath_symbols)
#----------- BIG QUERY ----------------
from google.cloud import bigquery
PROJECT_ID = 'keen-oasis-204713'
client = bigquery.Client()
# create a new datset
#client.create_dataset("new_dataset")
dataset_id = 'finance'
table_id = 'aitrend'
#client.delete_table(f"{PROJECT_ID}.{dataset_id}.{table_id}", not_found_ok=True)  # Make an API request.
#client.create_table(f"{PROJECT_ID}.{dataset_id}.{table_id}")

#Dataset(DatasetReference('kaggle-bq-test-248917', 'new_dataset'))
# some variables
filename = "G:\\My Drive\\ml_results\\ds\\aitrend.csv" # this is the file path to your csv
#filename = r'C:\ai_work\symbols\test.csv' # this is the file path to your csv
#----------- BIG QUERY ----------------



# tell the client everything it needs to know to upload our csv
dataset_ref = client.dataset(dataset_id)
table_ref = dataset_ref.table(table_id)
job_config = bigquery.LoadJobConfig()
job_config.source_format = bigquery.SourceFormat.CSV
job_config.autodetect = True
job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
folderpath_jobs ="G:\\My Drive\\ml_jobs\\"


#listSymbol()
"""
full_data_D1 = pd.read_csv (folderpath_his + "full_all_D1.csv")
full_data_W1 = pd.read_csv (folderpath_his + "full_all_W1.csv")
full_data_MON1 = pd.read_csv (folderpath_his + "full_all_MON1.csv")
full_data_1Q = pd.read_csv (folderpath_his + "full_all_1Q.csv")
full_data_2Q = pd.read_csv (folderpath_his + "full_all_2Q.csv")
full_data_1Y = pd.read_csv (folderpath_his + "full_all_1Y.csv")
"""
today = str(datetime.now().date() ) 


# %%
def uploadFileToBQ (filename):
    
  # load the csv into bigquery
  with open(filename, "rb") as source_file:
     job = client.load_table_from_file(source_file, table_ref , job_config=job_config )

  job.result()  # Waits for table load to complete.

  # looks like everything worked :)
  print("Loaded {} rows into {}:{}.".format(job.output_rows, dataset_id, table_id))
  
def trainsModelMarkets (  symbol, stock_data,  s_key, n_preds, n_tail , type ):

       lastBars = len(stock_data)-n_preds
       df_train = stock_data[:lastBars]
       df_train.dropna(inplace=True)
      
       
       #print (" Df TRAIN ----------------------------" , lastBars )
       #print ( df_train)
       
       #stock_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
       df_train.dropna(inplace=True)
       col_no = len(df_train.columns)
       col_train = col_no -2
       col_fc = col_no -1
       
       df_train = df_train.tail (n_tail)
       
       #print (df_train)

       #df = stock_data.drop(['next_close'], axis=1, inplace=True)
       
       #print (df_train.info())
       #print ("train ", df_train[["date","close","next_close"]])
       x = df_train.iloc[:,1:col_train].copy()
      
       y = df_train.iloc[:,col_fc].copy()

       #x = df_train.iloc[:,1:123].copy()
       #y = df_train.iloc[:,124].copy()
       
       

       x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
       

       Classifier = DecisionTreeRegressor()
       Classifier.fit(x_train,y_train)
       
       model_filepath  = folderpath_models+symbol + s_key+"_"+type + "_"+str(n_preds)+"_.joblib"
       dump(Classifier, model_filepath)
       clf = load(model_filepath)
       return clf 

# %%
def getDataFrame (symbol, timeframe ):
    
    #folder = "G:/My Drive/ml_data/yh_his/"+timeframe +"/"
    folder = "/Users/lehakhanh/Documents/ai_dev/data/yh_his/"+timeframe +"/"
    df = pd.read_csv (folder+symbol+".csv" )
    #df = df[:len(df)-0]
    df = df[["date","open","high","low","close","volume","adjclose"]]
    lastDate = str(df['date'].iloc[ len(df) -1])
    last_his_date = datetime.strptime(lastDate, '%Y-%m-%d' ).date()
    wday = last_his_date.weekday()
    day = last_his_date.day
    #print ( " TF ", timeframe ,  " Res: ", wday, " Day ", day )
    #if (timeframe =="W1" and wday != 0 and wday != 6) or ( (timeframe =="MON1" or timeframe =="MON3")  and day != 1):
    #    df = df [:len(df) -1]
    #print (df)
    #df = df [:len(df)-0]
    df.sort_values(by=['date'], inplace=True, ascending=True)
    #if timeframe =="D1":
    #    df = df.tail(400)
    #print (df)
    df = df[:len(df) - 0]
    df = df.tail (1000)
    if (timeframe =="D12"):
            df = df.tail (200)
    #print ("TF " , timeframe, " : ", df )
    return df


# %%
def convertWeekdate (fc_date, source_datetime):
    w_day = fc_date.weekday() +2 
    source_w_day = source_datetime.weekday() + 2
    if  w_day == 8 and source_w_day  <= 6:
            fc_date = fc_date  + timedelta(days = 1)
            
    if w_day == 7 and source_w_day  <= 6:
                fc_date = fc_date  + timedelta(days = 2)
                
    return str(fc_date)

# %%
def getFCdate (timeframe, his_datetime, n_preds):
        forecast_datetime  = datetime.strptime(his_datetime, '%Y-%m-%d' ).date() 
        source_datetime = forecast_datetime
        n_adddays = (n_preds /4)*7
        forecast_datetime = forecast_datetime  + timedelta(days = n_adddays)
        """
        if timeframe =="D1":
                delta = relativedelta(hours=23, day=0)
         
                forecast_datetime = forecast_datetime + delta
        if timeframe != "D1":
                forecast_datetime = forecast_datetime  + timedelta(days = n_preds)

        """
        forecast_datetime = convertWeekdate (forecast_datetime, source_datetime)

        return forecast_datetime


# %%
  
def getPredictMulti_SeriesMarkets ( symbol, timeframe, stock_data,  n_preds, fctype, writer, retrain, BarsLastHIS):

       #1.15083
       
       
       
       #stock_data.set_index('date', inplace=True)
       #stock_data = df_data

       #stock_data['date']= pd.to_datetime(stock_data['date'])
       #print ("tock_data", stock_data)
       #print (stock_data["date"])
       #stock_data = pd.DataFrame (stock_data)
       #pd.read_csv (filepath)
       #= pd.DataFrame(df_data)
       #stock_data = stock_data[:len(stock_data) - BarsLastHIS]
       
       
       
       #lastdate_df = str(stock_data["date"].iloc[len(stock_data)-1])
       #last_his_date = datetime.strptime(lastdate_df, '%Y-%m-%d %H:%M:%S' ).date()
       #lastDate = last_his_date
       print (stock_data)
       lastDate = stock_data["date"].iloc[len(stock_data) -1]
       print (lastDate)
       #lastDate = str(stock_data['date'].iloc[ len(stock_data) -1])
       #lastDate = str(stock_data[0].iloc[ len(stock_data) -1])
       lastClose = str(stock_data['close'].iloc[ len(stock_data) -1])
       
       #stock_data.tail (20)
       #stock_data = ta.add_all_ta_features(stock_data, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
       s_key = "_C"
       if fctype =="(C)":
         stock_data['next_close'] = stock_data['close'].shift(-n_preds)
        
        
       if fctype =="(H)":
           stock_data['next_close'] = stock_data['high'].shift(-n_preds)
           s_key = "_H"
       if fctype =="(L)":
           stock_data['next_close'] = stock_data['low'].shift(-n_preds)
           s_key = "_L"
      
       #print (" Df STOCK DATA  ----------------------------")
       #print ( stock_data)

       
       
       
       #df_preds = df_data.tail(future_steps)
       #df_preds = stock_data [ len(stock_data) - future_steps+1 : len(stock_data) - future_steps +2 ]
       df_preds = stock_data.tail(1)
       print (stock_data)
       #ORG df_preds = stock_data.tail(n_preds)
       #
       # 
       # = stock_data.tail(1)

       
       
      

       s_type = timeframe+"-ML-"
       algo = s_type+fctype
       #if ( n_preds == 1): algo = "D1-ML-"+fctype
       #if ( n_preds == 5): algo = "W1-ML-"+fctype
       #if ( n_preds == 20): algo = "MON1-ML-"+fctype

       algo1= "DS-ML"
       
       his_date =""
       df_preds = pd.DataFrame (df_preds)
       
       i = 0 
       print ( "PREDS", df_preds)
       fc_list =[]
       model_filepath  = folderpath_models+timeframe + "\\" +symbol +s_key+".joblib"
       #model_filepath  = "C:\\ai_work\\ML\\models\\ds\\W2\\" +symbol +".joblib"
       file_exists = os.path.exists(model_filepath)
       clf = ""
      
       if (file_exists == False or retrain == 1 ) :  
                print ("try Train Models ")
                clf  = trainsModelMarkets (symbol, timeframe, stock_data, n_preds, fctype, s_key )
       else: 
                clf = load(model_filepath)

       for row in df_preds.values:
              i = i +1
              his_date = row[0]
              his_close = row[4]
              if fctype =="(H)":
                his_close = row[2]
              if fctype =="(L)":
                his_close = row[3]

              
              #q = row[:,1:92]
              query =[]
              q= row.tolist()[1:col_train]
              print ("q ", q)
              #q= row.tolist()[1:123]
              query.append (q)
              
              source_close = his_close
              #q = q.values 

              #query  = row.tolist()
              #print ( his_date , " Q: " , query)
              #df_x = row.iloc[:,1:92].copy()
              # print ( " DF X " , df_x.values.tolist())
              #print (row)

              #print (q)
              
              print ("query ", query)
                
              if  (  i > 0  ): 
                 #pred_C = Classifier.predict(query)[0]
                 pred_C = clf.predict(query)[0]
                 
                 #if fctype =="(C)":
                 fc_list.append(pred_C)
                 pred_change = 0
                 if float(pred_C) > 0  and float (source_close) > 0:
                     pred_change = round ( (float(pred_C)  - float(source_close))/float(source_close) , 3  )

                 fc_date = getFCdate (timeframe, his_date, n_preds)

                 #print (symbol , " tf: ", algo,  " t: ",  fctype ,  " his_date", his_date , " his_close" , his_close,  " -- > fc date ", fc_date , " FC: ",  pred_C)
                 """
                 if fctype =="(H)" or fctype =="(L)":
                     writer.writerow({'Symbol': symbol, 'Type': algo,  'Date': str(his_date) , 'Close': str(pred_C), 'Source': str(his_close), 'Change': "0" }) 
                     writer.writerow({'Symbol': symbol, 'Type': algo,  'Date': str(fc_date) , 'Close': str(pred_C), 'Source': str(source_close), 'Change': str(pred_change) }) 
                 """

                 #writer.writerow({'Symbol': symbol, 'Type': algo,  'Date': str(his_date) , 'Close': str(his_close), 'Source': str(his_close), 'Change': "0" }) 
                 #writer.writerow({'Symbol': symbol, 'Type': algo,  'Date': str(fc_date) , 'Close': str(pred_C), 'Source': str(source_close), 'Change': str(pred_change) }) 

                 #if fctype =="(C)":
                 #       writer.writerow({'Symbol': symbol, 'Type': algo1,  'Date': str(fc_date) , 'Close': str(pred_C), 'Source': str(source_close), 'Change': str(pred_change) }) 
                 
       
       
       #his_date = lastDate
       
       fc_date = getFCdate (timeframe, lastDate, n_preds)
       
       fc_list = pd.DataFrame (fc_list)
       
       fc_mean = round ( fc_list.mean()[0], 4) 
       fc_max = round ( fc_list.max()[0], 4) 
       fc_min = round ( fc_list.min()[0] , 4)
       fc_end =  fc_list.values[len(fc_list) -1 ]
       fc_end = fc_mean
     

# %%

def checkDSBySymbol (symbol,  retrain, BarsLastHIS):
        folderpath_results ="G:\\My Drive\\ml_results\\ds\\"
        csvfileresult  = folderpath_results+symbol + ".csv"
        
        df_data = getDataFrame (symbol, "D1")
        
        
        df_data.set_index('date', inplace=True)
        #frames = [df_data, df_futures ]
        df_res = df_data
        
        
        #df_res = pd.concat([df_data, df_futures], axis=1, join="inner")
        
        #df = pd.DataFrame (df_res)
        
        #df_res= df_res["date","open","high","low","close","volume", "sp500_open","sp500_close","vix_close","oil_open","oil_volume"]
        
        
        filepath = folderpath_df +symbol +".csv"
        df_res.to_csv (filepath)
        
        df = pd.read_csv (filepath)
        
        #df_data.sort_values(by=['time'], inplace=True, ascending=True)
        df.dropna(inplace=True)
        
        #print ( " DF " , df)

        #df_res = pd.concat (frames)
        
        weekday  = datetime.now().weekday()

        print (" ************************ weekday ", weekday)
        
        with open(csvfileresult , 'w', newline='') as csvfilerow:
             fieldnames = ['Symbol','Type','Date','Close','Source','Change']
             writer = csv.DictWriter(csvfilerow, fieldnames=fieldnames)
             writer.writeheader()
             
             #prevTrain (symbol, "D1", 1, writer, df_data)
             #prevTrain (symbol, "D2", 2, writer, df_data)
             #prevTrain (symbol, "D1", 5, writer, df_data)
             
             n_preds = 1 
             
             #prevTrain (symbol, "D1", n_preds, writer, df , retrain, BarsLastHIS )
             getPredictMulti_SeriesMarkets  ( symbol, "D1", df_data, n_preds, "(C)", writer, retrain, BarsLastHIS)      


# %%


#folderpath = '/Users/lehakhanh/Documents/ai_dev/data/yh_his/D1/'
folderpath = 'M:\\My Drive\\analysis\\data\\df_features\\'
folderpath_models ="D:\\Projects\\ML\\models\\DS\\features\\"
#"C:\\ai_work\\GitHub\\ml_tensorflow\\decisiontree\\models\\"




# %%

#df_pred_H = pd.DataFrame( columns=['date','prediction'])
#df_pred_L = pd.DataFrame( columns=['date','prediction'])
 
def getPredictionList (symbol, stock_data, s_key, n_preds, n_trail , type,writer ):
   
   #stock_data['next_high'] = stock_data['high'].shift(-n_preds)
   #stock_data['next_low'] = stock_data['high'].shift(-n_preds)
   #df_preds = stock_data.tail(n_preds)
   
   fc_type = type
       
   clf =""
   if s_key =="_C":
      stock_data['next_close'] = stock_data['close'].shift(-n_preds)
      clf  = trainsModelMarkets ( symbol,  stock_data,  s_key, n_preds,  n_trail , type)
   if s_key =="_H":
      stock_data['next_close'] = stock_data['high'].shift(-n_preds)
      clf  = trainsModelMarkets ( symbol, stock_data,  s_key, n_preds,  n_trail ,type )
   if s_key =="_L":
      stock_data['next_close'] = stock_data['low'].shift(-n_preds)
      clf  = trainsModelMarkets (symbol, stock_data,  s_key, n_preds,  n_trail , type )
   col_no = len(stock_data.columns)
   col_train = col_no -2
   col_fc = col_no -1
   
   df_preds = stock_data.tail(1)
   df_pred = pd.DataFrame( columns=['date','prediction'])
   n_weeks = n_preds /5
 
   pred = 0
   i = 0
   for row in df_preds.values:
              i = i +1
              his_date = row[0]
              his_close = row[1]
              his_high = row[4]
              his_low = row[5]
          
              #q = row[:,1:92]
              query =[]
              q= row.tolist()[1:col_train]
              
              #q= row.tolist()[1:123]
              query.append (q)
              
              source_close = his_close
              
              his_lastclose = his_close
              
                                      
              if s_key =="_H": 
                 his_lastclose =his_high
               
                 
              if s_key =="_L": 
                 his_lastclose =his_low
               
              
              
              new_row = {"date": datetime.strptime(  his_date, '%Y-%m-%d' ).date() , "prediction": his_lastclose}
              df_pred = df_pred.append(new_row, ignore_index=True)
              
              
              if n_weeks >= 1: 
                    fc_date = datetime.strptime(  his_date, '%Y-%m-%d' ).date()    +  relativedelta(weeks=n_weeks)
                    
              if n_weeks < 1:
                    fc_date = datetime.strptime(  his_date, '%Y-%m-%d' ).date()    +  relativedelta(days=n_preds)
              w_day = fc_date.weekday() +2 
              
              if ( w_day == 8):
                        fc_date = fc_date  + timedelta(days = 1)
              if ( w_day == 7):
                fc_date = fc_date  + timedelta(days = 2)
                
                
              pred = clf.predict(query)[0]
              new_row = {"date": fc_date, "prediction": pred}
              df_pred = df_pred.append(new_row, ignore_index=True)
              fc_change = round ( (float(pred) - float (his_lastclose))/his_lastclose , 3)
              writer.writerow({'Symbol': symbol, 'Type': fc_type,  'Date': str(fc_date) , 'Close': str(pred), 'Source': str(his_lastclose), 'Change': str(fc_change) })
              #df_pred_Sum.append(new_row, ignore_index=True)
              
              
              
              
   return df_pred   
       

     

# %%
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# %%
def getPredData (df, prediction):
  #print (prediction)
  lastdate_df = str(df['date'].iloc[len(df)-1])
  lastclose_df = str(df['close'].iloc[len(df)-1])
  #last_his_date = datetime.strptime(lastdate_df, '%Y-%m-%d %H:%M:%S' ).date()
  last_his_date = datetime.strptime(lastdate_df, '%Y-%m-%d' ).date()
  #print (prediction)
  fc_date = last_his_date
  df_pred = pd.DataFrame( columns=['date','prediction'])
  
  
  #data = pd.DataFrame( columns=['date','prediction'])
  days = []
  #for i in range(0, len(prediction)):
  i = 0 
  for s  in  prediction.values:
    i = i+ 1
    #print(f"Day {i}: Predicted Price = {prediction[0]}")
    #print ( fc_date  )
    #pred = prediction[i][0]
    
    pred = s
    
    fc_date = fc_date + timedelta (days= 1)
    w_day = fc_date.weekday() +2 
    if ( w_day == 8):
            fc_date = fc_date  + timedelta(days = 1)
    if ( w_day == 7):
                fc_date = fc_date  + timedelta(days = 2)
                
    #df1 = pd.DataFrame({"date": fc_date,"prediction": pred},  index=["date"])
    df_pred.append ([fc_date, pred])
    
    new_row = {"date": fc_date, "prediction": pred}
    # Add a single row using the append() method
    df_pred = df_pred.append(new_row, ignore_index=True)
    days.append ( i)
    
   
  
  #print ("df pred ", df_pred)
  return df_pred

# %%
def getPredDataLinear (df, pred):
  #print (prediction)
  lastdate_df = str(df['date'].iloc[len(df)-1])
  lastclose_df = str(df['close'].iloc[len(df)-1])
  #last_his_date = datetime.strptime(lastdate_df, '%Y-%m-%d %H:%M:%S' ).date()
  last_his_date = datetime.strptime(lastdate_df, '%Y-%m-%d' ).date()
  #print (prediction)
  fc_date = last_his_date
  df_pred = pd.DataFrame( columns=['date','prediction'])
  new_row = {"date": last_his_date, "prediction": lastclose_df}
  df_pred = df_pred.append(new_row, ignore_index=True)
  n_week = n_preds/5
  
  if n_weeks >= 1: 
      fc_date = datetime.strptime(  lastdate_df, '%Y-%m-%d' ).date()    +  relativedelta(weeks=n_weeks)
  if n_weeks < 1: 
      fc_date = datetime.strptime(  lastdate_df, '%Y-%m-%d' ).date()    +  relativedelta(days=n_preds)
  
  new_row = {"date": fc_date, "prediction": pred}
  df_pred = df_pred.append(new_row, ignore_index=True)
   
  df_pred = pd.DataFrame(df_pred , columns=['date','prediction'])
  #print ("df pred ", df_pred)
  return df_pred

# %%
def displayChart ( title, df_full, df_pred, df_pred1, df_pred2, df_pred3): 
  
  # Visualizing the data
  #datetime.strptime(lastdate_df, '%Y-%m-%d %H:%M:%S' ).date()
  df_display = df_full.tail(60+BarsLastHIS*1)
  df_display['date']= pd.to_datetime(df_display['date'])
  
  
  #df_pred['date'] = pd.to_datetime(df['date'])
  #df_pred['date'] = df_pred['date'].datetime.strftime('%Y-%m-%d')
  #df_pred.set_index('date', inplace=True)
    
  plt.figure(figsize=(16,6)) 
  plt.title( title)
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price', fontsize=18)
  plt.plot(df_display['date'], df_display['close'])
  #plt.plot(datetime.strptime(df_display['date'], '%Y-%m-%d %H:%M:%S' ).date() , df_display['close'])
  print ( "df_pred3: ", df_pred3)
  plt.plot(df_pred['date'],df_pred['prediction'])
  plt.plot(df_pred1['date'],df_pred1['prediction'])
  plt.plot(df_pred2['date'],df_pred2['prediction'])
  #plt.plot(df_pred3['date'],df_pred3['prediction'])
   #plt.plot(df1['date'].iloc[n_length_train:],valid[['close', 'Predictions1']])
  #plt.plot(df1['date'].iloc[n_length_train:],valid[['close']])
  #plt.plot(df_full['date'],df_full['close'])

  plt.legend(['ACTUAL', "PRED-CLOSE", "PRED-HIGH", "PRED-LOW"])
  plt.show()
  


# %%


# %%
def checkForecastDataFeatures (symbol,  BarsLastHIS, writer ):
         
   
          
   type ="DS-FEATURES" 
   df_data = pd.read_csv (folderpath+symbol+".csv" )
   df_data = df_data[["date", "close", "volume", "sp500_close", "sp500_volume", "oil_close","oil_volume", "tn2y_close", "tn5y_close", "udx_close", "ty5Y_close", "tn10y_close", "ty10Y_close","ty30Y_close"]]
   df_full = df_data
   df_full.dropna(inplace=True)
   print (df_full)
   df_pred_Sum = pd.DataFrame( columns=['date','prediction'])
   lastDate = df_full['date'].iloc[ len(df_full) -1]
   lastClose= df_full['close'].iloc[ len(df_full) -1]
   #lastDate =  datetime.strptime(lastDate, '%Y-%m-%d' ).date()
    
   
   new_row = {"date": lastDate, "prediction": lastClose}
   df_pred_Sum = df_pred_Sum.append(new_row, ignore_index=True)
   print (" df_pred_Sum ", df_pred_Sum)

   #df = df[:len(df)-0]
   #df_data = df_data[["date","open","high","low","close","volume","adjclose"]]
   #df_data = df_data[["date", "adjclose", "volume", "open", "high", "low","close","sp500_close","sp500_volume","oil_close","oil_volume","tn2y_close","tn2y_volume","tn5y_close","tn5y_volume","tn10y_close","tn10y_volume","tn5y_close","tn5y_volume","tn10y_close","tn10y_volume","ty10Y_close"]]

   df_data = df_data[:len(df_data) - BarsLastHIS]
   
   #df_data.set_index('date', inplace=True)
   #print ( df_data["date"] ) 
   stock_data = df_data
   
   """
   ORG
   df_pred = getPredictionList ( symbol, stock_data, "_C", n_preds,  n_train_tail)
   df_pred_H = getPredictionList (symbol, stock_data, "_H" , n_preds,  n_train_tail)
   df_pred_L = getPredictionList (symbol, stock_data, "_L" , n_preds,  n_train_tail)
   """
   
   
   #getPredictionList (symbol, stock_data, "_H" , n_preds,  n_train_tail)
   
   n_preds = 5
   n_train_tail = 20
   df_pred_S = getPredictionList ( symbol, stock_data, "_C", n_preds,  n_train_tail, type, writer)
   
   n_preds = 10
   n_train_tail = 60
   df_pred_M =  getPredictionList ( symbol, stock_data, "_C", n_preds,  n_train_tail, type,writer)
   
   
   n_preds = 100
   n_train_tail = 200
   df_pred = getPredictionList ( symbol, stock_data, "_C", n_preds,  n_train_tail, type, writer)
   
   #df_pred = pd.DataFrame(df_pred , columns=['date','prediction'])
   title =" Futures of " + symbol
   
   
   
   
   
              
   #print (" df_pred_Sum ", df_pred_Sum)
   print (df_pred_S)
   #df_pred_Sum = df_pred_Sum.append(new_row, ignore_index=True)
   new_row = df_pred_S[1:len(df_pred_S) ]
   #new_row = {"date": lastDate, "prediction": lastClose}
   print ("new_row ", new_row)
   #df_pred_Sum = df_pred_Sum.append(new_row, ignore_index=True)
   #df_pred_Sum = df_pred_Sum.append(df_pred_S[:len(df_pred_S) - 1], ignore_index=True)
   #df_pred_Sum.append (df_pred_S[:len(df_pred_S) - 1])
   print (" df_pred_Sum " , df_pred_Sum)
   
   #displayChart ( title, df_full, df_pred, df_pred_M, df_pred_S, df_pred_Sum)

# %%
def checkForecastDataBase (symbol,  BarsLastHIS, writer ):
   type ="DS-BASE" 
   df_data = pd.read_csv (folderpath_his+"D1\\"+symbol+".csv" )
   df_full = df_data

   #df = df[:len(df)-0]
   #df_data = df_data[["date","open","high","low","close","volume","adjclose"]]
   df_data = df_data[["date", "adjclose", "volume", "open", "high", "low","close"]]

   df_data = df_data[:len(df_data) - BarsLastHIS]
   df_data.dropna(inplace=True)
   #df_data.set_index('date', inplace=True)
   #print ( df_data["date"] ) 
   stock_data = df_data
   
   
   
   
   #getPredictionList (symbol, stock_data, "_H" , n_preds,  n_train_tail)
   
   n_preds = 5
   n_train_tail = 20
   df_pred_L = getPredictionList ( symbol, stock_data, "_C", n_preds,  n_train_tail, type, writer)
   
   n_preds = 10
   n_train_tail = 30
   df_pred_H =  getPredictionList ( symbol, stock_data, "_C", n_preds,  n_train_tail, type, writer)
   
   
   
   n_preds = 100
   n_train_tail = 200
   df_pred = getPredictionList ( symbol, stock_data, "_C", n_preds,  n_train_tail, type,  writer)
   
   #getPredictionList (symbol, stock_data, "_L" , n_preds,  n_train_tail)
   
   
   #df_pred = pd.DataFrame(df_pred , columns=['date','prediction'])
   title =" Base of " + symbol
   
   #displayChart (title, df_full, df_pred, df_pred_H, df_pred_L, df_pred_L)

# %%
#symbol ="GBPUSD"
BarsLastHIS = 0
n_preds = 100
n_train_tail = 200

fc_symbol = ""
folder ="G:\\My Drive\\ml_data\\yh_his\\D1\\"
folder_results =  "G:\\My Drive\\ml_results\\ds_features\\"


current_date_time = datetime.now()
# Get only the current date
current_date = current_date_time.date()

def checkTehnical (symbol, lastclose ):
   result = []
   
 
   filepath_df ="G:\\My Drive\\ml_results\\technical\\"+symbol + ".csv"
   if os.path.isfile(filepath_df) == False: return result
   df_fc = pd.read_csv (filepath_df)
   print (symbol, df_fc, " len ", len (df_fc))
   #key = 'D1'
   if len(df_fc) == 0:  return result

   ta_trend_S = df_fc["S"][0]
   ta_trend_M = df_fc["M"][0]
   ta_trend_L = df_fc["L"][0]
   ta_trend = 0; 
   if ta_trend_S > 0 and ta_trend_M > 0 and ta_trend_L > 0: ta_trend = 1
   if ta_trend_S > 0 and ta_trend_M > 0 and ta_trend_L >= 0 and ta_trend == 0 : ta_trend = 2
   if ta_trend_S >= 0 and ta_trend_M > 0 and ta_trend_L >= 0 and ta_trend == 0 : ta_trend = 3

   if ta_trend_S < 0 and ta_trend_M < 0 and ta_trend_L < 0: ta_trend = -1
   if ta_trend_S < 0 and ta_trend_M < 0 and ta_trend_L <= 0 and ta_trend == 0 : ta_trend = -2
   if ta_trend_S <= 0 and ta_trend_M < 0 and ta_trend_L <= 0 and ta_trend == 0 : ta_trend = -3
   
   
   ta_signal = df_fc["ta_signal"][0]
   
   
   result = (ta_trend_S, ta_trend_M, ta_trend_L , ta_trend )
   print ("=====TECHNICAL:ta_trend_S, ta_trend_M, ta_trend_L , ta_trend", result)
   return (result)
  
   
   
   
   
def checkLinear (symbol, lastclose ):
   print ( "+++ Linear  ", symbol)

   result = []
   filepath_df ="G:\\My Drive\\ml_results\\linear\\"+symbol + ".csv"
   if os.path.isfile(filepath_df) == False: 
       print (" NO LINEAR FC ", symbol)
       return result
   
   df_fc = pd.read_csv (filepath_df)

        
   #df_fc.set_index('Type', inplace=True)
   # Retrieve a row by key (index value)
   #key = 'FB-10D-trend'
   trend_S = 0 
   trend_M = 0 
   trend_L = 0
   signal_S = 0
   tp1 = 0
   tp2  = 0 
   tp3 = 0 
   
   for row in df_fc.values: 
      st_type = row [1]
      fc_close = row [3]
      if st_type == "LN-D1-close" :
           trend_S = round ( (fc_close - lastclose ) /lastclose , 3)
           tp1 = fc_close
      if st_type == "LN-W1-close" :
           trend_M = round ( (fc_close - lastclose ) /lastclose , 3)
           tp2 = fc_close
      if st_type == "LN-MON1-close" :
           trend_L = round ( (fc_close - lastclose ) /lastclose , 3)
           tp3 = fc_close 
      
      #print (row)
   df_fc.set_index('Type', inplace=True)
   # Retrieve a row by key (index value)
   key = 'LN-D1-close'
   row = df_fc.loc[key]
   
   
   D1_High = df_fc.loc['LN-D1-high']['Close']
   D1_Low = df_fc.loc['LN-D1-low']['Close']
   D1_Close = df_fc.loc['LN-D1-close']['Close']
   
   
   change_D1_High = round((D1_High - lastclose)/lastclose, 4) 
   change_D1_Low = round((D1_Low - lastclose)/lastclose, 4) 
   change_D1_Close = round((D1_Close - lastclose)/lastclose, 4) 
   
   trend_S = change_D1_Close
  
   
   key = 'LN-W1-close'
   row = df_fc.loc[key]
   W1_High = df_fc.loc['LN-W1-high']['Close']
   W1_Low = df_fc.loc['LN-W1-low']['Close']
   W1_Close = df_fc.loc['LN-W1-close']['Close']
   
   
   change_W1_High = round((W1_High - lastclose)/lastclose, 4) 
   change_W1_Low = round((W1_Low - lastclose)/lastclose, 4) 
   change_W1_Close = round((W1_Close - lastclose)/lastclose, 4) 
   
   trend_M = change_W1_Close
   
   
   key = 'LN-MON1-close'
   row = df_fc.loc[key]
   
   MON1_High = df_fc.loc['LN-MON1-high']['Close']
   MON1_Low = df_fc.loc['LN-MON1-low']['Close']
   MON1_Close = df_fc.loc['LN-MON1-close']['Close']
   
   change_MON1_High = round((MON1_High - lastclose)/lastclose, 4) 
   change_MON1_Low = round((MON1_Low - lastclose)/lastclose, 4) 
   change_MON1_Close = round((MON1_Close - lastclose)/lastclose, 4) 
   
   trend_L = change_MON1_Close
   tp1 = D1_Close
   tp2 = W1_Close 
   tp2 = MON1_Close 
   if trend_S > 0: 
      signal_S = df_fc.loc['LN-D1-low']['Close'] - lastclose
      tp1 = df_fc.loc['LN-D1-high']['Close']
      
   if trend_S < 0:  
      signal_S = round ( (df_fc.loc['LN-D1-high']['Close'] - lastclose )/lastclose, 4)
      tp1 = df_fc.loc['LN-D1-low']['Close']
      
   
   if trend_M > 0: 
      signal_M =round ((df_fc.loc['LN-W1-low']['Close'] - lastclose)/lastclose, 4) 
      tp_M = df_fc.loc['LN-W1-high']['Close']
      
   if trend_M < 0:  
      signal_M = round ( (df_fc.loc['LN-W1-high']['Close'] - lastclose )/lastclose, 4)
      tp1 = df_fc.loc['LN-W1-low']['Close']
   
   
   signal = 0 
  
   if (trend_S > 0 and (signal_S >0 or signal_M > 0 )): signal = 1
   if (trend_S <  0 and (signal_S < 0 or signal_M < 0 )): signal = -1
   
   
   
   result = (tp1 ,  tp2,   tp3, trend_S,  trend_M, trend_L )
   
   print ( "=====LINEAR: tp1 ,  tp2,   tp3, trend_S, trend_M , trend_L", result)
   return result
   
   
def checkTensorTS (symbol, lastclose):
   result = []
   ts_trend = 0
   ts_sihnal = 0 
   trend_S = 0 
   trend_M = 0 
   trend_L = 0 
   tp1 = 0 
   tp2 = 0 
   tp3 = 0
   tp4 = 0 


   pricelevel1= 0 
   pricelevel2= 0
   #1d
   trend_1d = 0
   trend_5d = 0 
   trend_20d = 0
   trend_60d = 0 
   
   
   filepath_ts = "D:\\OneDrive\\AI Workspace\\results\\ts\\ts1\\"+symbol + ".csv"
   #"G:\\My Drive\\ml_results\\ts_results\\ts1\\"+symbol + ".csv"

   
   if os.path.isfile(filepath_ts) == True: 
     df_ts_D1 = pd.read_csv (filepath_ts)
     print ( len(df_ts_D1))
     i = 0 
    
     if len(df_ts_D1) > 0: 
         
       
       df_ts_D1.set_index('Type', inplace=True)
       D1_close = df_ts_D1.loc['TS-1D-TR']['Close']
       print ( " D1_close ", D1_close, " lastclose : ", lastclose)
       trend_1d =round ((D1_close - lastclose)/lastclose, 4)
       tp1 = D1_close 
   
  
   #5d
   print ("5D")
   filepath_ts5 = "D:\\OneDrive\\AI Workspace\\results\\ts\\ts5\\"+symbol + ".csv"
   if os.path.isfile(filepath_ts5) == True: 
   
     df_ts_fc = pd.read_csv (filepath_ts5)
     i = 0 
     
     if len(df_ts_fc): 
     #print (" TS df_ts_D1 " , df_ts_fc)
      df_ts_fc.set_index('Type', inplace=True)
      fc_close = df_ts_fc.loc['TS-5D-TR']['Close']
      print (fc_close)
      trend_5d =round ((fc_close - lastclose)/lastclose, 4)
      tp2 = fc_close
   print ( " **** Check TS20 ")
   filepath_ts20 = "D:\\OneDrive\\AI Workspace\\results\\ts\\ts20\\"+symbol + ".csv"
   if os.path.isfile(filepath_ts20) == True: 
    
     df_ts_fc = pd.read_csv (filepath_ts20)
     print (  " **** Check TS20 ", df_ts_fc )
     i = 0
     
     if len(df_ts_fc): 
      df_ts_fc.set_index('Type', inplace=True)
      fc_close = df_ts_fc.loc['TS-20D-TR']['Close']
      trend_20d =round ((fc_close - lastclose)/lastclose, 4)
      tp3 = fc_close
   
   filepath_ts60 = "D:\\OneDrive\\AI Workspace\\results\\ts\\ts60\\"+symbol + ".csv"
   
   

   if os.path.isfile(filepath_ts60) == True: 
    
     df_ts_fc = pd.read_csv (filepath_ts60)
     i = 0
     
     #print (" TS df_ts_D1 " , df_ts_fc)
     if len(df_ts_fc): 
      df_ts_fc.set_index('Type', inplace=True)
      fc_close = df_ts_fc.loc['TS-60D-TR']['Close']
      trend_60d =round ((fc_close - lastclose)/lastclose, 4)
      tp4 = fc_close

   
   print ("=====TS : trend_1d: ", trend_1d, " tp1:  ", tp1, " trend_5d:",  trend_5d, " trend_20d ", trend_20d ) 
   ts_trend = 0 
   if (tp1 > 0 and tp2 > 0 and tp3 > 0 and tp4 > 0 ):
       # Trend 1
       if ts_trend == 0 and trend_60d > trend_20d and trend_20d > trend_5d and trend_5d > trend_1d and trend_1d > 0 : ts_trend = trend_20d
       if ts_trend == 0  and trend_60d < trend_20d and trend_20d < trend_5d and trend_5d < trend_1d and trend_1d < 0 : ts_trend = trend_20d

       # Trend 2
       if ts_trend == 0 and trend_60d > trend_20d and trend_20d > trend_5d  and trend_5d > 0 : ts_trend = trend_60d
       if ts_trend == 0  and trend_60d < trend_20d and trend_20d < trend_5d  and trend_5d < 0  : ts_trend = trend_60d

        # Trend 3
       if ts_trend == 0 and trend_20d > trend_5d  and trend_5d > trend_1d and trend_1d > 0 : ts_trend = trend_5d
       if ts_trend == 0 and trend_20d < trend_5d  and trend_5d < trend_1d and trend_1d < 0 : ts_trend = trend_5d

   if (tp1 > 0 and tp2 > 0 and tp3 > 0  ):
       if ts_trend == 0 and trend_1d > 0 and trend_5d > trend_1d  and trend_20d > trend_5d: ts_trend = trend_5d
       if ts_trend == 0 and trend_1d < 0 and trend_5d < trend_1d  and trend_20d < trend_5d: ts_trend = trend_5d

   result = []
   
   result = (trend_1d ,  trend_5d,  trend_20d, trend_60d,   tp1,  tp2, tp3, tp4 , ts_trend)
   
   
   print ("trend_1d ,  trend_5d,  trend_20d, trend_60d,    tp1,  tp2, tp3 , tp4", result)
   return result
   
   
def checkTensorTA (symbol, lastclose):
   ts_ta_trend = 0
   ts_ta_sihnal = 0 
   df_ts_ta = pd.read_csv ("G:\\My Drive\\ml_results\\ta_results\\"+symbol + ".csv")
   df_ts_org = df_ts_ta

   df_ts_ta.set_index('Type', inplace=True)
   key = 'TA'
   row = df_ts_ta.loc[key]
   row.sort_values(by=['Date'], inplace=True, ascending=True)
   row_s = row[:len(row)-(len(row)-5)]
   
   row_s_mean = row_s['Close'].mean ()

   trend_S = 0
   trend_M = 0
   trend_L = 0 
   tp1 = 0
   tp2 = 0
   tp3 = 0 
   # 5 Days
   #row_s = row [: len(row) - (len(row)-5)]
   if row_s_mean > 0 and lastclose > 0: 
      trend_S = round ( (row_s_mean - lastclose )/lastclose ,4 ) 
      tp1 = row_s_mean
   

   
   row_last = row.tail (1)
   

   ts_ta_end  = float ( row['Close'][len(row)-1]) 
   ts_ta_end_date  = str ( row['Date'][len(row)-1]) 
   

   ts_ta_mean = row['Close'].mean ()
   #df_ts_ta.loc['TA-MEAN']['Close'][0]

   row.sort_values(by=['Close'], inplace=True, ascending=True)

  

   row_max = row[len(row)-1:len(row)]
   row_min = row[:1]
   ts_ta_max = row_max['Close'][0]
   ts_ta_date_max = row_max['Date'][0]

   #df_ts_ta.loc['TA-MAX']['Close'][0]
   ts_ta_min = row_min['Close'][0]
   ts_ta_date_min = row_min['Date'][0]

   #df_ts_ta.loc['TA-MIN']['Close'][0]
   row_org = df_ts_ta.loc["TA-D1"]
   ts_org_end   = float ( row_org['Close'][len(row_org)-1]) 

  

   

   
   
   if ts_ta_mean > 0 and lastclose > 0: 
      trend_M= round ( (ts_ta_mean - lastclose )/lastclose ,4 ) 
      tp2 = ts_ta_mean

   if ts_org_end > 0 and lastclose > 0: 
      #trend_L= round ( (ts_ta_end - lastclose )/lastclose ,4 ) 
      trend_L= round ( (ts_org_end - lastclose )/lastclose ,4 ) 
      

      tp3 = ts_org_end



   
   #print ("TA TS END  ", ts_ta_end ,"  Dateend: ", ts_ta_end_date ,   " ts_ta_mean " , ts_ta_mean ,  " ts_ta_max " , ts_ta_max , " ts_ta_min: ", ts_ta_min )
   
   #print  ( " trend_S ", trend_S , " trend_M: ", trend_M , " trend_L: ", trend_L)

   
   result = []
   signal = 0 
   result = (trend_S , trend_M,  trend_L , tp1, tp2, tp3, signal  )

   print ("=====TS TA Trend ----trend_S , trend_M,  trend_L , tp1, tp2, tp3 ", result)
   #print ( " TA MAX DATE  ", ts_ta_date_max ,  " -ts_ta_max: ", ts_ta_max, " ts_ta_date_min: ", ts_ta_date_min ,  " ts_ta_min: ", ts_ta_min)
   return result

   

   
def checkTrendFB (symbol, lastclose):
   #ds_result_file = folder + symbol + ".csv"
   #df_his=pd.read_csv(ds_result_file)
   #lastclose = float(df_his['close'].iloc[len(df_his)-1])
   #lastdate = str(df_his['date'].iloc[len(df_his)-1])
   
   
   #check FB10
   folderpath__fb = "D:\\OneDrive\\AI Workspace\\results\\fb_results\\"
   df_fb10 = pd.read_csv (folderpath__fb+"fb10\\"+symbol + ".csv")
   df_fb10_org =  pd.DataFrame (df_fb10)
   
   

   df_fb10.set_index('Type', inplace=True)
   # Retrieve a row by key (index value)
   key = 'FB-10D-trend'
   
    
   row = df_fb10.loc[key]
   row = row [1:len(row)]
   change = row["Change"][0]
   
   
   key = 'FB-10D(C)'
   row = df_fb10.loc[key]
   row = row.tail(1)
   
   fb10_close  = float ( row["Close"][0]) 
   
   change_fb10 = round ( (fb10_close - lastclose ) /lastclose , 4) 
   #row["Change"][0]
   
   #Today values of FB: 
   

   row = df_fb10.loc["FB10-C"]
   #print (datetime.strptime(row["Date"], '%y-&m-%d').date() )


   #row["Date"] =  datetime.strptime(row["Date"], '%y-&m-%d %H:%M:%S').date()

  
   #row = row.set_index('Date', inplace=True)
   #row = row.loc[today]
   #row.sort_values(by=['Date'], inplace=True, ascending=False)
   #fb10_close_today =  float ( row["Close"][0]) 
   fb10_close_today = 0 
   
   df_fb10_org = pd.DataFrame (df_fb10_org)

   


   today_fb10 = [d  for d in df_fb10_org.values if str(datetime.strptime( str( d[2]), '%Y-%m-%d %H:%M:%S').date()) == today ]
   today_fb10 = pd.DataFrame (today_fb10)
   today_fb10_C = 0
   today_fb10_H = 0
   today_fb10_L = 0
   for s in today_fb10.values: 
        if s[1] =="FB10-H": today_fb10_H = s[3]
        if s[1] =="FB10-C": today_fb10_C = s[3]
        if s[1] =="FB10-L": today_fb10_L = s[3]
   
   
   change_today_fb10 = round ( (today_fb10_C - lastclose ) /lastclose , 4) 
   
   print (symbol , " --- change_today_fb10: -- ", change_today_fb10)



   
   
   #fb10_today_close = row["Close"][0]
   #print ("fb10_today_close ", fb10_today_close)


   
   
   #FB60 
   
   df_fb60 = pd.read_csv (folderpath__fb+"fb60\\"+symbol + ".csv")
   df_fb60.set_index('Type', inplace=True)
   # Retrieve a row by key (index value)
   key = 'FB-60D(C)'
   row = df_fb60.loc[key]
   row = row.tail(1)
   
   fb60_close  = float ( row["Close"][0]) 
   change_fb60 = round ( (fb60_close - lastclose ) /lastclose , 4) 

   key = 'FB-60D(H)'
   row = df_fb60.loc[key]
   row = row.tail(1)
   
   fb60_high  = float ( row["Close"][0]) 
   change_fb60_H = round ( (fb60_high - lastclose ) /lastclose , 4) 

   key = 'FB-60D(L)'
   row = df_fb60.loc[key]
   row = row.tail(1)
   
   fb60_low  = float ( row["Close"][0]) 
   change_fb60_L = round ( (fb60_low - lastclose ) /lastclose , 4) 
   
   print (symbol , " --- change_fb60_L: -- ", change_fb60_L)
   
   #FB200
   
   df_fb200 = pd.read_csv (folderpath__fb+"fb200\\"+symbol + ".csv")
   df_fb200.set_index('Type', inplace=True)
   # Retrieve a row by key (index value)
   key = 'FB-200D(C)'
   row = df_fb200.loc[key]
   row = row.tail(1)
   
   fb200_close  = float ( row["Close"][0]) 
   
   change_fb200 = round ( (fb200_close - lastclose ) /lastclose , 4) 

   key = 'FB-200D(H)'
   row = df_fb200.loc[key]
   row = row.tail(1)
   fb200_high  = float ( row["Close"][0]) 
   change_fb200_H = round ( (fb200_high - lastclose ) /lastclose , 4) 

   key = 'FB-200D(L)'
   row = df_fb200.loc[key]
   row = row.tail(1)
   fb200_low  = float ( row["Close"][0]) 
   change_fb200_L = round ( (fb200_low - lastclose ) /lastclose , 4) 
   
   
   
    #FB400
   
   df_fb400 = pd.read_csv (folderpath__fb+"fb400\\"+symbol + ".csv")
   df_fb400_org = pd.DataFrame (df_fb400)
   df_fb400.set_index('Type', inplace=True)
   # Retrieve a row by key (index value)
   key = 'FB400-C'
   row_org = df_fb400.loc[key]
   row = row_org.tail(1)
  
   fb400_close  = float ( row["Close"][0])
  
   #print ( " TEST ", row_org['Date'][:8] ) 
   #print (row_org)
   #df_fb400_org['Date'] = df_fb400_org["Date"][0:10]

   #datetime.strptime( str( df_fb400_org["Date"]), '%y-&m-%d %H:%M:%S').date()
   #print (row_org)
   
   #df_fb400_org = df_fb400_org.set_index('Date', inplace=False)
   #df_fb400_org = df_fb400_org.loc[datetime.now().date()]
  

   today_fb400 = [d  for d in df_fb400_org.values if str(datetime.strptime( str( d[2]), '%Y-%m-%d %H:%M:%S').date()) == today ]
   today_fb400 = pd.DataFrame (today_fb400)
   today_fb400_H = 0 
   today_fb400_C = 0 
   for s in today_fb400.values: 
        if s[1] =="FB400-H": today_fb400_H = s[3]
        if s[1] =="FB400-L": today_fb400_L = s[3]
        if s[1] =="FB400-C": today_fb400_C = s[3]
   
   
   #print ( " today_fb400_H: " , today_fb400_H , " today_fb400_L: ", today_fb400_L,  " today_fb400_C: ", today_fb400_C )

   change_fb400 = round ( (fb400_close - lastclose ) /lastclose , 4) 
   
   result = []
   trend_S = 0
   trend_M = 0
   trend_L = 0
   tp1 = 0 
   tp2 = 0 
   tp3 = 0
   signal = 0 
   
   if ( trend_S == 0 and change_fb10 > 0 and change_fb60 > 0   ): 
        trend_S = change_fb10
        tp1 = today_fb10_H
        if tp1 < lastclose: 
             tp1 = today_fb400_H
        if tp1 < lastclose:
             tp1 = fb10_close
             

   if ( trend_S == 0 and change_fb10 < 0 and change_fb60 < 0 and change_fb200< 0  ): 
        trend_S = change_fb10
        tp1 = today_fb10_L
        if tp1 > lastclose: 
             tp1 = today_fb400_L
        if tp1 > lastclose: 
             tp1 = fb10_close

   if ( trend_S > 0 and today_fb10_C > lastclose  ): signal = 1
   if ( trend_S < 0 and today_fb10_C < lastclose  ): signal = -1
   
   #M ----- 
   if ( trend_M == 0 and change_fb60 > 0 and ( change_fb200 > 0 or change_fb400 > 0 )   ): 
        trend_M = change_fb60
        tp2 = fb60_close
        if fb60_close > fb200_close:
             tp2 = fb200_close
             
   if ( trend_M == 0 and change_fb60 < 0 and ( change_fb200 < 0 or change_fb400 <0 )   ): 
        trend_M = change_fb60
        tp2 = fb60_close
        if fb200_close < fb60_close:
             tp2 = fb200_close

   #L ----- 
   if ( trend_L == 0 and change_fb200 > 0 and change_fb400 > 0   ): 
        trend_L = change_fb400
        tp2 = fb200_close
        if fb400_close > fb200_close:
             tp2 = fb400_close
             
   if ( trend_L == 0 and change_fb200 < 0 and change_fb400 < 0   ): 
        trend_L = change_fb400
        tp2 = fb200_close
        if fb400_close < fb200_close:
             tp2 = fb400_close

   result = []
   result = (change_fb10, change_fb60, change_fb200, change_fb400, fb10_close, fb60_close, fb200_close, fb400_close, today_fb10_C, today_fb400_C)

   #result = (trend_S, trend_M, trend_L, tp1 , tp2, today_fb10)
   
   print ( "=====FB-TREND: change_fb10, change_fb60, change_fb200, change_fb400, fb10_close, fb60_close, fb200_close, fb400_close, today_fb10_C, today_fb400_C ----\n ", result)

   return result

def checkTrendDS_ML (symbol):
   result = []
   
   filepath_df = "G:\\My Drive\\ml_results\\ds\\"+symbol+".csv"
   
   if os.path.isfile(filepath_df) == False:  return result
   
   df_ft=pd.read_csv(filepath_df)
   #lastclose = float(df_ft['Source'].iloc[len(df_ft)-1])
                  
   i = 0 
   action = 0
   trend_S = 0 
   trend_M = 0 
   trend_L = 0
   tp1 = 0 
   tp2  = 0
   tp3 = 0 
   df_ft.set_index('Type', inplace=True)
   key ='DS-ML'
   row_org = df_ft.loc[key]
   #row = row_org.tail(1)
   
   for s in row_org.values: 
                       print (s[1])
                       i = i+1
                       fc_date = s[1]
                       fc = s[2]
                       fc_change= s[4]
                       print ( i, fc_date , " fc: ", fc , " Change ", fc_change)
                       if i == 1: 
                           trend_S = fc_change
                           tp1 = fc 
                       if i == 2: 
                           trend_M = fc_change
                           tp2 = fc 
                       if i == 3: 
                           trend_L = fc_change
                           tp3 = fc 
   trend_ds = 0 
   if (trend_S > 0 and trend_M > 0 and trend_L > 0 ) : trend_ds = 4 
   if ( trend_ds == 0 and trend_S > 0 and trend_M > 0  ) : trend_ds = 3
   if ( trend_ds == 0 and trend_S > 0 and ( trend_M > 0 or trend_L > 0)  ) : trend_ds = 2
   if (trend_S < 0 and trend_M < 0 and trend_L < 0 ) : trend_ds = -4 
   if ( trend_ds == 0 and trend_S < 0 and trend_M < 0  ) : trend_ds = -3
   if ( trend_ds == 0 and trend_S < 0 and ( trend_M < 0 or trend_L < 0)  ) : trend_ds = -2


   result = (trend_S, trend_M, trend_L, tp1, tp2, tp3, trend_ds)
   print (" TREND DS_ML: trend_S, trend_M, trend_L, tp1, tp2, tp3, trend_ds  ", result)
   return result
def checkTrendDSBase (symbol):
   result = []
   
   filepath_df = "G:\\My Drive\\ml_results\\ds_base\\"+symbol+".csv"
   
   if os.path.isfile(filepath_df) == False:  return result
   
   df_ft=pd.read_csv(filepath_df)
   #lastclose = float(df_ft['Source'].iloc[len(df_ft)-1])
                  
   i = 0 
   action = 0
   trend_S = 0 
   trend_M = 0 
   trend_L = 0
   tp1 = 0 
   tp2  = 0
   tp3 = 0 
   
   
   for s in df_ft.values: 
                       print (s[5])
                       i = i+1
                       if i == 1:
                           trend_S = s[5]
                           tp1 = s[3]
                       if i == 2: 
                           trend_M = s[5]
                           tp2 = s[3]

                       if i == 3:
                           trend_L = s[5]
                           tp3 = s[3]

                       action = 0
                       if action == 0 and trend_S > 0 and trend_M > 0 and trend_L > 0 : action = 1
                       if action == 0 and trend_S > 0 and trend_M > 0 and trend_L <= 0 : action = 2
                       if action == 0 and trend_S > 0 and (trend_M > 0 or trend_L > 0) : action = 3
                       if action == 0 and trend_S <= 0 and (trend_M > 0 and  trend_L > 0) : action = 4
             
             
                       if action == 0 and trend_S < 0 and trend_M < 0 and trend_L < 0 : action = -1
                       if action == 0 and trend_S < 0 and trend_M < 0 and trend_L >= 0 : action = -2
                       if action == 0 and trend_S < 0 and (trend_M < 0 or trend_L < 0) : action = -3
                       if action == 0 and trend_S >= 0 and (trend_M < 0 and  trend_L < 0) : action = -4
             
   result = []
   signal = action
   result = (trend_S , trend_M,  trend_L , tp1, tp2, tp3, signal  )
   print ("=====FC-BASE:  trend_S , trend_M,  trend_L , tp1, tp2, tp3 *** ", result)
          
   return result
def checkTrendDSFeatuews (symbol):
   result = []
   
   filepath_df =  "G:\\My Drive\\ml_results\\ds_features\\"+symbol+".csv"
   
   if os.path.isfile(filepath_df) == False:  return result
   df_ft=pd.read_csv("G:\\My Drive\\ml_results\\ds_features\\"+symbol+".csv")
   i = 0 
   action = 0
   trend_S = 0 
   trend_M = 0 
   trend_L = 0
   tp1 = 0 
   tp2  = 0
   tp3 = 0 
   
   
   for s in df_ft.values: 
                       print (s[5])
                       i = i+1
                       if i == 1:
                           trend_S = s[5]
                           tp1 = s[3]
                           
                       if i == 2: 
                           trend_M = s[5]
                           tp2 = s[3]

                       if i == 3:
                           trend_L = s[5]
                           tp3 = s[3]

                       action = 0
                       if action == 0 and trend_S > 0 and trend_M > 0 and trend_L > 0 : action = 1
                       if action == 0 and trend_S > 0 and trend_M > 0 and trend_L <= 0 : action = 2
                       if action == 0 and trend_S > 0 and (trend_M > 0 or trend_L > 0) : action = 3
                       if action == 0 and trend_S <= 0 and (trend_M > 0 and  trend_L > 0) : action = 4
             
             
                       if action == 0 and trend_S < 0 and trend_M < 0 and trend_L < 0 : action = -1
                       if action == 0 and trend_S < 0 and trend_M < 0 and trend_L >= 0 : action = -2
                       if action == 0 and trend_S < 0 and (trend_M < 0 or trend_L < 0) : action = -3
                       if action == 0 and trend_S >= 0 and (trend_M < 0 and  trend_L < 0) : action = -4
             
   result = []
   signal = action
   result = (trend_S , trend_M,  trend_L , tp1, tp2, tp3, signal )
   print ("=====FC-FEATURES:  trend_S , trend_M,  trend_L , tp1, tp2, tp3 *** ", result)
          
   return result

def checkTrendDSClassifier (symbol):
      result = []
      filepath_trend = "G:\\My Drive\\ml_results\\ds_classifier\\"+symbol+".csv"
      if os.path.isfile(filepath_trend) == False:
        result = (0,0,0,0)
        return result


      df_trend=pd.read_csv(filepath_trend)
     
      trend = 0 
      print ( "CLASSIFiFER:  " , df_trend)
      trend_1d = df_trend['1d'].values[0]
      trend_5d = df_trend['5d'].values[0]
      trend_1mo = df_trend['1mo'].values[0]
      trend_3mo = df_trend['3mo'].values[0]
      #trend = df_trend['trend'].values[0]
      if (trend_1d > 0 and (trend_5d>= 0 or trend_1mo >= 0) ): trend = 1 
      if (trend_1d < 0 and (trend_5d<= 0 or trend_1mo <= 0) ): trend = -1 
      

      result = (trend_1d, trend_5d, trend_1mo, trend_3mo, trend )
      print ( " TrendDSClassifier trend_1d, trend_5d, trend_1mo, trend_3mo, trend ", result)


      
      
      return result
folderpath_results = r"C:/ai_work/temp/"
#"D:\\OneDrive\\AI Workspace\\analyse\\"
#"C:\\ai_work\\temp\\"
#"D:\\OneDrive\\AI Workspace\\results\\analyse\\"
csvfileresult_base =folderpath_results + "aitrend.csv"   
csvfileresult_forecastfinal =folderpath_results+ "forecast_final.csv" 
#"C:\\ai_work\\temp\\" + "forecast_final.csv" 
def checkAItrend (fc_symbol, writer_final):
     #csvfileresult_base ="G:\\My Drive\\ml_results\\" + "aitrend.csv"
     """
     “1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”
     """
     i = 0 
     with open(csvfileresult_base , 'w', newline='') as csvfilerow:
               fieldnames = ['Symbol','Date','1d','5d','1mo','1y','2y','signal', 'power',  "price_level1", 'price_level2', 'tp1', 'tp2', 'avg_change','avg_volume', 'lastclose', 'tec','dsclass']
               writer_today = csv.DictWriter(csvfilerow, fieldnames=fieldnames)
               writer_today.writeheader()
               listOfFiles = os.listdir(folder_results)
               fckey = ".csv"
               i = 0 
               for entry in listOfFiles:
                  i = i +1 
                  s = entry.split (fckey)
                  symbol = s[0]
                  #print (i,  "CHECK" , symbol)
                  i_run = 1
                  if fc_symbol and symbol != fc_symbol:
                    i_run = 0
                  filepath_his = "G:\\My Drive\\ml_data\\yh_his\\D1\\"+symbol+".csv"
                  if  i_run == 1 and os.path.isfile(filepath_his) == False: 
                       print (" NO data of ", filepath_his)
                       i_run = 0 
                  dscl_trend = (0,0,0,0)
                  if  i_run > 0 :
                       try:
                        dscl_trend  = checkTrendDSClassifier (symbol)
                        dscl_trend = dscl_trend[4]
                        if  dscl_trend == 0: i_run = 0
                        
                       except: 
                           print (" Class error ", symbol)
                       

                  #print ( i, ". ", symbol ,  " ----- dscl_trend ****** ", i , dscl_trend ,  " ----- AI Trend symbol: " , " i_run: ", i_run)
                       
                  if i_run == 1:
                      
               
                    try:
                      i = i +1
                     
                     #dscl_trend = dscl_trend[4]
                      print ( " dscl_trend ****** ", i , dscl_trend ,  " ----- AI Trend symbol: " , symbol)
                      
                      df_his = pd.read_csv(filepath_his)
                      df_his_last10 = df_his.tail(10)
                      lastclose = float(df_his_last10['close'].iloc[len(df_his_last10)-1])
                      high_mean =  df_his_last10['high'].mean ()
                      low_mean =  df_his_last10['low'].mean ()
                      avg_volume = df_his_last10['volume'].mean ()
                      #print (" df_his_last10: ", df_his_last10)
                      avg_change = round ( (high_mean - low_mean )/low_mean, 4) 
                      #print (" avg_volume: ", avg_volume)
                      #Classifier Trend: 
                      
                      

                      result_fb = checkTrendFB (symbol, lastclose)
                      
                      trend_fb10 = result_fb[0]
                      trend_fb60 = result_fb[1]
                      trend_fb200 = result_fb[2]
                      trend_fb400 = result_fb[3]
                      close_fb10 = result_fb[4]
                      close_fb60  = result_fb[5]
                      close_fb200  = result_fb[6]
                      close_fb400  = result_fb[7]
                      today_fb10  = result_fb[8]
                      today_fb400  = result_fb[9]

                      pricelevel_fb1 = today_fb10
                      pricelevel_fb2 = today_fb400
                      
                      #change_fb10, change_fb60, change_fb200, change_fb400
                      
                      

                     
                      
                      #lastclose = float(df_ft['Source'].iloc[len(df_ft)-1])
                      result_ds_ml = checkTrendDS_ML  (symbol)
                      
                      trend_ds_ml = 0 
                      ds_ml_tp1 = 0
                      ds_ml_tp2 = 0
                      ds_ml_tp3 = 0
                      

                      if len(result_ds_ml) > 0 : 
                          trend_ds_ml = result_ds_ml[6]
                          ds_ml_tp1 = result_ds_ml[3]
                          ds_ml_tp2 = result_ds_ml[4]
                          ds_ml_tp3 = result_ds_ml[5]

                      ds_ml_list = (ds_ml_tp1, ds_ml_tp2, ds_ml_tp3)

                      result_dsft = checkTrendDSFeatuews (symbol)
                      print (" RESULT DSFT ", len(result_dsft))
                      
 
                      dsft_S = 0 
                      dsft_M = 0 
                      dsft_L = 0 
                      dsft_tp1 = 0 
                      dsft_tp2 = 0
                      dsft_tp3 = 0
                      dsft_signal = 0 
                     
                      if len(result_dsft) > 0 : 
                       dsft_S = result_dsft[0]
                       dsft_M = result_dsft[1]
                       dsft_L = result_dsft[2]
                       dsft_tp1 = result_dsft[3]
                       dsft_tp2 = result_dsft[4]
                       dsft_tp3 = result_dsft[5]
                       dsft_signal = result_dsft[6]

                     

                      #df_base=pd.read_csv("G:\\My Drive\\ml_results\\ds_base\\"+symbol+".csv")
                      result_dsbase =checkTrendDSBase (symbol)
                      dsbase_S = 0 
                      dsbase_M = 0 
                      dsbase_L = 0 
                      dsbase_tp1 = 0
                      dsbase_tp2 = 0
                      dsbase_tp3 = 0

                      if len(result_dsbase) > 0: 
                       dsbase_S = result_dsbase[0]
                       dsbase_M = result_dsbase[1]
                       dsbase_L = result_dsbase[2]
                       dsbase_tp1 = result_dsbase[3]
                       dsbase_tp2 = result_dsbase[4]
                       dsbase_tp3 = result_dsbase[5]
                       dsbase_signal = result_dsbase[6]
                    
                     
                      


                   
                      
                      result_linear = checkLinear (symbol, lastclose)
                      ln_tp1 = result_linear[0]
                      ln_tp2 = result_linear[1]
                      ln_tp3 = result_linear[2]
                      ln_signal = result_linear[3]
                      
                      
                      #ta_trend_S, ta_trend_M, ta_trend_L ,  ta_signal
                      result_tec = checkTehnical (symbol, lastclose)
                      ta_trend_S =0 
                      ta_trend_M = 0 
                      ta_trend_L = 0 
                      ta_trend = 0 

                      if len(result_tec) > 0: 
                       ta_trend_S =result_tec [0]
                       ta_trend_M =result_tec [1]
                       ta_trend_L =result_tec [2]
                       ta_trend = result_tec [3]
                      # print ("trend_1d ,  trend_5d,  trend_20d, trend_60d,    tp1,  tp2, tp3 , tp4", result)
                      result_ts = checkTensorTS (symbol, lastclose)
                      ts_trend_1 = 0 
                      ts_trend_5 = 0
                      ts_trend_20 = 0
                      ts_trend_60 = 0
                      ts_trend  = 0 
                      ts_tp1 = 0 
                      ts_tp2 = 0
                      ts_tp3 = 0 
                      ts_tp4 = 0 

                      if len(result_ts) > 1: 
                       ts_trend_1 =result_ts [0]
                       ts_trend_5=result_ts [1]
                       ts_trend_20 =result_ts [2]
                       ts_trend_60 =result_ts [3]
                       
                       ts_tp1  = result_ts [4]
                       ts_tp2  = result_ts [5]
                       ts_tp3  = result_ts [6]
                       ts_tp4  = result_ts [7]
                       ts_trend  = result_ts [8]
                       

                      # fieldnames = ['Symbol','Date','Type','Close','Source','Change','Horizont']
                      avg_tp1 = 0
                      i = 0 
                      if dsbase_tp1 > 0 : i=i+1
                      if dsft_tp1 > 0 : i = i +1
                      if ln_tp1 > 0 : i = i +1 
                      if ts_tp1 > 0 : i = i+1
                      print (i, " CAL AVG 1 ** ",  dsbase_tp1 , "-",  dsft_tp1 , "-",  ln_tp1 , "-",   ts_tp1 )
                      avg_tp1 =( ( dsbase_tp1 +dsft_tp1 + ln_tp1 + ts_tp1 )/i)

                      avg_tp2 = 0
                      i = 0 
                      if dsbase_tp2 > 0 : i=i+1
                      if dsft_tp2 > 0 : i = i +1
                      if ln_tp2 > 0 : i = i +1 
                      if ts_tp2 > 0 : i = i+1
                      avg_tp2 =( ( dsbase_tp2 +dsft_tp2 + ln_tp2 + ts_tp2 )/i)

                      avg_tp3 = 0
                      i = 0 
                      if dsbase_tp3 > 0 : i=i+1
                      if dsft_tp3 > 0 : i = i +1
                      if ln_tp3 > 0 : i = i +1 
                      if ts_tp3 > 0 : i = i+1
                      avg_tp3 =( ( dsbase_tp3 +dsft_tp3 + ln_tp3 + ts_tp3 )/i)

                      change_avg_tp1 = ((avg_tp1 - lastclose ) /lastclose)
                      change_avg_tp2 = ((avg_tp2 - lastclose ) /lastclose)
                      change_avg_tp3 = ((avg_tp3 - lastclose ) /lastclose)
                      
                      print ( " avg_tp1 ", avg_tp1 , " avg_tp2 ", avg_tp2 , " avg_tp3 ", avg_tp3)
                      print ( " CHANG AVG TPs **** ", " change_avg_tp1 ", change_avg_tp1, " change_avg_tp2 ", change_avg_tp2 , " change_avg_tp3: ", change_avg_tp3)
                      #result_ts_ta = checkTensorTA (symbol, lastclose)
                      
                     
                      #TREND
                      trend_S = 0 
                      trend_M = 0 
                      trend_L = 0 
                      tp1 = 0
                      tp2 = 0 
                      power = 0
                      signal = 0
                      price_level1 = pricelevel_fb1
                      price_level2 = pricelevel_fb2

                     #TRADE SIGNAL
                      #FB Signal --- 
                      
                      if   signal == 0  and trend_fb10 > 0 and trend_fb60 > 0 and ( trend_fb200 > 0 or trend_fb400 > 0 )    :
                          signal = 1
                          power = 1
                          
                          tp1 = close_fb60
                          tp2 = close_fb200
                      
                      if   signal == 0  and trend_fb10 < 0 and trend_fb60 < 0 and ( trend_fb200 < 0 or trend_fb400 < 0 )    :
                          signal = -1
                          power = 1
                          
                          tp1 = close_fb60
                          tp2 = close_fb200


                      if   signal == 0  and trend_fb10 > 0 and trend_fb60 > 0  and  change_avg_tp2 > 0 :
                          signal = 2
                          power = 2
                          if trend_fb200 > 0: power = 1
                          tp1 = close_fb10
                          tp2 = close_fb60
                      if   signal == 0  and trend_fb10 < 0 and trend_fb60 < 0  and  change_avg_tp2 < 0 :
                          signal = -2
                          power = 2
                          if trend_fb200 < 0: power = 1
                          tp1 = close_fb10
                          tp2 = close_fb60
                      
                      if   signal == 0  and trend_fb60 > 0 and trend_fb200 > 0 and change_avg_tp1 > 0 and change_avg_tp2 > 0  and ts_trend > 0 :
                          signal = 3
                          power = 3
                         
                          tp1 = close_fb60
                          tp2 = close_fb200
                      if   signal == 0  and trend_fb60 < 0 and trend_fb200 < 0 and change_avg_tp1 < 0 and change_avg_tp2 < 0 and ts_trend < 0 :
                          signal = -3
                          power = 3
                          
                          
                          tp1 = close_fb60
                          tp2 = close_fb200
                      #TREDN OTHER
                      if   signal == 0  and ts_trend > 0 and trend_fb10  > 0  :
                          signal = 4
                          power = 4
                          tp1 = lastclose + lastclose*ts_trend
                      if   signal == 0  and ts_trend < 0 and trend_fb10  < 0  :
                          signal = -4
                          power = 4
                          tp1 = lastclose + lastclose*ts_trend
                      
                      if   signal == 0  and  trend_fb10 > 0 and dscl_trend > 0 and ta_trend > 0  :
                          signal = 5
                          power = 4
                          tp1 = 0
                      if   signal == 0  and  trend_fb10 < 0 and dscl_trend < 0 and ta_trend < 0   :
                          signal = -5
                          power = 4
                          tp1 = 0

                      if   signal == 0  and ts_trend > 0 and change_avg_tp1 > 0 and trend_fb10 > 0 :
                          signal = 6
                          power = 4
                          tp1 = lastclose + lastclose*change_avg_tp1
                     
                     


                      # Power1: 
                      """
                      if   signal == 0 and dsft_S > 0 and fb_S > 0  : 
                           
                           signal = 5
                           tp1 = dsft_tp1
                           tp2 = dsft_S
                           power = 4
                           if fb_M > 0 and fb_L > 0: power = 1
                           if fb_M <= 0 and fb_L > 0: power = 2
                           if fb_M > 0 and fb_L <= 0: power = 3
                           if fb_M > 0 and ( dsft_M or dsft_L > 0 ) and power == 4 : power = 3

                     
                      if   signal == 0 and dsft_S < 0 and fb_S < 0   : 
                           signal = -5
                           tp1 = dsft_tp1
                           tp2 = dsft_S
                           power = 4
                           if fb_M < 0 and fb_L < 0: power = 1
                           if fb_M >= 0 and fb_L < 0: power = 2
                           if fb_M < 0 and fb_L >= 0: power = 3
                           if fb_M < 0 and ( dsft_M or dsft_L > 0 ) and power == 4: power = 3
                      
                    
                      """
                      #CANCEL Signal 
                      print ( symbol, " Before Trend Result ",  " signal:  ", signal , " power: ", power , " tp1 ", tp1 , " tp2: ", tp2 )
                      check_signal = 0 
                      if signal > 0 and dscl_trend > 0   and ta_trend >  0 :  check_signal = signal
                      if signal < 0 and dscl_trend < 0  and ta_trend <  0 :  check_signal = signal

                      signal = check_signal

                      if   signal == 0  and ts_trend > 0  and trend_ds_ml > 0  and trend_fb10  > 0  :
                          signal = 7
                          power = 4
                          tp1 = max (ds_ml_list)
                      if   signal == 0  and ts_trend < 0  and trend_ds_ml < 0  and trend_fb10  < 0  :
                          signal = -7
                          power = 4
                          tp1 = min (ds_ml_list)
                      

                      
                      print ( symbol, " Trend Result ",  " signal:  ", signal , " power: ", power , " tp1 ", tp1 , " tp2: ", tp2 )

                      #['Symbol','Date','Type','Close','Source','Change','Horizont']
                      fc_type ="AI"
                      fc_s = 0 
                      #writer_today.writerow({'Symbol': symbol,   'Date': str(current_date) , 'Type': fc_type, 'Close': str(fc_s) })
                      writer_today.writerow({'Symbol': symbol,   'Date': str(current_date) , '1d': str(change_avg_tp1), '5d': str(change_avg_tp2), '1mo': str(change_avg_tp3), '1y': str(trend_fb200) , '2y': str(trend_fb400) , 'signal': str(signal), 
                                             'power':str(power), 'price_level1': str(price_level1) , 'price_level2': str(price_level2), 'tp1':str(tp1), 'tp2':str(tp1), 'avg_change':str(avg_change), 
                                             'avg_volume': str(avg_volume), 'lastclose': str(lastclose), 'tec':str(ta_trend), 'dsclass':str(dscl_trend) })
                      
                     
                      
                    except: 
                      print ("Error HIS Files >>>>>>>>  ",symbol )
                  
               
                  
               
       
def runFromFolder (fc_symbol, retrain, BarsLastHIS):
    #df_futures = markets.getDataFrameMarkets ("D1")
   
    listOfFiles = os.listdir(folder)
    fckey = ".csv"
    for entry in listOfFiles:
        s = entry.split (fckey)
        symbol = s[0]
        i_run = 1
        #print ("Running DS " , symbol)
        #Check Symbol for Check Only
        if fc_symbol and symbol != fc_symbol:
           i_run = 0

        if i_run == 1:
          #print ("Try DS:  " , symbol)
          #checkDSBySymbol (symbol,  retrain, BarsLastHIS)  
          try: 
             
             csvfileresult ="G:\\My Drive\\ml_results\\ds_features\\"+symbol+".csv"
             with open(csvfileresult , 'w', newline='') as csvfilerow:
               fieldnames = ['Symbol','Type','Date','Close','Source','Change']
               writer = csv.DictWriter(csvfilerow, fieldnames=fieldnames)
               writer.writeheader()
             
              
               checkForecastDataFeatures (symbol,  BarsLastHIS, writer )
               
             
             
          except:
             print ("Error Features  " , symbol)  
             
          try: 
             
             csvfileresult_base ="G:\\My Drive\\ml_results\\ds_base\\"+symbol+".csv"
             with open(csvfileresult_base , 'w', newline='') as csvfilerow:
               fieldnames = ['Symbol','Type','Date','Close','Source','Change']
               writer = csv.DictWriter(csvfilerow, fieldnames=fieldnames)
               writer.writeheader()
             
               print ("Base ", symbol)
               checkForecastDataBase (symbol,  BarsLastHIS, writer )
               #checkForecastDataBase (symbol,  BarsLastHIS )

             
          except:
             print ("Error BASE  " , symbol)    
              
import build_Jobs as jobs
jobfile  = "fcft_ds_5.txt" 
jobfile_upload ="upload_ft_ds.txt"
jobfile_trend_analysis ="trendanalysis_ds.txt"
folderpath_jobs ="G:\\My Drive\\ml_jobs\\trend\\"
existsJob = jobs.checkExistsFile (folderpath_jobs, jobfile)
#existsJob =1
import os 
#query = " products not like 'stocks%'  and (iswatching > 0 or isinvest > 0) and active > 0 "
#query = "  iswatching > 0 or isinvest > 0  order by opentime asc"

folder ="G:\\My Drive\\ml_data\\yh_his\\D1\\"

BarsLastHIS = 0 
retrain = 0
weekday = datetime.now().weekday()
# 0: Mo  1: Di 2: Mi 3: Do, 4: Fr, 5: samstag , 6: Sonntag
if weekday == 3:
        retrain = 1

def delete_files_in_folder(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Iterate through the files and delete them
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("All files deleted successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
   
   
def checkJobs (folderpath_jobs):
  
    listOfFiles = os.listdir(folderpath_jobs)
    fckey = "trend_"
    i = 0 
    count_fc = 7
    status = 0
    filelist = ['trend_linear.txt','trend_ds.txt','trend_dsclass.txt','trend_dsft.txt','trend_fb10.txt','trend_tec.txt','trend_ts1.txt','trend_ts5.txt']


    for entry in listOfFiles:
       print (entry)
       if entry in filelist:
           i= i +1
           print (entry, " -> OK ")
       
       
    if i >= count_fc:
       status =1
    
    print (" Count job fies:  ", i, "Status: ", status )
    return status
       
    

jobfile_upload ="upload_trend.txt"
#fc_symbol =""
#existsJob = 1 # TEST
def runAnalyseByJobs (fc_symbol): 
  existsJob = checkJobs (folderpath_jobs)
  #existsJob = 1 
  if  existsJob == 1:
    
    
    st_start  = str(datetime.now()) 
    delete_files_in_folder (folderpath_jobs )
    #checkAItrend (fc_symbol)
  
    try: 
       
       #runFromFolder(fc_symbol,retrain, BarsLastHIS)  
       print ("Try to fc ")
       
       
       with open(csvfileresult_forecastfinal , 'w', newline='') as csvfilerow:
               fieldnames = ['Symbol','Date','Type','Close','Source','Change','Horizont']
               writer_final = csv.DictWriter(csvfilerow, fieldnames=fieldnames)
               writer_final.writeheader()
               checkAItrend (fc_symbol, writer_final)

               
                  
    except:
            print ("Error FC " )
    
    
    jobs.createJobFile (jobfile_upload , st_start, "OK")
    jobs.createJobFile ("upload_bq_forecast.txt" , st_start, "OK")
    jobs.createJobLogFile  ( "Finished_analysetrend.txt", st_start, "OK" )

    #Upload to BigQuery: 
    uploadFileToBQ (csvfileresult_base)
    filepath_uploadtrend = "G:\\My Drive\\ml_jobs\\"+jobfile_upload
    if os.path.isfile(csvfileresult_base) == True and os.path.isfile(filepath_uploadtrend) == True: 
            print ("Try to upload ... ")

            
  
def runAnalyseNow (fc_symbol):
   
      with open(csvfileresult_forecastfinal , 'w', newline='') as csvfilerow:
               fieldnames = ['Symbol','Date','Type','Close','Source','Change','Horizont']
               writer_final = csv.DictWriter(csvfilerow, fieldnames=fieldnames)
               writer_final.writeheader()
               print ("Check row ")
               checkAItrend (fc_symbol, writer_final)
   
      #uploadFileToBQ (csvfileresult_base)
   

#checkAItrend ("AUDUSD")

#existsJob = 1
#checkTrendDS ()
#checkTrendFB ("BRENT")


#uploadFileToBQ (csvfileresult_base)



# %%



