#from yahoo_fin import stock_info as si
import time
#import pyodbc
import csv
from datetime import datetime
from datetime import timedelta

import pandas as pd 
#import buildYHTechinsights


"""
DOCU: 
http://theautomatic.net/yahoo_fin-documentation/
"""
#from datetime import datetime

#import pyodbc
import csv
#from yahoo_fin import stock_info as si
from datetime import datetime
from datetime import timedelta
import json
import os 
#from yahooquery import Ticker
#import yahooquery as yq

import pandas as pd

utc_now = datetime.now()
utc_now = utc_now.date()
algo = "YH-1Y-FC"
utc_end = utc_now  + timedelta(days = 365)
#from yahooquery import Ticker



def exportJSONFile (jsondata,filepath):
    #folderpath_fc ="D:/Projects/ai_system/data/lstm_socket/fc/"
    ticker_data_filename = filepath
    with open( ticker_data_filename, "w") as json_file:
        json.dump(jsondata, json_file)   




# ============ SETTINGS ==================
folderPath = "/Users/Shared/ai_work/Trainingdata/fadata/fundamentals"
folderpath_jobs = "/Users/Shared/ai_work/Trainingdata/ml_jobs/"

fileName ="fundamental_enterprise_analyse.csv"
fileName_checklist ="fa_checklist.csv"
csvfileresult  = "/Users/Shared/ai_work/Trainingdata/ml_temp/fa/"+ fileName
#"D:/OneDrive/AI Workspace/results/fa_analysis/"+ fileName
csvfile_joblist  = folderpath_jobs + fileName_checklist
folder ="/Users/Shared/ai_work/Trainingdata/fadata/fundamentals/"
#"D:/OneDrive/AI Workspace/data/fundamentals/"
fileName ="fundamental_analysis.csv"
csvfileresult  = "/Users/Shared/ai_work/Trainingdata/fadata/analyse/"+ fileName

# ============ END SETTINGS ==================

import os

def check_summary_detail ( symbol,ticker, folderPath): 
      result = ("ERROR")
      yield_min = 0.1
      mos = 0.2
      years = 10 
      
      epsTrailingTwelveMonths=0
      forwardPE = 0
      marketCap = 0
      trailingPE = 0
      price = 0
      #filePath = folderPath+symbol +"_quotes.json"
      filePath = folderPath+ "info.json"
      
      dividendYield = 0 
      fiveYearAvgDividendYield = 0 
      payoutRatio = 0 

      existFilePath = os.path.exists(filePath)

      if existFilePath == True: 
          # Load the JSON data
        with open(filePath, 'r') as f:
          js_quotes = json.load(f)
        #data_quotes =pd.read_json (filePath)
        #symbol_data.quotes
        #js_quotes = data_quotes[ticker]
        
        ask = js_quotes['ask']
        bid = js_quotes['bid']
        price = (ask + bid)/2
        
        if "dividendYield" in js_quotes:
            dividendYield = js_quotes['dividendYield']

        if "fiveYearAvgDividendYield" in js_quotes:
            fiveYearAvgDividendYield = js_quotes['fiveYearAvgDividendYield']

        if "epsTrailingTwelveMonths" in js_quotes:
            epsTrailingTwelveMonths = js_quotes['epsTrailingTwelveMonths']
    

        if "forwardPE" in js_quotes:
            forwardPE = js_quotes['forwardPE']
        if "trailingPE" in js_quotes:
            trailingPE = js_quotes['trailingPE']
        if "marketCap" in js_quotes:
            marketCap = js_quotes['marketCap']

        if "payoutRatio" in js_quotes: payoutRatio = js_quotes['payoutRatio']
        
         
         
        dividendYield_growth = 0 
        if fiveYearAvgDividendYield > 0 and dividendYield >0 : dividendYield_growth = (dividendYield - fiveYearAvgDividendYield)/fiveYearAvgDividendYield
        result = (trailingPE,forwardPE, marketCap, dividendYield, dividendYield_growth , payoutRatio)
        
        print ("Result:  ---- ", result)
      return result

      
def check_FinanceAnalysis (ticker, symbol):
      marketCap = 0
      result = (0,0,0)
       
      filePath = folderPath+symbol +"_quotes.json"
      existFilePath = os.path.exists(filePath)
      if existFilePath == True: 
        data_quotes =pd.read_json (folderPath+symbol +"_quotes.json")
        #symbol_data.quotes
        js_quotes = data_quotes[ticker]
      
        if "marketCap" in js_quotes:
            marketCap = js_quotes['marketCap']

      
      #---- CASH FLOW 
      
      filePath =folderPath+symbol +"_all_financial_data.csv"
      #"_cash_flow.csv"
     
      df_data= pd.read_csv (filePath)
      i = 0 
      stocks_Repurchase = 0 
     
      if "RepurchaseOfCapitalStock" in df_data:
            
        df_s = df_data[['periodType', 'RepurchaseOfCapitalStock']]
        #df[["date","open","high","low","close","volume","adjclose"]]
      
        for s in df_s.values:
            
            
            
            if  s[0] !='TTM' and  str(s[1]).lower  != 'nan' and float(s[1]) < 0 :
              i = i +1 
              
              stocks_Repurchase = stocks_Repurchase + (-1*float(s[1]))
             

            
            
      marketCap_change = stocks_Repurchase / marketCap
      marketCap_change = round (marketCap_change, 4)

    
      #---- INCOME STATEMENT 
      filePath =folderPath+symbol +"_income_statement.csv"
     
      #data = pd.read_csv (filePath)
     
      i = 0 
      valided_income = 1
      min_pretaxincome = 75000000
      sum_pretaxincome = 0
      pretaxincome = 0 
      
      if "PretaxIncome" in df_data:
            df_s = df_data[['periodType', 'PretaxIncome']]
            for s in df_s.values:
      
            
             if  s[0] !='TTM' and  str(s[1])  != 'nan':
                  i = i +1
                  if s[1] < min_pretaxincome: valided_income = 0
                  pretaxincome = s[1] 
                  
                  sum_pretaxincome = sum_pretaxincome + pretaxincome
                  
      
      
      avg_pretaxincome = sum_pretaxincome /(i)
      perf_pretaxincome = (pretaxincome - avg_pretaxincome)/avg_pretaxincome
      pretaxstatus = 0
      if pretaxincome > min_pretaxincome :pretaxstatus = 2
      
      if pretaxincome < min_pretaxincome :pretaxstatus = -2
      if perf_pretaxincome < 0 :pretaxstatus = pretaxstatus -1
      if perf_pretaxincome > 0 :pretaxstatus = pretaxstatus +1

      result = (marketCap_change, avg_pretaxincome, pretaxincome, perf_pretaxincome)
      print ( i, "marketCap_change: ", marketCap_change ,  "LAST PRETAX:  ",  pretaxincome , " // SUM: ",sum_pretaxincome, " avg_pretaxincome: " , avg_pretaxincome, "perf_pretaxincome: ", perf_pretaxincome, " pretaxstatus: ", pretaxstatus)

      return (result)

#import nltk 
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
def check_Insights (symbol,ticker, folderPath):
     result = ("ERROR",0,0,0,0)
     filePath = folderPath+symbol+"_technical_insights.json"
     existFilePath = os.path.exists(filePath)
     if existFilePath == False : return result
     with open(filePath, 'r') as f:
          js_data = json.load(f)


     #js_data = pd.read_json (filePath)

     #RECOMMENDATION ---- 
     targetPrice = 0 
     rating = 0 
     support = 0 
     resistance = 0 
     stopLoss = 0 

     #RECOMMENDATION
     if "recommendation" in js_data[ticker]:
          
      js_recom= js_data[ticker]['recommendation']
      
      if "targetPrice" in js_recom:  targetPrice = js_recom['targetPrice']
      if "rating" in js_recom:  rating = js_recom['rating']
     print ( " RECOM: ", targetPrice , " R: ", rating)
     #valuation - Bewewrtung
     description =""
     discount = 0
     if "instrumentInfo" in js_data[ticker] and  "valuation" in js_data[ticker]['instrumentInfo']:
      js_valuation= js_data[ticker]['instrumentInfo']['valuation']
      if "description" in js_valuation:  description = js_valuation['description']
      
      if "discount" in js_valuation: discount = js_valuation['discount']
     #relativeValue = js_valuation['relativeValue']
     print ( "valuation " , discount , " description: ", description )


     #SIGNALS ------ 
     tradingHorizon =""
     tradeType =""
     startDate =""
     endDate =""
     
     if "events" in js_data[ticker]:
      js_signal= js_data[ticker]['events'][0]
      start_timestamp =  js_signal['startDate']
      end_timestamp =  js_signal['endDate']
      tradingHorizon =  js_signal['tradingHorizon']
      tradeType =  js_signal['tradeType']
   
      startDate = str(datetime.fromtimestamp(start_timestamp))
      endDate = str(datetime.fromtimestamp(end_timestamp))
     
     #---TREND 
     trend_S = 0
     trend_I = 0
     trend_L = 0

     if  "instrumentInfo" in js_data[ticker] : 
      js_ta = js_data[ticker]['instrumentInfo']['technicalEvents']
     
      if "shortTermOutlook" in js_ta:
         direction = js_ta['shortTermOutlook']['direction']
         score = js_ta['shortTermOutlook']['score']
         if direction =="Bullish": trend_S = 1*score
         if direction =="Bearish": trend_S = -1*score

      if "intermediateTermOutlook" in js_ta:
         direction = js_ta['intermediateTermOutlook']['direction']
         score = js_ta['intermediateTermOutlook']['score']
         if direction =="Bullish" : trend_I = 1*score
         if direction =="Bearish": trend_I = -1*score

      if "longTermOutlook" in js_ta:
         direction = js_ta['longTermOutlook']['direction']
         score = js_ta['longTermOutlook']['score']
         if direction =="Bullish": trend_L = 1*score
     
      if "keyTechnicals" in js_data[ticker]['instrumentInfo']:
          js_keyTechnicals= js_data[ticker]['instrumentInfo']['keyTechnicals']
          
          if "support" in  js_keyTechnicals: support = js_keyTechnicals['support']
          if "resistance" in  js_keyTechnicals: resistance = js_keyTechnicals['resistance']
          if "stopLoss" in  js_keyTechnicals: stopLoss = js_keyTechnicals['stopLoss']
          
         
     #COMPANY SNAPSHOT
     innovativeness = 0
     hiring = 0 
     sustainability = 0 
     insiderSentiments = 0 
     earningsReports = 0 
     i = 0

     if "companySnapshot" in js_data[ticker]:
          js_company= js_data[ticker]['companySnapshot']['company']
          if "innovativeness" in  js_company : innovativeness = js_company['innovativeness']
          if "hiring" in  js_company: hiring = js_company['hiring']
          if "sustainability" in js_company: sustainability = js_company['sustainability']
          if "insiderSentiments" in js_company: insiderSentiments = js_company['insiderSentiments']
          if "earningsReports" in js_company: earningsReports = js_company['earningsReports']

     if innovativeness > 0: i= i+1
     if hiring > 0 : i = i+1
     if insiderSentiments > 0 : i=i+1
     if earningsReports > 0 : i=i+1

     companySnapshot = 0 
     if i > 0: companySnapshot = (innovativeness + hiring +insiderSentiments + earningsReports )/ i 

     

     
     print ("Company Info ", "innovativeness: ", innovativeness , " hiring: ", hiring, " sustainability: ", sustainability, " insiderSentiments: ", insiderSentiments, " earningsReports: ", earningsReports)
     
     print (  "support: ", support , " resistance: ", resistance , " stopLoss: ", stopLoss ,   "trend_S: ", trend_S ,  "trend_I: ", trend_I ,  "trend_L: ", trend_L ,  "SIGNAL ", startDate , "tradingHorizon: ", tradingHorizon,  "tradeType: ", tradeType , " END ", endDate )
     if discount !=0:
        discount = float (discount.replace ('%', ''))/100

     result = (targetPrice, discount, rating, tradeType,  tradingHorizon, trend_S, trend_I, trend_L, companySnapshot,  innovativeness, insiderSentiments  )
     return result

def check_earnings_trend ( symbol,ticker, folderPath): 
     
     result = ("ERROR",0,0,0,0)
     filePath = folderPath+"info.json"
     filePath1 = folderPath+"earnings_estimate.csv"
     existFilePath = os.path.exists(filePath)
     dfEarningEST = pd.read_csv(filePath1)

     # Get rows where city == 'Berlin'
     filtered_rows = dfEarningEST[dfEarningEST['period'] == '+1y']

     print(filtered_rows)


     if existFilePath == False : return result

     growth_1y = 0 
     growth_past_5y = 0
     
     #data_earnings_trend = pd.read_json (filePath)
     
     # Load the JSON data
     with open(filePath, 'r') as f:
       data = json.load(f)
     growth_1y = data.get('earningsQuarterlyGrowth')
     growth_past_5y = data.get('earningsGrowth')
     growth_5y =filtered_rows["growth"].values[0]
     print ("growth_5y: ", growth_5y)
     result=  (growth_1y,growth_5y, growth_past_5y )
     
     return result

        #(growth_1y + growth_5y + growth_past_5y)/3
        #=I10*(1+I15)^I43
     """
        eps_in_years = epsTrailingTwelveMonths* (1+growth)**years
    
        #=I10*(1+I15)^I43SS
        price_in_years = round(eps_in_years*forwardPE , 2)
        #=C18/(1+C13)^C15
        faire_value = round(price_in_years/(1+yield_min)**years,3)
        #=I47*(1-I42)
        price_mos = round( faire_value*(1-mos),2)
        price_discount = round ( (faire_value - price )/price ,3)
    """
     


def check_Key_Stats ( symbol,ticker, folderPath): 
     
     result = ("ERROR",0,0,0,0)
     filePath = folderPath+"info.json"
     existFilePath = os.path.exists(filePath)

     if existFilePath == False : return result
     
     #js_data = pd.read_json (filePath)
     #js_data = js_data[ticker]
     # Load the JSON data
     
     with open(filePath, 'r') as f:
       js_data = json.load(f)
     
     profitMargins = 0 
     shortRatio = 0 
     shortPercentOfFloat = 0 
     bookValue = 0 
     priceToBook = 0 
     trailingEps = 0 
     forwardEps = 0 
     pegRatio = 0 
     enterpriseToRevenue = 0 
     EpsGrowth = 0 
     sharesOutstanding = 0
     enterpriseValue = 0 

     if "profitMargins" in js_data:  profitMargins = js_data["profitMargins"]
     if "shortRatio" in js_data:  shortRatio = js_data["shortRatio"]
     if "shortPercentOfFloat" in js_data:  shortPercentOfFloat = js_data["shortPercentOfFloat"]
     if "bookValue" in js_data:  bookValue = js_data["bookValue"] # Eigenkapital / Volume. hohe bookValue Good 
     if "priceToBook" in js_data:  priceToBook = js_data["priceToBook"]
     if "trailingEps" in js_data:  trailingEps = js_data["trailingEps"]
     if "forwardEps" in js_data:  forwardEps = js_data["forwardEps"]
     if "pegRatio" in js_data:  pegRatio = js_data["pegRatio"] #KGV / Wachstum > 1 Überbewertet , < 1 unterbewertet
     if "enterpriseToRevenue" in js_data:  enterpriseToRevenue = js_data["enterpriseToRevenue"]
     if forwardEps > 0 and trailingEps  > 0: EpsGrowth = (forwardEps - trailingEps)/trailingEps
     
     if "sharesOutstanding" in js_data:  sharesOutstanding = js_data["sharesOutstanding"]
     if "enterpriseValue" in js_data:  enterpriseValue = js_data["enterpriseValue"]

   

     #Importent: priceToBook
     result = (trailingEps, forwardEps, EpsGrowth, pegRatio, priceToBook, profitMargins, shortRatio, shortPercentOfFloat, sharesOutstanding , enterpriseValue)
     return result

def check_financial_data ( symbol,ticker, folderPath): 
     
     result = ("ERROR",0,0,0,0)
     #filePath = folderPath+symbol+"_financial_data.json"

     filePath = folderPath+"info.json"
     print(filePath)
     existFilePath = os.path.exists(filePath)

     if existFilePath == False : return result

      # Load the JSON data
     with open(filePath, 'r') as f:
      js_data = json.load(f)

     #js_data = pd.read_json (filePath)
     #js_data = js_data[ticker]
    
     currentPrice = 0 
     targetMeanPrice = 0 
     recommendationMean = 0 
     recommendationKey = ""
     totalCashPerShare = 0 
     quickRatio = 0  # >1: OK!. kurzfristigen Verbindlichkeiten mit seinen am schnellsten verfügbaren liquiden 
     currentRatio  = 0 
     debtToEquity = 0 
     revenuePerShare = 0 

     psr = 0 #KUV
     returnOnAssets = 0 #ROA. Gesamtkapitalrenditen > hoohe ok. Sekor (2)
     returnOnEquity = 0 #ROE . Eigenkapotalrenditen. Hohe:OK  (2)
     revenueGrowth = 0 
     grossMargins = 0 # company's profitability.  Bruttomargen
     operatingMargins = 0 
     profitMargins = 0 
     

     totalRevenue = 0 
     totalCashPerShare = 0
     cashPriceRatio = 0
     earningsGrowth = 0 
     totalCash = 0 


     #kurzfristigen Verbindlichkeiten mit seinen am schnellsten verfügbaren liquiden 
     # > 1: OK 

     if "currentPrice" in js_data:  currentPrice = js_data["currentPrice"]
     
     if "targetMeanPrice" in js_data:  targetMeanPrice = js_data["targetMeanPrice"]
     if "recommendationMean" in js_data:  recommendationMean = js_data["recommendationMean"]
     if "recommendationKey" in js_data:  recommendationKey = js_data["recommendationKey"]
    
     if "totalCashPerShare" in js_data:  totalCashPerShare = js_data["totalCashPerShare"]
     if "quickRatio" in js_data:  quickRatio = js_data["quickRatio"]
     if "currentRatio" in js_data:  currentRatio = js_data["currentRatio"]
     if "debtToEquity" in js_data:  debtToEquity = js_data["debtToEquity"]

     if "revenuePerShare" in js_data:  revenuePerShare = js_data["revenuePerShare"]
     if "returnOnAssets" in js_data:  returnOnAssets = js_data["returnOnAssets"]
     if "returnOnEquity" in js_data:  returnOnEquity = js_data["returnOnEquity"]

     if "revenueGrowth" in js_data:  revenueGrowth = js_data["revenueGrowth"]
     if "grossMargins" in js_data:  grossMargins = js_data["grossMargins"]
     if "operatingMargins" in js_data:  operatingMargins = js_data["operatingMargins"]
     if "profitMargins" in js_data:  profitMargins = js_data["profitMargins"]

     if "totalRevenue" in js_data:  totalRevenue = js_data["totalRevenue"]
     if "totalCashPerShare" in js_data:  totalCashPerShare = js_data["totalCashPerShare"]
     if "earningsGrowth" in js_data:  earningsGrowth = js_data["earningsGrowth"]
     if "totalCash" in js_data:  totalCash = js_data["totalCash"]
     
     

     
     if revenuePerShare > 0 and currentPrice > 0: psr = round ( currentPrice / revenuePerShare, 2) 
     if totalCashPerShare > 0 and currentPrice > 0 : cashPriceRatio = round ( totalCashPerShare / currentPrice , 2) 
     result = ( currentPrice, targetMeanPrice, psr, currentRatio, revenueGrowth, debtToEquity, recommendationMean, recommendationKey, returnOnAssets, returnOnEquity, profitMargins, operatingMargins, 
               totalRevenue, totalCashPerShare, cashPriceRatio, earningsGrowth, totalCash
               )
     return result


#check_financial_data ("#NVDA","", folderPath)

def check_asset_profile ( symbol,ticker, folderPath): 
     
     result = ("ERROR",0,0,0,0)
     filePath = folderPath+symbol+"_asset_profile.json"
     existFilePath = os.path.exists(filePath)

     if existFilePath == False : return result
     js_data = pd.read_json (filePath)
     js_data = js_data[ticker]
     country =""
     industry =""
     sector =""
     fullTimeEmployees =""
     overallRisk =""

     if "country" in js_data:  country = js_data["country"]
     if "industry" in js_data:  industry = js_data["industry"]
     if "sector" in js_data:  sector = js_data["sector"]
     if "fullTimeEmployees" in js_data:  fullTimeEmployees = js_data["fullTimeEmployees"]
     if "overallRisk" in js_data:  overallRisk = js_data["overallRisk"]
     result = ( country, industry, sector, fullTimeEmployees, overallRisk)
     return result
    
def check_info ( symbol, folderPath): 
     
     result = ("ERROR",0,0,0,0)
     filePath = folderPath+"info.json"
     print ( "CheckInfo " , filePath)
     existFilePath = os.path.exists(filePath)

     if existFilePath == False : return result
    

     
     
     js_data = {}
     with open(filePath, 'r') as f:
          js_data = json.load(f)

     longName =""
     industry =""
     sector =""
     fullTimeEmployees =""
     overallRisk =""
     industryKey = ""
     sectorKey =""
     overallRisk =""
     ticker = ""
     country =""

     ticker = js_data["symbol"]
     print ( " Ticker Info ORG : ", ticker)
     if "country" in js_data:  country = js_data["country"]
     if "industry" in js_data:  industry = js_data["industry"]
     if "sector" in js_data:  sector = js_data["sector"]
     if "fullTimeEmployees" in js_data:  fullTimeEmployees = js_data["fullTimeEmployees"]
     if "overallRisk" in js_data:  overallRisk = js_data["overallRisk"]
     if "industryKey" in js_data:  overallRisk = js_data["industryKey"]
     if "sectorKey" in js_data:  overallRisk = js_data["sectorKey"]
    
     result = ( country, industry, sector, fullTimeEmployees, overallRisk, industryKey, sectorKey, fullTimeEmployees, ticker)
     return result
    

from google.cloud import bigquery
def loadCSVFileToBigQuery (filePath_fa):
    existFilePath = os.path.exists(filePath_fa)
    if existFilePath == False:
          print ("Error filePath FA")
          return 
    PROJECT_ID = 'keen-oasis-204713'
    client = bigquery.Client()
    # create a new datset
    #client.create_dataset("new_dataset")
    dataset_id = 'finance'
    table_id = 'fundamental_analysis'
    
    client.delete_table(f"{PROJECT_ID}.{dataset_id}.{table_id}", not_found_ok=True)  # Make an API request.
    client.create_table(f"{PROJECT_ID}.{dataset_id}.{table_id}")

    #Dataset(DatasetReference('kaggle-bq-test-248917', 'new_dataset'))
    # some variables
    
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.autodetect = True
    job_config.max_bad_records = 300
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    # load the csv into bigquery
    with open(filePath_fa, "rb") as source_file:
      job = client.load_table_from_file(source_file, table_ref , job_config=job_config )

    job.result()  # Waits for table load to complete.

    # looks like everything worked :)
    print("Loaded {} rows into {}:{}.".format(job.output_rows, dataset_id, table_id))

def getFaireValueDCF (totalCash, sharesOutstanding, enterpriseValue):
     
     fairevalue = 0
     ticker = 'TSM'
     growth_rate = 0.12
     terminal_growth_rate = 0.03
     wacc = 0.071
     
     net_cash = totalCash#21_000_000_000  # in USD
     years = [2025, 2026, 2027, 2028, 2029]
     fcfs = []
     fcf = totalCash
     for year in years:
       fcf *= (1 + growth_rate)
       fcfs.append(fcf)

    # Berechnung des Terminal Value
     terminal_value = fcfs[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

     # Diskontierung der FCFs und des Terminal Value
     discounted_fcfs = [fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcfs)]
     discounted_terminal_value = terminal_value / ((1 + wacc) ** len(fcfs))

     # Unternehmenswert berechnen
     enterprise_value = enterpriseValue
     #sum(discounted_fcfs) + discounted_terminal_value

     # Eigenkapitalwert berechnen
     equity_value = enterprise_value + net_cash

     # Fairer Wert pro Aktie
     fair_value_per_share = equity_value / sharesOutstanding

     return fair_value_per_share



def analyseSymbolFAData ( symbol, writer):
      """
      importante Identificators
      -----Retability: 
      trailingEps, forwardEps
      KGV: trailingPE, forwardPE
      pegRatio: < 1: Under fairevalue OK  , > 1 About fairevalue
      ----------


      """
      #datenow= str(datetime.now.date())
      foderpath_fa = folderPath+"/"+symbol +"/"

      filePath = foderpath_fa+"info.json"
      existFilePath = os.path.exists(filePath)
      if existFilePath == False : return False
      js_info = {}
      with open(filePath, 'r') as f:
          js_info = json.load(f)
      #ticker = js_info["symbol"] 
      #print ( " Ticker Info ORG : ", ticker)
      
     #------- INFO  ----------------------------------------------------------
      st_check_info_data = check_info ( symbol, foderpath_fa)
      ticker = st_check_info_data[8]
      print ( " Ticker: ", ticker)

      #result = ( country, industry, sector, fullTimeEmployees, overallRisk, industryKey, sectorKey)

      #price = js_info['price']
      #filePath = foderpath_fa+ symbol +"_all_financial_data.csv"
      
      #print ( filePath)
      #existFilePath = os.path.exists(filePath)
      
      
      
      #-------FINANCIAL DATA ----------------------------------------------------------
      st_check_financial_data = check_financial_data (symbol, ticker, foderpath_fa)
      price = st_check_financial_data[0]
      #result = ( currentPrice, targetMeanPrice, psr, currentRatio, revenueGrowth, debtToEquity, recommendationMean, recommendationKey)
      

      targetMeanPrice = 0 
      
      psr = 0 #KUV
      currentRatio = 0 
      revenueGrowth = 0 
      debtToEquity = 0 
      tp1  =  0 
      recommendationMean = 0 
      ratting1 = ""
      returnOnAssets = 0 
      returnOnEquity = 0 
      profitMargins = 0 
      operatingMargins = 0

      #totalRevenue, totalCashPerShare, cashPriceRatio, earningsGrowth
      totalCashPerShare = 0 
      totalRevenue = 0 
      cashPriceRatio = 0 
      earningsGrowth = 0 
      totalCash  = 0 
      cashToRevenueRatio = 0 
      if price !="ERROR":
           targetMeanPrice = st_check_financial_data[1]
           #Income Statement
           psr = st_check_financial_data[2]

           #---Balance Sheet
           currentRatio = st_check_financial_data[3]
           revenueGrowth = st_check_financial_data[4]
           debtToEquity = st_check_financial_data[5]
           #------
           recommendationMean = st_check_financial_data[6]
           ratting1 = st_check_financial_data[7]

           #--Management Effectiveness 
           returnOnAssets = st_check_financial_data[8]
           returnOnEquity = st_check_financial_data[9]

           #--Profitability 
           profitMargins = st_check_financial_data[10]
           operatingMargins = st_check_financial_data[11]

           
           totalRevenue = st_check_financial_data[12]
           totalCashPerShare = st_check_financial_data[13]
           cashPriceRatio = st_check_financial_data[14]
           earningsGrowth = st_check_financial_data[15]
           totalCash =  st_check_financial_data[16]
           if totalCash > 0 and totalRevenue > 0 : cashToRevenueRatio = round ( (totalCash / totalRevenue),3) 

      
   
      
      
      st_check_summary_detail = check_summary_detail ( symbol, ticker, foderpath_fa)

      trailingPE = 0 # Laufende KGV
      forwardPE = 0 # 3. Zukunft KGV
      
      marketCap = 0
      dividendYield = 0 
      dividendYield_growth = 0 
      payoutRatio = 0 

      trailingPE = st_check_summary_detail[0]
      if trailingPE !="ERROR":
           forwardPE = st_check_summary_detail[1]
           marketCap = st_check_summary_detail[2]
           dividendYield = st_check_summary_detail[3]
           dividendYield_growth = st_check_summary_detail[4]
           payoutRatio = st_check_summary_detail[5]
           
       
      #result = (trailingPE,forwardPE, marketCap, dividendYield, dividendYield_growth )


      print ( " price: ", price)

      st_check_Insights = check_Insights (symbol, ticker, foderpath_fa)
      #(targetPrice, discount, rating, tradeType,  tradingHorizon, trend_S, trend_I, trend_L, companySnapshot,  innovativeness, insiderSentiments  )
      # result = (targetPrice, discount, rating, tradeType,  tradingHorizon, trend_S, trend_I, trend_L, companySnapshot,  innovativeness, insiderSentiments  )
      targetPrice = 0 
      discount = 0 
      ratting2 =""
      tradeType =""
      tradingHorizon = ""
      trend_S = 0 
      trend_I = 0
      trend_L = 0 
      companySnapshot = 0
      innovativeness = 0 
      insiderSentiments = 0 
      fairevalue = 0 


      targetPrice = st_check_Insights[0]

      if targetPrice !="ERROR": 
         discount = st_check_Insights[1]
         print(" price ", price , "discount" , discount)
         fairevalue = float(price) + discount*float(price)
         ratting2 = st_check_Insights[2]

         tradeType = st_check_Insights[3]
         tradingHorizon = st_check_Insights[4]
         trend_S = st_check_Insights[5]
         trend_I = st_check_Insights[6]
         trend_L = st_check_Insights[7]
         companySnapshot = st_check_Insights[8]
         innovativeness = st_check_Insights[9]
         insiderSentiments = st_check_Insights[10]


      
      
      # Key Stats: 
      # ---- KEY STATS -------------------
      st_check_Key_Stats = check_Key_Stats (symbol, ticker, foderpath_fa)
      #result = (trailingEps, forwardEps, espValue, pegRatio, priceToBook, profitMargins, shortRatio, shortPercentOfFloat )
      trailingEps = 0 
      forwardEps = 0 
      EpsGrowth = 0 ; 
      pegRatio = 0 
      priceToBook = 0 
      
      shortRatio = 0
      shortPercentOfFloat = 0 
      sharesOutstanding = 0
      enterpriseValue = 0 
      trailingEps = st_check_Key_Stats[0]
      if trailingEps !="ERROR":
           forwardEps = st_check_Key_Stats[1]
           EpsGrowth = st_check_Key_Stats[2]
           pegRatio = st_check_Key_Stats[3]
           priceToBook = st_check_Key_Stats[4]
           profitMargins = st_check_Key_Stats[5]
           shortRatio = st_check_Key_Stats[6]
           shortPercentOfFloat = st_check_Key_Stats[7]

           sharesOutstanding = st_check_Key_Stats[8]
           enterpriseValue = st_check_Key_Stats[9]
      

      #------ EARNING TREND -------------------------
      st_check_earnings_trend = check_earnings_trend (symbol, ticker, foderpath_fa)
      #result=  (growth_1y,growth_5y, growth_past_5y )
      growth_1y = 0 
      growth_5y = 0 
      growth_past_5y = 0 
      growth_1y = st_check_earnings_trend[0]
      if growth_1y !="ERROR":
           growth_5y = st_check_earnings_trend[1]
           growth_past_5y = st_check_earnings_trend[2]
      
      #----Fairevalue Cal ++
      n_years = 5
      fairevalue_eps_5y = 0 
      fairevalue_eps_today = 0 
      fairevalue_eps_forward_5y = 0 
      fairevalue_eps_forward_today = 0 
      fairevalue_graham_5y = 0
      growth_5y = float(growth_5y) 
      trailingEps = float(trailingEps)
      trailingPE  = float(trailingPE)
      discount_rate = 0.12 # Rendite erwartung für this stocks 0.08 - 0.12  ( %)
      #if growth_past_5y > 0 and trailingEps > 0 : fairevalue_graham = trailingEps *(8.5+2*growth_past_5y)*0.65
      if growth_5y > 0 and forwardEps > 0 : fairevalue_graham_5y = forwardEps *(8.5+2*growth_5y)*0.65
      if growth_5y > 0 and trailingEps > 0 and trailingPE > 0: 
          
           EPS_5y = trailingEps*(1+growth_5y)**n_years
           fairevalue_eps_5y =EPS_5y*trailingPE
           if forwardPE > 0: fairevalue_eps_forward_5y = EPS_5y*forwardPE
      
      if fairevalue_eps_5y > 0: fairevalue_eps_today = round (fairevalue_eps_5y / ((1+discount_rate)**n_years) ,2) 
      if fairevalue_eps_forward_5y > 0: fairevalue_eps_forward_today = round (fairevalue_eps_forward_5y / ((1+discount_rate)**n_years) ,2) 
      
      #Check Asset Profile 
      print (" DCF ", totalCash , " sharesOutstanding " , sharesOutstanding , " enterpriseValue ", enterpriseValue)
      fairevalueDCF = getFaireValueDCF (totalCash, sharesOutstanding, enterpriseValue)

      #--- SCORING FA Data ---- 
       
      """
       psr #hlow > OK 
       debtToEquity 
       priceToBook # 
       """
      print ( "'''''' ", symbol ,  " fairevalueDCF: ", fairevalueDCF)
      scoring_fa = "1"
      #---- Write Reults 
      #result = ( country, industry, sector, fullTimeEmployees, overallRisk, industryKey, sectorKey)

      writer.writerow({'symbol': symbol, 'price': str(price) , 'scoring_fa':str(scoring_fa), 'targetMeanPrice': str(targetMeanPrice) , 'psr': str(psr), 'currentRatio': str(currentRatio),'revenueGrowth': str(revenueGrowth), 'ratting1': str(ratting1),'ratting2': str(ratting2),
                 'trailingPE': str(trailingPE), 'forwardPE': str(forwardPE),  'marketCap': str(marketCap), 'dividendYield': str(dividendYield),'dividendYield_growth':str(dividendYield_growth), 'targetPrice': str(targetPrice),'discount': str(discount),
                 'fairevalue_base': str(fairevalueDCF) ,
				 'fairevalue_eps_today': str(fairevalue_eps_today), 'fairevalue_eps_5y': str(fairevalue_eps_5y), 'fairevalue_eps_forward_today': str(fairevalue_eps_forward_today), 'fairevalue_eps_forward_5y': str(fairevalue_eps_forward_5y), 
                 'fairevalue_graham_5y': str(fairevalue_graham_5y), 'fairevalue': str(fairevalue), 'tradeType': str(tradeType), 'tradingHorizon': str(tradingHorizon), 'trend_S': str(trend_S), 'trend_I': str(trend_I), 'trend_L': str(trend_L), 'companySnapshot': str(companySnapshot),
				 'innovativeness': str(innovativeness),'insiderSentiments': str(insiderSentiments), 'trailingEps': str(trailingEps), 'forwardEps': str(forwardEps), 'EpsGrowth': str(EpsGrowth), 'pegRatio': str(pegRatio),
				 'priceToBook': str(priceToBook), 'profitMargins': str(profitMargins), 'operatingMargins': str(operatingMargins), 'returnOnAssets': str(returnOnAssets), 'returnOnEquity': str(returnOnEquity), 
                 'shortRatio': str(shortRatio), 'shortPercentOfFloat': str(shortPercentOfFloat), 'growth_1y': str(growth_1y),
				 'growth_5y': str(growth_5y), 'growth_past_5y': str(growth_past_5y), 'debtToEquity':str(debtToEquity), 
                 'totalRevenue':str(totalRevenue), 'totalCashPerShare':str(totalCashPerShare), 'cashPriceRatio': str(cashPriceRatio), 'earningsGrowth':str(earningsGrowth), 
                 'totalCash':str(totalCash), 'cashToRevenueRatio':str(cashToRevenueRatio), 'payoutRatio':str(payoutRatio), 'created':str(utc_now),
                 'country': str(st_check_info_data[0]), 'industry': str(st_check_info_data[1]), 'sector': str(st_check_info_data[2]),
                     'industryKey': str(js_info['industryKey']), 'sectorKey': str(js_info['sectorKey']), 'fullTimeEmployees': str(st_check_info_data[3]), 'overallRisk':  str(js_info['overallRisk']), 
                         'longName': str(js_info['longName']), 'ticker': str(ticker), 'currency': str(js_info['currency']), 'market': str(js_info['market'])
                     


				 
	              })  



def runFromFolder (fc_symbol):
    #df_futures = markets.getDataFrameMarkets ("D1")
    print ("FILE: " +  csvfileresult )
    with open(csvfileresult , 'w', newline='') as csvfilerow:
                           fieldnames = ['symbol','ticker', 'longName','price','scoring_fa', 'targetMeanPrice','psr','currentRatio', 'revenueGrowth','ratting1','ratting2', 'trailingPE','forwardPE','marketCap','dividendYield','dividendYield_growth', 
                                         'targetPrice','discount', 'fairevalue_base','fairevalue_fwd','fairevalue', 'fairevalue_eps_5y', 'fairevalue_eps_forward_5y', 'fairevalue_eps_today', 'fairevalue_eps_forward_today', 'fairevalue_graham_5y', 'tradeType', 'tradingHorizon', 'trend_S', 'trend_I', 'trend_L', 'companySnapshot', 'innovativeness', 'insiderSentiments', 'trailingEps', 'forwardEps', 'EpsGrowth', 'pegRatio', 
			                             'priceToBook', 'profitMargins', 'operatingMargins', 'returnOnAssets','returnOnEquity', 'shortRatio', 'shortPercentOfFloat', 'growth_1y', 'growth_5y', 'growth_past_5y',
                                         'debtToEquity', 'totalRevenue', 'totalCashPerShare', 'cashPriceRatio', 'earningsGrowth', 'totalCash', 'cashToRevenueRatio','payoutRatio',
                                         'country','industry','sector','industryKey', 'sectorKey' , 'fullTimeEmployees','overallRisk','longName','currency','market',
                                         'created']
                           writer = csv.DictWriter(csvfilerow, fieldnames=fieldnames)
                           writer.writeheader()
                           listOfFiles = os.listdir(folder)
                           for entry in listOfFiles:
      
                              symbol = entry
                              i_run = 1
       
                              if fc_symbol and symbol != fc_symbol:
                                 i_run = 0

                              if i_run == 1:
                                  #analyseSymbolFAData (symbol, writer)
                               
                                  try: 
                                        print ("check FA " , symbol)
                                        analyseSymbolFAData (symbol, writer)
                                  except:
                                        print ("Error FA Analysis  " , symbol)          
                                
    loadCSVFileToBigQuery (csvfileresult)





datetime_utc_cv =  " dateadd(hh, -1, getDate() )  "
weekday_utc = "DATEPART ( weekday ," + datetime_utc_cv+  ")"
#query = "  (iswatching > 0 or isinvest > 0 ) and category ='Stock'   and weekdayopen <= " + weekday_utc + " and weekdayend >= " + weekday_utc+ " and cast (opentime as time) <=  cast(getDate() as time)  and  cast (closetime as time) > cast ( dateadd(hh, 4, getDate() ) as time) and cast(getDate() as datetime)  > cast( dateadd(hh,2 ,i_lastupdated) as datetime) "
#query = " symbol ='#PLUG' and (iswatching > 0 or isinvest > 0 ) and category ='Stock'   and weekdayopen <= " + weekday_utc + " and weekdayend >= " + weekday_utc+ " and  cast (closetime as time) > cast ( dateadd(hh, 4, getDate() ) as time) and cast(getDate() as datetime)  > cast( dateadd(hh,2 ,i_lastupdated) as datetime) "

query = " (iswatching > 0 or isinvest > 0 ) and ( symbol ='#PYPL' )  "

#query = " isinvest =5 "







fc_symbol = ""
#runFromFolder (fc_symbol)
st_start = str(datetime.now())

jobnamefile ="finished_analyse_fa.txt"
actionjob ="analyse_fa.txt"
filePath_actionjob = folderpath_jobs + actionjob
existJob =  os.path.exists(filePath_actionjob)
#fa_analyse.runFromFolder (fc_symbol)
#runFromFolder (fc_symbol)
if os.path.exists(filePath_actionjob):
    print ("Start FA Analysis: ", filePath_actionjob)
    
    os.remove (filePath_actionjob)
    runFromFolder (fc_symbol)
    f= open( os.path.join( folderpath_jobs ,  jobnamefile),"w+")
    st_end =  str(datetime.now())
    description ="upload FA"
    f.write ( jobnamefile +"; " + st_start+ "; " + st_end +"; OK"+ "; DESC: " + description)

