import numpy as np
from binance.client import Client
import pandas as pd
import time
from datetime import timedelta
import pandas_ta as ta
from sys import exit
import yaml
import argparse
import os
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import requests
import aiohttp
import asyncio
from datetime import date
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mpticker
import datetime
import pytz
from kneed import KneeLocator
from datetime import datetime

def mydate(x,pos):
    try:
        est = pytz.timezone('US/Eastern');date_format = "%H:%M"
        return datetime.fromtimestamp(x, tz=est).strftime(date_format)
    except IndexError:
        return ''

def plot_stock_data(data):
    fig, ax = plt.subplots()
    ax1 = plt.subplot2grid((5,1), (0,0), rowspan=4)
    ax2 = plt.subplot2grid((5,1), (4,0), sharex=ax1)
    symbol='CryptoUSDT'; start ='12345'; 

    ax1.set_title("{} - {}".format(symbol, start))
    ax1.set_facecolor("#131722")
    ax1.xaxis.set_major_formatter(mpticker.FuncFormatter(mydate))

    candlestick_ohlc(ax1, data.to_numpy(), width=8, colorup='#77d879', colordown='#db3f3f')

    ax2.bar(data['Time'], data['Volume'], width=30)
    ax2.xaxis.set_major_formatter(mpticker.FuncFormatter(mydate))
    fig.subplots_adjust(hspace=0)
    fig.autofmt_xdate()
    return ax1

def get_optimum_clusters(df):
    sum_of_sq_distances = []
    X = np.array(df)
    K = range(1,10)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X.reshape(-1,1))
        sum_of_sq_distances.append(km.inertia_)
    kn = KneeLocator(K, sum_of_sq_distances,S=1.0, curve="convex", direction="decreasing")
    kmeans = KMeans(n_clusters= kn.knee).fit(X.reshape(-1,1))
    c = kmeans.predict(X.reshape(-1,1))
    min_and_max = []
    for i in range(kn.knee):
        min_and_max.append([-np.inf,np.inf])
    for i in range(len(X)):
        cluster = c[i]
        if X[i] > min_and_max[cluster][0]:
            min_and_max[cluster][0] = X[i]
        if X[i] < min_and_max[cluster][1]:
            min_and_max[cluster][1] = X[i]
    return min_and_max


def s_r(df):
  df["Time"] = [d.timestamp() for d in df.Time]
  df['Open']=df['Open'].astype(float)
  df['Close']=df['Close'].astype(float)
  df['Low']=df['Low'].astype(float)
  df['High']=df['High'].astype(float)
  df['Volume']=df['Volume'].astype(float)
  df=df.drop(['close_time', 'ignore','tb_quote_av','quote_av','trades','tb_base_av'], axis = 1)

  ax = plot_stock_data(df)

  close = pd.DataFrame(data=df, index=df.index, columns=["Close"])
  clusters = get_optimum_clusters(close)
  for low in clusters:
    support =  min(low[1], key=lambda x:abs(x-df['Close'].astype(float).iloc[-1]))
    resistance= min(low[0], key=lambda x:abs(x-df['Close'].astype(float).iloc[-1]))
    ax.axhline(low[0], color='yellow', ls='--')
    ax.axhline(low[1], color='red', ls='--')
  plt.show()
  return support, resistance


def str_to_json(output):
  str1=''
  for k, v in output.items():
    for i,j in v.items():
      str1=str1+"           "+i
      for l,m in j.items():
        str1=str1+" "+l+" :"+str(m)
  return str1        

def telegram_bot_sendtext(bot_message):  
    bot_token = '5050978424:AAHdYxFryB7JuBp8vp8ufi0mLNgHkpHxrPs'
    bot_chatID = '5093729255'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return response.json()
      
def main():  
  access_key="8maDcVyKIwBtc0i32IvLD0jwca8nZ6VfrFkFprWThGkntHYH5gMqY2tuu9yOpkO1"
  secret_key ="dqQXXSZGaBL9jCJw3ttz2aKD6TW5IyeGirFoDo8NmkkhYMmycn492I9SASaG9meR"
  client = Client(access_key, secret_key)
  return client

def get_data(coin, kline_interval, since):
    klines = client.get_historical_klines(coin, kline_interval, since)
    df = pd.DataFrame(klines, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    df['Time'] = pd.to_datetime(df.Time, unit='ms')
    return df
    

def remove_coins(prices):
  initial_price = {}
  FIATS=['EURUSDT','GBPUSDT','JPYUSDT','USDUSDT','DOWN','UP']
  for coin in prices:
    if 'USDT' in coin['symbol'] and all(item not in coin['symbol'] for item in FIATS):
      initial_price[coin['symbol']] = { 'price': coin['price'], 'time': datetime.now()}
  return initial_price

def check():
  volatile_coins = {}
  prices = client.get_all_tickers()
  initial_price=remove_coins(prices)
  while initial_price['BNBUSDT']['time'] > datetime.now() - timedelta(minutes=15):
    print(f'not enough time has passed yet...')
    time.sleep(900)
  else:
    prices = client.get_all_tickers()
    last_price=remove_coins(prices)
  for coin in initial_price:
    threshold_check = (float(last_price[coin]['price']) - float(initial_price[coin]['price'])) / float(initial_price[coin]['price']) * 100
    if threshold_check  > 2.5:
      volatile_coins[coin] = threshold_check
      volatile_coins[coin] = round(volatile_coins[coin], 3)
  return volatile_coins

def var():
  KLINE_INTERVAL_15MINUTE='15m'; KLINE_INTERVAL_30MINUTE='30m'; KLINE_INTERVAL_1HOUR='1h';KLINE_INTERVAL_6HOUR='6h';  
  map=[{'kline':KLINE_INTERVAL_15MINUTE,'since':'2 day ago','window':20,'weights':0.17},{'kline':KLINE_INTERVAL_15MINUTE,'since':'2 day ago','window':50,'weights':0.13},
       {'kline':KLINE_INTERVAL_15MINUTE,'since':'2 day ago','window':100,'weights':0.08},{'kline':KLINE_INTERVAL_30MINUTE,'since':'5 day ago','window':20,'weights':0.11},
       {'kline':KLINE_INTERVAL_30MINUTE,'since':'5 day ago','window':50,'weights':0.07},{'kline':KLINE_INTERVAL_30MINUTE,'since':'5 day ago','window':100,'weights':0.05},
       {'kline':KLINE_INTERVAL_1HOUR,'since':'20 day ago','window':20,'weights':0.10},{'kline':KLINE_INTERVAL_1HOUR,'since':'20 day ago','window':50,'weights':0.07},
       {'kline':KLINE_INTERVAL_1HOUR,'since':'20 day ago','window':100,'weights':0.05},{'kline':KLINE_INTERVAL_6HOUR,'since':'50 day ago','window':20,'weights':0.09},
       {'kline':KLINE_INTERVAL_6HOUR,'since':'50 day ago','window':50,'weights':0.06},{'kline':KLINE_INTERVAL_6HOUR,'since':'50 day ago','window':100,'weights':0.02}]
  return map

def trend_score(coin):
  map=var()
  trend_indicator=0
  for i in range(0,len(map)):
    kline=map[i].get('kline'); since=map[i].get('since');
    weight=map[i].get('weights'); w=map[i].get('window');
    df=get_data(coin,kline,since)
    df['Close']=df['Close'].astype(float)
    df['ma']=ta.ema(df['Close'], length=w, talib=None, offset=None)
    df['ma']=df['ma'].astype(float)
    if int(df["Close"].iloc[-1]>df['ma'].iloc[-1]):
      trend_indicator = trend_indicator + weight * 1
    else:
      trend_indicator = trend_indicator + (weight * -1)
  return round(trend_indicator,2) 

def momentum(closes):
    returns = np.log(closes)
    indices = np.arange(len(returns))
    slope, _, r, _, _ = linregress(indices, returns)
    return round((((np.exp(slope) ** 1440) - 1) * 100) * (r**2),2)

def beta(df):
  coin=['BTCUSDT'];kline_interval=Client.KLINE_INTERVAL_1MINUTE; since='24h';
  df1=get_data(coin[0],kline_interval,since)
  df1['log']=df1['Close'].astype(float).pct_change()
  df1 = df1[df1['log'].notna()]
  x = np.array(df['log']).reshape((-1,1))
  y = np.array(df1['log'])
  if len(x) == len(y):
    model = LinearRegression().fit(x, y)
    beta=model.coef_
    return beta
  else:
    return str('null')  

def get_donchi(df):
  if len(df)>0:
    df["upper_bound"] = df["High"].astype(float).shift(1).rolling(window=50).max()
    if (float(df['Close'].iloc[-1]) > float(df['upper_bound'].iloc[-1])):
      return 1
    else:
      return 0  


def Volatility_of_coin(coin,move):
  vol={}
  kline_interval=Client.KLINE_INTERVAL_5MINUTE; since='12h'
  df=get_data(coin,kline_interval,since)
  df['log']=(df['Close'].astype(float).pct_change())*100
  volatilty=df['log'].std()
  if volatilty > 0:
    donchi=get_donchi(df)
    trend=trend_score(coin)
    mom=momentum(df['Close'].astype(float))
    beta1=beta(df)
    support, resistance =s_r(df)
    if trend > 1 and mom > 0 and donchi == 1 :
      vol[coin]={'volatilty':volatilty , 'trend':trend ,'momentum':mom, 'beta': beta1 ,'donchian':donchi, 'support':support , 'resistance':resistance , 'move':move}
  return vol

if __name__ == '__main__':
  client=main()
  while True:
    coins=check()
    output={}
    coins = coins if isinstance(coins, list) else [coins]
    coins[0]=dict(sorted(coins[0].items(), key=lambda x:x[1],reverse=True))
    if coins[0] is not None:
      coin=list(coins[0].keys())
      for key in coin:
        move=coins[0].get(key)
        output[key]=Volatility_of_coin(key,move)
    str1=str_to_json(output)
    print(str1)
    test = telegram_bot_sendtext(str1)


