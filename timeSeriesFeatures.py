#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import yfinance as yf
#%%

def addTimeFeatures(data):
    
    data['Date'] = pd.to_datetime(data['Date'])
    data['year']    = data['Date'].dt.year
    data['quarter'] = data['Date'].dt.quarter
    data['month']   = data['Date'].dt.month
    monthCat = pd.DataFrame(to_categorical(data.loc[:,['month']], num_classes=13))
    data = pd.concat([data,monthCat.loc[:,1:]],axis=1)
    data['day']     = data['Date'].dt.day
    data['wday']    = data['Date'].dt.dayofweek
    dayCat = pd.DataFrame(to_categorical(data.loc[:,['wday']], num_classes=8),
                          columns=["Monday","Tuesday","Wednesday",
                                   "Thursday","Friday","Saturday","Sunday","wday"])
    data = pd.concat([data,dayCat.iloc[:,:-1]],axis=1)
    data = data.drop('Date',1) 
    cols = list(data)
    cols.insert(0, cols.pop(cols.index('Close')))
    data = data.loc[:, cols]
    
    return(data)


#%%
def addSMAFeature(data, colName, window_size):
    data[f'{colName} {window_size}d_sma'] = data[colName].rolling(window=window_size, min_periods=1).mean()
    return(data)

def addEMAFeature(data, colName, alpha):
    data[f'{colName} {alpha}_ema'] = data[colName].ewm(alpha=alpha, adjust=False).mean()
    return(data)

def addLagFeature(data, colName, lag):
    data[f'{colName} t-{lag}'] = data[colName].shift(lag).fillna(0)
    return(data)


