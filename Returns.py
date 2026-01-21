import pandas as pd
import numpy as np
import datetime
import math

def GetReturn(df,date,lookback):
    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date","S5SFTW","S5PHRM","S5CPGS","S5ENRSX","S5FDBT","S5TECH","S5RETL","S5BANKX","S5HCES","S5DIVF","S5UTILX","S5MEDA","S5REAL","S5TELSX","S5MATRX","S5INSU","S5FDSR","S5HOUS","S5SSEQX","S5TRAN","S5HOTR","S5CODU","S5AUCO","S5COMS"]].copy()
    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback) ]
    returns_df.drop(columns="Date",inplace=True)
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily
    return returns_df

#return a df of size (lookback, number of sectors) with log returns


def GetReturnSPX(df,date,lookback):
    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date","SPX"]].copy()

    date_list=returns_df.drop(columns="Date")
    date_index = returns_df.index[returns_df["Date"] == date][0]

    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback) ]
    returns_df.drop(columns="Date",inplace=True)

    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily

    return returns_df


#residual momentum
def GetFullReturnsForResidual(df):
    returns_df = df[["S5SFTW","S5PHRM","S5CPGS","S5ENRSX","S5FDBT","S5TECH","S5RETL","S5BANKX","S5HCES","S5DIVF","S5UTILX","S5MEDA","S5REAL","S5TELSX","S5MATRX","S5INSU","S5FDSR","S5HOUS","S5SSEQX","S5TRAN","S5HOTR","S5CODU","S5AUCO","S5COMS"]].copy()
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    return returns_df
    #print(returns_df.std().mean()) #verification if std is around 1% daily


def GetReturnDailyIndividual(df,date,lookback,ticker):

    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date",ticker]].copy()
    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback)]
    returns_df.drop(columns="Date",inplace=True)
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily
    return returns_df



def aa(df,date,lookback,ticker):

    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date",ticker]].copy()
    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback)]
    returns_df.drop(columns="Date",inplace=True)
    compute = np.log(returns_df.iloc[0,0]/ returns_df.iloc[-1,0])
    return compute

def GetFullReturnsForResidualSPXDaily(df,date,lookback):

    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date","SPX"]].copy()

    date_list=returns_df.drop(columns="Date")
    date_index = returns_df.index[returns_df["Date"] == date][0]

    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback) ]
    returns_df.drop(columns="Date",inplace=True)
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily

    return returns_df


def GetReturnMonthly(df,date,lookback):
    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date","S5SFTW","S5PHRM","S5CPGS","S5ENRSX","S5FDBT","S5TECH","S5RETL","S5BANKX","S5HCES","S5DIVF","S5UTILX","S5MEDA","S5REAL","S5TELSX","S5MATRX","S5INSU","S5FDSR","S5HOUS","S5SSEQX","S5TRAN","S5HOTR","S5CODU","S5AUCO","S5COMS"]].copy()
    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback)]
    returns_df.drop(columns="Date",inplace=True)
    returns_df=returns_df.iloc[::21].copy()
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily
    return returns_df

#return a df of size (lookback, number of sectors) with log returns
def GetReturnMonthlyIndividual(df,date,lookback,ticker):
    date = pd.to_datetime(date)
    if date not in df["Date"].values:  # add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date", ticker]].copy()
    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df = returns_df[(returns_df.index <= date_index) & (returns_df.index >= date_index - lookback)]
    returns_df.drop(columns="Date", inplace=True)
    returns_df=returns_df.iloc[::21].copy()
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily
    return returns_df

def GetReturnSPXMonthly(df,date,lookback):
    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date","SPX"]].copy()

    date_list=returns_df.drop(columns="Date")
    date_index = returns_df.index[returns_df["Date"] == date][0]

    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback) ]
    returns_df.drop(columns="Date",inplace=True)
    returns_df=returns_df.iloc[::21].copy()
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    #print(returns_df.std().mean()) #verification if std is around 1% daily

    return returns_df

#return a df of size (lookback, 1) with log returns of SPX

