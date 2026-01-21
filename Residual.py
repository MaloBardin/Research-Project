import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime

from Returns import GetReturnDailyIndividual

def Residual(dfretassets,dfretspx):
    rf_daily=(0.02/252)
    y=dfretassets-rf_daily
    x=dfretspx-rf_daily


    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    params = results.params
    alpha = params.loc["const"]             # scalaire si y est Series, Series si y est DataFrame
    beta  = params.loc["SPX"]               # idem

    residus = results.resid

    return alpha, beta, residus


def GetWhatToBuy(df,date,lookback,ticker):
    alpha,beta,residus=Residual(GetReturnDailyIndividual(df,date,21,ticker),GetReturnDailyIndividual(df,date,21,"SPX"))
    beta=beta.tolist()
    residus=residus.tolist()
    alpha=alpha.tolist()
    listofperf=[]
    zscore=np.sum(residus)/np.std(residus)
    listofperf.extend([alpha,beta,zscore])


    return listofperf