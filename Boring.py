# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fredapi import Fred
import warnings

from plotly.data import stocks
from prompt_toolkit.filters import renderer_height_is_known

warnings.filterwarnings("ignore")

def GetMyDf():
    df = pd.read_csv("ML csv/data.csv", sep=";", decimal=",")
    df = df.rename(columns={
        "Column1": "Date",
        "Column2": "SPX",
        "Column3": "S5SFTW",
        "Column4": "S5PHRM",
        "Column5": "S5CPGS",
        "Column6": "S5ENRSX",
        "Column7": "S5FDBT",
        "Column8": "S5TECH",
        "Column9": "S5RETL",
        "Column10": "S5BANKX",
        "Column11": "S5HCES",
        "Column12": "S5DIVF",
        "Column13": "S5UTILX",
        "Column14": "S5MEDA",
        "Column15": "S5REAL",
        "Column16": "S5TELSX",
        "Column17": "S5MATRX",
        "Column18": "S5INSU",
        "Column19": "S5FDSR",
        "Column20": "S5HOUS",
        "Column21": "S5SSEQX",
        "Column22": "S5TRAN",
        "Column23": "S5HOTR",
        "Column24": "S5CODU",
        "Column25": "S5AUCO",
        "Column26": "S5COMS",
    })
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    return df
# %%
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


def GetSigma(df,date,lookback):

    returns_df=GetReturn(df,date,lookback=lookback)
    #covariance matric from returns_df
    sigma_windowed=returns_df.cov()

    return sigma_windowed

def get_shrunk_covariance(df,date,lookback):

    returns=GetReturn(df,date,lookback)

    lw = OAS()
    lw.fit(returns)
    shrunk_cov = lw.covariance_

    delta = lw.shrinkage_
    if isinstance(returns, pd.DataFrame):
        shrunk_cov = pd.DataFrame(
            shrunk_cov,
            index=returns.columns,
            columns=returns.columns
        )
    return shrunk_cov


def getSigmaModified(df,date,lookback,listofbanneddays,periodison=False):

    date=pd.to_datetime(date)
    if date not in df["Date"].values:#add breaker if windows not in df
        raise ValueError("Date not in dataframe")
    returns_df = df[["Date","S5SFTW","S5PHRM","S5CPGS","S5ENRSX","S5FDBT","S5TECH","S5RETL","S5BANKX","S5HCES","S5DIVF","S5UTILX","S5MEDA","S5REAL","S5TELSX","S5MATRX","S5INSU","S5FDSR","S5HOUS","S5SSEQX","S5TRAN","S5HOTR","S5CODU","S5AUCO","S5COMS"]].copy()

    date_index = returns_df.index[returns_df["Date"] == date][0]
    returns_df=returns_df[(returns_df.index<=date_index) & (returns_df.index>=date_index-lookback)]
    #days selection

    #banned days
    for banned_date in listofbanneddays:
        mask = returns_df["Date"] == banned_date
        if mask.any():
            print("got one :", banned_date)
            returns_df.loc[mask, :] = np.nan

    returns_df.drop(columns="Date",inplace=True)
    returns_df.dropna(inplace=True)

    #calculation of returns
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)

    #covaraicne matrix using shrinkage
    lw = OAS()
    lw.fit(returns_df)
    shrunk_cov = lw.covariance_

    delta = lw.shrinkage_
    if isinstance(returns_df, pd.DataFrame):
        shrunk_cov = pd.DataFrame(
            shrunk_cov,
            index=returns_df.columns,
            columns=returns_df.columns
        )
    return shrunk_cov

    #return a cov matrix of size (number of sectors, number of sectors) we use lookback to have different window sizes

def GetRfDataframe(df):
    fred = Fred(api_key="5c742a53d96bd3085e9199dcdb5af60b")
    riskfree = fred.get_series('DFF')
    # riskfree = fred.get_series('DTB1MO')

    riskfree = riskfree.to_frame(name='FedFunds')
    riskfree.index.name = "Date"
    riskfree = riskfree[riskfree.index >= "2002-01-01"]
    riskfree["FedFunds"]=riskfree["FedFunds"]/100
    list_days_open = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    list_days_full = pd.to_datetime(riskfree.index, dayfirst=True, errors="coerce")

    list_days_open=[pd.to_datetime(date) for date in list_days_open]
    list_days_full=[pd.to_datetime(date) for date in list_days_full]


    list_days_open_pondered=[]
    riskfree_list=[]
    count_list=[]
    timestamp=0
    while timestamp < len(list_days_full)-1:

      if list_days_full[timestamp+1] in list_days_open:
            list_days_open_pondered.append(list_days_full[timestamp])
            riskfree_list.append(riskfree["FedFunds"].loc[list_days_full[timestamp]])
            count_list.append(1)
            timestamp += 1

      else:
          count = 0
          timestampbis = timestamp
          while (timestamp + 1 < len(list_days_full)) and (list_days_full[timestamp + 1] not in list_days_open):
              timestamp += 1
              count += 1

          list_days_open_pondered.append(list_days_full[timestampbis])  # jour de départ
          riskfree_list.append(riskfree["FedFunds"].loc[list_days_full[timestampbis]])
          count_list.append(count+1)
          timestamp += 1

    RfDf=pd.DataFrame({"Date":list_days_open_pondered,"Rf":riskfree_list,"Count":count_list})
    RfDf=RfDf.set_index("Date")
    return RfDf



def GetRiskFree(df,date,lookback,RfDf):
    positionOfStartDate=df.index[df["Date"]==pd.to_datetime(date)][0]-lookback
    #print(positionOfStartDate)
    startDate=pd.to_datetime(df.iloc[positionOfStartDate,0])
    endDate=pd.to_datetime(date)
    RfDf=RfDf[(RfDf.index >= startDate) & (RfDf.index <= endDate )].copy()
    CumulativeRf=[]

    for i in range(len(RfDf)):
      if i==0:
        CumulativeRf.append(pow((1+RfDf["Rf"].iloc[i]),(RfDf["Count"].iloc[i]/360)))
      else:
        CumulativeRf.append(pow((1+RfDf["Rf"].iloc[i]),(RfDf["Count"].iloc[i]/360))*CumulativeRf[i-1])

    RfDf["CumulativeRf"]=CumulativeRf
    RfDf["CumulativeRf"]= RfDf["CumulativeRf"]-1

    return RfDf["CumulativeRf"].iloc[-1]

#compute risk free dataframe using API from FRED and get the cumulative risk free rate between two dates in a df
# %%
def GetWeight(df,date):
    #for the moment we will use the equal weight
    weight_vector=np.zeros((24,1))
    for i in range(0,24):
        weight_vector[i]=1/24

    return weight_vector


#usual weighting scheme, for the moment equal weight
# %%


holderlambda=[]
datesss=[]
def GetLambda(df,date,timeofcalculation,RfDf):
    returns=GetReturn(df,date,timeofcalculation) #daily returns
    weight_vector=GetWeight(df=0,date=0)

    mean_return=np.mean(np.dot(returns,weight_vector))
    mean_annual=(1+mean_return)**252-1 #annualized mean return


    rf_temps=GetRiskFree(df,date,timeofcalculation,RfDf)
    rf_annual=(1+rf_temps)**(252/timeofcalculation)-1 #annualized risk free rate


    Sigma=get_shrunk_covariance(df,date,timeofcalculation)
    Sigma_annual=252*Sigma #annualized covariance matrix
    var = float((weight_vector.T @ Sigma_annual.values @ weight_vector).item())
    lambda_value=(mean_annual - rf_annual)/var


    excess = mean_annual - rf_annual
    sigma2 = var
    sigma  = np.sqrt(var)
    lam    = excess / sigma2
    sharpe = excess / sigma
    #print("Excess:", excess, " Var:", sigma2, " Vol:", sigma, " λ:", lam, " Sharpe:", sharpe)
    datesss.append(date)
    holderlambda.append(lambda_value)
    return lambda_value

#compute the lambda value using the mean return, risk free rate and variance of the portfolio

# %%
#add the Q matrix calculation

def QMatrixCalculation(df,date,lookback,proportion,performerc_daily,dailyperf_market,historical_returns):
    Q=np.zeros((proportion,1))
    factor=1
    for i in range(proportion):
        Q[i,0]=(performerc_daily[i][0]-dailyperf_market)/2


    return Q,historical_returns


# %%
def GetPMatrix(df,date, lookback,proportion=3,historical_returns=0):

    AssetColumns=["S5SFTW","S5PHRM","S5CPGS","S5ENRSX","S5FDBT","S5TECH","S5RETL","S5BANKX","S5HCES","S5DIVF","S5UTILX","S5MEDA","S5REAL","S5TELSX","S5MATRX","S5INSU","S5FDSR","S5HOUS","S5SSEQX","S5TRAN","S5HOTR","S5CODU","S5AUCO","S5COMS"]
    bestperformer = []
    performerc = []
    performerc_daily=[]
    returnBestPerformer=[]
    endDateIndex=df.index[df["Date"]==pd.to_datetime(date)][0]
    startDateIndex=df.index[df["Date"]==pd.to_datetime(date)][0]-lookback

    for i in range(2, df.shape[1]):  #loop through asset columns
        performerc.append((((float(df.iloc[endDateIndex, i]) / float(df.iloc[startDateIndex, i]) - 1) * 100), i - 2,df.columns[i])) #pos of best stock in a tuple
        # with its return
        performerc_daily.append(((float(df.iloc[endDateIndex, i]) / float(df.iloc[startDateIndex, i])) ** (1/lookback) - 1, i - 2,df.columns[i])) #daily version


    performerc.sort(reverse=True)
    performerc_daily.sort(reverse=True)
    #print(performerc)
    perfMarket= (float(df.iloc[endDateIndex, 1]) / float(df.iloc[startDateIndex, 1]) - 1) * 100
    dailyperf_market = (float(df.iloc[endDateIndex, 1]) / float(df.iloc[startDateIndex, 1])) ** (1/lookback) - 1

    for i in range(proportion):
        bestperformer.append(performerc_daily[i][1])
        returnBestPerformer.append(performerc_daily[i][0])


    P=np.zeros((proportion,24))
    Q=np.zeros((proportion,1))
    for lineview in range(proportion):
        for i in range(len(AssetColumns)):
            P[lineview,i]=-1/len(AssetColumns)
        P[lineview,bestperformer[lineview]]=1-1/len(AssetColumns)
        sum=0
        for i in range(len(AssetColumns)):
            sum+=P[lineview,i]
    Q,historical_returns=QMatrixCalculation(df,date,lookback,proportion,performerc_daily,dailyperf_market,historical_returns)


    return P, Q, historical_returns
# %%
def GetOmega(PMatrix, Sigma, c=0.99):
    #Omega is the uncertainty of the views

    factorC=(1/c-1)
    Omega=factorC*PMatrix@Sigma@np.transpose(PMatrix)

    return Omega

# %%
def LinkOmegaTau(Omega, Sigma, P):
    #Link omega to tau
    constant=36

    multiple= np.trace(np.transpose(P) @ np.linalg.inv(Omega) @ P) * constant
    numerator= np.trace(np.linalg.inv(Sigma*252))
    result= numerator / multiple
    return result


# %%
def LinkOmegaTau2(Omega, Sigma,P,tau):
    #Link omega to tau
    numerator= np.trace(np.linalg.inv(Sigma*tau))
    denominator= np.trace((np.transpose(P)@np.linalg.inv(Omega)@P))
    result=numerator/denominator
    return result


# %%
def GetPandQUsingRedisual(df,lookbackdate,lookback,proportion,historical_returns):
    #36 months data
    alpha,beta,residus=Residual(GetReturnMonthly(df,lookbackdate,lookback),GetReturnSPXMonthly(df,lookbackdate,lookback))
    beta=beta.tolist()
    alpha=alpha.tolist()
    # get the n last residual for the score computation
    backforscore=4 #in months
    listofperf=[]
    for i in range(residus.shape[1]):
        extract=[residus.iloc[residus.shape[0]-j-1,i] for j in range(0,backforscore)]
        zscore=np.sum(extract)/np.std(extract)
        listofperf.append((alpha[i],beta[i],zscore,i))

    listofperf=sorted(listofperf,key=lambda x:x[2],reverse=True)

    AssetColumns=["S5SFTW","S5PHRM","S5CPGS","S5ENRSX","S5FDBT","S5TECH","S5RETL","S5BANKX","S5HCES","S5DIVF","S5UTILX","S5MEDA","S5REAL","S5TELSX","S5MATRX","S5INSU","S5FDSR","S5HOUS","S5SSEQX","S5TRAN","S5HOTR","S5CODU","S5AUCO","S5COMS"]


    P=np.zeros((proportion,24))
    for lineview in range(proportion):
        for i in range(len(AssetColumns)):
            P[lineview,i]=-1/len(AssetColumns)
        P[lineview,listofperf[lineview][3]]=1-1/len(AssetColumns)



    #Q prediction !

    #compute the monthly returns
    dailyret_market=GetFullReturnsForResidualSPXDaily(df,lookbackdate,lookback)
    meandaily=np.mean(dailyret_market)
    meanmonthly= (1+meandaily)**21 - 1

    Q=np.zeros((proportion,1))
    for stuff in range(proportion):
        prediction=listofperf[stuff][0]+listofperf[stuff][1]*meanmonthly
        excess=prediction-meanmonthly
        excessdaily=(1+excess)**(1/21)-1
        Q[stuff,0]=excessdaily



    historical_returns=[]
    return P,Q,historical_returns

#GetPandQUsingRedisual(df,"2018-05-11",21*36)


# %%
StackLambda=[]
datesss2=[]
def BlackAndLittermanModel(backtestStartDate, rebalancingFrequency, lookbackPeriod, df,RfDf,confidence=0.75,proportion=4,tau=0.025,Lambda=3,historical_returns=0,modifiedlambda=0):
    #implement the full backtest of the black and litterman model


    Sigma=get_shrunk_covariance(df,backtestStartDate,lookback=60) #using 720 days to have better sigma of 2 years
    #Sigma=getSigmaModified(df,backtestStartDate,lookback=60,listofbanneddays=listofbanneddays) #using 720 days to have better sigma of 2 years


    PMatrix,Q,historical_returns= GetPMatrix(df,backtestStartDate, lookback=lookbackPeriod,proportion=proportion,historical_returns=historical_returns)
    Omega=GetOmega(PMatrix, Sigma, c=confidence)
    rf=GetRiskFree(df,backtestStartDate,lookbackPeriod,RfDf)
    weights = GetWeight(df, backtestStartDate)
    weights = np.array(weights).reshape(-1, 1)

    changingLambda=False
    if changingLambda==True:
        Lambda=3+0.05*GetLambda(df,backtestStartDate,timeofcalculation=60,RfDf=RfDf)
        if Lambda >=5:
            Lambda=5
        elif Lambda <=1:
            Lambda =1

    else :
        Lambda=3

    uimplied = Lambda * (Sigma @ weights) + rf
    #BL formula
    #tau=OmegaLinked
    StackLambda.append(Lambda)
    datesss2.append(backtestStartDate)
    optimizedReturn=(np.linalg.inv(np.linalg.inv(tau*Sigma)+np.transpose(PMatrix)@np.linalg.inv(Omega)@PMatrix)) @ (np.linalg.inv(tau*Sigma)@uimplied+np.transpose(PMatrix)@np.linalg.inv(Omega)@Q)
    LambdaMarkowitz=Lambda


    #MarkowitzAllocation
    WeightBL=np.linalg.inv(Sigma)@(optimizedReturn-rf)/LambdaMarkowitz
    WeightRF=1-np.sum(WeightBL)

    return WeightBL,WeightRF,historical_returns

# %%




















# %%
#TESTING PURPOSES

#Returns=GetReturn(df,"2020-05-11",lookback=180000)
#ReturnsSPX=GetReturnSPX(df,"2020-05-11",lookback=180)
#Sigma=GetSigma(df,"2020-05-11",lookback=10000)
#Weight=GetWeight(df,"2020-05-11")
#Lambda=GetLambda(df,"2024-01-11",timeofcalculation=3500,RfDf=RfDf)
#PMatrix,TempoQ=GetPMatrix(df,"2020-05-11",lookback=180,proportion=3)
#GetOmega(PMatrix,Sigma,0.2)
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime
from sklearn.covariance import LedoitWolf,OAS


# %%
def Annualization(final):
    AnnualizedDf = final[["Date", "SPX", "Money"]]
    AnnualizedDf['Date'] = pd.to_datetime(AnnualizedDf['Date'])
    AnnualizedDf['Year'] = AnnualizedDf['Date'].dt.year

    YearList = AnnualizedDf["Year"].unique()
    SPXAnnualized = pd.DataFrame(columns=YearList)
    StratAnnualized = pd.DataFrame(columns=YearList)

    for year in YearList:
        compteurPerYear = 0
        for i in AnnualizedDf.index:
            if AnnualizedDf.loc[i, "Year"] == year:
                if compteurPerYear == 0:
                    SPXAnnualized.loc[compteurPerYear, year] = AnnualizedDf.loc[i, "SPX"]
                    StratAnnualized.loc[compteurPerYear, year] = AnnualizedDf.loc[i, "Money"]
                else:
                    SPXAnnualized.loc[compteurPerYear, year] = AnnualizedDf.loc[i, "SPX"] / SPXAnnualized.loc[
                        0, year] * 100 - 100
                    StratAnnualized.loc[compteurPerYear, year] = AnnualizedDf.loc[i, "Money"] / StratAnnualized.loc[
                        0, year] * 100 - 100
                compteurPerYear += 1

    for year in YearList:
        SPXAnnualized.loc[0, year] = SPXAnnualized.loc[0, year] / SPXAnnualized.loc[0, year] * 100 - 100
        StratAnnualized.loc[0, year] = StratAnnualized.loc[0, year] / StratAnnualized.loc[0, year] * 100 - 100

    SPXAvg = []
    StratAvg = []
    for i in SPXAnnualized.index:
        sumSPX = 0
        sumStrat = 0
        for year in SPXAnnualized.columns:
            sumSPX += SPXAnnualized.loc[i, year]
            sumStrat += StratAnnualized.loc[i, year]
        SPXAvg.append(sumSPX / len(YearList))
        StratAvg.append(sumStrat / len(YearList))

    SPXAnnualized = SPXAnnualized.drop(columns=[2024, 2002])  # too much nan
    StratAnnualized = StratAnnualized.drop(columns=[2024, 2002])

    SPXAvg = []
    StratAVG = []

    for i in SPXAnnualized.index:
        sumSPX = 0
        sumStrat = 0
        for year in SPXAnnualized.columns:
            sumSPX += SPXAnnualized.loc[i, year]
            sumStrat += StratAnnualized.loc[i, year]
        SPXAvg.append(sumSPX / len(YearList))
        StratAVG.append(sumStrat / len(YearList))

    dff = pd.DataFrame({"Index": (range(len(SPXAvg))), "Portfolio": StratAVG, "SPX": SPXAvg})

    fig = px.line(dff, x="Index", y=["SPX", "Portfolio"], color_discrete_map={"SPX": "red", "Portfolio": "green"})
    fig.show()


# %%
#risk mesures
# %%
def calculate_historical_var_es(df, col_name='Money', confidence_level=0.95):


    returns = df[col_name].pct_change().dropna()

    cutoff = 1 - confidence_level

    var_value = returns.quantile(cutoff)

    worst_returns = returns[returns <= var_value]
    es_value = worst_returns.mean()

    return {
        "confidence_level": confidence_level,
        "VaR": -var_value,
        "ES": -es_value,
        "count_returns": len(returns),
        "count_breaches": len(worst_returns)
    }



def calculate_sharpe_ratio(df, col_name='close', risk_free_rate_annual=0.04):


    returns = (df[col_name] - df[col_name].shift(1)) / df[col_name].shift(1)
    returns = returns.dropna()
    rf_daily = risk_free_rate_annual / 252
    excess_returns = returns - rf_daily

    sharpe_daily = excess_returns.mean() / excess_returns.std()

    sharpe_annualized = sharpe_daily * np.sqrt(252)

    return sharpe_annualized

# %%

def displayrm(final):
    print("Portfolio Risk Measures:")
    print(calculate_historical_var_es(final, 'Money', 0.99))
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'Money', 0.03):.2f}")

    print("\nSPX Risk Measures:")
    print(calculate_historical_var_es(final, 'SPX', 0.99))
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'SPX', 0.03):.2f}")

def Visualisation(final):
    money_norm = (final["Money"] / 10000000 * 100) - 100
    spx_norm = (final["SPX"] / final["SPX"].iloc[0] * 100) - 100

    df_plot = pd.DataFrame({
        "Date": final["Date"],
        "Portfolio": money_norm,
        "SPX": spx_norm
    }).melt(id_vars="Date", var_name="Série", value_name="Évolution en %")

    fix = px.line(
        df_plot,
        x="Date",
        y="Évolution en %",
        color="Série",
        color_discrete_map={"SPX": "red", "Portfolio": "green"},
        title="Comparaison des évolutions en %"
    )

    fix.update_layout(hovermode="x unified")
    fix.show()
