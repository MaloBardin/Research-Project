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

    changingLambda=True
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
import pandas as pd

money_norm = (final["Money"]/10000000*100) - 100
spx_norm = (final["SPX"]/final["SPX"].iloc[0]*100) - 100

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

# %%
import plotly.express as px
import pandas as pd

df_lambda = pd.DataFrame({"lambda": StackLambda})

fig = px.histogram(
    df_lambda,
    x="lambda",
    nbins=100,
    title="Lambda distribution",
    labels={"lambda": "λ"},
    opacity=0.8
)

fig.show()
# %%
df_lambda = pd.DataFrame({
    "Lambda": StackLambda
})

fig = px.line(
    df_lambda,
    y="Lambda",
    title="Temporal evolution of λ",
    labels={"Lambda": "λ"}
)

fig.show()

# %%
#TESTING PURPOSES

#Returns=GetReturn(df,"2020-05-11",lookback=180000)
#ReturnsSPX=GetReturnSPX(df,"2020-05-11",lookback=180)
#Sigma=GetSigma(df,"2020-05-11",lookback=10000)
#Weight=GetWeight(df,"2020-05-11")
#Lambda=GetLambda(df,"2024-01-11",timeofcalculation=3500,RfDf=RfDf)
#PMatrix,TempoQ=GetPMatrix(df,"2020-05-11",lookback=180,proportion=3)
#GetOmega(PMatrix,Sigma,0.2)

# %%
AnnualizedDf=final[["Date","SPX","Money"]]
AnnualizedDf['Date'] = pd.to_datetime(AnnualizedDf['Date'])
AnnualizedDf['Year'] = AnnualizedDf['Date'].dt.year



YearList=AnnualizedDf["Year"].unique()
SPXAnnualized=pd.DataFrame(columns=YearList)
StratAnnualized=pd.DataFrame(columns=YearList)



for year in YearList:
  compteurPerYear=0
  for i in AnnualizedDf.index:
    if AnnualizedDf.loc[i,"Year"]==year:
      if compteurPerYear==0:
        SPXAnnualized.loc[compteurPerYear,year]=AnnualizedDf.loc[i,"SPX"]
        StratAnnualized.loc[compteurPerYear,year]=AnnualizedDf.loc[i,"Money"]
      else :
        SPXAnnualized.loc[compteurPerYear,year]=AnnualizedDf.loc[i,"SPX"]/SPXAnnualized.loc[0,year]*100-100
        StratAnnualized.loc[compteurPerYear,year]=AnnualizedDf.loc[i,"Money"]/StratAnnualized.loc[0,year]*100-100
      compteurPerYear+=1

for year in YearList:
  SPXAnnualized.loc[0,year]=SPXAnnualized.loc[0,year]/SPXAnnualized.loc[0,year]*100-100
  StratAnnualized.loc[0,year]=StratAnnualized.loc[0,year]/StratAnnualized.loc[0,year]*100-100



SPXAvg=[]
StratAvg=[]
for i in SPXAnnualized.index:
  sumSPX=0
  sumStrat=0
  for year in SPXAnnualized.columns:
    sumSPX+=SPXAnnualized.loc[i,year]
    sumStrat+=StratAnnualized.loc[i,year]
  SPXAvg.append(sumSPX/len(YearList))
  StratAvg.append(sumStrat/len(YearList))

SPXAnnualized=SPXAnnualized.drop(columns=[2024,2002]) #too much nan
StratAnnualized=StratAnnualized.drop(columns=[2024,2002])

SPXAvg=[]
StratAVG=[]

for i in SPXAnnualized.index:
  sumSPX=0
  sumStrat=0
  for year in SPXAnnualized.columns:
    sumSPX+=SPXAnnualized.loc[i,year]
    sumStrat+=StratAnnualized.loc[i,year]
  SPXAvg.append(sumSPX/len(YearList))
  StratAVG.append(sumStrat/len(YearList))

dff = pd.DataFrame({"Index": (range(len(SPXAvg))),"Portfolio": StratAVG,"SPX": SPXAvg})


fig = px.line(dff, x="Index", y=["SPX","Portfolio"], color_discrete_map={"SPX": "red","Portfolio": "green"})
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
print("Portfolio Risk Measures:")
print(calculate_historical_var_es(final, 'Money', 0.99))
print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'Money', 0.03):.2f}")

print("\nSPX Risk Measures:")
print(calculate_historical_var_es(final, 'SPX', 0.99))
print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'SPX', 0.03):.2f}")
# %%
import pandas as pd
import numpy as np
import plotly.express as px

df2 = final[["Date", "SPX", "Money"]].copy()
df2["Portfolio"] = df2["Money"]
df2.drop(columns="Money", inplace=True)
df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)
df2.set_index("Date", inplace=True)

daily_returns = df2.pct_change()
daily_returns.dropna(inplace=True)

daily_returns["Volatility of the Benchmark"] = daily_returns['SPX'].rolling(window=252).std() * np.sqrt(252)
daily_returns["Volatility of the Portfolio"] = daily_returns['Portfolio'].rolling(window=252).std() * np.sqrt(252)
data_to_plot = daily_returns.dropna()


fig = px.line(data_to_plot,
              x=data_to_plot.index,
              y=["Volatility of the Benchmark", "Volatility of the Portfolio"],
              labels={"value": "Volatility of the Benchmark", "variable": "Actif", "Date": "Date"},
              title="annualized vol : SPX vs Portfolio")

fig.show()
# %%
#one value :


# --- Data prep ---
df2 = final[["Date", "SPX", "Money"]].copy()
df2["Portfolio"] = df2["Money"]
df2.drop(columns="Money", inplace=True)

df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)
df2.set_index("Date", inplace=True)

# --- Daily returns ---
daily_returns = df2.pct_change().dropna()

# --- Single-number annualized vol over the whole backtest (20y) ---
ann_vol_spx = daily_returns["SPX"].std(ddof=1) * np.sqrt(252)
ann_vol_port = daily_returns["Portfolio"].std(ddof=1) * np.sqrt(252)

print(f"SPX annualized vol (global, from daily returns): {ann_vol_spx:.6f}  ({ann_vol_spx*100:.2f}%)")
print(f"Portfolio annualized vol (global, from daily returns): {ann_vol_port:.6f}  ({ann_vol_port*100:.2f}%)")
# %%
#multiple runs on variable start date :

numberofdays = 21
all_results = []

for i in range(numberofdays):
    current_start = 181 + i
    results_df = Backtester(dfbacktest, hold=hold, hist=hist, proportion=proportion,
                            df_toBL=df, RfDf=RfDf, confidence2=confidence,
                            proportion2=proportion, tau2=tau, Lambda2=Lambda,
                            start=current_start,modifiedlambda=0)

    money_norm = (results_df["Money"] / 10_000_000 * 100) - 100
    temp_df = pd.DataFrame({f"Iter_{i}": money_norm.values})

    dateresults = results_df["Date"]
    temp_df.index = dateresults

    all_results.append(temp_df)


global_df = pd.concat(all_results, axis=1)
global_df_clean = global_df.dropna()
print(global_df_clean.head())



# %%
global_df_clean
# %%
#add spx to compare

dfcopyfinal = final[["Date", "SPX"]].copy()
dfcopyfinal.index=dfcopyfinal["Date"]
dfcopyfinal.drop(columns="Date", inplace=True)

global_df_clean = global_df_clean.merge(dfcopyfinal,left_index=True,right_index=True,how="left")
spx_norm = (global_df_clean["SPX"]/global_df_clean["SPX"].iloc[0]*100) - 100
global_df_clean["SPX"] = spx_norm
# %%
global_df_clean.tail(1)
# %%
import numpy as np
import plotly.graph_objects as go


def animate_dataframe_plotly(df,step=5,speed=40,spx_col="SPX"):

    df_anim = df.iloc[::step]
    x = df_anim.index

    final_values = df_anim.iloc[-1]
    best_col = final_values.drop(spx_col, errors="ignore").idxmax()

    traces = []
    for col in df_anim.columns:
        if col == spx_col:
            color = "red"
            width = 2
            alpha = 1
        elif col == best_col:
            color = "green"
            width = 2
            alpha = 1
        else:
            color = "rgba(250, 250, 250,1)"
            width = 1
            alpha = 0.4
        traces.append(go.Scatter(x=[],y=[],mode="lines",line=dict(color=color, width=width),opacity=alpha,name=col,showlegend=False))

    frames = []
    for t in range(1, len(df_anim)):
        frame_data = []
        for col in df_anim.columns:
            frame_data.append(go.Scatter(x=x[:t],y=df_anim[col].values[:t]))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    layout = go.Layout(title="Multiple Backtest Runs",xaxis=dict(title="Date"),yaxis=dict(title="Perf (%)"),plot_bgcolor="black",paper_bgcolor="black",font=dict(color="white"),updatemenus=[dict(type="buttons",showactive=False,buttons=[dict(label="▶ Play",method="animate",args=[None,dict( frame=dict(duration=speed, redraw=True),fromcurrent=True)]),dict(label="⏸ Pause",method="animate",args=[[None],dict(frame=dict(duration=0), mode="immediate")])])])

    fig = go.Figure(data=traces,layout=layout,frames=frames)

    fig.show()


#animate_dataframe_plotly(global_df_clean,step=5,speed=20)

# %%
import plotly.express as px

fig = px.line(
    global_df_clean,
    title=f"Comparaison des {numberofdays} itérations de Backtest (Rolling)",
    labels={
        "index": "Date",
        "value": "Performance Normalisée (%)",
        "variable": "Scénario"
    },
    template="plotly_dark"
)

fig.update_traces(line=dict(width=1.5))

if "SPX_Benchmark" in global_df_clean.columns:
    fig.update_traces(
        selector={"name": "SPX_Benchmark"},
        line=dict(width=4, color="white", dash="dot")
    )

fig.show()
# %%

# %%
#modified lambda evolution

#multiple runs on variable start date :

numberofruns = 5
all_results = []
factor=0.1
for i in range(numberofruns):
    results_df = Backtester(dfbacktest, hold=hold, hist=hist, proportion=proportion,
                            df_toBL=df, RfDf=RfDf, confidence2=confidence,
                            proportion2=proportion, tau2=tau, Lambda2=Lambda,
                            start=181,modifiedlambda=-i*factor)

    money_norm = (results_df["Money"] / 10_000_000 * 100) - 100
    temp_df = pd.DataFrame({f"Iter_{i}": money_norm.values})

    dateresults = results_df["Date"]
    temp_df.index = dateresults

    all_results.append(temp_df)


global_df = pd.concat(all_results, axis=1)
global_df_clean = global_df.dropna()
print(global_df_clean.head())
# %%
dfcopyfinal = final[["Date", "SPX"]].copy()
dfcopyfinal.index=dfcopyfinal["Date"]
dfcopyfinal.drop(columns="Date", inplace=True)

global_df_clean = global_df_clean.merge(dfcopyfinal,left_index=True,right_index=True,how="left")
spx_norm = (global_df_clean["SPX"]/global_df_clean["SPX"].iloc[0]*100) - 100
global_df_clean["SPX"] = spx_norm
# %%
import plotly.express as px

fig = px.line(
    global_df_clean,
    title=f"Comparaison des {numberofdays} itérations de Backtest (Rolling)",
    labels={
        "index": "Date",
        "value": "Performance Normalisée (%)",
        "variable": "Scénario"
    },
    template="plotly_dark"
)

fig.update_traces(line=dict(width=1.5))

if "SPX_Benchmark" in global_df_clean.columns:
    fig.update_traces(
        selector={"name": "SPX_Benchmark"},
        line=dict(width=4, color="white", dash="dot")
    )

fig.show()
# %%
#perfect views

def PerfectView(df,date,space=21): #match start and date and space with the holding time (1mo = 21)
    date=pd.to_datetime(date)
    copydf=df.copy()
    copydf=copydf[copydf["Date"] >= date]
    for i in range(0,copydf.shape[0]):
        if i%space!=0:
            copydf.iloc[i,1]=np.nan

    copydf.dropna(inplace=True)
    datelist=copydf["Date"]
    copydf.drop(columns="Date", inplace=True)
    copydf = (copydf/ copydf.shift(1)) ** (1/space)-1
    copydf.dropna(inplace=True)
    if copydf.shape[0]!=0:
        views= copydf.iloc[0, :]
        views=views.to_list()
        spxreturns=views[0]
        views.pop(0)
        listofreturns=[]
        for i in range(len(views)):
            listofreturns.append((views[i],i))
    else :
        print("LAST RUN SO NO PREDICTION")
        listofreturns=[(0,0),(0,1),(0,2),(0,3)]
        spxreturns=0
    listofreturns.sort(reverse=True)
    listofreturns=listofreturns[0:4:1] #MODIFIER LA PROPORTION AU BESOIN ICI
    return listofreturns,spxreturns




def ModifiedBlackPerfectViews(backtestStartDate, rebalancingFrequency, lookbackPeriod, df,RfDf,confidence=0.75,proportion=4,tau=0.025,Lambda=3,historical_returns=0,modifiedlambda=0):
    #implement the full backtest of the black and litterman model

    #---------
    #PARAMETERS
    #---------
    datetoremove=pd.to_datetime("2018-04-06") #add date to remove
    listofbanneddays=[]
    Sigma=get_shrunk_covariance(df,backtestStartDate,lookback=60) #using 720 days to have better sigma of 2 years
    Sigma=getSigmaModified(df,backtestStartDate,lookback=60,listofbanneddays=listofbanneddays) #using 720 days to have better sigma of 2 years

    PMatrix=np.zeros((proportion,24))
    Q=np.zeros((proportion,1))



    bestperf,perfmarket=PerfectView(df,backtestStartDate,21)


    for lineview in range(len(bestperf)):
        for i in range(24):
            PMatrix[lineview,i]=-1/24
        PMatrix[lineview,bestperf[lineview][1]]=1-1/24

    for views in range(len(bestperf)):
        Q[views,0]=(bestperf[views][0]-perfmarket)

    Omega=GetOmega(PMatrix, Sigma, c=confidence)
    rf=GetRiskFree(df,backtestStartDate,lookbackPeriod,RfDf)
    weights = GetWeight(df, backtestStartDate)
    weights = np.array(weights).reshape(-1, 1)

    changingLambda=True
    if changingLambda==True:
        if pd.to_datetime(backtestStartDate) < pd.to_datetime("2006-04-06"):
            Lambda=3
        else :
            Lambda=3#+0.1*GetLambda(df,backtestStartDate,timeofcalculation=504,RfDf=RfDf)
    else :
        Lambda=3

    uimplied = Lambda * (Sigma @ weights) + rf
    #BL formula
    #tau=OmegaLinked




    optimizedReturn=(np.linalg.inv(np.linalg.inv(tau*Sigma)+np.transpose(PMatrix)@np.linalg.inv(Omega)@PMatrix)) @ (np.linalg.inv(tau*Sigma)@uimplied+np.transpose(PMatrix)@np.linalg.inv(Omega)@Q)
    LambdaMarkowitz=3

    #MarkowitzAllocation
    WeightBL=np.linalg.inv(Sigma)@(optimizedReturn-rf)/LambdaMarkowitz
    WeightRF=1-np.sum(WeightBL)
    #if not np.isclose(float(np.sum(WeightBL)), 1.0, atol=1e-6):
        #print(np.sum(WeightBL))
        #raise ValueError("Weights do not sum to 1, please investigate.")

    return WeightBL,WeightRF,historical_returns


ModifiedBlackPerfectViews("2018-05-11", rebalancingFrequency=3, lookbackPeriod=180, df=df,RfDf=RfDf)



# %%
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

console = Console()

#BACK TESTER
dfbacktest=df.copy()
dfbacktest["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
dfbacktest["MonthIndex"] = dfbacktest["Date"].dt.to_period("M")

df_length = dfbacktest.shape[1] - 2  # bcs of date and spx
last_rebalance = dfbacktest.loc[0, "Date"]  # première date
month_count = 0

# 🎨 AFFICHAGE STYLÉ (sans prompts)
hold = 1
hist = 0
proportion = 4
Lambda=3
tau=0.025
confidence=0.75

console.print(Panel.fit(
    "[bold cyan]📊 PORTFOLIO BACKTESTER[/bold cyan]\n"
    "[dim]Black-Litterman Model[/dim]",
    border_style="cyan"
))

console.print(f"\n[yellow]⚙️  Configuration :[/yellow]")
console.print(f"   • Hold period: [cyan]{hold}[/cyan] mois")
console.print(f"   • Historique: [cyan]{hist}[/cyan] mois")
console.print(f"   • Proportion: [cyan]{proportion}[/cyan]")
console.print(f"   • Lambda: [cyan]{Lambda:.4f}[/cyan]")
console.print(f"   • Confiance: [cyan]{confidence}[/cyan]")
console.print(f"   • Taux: [cyan]{tau}[/cyan]\n")

console.print("\n[yellow]⏳ Lancement du backtest...[/yellow]\n")

def Backtester(df,hold, hist, proportion,df_toBL, RfDf,confidence2,proportion2,tau2,Lambda2,start,modifiedlambda):
    #new dataframe for stock quantity

    StockQty = df.copy()
    StockQty.drop(columns="MonthIndex", inplace=True)
    historical_returns=[]

    StockQty.loc[:, :] = 0
    #starting data
    MoneyAtStart = 10000000
    month_count=0
    CurrentValue=MoneyAtStart
    spaceindays=0
    #first ligne
    StockQty.loc[start, "Money"] = MoneyAtStart
    StockQty.loc[start, "SPX"] = df.iloc[start, 1]
    StockQty.loc[start, "Date"] = df.iloc[start, 0]
    RiskFreeAmount=0
    #start of the algorithm

    for i in tqdm(range(start,df.shape[0]), desc="Backtesting"):
      StockQty.iloc[i,0]=df.iloc[i,0]
      StockQty.iloc[i,1]=df.iloc[i,1]
      fees=0


      if df.loc[i, "Date"].month != df.loc[i-1, "Date"].month:
        month_count += 1


    # Si on atteint la période voulue
      if i>= hist and spaceindays>21*hold:
        #print(f"🔁 Rebalancement déclenché à la date : {df.loc[i, 'Date'].date()}")
        #print(str(df.iloc[i,0]))

        spaceindays=0

        BLWeight,RiskFreeAmount,historical_returns=ModifiedBlackPerfectViews(str(df.iloc[i,0]),3,3*22,df_toBL,RfDf,confidence=confidence2,proportion=proportion2,tau=tau2,Lambda=Lambda2,historical_returns=historical_returns,modifiedlambda=modifiedlambda)
        #print(len(BLWeight))
        for index in range(len(BLWeight)):
            StockQty.iloc[i,index+2]=(BLWeight.iloc[index,0]*CurrentValue)/df.iloc[i,index+2] #qty = weight*total value/price
      else :
        spaceindays+=1
        for stocks in range(2,StockQty.shape[1]-1):
          StockQty.iloc[i,stocks]=StockQty.iloc[i-1,stocks] #same qty


      #value of pf

      GainOrLoss = 0
      for stocks in range(2, StockQty.shape[1]-1):
        qty = StockQty.iloc[i, stocks]

        if qty != 0.0:
            price_now = df.iloc[i, stocks]
            price_prev = df.iloc[i-1, stocks]
            GainOrLoss += qty * (price_now - price_prev)

      daily_rate = GetRiskFree(df, str(df.iloc[i,0]), 1, RfDf)
      interest_gain = (CurrentValue * RiskFreeAmount) * daily_rate
      CurrentValue += GainOrLoss + interest_gain - fees
      StockQty.iloc[i,-1]=CurrentValue


    StockQty = StockQty.iloc[start:].reset_index(drop=True)
    return StockQty
RfDf=GetRfDataframe(df)
final = Backtester(dfbacktest, hold=hold, hist=hist, proportion=proportion, df_toBL=df,RfDf=RfDf,confidence2=confidence,proportion2=proportion,tau2=tau,Lambda2=Lambda,start=181,modifiedlambda=0)

console.print("\n[green]✅ Backtest terminé avec succès ![/green]\n")
# %%
import pandas as pd

money_norm = (final["Money"]/10000000*100) - 100
spx_norm = (final["SPX"]/final["SPX"].iloc[0]*100) - 100

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
# %%
import pandas as pd
import numpy as np
import plotly.express as px

df2 = final[["Date", "SPX", "Money"]].copy()
df2["Portfolio"] = df2["Money"]
df2.drop(columns="Money", inplace=True)
df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)
df2.set_index("Date", inplace=True)

daily_returns = df2.pct_change()
daily_returns.dropna(inplace=True)

daily_returns["annualizedVolSPX"] = daily_returns['SPX'].rolling(window=252).std() * np.sqrt(252)
daily_returns["annualizedVolPf"] = daily_returns['Portfolio'].rolling(window=252).std() * np.sqrt(252)
data_to_plot = daily_returns.dropna()

fig = px.line(data_to_plot,
              x=data_to_plot.index,
              y=["annualizedVolSPX", "annualizedVolPf"],
              labels={"value": "annualized vol", "variable": "Actif", "Date": "Date"},
              title="annualized vol : SPX vs Portfolio")

fig.show()
# %%
print("Portfolio Risk Measures:")
print(calculate_historical_var_es(final, 'Money', 0.99))
print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'Money', 0.03):.2f}")

print("\nSPX Risk Measures:")
print(calculate_historical_var_es(final, 'SPX', 0.99))
print(f"Sharpe Ratio: {calculate_sharpe_ratio(final, 'SPX', 0.03):.2f}")
# %%
#residual momentum
def GetFullReturnsForResidual(df):
    returns_df = df[["S5SFTW","S5PHRM","S5CPGS","S5ENRSX","S5FDBT","S5TECH","S5RETL","S5BANKX","S5HCES","S5DIVF","S5UTILX","S5MEDA","S5REAL","S5TELSX","S5MATRX","S5INSU","S5FDSR","S5HOUS","S5SSEQX","S5TRAN","S5HOTR","S5CODU","S5AUCO","S5COMS"]].copy()
    returns_df = np.log(returns_df/ returns_df.shift(1))
    returns_df.dropna(inplace=True)
    return returns_df
    #print(returns_df.std().mean()) #verification if std is around 1% daily
GetFullReturnsForResidual(df)

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
# %%
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

GetReturnMonthly(df,"2018-05-11",21*10)
#return a df of size (lookback, number of sectors) with log returns


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
# %%
import statsmodels.api as sm
def Residual(dfretassets,dfretspx):
    rfmonthly=(0.02/12)
    y=dfretassets-rfmonthly
    x=dfretspx-rfmonthly

    x_with_const = sm.add_constant(x)

    model = sm.OLS(y, x_with_const)
    results = model.fit()

    params = results.params
    alpha = params.loc["const"]             # scalaire si y est Series, Series si y est DataFrame
    beta  = params.loc["SPX"]               # idem

    residus = results.resid

    return alpha, beta, residus
# %%
alpha,beta,residus=Residual(GetReturnMonthly(df,"2018-05-11",21*36),GetReturnSPXMonthly(df,"2018-05-11",21*36))
fig = px.line(residus,
              x=residus.index,
              y=residus["S5BANKX"],
              title="quick draw of the evolution of the residual")

fig.show()
# %%

# %%

# %%

# %%

# %%
