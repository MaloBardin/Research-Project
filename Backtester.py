from MomentumML import *


def GetProperDF():
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




def ComputingTheBackTest(df):
    from rich.console import Console
    from rich.panel import Panel
    from tqdm import tqdm

    console = Console()

    # BACK TESTER
    dfbacktest = df.copy()
    dfbacktest["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")


    df_length = dfbacktest.shape[1] - 2  # bcs of date and spx
    last_rebalance = dfbacktest.loc[0, "Date"]  # première date
    month_count = 0

    # 🎨 AFFICHAGE STYLÉ (sans prompts)
    hold = 3
    hist = 0
    proportion = 4
    Lambda = 3
    tau = 0.025
    confidence = 0.75

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
    datesHolder=[]
    def Backtester(df, hold, hist, proportion, df_toBL, RfDf, confidence2, proportion2, tau2, Lambda2, start,
                   modifiedlambda):
        # new dataframe for stock quantity
        StockQty = df.copy()
        historical_returns = []

        StockQty.loc[:, :] = 0
        # starting data
        MoneyAtStart = 10000000
        CurrentValue = MoneyAtStart
        interspaced = 0
        # first ligne
        StockQty.loc[start, "Money"] = MoneyAtStart
        StockQty.loc[start, "SPX"] = df.iloc[start, 1]
        StockQty.loc[start, "Date"] = df.iloc[start, 0]


        for i in tqdm(range(start, df.shape[0]), desc="Backtesting"):
            StockQty.iloc[i, 0] = df.iloc[i, 0]
            StockQty.iloc[i, 1] = df.iloc[i, 1]
            datesHolder.append(df.iloc[i, 0])

            # calculer value
            if i == start:
                CurrentValue = MoneyAtStart
            else:
                CurrentValue = 0
                for stocks in range(2, StockQty.shape[1] - 1):
                    CurrentValue += StockQty.iloc[i - 1, stocks] * df.iloc[i, stocks]

            StockQty.iloc[i, -1] = CurrentValue

            if interspaced > 21 * hold or i == start:
                interspaced = 0

                BLWeight,RiskFreeAmount,historical_returns=BlackAndLittermanModel(str(df.iloc[i,0]),21*hold,6*21,df_toBL,RfDf,confidence=confidence2,proportion=proportion2,tau=tau2,Lambda=Lambda2,historical_returns=historical_returns,modifiedlambda=modifiedlambda)

                for index in range(len(BLWeight)):
                    StockQty.iloc[i, index + 2] = (BLWeight.iloc[index, 0] * CurrentValue) / df.iloc[i, index + 2]

            else:
                interspaced += 1
                for stocks in range(2, StockQty.shape[1] - 1):
                    StockQty.iloc[i, stocks] = StockQty.iloc[i - 1, stocks]

        StockQty = StockQty.iloc[start:].reset_index(drop=True)
        return StockQty

    RfDf = GetRfDataframe(df)
    final = Backtester(dfbacktest, hold=hold, hist=hist, proportion=proportion, df_toBL=df, RfDf=RfDf,
                       confidence2=confidence, proportion2=proportion, tau2=tau, Lambda2=Lambda, start=180,
                       modifiedlambda=True)

    console.print("\n[green]✅ Backtest terminé avec succès ![/green]\n")
    return final

#%%
Visualisation(ComputingTheBackTest(GetProperDF()))