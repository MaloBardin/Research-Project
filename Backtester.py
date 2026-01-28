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


    StockQty= df.copy()
    StockQty.loc[:, :] = np.nan
    MoneyAtStart=1000000
    CurrentValue=MoneyAtStart
    StockQty.loc[start, "Money"] = MoneyAtStart
    StockQty.loc[start, "SPX"] = df.iloc[start, 1]
    StockQty.loc[start, "Date"] = df.iloc[start, 0]
    spaceindays=0

    for i in tqdm(range(start,df.shape[0])):
        StockQty.iloc[i,0]=df.iloc[i,0] #keep date
        StockQty.iloc[i,1]=df.iloc[i,1] #keep spx value

        #compute current value :
        if i==start:
            CurrentValue=MoneyAtStart
        else :
            CurrentValue=0
            for asset in df.columns[2:]:
                CurrentValue+=StockQty.iloc[i-1][asset]*df.iloc[i][asset]

        #rebalancing

        if i >= hist and spaceindays > 21 * hold:
            spaceindays=0
            BLWeight,RiskFreeAmount,historical_returns=BlackAndLittermanModel(str(df.iloc[i,0]),3,3*22,df_toBL,RfDf,confidence=confidence2,proportion=proportion2,tau=tau2,Lambda=Lambda2,historical_returns=historical_returns,modifiedlambda=modifiedlambda)
            for index in range(len(BLWeight)):
                StockQty.iloc[i,index+2]=(BLWeight.iloc[index,0]*CurrentValue)/df.iloc[i,index+2]
        else :
            StockQty.iloc[i,:]=StockQty.iloc[i-1,:]
            spaceindays+=1

        StockQty.iloc[i, -1] = CurrentValue

    StockQty = StockQty.iloc[start:].reset_index(drop=True)
    return StockQty
RfDf=GetRfDataframe(df)
final = Backtester(dfbacktest, hold=hold, hist=hist, proportion=proportion, df_toBL=df,RfDf=RfDf,confidence2=confidence,proportion2=proportion,tau2=tau,Lambda2=Lambda,start=181,modifiedlambda=0)

console.print("\n[green]✅ Backtest terminé avec succès ![/green]\n")