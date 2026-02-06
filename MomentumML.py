from Residual import GetWhatToBuy
from Returns import aa


def ComputeTraining(trainingDf,forwardwindow):
    X_train=0
    Y_train=[]


    #X train part
    cols = ["Date","Ticker","ret_5d","ret_10d","ret_21d","ret_63d","zscore","y"]
    working_df=pd.DataFrame(columns=cols)
    for date in tqdm.tqdm(range(forwardwindow,trainingDf.shape[0]-forwardwindow)):
        ranking=[]
        currentDate = trainingDf.iloc[date]["Date"]

        for asset in trainingDf.columns[2:]:
            working_row=[]
            currentTicker=asset
            ret_5d= aa(trainingDf,currentDate,5,asset)
            ret_10d= aa(trainingDf,currentDate,10,asset)
            ret_21d= aa(trainingDf,currentDate,21,asset)
            ret_63d= aa(trainingDf,currentDate,63,asset)
            tempo=GetWhatToBuy(trainingDf,currentDate,12,asset)

            alpha=tempo[0]
            beta=tempo[1]
            zscore=tempo[2]


            retin21d=aa(trainingDf,trainingDf.iloc[date+forwardwindow]["Date"],forwardwindow,asset)
            ranking.append((retin21d,asset))

            working_row.append({
                "Date": currentDate,
                "Ticker": currentTicker,
                "ret_5d": ret_5d,
                "ret_10d": ret_10d,
                "ret_21d": ret_21d,
                "ret_63d": ret_63d,
                "alpha": alpha,
                "beta": beta,
                "zscore": zscore,
                "y": np.nan
            })

            working_df.loc[len(working_df)] = working_row[0]
        #ranking for y
        ranking.sort(key=lambda x:x[0], reverse=True)
        top4_assets = {asset for _, asset in ranking[:4]}
        mask_date = working_df["Date"] == currentDate
        mask_top4 = working_df["Ticker"].isin(top4_assets)

        working_df.loc[mask_date, "y"] = 0
        working_df.loc[mask_date & mask_top4, "y"] = 1

    X_train=working_df.drop(columns=["y"])
    Y_train=working_df["y"]
    X_train=X_train[:80000]
    Y_train=Y_train[:80000]
    X_test=X_train[80000:]
    Y_test=Y_train[80000:]
    working_df.to_csv("MLset.csv",index=False)
    return X_train,Y_train,X_test,Y_test

def ModelComputation():
    workkk = pd.read_csv("ML csv/MLset.csv")

    X = workkk.drop(columns=["y"])
    y = workkk["y"]

    split = 80000

    X_train = X.iloc[:split]
    Y_train = y.iloc[:split]

    X_test  = X.iloc[split:]
    Y_test  = y.iloc[split:]

    print("Train:", X_train.shape, Y_train.shape)
    print("Test :", X_test.shape, Y_test.shape)
    features_to_drop = ["Date"]
    # TRAIN
    Xtr = X_train.drop(columns=features_to_drop)
    Xtr = pd.get_dummies(Xtr, columns=["Ticker"])

    # TEST
    Xte = X_test.drop(columns=features_to_drop)
    Xte = pd.get_dummies(Xte, columns=["Ticker"])

    # Alignement des colonnes (OBLIGATOIRE)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    # Targets
    ytr = Y_train.astype(int)
    yte = Y_test.astype(int)

    print(Xtr.shape, Xte.shape)
    print(Xtr.columns.equals(Xte.columns))


    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xte, yte)],
        verbose=True
    )

    y_pred_proba = model.predict_proba(Xte)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)


    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report
    )

    print("Accuracy :", accuracy_score(yte, y_pred))
    print("Precision:", precision_score(yte, y_pred))
    print("Recall   :", recall_score(yte, y_pred))
    print("F1-score :", f1_score(yte, y_pred))
    print("ROC AUC  :", roc_auc_score(yte, y_pred_proba))

    print("\nClassification report:\n")
    print(classification_report(yte, y_pred))

    X_test_eval = X_test.copy()
    X_test_eval["proba"] = y_pred_proba
    X_test_eval["y_true"] = yte.values

    def precision_at_4(group):
        top4 = group.sort_values("proba", ascending=False).head(4)
        return top4["y_true"].mean()

    precision_top4 = (
        X_test_eval
        .groupby("Date")
        .apply(precision_at_4)
        .mean()
    )

    print("Precision@4:", precision_top4)

import tqdm

def computeBBasicMomentum_vectorized_prices(trainingDf, forwardwindow, start=181, out_csv="verifff.csv"):
    assets = list(trainingDf.columns[2:])
    dates = trainingDf["Date"].to_numpy()

    prices = trainingDf[assets].astype(float)  # ne modifie pas trainingDf
    ret_63 = prices.pct_change(63)                          # trailing 63j
    fwd_ret = prices.shift(-forwardwindow) / prices - 1     # forward fw

    idx = np.arange(start, trainingDf.shape[0] - forwardwindow)
    pre = ret_63.iloc[idx].to_numpy()
    post = fwd_ret.iloc[idx].to_numpy()

    n_dates = len(idx)
    n_assets = len(assets)

    # indices des top4 sans trier toute la ligne
    top4_pre = np.argpartition(-pre, 4, axis=1)[:, :4]
    top4_post = np.argpartition(-post, 4, axis=1)[:, :4]


    position = np.zeros((n_dates, n_assets), dtype=np.int8)
    y = np.zeros((n_dates, n_assets), dtype=np.int8)
    row = np.arange(n_dates)[:, None]
    position[row, top4_pre] = 1
    y[row, top4_post] = 1

    out = pd.DataFrame({
        "Date": np.repeat(dates[idx], n_assets),
        "Ticker": np.tile(np.array(assets), n_dates),
        "ret_63d": pre.reshape(-1),
        "position": position.reshape(-1),
        "y": y.reshape(-1),
        "retin21d": post.reshape(-1),
    })

    out.to_csv("eee", index=False)
    return out

def quick_hit_rate(path="verifff.csv"):
    df = pd.read_csv(path)

    # sécurise les types
    pos = df["position"].fillna(0).astype(int)
    y   = df["y"].fillna(0).astype(int)

    n_pred_1 = (pos == 1).sum()
    n_hit = ((pos == 1) & (y == 1)).sum()

    hit_rate = n_hit / n_pred_1 if n_pred_1 > 0 else 0.0

    print(f"Pred=1: {n_pred_1} | Hits: {n_hit} | Hit rate: {hit_rate:.4%}")
    return hit_rate



from Boring import *



def compute_y(trainingDf, date, forward_window, top_k=4):
    future_date = trainingDf.iloc[
        trainingDf.index[trainingDf["Date"] == date][0] + forward_window
    ]["Date"]

    ranking = []
    for asset in trainingDf.columns[2:]:
        ret_fwd = aa(trainingDf, future_date, forward_window, asset)
        ranking.append((ret_fwd, asset))

    ranking.sort(reverse=True)
    top_assets = {a for _, a in ranking[:top_k]}

    y = []
    for asset in trainingDf.columns[2:]:
        y.append(1 if asset in top_assets else 0)

    return np.array(y)

def computeFeaturesatDate(trainingDf, currentDate):
    cols = ["Date", "Ticker", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "zscore"]
    working_df = pd.DataFrame(columns=cols)

    for asset in trainingDf.columns[2:]:
        working_row = []
        currentTicker = asset
        ret_5d = aa(trainingDf, currentDate, 5, asset)
        ret_10d = aa(trainingDf, currentDate, 10, asset)
        ret_21d = aa(trainingDf, currentDate, 21, asset)
        ret_63d = aa(trainingDf, currentDate, 63, asset)
        tempo = GetWhatToBuy(trainingDf, currentDate, 12, asset)

        alpha = tempo[0]
        beta = tempo[1]
        zscore = tempo[2]

        working_row.append({
            "Date": currentDate,
            "Ticker": currentTicker,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ret_21d": ret_21d,
            "ret_63d": ret_63d,
            "alpha": alpha,
            "beta": beta,
            "zscore": zscore,
        })

        working_df.loc[len(working_df)] = working_row[0]

    return working_df

def Coldtraining(trainingDf, burnin_dates, forward_window):
    first = True

    for date in burnin_dates:
        df_feat = computeFeaturesatDate(trainingDf, date)
        X = df_feat.drop(columns=["Date", "Ticker"])
        y = compute_y(trainingDf, date, forward_window)

        if first:
            model.partial_fit(X, y, classes=[0, 1])
            first = False
        else:
            model.partial_fit(X, y)





from sklearn.linear_model import SGDClassifier
model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=42
    )
buffer_X = {}

def getviewsfromml(trainingDf, currentDate, forward_window):
    """
    1) prédire à currentDate
    2) retourner les probas (pour BL)
    3) entraîner le modèle si le label passé est dispo
    """

    df_feat = computeFeaturesatDate(trainingDf, currentDate)
    X = df_feat.drop(columns=["Date", "Ticker"])

    proba = model.predict_proba(X)[:, 1]
    buffer_X[currentDate] = X.copy()

    idx = trainingDf.index[trainingDf["Date"] == currentDate][0]
    past_idx = idx - forward_window

    if past_idx >= 0:
        past_date = trainingDf.iloc[past_idx]["Date"]

        if past_date in buffer_X:
            X_past = buffer_X[past_date]
            y_past = compute_y(trainingDf, past_date, forward_window)

            model.partial_fit(X_past, y_past)

    # ---- SORTIE POUR BL ----
    out = df_feat[["Ticker"]].copy()
    out["proba_ml"] = proba

    return out



def BlackAndLittermanML(backtestStartDate, rebalancingFrequency, lookbackPeriod, df, RfDf, confidence, proportion, tau, Lambda,):

    #COMPUTATION OF SIGMA
    Sigma = get_shrunk_covariance(df, backtestStartDate, lookback=60)  # using 720 days to have better sigma of 2 years
    # Sigma=getSigmaModified(df,backtestStartDate,lookback=60,listofbanneddays=listofbanneddays) #using 720 days to have better sigma of 2 years


    #Compute the views from the ML model
    viewtandproba=getviewsfromml(df, backtestStartDate, rebalancingFrequency)






    """
    Omega=GetOmega(PMatrix, Sigma, c=confidence)
    rf=GetRiskFree(df,backtestStartDate,lookbackPeriod,RfDf)
    weights = GetWeight(df, backtestStartDate)
    weights = np.array(weights).reshape(-1, 1)

    Lambda=3
    LambdaMarkowitz=3

    uimplied = Lambda * (Sigma @ weights) + rf
    #BL formula
    #tau=OmegaLinked
    StackLambda.append(Lambda)
    datesss2.append(backtestStartDate)
    optimizedReturn=(np.linalg.inv(np.linalg.inv(tau*Sigma)+np.transpose(PMatrix)@np.linalg.inv(Omega)@PMatrix)) @ (np.linalg.inv(tau*Sigma)@uimplied+np.transpose(PMatrix)@np.linalg.inv(Omega)@Q)


    #MarkowitzAllocation
    WeightBL=np.linalg.inv(Sigma)@(optimizedReturn-rf)/LambdaMarkowitz
    WeightRF=1-np.sum(WeightBL)
    
    return WeightBL,WeightRf"""

    return 0,0












# Get BL inputs





