import numpy as np
import pandas as pd
from Grabber import getProperDf
from Residual import GetWhatToBuy
from Returns import aa
import tqdm


def CreateMLSets(df,proportion):
    split=int(df.shape[0]*proportion)
    trainingDf=df.iloc[:split,:]
    testingDf=df.iloc[split:,:]
    return trainingDf,testingDf


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
    workkk = pd.read_csv("MLset.csv")

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
        confusion_matrix,
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



import numpy as np
import pandas as pd
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

quick_hit_rate()









computeBBasicMomentum_vectorized_prices(getProperDf(),21)







