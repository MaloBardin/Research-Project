import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fredapi import Fred
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("data.csv", sep=";", decimal=",")
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