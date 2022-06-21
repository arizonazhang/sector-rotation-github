import pandas as pd
import numpy as np

for filename in ['dj_sectors', 'csi_sectors', 'hsci_sectors']:
    data = pd.read_csv(r'.\sector-rotation-data\{}.csv'.format(filename), index_col=0)
    data.index = pd.to_datetime(data.index)
    if filename == 'dj_sectors':
        data.columns = [col.split(" ")[0] for col in data.columns]
    if filename == "csi_sectors":
        ls = ["Energy", "Materials", "Industrials", "Consumer Discretionary", "Consumer Staples", "Health Care", "Financials", "Information Technology",
                        "Telecom", "Utilities"]
        data.columns = [item.lower() for item in ls]
    if filename == "hsci_sectors":
        data.columns = [col.split(" ")[0] for col in data.columns]
        data = data.drop(columns="HSCICO")
    ret_weekly = data[data.index.weekday == 4].pct_change().dropna(how='all',axis=0)
    ret_weekly = ret_weekly[ret_weekly.any(axis=1)]
    dt_range2 = pd.date_range(start = '2000-03-01', end = '2022-04-01', freq='BM')
    ret_monthly = data[data.index.isin(dt_range2)].pct_change().dropna(how='all', axis=0)
    ret_monthly = ret_monthly[ret_monthly.any(axis=1)]
    ret_weekly.to_csv(r'.\sector-rotation-data\{}_weekly.csv'.format(filename))
    ret_monthly.to_csv(r'.\sector-rotation-data\{}_monthly.csv'.format(filename))

