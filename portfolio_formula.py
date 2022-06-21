from utils import *
from scipy import linalg
import numpy as np
import pandas as pd

def cal_weights(cal_mean_var, gamma_reg, addErr):
    weights = {}
    rsqs = {}
    weeks = pd.date_range(start=df_sectors.index[0], end=df_sectors.index[-1], freq="W-FRI" if useWeekly else "BM", closed = "left")
    reb = len(weeks[weeks < df_sectors.index[window-1]])
    for t in range(T-window):
        if reb < len(weeks) and df_sectors.index[t + window-1] <= weeks[reb] and df_sectors.index[t+window] > weeks[reb]:
            idx_range = (df_sectors.index <= weeks[reb])&(df_sectors.index >= weeks[reb-window+1])
            Evec, Vmat = cal_mean_var(df_sectors[idx_range], p=p, gamma_reg=gamma_reg, useReg=True, useErr=addErr)
            w = np.dot(linalg.inv(Vmat), Evec)
            w = w/np.sum(w)
            weights[df_sectors.index[t + window].strftime("%Y-%m-%d")] = list(w)
            reb += 4
        else:
            new_weights = weights[df_sectors.index[t + window - 1].strftime("%Y-%m-%d")]*(1+df_sectors.iloc[t+window-1, :p])
            weights[df_sectors.index[t + window].strftime("%Y-%m-%d")] = new_weights/np.sum(new_weights)
    # for t in range(T - window):
    #     if t % rebal_interval == 0:
    #         Evec, Vmat = cal_mean_var(df_sectors.iloc[t:t + window], p=p, gamma_reg=gamma_reg, useReg=True, useErr=addErr)
    #         w = np.dot(linalg.inv(Vmat), Evec)
    #         w = w/np.sum(w)
    #         weights[df_sectors.index[t + window]] = list(w)
    #         # rsqs[df_sectors.index[t + window]] = rsq
    #     else:
    #         weights[df_sectors.index[t + window]] = weights[df_sectors.index[t + window-1]]
    weights = pd.DataFrame.from_dict(weights, orient="index", columns=df_sectors.columns[:p])
    # rsq = pd.DataFrame.from_dict(rsqs, orient="index", columns=df_sectors.columns[:p])
    return weights

if __name__ == "__main__":
    # parameters

    # useWeekly = False
    # sector_indices = ("hsci", "dj", "csi")
    # markets = ("hk", "us", "cn")
    addErr = True
    useWeekly = True
    sector_indices = ("hsci", "csi")
    markets = ("hk300", "cn800")

    window = 260 if useWeekly else 60
    # run results
    for est_input in [est_input_hist, est_input_capm, est_input_factor]:
        for i in range(len(markets)):
            ls_weight = []
            ls_return = []
            mkt = markets[i]
            index = sector_indices[i]
            df_sectors, p, T = load_data(index, mkt, useWeekly) # p stands for no. of sectors
            for gamma_reg in [0.001, 0.005, 0.01]:
                _weights = cal_weights(est_input, gamma_reg, addErr)
                for b_weight in [1.0, 0.5, 2/3]: # equal weight mixture not very accurate
                    weights = _weights*b_weight + (1 - b_weight)*(1/p)
                    weights["gamma_reg"] = gamma_reg
                    weights["b_weight"] = b_weight
                    ls_weight.append(weights)

                    returns = cal_return(df_sectors.iloc[window:, :p], weights.iloc[:,:p])
                    returns["gamma_reg"] = gamma_reg
                    returns["b_weight"] = b_weight
                    ls_return.append(returns)
                # rsqs.to_csv(r".\regre\{}_rsq.csv".format(mkt))
            try:
                suffix = "e" if addErr else ""
                pd.concat(ls_weight).to_csv(
                    r".\weights\{}\{}_formula_{}.csv".format("weekly" if useWeekly else "monthly", mkt,
                                                              est_input.__name__ + suffix))
                pd.concat(ls_return).to_csv(
                    r".\performances\{}\{}_formula_{}.csv".format("weekly" if useWeekly else "monthly", mkt,
                                                                   est_input.__name__ + suffix))
                print("Output successful, {}_formula{}_{}".format(mkt, "e" if addErr else "", est_input.__name__))
            except PermissionError:
                print("please close data outputs.")
