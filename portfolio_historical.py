from utils import load_data, cal_mean_var, sharpe, cal_perf, cal_return
from scipy import linalg
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

def cal_weights(gamma_reg=None):
    weights = {}
    for t in range(T - window):
        ## DEBUD USE
        Evec = np.mean(df_sectors.iloc[t:t + window, :p])
        Vmat = np.cov(df_sectors.iloc[t:t+window, :p], rowvar=False)
        if gamma_reg:
            Evec += np.ones(p)*gamma_reg
            Vmat += np.eye(p)*gamma_reg
        w = np.dot(linalg.inv(Vmat), Evec)
        w = w/np.sum(w)
        weights[df_sectors.index[t + window]] = list(w)
    weights = pd.DataFrame.from_dict(weights, orient="index", columns=df_sectors.columns[:p])
    return weights

# parameters
window = 60
sector_indices = ("hsci", "dj", "csi")
markets = ("hk", "us", "cn")

for i in range(3):
    ls_weight = []
    ls_return = []
    mkt = markets[i]
    index = sector_indices[i]
    df_sectors, p, T = load_data(index, mkt) # p stands for no. of sectors
    for gamma_reg in [0.01, 0.02, 0.05]:
        _weights = cal_weights(gamma_reg)
        for b_weight in [0, 0.5, 1.0]:
            weights = _weights*b_weight + (1 - b_weight)*(1/p)
            weights["gamma_reg"] = gamma_reg
            weights["b_weight"] = b_weight
            ls_weight.append(weights)

            returns = cal_return(df_sectors.iloc[window:, :p], weights.iloc[:,:p])
            returns["gamma_reg"] = gamma_reg
            returns["b_weight"] = b_weight
            ls_return.append(returns)
    pd.concat(ls_weight).to_csv(r".\weights\{}_historical.csv".format(mkt))
    pd.concat(ls_return).to_csv(r".\performances\{}_historical.csv".format(mkt))
