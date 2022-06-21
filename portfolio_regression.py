import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from utils import load_data, cal_mean_var, cal_perf, sharpe

sector_index = 'csi'
code = 'cn'
df_sectors, p, T = load_data(sector_index, code)
weights = {}
sharpes = []
# parameters
gamma_reg = 0.0001
gamma_op = 2
window = 60

for t in range(T-window):
    r_mean, r_vol = cal_mean_var(df_sectors.iloc[t:t + window], gamma_reg, p)

    # optimization part
    constraint = ({'type': 'eq', 'fun': lambda x: 1-np.sum(x)})
    bnds = [(0, 1)]*p
    w0 = np.ones(p)/p
    w = minimize(sharpe, w0, method = 'SLSQP', constraints = constraint, bounds = bnds, args = (r_mean, r_vol))
    if w.success:
        weights[df_sectors.index[t+window]] = w.x
        sharpes.append(-w.fun)
    else:
        weights[df_sectors.index[t+window]] = None

weights = pd.DataFrame.from_dict(weights, orient="index",columns=df_sectors.columns[:p])
weights.to_csv(r".\sector-rotation-data\weights_{}_{}.csv".format(code, "slsqp"))

# return computation
cal_perf(df_sectors.iloc[window:, :10], weights)