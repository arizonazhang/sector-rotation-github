from utils import load_data, cal_mean_var, sharpe, cal_perf, cal_return
import pandas as pd
from cvxopt import matrix, solvers

def cal_weights(gamma_reg, gamma_op, addErr, long_only=True):
    weights = {}
    for t in range(T - window):
        # calculate respective return and vol
        Evec, Vmat, rsq = cal_mean_var(df_sectors.iloc[t:t + window], gamma_reg, p, useReg=False, useErr=addErr)
        Vmat = Vmat.astype('float')  # must convert otherwise the package gives a warning
        Evec = Evec.astype('float')
        # Vmat.to_csv("vmat.csv", index=False)
        # pd.Series(Evec).to_csv("evec.csv", index=False)
        Q = gamma_op * matrix([list(Vmat[i, :]) for i in range(p)])
        q = -1 * matrix(list(Evec))
        G = matrix([[0.0] * i + [-1.0] + [0.0] * (p - i - 1) for i in range(p)]) if long_only else None
        h = matrix([0.0] * p) if long_only else None
        A = matrix([1.0] * p, (1, p))
        b = matrix(1.0)
        sol = solvers.qp(Q, q, G, h, A, b)
        weights[df_sectors.index[t + window]] = list(sol['x'])
        # w = np.array(list(sol['x']))
        # values = np.dot(w, np.dot(Vmat, w)) - np.dot(w, Evec)

    weights = pd.DataFrame.from_dict(weights, orient="index", columns=df_sectors.columns[:p])
    return weights

# parameters
window = 60
sector_indices = ("hsci", "dj", "csi")
markets = ("hk", "us", "cn")
addErr = False
long_only = False

for i in range(3):
    ls_weight = []
    ls_return = []
    mkt = markets[i]
    index = sector_indices[i]
    df_sectors, p, T = load_data(index, mkt) # p stands for no. of sectors
    for gamma_op in [2, 4, 8]:
        _weights = cal_weights(0, gamma_op, addErr, long_only = long_only)
        for b_weight in [0, 0.5, 2/3, 1]:
            weights = _weights*b_weight + (1 - b_weight)*(1/p)
            weights["gamma_op"] = gamma_op
            weights["b_weight"] = b_weight
            ls_weight.append(weights)

            returns = cal_return(df_sectors.iloc[window:, :p], weights.iloc[:,:p])
            returns["gamma_op"] = gamma_op
            returns["b_weight"] = b_weight
            ls_return.append(returns)

    suffix = ""
    suffix += "e" if addErr else ""
    suffix += "l" if not long_only else ""

    try:
        pd.concat(ls_weight).to_csv(r".\weights\{}_quadprog{}.csv".format(mkt, suffix))
        pd.concat(ls_return).to_csv(r".\performances\{}_quadprog{}.csv".format(mkt, suffix))
    except PermissionError:
        print("please close data outputs.")
    else:
        print("Output successful. {}_quadprog{}".format(mkt, suffix))

# for t in range(T-window):
#     # calculate respective return and vol
#     Evec, Vmat = cal_mean_var(df_sectors.iloc[t:t + window], gamma_reg, p)
#
#     Vmat = Vmat.astype('float') # must convert otherwise the package gives a warning
#     Evec = Evec.astype('float')
#     # Vmat.to_csv("vmat.csv", index=False)
#     # pd.Series(Evec).to_csv("evec.csv", index=False)
#     Q = gamma_op*matrix([list(Vmat.iloc[i,:]) for i in range(p)])
#     q = -1*matrix(list(Evec))
#     G = matrix([[0.0]*i + [-1.0] + [0.0]*(p-i-1) for i in range(p)])
#     h = matrix([0.0] * p)
#     A = matrix([1.0]*p, (1, p))
#     b = matrix(1.0)
#     sol = solvers.qp(Q, q, G, h, A, b)
#     weights[df_sectors.index[t+window]] = list(sol['x'])
#     # w = np.array(list(sol['x']))
#     # values = np.dot(w, np.dot(Vmat, w)) - np.dot(w, Evec)
#
# weights = pd.DataFrame.from_dict(weights, orient="index", columns=df_sectors.columns[:p])
# weights.to_csv(r".\sector-rotation-data\weights-quadprog_{}_{:.0f}_{:.0f}.csv".format(sector_index, gamma_op, -np.log10(gamma_reg)))

# return computation
# cal_perf(df_sectors.iloc[window:, :10], weights)

