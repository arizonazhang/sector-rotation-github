import warnings
from utils import *
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers # quadratic programming


def cal_weights(cal_mean_var, gamma_reg, gamma_op, addErr, model_log = False):
    """
    calculate the allocation based on the cal_mean_var function given and gamma_op specified
    return a dataframe containing weightings of all time points
    """
    weights = {}
    if model_log:
        evec = {}

    # "weeks" stands for the rebalacing weeks
    weeks = pd.date_range(start=df_sectors.index[0], end=df_sectors.index[-1], freq="W-FRI" if useWeekly else "BM")
    reb = len(weeks[weeks < df_sectors.index[window-1]])

    for t in range(T-window):
        # rebalancing week
        if reb < len(weeks) and df_sectors.index[t + window-1] <= weeks[reb] and df_sectors.index[t+window] > weeks[reb]:

            # calculate E and V
            idx_range = (df_sectors.index <= weeks[reb])&(df_sectors.index >= weeks[reb-window+1])

            if cal_mean_var.__name__ == "factorx":
                if exogs.empty:
                    exog_future = exogs
                else:
                    exog_t = exogs[(np.cumsum(idx_range)==np.sum(idx_range))&(~idx_range)].values[0]
                    exog_t1 = (np.sum(exogs[idx_range].iloc[-51:,])+exog_t)/52
                    exog_future = np.reshape(np.concatenate([exog_t, exog_t1, exog_t1, exog_t1]), (4, exogs.shape[1]))
                Evec, Vmat = cal_mean_var(df_sectors[idx_range], exog=exogs[idx_range] if not exogs.empty else exogs,
                                          exog_future=exog_future, p=p, useErr=addErr)
            else:
                Evec, Vmat = cal_mean_var(df_sectors[idx_range], gamma_reg=gamma_reg, p=p, useErr=addErr)
            Vmat = Vmat.astype('float')  # must convert otherwise the package gives a warning
            Evec = Evec.astype('float')

            # optimization
            solvers.options['show_progress'] = False  # do not show optimization output
            solvers.options['abstol'] = 0.1**-6
            solvers.options['reltol'] = 0.1**-5
            Q = gamma_op * matrix([list(Vmat[i, :]) for i in range(p)])
            q = -1 * matrix(list(Evec))
            G = matrix([[0.0] * i + [-1.0] + [0.0] * (p - i - 1) for i in range(p)])
            h = matrix([0.0] * p)
            A = matrix([1.0] * p, (1, p))
            b = matrix(1.0)

            try:
                sol = solvers.qp(Q, q, G, h, A, b)
                weights[df_sectors.index[t + window].strftime("%Y-%m-%d")] = list(sol['x'])
            except:
                print("cannot find solution at {}".format(t+window))
                weights[df_sectors.index[t + window].strftime("%Y-%m-%d")] = list(np.ones(p)/p)

            # save records if model log is TRUE
            if model_log:
                evec[df_sectors.index[t + window].strftime("%Y-%m-%d")] = list(Evec)
            reb += 4

        # non-rebalancing week
        else:
            new_weights = weights[df_sectors.index[t + window - 1].strftime("%Y-%m-%d")] * (
                        1 + df_sectors.iloc[t + window - 1, :p])
            weights[df_sectors.index[t + window].strftime("%Y-%m-%d")] = new_weights / np.sum(new_weights)

    # save weights in dataframe
    weights = pd.DataFrame.from_dict(weights, orient="index", columns=df_sectors.columns[:p])
    if model_log:
        evec = pd.DataFrame.from_dict(evec, orient='index', columns=df_sectors.columns[:p])
        evec.to_csv(r".\regre\evec_{}_{}.csv".format(mkt, cal_mean_var.__name__))
    return weights

def cal_ew():
    """
    calculate the weekly weighting of an equally weighted portfolio, rebalanced monthly
    """
    weights = {}
    weeks = pd.date_range(start=df_sectors.index[0], end=df_sectors.index[-1], freq="W-FRI" if useWeekly else "BM", closed = "left")
    reb = len(weeks[weeks < df_sectors.index[window-1]])
    for t in range(T-window):
        if reb < len(weeks) and df_sectors.index[t + window-1] <= weeks[reb] and df_sectors.index[t+window] > weeks[reb]:
            weights[df_sectors.index[t + window].strftime("%Y-%m-%d")] = np.ones(p)/p
            reb += 4
        else:
            new_weights = weights[df_sectors.index[t + window - 1].strftime("%Y-%m-%d")] * (
                        1 + df_sectors.iloc[t + window - 1, :p])
            weights[df_sectors.index[t + window].strftime("%Y-%m-%d")] = new_weights / np.sum(new_weights)
    weights = pd.DataFrame.from_dict(weights, orient="index", columns=df_sectors.columns[:p])
    return weights


warnings.simplefilter(action='ignore', category=FutureWarning)

# define parameters
addErr = True # no need to change
useWeekly = True # no need to change
sector_indices = ("dj", "csi", "hsci")
markets = ("us1500", "cn800", "hk400")
window = 260
func_list = [est_input_capm, est_input_factor, est_input_hist,
             est_input_factorx, est_input_capm_old, est_input_factor_old] # list of methods

for est_input in func_list:
# for est_input in [est_input_factorx]:
    for i in range(len(markets)):
        ls_weight = []
        ls_return = []
        mkt = markets[i]
        index = sector_indices[i]

        # get factor and sector return, p stands for no. of sectors
        df_sectors, exogs, p, T = load_data(index, mkt, useWeekly)
        _equal = cal_ew()
        for gamma_op in [2, 4, 8]:
            _weights = cal_weights(est_input, 0,
                                   gamma_op, addErr,
                                   model_log=True if gamma_op==2 else False) # save model outputs when gamma=2 (for checking)

            for b_weight in [0, 1]:
                # calculate weights with b_weight
                weights = _weights*b_weight + (1 - b_weight)*_equal
                weights["gamma_op"] = gamma_op
                weights["b_weight"] = b_weight
                ls_weight.append(weights)

                # calculate weekly return of the allocation
                returns = cal_return(df_sectors.iloc[window:, :p], weights.iloc[:, :p])
                returns["gamma_op"] = gamma_op
                returns["b_weight"] = b_weight
                ls_return.append(returns)

        suffix = "e" if addErr else ""

        try:
            pd.concat(ls_weight).to_csv(r".\weights\{}\{}_quadprog_{}{}.csv".format("weekly" if useWeekly else "monthly",
                                                                                  mkt,  est_input.__name__+suffix,
                                                                                    "3" if window < 260 else ""))
            pd.concat(ls_return).to_csv(r".\performances\{}\{}_quadprog_{}{}.csv".format("weekly" if useWeekly else "monthly",
                                                                                       mkt,  est_input.__name__+suffix,
                                                                                         "3" if window < 260 else ""))
        except PermissionError:
            print("please close data outputs.")
        else:
            print("Output successful, {}_quadprog_{}{}".format(mkt, est_input.__name__, suffix))