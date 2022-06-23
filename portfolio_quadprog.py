import warnings
from utils import *
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers  # quadratic programming


def cal_weights(cal_mean_var, gamma_op, model_log=False):
    """
    calculate the allocation based on the cal_mean_var function given and gamma_op specified
    return a dataframe containing weightings of all time points
    """
    weights = {}
    if model_log:
        evec = {}

    # "weeks" stands for the rebalacing weeks
    weeks = pd.date_range(start=df_sectors.index[0], end=df_sectors.index[-1], freq="W-FRI")
    reb = len(weeks[weeks < df_sectors.index[window - 1]])

    for t in range(T - window):
        # rebalancing week
        row_index = df_sectors.index[t + window].strftime("%Y-%m-%d")
        if reb < len(weeks) and df_sectors.index[t + window - 1] <= weeks[reb] and df_sectors.index[t + window] > weeks[
            reb]:
            # select the calculation time horizon (last 260 weeks including holidays)
            idx_range = (df_sectors.index <= weeks[reb]) & (df_sectors.index >= weeks[reb - window + 1])
            df_window = df_sectors[idx_range].dropna(how='any', axis=1)
            p_window = len(df_window.columns) - n

            # calculate E and V using the method specified
            if cal_mean_var.__name__ == "factorx":
                exog_t = exogs[(np.cumsum(idx_range) == np.sum(idx_range)) & (~idx_range)].values[0]
                exog_t1 = (np.sum(exogs[idx_range].iloc[-51:, ]) + exog_t) / 52
                exog_future = np.reshape(np.concatenate([exog_t, exog_t1, exog_t1, exog_t1]), (4, exogs.shape[1]))

                Evec, Vmat = cal_mean_var(df_window, exog=exogs[idx_range], exog_future=exog_future, p=p_window)
            else:
                Evec, Vmat = cal_mean_var(df_window, p=p_window)
            Vmat = Vmat.astype('float')  # must convert otherwise the package gives a warning
            Evec = Evec.astype('float')

            # optimization
            solvers.options['show_progress'] = False  # do not show optimization output
            solvers.options['abstol'] = 0.1 ** -6
            solvers.options['reltol'] = 0.1 ** -5
            Q = gamma_op * matrix([list(Vmat[i, :]) for i in range(p_window)])
            q = -1 * matrix(list(Evec))
            G = matrix([[0.0] * i + [-1.0] + [0.0] * (p_window - i - 1) for i in range(p_window)])
            h = matrix([0.0] * p_window)
            A = matrix([1.0] * p_window, (1, p_window))
            b = matrix(1.0)

            try:
                sol = solvers.qp(Q, q, G, h, A, b)
                weights[row_index] = dict(zip(list(df_window.columns), list(sol['x'])))
            except:
                print("cannot find solution at {}".format(t + window))
                weights[row_index] = dict(zip(list(df_window.columns), list(np.ones(p_window) / p_window)))

            # save records if model log is TRUE
            if model_log:
                evec[row_index] = dict(zip(list(df_window.columns), list(Evec)))
            reb += 4

        # non-rebalancing week
        else:
            row_index_pre = df_sectors.index[t + window - 1].strftime("%Y-%m-%d")
            new_weights = pd.Series(weights[row_index_pre])*(1 + df_sectors.iloc[t + window - 1, :p])
            new_weights = new_weights / np.sum(new_weights)
            weights[row_index] = new_weights.to_dict()


    # save weights in dataframe
    weights = pd.DataFrame.from_dict(weights, orient="index")
    if model_log:
        evec = pd.DataFrame.from_dict(evec, orient='index')
        evec.to_csv(r".\regre\evec_{}_{}.csv".format(mkt, cal_mean_var.__name__))
    return weights


def cal_ew(df_sectors):
    """
    calculate the weekly weighting of an equally weighted portfolio, rebalanced monthly
    """
    weights = {}
    weeks = pd.date_range(start=df_sectors.index[0], end=df_sectors.index[-1], freq="W-FRI", closed="left")
    reb = len(weeks[weeks < df_sectors.index[window - 1]])
    for t in range(T - window):
        row_index = df_sectors.index[t + window].strftime("%Y-%m-%d")
        if reb < len(weeks) and df_sectors.index[t + window - 1] <= weeks[reb] and df_sectors.index[t + window] > weeks[reb]:
            idx_range = (df_sectors.index <= weeks[reb]) & (df_sectors.index >= weeks[reb - window + 1])
            df_window = df_sectors[idx_range].dropna(how='any', axis=1) # TODO: could optimize
            p_window = len(df_window.columns) - n

            weights[row_index] = dict(zip(list(df_window.columns), list(np.ones(p_window) / p_window)))
            reb += 4
        else:
            row_index_pre = df_sectors.index[t + window - 1].strftime("%Y-%m-%d")
            new_weights = pd.Series(weights[row_index_pre]) * (1 + df_sectors.iloc[t + window - 1, :p])
            new_weights = new_weights / np.sum(new_weights)
            weights[row_index] = new_weights.to_dict()

    weights = pd.DataFrame.from_dict(weights, orient="index")
    return weights

if __name__ == "__main__":
    # filter out future warning for pd.concat
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # define parameters
    # sector_indices = ("hsci", "csi", "dj")
    # markets = ("hk", "cn", "us")
    # level = "sector"
    total_ret = False

    markets = ("hk", "us")
    level = "industry_group"

    window = 260
    func_list = [est_input_capm, est_input_factor, est_input_hist,
                 est_input_factorx]  # list of methods (est_input_capm_old, est_input_factor_old)
    for i in range(len(markets)):
        for est_input in func_list:
        # for est_input in [est_input_factorx]:
            ls_weight = []
            ls_return = []
            mkt = markets[i]

            # get from local files
            # df_sectors, exogs, p, n, T = load_data(mkt, level)
            # or, get from mysql database
            df_sectors, exogs, p, n, T = load_data_sql(mkt, level)

            for gamma_op in [2, 4, 8]:
                weights = cal_weights(est_input, gamma_op,
                                      model_log=True if gamma_op == 2 else False)  # save model outputs when gamma=2 (for checking)
                weights["gamma_op"] = gamma_op
                ls_weight.append(weights)

                # calculate weekly return of the allocation
                returns = cal_return(df_sectors.iloc[window:, :p], weights.iloc[:, :p])
                returns["gamma_op"] = gamma_op
                ls_return.append(returns)

            # calculate the return of equally-weighted portfolio - for benchmarking
            equal = cal_ew(df_sectors)
            equal_return = cal_return(df_sectors.iloc[window:, :p], equal)

            if total_ret: # suffix for test cases
                suffix = "t"
            else: # default suffix
                suffix = "e"

            try:
                w_suffix = "3" if window < 260 else "" # add suffix in file name to indicate 3y window

                pd.concat(ls_weight).to_csv(rf".\weights\{mkt}_{level}_{est_input.__name__ + suffix}{w_suffix}.csv")
                pd.concat(ls_return).to_csv(rf".\performances\{mkt}_{level}_{est_input.__name__ + suffix}{w_suffix}.csv")

                equal.to_csv(rf".\weights\{mkt}_{level}_ew{w_suffix}.csv")
                equal_return.to_csv(rf".\performances\{mkt}_{level}_ew{w_suffix}.csv")

            except PermissionError:
                print("please close data outputs.")
            else:
                print(f"Output successful, {mkt}_{level}_{est_input.__name__ + suffix}{w_suffix}")



