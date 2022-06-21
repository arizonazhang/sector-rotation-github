'''git version 2'''
import pandas as pd
import numpy as np
import numpy.linalg as linalg
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine


def connectDB():
    config = {
        'user': 'infoport',
        'password': 'HKaift-123',
        'host': '192.168.2.81',
        'database': 'AlternativeData',
        'raise_on_warnings': False
    }
    try:
        cnx = mysql.connector.connect(**config)
        return cnx
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        return 0


def _get_exogs(code):
    """
    extract exogenous variable for the given country
    exogenous variables are the market return of other markets, with market codes specified in exog_group; external variables
    :param code: country code
    :return:
    """
    exog_group = {'hk400': ['cn800', 'us1500'],
                  'cn800': ['hk400', 'us1500'],
                  'us1500': ['hk400', 'cn800'],
                  'hk300': ['cn800', 'us1500']}  # exogenous variable for different markets respectively
    exog_common = ['.dMIEF00000G', '.dMIEA00000G', 'JPY=', 'US10YT=RR']  # common exogenous variable
    exog = pd.DataFrame()

    # get market data of the exogenous variable
    for c in exog_group[code]:
        data = pd.read_csv(r'.\input\factor\ten_factor_vw_{}_week_5.csv'.format(c), index_col=0)
        data.index = pd.to_datetime(data.index)
        one_exog = data['exmkt'] #TODO: revised
        one_exog.name = '%s_market' % c[:2]
        if exog.empty:
            exog[one_exog.name] = one_exog
            exog.index = one_exog.index
        else:
            exog = exog.join(one_exog, how='outer')

    # get other exogenous variables
    exog_external = pd.read_csv(r'.\input\factor\exogs_weekly.csv', index_col=0)
    exog_external.index = pd.to_datetime(exog_external.index)
    exog_external = exog_external[exog_common]
    exog = exog.join(exog_external, how='inner')

    return exog


def _get_return(query, cnx, *args):
    df = pd.read_sql(query, cnx, params=list(args))
    df = df.pivot(index='date', columns='col', values='value')
    df.index = pd.to_datetime(df.index)
    return df


def load_data_sql(code, level, start_dt="2005-01"):
    factor_query = """
        SELECT date, code as col, value  FROM AlternativeData.SectorRotationRet 
        where markets = %s and date >= %s and side = "LS" and code in ({c});
    """
    sector_query = """
        SELECT date, code as col, value FROM AlternativeData.SectorRotationRet 
        where markets = %s and date >= %s;
    """
    exog_query = """
        SELECT date, code as col, value FROM AlternativeData.SectorRotationRet 
        where markets = "common_exogs" and date >= %s and code in ({c});
    """
    exog_query_2 = """
        select date, markets as col, value from AlternativeData.SectorRotationRet 
        where date >= %s and markets in (%s, %s) and code = "exmkt"
    """
    rf_query = """
        select date, markets as col, value from AlternativeData.SectorRotationRet 
        where date >= %s and code = "rf"
    """

    engine = create_engine("mysql+mysqlconnector://infoport:HKaift-123@192.168.2.81/AlternativeData") # TODO: revised
    common_exogs = ['.dMIEF00000G', '.dMIEA00000G', '.DXY', 'US10YT=RR']
    exog_group = {'hk': ['cn_factor', 'us_factor'],
                  'cn': ['hk_factor', 'us_factor'],
                  'us': ['hk_factor', 'cn_factor']}  # exogenous variable for different markets respectively
    common_factors = ['exmkt', 'size', 'bm', 'ep', 'roe', 'ag', 'm12', 'm1', 'beta', 'idvc', 'dtvm']
    cnx = connectDB()

    # query sector data from sql
    df_sectors = _get_return(sector_query, engine, f"{code}_{level}", start_dt)

    # query factor data from sql
    factor_query = factor_query.format(c=",".join(['%s']*len(common_factors))) #TODO: revised
    df_factors = _get_return(factor_query, engine, f"{code}_factor", start_dt, *common_factors)

    exog_query = exog_query.format(c=",".join(['%s']*len(common_exogs)))
    df_exogs = _get_return(exog_query, engine, start_dt, *common_exogs) #TODO: revised (cnx changed to engine)
    df_exogs2 = _get_return(exog_query_2, engine, start_dt, *exog_group[code])
    df_exogs2.columns = ['%s_market' %col for col in df_exogs2.columns]
    df_exogs2 = df_exogs2.fillna(0) # fill empty a-share market return during holidays
    df_rf = _get_return(rf_query, engine, start_dt)

    # join table
    df_sectors = df_sectors.join(df_factors, how="inner")
    T, p = df_sectors.shape
    n = len(df_factors.columns)
    p = p - n

    # convert sector return to excess return
    df_sectors.iloc[:, :p] = df_sectors.iloc[:, :p].subtract(df_rf[f"{code}_factor"], axis=0) #TODO: revised

    # exogneous data table
    df_exogs = df_exogs.join(df_exogs2.iloc[:, :len(exog_group[code])])
    df_exogs = df_exogs.join(df_sectors, how="right").iloc[:, :len(df_exogs.columns)]
    df_exogs = df_exogs.shift(1) # shift by one period for lagging
    # fill empty exogs data with 0, just applicable to cn_market (no trading on Spring festival and National Day)
    df_exogs = df_exogs.fillna(0)

    return df_sectors, df_exogs, p, n, T

def load_data(code, level, start_dt="2005-01"):
    """
    loading factor return, sector return and exogenous data; all return are EXCESS return
    :param sector_index: csi, hsci or dj
    :param code: cn, hk or us
    :param start_dt: start date of time series
    :return: df containing sector and factor return, df containing exogenous variable, no. of sectors, no. of observations
    """

    suffix = {'hk':'400', 'cn':'800', 'us':'1500'}
    factor_code = code + suffix[code] #TODO: revised
    df_factors = pd.read_csv(r".\input\factor\ten_factor_vw_{}_week_5.csv".format(factor_code), index_col=0)
    df_sectors = pd.read_csv(r".\input\sector\{}_{}_weekly.csv".format(code, level), index_col=0)

    df_factors.index = pd.to_datetime(df_factors.index)
    df_sectors.index = pd.to_datetime(df_sectors.index)

    df_sectors = df_sectors[df_sectors.index >= pd.to_datetime(start_dt)]
    p = len(df_sectors.columns)
    df_sectors = df_sectors.join(df_factors, how="inner")

    # convert sector return to excess return
    df_sectors.iloc[:, :p] = df_sectors.iloc[:, :p].subtract(df_sectors.rf, axis=0)
    del df_sectors['rf'] #TODO:revised
    n = len(df_sectors.columns) - p
    T = len(df_sectors.index)

    exogs = _get_exogs(factor_code)
    if not exogs.empty:
        # align exogenous variable with main data - df_sectors
        exogs = exogs.join(df_sectors, how="right").iloc[:, :len(exogs.columns)]
        # shift by one period for lagging
        exogs = exogs.shift(1)
        # fill empty exogs data with 0, just applicable to cn_market (no trading on Spring festival and National Day)
        exogs = exogs.fillna(0)

    return df_sectors, exogs, p, n, T


def est_input_factor_old(df_sectors, p, use_err=True, **kwargs):
    """
    estimate E, V using 11-factor model;
    expected factor return are calculated as average return of the past 5 years
    """
    alphas = []
    betas = []
    err = []
    X0 = df_sectors.iloc[:, p:(p + 11)]
    X = sm.add_constant(X0)
    rsq = []

    for i in range(p):
        y = df_sectors.iloc[:, i]
        res = sm.OLS(y, X).fit()
        betas.append(res.params[1:])
        alphas.append(res.params[0])
        err.append(res.resid)
        rsq.append(res.rsquared)

    # calculate weightings
    alphas = np.array(alphas)
    betas = pd.concat(betas, axis=1)
    err = pd.concat(err, axis=1)
    r_mean = np.dot(X0.mean(), betas)
    r_var = np.dot(betas.T, np.dot(X0.cov(), betas))
    if use_err:
        r_var += np.eye(p) * np.diag(err.cov())
    return r_mean, r_var  # , rsq


def est_input_factor(df_sectors, p, use_err=True, **kwargs):
    """
    estimate E, V using 11-factor model;
    expected factor return are calculated as 4-step ahead predicted value from the VAR(1) model
    """
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    betas = []
    alphas = []
    err = []
    X0 = df_sectors.iloc[:, p:(p + 11)]
    X = sm.add_constant(X0)
    rsq = []

    for i in range(p):
        y = df_sectors.iloc[:, i]
        res = sm.OLS(y, X).fit()
        betas.append(res.params[1:])
        alphas.append(res.params[0])
        err.append(res.resid)
        rsq.append(res.rsquared)

    # calculate the expected factor value using VAR
    X0.index = pd.Series(range(len(X0.index)))
    var_model = VAR(X0)
    results = var_model.fit(1)

    n = len(X0.columns)
    A = results.params.iloc[1:, :].T
    term1 = linalg.inv(np.diag(np.ones(n * n)) - np.kron(A, A))
    term2 = np.reshape(results.sigma_u.values, newshape=n * n, order='F')
    f_sigma_vec = np.dot(term1, term2)
    f_sigma = np.reshape(f_sigma_vec, (n, n))

    # f_sigma = results.forecast_cov(4)[3]
    f_mean = results.forecast(X0.values[-1:], 4)[3]

    # calculate weightings
    betas = pd.concat(betas, axis=1)
    alphas = np.array(alphas)
    err = pd.concat(err, axis=1)
    r_mean = f_mean.dot(betas)
    r_var = np.dot(betas.T, np.dot(f_sigma, betas))
    if use_err:
        r_var += np.eye(p) * np.diag(err.cov())
    return r_mean, r_var  # , rsq


def est_input_capm(df_sectors, p, use_err=True):
    """
    estimate E, V using CAPM model;
    expected factor return are calculated as 4-step ahead predicted value from the AR(1) model
    """

    import statsmodels.api as sm
    from statsmodels.tsa.api import AutoReg

    alphas = []
    betas = []
    err = []
    X0 = df_sectors.exmkt
    X = sm.add_constant(X0)
    rsq = []

    for i in range(p):
        y = df_sectors.iloc[:, i]
        res = sm.OLS(y, X).fit()
        betas.append(res.params[1:])
        alphas.append(res.params[0])
        err.append(res.resid)
        rsq.append(res.rsquared)

    # calculate weightings
    alphas = np.array(alphas)
    betas = pd.concat(betas, axis=1)
    err = pd.concat(err, axis=1)

    # estimate expected factor mean and covariance
    X0.index = pd.Series(range(len(X0.index)))
    model = AutoReg(X0, 1, old_names=False)
    res = model.fit()  # np.sum(res.params*np.array([1, X0[-1]]))
    res.summary()
    f_mean = res.predict(start=len(X0), end=len(X0) + 3)[len(X0) + 3]
    f_var = res.sigma2 / (1 - res.params['exmkt.L1'])

    r_mean = np.squeeze(np.array(betas) * f_mean)
    r_var = f_var * np.dot(betas.T, betas)
    if use_err:
        r_var += np.eye(p) * np.diag(err.cov())
    return r_mean, r_var  # , rsq


def est_input_hist(df_sectors, p, **kwargs):
    """
    estimate E, V using asset historical mean and variance
    """
    r_mean = np.mean(df_sectors.iloc[:, :p])
    # np.cov calculated the sample covariance, variance term is divided by N - 1, unlike np.var where variance term is divided by N
    # could set bias = True to make np.cov divide by N
    r_var = np.cov(df_sectors.iloc[:, :p], rowvar=False)
    return r_mean, r_var


def cal_return(returns, weights):
    """
    calculate portfolio return given sector return and allocations
    """
    # pd.DataFrame(np.sum(np.array(returns)*np.array(weights), axis=1), index=returns.index)
    ret_ls = np.nansum(np.array(returns) * np.array(weights), axis=1)
    ret_ls = pd.DataFrame(ret_ls, index=weights.index)
    return ret_ls


def est_input_capm_old(df_sectors, p, use_err=True, **kwargs):
    """
    estimate E, V using CAPM model;
    expected factor return are calculated as average return of the rolling window
    """
    alphas = []
    betas = []
    err = []
    X0 = df_sectors.exmkt
    X = sm.add_constant(X0)
    rsq = []

    for i in range(p):
        y = df_sectors.iloc[:, i]
        res = sm.OLS(y, X).fit()
        betas.append(res.params[1:])
        alphas.append(res.params[0])
        err.append(res.resid)
        rsq.append(res.rsquared)

    # calculate weightings
    alphas = np.array(alphas)
    betas = pd.concat(betas, axis=1)
    err = pd.concat(err, axis=1)
    r_mean = np.array(betas) * np.mean(X0)
    r_var = np.var(X0) * np.dot(betas.T, betas)
    r_mean = np.squeeze(r_mean)
    if use_err:
        r_var += np.eye(p) * np.diag(err.cov())
    return r_mean, r_var


def est_input_factorx(df_sectors, exog, exog_future, p, use_err=True):
    """
    estimate E, V using 11-factor model;
    expected factor return are calculated as 4-step ahead predicted value from the VARX(1) model
    VARX(1) introduce with exogenous variable with 1 time lag
    """
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    betas = []
    alphas = []
    err = []
    X0 = df_sectors.iloc[:, p:(p + 11)]
    X = sm.add_constant(X0)
    rsq = []

    for i in range(p):
        y = df_sectors.iloc[:, i]
        res = sm.OLS(y, X).fit()
        betas.append(res.params[1:])
        alphas.append(res.params[0])
        err.append(res.resid)
        rsq.append(res.rsquared)

    # calculate the expected factor value using VAR
    if not exog.empty:
        endog = X0
        endog.index = range(len(endog.index))
        exog.index = range(len(exog.index))
    else:
        exog = None
        endog = X0
        endog.index = range(len(endog.index))
        exog_future = None

    var_model = VAR(endog, exog=exog)
    results = var_model.fit(1)

    n = len(endog.columns)
    k = len(exog.columns)
    A = results.params.iloc[1 + k:, :].T
    B = results.params.iloc[1:1 + k, :].T
    term1 = linalg.inv(np.diag(np.ones(n * n)) - np.kron(A, A))
    term2 = np.dot(np.kron(B, B), np.reshape(exog.cov().values, k * k)) + np.reshape(results.sigma_u.values,
                                                                                         newshape=121, order='F')
    f_sigma_vec = np.dot(term1, term2)
    f_sigma = np.reshape(f_sigma_vec, (n, n))
    f_mean = results.forecast(X0.values[-1:], steps=4, exog_future=exog_future)[3]

    if (linalg.eigvals(f_sigma) < 0).any():
        print("not positive semi-definite! {}".format(df_sectors.index[0]))

    # calculate weightings
    betas = pd.concat(betas, axis=1)
    alphas = np.array(alphas)
    err = pd.concat(err, axis=1)
    r_mean = f_mean.dot(betas)
    r_var = np.dot(betas.T, np.dot(f_sigma, betas))
    if use_err:
        r_var += np.eye(p) * np.diag(err.cov())
    return r_mean, r_var


est_input_factor_old.__name__ = "factorhist"
est_input_capm_old.__name__ = "capmhist"
est_input_hist.__name__ = "hist"
est_input_factor.__name__ = "factor"
est_input_capm.__name__ = "capm"
est_input_factorx.__name__ = "factorx"


if __name__ == '__main__':
    data1, data2, p, n, T = load_data_sql('us', start_dt="2005-01")
    print(data1.head())