"""
This is a package for uploading data into the database for the sector rotation projects.

The requirements for data are:
- Since weekly data is used, each record should bare a date that is a friday.
- For return data, units are 1 (not %)
- For Chinese factor return, empty records during long holiday like CNY and National Day are dropped
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import eikon as ek
ek.set_app_key('51dd084a4cea45dab6ac09cdbfe75cd83d17caa8')


class api_connector():
    """
    this class serves as an api connection to ek.
    It also helps to keep track of all the requests made and data points downloaded.
    """
    ##TODO: can add other functions in here
    def __init__(self):
        self.requsts = 0
        self.datapoints = 0

    def add_request(self):
        self.requests += 1

    def add_datapoints(self, points):
        self.datapoints += points

    def get_api(self, rics, start_dt, end_dt):
        """query data from api, add number of request and data points downloaded"""
        try:
            data = ek.get_timeseries(rics, fields='CLOSE', interval='weekly',
                                     start_date=start_dt, end_date=end_dt,
                                     debug=True)
            data.index = pd.to_datetime(data.index)

            if len(rics) == 1:
                data.columns = rics
            if len(data.columns) != len(rics):
                raise KeyError

            self.requsts += 1
            self.datapoints += len(data.index)
            return data

        except KeyError:
            print("Error: some indices not available")
        except ValueError:
            print("Error: a parameter type or value is wrong")
        except Exception:
            print("Error: request failed")


    def print_counter(self):
        """print counter records"""
        print("API download records:")
        print(f"No. of requests made: {self.requsts}")
        print(f"No. of data points downloaded: {self.datapoints}")


# get mapping dictionary
def get_mapping():
    """obtain ticker and name mapping table from sql"""
    # from sql
    engine = create_engine("mysql+mysqlconnector://infoport:HKaift-123@192.168.2.81/AlternativeData")
    mdict = pd.read_sql("Select * from SecRotMapping", engine)
    mdict = mdict.set_index('code').to_dict()['name']

    # from local csv (comment the above and un
    # mdict = pd.read_csv(r'.\input\code_mapping.csv').set_index('code').to_dict()['name']

    return mdict

mdict = get_mapping()

ric_dict = {'us_sector': ['.TR15GSPE', '.TR15GSPM', '.TR15GSPI', '.TR15GSPD', '.TR15GSPS',
                          '.TR15GSPA', '.TR15GSPF', '.TR15GSPT', '.TR15GSPL', '.TR15GSPU', '.TR15GSPREC'],
            'cn_sector': ['.CSIH00929', '.CSIH00934', '.CSIH00933', '.CSIH00930',
                          '.CSIH00928', '.CSIH00935', '.CSIH00936', '.CSIH00937',
                          '.CSIH00931', '.CSIH00932'],
            'hk_sector': ['.HSCIM', '.HSCIF', '.HSCIH', '.HSCIIG', '.HSCIE', '.HSCIIT',
                          '.HSCIT', '.HSCIU', '.HSCICD', '.HSCICS', '.HSCIPC'],  # other two?
            'us_industry_group': ['.TR15GSPE', '.TR15GSPM', '.TR15GSPIC', '.TR15GSPCS',
                                  '.TR15GSPTRN', '.TR15GSPAU', '.TR15GSPLP', '.TR15GSPHR',
                                  '.TR15GSPMS', '.TR15GSPFDGR', '.TR15GSPFBT', '.TR15GSPHHPE',
                                  '.TR15GSPHC', '.TR15GSPPHB', '.TR15GSPBK', '.TR15GSPDF',
                                  '.TR15GSPINSC', '.TR15GSPIS', '.TR15GSPTEHW', '.TR15GSPSEQP',
                                  '.TR15GSPTS', '.TR15GSPME', '.TR15GSPU', '.TR15GSPREC'],
            'hk_industry_group': ['.TRXFLDHDTA1', '.TRXFLDHDTE1', '.TRXFLDHDTE2', '.TRXFLDHDTF1',
                                  '.TRXFLDHDTF4', '.TRXFLDHDTH1', '.TRXFLDHDTH2', '.TRXFLDHDTI1',
                                  '.TRXFLDHDTI2', '.TRXFLDHDTI4', '.TRXFLDHDTM1', '.TRXFLDHDTM2',
                                  '.TRXFLDHDTM3', '.TRXFLDHDTN1', '.TRXFLDHDTN2', '.TRXFLDHDTT1',
                                  '.TRXFLDHDTT2', '.TRXFLDHDTU1', '.TRXFLDHDTY1', '.TRXFLDHDTY2',
                                  '.TRXFLDHDTY3', '.TRXFLDHDTY4'],
            'exogs': ['.dMIEF00000G', '.dMIEA00000G', 'US10YT=RR', '.DXY']}


def check_complete(data):
    """
    sanity check of date coverage completeness.
    The program prints out total number of fridays covered and prints out missing fridays in between
    """
    # check missing dates in between
    dates = pd.date_range(start=data.index[0], end=data.index[-1], freq="W-FRI")
    missing_dates = dates[~dates.isin(data.index)].strftime("%Y-%m-%d").values
    if len(missing_dates) > 0:
        print("These fridays are not covered: ")
        print(list(missing_dates))
    print(f"In total the data has records on {len(data.index)} fridays")
    return 0

def get_factor_single(country, code, save_csv=False):
    """
    get factor data (long-short) for a given country and unstack the data to upload to database
    """
    data = pd.read_csv(r'.\input\ten_factor_vw_{}_week_5.csv'.format(code), index_col=0)
    data.index = pd.to_datetime(data.index)

    # adjust column names
    data.columns = ["rf" if item == "rf_week" else item for item in data.columns]

    # convert dates to Fridays and merge same Fridays together
    if (data.index.weekday != 4).any():
        data.index = pd.Series(list(map(lambda x: x + np.timedelta64(4 - x.weekday(), 'D'), data.index)))
        data = (data + 1).groupby(data.index).prod() - 1

    check_complete(data)

    # stack data and add other columns
    data['exmkt'] = data['market'] - data['rf']
    data = data.drop(columns=['market'])

    if save_csv:
        data.to_csv(r'.\input\factor\ten_factor_vw_{}_week_5.csv'.format(code))

    df = pd.DataFrame(data.stack()).reset_index()
    df.columns = ['date', 'code', 'value']
    df['markets'] = country + "_factor"
    df['name'] = df.code.apply(lambda x: mdict[x])
    df['side'] = "LS"
    return df


def get_factor():
    """
    get factor data for all markets
    """
    factor_indices = {'cn': 'cn800', 'hk': 'hk400', 'us': 'us1500'}
    ls_factor = []
    for country, code in factor_indices.items():
        df = get_factor_single(country, code)
        ls_factor.append(df)
    factors = pd.concat(ls_factor)
    return factors


def get_factor_single_ls(country, code, side, save_csv=False):
    """
    get factor data (long and short separately) for a given country and unstack the data to upload to database
    """
    data = pd.read_csv(r'.\input\ten_factor_{}_week_{}_5.csv'.format(code, side), index_col=0)
    data.index = pd.to_datetime(data.index)

    # convert dates to Fridays and merge same Fridays together
    if (data.index.weekday != 4).any():
        data.index = pd.Series(list(map(lambda x: x + np.timedelta64(4 - x.weekday(), 'D'), data.index)))
        data = (data + 1).groupby(data.index).prod() - 1

    if save_csv:
        data.to_csv(r'.\input\factor\ten_factor_{}_week_{}_5.csv'.format(code, side))

    check_complete(data)

    # stack data and add other columns
    df = pd.DataFrame(data.stack(), columns=['ret']).reset_index()
    df.columns = ['date', 'code', 'value']
    df['markets'] = country + "_factor"
    df['name'] = df.code.apply(lambda x: mdict[x])
    df['side'] = side[0].upper()
    print("Length: {}, code: {}, side: {}".format(len(data.index), code, side))
    return df


def get_factor_ls():
    """
    get factor data (long and short separately) for all markets
    """
    factor_indices = {'cn': 'cn800', 'hk': 'hk400', 'us': 'us1500'}
    ls_factor_sides = []
    for country, code in factor_indices.items():
        for side in ['long', 'short']:
            df = get_factor_single_ls(country, code, side)
            ls_factor_sides.append(df)
    factors_sides = pd.concat(ls_factor_sides)
    return factors_sides


def upload_return(data):
    """
    upload data to database.
    """
    # re-order columns TODO: sanity check for columns to not existing, only support 'update' mode
    data = data[['date', 'markets', 'code', 'name', 'side', 'value']]

    # reduce upload numbers
    engine = create_engine("mysql+mysqlconnector://infoport:HKaift-123@192.168.2.81/AlternativeData")
    for name, one_group in data.groupby(['markets', 'side']):
        max_date = pd.read_sql("select max(date) from SectorRotationRet where markets = %s and side = %s", engine,
                               params=list(name))
        max_date = max_date.values[0, 0]
        if max_date:
            one_group = one_group[one_group['date'] > max_date]

        try:
            one_group.to_sql("SectorRotationRet", engine, if_exists="append", index=False)
            print(f"(market: {name[0]}, side: {name[1]}) {one_group.shape[0]} rows uploaded. ")
        except:
            print(f"(market: {name[0]}, side: {name[1]}) failed to upload data. ")

    return 0


def get_sector_single(country, level, start_dt, end_dt, con):
    """
    Get sector weekly return data using refinitiv api.
    Note that the time horizon (as defined by start_dt and end_dt)
    must be at least a week to ensure return can be computed
    :param country: "cn", "us" or "hk"
    :param level: "sector" or "industry_group"
    :return:
    """
    # data = pd.read_csv(r'.\input\sector\{}_{}_weekly.csv'.format(country, level), index_col=0)
    # data.index = pd.to_datetime(data.index)

    tickers = ric_dict[f'{country}_{level}']
    data = con.get_api(tickers, start_dt, end_dt)
    data = data.pct_change().dropna(how='all')

    check_complete(data)

    df = data.stack().reset_index()
    df.columns = ['date', 'code', 'value']
    df['markets'] = "_".join([country, level])
    df['name'] = df.code.apply(lambda x: mdict[x])
    df['side'] = 'NA'
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_sector(level, start_dt, end_dt, con):
    """
    get aggregate index return of a certain level
    :param level: "sector" or "industry_group"
    :return:
    """
    sector_indices = ['cn', 'hk', 'us']
    ls_sector = []
    for country in sector_indices:
        df = get_sector_single(country, level, start_dt, end_dt, con)
        ls_sector.append(df)

    sectors = pd.concat(ls_sector)
    return sectors


def get_exog(start_dt, end_dt, con):
    """
    get exogenous data from api
    """
    # data = pd.read_csv(r'.\input\factor\exogs_weekly.csv', index_col=0)
    tickers = ric_dict['exogs']
    data = con.get_api(tickers, start_dt, end_dt, con)

    cols_pct = [col for col in data.columns if col not in ['US10YT=RR']]
    if 'US10YT=RR' in data.columns:
        data['US10YT=RR'] = data['US10YT=RR'] / 100
    data[cols_pct] = data[cols_pct].pct_change()
    data = data.dropna(how='all')

    check_complete(data)

    df = data.dropna().unstack().reset_index()
    df.columns = ['code', 'date', 'value']
    df['name'] = df.code.apply(lambda x: mdict[x])
    df['markets'] = 'common_exogs'
    df['side'] = 'NA'
    df['date'] = pd.to_datetime(df['date'])
    return df


def save_local(data, file_name):
    """
    combine new data with data in local file
    :param data: dataframe matching the structure in local file
    :param file_name:
    :return:
    """
    data = data.pivot(index='date', columns='code', values='value')
    if os.path.exists(file_name):
        old_data = pd.read_csv(file_name, index_col=0)
        old_data.index = pd.to_datetime(old_data.index, infer_datetime_format=True)
        data = pd.concat([old_data, data])
        data = data.drop_duplicates().sort_index()

    try:
        print(data.tail(10))
        if input("Save file? ") == 'Y':
            data.to_csv(file_name)
            print("File saved. ")
    except PermissionError:
        print("The destination file is open. Please close the file.")


def new_code_name(df):
    """
    upload new code-name mapping pairs to database and local file
    :param df: two columns, with codes on the left and names on the right
    :return:
    """
    df.columns = ['code', 'name']

    # save to database
    engine = create_engine("mysql+mysqlconnector://infoport:HKaift-123@192.168.2.81/AlternativeData")
    df.to_sql("SecRotMapping", engine, if_exists="append", index=False)

    # save to local file
    allcodes = pd.read_csv(r'.\input\code_mapping.csv')
    allcodes = pd.concat([allcodes, df]).drop_duplicates()
    allcodes.to_csv(r'.\input\code_mapping.csv', index=False)


if __name__ == '__main__':
    # the codes below are for testing purpose
    # sectors = get_sector()
    # factors = get_factor()
    # factors_ls = get_factor_ls()
    # exogs = get_exog()
    #
    # sectors = sectors[sectors.date <= pd.to_datetime('2022-03-20')]
    # factors = factors[factors.date <= pd.to_datetime('2022-03-20')]
    # factors_ls = factors_ls[factors_ls.date <= pd.to_datetime('2022-03-20')]
    # exogs = exogs[exogs.date <= pd.to_datetime('2022-03-20')]
    #
    # data = pd.concat([sectors, factors, factors_ls, exogs])
    # upload_return(data)

    con = api_connector()
    data = get_sector_single('us', 'industry_group', '2021-05-01', '2022-06-17', con)
    upload_return(data)
    print(data.head(5))
    print(data.tail(5))
    con.print_counter()