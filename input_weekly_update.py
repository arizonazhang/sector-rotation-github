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
import mysql.connector
ek.set_app_key('51dd084a4cea45dab6ac09cdbfe75cd83d17caa8')


class api_connector():
    """
    this class serves as an api connection to ek.
    It also helps to keep track of all the requests made and data points downloaded.
    """
    def __init__(self):
        self.requests = 0
        self.datapoints = 0

    def add_request(self):
        self.requests += 1

    def add_datapoints(self, points):
        self.datapoints += points

    def get_api(self, func, rics, **args):
        """
        :param func: can be ek.get_timeseries or ek.get_data
        :param rics: list of tickers
        :param args: arguments to be put into the func
        :return:
        """
        try:
            data = func(rics, **args)
            if func == ek.get_data:
                data, err = data
                if err:
                    print(err)

                # unstack the dataframe
                data = data.pivot(columns='Instrument', index='Date')
                # remove timezone in the timestamp
                data.index = pd.to_datetime(data.index).tz_localize(None)

            elif func == ek.get_timeseries:
                data.index = pd.to_datetime(data.index)
                data.columns = rics

                # adjust column names
                if len(rics) == 1:
                    data.columns = rics
                if len(data.columns) != len(rics):
                    print("Warning: Some indices not available")

            self.add_request()
            self.add_datapoints(data.shape[0]*data.shape[1])
            return data

        except ValueError:
            print("Error: a parameter type or value is wrong")
        except Exception:
            print("Error: request failed")

    def get_group_data(self, markets, rics, api_func, proc_func, **args):
        """
        :param markets: market code in the SectorRotationRet
        :param rics: list of tickers
        :param api_func: eikon function to use, ek.get_timeseries or ek.get_data
        :param proc_func:  function to process data from Refinitiv, depends on how the data should be processed, i.e. pct_change
        :param args: arguments to pass api function
        :return:
        """
        data = self.get_api(api_func, rics, **args)
        data = proc_func(data)
        check_complete(data)

        # stack the dataframe to fit database format
        df = data.stack().reset_index()
        df.columns = ['date', 'code', 'value']
        df['markets'] = markets
        df['name'] = df.code.apply(lambda x: mdict[x])
        df['side'] = 'NA'
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
        return df

    def print_counter(self):
        """print counter records"""
        print("API download records:")
        print(f"No. of requests made: {self.requests}")
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
            'hk_industry_group': ['.TRXFLDHKTA1', '.TRXFLDHKTE1', '.TRXFLDHKTE2', '.TRXFLDHKTF1',
                                    '.TRXFLDHKTF3', '.TRXFLDHKTF4', '.TRXFLDHKTH1', '.TRXFLDHKTH2',
                                    '.TRXFLDHKTI1', '.TRXFLDHKTC1', '.TRXFLDHKTI2', '.TRXFLDHKTI4',
                                    '.TRXFLDHKTM1',  '.TRXFLDHKTM2', '.TRXFLDHKTM3', '.TRXFLDHKTN1',
                                    '.TRXFLDHKTN2', '.TRXFLDHKTN3', '.TRXFLDHKTT1', '.TRXFLDHKTT2',
                                     '.TRXFLDHKTU1', '.TRXFLDHKTY1',  '.TRXFLDHKTY2', '.TRXFLDHKTY3', '.TRXFLDHKTY4'],
            'common_exogs': ['.dMIEF00000G', '.dMIEA00000G', 'US10YT=RR', '.DXY', 'HIHKD1MD='],
            'common_exogs_monthly': ['.dMIEF00000G', '.dMIEA00000G', 'US10YT=RR', '.DXY', 'HIHKD1MD=']} #TODO: complete the list


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


def get_factor_single(country, code):
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

    df = pd.DataFrame(data.stack()).reset_index()
    df.columns = ['date', 'code', 'value']
    df['markets'] = country + "_factor"
    df['name'] = df.code.apply(lambda x: mdict[x])
    df['side'] = "LS"
    return df


def get_factor_single_ls(country, code, side):
    """
    get factor data (long and short separately) for a given country and unstack the data to upload to database
    """
    data = pd.read_csv(r'.\input\ten_factor_{}_week_{}_5.csv'.format(code, side), index_col=0)
    data.index = pd.to_datetime(data.index)

    # convert dates to Fridays and merge same Fridays together
    if (data.index.weekday != 4).any():
        data.index = pd.Series(list(map(lambda x: x + np.timedelta64(4 - x.weekday(), 'D'), data.index)))
        data = (data + 1).groupby(data.index).prod() - 1

    check_complete(data)

    # stack data and add other columns
    df = pd.DataFrame(data.stack(), columns=['ret']).reset_index()
    df.columns = ['date', 'code', 'value']
    df['markets'] = country + "_factor"
    df['name'] = df.code.apply(lambda x: mdict[x])
    df['side'] = side[0].upper()
    print("Length: {}, code: {}, side: {}".format(len(data.index), code, side))
    return df


def get_factor(save_file=False):
    """
    get factor data for all markets
    """
    factor_indices = {'cn': 'cn800', 'hk': 'hk400', 'us': 'us1500'}
    ls_factor = []
    for country, code in factor_indices.items():
        df = get_factor_single(country, code)
        if save_file:
            save_local(df, rf".\input\factor\ten_factor_vw_{code}_week_5.csv")
        ls_factor.append(df)
    factors = pd.concat(ls_factor)
    return factors


def get_factor_ls(save_file=False):
    """
    get factor data (long and short separately) for all markets
    """
    factor_indices = {'cn': 'cn800', 'hk': 'hk400', 'us': 'us1500'}
    ls_factor_sides = []
    for country, code in factor_indices.items():
        for side in ['long', 'short']:
            df = get_factor_single_ls(country, code, side)
            ls_factor_sides.append(df)
            if save_file:
                save_local(df, rf'.\input\factor\ten_factor_{code}_week_{side}_5.csv')
    factors_sides = pd.concat(ls_factor_sides)
    return factors_sides


def replace_into_mysql(query, table_name, upload_data):
    """run the query to insert/replace data in the table"""
    config = {
        'user': 'infoport', 'password': 'HKaift-123',
        'host': '192.168.2.81', 'database': 'AlternativeData',
        'raise_on_warnings': False
    }

    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    cnt = 0
    for i in range(len(upload_data.index)): # upload the data into the table row by row
        data_query = upload_data.iloc[i,:].to_dict()
        try:
            cursor.execute(query, data_query)
        except Exception as e:
            print(e)
            cnt += 1
    cnx.commit()
    cursor.close()
    cnx.close()
    print(f"Uploaded {len(upload_data.index)-cnt}/{len(upload_data.index)} records into table [{table_name}].")


def upload_data(data):
    query = """
    replace into SectorRotationRet (date, markets, code, name, side, value) values
     (%(date)s, %(markets)s, %(code)s, %(name)s, %(side)s, %(value)s)
     """
    replace_into_mysql(query, "SectorRotationRet", data)


def save_local(data, file_name):
    """
    combine new data with data in local file
    :param data: dataframe with three columns, date, code and value
    :param file_name:
    :return:
    """
    data = data.pivot(index='date', columns='code', values='value')
    data = data.astype(float).round(6)
    data.index = pd.to_datetime(data.index)

    if os.path.exists(file_name):
        old_data = pd.read_csv(file_name, index_col=0)
        old_data.index = pd.to_datetime(old_data.index, infer_datetime_format=True)
        old_data = old_data.astype(float).round(6)
        data = pd.concat([old_data, data])
        data = data.drop_duplicates()
        data = data.sort_index()

    try:
        print(data.tail(10))
        if input("Save file? ") == 'Y':
            data.to_csv(file_name)
            print(f"File ({file_name}) updated. ")
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
    query = """replace into SecRotMapping (code, name) values ( %(code)s, %(name)s)"""
    replace_into_mysql(query, table_name='SecRotMapping', upload_data=df)

    # save to local file
    allsql = pd.read_sql("SELECT * FROM AlternativeData.SecRotMapping;", engine)
    allcsv = pd.read_csv(r'.\input\code_mapping.csv')
    all = pd.concat([allsql, allcsv]).drop_duplicates()
    all.to_csv(r'.\input\code_mapping.csv', index=False)


def weekly_pct_chg(df):
    return df.pct_change().dropna(how='all')


def monthly_pct_chg(df):
    return df.pct_change(4).dropna(how='all')


def weekly_update(end_date):
    def weekly_pct_chg(df):
        return df.pct_change().dropna(how='all')

    def monthly_pct_chg(df):
        return df.pct_change(4).dropna(how='all')

    update_list = ['cn_sector', 'us_sector', 'hk_sector', 'us_industry_group', 'hk_industry_group', 'common_exogs', 'common_exogs_monthly']
    con = api_connector()
    for market in update_list:
        print(f" ----- Updating the list: {market}")
        ric = ric_dict[market]

        # monthly change data
        if market == 'common_exogs_monthly':
            start_date = pd.to_datetime(end_date) - np.timedelta64(28, 'D') # start date is one month ago to get monthly change
            start_date = str(start_date)[:10]
            data = con.get_group_data(market, ric, proc_func=monthly_pct_chg, api_func=ek.get_timeseries,
                               start_date=start_date, end_date=end_date, fields='CLOSE', interval='weekly')
            upload_data(data)
        # weekly change data
        else:
            start_date = pd.to_datetime(end_date) - np.timedelta64(7, 'D') # start date is one week ago to get monthly change
            start_date = str(start_date)[:10]
            data = con.get_group_data(market, ric, proc_func=weekly_pct_chg, api_func=ek.get_timeseries,
                               start_date=start_date, end_date=end_date, fields='CLOSE', interval='weekly')
            upload_data(data)
    con.print_counter()

mdict = get_mapping()

if __name__ == '__main__':
    ## TODO: add scheduler
    end_dt = '2022-07-01'
    weekly_update(end_dt)




