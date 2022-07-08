import pandas as pd
import pymongo
from input_weekly_update import mdict


def read_weighting_data(code, model, group, appetite):
    data = pd.read_csv(rf".\weights\{code}_{group}_{model}e.csv")

    data = data[data.gamma_op == appetite]
    del data['gamma_op']
    names = list(map(lambda x: mdict[x], data.columns[1:]))  # map column name to its industry name
    data.columns = ['date'] + names

    ls_weight = data.to_dict('records')  # turn dataframe into a list of dictionaries
    ls_params = {'market': code, 'timeframe': 5, 'model': model, 'appetite': appetite,
                 'level': group, 'rebalance': 'monthly'} #TODO: could modify 'rebalance' and 'timeframe' later

    for d in ls_weight:
        d['params'] = ls_params # add params as dictionary into each one

    return ls_weight


def read_return_data(code, model, group, appetite):
    data = pd.read_csv(rf".\performances\{code}_{group}_{model}e.csv")

    data = data[data.gamma_op == appetite]
    del data['gamma_op']
    data.columns = ['date', 'return']

    ls_weight = data.to_dict('records')
    ls_params = {'market': code, 'timeframe': 5, 'model': model, 'appetite': appetite,
                 'level': group, 'rebalance': 'monthly'}

    for d in ls_weight:
        d['params'] = ls_params

    return ls_weight


def upload_mongo(data_dict, coll_name):
    """upload data_dict (a list of dictionaries) to mongo table: coll_name"""
    myclient = pymongo.MongoClient("mongodb://app_developer:hkaift123@192.168.2.85:4010/")
    db = myclient["app_data"]
    coll = db[coll_name]
    try:
        result = coll.insert_many(data_dict)
        print(f"Uploaded {len(data_dict)} into [{coll_name}]")
    except Exception as e:
        print(e)
        print(f"Failed to inserting value into [{coll_name}]")


if __name__ == '__main__':
    codes = ['us', 'cn', 'hk']
    models = ['capm', 'factor', 'factorx']
    classification = ['sector', 'industry_group']
    appetite = [2, 8]

    for code in codes:
        for model in models:
            for level in classification:
                for gamma in appetite:

                    if code == 'cn' and level == 'industry_group': # skip cn_industry_group
                        continue

                    print(f"---- {code} {model} {level} {gamma}")

                    # upload sector portfolio return
                    return_ls = read_return_data(code, model, level, gamma)
                    upload_mongo(return_ls, "sector_rotation_return")

                    # upload weightings
                    weight_ls = read_weighting_data(code, model, level, gamma)

                    print(f"---------------------------------")