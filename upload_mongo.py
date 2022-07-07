import pandas as pd
import numpy as np
import scipy.stats
import mysql.connector
from sqlalchemy import create_engine
import pymongo
from utils_db import mdict

def read_data(code, model, group, appetite):
    data = pd.read_csv(rf".\weights\weekly\{code}_{group}_{model}e.csv")
    names = list(map(lambda x: mdict[x], data.columns[1:-2]))
    data.columns = ['date'] + names + ['']
    data = data[data.weight == 1]
    data = data.drop(columns="weight")
    data['market'] = code
    data['model'] = model

    names.extend(["date"])

    ls_weight = data[names].to_dict('records')
    ls_params = data[['market', 'timeframe', 'model', 'appetite']].to_dict('records')

    for i, d in enumerate(ls_weight):
        d['params'] = ls_params[i]

codes = ['us1500', 'cn800', 'hk400']
models = ['capm', 'factor', 'factorx'] #hist?
classification = ['sector', 'industry_group']
appetite = [2, 8]
# adjust industry codes
for code in codes:
    for model in models:
        for group in classification:
            for gamma in appetite:

                    data = pd.read_csv(r".\weights\weekly\{}_quadprog_{}e.csv".format(code, model))
                    names = list(map(lambda x: mdict[x], data.columns[1:-2]))
                    data.columns = ['date'] + names + list(['appetite','weight'])
                    data = data[data.weight==1]
                    data = data.drop(columns="weight")
                    data['market'] = 'us'
                    data['timeframe'] = 5
                    data['model'] = 'capm'

                    names.extend(["date"])

                    ls_weight = data[names].to_dict('records')
                    ls_params = data[['market', 'timeframe', 'model', 'appetite']].to_dict('records')

                    for i, d in enumerate(ls_weight):
                        d['params'] = ls_params[i]




myclient = pymongo.MongoClient("mongodb://app_developer:hkaift123@192.168.2.85:4010/")
db = myclient["app_data"]
coll = db["sector_allocation"]
result = coll.insert_many(ls_weight)