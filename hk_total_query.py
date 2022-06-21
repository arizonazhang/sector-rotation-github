from utils import connectDB
import pandas as pd
import numpy as np

query = """
select IndexCode, TradingDay, s.EngName, ClosePrice from jydb.QT_OSIndexQuote m
left join jydb.SecuMain s on s.InnerCode = m.IndexCode
where IndexCode in (1151196, 1151198, 1151200, 1151201, 1151202, 1151204, 1151205, 1151208, 1151211, 1151213, 1228492, 1228493, 1228494);
"""
cnx = connectDB()
data = pd.read_sql(query, cnx)

data = data.pivot(index="TradingDay", columns="IndexCode", values="ClosePrice")
data.to_csv(".\input\hsci_total_sectors.csv")

