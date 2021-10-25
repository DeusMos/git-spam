# import stockbot
# import data
# import stockList
# stocks = stockList.getStocks()
# sb = stockbot.stockbot(num_df_cols=data.Get_num_df_cols(stocks))
# sb.load(stocks)
# # sb.trainOnMarket(test=True)#This will train a new market model this will take a long time! It is the Ai learning how stocks work in general.
# sb.trainOnStocks(test=True)#this cross trains a new model for each symbol. Think of this as the Ai learing that how that one companies stock works.
# for stock in stockList.myStocks:
#     print("Stock {}, pred: {}".format(stock, sb.predictFuture(stock)))
import stockList
from yahoo_fin import stock_info
import datetime
import numpy as np
import pandas as pd

import data

def pad_stock(symbol,stocks,dummies):
    dumdums = np.zeros(len(stocks))
    dumdums[list(dummies.columns.values).index(symbol)] = 1.
    return dumdums

stocks = stockList.getStocks()
pf = data.GetData(stocks)

stock = "ABNB"
end = datetime.datetime.now()
start = end -datetime.timedelta(days=(5*365))
# df = stock_info.get_data(stock,start_date=start,end_date= end,index_as_date=False)
# print(df)

# print(df)
# df = data.GetData(["ABNB"])
# print(df.keys())
df = data.Get1Data(stock)
today = df[0].iloc[-1].drop(['tomo_gain', 'symbol'])
dummies = pd.get_dummies(df[0]['symbol'], columns=['symbol'])
print(dummies)
# print(df)