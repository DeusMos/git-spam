from pandas_datareader import data
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import datetime
import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pandas
jsonpickle_pandas.register_handlers()
# from yahoo_fin.stock_info import get_data, tickers_sp500, tickers_nasdaq, tickers_other, get_quote_table
from yahoo_fin import stock_info
def Get1Data(stock):
    end = datetime.datetime.now()
    start = end -datetime.timedelta(days=(5*365))
    df = stock_info.get_data(stock,start_date=start,end_date= end,index_as_date=False)
    df=df.reset_index()
    df['symbol'] = stock
    df['begins_at'] = df['date']
    df['close_price']=df['close']
    df['high_price']=df['high']
    df['interpolated'] = False
    df['low_price']=df['low']
    df['open_price']=df['open']
    df['session'] = 'reg'
    df['volume']=df['volume']
    del(df['date'])
    del(df['high'])
    del(df['low'])
    del(df['open'])
    del(df['close'])
    del(df['index'])
    del(df['adjclose'])
    del(df['ticker'])
    # print(df)
    for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume'):
        df[col] = df[col].astype(float)
        try:
            df[col] = signal.detrend(df[col])
        except:
            continue
    df['mean_close_price_2'] = df['close_price'].rolling(window=2).mean()
    df['mean_close_price_3'] = df['close_price'].rolling(window=3).mean()
    df['std_close_price_2'] = df['close_price'].rolling(window=2).std()
    df['std_close_price_3'] = df['close_price'].rolling(window=3).std()
    df['tomo_gain'] = df['close_price'].shift(-1) - df['close_price']
    df['yday_gain'] = df['tomo_gain'].shift(1)
    as_date = df['begins_at'].dt
    df['dayofweek'] = as_date.dayofweek
    df['quarter'] = as_date.quarter
    df['weekofyear'] = as_date.weekofyear
    df = df.drop(['begins_at', 'interpolated', 'session'], axis=1)
    # df = df.dropna(axis=0) # Due to window, first two rows now contain nans
    # df = df.iloc[2: , :]
    df.fillna(0)
    # df = df.reset_index(drop=True)
    # outliers = abs(df[df['symbol'] == stock]['tomo_gain']) < df[df['symbol'] == stock]['tomo_gain'].std() * 50
    # df[df['symbol'] == stock] = df[df['symbol'] == stock].loc[:, :][outliers]

    # df = df.drop(df.index[len(df[df['symbol'] == stock]) - 1], axis=0)
    # outliers = abs(df[df['symbol'] == stock]['tomo_gain']) < df[df['symbol'] == stock]['tomo_gain'].std() * 50
    # df[df['symbol'] == stock] = df[df['symbol'] == stock].loc[:, :][outliers]
    
    # df = df.drop(df[df['symbol'] == stock].iloc[-1].name) # get rid of last because next is a different stock
    X_scalers, y_scalers = loadScalers()
    for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume', 'mean_close_price_2', \
            'mean_close_price_3', 'std_close_price_2', 'std_close_price_3', 'yday_gain'):
        pre_x = df[col]
        df[col] = X_scalers[col].fit_transform(pre_x.values.reshape(-1,1))
    pre_y = df['tomo_gain'].values
    df.loc['tomo_gain'] = y_scalers.fit_transform(pre_y.reshape(-1, 1)).reshape(-1)
    df.fillna(0)
    return df 
def GetData(stocks,start = None,end = None):
    if None == end:
        end = datetime.datetime.now()
    if None == start:
        start = end -datetime.timedelta(days=(5*365))
        # start = end -datetime.timedelta(days=(30))
    dfs = []
    for symbol in stocks:
        df = {}
        try:
            with open("data/{}_{}.json".format(symbol, datetime.date.today()), "r") as jsonfile:
                df = jsonpickle.decode( jsonfile.readline())
                assert len(df) >= 1, "no predictions on disk" 
        except:

            print("cached data not found loading data for {}".format(symbol))
            try:
                df = stock_info.get_data(symbol,start_date=start,end_date= end,index_as_date=False)
                # df = data.DataReader(symbol,'yahoo',start,end)

            except:
                print("got no data for {} ".format(symbol))
                continue
            df=df.reset_index()
            df['symbol'] = symbol
            df['begins_at'] = df['date']
            df['close_price']=df['close']
            df['high_price']=df['high']
            df['interpolated'] = False
            df['low_price']=df['low']
            df['open_price']=df['open']
            df['session'] = 'reg'
            df['volume']=df['volume']
            # del(df['date'])
            del(df['high'])
            del(df['low'])
            del(df['open'])
            del(df['close'])
            del(df['index'])
            del(df['adjclose'])
            del(df['ticker'])

            with open("data/{}_{}.json".format(symbol, datetime.date.today()), "w+") as text_file:
                text_file.write("{}".format(jsonpickle.encode( df)))    
        dfs.append(df)
    df = pd.concat(dfs)
    X_scalers = {}
    y_scalers = {}
    for stock in stocks:
        for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume'):
            df[col] = df[col].astype(float)
            try:
                df.loc[df['symbol'] == stock, col] = signal.detrend(df[df['symbol'] == stock][col])
            except:
                continue
        df.loc[df['symbol'] == stock, 'mean_close_price_2'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=2).mean()
        df.loc[df['symbol'] == stock, 'mean_close_price_3'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=3).mean()
        df.loc[df['symbol'] == stock, 'std_close_price_2'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=2).std()
        df.loc[df['symbol'] == stock, 'std_close_price_3'] = df.loc[df['symbol'] == stock, 'close_price'].rolling(window=3).std()
    X_scalers = {stock:{} for stock in stocks}
    y_scalers = {}
    df['tomo_gain'] = df['close_price'].shift(-1) - df['close_price']
    df['yday_gain'] = df['tomo_gain'].shift(1)
    as_date = df['begins_at'].dt
    df['dayofweek'] = as_date.dayofweek
    df['quarter'] = as_date.quarter
    df['weekofyear'] = as_date.weekofyear
    df = df.drop(['begins_at', 'interpolated', 'session'], axis=1)
    df = df.dropna(axis=0) # Due to window, first two rows now contain nans
    df = df.reset_index(drop=True)
    for stock in stocks:
        df = df.drop(df.index[len(df[df['symbol'] == stock]) - 1], axis=0)
        outliers = abs(df[df['symbol'] == stock]['tomo_gain']) < df[df['symbol'] == stock]['tomo_gain'].std() * 50
        df[df['symbol'] == stock] = df[df['symbol'] == stock].loc[:, :][outliers]
        try:
            df = df.drop(df[df['symbol'] == stock].iloc[-1].name) # get rid of last because next is a different stock
        except:
            print("failed to drop!",stock)
            print(df)
            continue
        pre_y = df[df['symbol'] == stock]['tomo_gain'].values
        y_scalers[stock] = make_pipeline(StandardScaler(), MinMaxScaler(feature_range=(-1, 1)))
        for col in ('close_price', 'high_price', 'low_price', 'open_price', 'volume', 'mean_close_price_2', \
                'mean_close_price_3', 'std_close_price_2', 'std_close_price_3', 'yday_gain'):
            pre_x = df[df['symbol'] == stock][col]
            X_scalers[stock][col] = make_pipeline(StandardScaler(), MinMaxScaler(feature_range=(-1, 1)))
            df.loc[df['symbol'] == stock, col] = X_scalers[stock][col].fit_transform(pre_x.values.reshape(-1,1))
        df.loc[df['symbol'] == stock, 'tomo_gain'] = y_scalers[stock].fit_transform(pre_y.reshape(-1, 1)).reshape(-1)
    pickle.dump(X_scalers, open('x_scalers.pkl', 'wb'))
    pickle.dump(y_scalers, open('y_scalers.pkl', 'wb'))
    df = df.dropna(axis=0)
    
    
    return df , X_scalers , y_scalers
def Get_num_df_cols(stocks,start = None,end = None):
    return 13 + len(stocks)
   
def loadScalers():
    X_scalers=  pickle.load(open('x_scalers.pkl', 'rb'))
    y_scalers= pickle.load(open('y_scalers.pkl', 'rb'))
    return X_scalers,y_scalers