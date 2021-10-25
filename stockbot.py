from os import execle, environ

environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
import datetime
import pandas as pd
from tensorflow.python.training.tracking.util import Checkpoint
import data
import stockList
from sklearn.model_selection import train_test_split
import traceback 
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback , ModelCheckpoint, EarlyStopping
def pad_stocks(symbol,stocks,dummies):
    dumdums = np.zeros(len(stocks))
    dumdums[list(dummies.columns.values).index(symbol)] = 1.
    return dumdums
def pad_stock(symbol,stocks,dummies):
    dumdums = np.zeros(len(stocks))
    dumdums[0] = 1.
    return dumdums
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='mse', value=0.001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # if current is None:
        #     warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True



class stockbot(object):
    models = {} 
    evals= {}
    num_df_cols = "0"
    def __init__(self,num_df_cols):
        self.num_df_cols = num_df_cols  
        
        self.models['market'] = tf.keras.Sequential()
        self.models['market'].add(tf.keras.layers.Dense(num_df_cols, input_shape=(1, num_df_cols)))
        self.models['market'].add(tf.keras.layers.LSTM(128,return_sequences = True))
        self.models['market'].add(tf.keras.layers.Dense(512, activation='sigmoid'))
        self.models['market'].add(tf.keras.layers.Dropout(0.1))
        self.models['market'].add(tf.keras.layers.LSTM(128,return_sequences = True))
        self.models['market'].add(tf.keras.layers.Dense(512, activation='sigmoid'))
        self.models['market'].add(tf.keras.layers.Dropout(0.1))
        # self.models['market'].add(tf.keras.layers.LSTM(64,return_sequences = True))
        # self.models['market'].add(tf.keras.layers.Dense(512, activation='sigmoid'))
        # self.models['market'].add(tf.keras.layers.Dropout(0.1))
        self.models['market'].add(tf.keras.layers.LSTM(128))
        self.models['market'].add(tf.keras.layers.Flatten())
        self.models['market'].add(tf.keras.layers.Dense(512, activation='sigmoid'))
        self.models['market'].add(tf.keras.layers.Dropout(0.4))
        self.models['market'].add(tf.keras.layers.Dense(128, activation='sigmoid'))
        self.models['market'].add(tf.keras.layers.Dense(1)) # dont squash output gradient
        self.models['market'].compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    def load(self,stocks= stockList.getStocks(),trainMissing = True):
        self.stocks= stocks
        self.models['market'] = tf.keras.models.load_model('market_model.h5')
        for stock in self.stocks:
            try:
                self.models[stock] = tf.keras.models.load_model("bots/{}.h5".format(stock))
            except:
                traceback.print_exc() 
                print("===========================================================Missing model for {}".format(stock))
                if trainMissing:
                    self.trainOnStocks([stock])

    def trainOnMarket(self,getStocks=stockList.getStocks,startTime = None,endTime = None,test = False):
        self.stocks= getStocks()
        
        if None == endTime:
            endTime = datetime.datetime.now()
        if None == startTime:
            startTime = endTime-datetime.timedelta(days=(5*365))
        df,_,_ = data.GetData(self.stocks,start=startTime,end=endTime)
        X = df.drop(['tomo_gain', 'symbol'], axis=1)
        print(X.keys())
        y = df['tomo_gain']
        dummies = pd.get_dummies(df[0]['symbol'], columns=['symbol'])
        X = np.append(X, dummies.values, axis=1)
        X = np.reshape(X, (-1, 1, self.num_df_cols))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        numEpochs = 12000 #1200 = about 1 hours with 100 stocks
        if test:
            numEpochs = 12
        print("training for a {} epochs!".format(numEpochs))
        log_dir = "logs/fit/market/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=100)
        mc = ModelCheckpoint("bots/{}.h5".format("market"), monitor='mse', mode='min', save_best_only=True,verbose=1)
        es = EarlyStopping(monitor='mse', mode='min', verbose=1, patience=100)
        callbacks=[tensorboard_callback,es,mc]
        self.models['market'].fit(X_train, y_train.values.reshape(-1,1), batch_size=1024, epochs=numEpochs, verbose=0, callbacks= callbacks)
        self.models['market'].fit(X_train, y_train.values.reshape(-1,1), batch_size=16, initial_epoch=numEpochs, epochs=1+int(numEpochs /1000 ), verbose=1, callbacks= callbacks)
        self.evals['market'] = self.models['market'].evaluate(X_test, y_test)
        self.models['market'].save('market_model.h5')
        
    def trainOnStocks(self, startTime = None,endTime = None,test=False):
        
        self.stocks= stockList.getStocks()
        # self.load(self.stocks,trainMissing=False)
        if self.models['market'] == None:
            self.models['market'] = tf.keras.models.load_model('bots/market.h5')
        self.models['market'].pop()
        self.models['market'].pop()
        self.models['market'].layers[0].trainable = False
        self.models['market'].layers[3].trainable = False
        self.models['market'].layers[2].trainable = False
        self.models['market'].layers[5].trainable = True
        # self.models['market'].layers[9].trainable = False
        self.models['market'].add(tf.keras.layers.Dense(128, activation='sigmoid'))
        self.models['market'].add(tf.keras.layers.Dense(1))
        stocks_dfs = {}
        specific_models = {}
        df,_,_ = data.GetData(self.stocks,start=startTime,end=endTime)
        for stock in self.stocks:
            stocks_dfs[stock] = df[df['symbol'] == stock]
            self.models[stock] = tf.keras.models.clone_model(self.models['market'])
            self.models[stock].set_weights(self.models['market'].get_weights())
            
            self.models[stock].compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
            print(stock)
        Xes = {}
        ys = {}
        dummies = pd.get_dummies(df['symbol'], columns=['symbol'])
        # toSkip= ['AAL','AAPL','ABNB','ACB','AMC','AMD','AMZN','APHA','AQB','ARKK', 'AZN' , 'BA' , 'BABA' , 'BAC' , 'BLNK' , 'BNGO', "CCL", "CGC", "CPRX"    ]
        print(f"the len of self.stocks is {len(self.stocks)}")
        for stock in self.stocks:
            # if stock in toSkip:
            #     continue
            print(stock)
            try:
                repeated_dummies = pad_stocks(stock,self.stocks,dummies).reshape(1,-1).repeat(len(stocks_dfs[stock]),axis=0)
                Xes[stock] = np.append(stocks_dfs[stock].drop(['tomo_gain', 'symbol'], axis=1).values, repeated_dummies, axis=1)
                Xes[stock] = np.reshape(Xes[stock], (-1, 1, self.num_df_cols))
                ys[stock] = stocks_dfs[stock]['tomo_gain'].values.reshape(-1,1)
                print(f"the Xes have shape = {Xes[stock].shape}")
                print(f"the Ys have shape = {ys[stock].shape}")
                Xes_train, ys_train, Xes_test, ys_test, best_model_scores, best_model = {}, {}, {}, {}, {}, {}
                best_model_scores[stock] = 1
                
                Xes_train[stock] = Xes[stock][:-5]
                ys_train[stock] = ys[stock][:-5]
                Xes_test[stock] = Xes[stock][-5:]
                ys_test[stock] = ys[stock][-5:]
                
                log_dir = "logs/fit/{}/".format(stock) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=100)
                mc = ModelCheckpoint("bots/{}.h5".format(stock), monitor='mse', mode='min', save_best_only=True,verbose=0)
                es = EarlyStopping(monitor='mse', mode='min', patience=2500)
                callbacks=[tensorboard_callback,EarlyStoppingByLossVal(monitor='mse'),es,mc]
                numEpochs = 100000
                self.models[stock].fit(Xes_train[stock], ys_train[stock], batch_size=128, epochs=numEpochs, verbose=1, callbacks = callbacks)
                self.models[stock].fit(Xes_train[stock], ys_train[stock], initial_epoch=numEpochs, batch_size=16,epochs=1+int(numEpochs /100 ), verbose=1,callbacks = callbacks)
                self.evals[stock] = self.models[stock].evaluate(Xes_test[stock], ys_test[stock])[0]
                
                # best_model[stock].save("bots/{}.h5".format(stock))
            except:
                traceback.print_exc() 
                continue

   


    def predictFuture(self,stock):
        df , X_scale, Y_scale= data.GetData(stocks=[stock])
        today = df[df['symbol'] == stock].iloc[-1].drop(['tomo_gain', 'symbol'])
        dummies = pd.get_dummies(df['symbol'], columns=['symbol'])
        today = np.append(today, pad_stocks(stock,self.stocks,dummies=dummies))
        # self.models[stock].reset_states()
        
        pred = self.models[stock].predict(np.reshape(today, (-1, 1, self.num_df_cols)).astype('float32'))
        pred = Y_scale[stock].inverse_transform(pred)
        # print("Stock {}, pred: {}".format(stock, pred.item()))
        result = pred.item()
        direction = ""
        todaysdate = datetime.date.today() 
        thenextdate = todaysdate + datetime.timedelta(days=1)
        if result < 0.0:
            direction = 'lower'
        else:
            direction = 'higher'
        with open('pred.txt', 'a') as file:
            file.write("on {} the {} stock will be {} than today {} with a scaler of, {}\n".format(thenextdate,stock,direction,todaysdate, result))
        return "on {} the {} stock will be {} than today {} with a scaler of, {}\n".format(thenextdate,stock,direction,todaysdate, result)
    def predict(self,stock):
        # fig = plt.figure(figsize=(8, 6), dpi=80)
        X_scalers, y_scalers = data.loadScalers()
        df = data.Get1Data(stock)
        today = df.iloc[-1].drop(['tomo_gain', 'symbol'])
        dummies = pd.get_dummies(df, columns=['symbol'])
        print(dummies)
        today = np.append(today, pad_stock(stock,self.stocks,dummies=dummies))
        pred = self.models[stock].predict(np.reshape(today, (-1, 1, self.num_df_cols)).astype('float32'))
        # ax0 = fig.add_subplot(3,1,1)
        # ax0.plot(df)
        pred = y_scalers[stock].inverse_transform(pred)
        print(pred.item())
        # ax1 = fig.add_subplot(3,1,2)
        # ax1.plot(pred)
        # ax2 = fig.add_subplot(3,1,3)
        # ax2.plot((df[-30:]))
        # plt.subplot(4,1,3) = plt.plot(df[df['symbol'] == stock]['close_price'])
        # fig.savefig("static/{}_{}.png".format(datetime.date.today(), stock))
        return pred.item()
    
    # def GenGraph(self, stock):
    #     fig = plt.figure()
    #     df , X_scale, Y_scale= data.GetData(stocks=[stock])
    #     today = df[df['symbol'] == stock].iloc[-1].drop(['tomo_gain', 'symbol'])
    #     dummies = pd.get_dummies(df['symbol'], columns=['symbol'])
    #     today = np.append(today, pad_stock(stock,self.stocks,dummies=dummies))
    #     pred = self.models[stock].predict(np.reshape(today, (-1, 1, self.num_df_cols)).astype('float32'))
    #     ax1 = fig.add_subplot(2,1,1)
    #     ax1.plot(Y_scale[stock].inverse_transform(pred[-30:]))
    #     ax2 = fig.add_subplot(2,1,2)
    #     ax2.plot(Y_scale[stock].inverse_transform(df[-30:]))
    #     # plt.subplot(4,1,3) = plt.plot(df[df['symbol'] == stock]['close_price'])
    #     fig.savefig("static/{}_{}.png".format(datetime.date.today(), stock))
    #     return fig