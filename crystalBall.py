from matplotlib.pyplot import yscale
from stockbot import stockbot
from flask import Flask, send_from_directory,render_template
from data import Get_num_df_cols
from stockList import getMyStocks,getStocks
import os
from result import result
import datetime
import jsonpickle   
import traceback 
import data
app = Flask(__name__)
mybot = stockbot(Get_num_df_cols(getMyStocks()))
cache = {}
mybot.load(getMyStocks(),trainMissing=False)
predictions =[]
@app.route('/')
def default():
    global predictions
    if len(predictions) == 0:
        buildPredictions()
        
    stocks  = getMyStocks()
    return render_template('homepage.html',predictions = predictions,myStocks =    stocks,evals =mybot.evals)
@app.route('/<symbol>/')
def infer(symbol):
    if cache[symbol] != None:
        result = cache[symbol]
    else :
        result = mybot.predictFuture(symbol)
        cache[symbol] = result
    return render_template('predict.html',result = result,myStocks =   getMyStocks())
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/graphs/<symbol>/')
def getGraphs(symbol):
    return send_from_directory(os.path.join(app.root_path, 'static'),"{}_{}.png".format(datetime.date.today(), symbol))

@app.route('/api/precache/')
def precache():
    for stocks in getMyStocks():
        _ = infer(stock)
    return "done"

def buildPredictions():
    global predictions
    # try:
    #     with open("predictions/{}predictions.json".format(datetime.date.today()), "r") as text_file:
    #         predictions = jsonpickle.decode( text_file.readline())
    #         assert len(predictions) >= 1 , "no predictions to load."
    # except:
    traceback.print_exc() 
    
    print("failed to load cached predictions rebuilding. This will take a while.")
    for stock in getStocks():
        try:
            today = datetime.date.today()
            r = mybot.predict(stock)
           
        except Exception as e:
            traceback.print_exc() 
            print("failed to predict on {} with error {}".format(stock,e))
            continue
        thisResult = result(today,stock,r )
        print(thisResult)
        predictions.append(thisResult)
        
    predictions.sort(key=lambda x: x.theScaler, reverse=True)
    predictionsJson  = jsonpickle.encode(predictions)
    with open("predictions/{}predictions.json".format(datetime.date.today()), "w") as text_file:
        text_file.write("{}".format(predictionsJson))


for stock in getMyStocks():
    cache[stock] = None

if __name__ == '__main__':
    buildPredictions()
    app.run(host='0.0.0.0',port=8080)