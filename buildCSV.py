import stockbot
import data
import stockList
import datetime
from result import result
stocks = stockList.getStocks()
sb = stockbot.stockbot(num_df_cols=data.Get_num_df_cols(stocks))
sb.load(stocks)


myResults = []

for stock in stockList.myStocks:
    try:
        today = datetime.date.today()
        result = sb.predict(stock)
    except:
        print("failed to predict on {}".format(stock))
        continue
    thisResult = result(today,stock,result)
    print(thisResult)
    myResults.append(thisResult)
    
myResults.sort(key=lambda x: x.theScaler, reverse=True)
        
with open('pred.txt', 'a') as file:
    for r in myResults :
        file.write("{}\n".format(r))