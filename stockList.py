stocks = {}
myStocks = ["TSLA","DIS","LW","SBTX","BNGO","OPK","GSAT","SRPT","SPCE","CPRX","BABA","AMZN","SNE","APHA","GOOGL","MSFT"]
myStocks = ["TSLA","DIS"]
#this is a list of the stocks you care about. that you want individual modles and predictions for. Long lists can take a long time.
stocks = ["HOOD","GOOG","EDU","COIN","AMAT","LW","MDB","SPCE","CPRX","BABA","AMZN","GOOGL","MSFT","NRZ","PLUG","CRM","PTON","NKE","PYPL","FB","GM","KO","V","UBER","ZNGA","TXMD","LI","JNJ","RCL","WMT","DKNG","AZN","JPM","PENN","SNAP","GE","ET","NOK","DAL","LUV","DIS","SIRI","NFLX","NVDA","BAC","AAPL","NIO","OGI","WORK","PFE","SQ","SBUX","ZM","KOS","UAL","SAVE","AMD","BA","NCLH","INTC","T","JBLU","MRO","INO","TSLA","CRON","TWTR","CGC","MGM","AAL","F","ACB","GPRO","TLRY","MFA","CCL","XOM","SNDL","PLTR","FCEL","NKLA","AMC","VOO","HEXO","WKHS","SPY","IDEX","ABNB","GNUS","PSEC","BNGO","IVR","GUSH","ARKK","USO","XPEV","BLNK","QS","SPHD","VTI","AQB","EYES"]
stocks.sort()
myStocks = stocks


#stocks = ["SPCE","CPRX","BABA","AMZN","SNE","APHA","GOOGL","MSFT","NRZ","PLUG","CRM","PTON","NKE","PYPL","FB","GM","KO","V","UBER","MRN","ZNGA","TXMD","LI","JNJ","RCL","WMT","DKNG","AZN","JPM","PENN","SNAP","GE","ET","NOK","DAL","LUV"]
#stocks = ["DIS","SIRI","NFLX","NVDA","BAC","AAPL","NIO","OGI","WORK","PFE","SQ","SBUX","ZM","KOS","UAL","SAVE","AMD","BA","NCLH","INTC","T","JBLU","MRO","INO","TSLA","CRON","TWTR","CGC","MGM","AAL","F","ACB","GPRO"]
#stocks = ["NKE","PYPL","FB"]
#stocks =["GM","KO","V","UBER"]
#stocks = ["MRN"] this one breakes things 
#stocks = ["ZNGA"]
#stocks = ["TXMD","LI"]
#stocks = ["JNJ","RCL","WMT","DKNG","AZN"]
#this is a list of the top 100 stocks from robinhood

def getStocks():
    return stocks
def getMyStocks():
    return myStocks
def addStock(stock):
    stocks.append(stock)
def removeStock(stock):
    stocks.remove(stock)
