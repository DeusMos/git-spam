import stockbot
import data
import stockList
stocks = stockList.getStocks()
sb = stockbot.stockbot(num_df_cols=data.Get_num_df_cols(stocks))
sb.load()
sb.trainOnStocks()
from account import Account
class strategy(object):
    def __init__(self) -> None:
        self.account = Account(5000.00,"2015-01-1")
        super().__init__()
    def run(self):
        
        for stock in self.account.Stocks:
            

