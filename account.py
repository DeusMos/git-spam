class Account(object):
    def __init__(self,startFunds,startDate) -> None:
        self.StartFunds = startFunds
        self.CurrentFunds = startFunds
        self.StartDate = startDate
        self.Stocks = []
        super().__init__()