class Stock(object):
    def __init__(self,symbol) -> None:
        self.Symbol = symbol
        self.quantity = 0
        self.cost = 0 #how much we have paid for our quantity
        self.value = 0
        super().__init__()
    def buy(self,dollarAmount):
        self.price = self.findCurrentPrice()
        self.cost += dollarAmount
        self.quantity += dollarAmount/self.price
        self.value = self.quantity * self.price
    def sell(self,dollarAmount):
        self.price = self.findCurrentPrice()
        self.cost -= dollarAmount
        self.quantity -= dollarAmount/self.price
    def findCurrentPrice(self):
        return 1 # todo get the real price.
    