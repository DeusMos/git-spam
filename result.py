class result(object):
    theDate = ""
    theStock = ""
    theScaler = 0.0
    def __init__(self,theDate,theStock,theScaler):
        self.theDate = theDate
        self.theStock = theStock
        self.theScaler = float(theScaler)
    def __str__(self):
        return "{}, {}, {}".format(self.theDate,self.theStock,self.theScaler)