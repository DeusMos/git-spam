from glob import glob


class Stats():
    source = ""
    total = 0
    removed = 0
    reduction = 0.0
    version = ""
    sanitationScore = 0
    alive = 0.0

    def __init__(self,product,sensor):
        fmCount = len(glob(f"/data/{product}/{sensor}/que/fm/*.png"))+ len(glob(f"/data/{product}/{sensor}/fm/*.png"))
        productCount = len(glob(f"/data/{product}/{sensor}/que/product/*.png"))+ len(glob(f"/data/{product}/{sensor}/product/*.png"))
        self.source = sensor
        self.total = fmCount + productCount
        self.removed = productCount
        if self.total > 0 :
            self.reduction = (productCount / self.total) * 100
        else:
            self.redcution = 100.0
        self.version = "20210615" #todo look these up.
        self.sanitationScore = 0 #todo look these up.
        self.alive = 0.0 #todo look these up.