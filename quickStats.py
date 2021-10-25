import statistics
class qStat():
    def __init__(self,data):
        self.mean = statistics.mean(data)
        self.median = statistics.median(data)
        self.stdev = statistics.stdev(data)
        self.variance = statistics.variance(data)
        self.distro = statistics.NormalDist.from_samples(data)