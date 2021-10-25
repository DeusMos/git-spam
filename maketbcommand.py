import stockList
command = "tensorboard --logdir_spec "
for stock in stockList.getStocks():
    command = "{}{}".format(command,"{}:logs/fit/{}/,".format(stock,stock))
print(command)
