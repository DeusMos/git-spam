from flask import Flask, app, request, render_template,send_from_directory
import glob

classifications = ['product','fm']
@app.route('/view/<product>/<sensor>/<classification>/')
def view(product,sensor,classification):
    g = "/data/{}/{}/que/{}/*.png".format(product,sensor,classification)
    paths = glob(g)

    urls = []
    sampleIDs = []
    for p in paths:        
        urls.append(p.replace("/data/","/"))
        sampleIDs.append(p.split("/")[-1])

    print(g,paths)
    if classification == classifications[0]:#set the default to the first class unless thats the class we are showing
        trgClass = classifications[1]
    else:
        trgClass = classifications[0]
    return render_template("view.html",paths = urls,product = product,sensor = sensor,srcClass= classification, sampleIDs=sampleIDs, trgClass = trgClass,zip = zip)

@app.route("/view/")