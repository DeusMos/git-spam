import os
import shutil
from flask import Flask, app, request, render_template,send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from glob import glob
import time
import train
import tarfile
import json


classifications = ['product','fm'] # change this according to what you've trained your model to do        
models = {}
product = "CrissCuts"
# product = "t"
sensor = "camera"
haveModel = True
if not os.path.isdir("/data/{}/{}/model/".format(product,sensor)):#if we don't have a new model
    if os.path.isdir("/mydata/{}/{}/model/".format(product,sensor)):# and we do have a baked in model
        shutil.move("/mydata/{}/{}/model/".format(product,sensor),"/data/{}/{}/model/".format(product,sensor))
    else:#we don't have a model to load!
        print("can't find a model for {}/{}".format(product,sensor))
        haveModel =False
if haveModel:
    models[sensor] = tf.keras.models.load_model("/data/{}/{}/model/".format(product,sensor))
product = "CrissCuts"
# product = "t"
sensor = "laser"
haveModel = True
if not os.path.isdir("/data/{}/{}/model/".format(product,sensor)):#if we don't have a new model
    if os.path.isdir("/mydata/{}/{}/model/".format(product,sensor)):# and we do have a baked in model
        shutil.move("/mydata/{}/{}/model/".format(product,sensor),"/data/{}/{}/model/".format(product,sensor))
    else:#we don't have a model to load!
        print("can't find a model for {}/{}".format(product,sensor))
        haveModel =False
if haveModel:
    print("loaded model for ","/data/{}/{}/model/".format(product,sensor))
    models[sensor] = tf.keras.models.load_model("/data/{}/{}/model/".format(product,sensor))
IMG_SIZE = (224, 224)

app = Flask(__name__, static_url_path="", static_folder="/data/",template_folder="/template/")
def ensure_dir(file_path):
    import os
    directory = os.path.dirname(file_path)
    path = ""
    parts = directory.split("/")
    for part in parts:
        path += "/"+part
        if not os.path.exists(path):
            print("{} did not exists so it was created.".format(path))
            os.makedirs(path)
@app.route('/infer/<product>/<sensortype>/<timestamp>/', methods=['POST'])
def infer(product, sensortype, timestamp):
    try:
        file = request.files['myFile']
        if file:
            start = time.time()
            ensure_dir("/data/{}/{}/que/".format(product, sensortype))
            productQueDir = "/data/{}/{}/que/".format(product, sensortype)
            filepath = os.path.join(productQueDir, "/sample{}.png".format(timestamp))
            ensure_dir(filepath)
            file.save(filepath)
            file.close()
            img = image.load_img(filepath,target_size=(224,224))
            img = image.img_to_array(img)
            img = np.expand_dims(img,axis=0)        
            
            pred = models[sensortype].predict(img)
            print(pred)
            logit = pred[0][0]
            if logit <= 0:#it is fm                
                
                p = np.exp(logit*-1)/(1+np.exp(logit*-1))
                ret = "{},{}".format(1-p,p)
                print("this sample is {} %{} ({})".format(classifications[1],100*p,ret))
                productFMPath = productQueDir + classifications[1] + "/sample{}.png".format(timestamp)
                ensure_dir(productFMPath)
                try:
                    shutil.move(filepath, productFMPath)
                except:
                    return ret
               # os.symlink(productFMPath,"/data/lastFM.png")
            else:#it is product                
                p = np.exp(logit)/(1+np.exp(logit))
                ret = "{},{}".format(p,1-p)
                print("this sample is {} %{} ({})".format(classifications[0],100*p,ret))
                productPath = productQueDir + classifications[0] +"/sample{}.png".format(timestamp)
                ensure_dir(productPath)
                try:
                    shutil.move(filepath, productPath)
                except:
                    return ret
               # os.symlink(productPath,"/data/lastProduct.png")
            end = time.time()
            print("result = ({}) in {}".format(ret,end-start))
            return ret
    except RuntimeError as e:
        print("Inference has failed with error\n {} \nreturning -1.0,-1.0".format(product, e))
        return "-1.0,-1.0"
@app.route('/view/')
def view():
    products = glob.glob("/data/*"):

    return render_template("view.html",products = products,zip = zip)

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

@app.route('/move/<product>/<sensor>/<srcClass>/<sampleID>/<trgClass>/')
def move(product,sensor,srcClass,sampleID,trgClass):    
    src = "/data/{}/{}/que/{}/{}".format(product,sensor,srcClass,sampleID)
    trg = "/data/{}/{}/que/{}/{}".format(product,sensor,trgClass,sampleID)
    ensure_dir(trg)
    shutil.move(src,trg)
    return view(product,sensor,srcClass)
@app.route('/archive/<product>/<sensor>/<srcClass>/')
def archive(product,sensor,srcClass):#move the files from the que to the training data directory
    ensure_dir("/data/{}/{}/train/{}/".format(product,sensor,srcClass))
    ensure_dir("/data/{}/{}/validation/{}/".format(product,sensor,srcClass))
    for p in glob("/data/{}/{}/que/{}/*.png".format(product,sensor,srcClass)):
        n = p.split(".png")[0][-1]
        if n > 6:#samples that end in 7,8,9 are validation
            shutil.move(p,"/data/{}/{}/validation/{}/{}".format(product,sensor,srcClass,p.split("/")[-1]))
        else:#samples that end in 0-6 are training 
            shutil.move(p,"/data/{}/{}/train/{}/{}".format(product,sensor,srcClass,p.split("/")[-1]))
    return view(product,sensor,srcClass)
@app.route("/retrain/<product>/<sensor>/")
def reTrain(product,sensor):
    accuracy = train.train(product,sensor,new=True)
    return "accuracy = {}".format(accuracy)
@app.route("/tar/<product>/<sensor>/")
def tar(product,sensor):
    path = "/data/{}/{}/*/*/*.png".format(product,sensor)
    paths = glob(path)
    tarpath = f"/data/{product}/{sensor}/{product}_{sensor}.tar.gz"
    tar = tarfile.open(tarpath,"w:gz")
    for p in paths:
        if p[-4:] == ".png":
            tar.add(p)
    tar.close()
    name = f"{product}_{sensor}.tar.gz"
    return send_from_directory(f"/data/{product}/{sensor}",filename=name,as_attachment=True,attachment_filename=name)

@app.route("/tar/<product>/<sensor>/<srcClass>/")
def tarClass(product,sensor,srcClass):
    path = "/data/{}/{}/{}/*.png".format(product,sensor,srcClass)
    paths = glob(path)
    path = "/data/{}/{}/que/{}/*.png".format(product,sensor,srcClass)
    paths.extend(glob(path))
    
    tarpath = f"/data/{product}/{sensor}/{product}_{sensor}_{srcClass}.tar.gz"
    tar = tarfile.open(tarpath,"w:gz")
    for p in paths:
        if p[-4:] == ".png":
            tar.add(p)
            print(f"adding {p}, to tar.")            
        else:
            print(f"did not add {p}")
    tar.close()
    name = f"{product}_{sensor}_{srcClass}.tar.gz"
    return send_from_directory(f"/data/{product}/{sensor}",filename=name,as_attachment=True,attachment_filename=name)

@app.route("/tarque/<product>/<sensor>/<srcClass>/")
def tarQue(product,sensor,srcClass):
    path = "/data/{}/{}/que/{}/*.png".format(product,sensor,srcClass)
    paths = glob(path)
    tarpath = f"/data/{product}/{sensor}/{product}_{sensor}_que.tar.gz"
    tar = tarfile.open(tarpath,"w:gz")
    for p in paths:
        if p[-4:] == ".png":
            tar.add(p)
    tar.close()
    name = f"{product}_{sensor}_que.tar.gz"
    return send_from_directory(f"/data/{product}/{sensor}",filename=name,as_attachment=True,attachment_filename=name)
@app.route("/tararchive/<product>/<sensor>/<srcClass>/")
def tarArchive(product,sensor,srcClass):

    path = "/data/{}/{}/{}/*.png".format(product,sensor,srcClass)
    paths = glob(path)
    tarpath = f"/data/{product}/{sensor}/{product}_{sensor}_archive.tar.gz"
    tar = tarfile.open(tarpath,"w:gz")
    for p in paths:
        if p[-4:] == ".png":
            print(f"adding {p}, to tar.")
            tar.add(p)
        else:
            print(f"did not add {p}")
    tar.close()
    name = f"{product}_{sensor}_archive.tar.gz"
    return send_from_directory(f"/data/{product}/{sensor}/" ,filename=name,as_attachment=True,attachment_filename=name)

@app.route("/count/<product>/")
def count(product):
    cameraString = ""
    laserString = ""
    fmCountCamera = len(glob(f"/data/{product}/camera/que/fm/*.png"))
    productCountCamera = len(glob(f"/data/{product}/camera/que/product/*.png"))
    if fmCountCamera+productCountCamera > 0 :        
        cameraString = f"fmCount = {fmCountCamera} productCount = {productCountCamera} reduction = { productCountCamera/(fmCountCamera+productCountCamera)}"
    fmCountLaser = len(glob(f"/data/{product}/laser/que/fm/*.png"))
    productCountLaser = len(glob(f"/data/{product}/laser/que/product/*.png"))
    if fmCountLaser+productCountLaser > 0:
        laserString = f"fmCount = {fmCountLaser} productCount = {productCountLaser} reduction = { productCountLaser/(fmCountLaser+productCountLaser)}"
    return f"Prodcut = {product}<br> Camera = {cameraString}<br>Laser = {laserString}"

@app.route("/count/")
def countAll():
    products = glob("/data/*/")
    outStr = ""
    for p in products:
        name = p.split("/")[-2]
        outStr += "<br>"+count(name)
    return outStr

import stats
@app.route("/stats/<product>/<sensor>/")
def getStats(product,sensor):
    return json.dumps(stats.Stats(product,sensor).__dict__)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,threaded=False)