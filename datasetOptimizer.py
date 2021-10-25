from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
import glob
import numpy as np
import shutil
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
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

#use model to inference each sample in inpath and save all the low confidence samples to outpath
#def optimizeDataset(model,inpath,outpath= "/tmp/optimizedDataset/",threshold=0.95):

#use model to inference each sample in inpath and save all the low confidence samples to outpath
def collectAllErrorsDataset(model,inpath,expectedClass,outpath= "/tmp/optimizedDataset/",threshold=0.95):
    ensure_dir(outpath)
    ensure_dir(outpath+"/fm/")
    ensure_dir(outpath+"/product/")
    files = glob.glob(inpath + "/*.png")
    l = len(files)
    print(f"the len of files is {l}")
    i =0
    for filepath in files:
        i  += 1 
        if i % 100 == 0:
            print(f"{i}/{l} = {i/l}")
        img = image.load_img(filepath,target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)        
        
        pred = model.predict(img)
        
        logit = pred[0][0]
        # print(logit)
        if logit <= 0.5:#the ai thinks it is fm
            if expectedClass != "fm":
                print("The Ai thinks {} is fm but it is in the {} dir".format(filepath,expectedClass))
                
                try:
                    shutil.move(filepath, outpath+"/product/")
                except Exception as err:
                    print(err)
                                
             
            
        else:#it is product
            if expectedClass != "product":
                print("The Ai thinks {} is product but it is in the {} dir".format(filepath,expectedClass))
                try:
                    shutil.move(filepath, outpath+"/fm/")
                except Exception as err:
                    print(err)
def collectAllErrorsProduct(product):
    sensor = "laser"
    model = tf.keras.models.load_model("/data/{}/{}/model/".format(product,sensor))
    # # collectAllErrorsDataset(model,"/data/S004938SkinOff/camera/que/fm/","fm",outpath= "/data/S004938SkinOff/camera/error/")
    collectAllErrorsDataset(model,"/data/{}/{}/train/fm/".format(product,sensor),"fm",outpath= "/data/mega/laser/error/")
    collectAllErrorsDataset(model,"/data/{}/{}/train/product/".format(product,sensor),"product",outpath= "/data/mega/laser/error/")
    collectAllErrorsDataset(model,"/data/{}/{}/validation/fm/".format(product,sensor),"fm",outpath= "/data/mega/laser/error/")
    collectAllErrorsDataset(model,"/data/{}/{}/validation/product/".format(product,sensor),"product",outpath= "/data/mega/laser/error/")
    sensor = "camera"
    model = tf.keras.models.load_model("/data/{}/{}/model/".format(product,sensor))
    collectAllErrorsDataset(model,"/data/{}/{}/train/fm/".format(product,sensor),"fm",outpath= "/data/mega/camera/error/")
    collectAllErrorsDataset(model,"/data/{}/{}/train/product/".format(product,sensor),"product",outpath= "/data/mega/camera/error/")
    collectAllErrorsDataset(model,"/data/{}/{}/validation/fm/".format(product,sensor),"fm",outpath= "/data/mega/camera/error/")
    collectAllErrorsDataset(model,"/data/{}/{}/validation/product/".format(product,sensor),"product",outpath= "/data/mega/camera/error/")
# collectAllErrorsProduct("mega")


def polyClassErrorSearch(path = "/data/polyclass/camera/"):
    IMG_SIZE = (224, 224)
    model = tf.keras.models.load_model(f"{path}/model/")
    test_datagen = ImageDataGenerator()
    gen = test_datagen.flow_from_directory(f"{path}/train/",target_size=(224,224),shuffle=False)
    # data = tf.keras(f"{path}/train/",label_mode="categorical" ,labels="inferred", shuffle=False, batch_size=1, image_size=IMG_SIZE)
    # val = model.evaluate(train_dataset)
    
    paths = gen.filenames
    classes = gen.classes
    classNames = list(gen.class_indices.keys())
    preds  = model.predict(gen)
    for n in classNames:
        try:
            os.makedirs(path+"/error/"+n)
        except :
            _ = 0
    print(classes, classNames)
    c = 0
    w = 0
    for s in zip(paths,preds,classes):
        p = s[1]
        
        clas = s[0].split("/")[-2]
        # print(clas)
        i = np.argmax(p)
        if i == s[2]:
            c += 1
            
        else:
            print(f"path = {s[0]},pred = {p}, the Ai thinks that is a {classNames[i]},{i},{p[i]} but the dir thinks it is a {classNames[s[2]]},{s[2]},{p[s[2]]}\n")
            shutil.move(path+"train/" + s[0],f"/data/polyclass/camera/error/{classNames[s[2]]}")
            w += 1
    print(f"the score was c/w = {c}/{w+c} = {c/(w+c)}")
            
    # print(paths)
    # print(pred.shape)
# polyClassErrorSearch()