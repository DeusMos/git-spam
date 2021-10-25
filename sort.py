from tensorflow.keras.preprocessing import image
import tensorflow as tf
import glob
import numpy as np
import shutil
from tensorflow.keras.preprocessing import image_dataset_from_directory
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
def sort(model,inglob,outpath= "/tmp/dataset/",threshold=0.95):
    ensure_dir(outpath)
    ensure_dir(outpath+"/fm/")
    ensure_dir(outpath+"/product/")
    files = glob.glob(inglob,recursive=True)
    print(f"the len of files is {len(files)}")
    l = len(files)
    i = 0
    for filepath in files:
        i += 1 
        if i % 100 == 0:
            print(f"{i}/{l}")
        try:
            img = image.load_img(filepath,target_size=(224,224))
            img = image.img_to_array(img)
            img = np.expand_dims(img,axis=0)              
            pred = model.predict(img)        
            logit = pred[0][0]
            # print(logit)
            if logit <= 0:#the ai thinks it is fm
                
                try:
                    shutil.copy(filepath, outpath+"/fm/")
                except OSError as err:
                    print(err) 
                                    
                
                
            else:#it is product
                try:
                    shutil.copy(filepath, outpath+"/product/")
                except OSError as err:
                    print(err)
        except Exception as err:
            print(err)    
            
product = "CrissCuts"
sensor = "camera"
model = tf.keras.models.load_model("/data/{}/{}/model/0643/".format(product,sensor))
sort(model,"/data/products/**/camera/**/*.png",outpath= "/data/mega/")

