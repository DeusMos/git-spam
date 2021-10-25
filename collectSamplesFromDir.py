import shutil
import os
import glob

def collectSamplesFromDataDir(dataDir):
    listOfcameraProduct = glob.glob(f"{dataDir}/*/camera/**/product/*.png",recursive=True)
    listOflaserProduct = glob.glob(f"{dataDir}/*/laser/**/product/*.png",recursive=True)
    listOfcamerafm = glob.glob(f"{dataDir}/*/camera/**/fm/*.png",recursive=True)
    listOflaserfm = glob.glob(f"{dataDir}/*/laser/**/fm/*.png",recursive=True)
    print(len(listOfcameraProduct))
    print(len(listOflaserProduct))
    print(len(listOfcamerafm))
    print(len(listOflaserfm))
    os.makedirs(f"{dataDir}/all", exist_ok=True)
    os.makedirs(f"{dataDir}/all/camera", exist_ok=True)
    os.makedirs(f"{dataDir}/all/camera/product", exist_ok=True)
    os.makedirs(f"{dataDir}/all/camera/fm", exist_ok=True)
    os.makedirs(f"{dataDir}/all/laser", exist_ok=True)
    os.makedirs(f"{dataDir}/all/laser/product", exist_ok=True)
    os.makedirs(f"{dataDir}/all/laser/fm", exist_ok=True)
    print(f"{dataDir}/all/laser/fm")
    for sample in listOfcameraProduct:
        if sample.__contains__("all"):
            print(f"sample is in all {sample}")
            continue
        
        else:
            shutil.copy(sample,f"{dataDir}/all/camera/product/")
            os.remove(sample)
    
    for sample in listOflaserProduct:
        if sample.__contains__("all"):
            continue
        
        else:
            shutil.copy(sample,f"{dataDir}/all/laser/product/")
            os.remove(sample)
    
    for sample in listOfcamerafm:
        if sample.__contains__("all"):
            continue
        
        else:
            shutil.copy(sample,f"{dataDir}/all/camera/fm/")
            os.remove(sample)
    
    for sample in listOflaserfm:
        if sample.__contains__("all"):
            continue
        
        else:
            shutil.copy(sample,f"{dataDir}/all/laser/fm/")
            os.remove(sample)

collectSamplesFromDataDir("/home/xp/Desktop/new/mega/data")