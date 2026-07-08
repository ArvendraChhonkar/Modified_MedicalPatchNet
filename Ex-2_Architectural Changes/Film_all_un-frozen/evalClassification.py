import imgRetrivalUtil
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision.transforms import v2

#from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utilFunc import readCSV
import pickle
import os

from NetworkModel import ScalePatchNet, FiLMPatchNet

CHEXLOCALIZE_BASE_PATH = "DataSet/CheXlocalize/"
CHEXPERT_PATH = CHEXLOCALIZE_BASE_PATH + "CheXpert-v1.0/"
MAP_OUT_FOLDER = "DataOut2/"

RAW_PATCH_NET = "RawPatchNet"
SCALED_PATCH_NET = "ScaledPatchNet"

PATCH_TYPE = SCALED_PATCH_NET

NORM_VAL = 200

def encodeClassImg(model, device, shiftLoader):
    imgFeatrueList = list()
    shiftTupList = list()

    for shiftImgBatch,shiftTupBatch in shiftLoader:
        shiftImgBatch = shiftImgBatch.to(device)
        if PATCH_TYPE == RAW_PATCH_NET:
            imgFeatures = model.forwardRawPatches(shiftImgBatch)
        elif PATCH_TYPE == SCALED_PATCH_NET:
            imgFeatures = model.forwardScaledPatches(shiftImgBatch)
        else: assert False
        imgFeatures = imgFeatures.detach()
        imgFeatrueList.append(imgFeatures)
        shiftTupList.append(shiftTupBatch)
    allImgFeatures = torch.cat(imgFeatrueList)
    allShiftTup = torch.cat(shiftTupList)
    return allImgFeatures,allShiftTup

def genLocalClassMap(imgFeatures, shiftTup,imgSize,patchSize):
    assert imgSize % patchSize == 0
    patchCount = int(imgSize / patchSize)
    elementImgCount = imgFeatures.shape[0]
    featureNum = imgFeatures.shape[-1]
    imgFeatures = torch.reshape(imgFeatures,(-1,patchCount,patchCount,featureNum))

    shiftTup = shiftTup * -1

    totalMap = torch.zeros((featureNum,imgSize,imgSize),dtype=imgFeatures.dtype,device=imgFeatures.device)

    for i in range(elementImgCount):
        elementImg = imgFeatures[i]
        elementImg = torch.permute(elementImg,(2,0,1))
        elementShift = shiftTup[i].tolist()

        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=1)
        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=2)
        elementImg = imgRetrivalUtil.createShiftImg(elementImg,*elementShift,patchSize=patchSize,imgSize=imgSize,useChannelDim=True)
        totalMap += elementImg
    totalMap = totalMap/elementImgCount
    return totalMap

def applyGradCamOneClass(model,img,targetId,CamAlg):
    target_layers = [model.baseBackbone.features[7]]
    targets = [ClassifierOutputTarget(targetId)]
    with CamAlg(model=model,target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = torch.tensor(grayscale_cam)
        return grayscale_cam

def applyGradCam(model,img,targetCount,CamAlg):
    mapList = list()
    for i in range(targetCount):
        oneClassMap = applyGradCamOneClass(model,img,i,CamAlg=CamAlg)
        mapList.append(oneClassMap)
    allClassMap = torch.cat(mapList)
    return allClassMap, None

def applyPatchLocalisation(model,img, imgSize, patchSize ,stepsPerPatch, device,):
    shiftImgDataset = imgRetrivalUtil.ShiftImgSet(img,stepsPerPatch,patchSize,imgSize)
    shiftLoader = iter(DataLoader(shiftImgDataset,batch_size=4,shuffle=False))
    imgFeatures, shiftTup = encodeClassImg(model,device,shiftLoader)
    localClassMap = genLocalClassMap(imgFeatures,shiftTup,imgSize,patchSize)
    return localClassMap,(imgFeatures,shiftTup)

def getChexpertImgPath(line):
    relative_path = line[0].replace("CheXpert-v1.0/", "", 1)
    path = CHEXPERT_PATH + relative_path
    return path

def scaleImg(img,size):
    return v2.Resize(size=size,antialias=True)(img)

def cropImg(img):
    width,height = img.size
    cropLen = min(width,height)
    widthMargin = width - cropLen
    heightMargin = height - cropLen
    img = img.crop((widthMargin//2,heightMargin//2,widthMargin//2+cropLen,heightMargin//2+cropLen))
    return img

def getImg(imgPath,crop=True):
    img = Image.open(imgPath).convert("L")
    origSize = img.size
    if crop: img = cropImg(img)
    img = torchvision.transforms.functional.to_tensor(img)
    return img,origSize

def fetchImg(line,size):
    imgPath = getChexpertImgPath(line)
    img,origSize = getImg(imgPath)
    scaledImg = scaleImg(img,size)
    return scaledImg,origSize

def genChexlocalizeMap(imgLine,head,model,mapFunc,outPath,split,imgSize,patch_size,device,stepsPerPatch=None,CamAlg=None):
    img,origSize = fetchImg(imgLine,(imgSize,imgSize))

    img = img.to(device,non_blocking=True)
    img = img[None]

    cutOff = 1 if split == "test" else 5
    labelNames = head[cutOff:]
    gtList = imgLine[cutOff:]
    allClassMap = None
    if mapFunc == applyGradCam:
        allClassMap,_ = applyGradCam(model,img,len(labelNames),CamAlg=CamAlg)
        allClassMap.detach().cpu()
    elif mapFunc == applyPatchLocalisation:
        allClassMap,_ = applyPatchLocalisation(model,img,imgSize=imgSize,patchSize=patch_size,stepsPerPatch=stepsPerPatch,device=device)
        allClassMap = (torch.clip(allClassMap,-1*NORM_VAL,NORM_VAL) + NORM_VAL)/(2*NORM_VAL)
        print("range:",torch.min(allClassMap),torch.max(allClassMap))
        allClassMap = allClassMap.detach().cpu()
    assert allClassMap is not None

    modelGlobalOutput = model(img)[0]
    modelGlobalOutput = modelGlobalOutput.detach().cpu()

    expandedCxr = img[0].expand(3,-1,-1).detach().cpu()

    for i,name in enumerate(labelNames):
        if name == "Lung Opacity": name = "Airspace Opacity"
        groundTruth = int(float(gtList[i]))

        prob = torch.nn.functional.sigmoid(modelGlobalOutput[i]).item()

        partClassMap = allClassMap[i].detach().cpu()
        partClassMap = partClassMap[None][None]
        retDict = {
                'map': partClassMap,
                'prob': prob,
                'task': name,
                'gt':groundTruth,
                'cxr_img':expandedCxr,
                'cxr_dims': origSize,
            }

        fileDescList = imgLine[0].split("/")
        patient = fileDescList[-3]
        study = fileDescList[-2]
        view = fileDescList[-1].replace(".jpg","")
        pklFileName = outPath +  patient + "_" + study + "_" + view + "_" + name + "_map.pkl"
        with open(pklFileName, 'wb') as handle:
            pickle.dump(retDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved",pklFileName)


def _load_model(modelPath, patchSize, imgSize, device="cuda"):
    """
    Load either a ScalePatchNet or FiLMPatchNet checkpoint.
    Detects the format from key names and applies translation if needed.
    """
    state_dict = torch.load(modelPath, weights_only=True, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    is_scalepatchnet_ckpt = any(k.startswith("baseBackbone.") for k in state_dict)

    # If the checkpoint has "eff_features.*" keys it's already FiLMPatchNet
    is_filmpatchnet_ckpt = any(k.startswith("eff_features.") for k in state_dict)

    if is_filmpatchnet_ckpt:
        model = FiLMPatchNet(patchSize=patchSize, outFeatures=14)
        model.load_state_dict(state_dict)
    elif is_scalepatchnet_ckpt:
        # Default: load as ScalePatchNet (original behaviour)
        model = ScalePatchNet(patchSize=patchSize, outFeatures=14)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Cannot detect model class from checkpoint: {modelPath}")

    return model.to(device)


def execEval(modelPath,usedSplit,CamAlg=None,imgSize=None,patchSize=None,stepsPerPatch=16,device="cuda"):

    modelName = modelPath.split("/")[-1].replace(".pt","")

    assert usedSplit in ["test","val",None]

    model = _load_model(modelPath, patchSize=patchSize, imgSize=imgSize, device=device)
    model.eval()

    head,*dataLines = readCSV(CHEXPERT_PATH + usedSplit + "_labels.csv")

    camName = PATCH_TYPE if CamAlg is None else CamAlg.__name__

    addName = ""
    if stepsPerPatch == 1:
        addName = "_oneStep"

    outFolder = MAP_OUT_FOLDER +usedSplit + addName + "_" + camName + "_" + modelName + "/"
    os.makedirs(outFolder,exist_ok=True)

    mapFunc = applyPatchLocalisation if CamAlg is None else applyGradCam
    cutOff = 1 if usedSplit == "test" else 5
    first_label = head[cutOff]
    if first_label == "Lung Opacity":
        first_label = "Airspace Opacity"

    skipped = 0

    for line in tqdm(dataLines):
        fileDescList = line[0].split("/")
        patient = fileDescList[-3]
        study   = fileDescList[-2]
        view    = fileDescList[-1].replace(".png", "")
        first_label = head[1 if usedSplit == "test" else 5]
        if first_label == "Lung Opacity":
            first_label = "Airspace Opacity"
        expected_pkl = outFolder + patient + "_" + study + "_" + view + "_" + first_label + "_map.pkl"

        if os.path.exists(expected_pkl):
            print(f"Skipping (already done): {patient}_{study}_{view}")
            continue

        genChexlocalizeMap(line, head, model, mapFunc, outFolder,
                       split=usedSplit, imgSize=imgSize,
                       patch_size=patchSize, device=device,
                       stepsPerPatch=stepsPerPatch, CamAlg=CamAlg)
    print(f"Done. Skipped {skipped} already-processed images.")

if __name__ == "__main__":

    PATCH_TYPE = RAW_PATCH_NET
    execEval("savedModels/MedicalPatchNet_weights.pt","test",stepsPerPatch=8,imgSize=512,patchSize=64)
    torch.cuda.empty_cache()
    print("DONE")