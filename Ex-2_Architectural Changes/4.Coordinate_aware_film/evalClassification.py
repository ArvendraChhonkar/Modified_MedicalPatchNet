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

from NetworkModel import ScalePatchNet

CHEXLOCALIZE_BASE_PATH = "DataSet/CheXlocalize/" #REPLACE: replace with your path
CHEXPERT_PATH = CHEXLOCALIZE_BASE_PATH + "CheXpert-v1.0/"
MAP_OUT_FOLDER = "DataOut_TXT/" #REPLACE: replace with your path

RAW_PATCH_NET = "RawPatchNet"
SCALED_PATCH_NET = "ScaledPatchNet"

# Evaluating raw patches (per user request): PATCH_TYPE is set to RAW_PATCH_NET
# here and re-affirmed in __main__ below.
PATCH_TYPE = RAW_PATCH_NET

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
    
    shiftTup = shiftTup * -1 # reverse shift, because relative to the image, the grid is shifted into the other direction
    
    totalMap = torch.zeros((featureNum,imgSize,imgSize),dtype=imgFeatures.dtype,device=imgFeatures.device)

    for i in range(elementImgCount):
        elementImg = imgFeatures[i]
        elementImg = torch.permute(elementImg,(2,0,1))# channel = features => first dimension
        elementShift = shiftTup[i].tolist()

        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=1)
        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=2)
        elementImg = imgRetrivalUtil.createShiftImg(elementImg,*elementShift,patchSize=patchSize,imgSize=imgSize,useChannelDim=True)
        totalMap += elementImg
    totalMap = totalMap/elementImgCount
    return totalMap

def applyGradCamOneClass(model,img,targetId,CamAlg):
    target_layers = [model.baseBackbone.features[7]] # check if correct
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
    # line[0] = "CheXpert-v1.0/patient64622/study1/view1_frontal.png"
    # CHEXPERT_PATH = "DataSet/CheXlocalize/CheXpert-v1.0/"
    # strip the "CheXpert-v1.0/" prefix from line[0] to avoid duplication
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
    img = img[None] # add batch dimension [batch size = 1]

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

    
    modelGlobalOutput = model(img)[0] # batch size = 1
    modelGlobalOutput = modelGlobalOutput.detach().cpu()

    
    
    expandedCxr = img[0].expand(3,-1,-1).detach().cpu()

    

    for i,name in enumerate(labelNames):
        if name == "Lung Opacity": name = "Airspace Opacity" # different label from Chexlocalize and Chexpert/MIMIC ( https://www.nature.com/articles/s42256-022-00536-x )
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


def loadScalePatchNetCheckpoint(modelPath, patchSize, outFeatures, device):
    """
    Load a ScalePatchNet checkpoint safely.

    ScalePatchNet includes the Soft Consensus Gate ("filter") submodule
    (`consensus_conv`). If the checkpoint predates that addition, a plain
    strict=True load_state_dict() would crash with missing-key errors.
    This loader uses strict=False and reports exactly what did/didn't
    transfer, so a mismatch is visible instead of silently failing or
    silently running with an untrained filter.
    """
    state_dict = torch.load(modelPath, map_location=device, weights_only=True)
    # the model may have been saved from a torch.compile()'d model => strip "_orig_mod." prefix
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = ScalePatchNet(patchSize=patchSize, outFeatures=outFeatures)
    modelStateDict = model.state_dict()

    # strict=False only tolerates missing/unexpected KEYS, not shape mismatches.
    # If a key exists in both but has a different shape (e.g. checkpoint was saved
    # from an older classifier head), load_state_dict still raises. Filter those
    # out ourselves and report them explicitly instead of crashing.
    filteredStateDict = {}
    shapeMismatched = []
    for k, v in state_dict.items():
        if k in modelStateDict:
            if modelStateDict[k].shape == v.shape:
                filteredStateDict[k] = v
            else:
                shapeMismatched.append((k, tuple(v.shape), tuple(modelStateDict[k].shape)))

    missing, unexpected = model.load_state_dict(filteredStateDict, strict=False)

    print(f"[ScalePatchNet] Loaded from: {modelPath}")
    print(f"  Missing keys       : {missing}")
    print(f"  Unexpected keys    : {unexpected}")
    if shapeMismatched:
        print("  Shape-mismatched keys (skipped, kept randomly-initialized):")
        for k, ckptShape, modelShape in shapeMismatched:
            print(f"    {k}: checkpoint {ckptShape} vs model {modelShape}")

    if any(k.startswith("consensus_conv") for k in missing):
        print(
            "  WARNING: 'consensus_conv' (the Soft Consensus Gate / filter) was NOT "
            "found in the checkpoint. It will run with randomly initialized weights, "
            "which will make forward()/forwardScaledPatches() outputs meaningless. "
            "forwardRawPatches() is unaffected since it doesn't use the gate."
        )
    if missing == [] and unexpected == []:
        print("  Checkpoint matches the current ScalePatchNet architecture exactly.")

    return model.to(device)


def execEval(modelPath,usedSplit,CamAlg=None,imgSize=None,patchSize=None,stepsPerPatch=16,device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    modelName = modelPath.split("/")[-1].replace(".pt","")

    assert usedSplit in ["test","val",None]

    model = loadScalePatchNetCheckpoint(modelPath, patchSize=patchSize, outFeatures=14, device=device)
    model.eval()

    head,*dataLines = readCSV(CHEXPERT_PATH + usedSplit + "_labels.csv")

    camName = PATCH_TYPE if CamAlg is None else CamAlg.__name__

    addName = ""
    if stepsPerPatch == 1:
        addName = "_oneStep"


    outFolder = MAP_OUT_FOLDER +usedSplit + addName + "_" + camName + "_" + modelName + "/"
    os.makedirs(outFolder,exist_ok=True)

    mapFunc = applyPatchLocalisation if CamAlg is None else applyGradCam
    # --- cutOff mirrors the logic inside genChexlocalizeMap ---
    cutOff = 1 if usedSplit == "test" else 5
    first_label = head[cutOff]
    if first_label == "Lung Opacity":
        first_label = "Airspace Opacity"

    skipped = 0

    for line in tqdm(dataLines):

        # Build the expected pkl filename for the FIRST label to check if already processed
        fileDescList = line[0].split("/")
        patient = fileDescList[-3]
        study   = fileDescList[-2]
        view    = fileDescList[-1].replace(".png", "")
        expected_pkl = outFolder + patient + "_" + study + "_" + view + "_" + first_label + "_map.pkl"

        if os.path.exists(expected_pkl):
            print(f"Skipping (already done): {patient}_{study}_{view}")
            skipped += 1
            continue  # ← skip this patient entirely

        genChexlocalizeMap(line, head, model, mapFunc, outFolder,
                       split=usedSplit, imgSize=imgSize,
                       patch_size=patchSize, device=device,
                       stepsPerPatch=stepsPerPatch, CamAlg=CamAlg)
    print(f"Done. Skipped {skipped} already-processed images.")


if __name__ == "__main__":

    
    '''
    PATCH_TYPE = RAW_PATCH_NET

    execEval("savedModels/MedicalPatchNet_weights.pt","val",stepsPerPatch=64,imgSize=512,patchSize=64)
    execEval("savedModels/MedicalPatchNet_weights.pt","test",stepsPerPatch=64,imgSize=512,patchSize=64)

    PATCH_TYPE = SCALED_PATCH_NET

    execEval("savedModels/MedicalPatchNet_weights.pt","val",stepsPerPatch=64,imgSize=512,patchSize=64)
    execEval("savedModels/MedicalPatchNet_weights.pt","test",stepsPerPatch=64,imgSize=512,patchSize=64)

    execEval("savedModels/EfficientNetB0_weights.pt","val",GradCAM,imgSize=512,patchSize=512)
    execEval("savedModels/EfficientNetB0_weights.pt","test",GradCAM,imgSize=512,patchSize=512)

    execEval("savedModels/EfficientNetB0_weights.pt","val",GradCAMPlusPlus,imgSize=512,patchSize=512)
    execEval("savedModels/EfficientNetB0_weights.pt","test",GradCAMPlusPlus,imgSize=512,patchSize=512)

    execEval("savedModels/EfficientNetB0_weights.pt","val",EigenCAM,imgSize=512,patchSize=512)
    execEval("savedModels/EfficientNetB0_weights.pt","test",EigenCAM,imgSize=512,patchSize=512)
    
    
    PATCH_TYPE = RAW_PATCH_NET

    execEval("savedModels/MedicalPatchNet_weights.pt","val",stepsPerPatch=64,imgSize=512,patchSize=64)
    '''
    PATCH_TYPE = RAW_PATCH_NET  # evaluating raw patches, as requested
    execEval("savedModels/TT_conv_neigh_gate_weights.pt","test",stepsPerPatch=64,imgSize=512,patchSize=64)
    torch.cuda.empty_cache()
    print("DONE")