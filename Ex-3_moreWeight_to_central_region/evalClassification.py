import imgRetrivalUtil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision.transforms import v2
from utilFunc import readCSV
import pickle
import os

from NetworlModel import ScalePatchNet
from patchWeighting import (
    get_foreground_weight_map,
    patch_weights_from_map,
    patch_weights_to_grid,
)

CHEXLOCALIZE_BASE_PATH = "DataSet/CheXlocalize/"
CHEXPERT_PATH = CHEXLOCALIZE_BASE_PATH + "CheXpert-v1.0/"
MAP_OUT_FOLDER = "DataOut2/"

RAW_PATCH_NET = "RawPatchNet"
SCALED_PATCH_NET = "ScaledPatchNet"
WEIGHTED_PATCH_NET = "WeightedPatchNet"

PATCH_TYPE = WEIGHTED_PATCH_NET
NORM_VAL = 200


def encodeClassImg(model, device, shiftLoader, patchSize, useWeights=False, minWeight=0.15):
    imgFeatureList = []
    shiftTupList = []
    weightList = []
    weightMapList = []

    for shiftImgBatch, shiftTupBatch in shiftLoader:
        shiftImgBatch = shiftImgBatch.to(device)

        if useWeights:
            weightMap = get_foreground_weight_map(shiftImgBatch, min_weight=minWeight)  # [B,1,H,W]
            patchWeights = patch_weights_from_map(weightMap, patch_size=patchSize)       # [B,P]

            outDict = model.forwardWeightedPatches(
                shiftImgBatch,
                patchWeights=patchWeights,
                normalize_weights=True,
                min_weight=minWeight
            )
            imgFeatures = outDict["weighted_patch_logits"]
            weightList.append(patchWeights.detach())
            weightMapList.append(weightMap.detach())
        else:
            if PATCH_TYPE == RAW_PATCH_NET:
                imgFeatures = model.forwardRawPatches(shiftImgBatch)
            elif PATCH_TYPE == SCALED_PATCH_NET:
                imgFeatures = model.forwardScaledPatches(shiftImgBatch)
            else:
                raise ValueError("Unknown PATCH_TYPE without weights")

        imgFeatures = imgFeatures.detach()
        imgFeatureList.append(imgFeatures)
        shiftTupList.append(shiftTupBatch)

    allImgFeatures = torch.cat(imgFeatureList)
    allShiftTup = torch.cat(shiftTupList)

    ret = {
        "img_features": allImgFeatures,
        "shift_tup": allShiftTup,
    }

    if useWeights:
        ret["patch_weights"] = torch.cat(weightList)
        ret["weight_maps"] = torch.cat(weightMapList)

    return ret


def genLocalClassMap(imgFeatures, shiftTup, imgSize, patchSize):
    assert imgSize % patchSize == 0
    patchCount = int(imgSize / patchSize)
    elementImgCount = imgFeatures.shape[0]
    featureNum = imgFeatures.shape[-1]
    imgFeatures = torch.reshape(imgFeatures, (-1, patchCount, patchCount, featureNum))

    shiftTup = shiftTup * -1
    totalMap = torch.zeros((featureNum, imgSize, imgSize), dtype=imgFeatures.dtype, device=imgFeatures.device)

    for i in range(elementImgCount):
        elementImg = imgFeatures[i]
        elementImg = torch.permute(elementImg, (2, 0, 1))
        elementShift = shiftTup[i].tolist()

        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=1)
        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=2)
        elementImg = imgRetrivalUtil.createShiftImg(
            elementImg,
            *elementShift,
            patchSize=patchSize,
            imgSize=imgSize,
            useChannelDim=True
        )
        totalMap += elementImg

    totalMap = totalMap / elementImgCount
    return totalMap


def aggregateWeightMaps(weightMaps, shiftTup, imgSize, patchSize):
    # weightMaps: [N,1,H,W]
    shiftTup = shiftTup * -1
    totalMap = torch.zeros((1, imgSize, imgSize), dtype=weightMaps.dtype, device=weightMaps.device)

    for i in range(weightMaps.shape[0]):
        elementImg = weightMaps[i]
        elementShift = shiftTup[i].tolist()
        elementImg = imgRetrivalUtil.createShiftImg(
            elementImg,
            *elementShift,
            patchSize=patchSize,
            imgSize=imgSize,
            useChannelDim=True
        )
        totalMap += elementImg

    totalMap = totalMap / weightMaps.shape[0]
    return totalMap


def applyPatchLocalisation(model, img, imgSize, patchSize, stepsPerPatch, device, useWeights=False, minWeight=0.15):
    shiftImgDataset = imgRetrivalUtil.ShiftImgSet(img, stepsPerPatch, patchSize, imgSize)
    shiftLoader = iter(DataLoader(shiftImgDataset, batch_size=4, shuffle=False))

    enc = encodeClassImg(
        model=model,
        device=device,
        shiftLoader=shiftLoader,
        patchSize=patchSize,
        useWeights=useWeights,
        minWeight=minWeight,
    )

    localClassMap = genLocalClassMap(enc["img_features"], enc["shift_tup"], imgSize, patchSize)

    extra = {
        "imgFeatures": enc["img_features"],
        "shiftTup": enc["shift_tup"],
    }

    if useWeights:
        weightMapAgg = aggregateWeightMaps(enc["weight_maps"], enc["shift_tup"], imgSize, patchSize)
        extra["weightMap"] = weightMapAgg
        extra["patchWeights"] = enc["patch_weights"]

    return localClassMap, extra


def getChexpertImgPath(line):
    relative_path = line[0].replace("CheXpert-v1.0/", "", 1)
    path = CHEXPERT_PATH + relative_path
    return path


def scaleImg(img, size):
    return v2.Resize(size=size, antialias=True)(img)


def cropImg(img):
    width, height = img.size
    cropLen = min(width, height)
    widthMargin = width - cropLen
    heightMargin = height - cropLen
    img = img.crop((widthMargin // 2, heightMargin // 2, widthMargin // 2 + cropLen, heightMargin // 2 + cropLen))
    return img


def getImg(imgPath, crop=True):
    img = Image.open(imgPath).convert("L")
    origSize = img.size
    if crop:
        img = cropImg(img)
    img = torchvision.transforms.functional.to_tensor(img)
    return img, origSize


def fetchImg(line, size):
    imgPath = getChexpertImgPath(line)
    img, origSize = getImg(imgPath)
    scaledImg = scaleImg(img, size)
    return scaledImg, origSize


def getGlobalOutput(model, img, patchSize, useWeights=False, minWeight=0.15):
    if useWeights:
        weightMap = get_foreground_weight_map(img, min_weight=minWeight)
        patchWeights = patch_weights_from_map(weightMap, patch_size=patchSize)
        outDict = model.forwardWeightedPatches(
            img,
            patchWeights=patchWeights,
            normalize_weights=True,
            min_weight=minWeight
        )
        return outDict["global_logits"][0], weightMap
    else:
        return model(img)[0], None


def genChexlocalizeMap(
    imgLine,
    head,
    model,
    outPath,
    split,
    imgSize,
    patchSize,
    device,
    stepsPerPatch=16,
    useWeights=False,
    minWeight=0.15,
):
    img, origSize = fetchImg(imgLine, (imgSize, imgSize))
    img = img.to(device, non_blocking=True)
    img = img[None]

    cutOff = 1 if split == "test" else 5
    labelNames = head[cutOff:]
    gtList = imgLine[cutOff:]

    allClassMap, extra = applyPatchLocalisation(
        model,
        img,
        imgSize=imgSize,
        patchSize=patchSize,
        stepsPerPatch=stepsPerPatch,
        device=device,
        useWeights=useWeights,
        minWeight=minWeight,
    )

    allClassMap = (torch.clip(allClassMap, -1 * NORM_VAL, NORM_VAL) + NORM_VAL) / (2 * NORM_VAL)
    allClassMap = allClassMap.detach().cpu()

    modelGlobalOutput, directWeightMap = getGlobalOutput(
        model,
        img,
        patchSize=patchSize,
        useWeights=useWeights,
        minWeight=minWeight,
    )
    modelGlobalOutput = modelGlobalOutput.detach().cpu()

    expandedCxr = img[0].expand(3, -1, -1).detach().cpu()

    savedWeightMap = None
    if useWeights and "weightMap" in extra:
        savedWeightMap = extra["weightMap"].detach().cpu()
    elif useWeights and directWeightMap is not None:
        savedWeightMap = directWeightMap[0].detach().cpu()

    for i, name in enumerate(labelNames):
        if name == "Lung Opacity":
            name = "Airspace Opacity"

        groundTruth = int(float(gtList[i]))
        prob = torch.sigmoid(modelGlobalOutput[i]).item()

        partClassMap = allClassMap[i].detach().cpu()[None][None]

        retDict = {
            "map": partClassMap,
            "prob": prob,
            "task": name,
            "gt": groundTruth,
            "cxr_img": expandedCxr,
            "cxr_dims": origSize,
            "use_weights": useWeights,
        }

        if savedWeightMap is not None:
            retDict["weight_map"] = savedWeightMap

        fileDescList = imgLine[0].split("/")
        patient = fileDescList[-3]
        study = fileDescList[-2]
        view = fileDescList[-1].replace(".jpg", "").replace(".png", "")

        suffix = "_weighted" if useWeights else "_plain"
        pklFileName = outPath + patient + "_" + study + "_" + view + "_" + name + suffix + "_map.pkl"

        with open(pklFileName, "wb") as handle:
            pickle.dump(retDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("saved", pklFileName)


def execEval(
    modelPath,
    usedSplit,
    imgSize=512,
    patchSize=64,
    stepsPerPatch=64,
    device="cuda",
    useWeights=False,
    minWeight=0.15,
):
    modelName = modelPath.split("/")[-1].replace(".pt", "")
    assert usedSplit in ["test", "val", None]

    state_dict = torch.load(modelPath, weights_only=True)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = ScalePatchNet(patchSize=patchSize, outFeatures=14)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    head, *dataLines = readCSV(CHEXPERT_PATH + usedSplit + "_labels.csv")

    tag = "WeightedPatchNet" if useWeights else PATCH_TYPE
    outFolder = MAP_OUT_FOLDER + usedSplit + "_" + tag + "_" + modelName + "/"
    os.makedirs(outFolder, exist_ok=True)

    for line in tqdm(dataLines):
        genChexlocalizeMap(
            line,
            head,
            model,
            outFolder,
            split=usedSplit,
            imgSize=imgSize,
            patchSize=patchSize,
            device=device,
            stepsPerPatch=stepsPerPatch,
            useWeights=useWeights,
            minWeight=minWeight,
        )


if __name__ == "__main__":
    PATCH_TYPE = RAW_PATCH_NET
    execEval(
        "savedModels/MedicalPatchNet_weights.pt",
        "test",
        imgSize=512,
        patchSize=64,
        stepsPerPatch=8,
        device="cuda",
        useWeights=False,
        minWeight=0.0,
    )
    torch.cuda.empty_cache()
    print("DONE")