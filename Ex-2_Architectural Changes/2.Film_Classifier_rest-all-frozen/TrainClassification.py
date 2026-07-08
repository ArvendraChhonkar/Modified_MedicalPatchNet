from pyexpat import model

import argParser as ARG
ARG.initArgs()

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from ChexpertDataset import ClassificationChexpertDataset
from NetworkModel import getModelClass, FiLMPatchNet
from tqdm import tqdm
import torchmetrics.functional as tmf

import wandb

import multiprocessing as mp

def initWandB():
    
    wandb.login()
    runVar = wandb.init(
        project="MEDICAL_PATCH_NET",
        config={
            "learning_rate": ARG.LEARNING_RATE,
            "epoch_num":ARG.EPOCH_NUM,
            "batch_size":ARG.BATCH_SIZE,
            "patch_size":ARG.PATCH_SIZE,
            "wandb_name":ARG.WANDB_NAME
        },
        name=ARG.WANDB_NAME
        )

def log(name,val,printLog=True,commit=False):
    if isinstance(val,torch.Tensor): val = val.item()
    if printLog: print("LOG",name,val)
    if ARG.USE_WANDB: wandb.log({name:val},commit=commit)

NUM_PROC = 24
DEFAULT_MODEL_PATH = "savedModels/"

TEST = "test"

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        trainLoader: DataLoader,
        validLoaderList: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        gpuId: int,
        saveEvery: int = 1,
    ) -> None:

        self.gpuId = gpuId
        self.model = model.to(gpuId)
        self.trainLoader = trainLoader
        self.validLoaderList = validLoaderList
        self.optimizer = optimizer
        self.saveEvery = saveEvery
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.ampScaler = GradScaler()

    def _runBatch(self, img, label):
        self.optimizer.zero_grad()

        with autocast(device_type='cuda'):
            output = self.model(img)
            loss = self.criterion(output,label)

        self.ampScaler.scale(loss).backward()
        self.ampScaler.step(self.optimizer)
        self.ampScaler.update()

        self.scheduler.step()

        log("learningRate",self.scheduler.get_last_lr()[0],printLog=False)
        log("loss",loss,commit=True,printLog=False)

    def _runEpoch(self, epoch):
        self.model.train()
        stepNum = len(self.trainLoader)
        print(f"Epoch: {epoch} | Batchsize: {ARG.BATCH_SIZE} | Steps: {stepNum}")

        for i, (img,label) in enumerate(tqdm(self.trainLoader)):
            img = img.to(self.gpuId,non_blocking=True,dtype=torch.float)
            label = label.to(self.gpuId,non_blocking=True)
            self._runBatch(img,label)

        if hasattr(self.model, 'film_gamma'):
            gamma = self.model.film_gamma.data
            beta  = self.model.film_beta.data
            log("film_gamma/mean", gamma.mean())
            log("film_gamma/std",  gamma.std())
            log("film_beta/mean",  beta.mean())
            log("film_beta/std",   beta.std())
            # Optional: log a histogram (more detailed, but heavier)
            if ARG.USE_WANDB:
                wandb.log({"film_gamma/hist": wandb.Histogram(gamma.cpu().numpy())}, commit=False)
                wandb.log({"film_beta/hist":  wandb.Histogram(beta.cpu().numpy())}, commit=False)
    
        for el in self.validLoaderList:
            if el[2] == TEST and (epoch != ARG.EPOCH_NUM - 1): continue
            self.validate(*el)
        print(
            "Classifier Weight Norm:",
            self.model.classifier.weight.norm().item()
        )
        print(
            "Gamma mean:",
            self.model.film_gamma.mean().item()
        )

        print(
            "Gamma std:",
            self.model.film_gamma.std().item()
        )

        print(
            "Beta mean:",
            self.model.film_beta.mean().item()
        )

        print(
            "Beta std:",
            self.model.film_beta.std().item()
        )

    def _saveCheckpoint(self, epoch):
        saveName = DEFAULT_MODEL_PATH + ARG.MODEL_SAVE_PATH
        torch.save(self.model, saveName)
        torch.save(self.model.state_dict(), saveName.replace(".pt","_weights.pt"))

        print(f"Epoch {epoch} | Checkpoint saved at {saveName}")

    def train(self, maxEpochs: int):
        for epoch in range(maxEpochs):
            log("epoch",epoch)
            self._runEpoch(epoch)
            if epoch % self.saveEvery == 0:
                self._saveCheckpoint(epoch)
        if ARG.USE_WANDB: wandb.log({},commit=True)

    def calculateThresholds(self, thresholdLoader):
        self.model.eval()
        outputList = list()
        labelList = list()
        for img,label in tqdm(thresholdLoader,desc="optThreshold"):
            img = img.to(self.gpuId, non_blocking=True, dtype=torch.float)
            with torch.no_grad():
                output = self.model(img).detach().cpu()
            outputList.append(nn.functional.sigmoid(output))
            labelList.append(label.cpu())

        allOutput = torch.cat(outputList).to(dtype=torch.float32)
        allLabel = torch.cat(labelList).to(dtype=torch.int32)
        labelNameList = thresholdLoader.dataset.getLabelNames()
        thrsDict = dict()

        for i,labelName in enumerate(labelNameList):
            preds = allOutput[:, i]
            targets = allLabel[:, i]
            fpr, tpr, thresh = tmf.roc(preds, targets, task="binary")
            optIdx = torch.argmax(tpr - fpr)
            thrsDict[labelName] = thresh[optIdx].item()

        print(thrsDict)
        self.model.train()
        return thrsDict

    def validate(self,dataLoader,datasetName,validType,thresholdValidLoader=None):
        self.model.eval()
        outputList = list()
        labelList = list()
        for img,label in tqdm(dataLoader):
            img = img.to(self.gpuId,non_blocking=True,dtype=torch.float)
            output = self.model(img).detach().to("cpu",non_blocking=True)
            label = label.to("cpu",non_blocking=True,dtype=torch.int)
            outputList.append(output)
            labelList.append(label)

        allOutput = torch.cat(outputList)
        allOutput = nn.functional.sigmoid(allOutput)
        allLabel = torch.cat(labelList)

        labelNameList = dataLoader.dataset.getLabelNames()

        avgDict = dict()
        if thresholdValidLoader is not None: thrsDict = self.calculateThresholds(thresholdValidLoader)
        print("LEN:",len(labelNameList),labelNameList,allOutput.size(),allLabel.size())
        for i,labelName in enumerate(labelNameList):
            preds = allOutput[:,i]
            targets = allLabel[:,i]

            def execMetric(func,name=None):
                metricName = func.__name__ if name is None else name
                metr = func(preds,targets,task="binary")
                log(datasetName+"/"+labelName + "/" + metricName,metr)
                if metricName not in avgDict: avgDict[metricName] = list()
                avgDict[metricName].append(metr)

            def execThrsMetric(func,threshold,name=None,execWithoutThreshold=True):
                if execWithoutThreshold: execMetric(func,name)
                metricName = func.__name__ if name is None else name
                metr = func(preds,targets,task="binary",threshold=threshold)
                log(datasetName+"/"+labelName + "/thrs_" + metricName,metr)

            def execBootstrapMetric(func,name=None,threshold=None,bootstrapNum = 100000):
                argList = [(preds,targets,func,threshold) for _ in range(bootstrapNum)]

                with mp.Pool(ARG.PROC_NUM) as pool:
                    metrList = pool.map(execOneBootStrap,argList)

                metrList.sort()

                lowConfIdx = round(len(metrList)*0.025)
                highConfIdx = round(len(metrList)*0.975)
                medianIdx = round(len(metrList)*0.5)

                meanVal = sum(metrList)/len(metrList)
                metricName = func.__name__ if name is None else name
                thrsTag = "" if threshold is None else "thrs_"
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_low_" + metricName,metrList[lowConfIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_high_" + metricName,metrList[highConfIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_median_" + metricName,metrList[medianIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_mean_" + metricName,meanVal)

            execMetric(tmf.auroc)

            if thresholdValidLoader is not None:
                optThrs = thrsDict[labelName]
                log(datasetName+"/"+labelName+"/opt_thrs",optThrs)

                execThrsMetric(tmf.accuracy,threshold=optThrs)
                execThrsMetric(tmf.precision,threshold=optThrs)
                execThrsMetric(tmf.recall,threshold=optThrs,name="sensitivity")
                execThrsMetric(tmf.specificity,threshold=optThrs)
                execThrsMetric(tmf.f1_score,threshold=optThrs)

            if validType == TEST:
                execBootstrapMetric(tmf.auroc)
                execBootstrapBoth = lambda func,name=None: (execBootstrapMetric(func,name=name),execBootstrapMetric(func,name=name,threshold=optThrs))
                execBootstrapBoth(tmf.accuracy)
                execBootstrapBoth(tmf.precision)
                execBootstrapBoth(tmf.recall,name="sensitivity")
                execBootstrapBoth(tmf.specificity)
                execBootstrapBoth(tmf.f1_score)

        for key in avgDict.keys():
            valList = avgDict[key]
            avgVal = sum(valList)/len(valList)
            log(datasetName+"/avg/"+key,avgVal)

        self.model.train()

def execOneBootStrap(inp):
    preds,targets,func,threshold = inp

    assert len(preds) == len(targets)
    valCount = len(preds)

    sampled_indices = torch.multinomial(torch.ones(valCount), valCount, replacement=True)
    sampledPreds = preds[sampled_indices]
    sampledTargets = targets[sampled_indices]

    if threshold is None:
        metr = func(sampledPreds,sampledTargets,task="binary").item()
    else:
        metr = func(sampledPreds,sampledTargets,task="binary",threshold=threshold).item()

    return metr

def prepareDataloader(dataset: Dataset, batchSize: int, training: bool):
    return DataLoader(
        dataset,
        batch_size=batchSize,
        num_workers=NUM_PROC,
        pin_memory=True,
        shuffle=training,
        drop_last=training,
    )


def loadModelFromCheckpoint(weightPath):
    """
    Load a checkpoint into the model class set by -model flag.

    Detects checkpoint format automatically:
      • ScalePatchNet checkpoint (keys start with "baseBackbone.*"):
        When target class is FiLMPatchNet, applies key translation via
        FiLMPatchNet.from_scalepatchnet_checkpoint().
      • FiLMPatchNet checkpoint (keys start with "eff_features.*"):
        Loaded directly with load_state_dict().
      • ScalePatchNet → ScalePatchNet: original behaviour unchanged.
    """
    ModelClass = getModelClass(ARG.MODEL_CLASS)

    raw = torch.load(weightPath, map_location="cpu", weights_only=True)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}

    is_scalepatchnet_ckpt = any(k.startswith("baseBackbone.") for k in state_dict)

    if ModelClass is FiLMPatchNet and is_scalepatchnet_ckpt:
        # Translate ScalePatchNet checkpoint keys into FiLMPatchNet keys
        model = FiLMPatchNet.from_scalepatchnet_checkpoint(
            weightPath,
            patchSize=ARG.PATCH_SIZE,
            outFeatures=14,
        )
    else:
        model = ModelClass(patchSize=ARG.PATCH_SIZE, outFeatures=14)
        model.load_state_dict(state_dict)

    model = torch.compile(model)
    return model


def loadTrainObjs():
    trainSet = ClassificationChexpertDataset("train", augementImg=True)

    ModelClass = getModelClass(ARG.MODEL_CLASS)
    model = ModelClass(patchSize=ARG.PATCH_SIZE, outFeatures=14)

    # If a checkpoint path was provided and the target model is FiLMPatchNet,
    # use the special loader that translates ScalePatchNet keys into FiLMPatchNet.
    # This copies the backbone and classifier weights; γ=1 and β=0 by default.
    if ARG.MODEL_LOAD_PATH is not None:
        if ModelClass is FiLMPatchNet:
            print(f"[FiLM] Initialising from ScalePatchNet checkpoint: {ARG.MODEL_LOAD_PATH}")
            model = FiLMPatchNet.from_scalepatchnet_checkpoint(
                ARG.MODEL_LOAD_PATH,
                patchSize=ARG.PATCH_SIZE,
                outFeatures=14,
            )
        else:
            # For other models (e.g., ScalePatchNet), just load the state_dict.
            checkpoint = torch.load(ARG.MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded checkpoint into {ModelClass.__name__}")

    # Only trainable parameters (backbone is frozen in FiLMPatchNet, so only FiLM + classifier)
    model = torch.compile(model)
    
    # Count trainable parameters
    total_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    
    print("\n==========================")
    print("Trainable Parameters")
    print("==========================")
    print(total_trainable)
    
    # Show exactly what is trainable
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=ARG.LEARNING_RATE
    )
    
    return trainSet, model, optimizer



def loadValidationLoaderList():
    validLoaderList = list()

    chexpertValidationSet = ClassificationChexpertDataset("validate",False)
    chexpertValidLoader = prepareDataloader(chexpertValidationSet,ARG.BATCH_SIZE,training=False)

    validLoaderList.append((chexpertValidLoader,"val_Chexpert", "val", chexpertValidLoader))

    chexpertTestSet = ClassificationChexpertDataset(TEST,False)
    chexpertTestLoader = prepareDataloader(chexpertTestSet,ARG.BATCH_SIZE,training=False)

    validLoaderList.append((chexpertTestLoader,"test_Chexpert", TEST, chexpertValidLoader))

    return validLoaderList

def trainAndEval(device, totalEpochs, saveEvery, batchSize):
    trainSet, model, optimizer = loadTrainObjs()
    trainLoader = prepareDataloader(trainSet, batchSize, training=True)
    validLoaderList = loadValidationLoaderList()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,total_steps=ARG.EPOCH_NUM*len(trainLoader),max_lr=ARG.LEARNING_RATE,pct_start=0.05)

    trainer = Trainer(
        model=model,
        trainLoader=trainLoader,
        validLoaderList=validLoaderList,
        optimizer=optimizer,
        scheduler=scheduler,
        gpuId=device,
        saveEvery=saveEvery,
    )
    trainer.train(totalEpochs)

def evalOnly(device,modelPath):
    validLoaderList = loadValidationLoaderList()
    model = loadModelFromCheckpoint(modelPath)
    trainer = Trainer(
        model=model,
        trainLoader=None,
        validLoaderList=None,
        optimizer=None,
        scheduler=None,
        gpuId=device,
        saveEvery=-1,
    )
    for el in validLoaderList:
        trainer.validate(*el)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    if ARG.USE_WANDB: initWandB()

    device = 0
    if ARG.EVAL_ONLY:
        assert ARG.MODEL_LOAD_PATH is not None, "provide a path to model weights"
        evalOnly(
            device=device,
            modelPath=ARG.MODEL_LOAD_PATH,
        )
    else:
        trainAndEval(
            device=device,
            totalEpochs=ARG.EPOCH_NUM,
            saveEvery=1,
            batchSize=ARG.BATCH_SIZE,
        )