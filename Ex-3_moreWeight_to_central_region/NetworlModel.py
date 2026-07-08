import torchvision
import torch.nn as nn
import torch

    
class ScalePatchNet(nn.Module):
    def __init__(self, patchSize, outFeatures=512) -> None:
        super(ScalePatchNet, self).__init__()
        self.baseBackbone = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.baseBackbone.features[0][0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.baseBackbone.classifier[1] = nn.Linear(
            in_features=1280, out_features=outFeatures, bias=True
        )
        self.patchSize = patchSize

    def forward(self, x):
        patch_logits = self.forwardRawPatches(x)
        global_logits = torch.mean(patch_logits, dim=1)
        return global_logits

    def forwardRawPatches(self, x):
        batchSize = x.size(0)

        x = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, self.patchSize, self.patchSize)

        x = self.baseBackbone(x)

        assert x.size(0) % batchSize == 0
        patchCount = int(x.size(0) / batchSize)
        batchList = torch.split(x, patchCount)
        x = torch.stack(batchList)
        return x  # [B, P, C]

    def forwardScaledPatches(self, x):
        rawPatchLogits = self.forwardRawPatches(x)
        globalLogits = torch.mean(rawPatchLogits, dim=1)
        globalProb = torch.sigmoid(globalLogits)
        globalProb = globalProb.unsqueeze(1).unsqueeze(2)
        scaledPatches = rawPatchLogits * globalProb
        return scaledPatches

    def forwardWeightedPatches(self, x, patchWeights, normalize_weights=True, min_weight=0.0):
        rawPatchLogits = self.forwardRawPatches(x)  # [B, P, C]

        if min_weight > 0:
            patchWeights = torch.clamp(patchWeights, min=min_weight, max=1.0)
        else:
            patchWeights = torch.clamp(patchWeights, min=0.0, max=1.0)

        weightExp = patchWeights.unsqueeze(-1)  # [B, P, 1]
        weightedPatchLogits = rawPatchLogits * weightExp

        if normalize_weights:
            denom = weightExp.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            globalLogits = weightedPatchLogits.sum(dim=1) / denom
        else:
            globalLogits = weightedPatchLogits.mean(dim=1)

        return {
            "global_logits": globalLogits,
            "weighted_patch_logits": weightedPatchLogits,
            "raw_patch_logits": rawPatchLogits,
            "patch_weights": patchWeights,
        }

    def imageWeightMapToPatchWeights(self, weightMap):
        # weightMap: [B,1,H,W], values in [0,1]
        B, C, H, W = weightMap.shape
        assert C == 1
        assert H % self.patchSize == 0 and W % self.patchSize == 0

        patchWeights = weightMap.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        patchWeights = patchWeights.contiguous().view(B, 1, -1, self.patchSize, self.patchSize)
        patchWeights = patchWeights.mean(dim=(-1, -2)).squeeze(1)  # [B, P]
        return patchWeights


MODEL_CLASS_LIST = [
    ScalePatchNet,
]


def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName:
            return ModelClass
    assert False, modelClassName + " not in " + str([x.__name__ for x in MODEL_CLASS_LIST])