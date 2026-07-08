from PIL.ImageFilter import Kernel
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  ScalePatchNet  —  original model, kept verbatim
# ─────────────────────────────────────────────────────────────────────────────

#adding - 3x3 or anyother size filter

class ScalePatchNet_filter(nn.Module):
    def __init__(self,patchSize,outFeatures=14) -> None:
        super(ScalePatchNet_filter, self).__init__()
        self.baseBackbone = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.baseBackbone.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.baseBackbone.classifier[1] = nn.Linear(in_features=1280, out_features=outFeatures, bias=True)
        self.patchSize = patchSize

        #==============================
        #Soft Consensus Gate
        #=============================
        kernel_size = 3 #change this to checnge the filter size (now it is 3x3)
        #you can set 1x1 or 5x5 or any other size filter

        self.consensus_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=14,
                out_channels=14,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=14
            ),

            nn.Conv2d(
                in_channels=14,
                out_channels=14,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self,x):
        patch_logits = self.forwardRawPatches(x)
        patch_logits = self.applyConsensusGate(patch_logits)
        global_logits = torch.mean(patch_logits,dim=1)
        return global_logits

    def applyConsensusGate(self,patch_logits):
        B = patch_logits.size(0)
        C = patch_logits.size(2)
        grid = int(patch_logits.size(1) ** 0.5)
        assert grid * grid == patch_logits.size(1)

        # (b,64,14) -> (B,14,8,8)
        gate_input = patch_logits.view(
            B, grid , grid , C
        ).permute(0,3,1,2)

        gate = self.consensus_conv(gate_input)

        gate = 2*gate
        gate = gate.permute(0, 2, 3, 1).reshape(B, grid * grid, C)
        gated_logits = patch_logits * gate
        return gated_logits

    def forwardRawPatches(self,x):
        batchSize = x.size()[0]
        x = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, self.patchSize, self.patchSize)
        x = self.baseBackbone(x)
        assert x.size()[0]%batchSize == 0
        patchCount = int(x.size()[0]/batchSize)
        batchList = torch.split(x,patchCount)
        x = torch.stack(batchList)
        return x

    def forwardScaledPatches(self,x):
        rawPatchLogits = self.forwardRawPatches(x)
        rawPatchLogits = self.applyConsensusGate(rawPatchLogits)
        globalLogits = torch.mean(rawPatchLogits,dim=1)
        globalProb = torch.sigmoid(globalLogits)
        scaled = rawPatchLogits * globalProb.unsqueeze(1)
        return scaled


MODEL_CLASS_LIST = [ #To make it easy to add other models
    ScalePatchNet_filter,
]

def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName:
            return ModelClass
    assert False, modelClassName + " not in " + str([x.__name__ for x in MODEL_CLASS_LIST])