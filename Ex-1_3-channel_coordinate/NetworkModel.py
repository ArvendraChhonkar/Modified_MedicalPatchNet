import torchvision
import torch.nn as nn
import torch

class ScalePatchNet(nn.Module):
    def __init__(self,patchSize,outFeatures=512) -> None:
        super(ScalePatchNet, self).__init__()
        self.baseBackbone = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.baseBackbone.features[0][0] = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) #3 channels 1 for patches and other 2 for coordinates
        self.baseBackbone.classifier[1] = nn.Linear(in_features=1280, out_features=outFeatures, bias=True)
        self.patchSize = patchSize

    def forward(self,x): #normal forward to train 
        x = self.forwardRawPatches(x)
        x = torch.mean(x,dim=1)
        return x
    
    def forwardRawPatches(self,x): #forward raw patches
        batchSize = x.size()[0]
        #put patches in batch dimension
        x = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, self.patchSize, self.patchSize)
        x = self.baseBackbone(x)
        assert x.size()[0]%batchSize == 0
        #recreate image batches
        patchCount = int(x.size()[0]/batchSize)
        batchList = torch.split(x,patchCount)
        x = torch.stack(batchList)
        return x
    
    def forwardScaledPatches(self,x):
        rawPatchLogits = self.forwardRawPatches(x)
        globalLogits = torch.mean(rawPatchLogits,dim=1)
        globalProb = torch.nn.functional.sigmoid(globalLogits)
        globalProb = globalProb.unsqueeze(1).unsqueeze(2)
        scaledPatches = rawPatchLogits * globalProb
        return scaledPatches

    
MODEL_CLASS_LIST = [ #To make it easy to add other models
    ScalePatchNet,
]

def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName: return ModelClass
    assert False, modelClassName + "not in" + str([x.__name__ for x in MODEL_CLASS_LIST])


def adapt_first_conv(new_model, old_weight_path):
    # Load old state dict (1‑channel model)
    old_state = torch.load(old_weight_path, map_location='cpu')
    old_state = {k.replace('_orig_mod.', ''): v for k, v in old_state.items()}

    # Extract the old first conv weight
    key = 'baseBackbone.features.0.0.weight'   # this is the name inside the loaded dict
    old_conv_weight = old_state[key]            # shape (24, 1, 3, 3)

    # Prepare new weight (24, 3, 3, 3)
    new_conv_weight = torch.zeros_like(new_model.baseBackbone.features[0][0].weight.data)
    # Copy the old kernel into channel 0
    new_conv_weight[:, 0:1, :, :] = old_conv_weight
    # Channels 1 and 2 remain zero

    # Copy into the model
    new_model.baseBackbone.features[0][0].weight.data = new_conv_weight

    # Also copy the rest of the backbone (excluding the classifier)
    new_state = new_model.state_dict()
    for k, v in old_state.items():
        if k in new_state and 'classifier' not in k and k != key:
            new_state[k] = v
    new_model.load_state_dict(new_state)
    print("First conv adapted and other weights copied.")