import torchvision
import torch.nn as nn
import torch


# ─────────────────────────────────────────────────────────────────────────────
#  ScalePatchNet  —  original model, kept verbatim
# ─────────────────────────────────────────────────────────────────────────────

class ScalePatchNet(nn.Module):
    def __init__(self,patchSize,outFeatures=512) -> None:
        super(ScalePatchNet, self).__init__()
        self.baseBackbone = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.baseBackbone.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.baseBackbone.classifier[1] = nn.Linear(in_features=1280, out_features=outFeatures, bias=True)
        self.patchSize = patchSize

    def forward(self,x):
        x = self.forwardRawPatches(x)
        x = torch.mean(x,dim=1)
        return x

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
        globalLogits = torch.mean(rawPatchLogits,dim=1)
        globalProb = torch.nn.functional.sigmoid(globalLogits)
        globalProb = globalProb.unsqueeze(1).unsqueeze(2)
        scaledPatches = rawPatchLogits * globalProb
        return scaledPatches

class FiLMPatchNet(nn.Module):

    def __init__(self, patchSize, outFeatures=14):
        super(FiLMPatchNet, self).__init__()
        self.patchSize = patchSize

        # ── Build backbone ────────────────────────────────────────────────
        backbone = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        # Grayscale input — same modification as ScalePatchNet
        backbone.features[0][0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        # Split into feature extractor and pooling; discard the ImageNet head
        self.eff_features = backbone.features   # all EfficientNet blocks
        self.eff_avgpool  = backbone.avgpool    # AdaptiveAvgPool2d(1, 1)

        # ── FiLM parameters ───────────────────────────────────────────────
        # γ and β are plain nn.Parameters — no conditioning network,
        # no coordinates.  They are global learnable feature-wise scalars.
        self.film_gamma = nn.Parameter(torch.ones(1280))   # init: γ = 1
        self.film_beta  = nn.Parameter(torch.zeros(1280))  # init: β = 0

        # ── Classifier ────────────────────────────────────────────────────
        # Same shape as ScalePatchNet's replaced head: Linear(1280, 14).
        # Call from_scalepatchnet_checkpoint() to copy the pretrained weights.
        self.classifier = nn.Linear(1280, outFeatures)
        
        print("\nTrainable Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
        

    def train(self, mode=True):
        super().train(mode)               # put classifier, FiLM params in train/eval
        self.eff_features.eval()          # backbone always stays in eval
        self.eff_avgpool.eval()
        return self


    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_features(self, patches):
        x = self.eff_features(patches)   # (N, 1280, H', W')
        x = self.eff_avgpool(x)          # (N, 1280, 1, 1)
        x = torch.flatten(x, 1)          # (N, 1280)
        return x

    # ── Shared patch-level forward ────────────────────────────────────────────

    def _run_patches(self, x):
        B = x.size(0)

        # Unfold exactly like ScalePatchNet.forwardRawPatches
        patches = (
            x.unfold(2, self.patchSize, self.patchSize)
             .unfold(3, self.patchSize, self.patchSize)
             .permute(0, 2, 3, 1, 4, 5)
             .reshape(-1, 1, self.patchSize, self.patchSize)
        )  # (B*P, 1, patchSize, patchSize)

        P = patches.size(0) // B

        # Frozen backbone — no gradient computation needed
        with torch.no_grad():
            feats = self.extract_features(patches)  # (B*P, 1280)

        # FiLM:  modulated = γ ⊙ feats + β
        # film_gamma and film_beta broadcast over the batch dimension
        feats = self.film_gamma * feats + self.film_beta  # (B*P, 1280)

        # Classifier
        logits = self.classifier(feats)   # (B*P, outFeatures)

        return logits.view(B, P, -1)      # (B, P, outFeatures)

    # ── Public API (drop-in compatible with ScalePatchNet) ────────────────────

    def forward(self, x):
        return self._run_patches(x).mean(dim=1)   # (B, outFeatures)

    def forwardRawPatches(self, x):
        return self._run_patches(x)               # (B, P, outFeatures)

    def forwardScaledPatches(self, x):
        rawPatchLogits = self._run_patches(x)
        globalLogits   = rawPatchLogits.mean(dim=1)
        globalProb     = torch.sigmoid(globalLogits).unsqueeze(1).unsqueeze(2)
        return rawPatchLogits * globalProb        # (B, P, outFeatures)

    # ── Checkpoint utilities ──────────────────────────────────────────────────

    @classmethod
    def from_scalepatchnet_checkpoint(
        cls,
        checkpoint_path,
        patchSize=64,
        outFeatures=14,
        device="cpu",
    ):
        model = cls(patchSize=patchSize, outFeatures=outFeatures)

        raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Strip torch.compile()'s "_orig_mod." prefix if present
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}

        translated = {}
        for k, v in state_dict.items():
            if k.startswith("baseBackbone.features."):
                translated[k.replace("baseBackbone.features.", "eff_features.")] = v
            elif k.startswith("baseBackbone.avgpool."):
                translated[k.replace("baseBackbone.avgpool.", "eff_avgpool.")] = v
            elif k == "baseBackbone.classifier.1.weight":
                translated["classifier.weight"] = v
            elif k == "baseBackbone.classifier.1.bias":
                translated["classifier.bias"] = v
            # everything else (dropout, etc.) is intentionally skipped

        missing, unexpected = model.load_state_dict(translated, strict=False)

        print(f"[FiLMPatchNet] Loaded from: {checkpoint_path}")
        print(f"  Transferred  : {len(translated)} tensors  (backbone + classifier)")
        print(f"  Missing      : {missing}")   # should only be film_gamma, film_beta
        print(f"  Unexpected   : {unexpected}")

        return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CLASS_LIST = [
    ScalePatchNet,
    FiLMPatchNet,
]

def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName:
            return ModelClass
    assert False, modelClassName + " not in " + str([x.__name__ for x in MODEL_CLASS_LIST])