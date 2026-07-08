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



class CoordFiLMPatchNet(nn.Module):
    """
    Coordinate-aware FiLM: an MLP maps normalized patch coordinates (x,y)
    to per-patch modulation parameters (γ, β).

    Backbone stays frozen; only the coordinate MLP and the classifier are trained.
    """
    def __init__(self, patchSize, outFeatures=14, coord_hidden_dim=64):
        super().__init__()
        self.patchSize = patchSize

        # ── Build frozen backbone ────────────────────────────────────────
        backbone = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        backbone.features[0][0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.eff_features = backbone.features
        self.eff_avgpool  = backbone.avgpool

        for param in self.eff_features.parameters():
            param.requires_grad = False
        for param in self.eff_avgpool.parameters():
            param.requires_grad = False

        # ── Coordinate MLP → (γ, β) ──────────────────────────────────────
        # Input: 2D normalized coordinates (x, y) ∈ [-1, 1]
        # Output: 2560 numbers → split into γ and β (1280 each)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, coord_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_hidden_dim, 2 * 1280)
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Linear(1280, outFeatures)

        # Initialize last MLP layer close to identity mapping (γ≈1, β≈0)
        with torch.no_grad():
            self.coord_mlp[-1].weight.data *= 0.01
            self.coord_mlp[-1].bias.data.zero_()

        print("\nTrainable Parameters (CoordFiLM):")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    # ── Freeze backbone in eval mode always ───────────────────────────────
    def train(self, mode=True):
        super().train(mode)
        self.eff_features.eval()
        self.eff_avgpool.eval()
        return self

    # ── Extract features from frozen backbone ────────────────────────────
    def extract_features(self, patches):
        x = self.eff_features(patches)   # (N, 1280, H', W')
        x = self.eff_avgpool(x)          # (N, 1280, 1, 1)
        return torch.flatten(x, 1)       # (N, 1280)

    # ── Core per‑patch forward with coordinate‑conditioned FiLM ──────────
    def _run_patches(self, x):
        B = x.size(0)

        # Unfold patches
        patches = (
            x.unfold(2, self.patchSize, self.patchSize)
             .unfold(3, self.patchSize, self.patchSize)
             .permute(0, 2, 3, 1, 4, 5)
             .reshape(-1, 1, self.patchSize, self.patchSize)
        )
        P = patches.size(0) // B
        grid = int(P ** 0.5)          # assumed square grid, e.g. 8x8

        # Normalized coordinates for every patch
        xs = torch.linspace(-1, 1, grid, device=x.device)
        ys = torch.linspace(-1, 1, grid, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coord = torch.stack([gx, gy], dim=-1)                    # (grid, grid, 2)
        coord = coord.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * P, 2)

        # Frozen backbone features
        with torch.no_grad():
            feats = self.extract_features(patches)               # (B*P, 1280)

        # Per‑patch γ,β
        gamma_beta = self.coord_mlp(coord)                       # (B*P, 2560)
        gamma, beta = gamma_beta.chunk(2, dim=1)                 # each (B*P, 1280)

        # Modulate + classify
        modulated = gamma * feats + beta
        logits = self.classifier(modulated)                      # (B*P, outFeatures)
        return logits.view(B, P, -1)

    # ── Public API (same as ScalePatchNet / FiLMPatchNet) ────────────────
    def forward(self, x):
        return self._run_patches(x).mean(dim=1)

    def forwardRawPatches(self, x):
        return self._run_patches(x)

    def forwardScaledPatches(self, x):
        raw = self._run_patches(x)
        global_prob = torch.sigmoid(raw.mean(dim=1)).unsqueeze(1).unsqueeze(2)
        return raw * global_prob

    # ── Checkpoint utility ───────────────────────────────────────────────
    @classmethod
    def from_scalepatchnet_checkpoint(
        cls,
        checkpoint_path,
        patchSize=64,
        outFeatures=14,
        device="cpu",
        coord_hidden_dim=64,
    ):
        """
        Build a CoordFiLMPatchNet and load backbone + classifier weights
        from a ScalePatchNet checkpoint. The coordinate MLP starts fresh.
        """
        model = cls(patchSize=patchSize,
                    outFeatures=outFeatures,
                    coord_hidden_dim=coord_hidden_dim)

        raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
        # film_gamma/beta are ignored, coord_mlp stays at init

        missing, unexpected = model.load_state_dict(translated, strict=False)
        print(f"[CoordFiLMPatchNet] Loaded from: {checkpoint_path}")
        print(f"  Transferred  : {len(translated)} tensors  (backbone + classifier)")
        print(f"  Missing      : {missing}")   # coord_mlp.*
        print(f"  Unexpected   : {unexpected}")
        return model.to(device)



#==================================================================

#                   uNFROZEN

#====================================================================

class CoordFiLMPatchNetUnfrozen(nn.Module):
    """
    Coordinate-aware FiLM – fully trainable version.
    Backbone is initialized from ImageNet-pretrained EfficientNet-V2-S
    and trained together with the coordinate MLP and classifier.
    """
    def __init__(self, patchSize, outFeatures=14, coord_hidden_dim=64):
        super().__init__()
        self.patchSize = patchSize

        # ── Build backbone (trainable, pretrained) ───────────────────────
        backbone = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        # Grayscale input
        backbone.features[0][0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.eff_features = backbone.features
        self.eff_avgpool  = backbone.avgpool
        # NO freezing – all parameters are trainable

        # ── Coordinate MLP → (γ, β) ──────────────────────────────────────
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, coord_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_hidden_dim, 2 * 1280)
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Linear(1280, outFeatures)

        # Initialize last MLP layer close to identity mapping (γ≈1, β≈0)
        with torch.no_grad():
            self.coord_mlp[-1].weight.data *= 0.01
            self.coord_mlp[-1].bias.data.zero_()

        print("\nTrainable Parameters (CoordFiLM Unfrozen):")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    # ── Feature extraction (trainable backbone) ──────────────────────────
    def extract_features(self, patches):
        x = self.eff_features(patches)   # (N, 1280, H', W')
        x = self.eff_avgpool(x)          # (N, 1280, 1, 1)
        return torch.flatten(x, 1)       # (N, 1280)

    # ── Core per‑patch forward with coordinate‑conditioned FiLM ──────────
    def _run_patches(self, x):
        B = x.size(0)

        # Unfold patches
        patches = (
            x.unfold(2, self.patchSize, self.patchSize)
             .unfold(3, self.patchSize, self.patchSize)
             .permute(0, 2, 3, 1, 4, 5)
             .reshape(-1, 1, self.patchSize, self.patchSize)
        )
        P = patches.size(0) // B
        grid = int(P ** 0.5)          # assumed square grid

        # Normalized coordinates for every patch
        xs = torch.linspace(-1, 1, grid, device=x.device)
        ys = torch.linspace(-1, 1, grid, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coord = torch.stack([gx, gy], dim=-1)                    # (grid, grid, 2)
        coord = coord.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * P, 2)

        # Backbone features (gradients flow through here now)
        feats = self.extract_features(patches)                   # (B*P, 1280)

        # Per‑patch γ,β
        gamma_beta = self.coord_mlp(coord)                       # (B*P, 2560)
        gamma, beta = gamma_beta.chunk(2, dim=1)                 # each (B*P, 1280)

        # Modulate + classify
        modulated = gamma * feats + beta
        logits = self.classifier(modulated)                      # (B*P, outFeatures)
        return logits.view(B, P, -1)

    # ── Public API ─────────────────────────────────────────────────────
    def forward(self, x):
        return self._run_patches(x).mean(dim=1)

    def forwardRawPatches(self, x):
        return self._run_patches(x)

    def forwardScaledPatches(self, x):
        raw = self._run_patches(x)
        global_prob = torch.sigmoid(raw.mean(dim=1)).unsqueeze(1).unsqueeze(2)
        return raw * global_prob

    # ── Checkpoint loading (from ScalePatchNet) ─────────────────────────
    @classmethod
    def from_scalepatchnet_checkpoint(
        cls,
        checkpoint_path,
        patchSize=64,
        outFeatures=14,
        device="cpu",
        coord_hidden_dim=64,
    ):
        """
        Build a CoordFiLMPatchNetUnfrozen and load backbone + classifier
        weights from a ScalePatchNet checkpoint. The coordinate MLP starts fresh.
        Backbone is NOT frozen (gradients enabled).
        """
        model = cls(patchSize=patchSize,
                    outFeatures=outFeatures,
                    coord_hidden_dim=coord_hidden_dim)

        raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
        # film_gamma/beta ignored; coord_mlp stays at initialisation

        missing, unexpected = model.load_state_dict(translated, strict=False)
        print(f"[CoordFiLMPatchNetUnfrozen] Loaded from: {checkpoint_path}")
        print(f"  Transferred  : {len(translated)} tensors  (backbone + classifier)")
        print(f"  Missing      : {missing}")   # coord_mlp.*
        print(f"  Unexpected   : {unexpected}")
        return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CLASS_LIST = [
    ScalePatchNet,
    CoordFiLMPatchNet,
    CoordFiLMPatchNetUnfrozen,
]


def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName:
            return ModelClass
    assert False, modelClassName + " not in " + str([x.__name__ for x in MODEL_CLASS_LIST])