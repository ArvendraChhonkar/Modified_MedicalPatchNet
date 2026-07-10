# Modified_MedicalPatchNet

**Explainable Chest X-Ray AI — classifying 14 diseases and generating self-explanatory localization heatmaps.**

This repository documents a series of research experiments extending **MedicalPatchNet**, a patch-based, self-explainable deep learning architecture for chest X-ray disease classification and localization. Each experiment tests a different hypothesis about how to improve the base architecture's classification accuracy, localization quality (heatmaps), or robustness, while preserving its core explainability property: predictions are generated directly from per-patch logits, with no post-hoc saliency method (e.g. Grad-CAM) required.

All experiments are built on the same backbone (EfficientNetV2-S, ImageNet-1K pretrained, grayscale-adapted) and the same dataset (CheXpert, evaluated in part against CheXlocalize segmentation masks), so results across experiments are directly comparable.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Base Architecture: Original_Medical_PatchNet](#3-base-architecture-original_medical_patchnet)
4. [Common Experimental Setup](#4-common-experimental-setup)
5. [Experiment 1 — 3-Channel Coordinate Input](#5-experiment-1--3-channel-coordinate-input)
6. [Experiment 2 — Architectural Changes (FiLM, Coordinate-FiLM, Mini-Classifiers)](#6-experiment-2--architectural-changes-film-coordinate-film-mini-classifiers)
7. [Experiment 3 — Spatial Prior Patch Weighting](#7-experiment-3--spatial-prior-patch-weighting)
8. [Experiment 4 — Patch-Consensus Filtering (1x1/3x3/5x5)](#8-experiment-4--patch-consensus-filtering-1x13x35x5)
9. [Cross-Experiment Comparison](#9-cross-experiment-comparison)
10. [Overall Findings](#10-overall-findings)
11. [Setup and Installation](#11-setup-and-installation)
12. [Usage](#12-usage)
13. [Future Work](#13-future-work)
14. [Citation](#14-citation)
15. [License](#15-license)

---

## 1. Project Overview

MedicalPatchNet classifies chest X-rays for 14 CheXpert disease labels and localizes disease regions without any separate post-hoc explainability step. It does this by:

1. Splitting each 512x512 chest X-ray into an 8x8 grid of 64 non-overlapping 64x64 patches.
2. Running every patch independently through a **shared** EfficientNetV2-S backbone to get 14-class disease logits per patch.
3. Averaging the 64 sets of patch logits into a single global, image-level prediction (via sigmoid).
4. Rendering the un-averaged per-patch logits directly as an 8x8 heatmap — since each value is literally "what does the network think this specific region shows," no Grad-CAM or similar saliency method is needed.

The problem statement motivating this whole repository: traditional CNNs are accurate but are black boxes, and bolting on explainability after the fact (Grad-CAM, Grad-CAM++) does not always produce clinically reliable results. MedicalPatchNet is explainable by construction, but its simple architecture (frozen patch size, one linear classifier, uniform-mean aggregation) leaves several open questions — does the model need spatial awareness? Can feature modulation improve it without breaking its pretrained features? Does treating all patches equally hurt performance? Experiments 1 through 4 in this repository investigate exactly these questions, one variable at a time.

---

## 2. Repository Structure

```
Modified_MedicalPatchNet/
│
├── Original_Medical_PatchNet/              # Baseline: unmodified MedicalPatchNet
│   ├── NetworkModel.py                     # ScalePatchNet (base architecture)
│   ├── ChexpertDataset.py                  # CheXpert dataset loader
│   ├── trainClassification.py              # Training script
│   ├── evalClassification.py               # Evaluation script
│   ├── tune_heatmap_threshold_opt.py        # Heatmap threshold tuning
│   ├── figureGeneration.py                 # Figure/plot generation
│   ├── imgRetrivalUtil.py                  # Image retrieval utilities
│   ├── utilFunc.py                         # Shared utility functions
│   ├── argParser.py                        # CLI argument parsing
│   ├── environment.yml                     # Conda environment spec
│   ├── runTraining.sh / runEval.sh          # Shell scripts for training/eval
│   ├── preprocessing/                      # Dataset preprocessing scripts
│   └── savedModels/                        # Pretrained/trained checkpoints
│
├── Ex-1_3-channel_coordinate/              # Experiment 1: coordinate channel concatenation
│   ├── NetworkModel.py
│   ├── ChexpertDataset.py
│   ├── trainClassification.py
│   ├── evalClassification.py
│   ├── utilFunc.py
│   └── argParser.py
│
├── Ex-2_Architectural Changes/             # Experiment 2: FiLM / coordinate-FiLM / mini-classifiers
│   └── NetworkModel.py
│
├── Ex-3_moreWeight_to_central_region/      # Experiment 3: inference-time spatial-prior weighting
│   ├── NetworlModel.py
│   ├── patchWeighting.py
│   ├── evalClassification.py
│   └── figureGeneration.py
│
├── Ex-4_filers(1x1_3x3_5x5)/                # Experiment 4: learned patch-consensus gating
│   ├── default3x3/
│   │   └── NetworkModel.py
│   ├── default_filter_with_miniclassifiers/
│   │   └── NetworkModel.py
│   └── implemented 3x3 filter twice - really high perecision f1 specificity and senesitivity/
│       └── NetworkModel.py
│
├── LICENSE                                 # MIT License
└── README.md                               # This file
```

---

## 3. Base Architecture: `Original_Medical_PatchNet`

This folder holds the unmodified reference implementation that every experiment branches from.

### 3.1 Pipeline

```
Input ChestX-ray (1, 512, 512) grayscale
        │
        ▼
Unfold into 8x8 grid → 64 patches, each (1, 64, 64)
        │
        ▼
Shared EfficientNetV2-S backbone (grayscale-adapted first conv layer)
        │
        ▼
Linear(1280 → 14) classifier head
        │
        ▼
Reshape → (B, 64, 14) per-patch, per-disease logits
        │
        ▼
Mean over 64 patches → (B, 14) global logits
        │
        ▼
Sigmoid → 14 disease probabilities
```

### 3.2 Core Model Code (`ScalePatchNet`)

```python
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
        x = self.forwardRawPatches(x)
        x = torch.mean(x, dim=1)
        return x

    def forwardRawPatches(self, x):
        batchSize = x.size()[0]
        x = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, self.patchSize, self.patchSize)
        x = self.baseBackbone(x)
        assert x.size()[0] % batchSize == 0
        patchCount = int(x.size()[0] / batchSize)
        batchList = torch.split(x, patchCount)
        x = torch.stack(batchList)
        return x

    def forwardScaledPatches(self, x):
        rawPatchLogits = self.forwardRawPatches(x)
        globalLogits = torch.mean(rawPatchLogits, dim=1)
        globalProb = torch.sigmoid(globalLogits)
        globalProb = globalProb.unsqueeze(1).unsqueeze(2)
        return rawPatchLogits * globalProb
```

### 3.3 Datasets

| Dataset | Role |
|---|---|
| **CheXpert** | >223,000 chest X-rays from >65,000 patients, 14 disease labels, multi-label classification. Used for training and classification evaluation. |
| **CheXlocalize** | Extends CheXpert with expert-annotated segmentation masks for several diseases, enabling quantitative localization metrics (IoU, Dice, Hit Rate). |

### 3.4 Utilities in This Folder

- `preprocessing/` — scripts to convert raw CheXpert/CheXlocalize downloads into the tensors used by `ChexpertDataset.py`.
- `tune_heatmap_threshold_opt.py` — tunes the probability threshold used to binarize heatmaps for IoU/Dice computation.
- `imgRetrivalUtil.py` — utilities for retrieving and matching images/annotations across datasets.
- `runTraining.sh` / `runEval.sh` — shell wrappers around `trainClassification.py` and `evalClassification.py` for repeatable command-line runs.
- `environment.yml` — the Conda environment definition (Python, PyTorch, Torchvision, NumPy, OpenCV) used across all experiments.

---

## 4. Common Experimental Setup

To keep comparisons fair, most experiments share this training configuration:

| Setting | Value |
|---|---|
| Dataset | CheXpert |
| Training subset | ~29,000 images, patients 30,001–40,000 |
| Input image size | 512x512 |
| Patch size | 64x64 (→ 8x8 grid, 64 patches) |
| Backbone | EfficientNetV2-S (ImageNet-1K pretrained) |
| Disease classes | 14 |
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| Epochs | 20 |
| Loss function | Binary Cross-Entropy (BCE) |
| Evaluation metrics | AUROC, F1-score, Sensitivity, Specificity, Dice Score, Mean IoU |
| Hardware | L4 and RTX 2050 GPUs, CUDA |
| Software | Python, PyTorch, Torchvision, NumPy, OpenCV |

Fixing the patient subset and training protocol across experiments means that any measured performance differences can be attributed to the architectural or algorithmic change under test, not to data variation.

---

## 5. Experiment 1 — 3-Channel Coordinate Input

**Folder:** `Ex-1_3-channel_coordinate/`

### 5.1 Objective

Tests whether giving the network explicit knowledge of *where* in the radiograph a patch is located — via raw coordinate channels — improves classification and localization.

### 5.2 Method

Each 64x64 grayscale patch is extended from 1 channel to 3 channels:

| Channel | Content |
|---|---|
| 0 | Original grayscale X-ray intensities |
| 1 | Normalized horizontal (x) coordinate map |
| 2 | Normalized vertical (y) coordinate map |

The modified backbone's first convolution accepts 3 input channels instead of 1, and training proceeds with the pretrained weights from the original paper as a starting point, with the backbone initially frozen. Because two of the three input channels are coordinate grids rather than pixel intensities, saliency methods like Grad-CAM and Grad-CAM++ become unreliable (they assume all channels carry visual content), so **localization was evaluated with IoU metrics rather than saliency maps.**

### 5.3 Progressive (Batched) Training Strategy

Because CheXpert is large, training was split into successive batches of ~10,000 patients each, and the model was evaluated after each stage:

- Batch 1: patients 1–10,000
- Batch 2: patients 10,001–20,000
- Batch 3: patients 20,001–30,000
- Batch 4: patients 30,001–40,000

### 5.4 Training Dynamics (Batch 1, across epochs)

| Metric | Epoch 1 | Epoch 6 | Epoch 12 |
|---|---|---|---|
| Macro AUROC | 0.571 | 0.616 | 0.564 |
| Macro F1 | 0.242 | 0.290 | 0.266 |
| Average Sensitivity | 0.144 | 0.146 | 0.152 |
| Average Specificity | 0.931 | 0.927 | 0.925 |

Macro AUROC peaked at Epoch 6; training beyond that point hurt both AUROC and F1, indicating overfitting. The sharpest class-wise degradation was in **No Finding**, which fell from 0.680 (Epoch 6) to 0.473 (Epoch 12) — evidence the model drifted toward over-predicting disease as training continued. Other notable drops: Atelectasis (−0.148), Lung Opacity (−0.123), Pleural Effusion (−0.086).

### 5.5 Classification and Localization Results

**Patients 1–10,000 (Baseline vs. Coordinate-aware model):**

| Metric | Baseline | Coordinate Model |
|---|---|---|
| Accuracy | 0.816 | 0.386 |
| Precision | 0.865 | 0.379 |
| Recall | 0.825 | 0.021 |
| F1 Score | 0.844 | 0.039 |
| Mean IoU | 0.069 | 0.021 |

**Patients 10,001–20,000:** Recall fell sharply across most diseases (e.g., Airspace Opacity F1 0.890 → 0.280; Atelectasis F1 0.821 → 0.055; Cardiomegaly F1 0.736 → 0.194; Enlarged Cardiomediastinum and Consolidation recall dropped to zero). A few localization scores improved slightly (Lung Lesion +0.12 IoU, Support Devices +0.03 IoU, Pneumothorax +0.05 IoU), but overall positive-case mean IoU stayed essentially flat (Baseline 0.1067 vs. Proposed 0.1020).

**Patients 20,001–30,000 and 30,001–40,000 (progressive training):**

| Metric | Baseline | After 30k patients | After 40k patients |
|---|---|---|---|
| Accuracy | 0.824 | 0.515 → 0.502 | 0.502 → 0.515 |
| Precision | 0.734 | 0.458 → 0.545 | 0.545 → 0.577 |
| Recall | 0.810 | 0.107 → 0.299 | 0.299 → 0.397 |
| F1 Score | 0.749 | 0.161 → 0.286 | 0.286 → 0.334 |
| ROC AUC | 0.876 | 0.634 → 0.650 | 0.650 → 0.647 |
| mIoU (positive) | 0.107 | 0.102 → 0.092 | 0.092 → 0.098 |

Recall and F1 continued improving with more training data, but the gap to baseline remained large throughout.

### 5.6 Discussion and Conclusion

Appending raw coordinate channels **did not improve** MedicalPatchNet. Global spatial context was available in principle, but the model could not integrate it effectively with local visual features — classification stayed consistently below baseline and localization changed only marginally. The gradual recall improvement across patient batches suggests the *idea* of spatial context might still be valuable, but **direct channel concatenation is the wrong fusion mechanism.** This finding directly motivated Experiment 2's shift to FiLM-based modulation as an alternative fusion strategy.

---

## 6. Experiment 2 — Architectural Changes (FiLM, Coordinate-FiLM, Mini-Classifiers)

**Folder:** `Ex-2_Architectural Changes/`

### 6.1 Objective

Following Experiment 1's conclusion that raw coordinate concatenation doesn't work, Experiment 2 tests **Feature-wise Linear Modulation (FiLM)** as a more structured way to inject conditioning signal (global or spatial) into the pretrained backbone's features, without disturbing those pretrained features. It also tests classifier-head depth independently.

Planned research stages for this line of work:
1. Baseline MedicalPatchNet evaluation
2. Static FiLM feature modulation
3. Coordinate-conditioned FiLM
4. Coordinate representation study
5. Enhanced patch classifier
6. Residual calibration network
7. Learnable patch weighting network
8. Aggregation strategy comparison
9. Integration of best-performing modules
10. Comprehensive ablation study

### 6.2 Phase-1 Version 1 (V1) — Static FiLM, Backbone + Classifier Frozen

A FiLM layer applies a learned, **global** (non-spatial) affine transform to the 1280-dim feature vector produced by the backbone, before the classifier:

```python
self.film_gamma = nn.Parameter(torch.ones(1280))    # init: γ = 1
self.film_beta  = nn.Parameter(torch.zeros(1280))   # init: β = 0
...
feats = self.film_gamma * feats + self.film_beta    # FiLM modulation
logits = self.classifier(feats)
```

Both the EfficientNetV2-S backbone and the classifier head are frozen; **only `film_gamma` and `film_beta` are trained.** The classifier weights are transferred directly from a pretrained `ScalePatchNet` checkpoint via `from_scalepatchnet_checkpoint()`.

**Result:** Preserved the original model's behavior almost perfectly, with mean AUROC improving slightly:

| Model | Mean AUROC |
|---|---|
| Baseline | 0.64097 |
| Phase-1 V1 | 0.64249 (Δ = +0.00152) |

Largest per-disease AUROC gains: Pleural Other (+0.0076), Fracture (+0.0050), Pneumonia (+0.0037), Edema (+0.0036), Pleural Effusion (+0.0033). Most other classes were essentially unchanged.

**Win-count analysis** (number of the 14 diseases each model led on by AUROC):

| Model | Number of Wins |
|---|---|
| Baseline | 2 |
| Phase-1 V1 | 6 |
| Phase-1 V2 | 4 |
| Tie | 1 (of 14 total) |

**Conclusion: Phase-1 V1 is safe — it preserves the original model's behavior while producing small, consistent AUROC gains, and was chosen as the foundation for later coordinate-conditioning and spatial-embedding experiments.**

### 6.3 Phase-1 Version 2 (V2) — FiLM + Trainable Classifier, Backbone Frozen

Same FiLM layer, but the classifier head is retrained jointly rather than reused from checkpoint.

**Result:** Larger, more aggressive shifts in decision boundaries. Gains on some diseases came paired with losses on others:

| Disease | AUROC Change |
|---|---|
| Consolidation | +0.067 |
| Pneumonia | +0.052 |
| No Finding | +0.022 |
| Fracture | −0.115 |
| Pleural Other | −0.096 |
| Lung Lesion | −0.083 |
| Edema | −0.053 |
| Enlarged Cardiomediastinum | −0.040 |

**Conclusion:** Retraining the classifier alongside FiLM trades robustness for specialization — it improves specific diseases (Pneumonia, Consolidation) at the cost of others, and won fewer disease-level comparisons overall (4/14) than V1 (6/14).

**Overall Phase-1 ranking:** 1. Phase-1 V1, 2. Baseline, 3. Phase-1 V2.

### 6.4 Coordinate-Conditioned FiLM (`CoordFiLMPatchNet`)

Extends the static FiLM idea to be **spatially aware**: instead of a single global `(γ, β)` pair, a small MLP maps each patch's normalized `(x, y)` grid coordinate to a *per-patch* `(γ, β)` pair.

```python
self.coord_mlp = nn.Sequential(
    nn.Linear(2, coord_hidden_dim),
    nn.ReLU(inplace=True),
    nn.Linear(coord_hidden_dim, 2 * 1280)   # → split into per-patch γ, β
)
```

The last MLP layer is initialized near-identity (`weight *= 0.01`, `bias = 0`) so training starts close to the unmodulated baseline. Two variants exist:

- **`CoordFiLMPatchNet`** — backbone frozen (`eff_features`/`eff_avgpool.eval()`, `requires_grad=False`); only `coord_mlp` and the classifier train.
- **`CoordFiLMPatchNetUnfrozen`** — identical coordinate-MLP design, but the backbone is fully trainable alongside it.

Both support `from_scalepatchnet_checkpoint()` to initialize backbone + classifier weights from a pretrained `ScalePatchNet`, leaving `coord_mlp` freshly initialized. This is framed as a more structured alternative to Experiment 1's raw coordinate-channel concatenation, since the coordinate signal here only ever rescales/shifts existing backbone features rather than being fed through the convolutional stack as new pixel content.

### 6.5 Multiple Mini-Classifiers (`ScalePatchNet_MiniMiniClassifier`)

Replaces the single `Linear(1280, 14)` head with a deeper cascade:

```python
self.baseBackbone.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 512), nn.ReLU(inplace=True),
    nn.Linear(512, 256),  nn.ReLU(inplace=True),
    nn.Linear(256, outFeatures)
)
```

This tests whether a progressively-compressing classifier head (1280 → 512 → 256 → 14) extracts more disease-relevant signal than a single linear layer, independent of any FiLM modulation.

### 6.6 Variant Summary Table

| Variant | Backbone | Classifier | Modulation | Result |
|---|---|---|---|---|
| Baseline (`ScalePatchNet`) | frozen (pretrained) | linear head | none | Reference; wins 2/14 disease comparisons |
| FiLM V1 | frozen | frozen (reused) | global static FiLM | **Best balance**; ΔAUROC +0.00152; wins 6/14 |
| FiLM V2 | frozen | trainable | global static FiLM | High variance; wins 4/14 |
| CoordFiLMPatchNet | frozen | linear head | per-patch coordinate-conditioned FiLM | Structured alternative to raw coord concat (Ex-1) |
| CoordFiLMPatchNetUnfrozen | trainable | linear head | per-patch coordinate-conditioned FiLM | All parameters jointly trained |
| Mini-classifier | frozen | 3-layer MLP head | none | Deeper head tested independently of FiLM |

---

## 7. Experiment 3 — Spatial Prior Patch Weighting

**Folder:** `Ex-3_moreWeight_to_central_region/`

### 7.1 Motivation

MedicalPatchNet averages patch predictions **equally**, but chest abnormalities are not evenly distributed across the image — the lungs, heart, and mediastinum occupy distinct central regions, while image borders contribute little diagnostic information.

### 7.2 Hypothesis

Most diagnostically relevant anatomy sits near the center of a chest radiograph. Upweighting central patches and downweighting peripheral ones might improve localization and prediction confidence — **without needing to retrain anything.**

### 7.3 Methodology

Unlike Experiment 1's retraining approach, this modification is applied **entirely at inference time**; the backbone is left completely untouched.

1. Generate a soft weight map for the input image.
2. Assign larger weights to central image regions.
3. Convert the image-level weight map into patch-level weights.
4. Multiply each patch prediction by its corresponding weight.
5. Aggregate weighted patch logits to obtain the final prediction.

The weight map combines three components — image intensity, local contrast, and distance from the image center:

\[ W = 0.55I + 0.25C + 0.20P \]

where \(I\) is normalized image intensity, \(C\) is local contrast, and \(P\) is the center-prior map. Patch weights are computed by averaging all pixel weights within each patch (implemented in `patchWeighting.py`).

### 7.4 Experimental Setup

Ran on a subset of 50 test images, no parameter updates. The same pretrained MedicalPatchNet was evaluated under two conditions: raw (unweighted) patch aggregation and weighted patch aggregation, using `evalClassification.py` and `figureGeneration.py` for visualization.

### 7.5 Results

| Task | Prob (Raw) | Prob (Weighted) | IoU (Raw) | IoU (Weighted) |
|---|---|---|---|---|
| Airspace Opacity | 0.6665 | 0.6652 | 0.0606 | 0.0607 |
| Atelectasis | 0.5298 | 0.5284 | 0.0337 | 0.0336 |
| Cardiomegaly | 0.4688 | 0.4770 | 0.0587 | 0.0585 |
| Consolidation | 0.3555 | 0.3664 | 0.0000 | 0.0000 |
| Edema | 0.1242 | 0.1239 | 0.0000 | 0.0000 |
| Enlarged Cardiomediastinum | 0.5926 | 0.5965 | 0.1045 | 0.1052 |
| Lung Lesion | 0.1802 | 0.1899 | 0.0028 | 0.0034 |
| Pleural Effusion | 0.2807 | 0.2814 | 0.0092 | 0.0091 |
| Pneumothorax | 0.1800 | 0.1838 | 0.0000 | 0.0000 |
| Support Devices | 0.7135 | 0.7067 | 0.0488 | 0.0486 |

Changes in both classification confidence and localization were **marginal** — for most diseases, weighted and unweighted predictions were essentially identical.

### 7.6 Discussion

The weighting approach successfully injected spatial priors into inference without touching the network or retraining, but the **center-prior assumption produced no meaningful improvement**. The core problem: disease locations vary substantially by condition. Cardiomegaly sits near the cardiac silhouette (central), but pleural effusion, pneumothorax, and lung lesions tend toward the periphery — a single universal center-weighted map risks suppressing exactly the patches that matter most for some conditions.

### 7.7 Conclusion and Future Direction

Patch-level weighting can be grafted onto MedicalPatchNet without retraining, but a generic center prior produced only negligible improvements. The logical next step (not yet implemented) is **disease-specific probability maps** built from CheXlocalize segmentation annotations — rather than a single center prior, each disease would get its own prior derived by aggregating training masks, capturing the most probable anatomical location for that specific pathology.

---

## 8. Experiment 4 — Patch-Consensus Filtering (1x1/3x3/5x5)

**Folder:** `Ex-4_filers(1x1_3x3_5x5)/`

### 8.1 Objective

Introduces a **learned Soft Consensus Gate** that inspects each patch's local neighborhood in logit-space (separately per disease class) and suppresses or boosts patches based on whether their neighbors agree, before mean-aggregation. Suggested by the project guide as a way to reject spatially-isolated outlier patches (e.g., imaging artifacts, rib overlap) that would otherwise skew the global average.

### 8.2 Consensus Gate Design

```python
self.consensus_conv = nn.Sequential(
    nn.Conv2d(14, 14, kernel_size=3, padding=1, groups=14),  # depthwise, per-class
    nn.Conv2d(14, 14, kernel_size=1),                         # pointwise recalibration
    nn.Sigmoid()
)

def applyConsensusGate(self, patch_logits):
    B, P, C = patch_logits.shape
    grid = int(P ** 0.5)
    gate_input = patch_logits.view(B, grid, grid, C).permute(0, 3, 1, 2)  # (B,14,8,8)
    gate = 2 * self.consensus_conv(gate_input)   # rescale sigmoid output to (0,2)
    gate = gate.permute(0, 2, 3, 1).reshape(B, grid * grid, C)
    return patch_logits * gate

def forward(self, x):
    patch_logits = self.forwardRawPatches(x)
    patch_logits = self.applyConsensusGate(patch_logits)
    return torch.mean(patch_logits, dim=1)
```

The depthwise convolution (`groups=14`) ensures each disease's spatial-agreement computation is independent — the gate for "Cardiomegaly" never mixes with the gate for "Pneumothorax," preserving per-disease interpretability.

### 8.3 Three Variants Tested

| Folder | Classifier Head | Gate Formula | Result |
|---|---|---|---|
| `default3x3/` | `Linear(1280, 14)` | `logits * gate`, single 3x3 pass, gate ∈ (0,2) | Good, stable metrics — solid baseline |
| `default_filter_with_miniclassifiers/` | 3-layer MLP (1280→512→256→14) | `logits * gate`, single 3x3 pass | Weaker metrics than the linear-head baseline |
| `implemented 3x3 filter twice.../` | `Linear(1280, 14)` | `logits * (1 + gate)`, residual, gate computed twice per forward | **Best result** — high precision, F1, sensitivity, specificity |

### 8.4 Why the "Filter Twice" Variant Wins

The winning variant uses `logits * (1 + gate)` instead of `logits * gate`. With `gate` bounded to `(0, 2)`, this makes the effective multiplier `(1, 3)` — meaning it can only ever **preserve or boost** a patch's logit, never suppress it below its original value, unlike the plain multiplicative form (multiplier range `(0, 2)`, which can zero out a patch entirely). This residual-style formulation behaves like an easier optimization target (similar in spirit to ResNet skip connections): the network only has to learn a *correction* on top of a working baseline rather than a scale factor from scratch, which appears to make training more stable.

### 8.5 Filter-Size Sweep Context

The folder name `(1x1_3x3_5x5)` reflects that `kernel_size` in `consensus_conv` is a one-line configurable variable, intended to be swept across 1, 3, and 5. All three variants documented above specifically used `kernel_size = 3`; the single 3x3 filter (applied via the residual/double-call pattern) outperformed both the deeper-classifier combination and (per the experiment log) the alternative default multi-scale filter-bank attempts.

### 8.6 Complexity

The consensus gate is extremely cheap relative to the ~20.2M-parameter EfficientNetV2-S backbone:

| Component | Parameters (kernel_size=3) |
|---|---|
| Depthwise conv (14 channels, 3x3, groups=14) | 140 |
| Pointwise conv (14→14, 1x1) | 210 |
| **Total gate parameters** | **350** (~0.002% of backbone) |

---

## 9. Cross-Experiment Comparison

| Experiment | Mechanism | Trained end-to-end? | Retraining required? | Outcome |
|---|---|---|---|---|
| Ex-1: Coordinate channels | Raw (x,y) grids concatenated as extra input channels | Yes | Yes | **Failed** — hurt classification and localization vs. baseline |
| Ex-2: FiLM (V1) | Global learnable feature-wise affine modulation, frozen backbone+classifier | Partial (FiLM params only) | Minimal | **Succeeded** — small consistent AUROC gains, safest change |
| Ex-2: FiLM (V2) | Same FiLM + trainable classifier | Partial | Yes (classifier) | Mixed — gains on some diseases, losses on others |
| Ex-2: Coordinate-FiLM | Per-patch (x,y)-conditioned FiLM via small MLP | Partial or full | Partial | Proposed structured fix for Ex-1's failure mode |
| Ex-3: Center-weighting | Fixed, hand-designed spatial prior, inference-only | No | No | **Negligible effect** — center prior too generic across diseases |
| Ex-4: Consensus gating | Learned depthwise-conv neighborhood agreement gate on patch logits | Yes (gate + classifier) | Yes | **Best result** — especially the residual "filter twice" formulation |

### Progression of Spatial-Awareness Strategies

1. **Ex-1** — inject spatial info as raw input signal (failed: model couldn't integrate it with visual features).
2. **Ex-2** — inject spatial/global info as a *feature-space* modulation instead of raw input (safer; small gains).
3. **Ex-3** — inject spatial info as a fixed, hand-designed *output-space* prior with no learning (safe but ineffective; too generic).
4. **Ex-4** — inject spatial info as a *learned, trained* output-space consistency check (best measured outcome).

The general lesson across all four experiments: **how** spatial or contextual information is fused into the network matters far more than **whether** it is included at all. Naively concatenating or hand-crafting spatial signal (Ex-1, Ex-3) underperforms structured, learned modulation or gating mechanisms (Ex-2 V1, Ex-4) that are designed to leave the pretrained backbone's feature space intact.

---

## 10. Overall Findings

- Raw coordinate-channel concatenation is the wrong way to inject spatial context into MedicalPatchNet — it actively hurts both classification and localization.
- FiLM modulation with a frozen backbone and reused classifier (Ex-2 V1) is the safest and most reliable architectural change tested, and was adopted as the foundation for further spatial-conditioning work.
- Retraining the classifier head alongside FiLM increases variance across diseases rather than delivering a clean net improvement.
- A fixed, hand-designed center-weighting prior (Ex-3) is easy to apply (no retraining) but too generic — different diseases occupy very different anatomical regions.
- A learned, trained neighborhood-consensus gate (Ex-4) is the strongest intervention tested, and its **residual gating formulation** (`logits * (1 + gate)`) clearly outperforms the plain multiplicative form (`logits * gate`).
- Deeper mini-classifier heads, tested independently in both Ex-2 and Ex-4, consistently underperformed a single linear classifier layer in this codebase — added classifier capacity was not the bottleneck in any experiment.

---

## 11. Setup and Installation

```bash
git clone https://github.com/ArvendraChhonkar/Modified_MedicalPatchNet.git
cd Modified_MedicalPatchNet/Original_Medical_PatchNet

conda env create -f environment.yml
conda activate <env-name-from-yml>
```

Datasets required:
- **CheXpert** — download from the official Stanford ML Group release.
- **CheXlocalize** — download separately for localization/IoU evaluation; used alongside CheXpert in `preprocessing/`.

---

## 12. Usage

### 12.1 Baseline Training/Evaluation

```bash
cd Original_Medical_PatchNet
bash runTraining.sh
bash runEval.sh
```

### 12.2 Running a Specific Experiment

Each experiment folder is self-contained with its own `NetworkModel.py` (or equivalently named file) and, where applicable, its own training/eval scripts:

```python
# Example: Experiment 2, static FiLM (V1)
from NetworkModel import FiLMPatchNet

model = FiLMPatchNet.from_scalepatchnet_checkpoint(
    checkpoint_path="Original_Medical_PatchNet/savedModels/<checkpoint>.pt",
    patchSize=64,
    outFeatures=14,
)
logits = model(x)   # x: (B, 1, 512, 512) grayscale batch
```

```python
# Example: Experiment 4, best-performing consensus-gated model
from importlib import import_module
mod = import_module(
    "Ex-4_filers(1x1_3x3_5x5)."
    "implemented 3x3 filter twice - really high perecision f1 specificity and senesitivity.NetworkModel"
)
model = mod.getModelClass("ScalePatchNet_1x1")(patchSize=64, outFeatures=14)
```

> Note: folder names containing spaces (as in Experiment 4's best variant) cannot be imported with a plain `import` statement — use `importlib.import_module` with the literal path, or rename the folder to a valid identifier.

---

## 13. Future Work

- Combine Ex-2's Phase-1 V1 FiLM with Ex-4's residual consensus gate — both are lightweight, backbone-preserving modifications, and their effects may be complementary.
- Replace Ex-3's generic center-weighting prior with disease-specific probability maps derived from CheXlocalize segmentation masks (per-pathology, rather than one-size-fits-all).
- Complete the remaining planned Ex-2 stages: residual calibration network, learnable patch weighting network, aggregation strategy comparison, and full ablation study.
- Fix known implementation issues in Experiment 4 (missing `return` in `forwardScaledPatches` for the "filter twice" variant; redundant duplicate gate computation) before further benchmarking.
- Run a controlled kernel-size sweep (1x1 vs. 3x3 vs. 5x5) in Experiment 4 with the gating formula held fixed, to separate the effect of receptive field from the multiplicative-vs-residual gating change.

---

## 14. Citation

If you use or build on this repository, please cite:

Chhonkar, A. (2026). *Research and Development of Explainable Deep Learning Models for Chest X-Ray Classification and Localization using MedicalPatchNet.* IIT (BHU) Internship Report.

---

## 15. License

Released under the MIT License. See [LICENSE](./LICENSE) for details.
