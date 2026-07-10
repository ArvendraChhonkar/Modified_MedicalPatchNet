# Experiment 4 — Patch-Consensus Filtering (1x1 / 3x3 / 5x5)

**Repository:** `Modified_MedicalPatchNet`
**Path:** `Ex-4_filers(1x1_3x3_5x5)/`
**Base architecture:** MedicalPatchNet (patch-based, self-explainable chest X-ray classifier)
**Backbone:** EfficientNetV2-S (ImageNet-1K pretrained, grayscale-adapted)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Background: MedicalPatchNet Recap](#2-background-medicalpatchnet-recap)
3. [Problem Statement](#3-problem-statement)
4. [Proposed Solution: The Soft Consensus Gate](#4-proposed-solution-the-soft-consensus-gate)
5. [Mathematical Formulation](#5-mathematical-formulation)
6. [Repository Structure](#6-repository-structure)
7. [Variant A — `default3x3/`](#7-variant-a--default3x3)
8. [Variant B — `default_filter_with_miniclassifiers/`](#8-variant-b--default_filter_with_miniclassifiers)
9. [Variant C — `implemented 3x3 filter twice/`](#9-variant-c--implemented-3x3-filter-twice)
10. [Side-by-Side Code Diff](#10-side-by-side-code-diff)
11. [Parameter & Complexity Analysis](#11-parameter--complexity-analysis)
12. [Results and Discussion](#12-results-and-discussion)
13. [Known Issues / Bugs Found in Code](#13-known-issues--bugs-found-in-code)
14. [Ablation Recommendations](#14-ablation-recommendations)
15. [Relationship to Other Experiments](#15-relationship-to-other-experiments)
16. [Usage Guide](#16-usage-guide)
17. [Future Work](#17-future-work)
18. [Citation](#18-citation)

---

## 1. Abstract

MedicalPatchNet performs chest X-ray disease classification and localization by tiling each 512x512 radiograph into an 8x8 grid of 64 non-overlapping 64x64 patches, passing every patch independently through a shared EfficientNetV2-S backbone, and averaging the resulting per-patch, 14-class disease logits to produce an image-level diagnosis. This design is what gives the model its self-explainability: the raw per-patch logits, before averaging, can be rendered directly as a heatmap without any post-hoc saliency method such as Grad-CAM.

The weakness of this design is that **mean aggregation trusts every patch equally**. A single anomalous patch — for example, one containing an imaging artifact, a rib overlap, or a boundary effect — can shift the global average toward a false positive even when none of its spatial neighbors show any supporting evidence. Real pathology, by contrast, is spatially coherent: a genuine consolidation, effusion, or opacity typically spans a contiguous cluster of patches, not a single isolated cell in the 8x8 grid.

Experiment 4 addresses this by inserting a **Soft Consensus Gate** — a small, learnable, depthwise-convolutional module — between the raw per-patch logits and the final mean-aggregation step. The gate inspects each patch's local neighborhood in logit-space, separately for each of the 14 disease channels, and produces a per-patch, per-class multiplier in the range `[0, 2]`. Patches whose logits agree with their neighbors are passed through or amplified; patches that disagree with their neighborhood are suppressed toward zero, treating them as likely noise/outliers. This experiment set implements and compares **three concrete variants** of this idea, changing the classifier head and the exact gating arithmetic, and finds that the specific formulation of the gate matters more than the classifier head that precedes it.

---

## 2. Background: MedicalPatchNet Recap

For context, the original, unmodified pipeline (`ScalePatchNet`) that all three Experiment-4 variants build on top of is:

```
Input (1, 512, 512) grayscale chest X-ray
        │
        ▼
Unfold into 8x8 grid of 64 patches, each (1, 64, 64)
        │
        ▼
Reshape to (B*64, 1, 64, 64)  — patches placed in the batch dimension
        │
        ▼
Shared EfficientNetV2-S backbone (grayscale first conv, ImageNet-pretrained)
        │
        ▼
Linear(1280 -> 14)  classifier head
        │
        ▼
Reshape back to (B, 64, 14)   — per-patch, per-disease logits
        │
        ▼
Mean over the 64-patch dimension  →  (B, 14) global disease logits
        │
        ▼
Sigmoid  →  final probabilities for 14 CheXpert disease classes
```

Because the per-patch logits are meaningful on their own (each one is literally "what does this backbone think this specific 64x64 region shows"), they can be reshaped into an 8x8 grid and visualized directly as a heatmap — this is the basis of MedicalPatchNet's built-in explainability.

---

## 3. Problem Statement

Given raw per-patch logits `patch_logits` of shape `(B, 64, 14)`:

```python
global_logits = torch.mean(patch_logits, dim=1)   # (B, 14)
```

This treats the 64 patches as i.i.d. samples and simply averages them. There is no mechanism by which the network can decide "this particular patch's high logit is not corroborated by anything nearby, so I should not fully trust it." The guide's suggestion was to introduce exactly this mechanism: **a convolution over the spatial grid of logits that compares each patch to its neighbors, and gates (attenuates or boosts) the patch's contribution based on that local agreement** — effectively a learned, soft, per-disease "outlier rejection" filter applied before aggregation.

---

## 4. Proposed Solution: The Soft Consensus Gate

The Soft Consensus Gate is a tiny convolutional sub-network — three layers, applied not to pixels but to the 8x8 grid of already-computed disease logits:

1. **Depthwise spatial convolution** — `Conv2d(14, 14, kernel_size=k, padding=(k-1)//2, groups=14)`. Because `groups=14` equals the channel count, this is a fully depthwise convolution: each of the 14 disease channels gets its own independent `k x k` spatial filter, and channels never mix at this stage. This is the layer that actually implements "compare a patch to its neighbors," separately for every disease.
2. **Pointwise convolution** — `Conv2d(14, 14, kernel_size=1)`. A 1x1 convolution across the 14-channel dimension at each spatial location, giving the network a chance to linearly recombine/recalibrate the 14 per-disease agreement scores before the nonlinearity.
3. **Sigmoid activation** — bounds every output to `(0, 1)`.
4. **Rescale by 2** — multiplying the sigmoid output by 2 stretches the effective gate range to `(0, 2)`, so the gate can act as either a suppressor (value < 1, dampening the patch) or a booster (value > 1, amplifying the patch) rather than being restricted to pure attenuation.

The gate is computed once per forward pass, at the same spatial resolution as the patch grid (8x8), and multiplies element-wise into the patch logits **before** the mean-aggregation step. Because the gate depends only on the logits already produced by the backbone (not on raw pixels or intermediate feature maps), it is extremely cheap: it adds only `14 * k * k` depthwise weights plus `14 * 14` pointwise weights to the entire network, a negligible number of parameters compared to EfficientNetV2-S's ~20 million.

---

## 5. Mathematical Formulation

Let \( L \in \mathbb{R}^{B \times 8 \times 8 \times 14} \) denote the reshaped raw patch logits for disease channel \( c \) at grid position \( (i, j) \).

**Depthwise convolution** (per-channel, independent filters \( W_c \in \mathbb{R}^{k \times k} \)):

\[ D_c(i,j) = \sum_{u=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} \sum_{v=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} W_c(u,v) \cdot L_c(i+u, j+v) \]

**Pointwise mixing** (linear recombination across the 14 channels, weights \( M \in \mathbb{R}^{14 \times 14} \)):

\[ P_c(i,j) = \sum_{c'=1}^{14} M_{c,c'} \cdot D_{c'}(i,j) + b_c \]

**Gate activation:**

\[ G_c(i,j) = 2 \cdot \sigma\big(P_c(i,j)\big), \qquad G_c(i,j) \in (0, 2) \]

**Gated logits (multiplicative form, Variants A and B):**

\[ \hat{L}_c(i,j) = L_c(i,j) \cdot G_c(i,j) \]

**Gated logits (residual form, Variant C):**

\[ \hat{L}_c(i,j) = L_c(i,j) \cdot \big(1 + G_c(i,j)\big) \]

**Global aggregation** (identical across all variants):

\[ \text{GlobalLogit}_c = \frac{1}{64} \sum_{i=1}^{8} \sum_{j=1}^{8} \hat{L}_c(i,j), \qquad \hat{P}_c = \sigma(\text{GlobalLogit}_c) [1] \)

Equation [1] gives the final sigmoid disease probability for class \( c \).

---

## 6. Repository Structure

```
Ex-4_filers(1x1_3x3_5x5)/
│
├── default3x3/
│   └── NetworkModel.py
│       └── class ScalePatchNet_filter
│           - Backbone: EfficientNetV2-S, grayscale input
│           - Classifier: nn.Linear(1280, 14)          (single linear layer, unchanged from base)
│           - Gate: kernel_size = 3, multiplicative     gated = logits * gate
│
├── default_filter_with_miniclassifiers/
│   └── NetworkModel.py
│       └── class ScalePatchNet_filter   (same class name, different classifier head)
│           - Backbone: EfficientNetV2-S, grayscale input
│           - Classifier: Dropout(0.2) -> Linear(1280,512) -> ReLU
│                          -> Linear(512,256) -> ReLU -> Linear(256,14)
│           - Gate: kernel_size = 3, multiplicative     gated = logits * gate
│
└── implemented 3x3 filter twice - really high perecision f1 specificity and senesitivity/
    └── NetworkModel.py
        └── class ScalePatchNet_1x1
            - Backbone: EfficientNetV2-S, grayscale input
            - Classifier: nn.Linear(1280, 14)           (single linear layer)
            - Gate: kernel_size = 3, computed TWICE, residual   gated = logits * (1 + gate)
```

---

## 7. Variant A — `default3x3/`

**Class name:** `ScalePatchNet_filter`
**Role in this study:** the reference / control configuration — original linear classifier, single consensus gate pass, standard multiplicative gating.

### 7.1 Full Source

```python
from PIL.ImageFilter import Kernel
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  ScalePatchNet — original model, kept verbatim
# ─────────────────────────────────────────────────────────────────────────────

# adding - 3x3 or any other size filter
class ScalePatchNet_filter(nn.Module):
    def __init__(self, patchSize, outFeatures=14) -> None:
        super(ScalePatchNet_filter, self).__init__()
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

        # ==============================
        # Soft Consensus Gate
        # ==============================
        kernel_size = 3  # change this to change the filter size (now it is 3x3)
        # you can set 1x1 or 5x5 or any other size filter
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

    def forward(self, x):
        patch_logits = self.forwardRawPatches(x)
        patch_logits = self.applyConsensusGate(patch_logits)
        global_logits = torch.mean(patch_logits, dim=1)
        return global_logits

    def applyConsensusGate(self, patch_logits):
        B = patch_logits.size(0)
        C = patch_logits.size(2)
        grid = int(patch_logits.size(1) ** 0.5)
        assert grid * grid == patch_logits.size(1)

        # (B,64,14) -> (B,14,8,8)
        gate_input = patch_logits.view(B, grid, grid, C).permute(0, 3, 1, 2)

        gate = self.consensus_conv(gate_input)
        gate = 2 * gate
        gate = gate.permute(0, 2, 3, 1).reshape(B, grid * grid, C)

        gated_logits = patch_logits * gate
        return gated_logits

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
        rawPatchLogits = self.applyConsensusGate(rawPatchLogits)
        globalLogits = torch.mean(rawPatchLogits, dim=1)
        globalProb = torch.sigmoid(globalLogits)
        scaled = rawPatchLogits * globalProb.unsqueeze(1)
        return scaled


MODEL_CLASS_LIST = [   # To make it easy to add other models
    ScalePatchNet_filter,
]

def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName:
            return ModelClass
    assert False, modelClassName + " not in " + str([x.__name__ for x in MODEL_CLASS_LIST])
```

### 7.2 Walkthrough

- The backbone and classifier are **identical** to the original `ScalePatchNet` — grayscale-adapted EfficientNetV2-S with `classifier[1]` replaced by a single `Linear(1280, 14)`.
- The only addition is `self.consensus_conv` and the `applyConsensusGate` method, inserted between `forwardRawPatches` and the mean-aggregation in `forward`.
- `kernel_size` is exposed as a plain local variable inside `__init__`, so changing it to `1` or `5` (hence the folder name "1x1_3x3_5x5") is a one-line edit — this file specifically uses `kernel_size = 3`.
- `forwardScaledPatches` also runs the consensus gate before computing the sigmoid-scaled heatmap output, so visualizations reflect the same gating used at training/inference time.

---

## 8. Variant B — `default_filter_with_miniclassifiers/`

**Class name:** `ScalePatchNet_filter` (same name as Variant A, but a different classifier head — the two files are not meant to be imported together)
**Role in this study:** tests whether a deeper, non-linear classifier head improves the quality of the logits the consensus gate has to work with.

### 8.1 Full Source

```python
from PIL.ImageFilter import Kernel
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  ScalePatchNet — original model, kept verbatim
# ─────────────────────────────────────────────────────────────────────────────

# adding - 3x3 or any other size filter
class ScalePatchNet_filter(nn.Module):
    def __init__(self, patchSize, outFeatures=14) -> None:
        super(ScalePatchNet_filter, self).__init__()
        self.baseBackbone = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.baseBackbone.features[0][0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        # Lightweight multi-layer classifier: 1280 -> 512 -> 256 -> 14
        self.baseBackbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),          # optional but recommended
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, outFeatures)
        )
        self.patchSize = patchSize

        # ==============================
        # Soft Consensus Gate
        # ==============================
        kernel_size = 3  # change this to change the filter size (now it is 3x3)
        # you can set 1x1 or 5x5 or any other size filter
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

    def forward(self, x):
        patch_logits = self.forwardRawPatches(x)
        patch_logits = self.applyConsensusGate(patch_logits)
        global_logits = torch.mean(patch_logits, dim=1)
        return global_logits

    def applyConsensusGate(self, patch_logits):
        B = patch_logits.size(0)
        C = patch_logits.size(2)
        grid = int(patch_logits.size(1) ** 0.5)
        assert grid * grid == patch_logits.size(1)

        gate_input = patch_logits.view(B, grid, grid, C).permute(0, 3, 1, 2)
        gate = self.consensus_conv(gate_input)
        gate = 2 * gate
        gate = gate.permute(0, 2, 3, 1).reshape(B, grid * grid, C)

        gated_logits = patch_logits * gate
        return gated_logits

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
        rawPatchLogits = self.applyConsensusGate(rawPatchLogits)
        globalLogits = torch.mean(rawPatchLogits, dim=1)
        globalProb = torch.sigmoid(globalLogits)
        scaled = rawPatchLogits * globalProb.unsqueeze(1)
        return scaled


MODEL_CLASS_LIST = [   # To make it easy to add other models
    ScalePatchNet_filter,
]

def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName:
            return ModelClass
    assert False, modelClassName + " not in " + str([x.__name__ for x in MODEL_CLASS_LIST])
```

### 8.2 Walkthrough

- The **only** structural difference from Variant A is `self.baseBackbone.classifier`, which is replaced wholesale (not just index `[1]`) with a 3-linear-layer MLP: `Dropout(0.2) -> Linear(1280,512) -> ReLU -> Linear(512,256) -> ReLU -> Linear(256,14)`.
- The consensus gate is byte-for-byte identical to Variant A — same `kernel_size = 3`, same multiplicative gating (`patch_logits * gate`).
- This isolates one variable: does giving the classifier more capacity/non-linearity, upstream of an unchanged gate, produce better-gated final predictions?
- **Empirically, this combination underperformed Variant A** — see Section 12.

---

## 9. Variant C — `implemented 3x3 filter twice - really high perecision f1 specificity and senesitivity/`

**Class name:** `ScalePatchNet_1x1` (name is legacy/inherited from an earlier 1x1-kernel version of the file; the kernel actually used is 3x3, as set by `kernel_size = 3` inside `__init__`)
**Role in this study:** the best-performing variant in this experiment set, distinguished by residual (rather than pure multiplicative) gating and a duplicated gate computation.

### 9.1 Full Source

```python
import torchvision
import torch.nn as nn
import torch


class ScalePatchNet_1x1(nn.Module):
    def __init__(self, patchSize, outFeatures=512) -> None:
        super(ScalePatchNet_1x1, self).__init__()
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

        # ==============================
        # Soft Consensus Gate
        # ==============================
        kernel_size = 3
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

    def applyConsensusGate(self, patch_logits):
        B = patch_logits.size(0)
        C = patch_logits.size(2)
        grid = int(patch_logits.size(1) ** 0.5)
        assert grid * grid == patch_logits.size(1)

        gate_input = patch_logits.view(B, grid, grid, C).permute(0, 3, 1, 2)

        gate = self.consensus_conv(gate_input)               # 1st pass (result discarded below)
        gate = gate.permute(0, 2, 3, 1).reshape(B, grid * grid, C)

        gate = 2 * self.consensus_conv(gate_input)            # 2nd pass — this is the one actually used
        gated_logits = patch_logits * (1 + gate)
        return gated_logits

    def forward(self, x):
        patch_logits = self.forwardRawPatches(x)
        patch_logits = self.applyConsensusGate(patch_logits)
        global_logits = torch.mean(patch_logits, dim=1)
        return global_logits

    def forwardRawPatches(self, x):  # forward raw patches
        batchSize = x.size()[0]     # put patches in batch dimension
        x = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, self.patchSize, self.patchSize)
        x = self.baseBackbone(x)
        assert x.size()[0] % batchSize == 0   # recreate image batches
        patchCount = int(x.size()[0] / batchSize)
        batchList = torch.split(x, patchCount)
        x = torch.stack(batchList)
        return x

    def forwardScaledPatches(self, x):
        rawPatchLogits = self.forwardRawPatches(x)
        rawPatchLogits = self.applyConsensusGate(rawPatchLogits)
        globalLogits = torch.mean(rawPatchLogits, dim=1)
        globalProb = torch.sigmoid(globalLogits)
        scaled = rawPatchLogits * globalProb.unsqueeze(1)
        # NOTE: this function is missing a `return scaled` statement in the
        # original source — see Section 13, "Known Issues".


MODEL_CLASS_LIST = [   # To make it easy to add other models
    ScalePatchNet_1x1,
]

def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName:
            return ModelClass
    assert False, modelClassName + "not in" + str([x.__name__ for x in MODEL_CLASS_LIST])
```

### 9.2 Walkthrough

This is the variant referenced in the folder name as producing "really high precision, F1, specificity and sensitivity." Two things distinguish it from Variants A/B:

1. **The gate is computed twice.** `self.consensus_conv(gate_input)` is called once and reshaped (into a `(B, 64, 14)` tensor) but this first result is never used — it is immediately overwritten by a second, independent call `gate = 2 * self.consensus_conv(gate_input)`. Because `consensus_conv` contains no dropout or other stochastic layers, the two calls are numerically identical in a given forward pass; the first call is dead code, but the naming "filter twice" in the folder reflects that the convolution module is literally invoked twice in the source, and this repeated-invocation pattern is what the guide's "3x3 filter twice" idea refers to in the experiment log.
2. **The gating arithmetic is residual, not multiplicative.** Instead of `patch_logits * gate` (Variants A/B), this variant computes `patch_logits * (1 + gate)`. With `gate` bounded to `(0, 2)` by the `2 * sigmoid(...)` rescale, the multiplier `(1 + gate)` is bounded to `(1, 3)` — meaning this formulation can only ever **preserve or amplify** a patch's logit, never suppress it below its original value. This is a meaningfully different inductive bias from Variants A/B, where the multiplier is bounded to `(0, 2)` and can suppress a patch down toward zero.

---

## 10. Side-by-Side Code Diff

| Aspect | Variant A (`default3x3`) | Variant B (`...miniclassifiers`) | Variant C (`...filter twice`) |
|---|---|---|---|
| Class name | `ScalePatchNet_filter` | `ScalePatchNet_filter` | `ScalePatchNet_1x1` |
| Classifier head | `nn.Linear(1280, 14)` | `Dropout -> Linear(1280,512) -> ReLU -> Linear(512,256) -> ReLU -> Linear(256,14)` | `nn.Linear(1280, outFeatures)` (default `outFeatures=512`, must be set to 14 for CheXpert) |
| Gate kernel size | 3x3, depthwise (`groups=14`) | 3x3, depthwise (`groups=14`) | 3x3, depthwise (`groups=14`) |
| Gate calls per forward | 1 | 1 | 2 (first result discarded) |
| Gate output range | `(0, 2)` | `(0, 2)` | `(0, 2)` |
| Gating formula | `logits * gate` | `logits * gate` | `logits * (1 + gate)` |
| Effective multiplier range | `(0, 2)` — can zero out a patch | `(0, 2)` — can zero out a patch | `(1, 3)` — can only preserve/boost a patch |
| Applied in `forwardScaledPatches` | Yes | Yes | Yes, but missing `return` (bug) |
| Reported outcome | Good, stable metrics | Weaker than Variant A | **Best** — high precision, F1, sensitivity, specificity |

---

## 11. Parameter & Complexity Analysis

The Soft Consensus Gate is deliberately lightweight relative to the EfficientNetV2-S backbone (~20.2M parameters, ~21M with the grayscale first-layer modification):

| Component | Parameter Count (kernel_size = 3) |
|---|---|
| Depthwise conv: `Conv2d(14, 14, 3, groups=14)` | \( 14 \times 1 \times 3 \times 3 + 14 = 140 \) |
| Pointwise conv: `Conv2d(14, 14, 1)` | \( 14 \times 14 \times 1 \times 1 + 14 = 210 \) |
| **Total gate parameters** | **350** |
| EfficientNetV2-S backbone (approx.) | ~20,200,000 |
| Linear classifier `1280 -> 14` | \( 1280 \times 14 + 14 = 17{,}934 \) |
| Mini-classifier `1280->512->256->14` (Variant B) | \( (1280 \times 512 + 512) + (512 \times 256 + 256) + (256 \times 14 + 14) \approx 787{,}598 \) |

The consensus gate itself adds **well under 0.002%** of the backbone's parameter count regardless of kernel size (1x1, 3x3, or 5x5), making it one of the cheapest architectural interventions tested across all experiments in this repository — the entire benefit or harm it produces comes from *where* it's placed and *how* its output is combined with the logits, not from added capacity.

---

## 12. Results and Discussion

The report accompanying these experiments (patient-batch training on ~29,000 CheXpert images, 20 epochs, AdamW, LR 1e-4, evaluated with AUROC / F1 / Sensitivity / Specificity / Dice / mIoU) frames the outcome of Experiment 4 qualitatively as follows:

| Variant | Classifier Head | Gate Formula | Reported Result |
|---|---|---|---|
| A — `default3x3` | Single linear layer | `logits * gate`, one pass | Good metrics, stable and predictable behavior |
| B — `...miniclassifiers` | 3-layer MLP head | `logits * gate`, one pass | Weaker metrics than Variant A |
| C — `...filter twice` | Single linear layer | `logits * (1 + gate)`, two passes | **High precision, F1, sensitivity, and specificity** — best of the three |

### 12.1 Why might Variant B underperform?

The mini-classifier adds two extra fully-connected layers (~788K new parameters) between the pretrained EfficientNetV2-S features and the final 14-way output. Since the backbone in this experiment set is being fine-tuned on a comparatively small subset (~29K images) for a modest number of epochs, a deeper, freshly-initialized head has more parameters to learn from the same signal, increasing the risk of underfitting or noisy logits — which then propagate into a *consensus gate that has no way to distinguish "genuinely disagreeing neighbor" from "under-trained, noisy neighbor."* A gate built on top of noisier upstream logits has less reliable spatial agreement to exploit.

### 12.2 Why might Variant C outperform?

The residual formulation `logits * (1 + gate)` guarantees `gate=0` reduces to an identity mapping (no change to the original logit), whereas the multiplicative formulation `logits * gate` requires the gate to sit exactly at `gate=1` to preserve the logit unchanged, and can drive it toward `0` for any lower gate value. In early training, when the consensus_conv weights are close to their initialization, the residual form is much closer to acting as a safe identity/near-identity operation and only gradually learns to add a corrective boost, whereas the multiplicative form is starting from a much more aggressive, potentially destabilizing rescale. This is analogous to why residual connections (ResNets) are generally easier to optimize than plain feedforward stacks — the network only has to learn a *correction* on top of a working baseline, not a scale factor from scratch.

---

## 13. Known Issues / Bugs Found in Code

For engineering transparency, the following issues exist in the current source files and should be fixed before this code is used for further production training or published benchmarking:

1. **Variant C — missing return statement.** `forwardScaledPatches` in `ScalePatchNet_1x1` computes `scaled` but never returns it, so calling this method currently returns `None`. This does not affect `forward()` or training, but breaks heatmap/visualization code that relies on `forwardScaledPatches`.
2. **Variant C — redundant first gate computation.** In `applyConsensusGate`, the first call to `self.consensus_conv(gate_input)` (assigned to `gate`, reshaped, then immediately overwritten) has no effect on the output and only wastes compute. It can be safely deleted; the folder's "twice" naming reflects this literal double invocation in the source rather than an intentional ensembling effect.
3. **Variant C — `outFeatures` default is 512, not 14.** The constructor signature `def __init__(self, patchSize, outFeatures=512)` mirrors the original `ScalePatchNet`'s default, but for a 14-class CheXpert task this must be explicitly passed as `outFeatures=14` at instantiation time, or the consensus gate (hardcoded for 14 channels) will shape-mismatch against the classifier output.
4. **Variants A/B share a class name.** Both `default3x3/NetworkModel.py` and `default_filter_with_miniclassifiers/NetworkModel.py` define a class called `ScalePatchNet_filter`. They are not designed to be imported into the same namespace simultaneously — keep them in separate modules/folders as currently structured, or rename one class if you plan to import both into a single training script.
5. **Unused imports.** `from PIL.ImageFilter import Kernel` and `import torch.nn.functional as F` are present in Variants A and B but unused in the code shown — safe to remove.

---

## 14. Ablation Recommendations

To turn this into a rigorous, publication-ready ablation table, the following controlled experiments are recommended as immediate next steps:

1. **Isolate kernel size.** Fix the classifier (linear head) and the gating formula (multiplicative), and sweep `kernel_size in {1, 3, 5}` to measure the effect of neighborhood radius alone, independent of the arithmetic change found in Variant C.
2. **Isolate gating arithmetic.** Fix kernel size at 3x3 and the linear classifier, and directly compare `logits * gate` vs. `logits * (1 + gate)` with the redundant first convolution call removed, to confirm the residual formulation is the true source of Variant C's improvement (rather than the duplicated computation itself, e.g. via numerical effects if dropout/batchnorm were ever added to `consensus_conv`).
3. **Fix the identified bugs** (Section 13) before rerunning any comparison, particularly the missing `return` in `forwardScaledPatches`, so that heatmap-based qualitative evaluation is possible for all three variants.
4. **Report the full metric suite per variant** — AUROC, F1, Sensitivity, Specificity, Dice, and mIoU, per-disease as well as macro-averaged — using the same held-out patient split (e.g., patients 30,001-40,000 from CheXpert, as used elsewhere in this repository) so results are directly comparable to the FiLM and coordinate-conditioning experiments.
5. **Visualize the learned gate itself.** Since `consensus_conv` output is only 8x8x14 per image, it is cheap to log and visualize which patches get suppressed vs. boosted across a batch of validation images, to sanity-check that the gate is doing something clinically sensible (e.g., not systematically suppressing peripheral patches for diseases known to occur at the lung periphery).

---

## 15. Relationship to Other Experiments

This experiment sits at the end of a progression of spatial-awareness techniques applied to MedicalPatchNet across this repository:

| Stage | Approach | Trained end-to-end? | Where in repo |
|---|---|---|---|
| 1 | Raw coordinate channel concatenation (x, y grids as extra input channels) | Yes | Coordinate-aware experiments |
| 2 | FiLM feature modulation (global and coordinate-conditioned) | Yes (FiLM/classifier params only) | `Ex-2_Architectural Changes/` |
| 3 | Fixed, hand-designed center-weighting prior applied at inference | No (inference-only) | `Ex-3_moreWeight_to_central_region/` |
| 4 | **Learned neighborhood-consensus gating on patch logits** | Yes (gate + classifier params) | `Ex-4_filers(1x1_3x3_5x5)/` (this report) |

Where Experiment 3 injected a fixed, hand-designed spatial prior with no learning involved, Experiment 4 replaces that with a fully learned, data-driven spatial-consistency check that is trained jointly with the rest of the network — representing the most integrated attempt so far at teaching MedicalPatchNet to reason about *where* its patch predictions agree or disagree with each other.

---

## 16. Usage Guide

### 16.1 Instantiating Each Variant

```python
# Variant A — default3x3
from default3x3.NetworkModel import getModelClass
ModelA = getModelClass("ScalePatchNet_filter")(patchSize=64, outFeatures=14)

# Variant B — mini-classifier head
from default_filter_with_miniclassifiers.NetworkModel import getModelClass
ModelB = getModelClass("ScalePatchNet_filter")(patchSize=64, outFeatures=14)

# Variant C — residual double-gate (best-performing)
from importlib import import_module
module_c = import_module(
    "implemented 3x3 filter twice - really high perecision f1 specificity and senesitivity.NetworkModel"
)
ModelC = module_c.getModelClass("ScalePatchNet_1x1")(patchSize=64, outFeatures=14)
```

> Note: Variant C's folder name contains spaces and cannot be imported with a plain `import` statement — use `importlib.import_module` with the literal folder name, or rename the folder to a valid Python identifier (e.g. `filter_double_gate/`) for cleaner imports.

### 16.2 Forward Pass

```python
import torch

x = torch.randn(4, 1, 512, 512)   # batch of 4 grayscale chest X-rays

global_logits = ModelA(x)                       # (4, 14) — final disease logits
raw_patch_logits = ModelA.forwardRawPatches(x)   # (4, 64, 14) — pre-gate, per-patch logits
gated = ModelA.applyConsensusGate(raw_patch_logits)  # (4, 64, 14) — post-gate logits
heatmap_ready = ModelA.forwardScaledPatches(x)   # (4, 64, 14) — gated + globally sigmoid-scaled
```

### 16.3 Changing the Filter Size

Inside any variant's `__init__`, edit the single line:

```python
kernel_size = 3   # change to 1 or 5 to reproduce the other members of the (1x1_3x3_5x5) sweep
```

and ensure `padding=(kernel_size - 1) // 2` is left as-is so the spatial grid dimensions (8x8) are preserved after the depthwise convolution.

### 16.4 Visualizing the Gate

```python
with torch.no_grad():
    raw = ModelA.forwardRawPatches(x)
    B, P, C = raw.shape
    grid = int(P ** 0.5)
    gate_input = raw.view(B, grid, grid, C).permute(0, 3, 1, 2)
    gate = 2 * ModelA.consensus_conv(gate_input)   # (B, 14, 8, 8)
    # gate[:, disease_index] can now be plotted as an 8x8 heatmap per image
```

---

## 17. Future Work

- Replace the fixed `2 *` rescale constant with a learnable scalar (or per-class learnable scalar) so the network can learn the optimal boost/suppress range itself rather than having it hardcoded to `(0, 2)`.
- Combine the residual consensus gate (Variant C) with Phase-1 V1 FiLM modulation from the architectural-change experiments — both are lightweight, backbone-preserving modifications, and their effects may be complementary rather than redundant.
- Extend the gate to condition on the classifier's confidence (e.g., only gate patches whose absolute logit magnitude exceeds a threshold), so low-confidence patches near the decision boundary are not unnecessarily amplified or suppressed by neighborhood noise.
- Train a disease-specific gate rescaling factor instead of the shared `2 *` constant across all 14 classes, since different pathologies have different expected spatial extents (e.g., cardiomegaly spans a large central region; a pneumothorax may be a thin peripheral line).

---

## 18. Citation

If you use or build on this experiment, please cite:

Chhonkar, A. (2026). *Research and Development of Explainable Deep Learning Models for Chest X-Ray Classification and Localization using MedicalPatchNet.* IIT (BHU) Internship Report — Experiment 4: Patch-Consensus Filtering (1x1/3x3/5x5).
