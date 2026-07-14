---
title: "WaterDetectionNet: A New Deep Learning Method for Flood Mapping With SAR Image Convolutional Neural Network"
authors: "Binbin Huang, Peng Li, Hongyuan Lu, Jiamin Yin, Zhenhong Li, Houjie Wang"
year: 2024
venue: "IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), vol. 17, 2024, pp. 14471-14485"
doi: "10.1109/JSTARS.2024.3440995"
bibtex_key: huang_waterdetectionnet_2024
zotero_pdf: "/Users/gersongarrido/Zotero/storage/VUX9HVMK/Huang et al. - 2024 - WaterDetectionNet A New Deep Learning Method for Flood Mapping With SAR Image Convolutional Neural.pdf"
pages: 15
sensors: [Sentinel-1]
task: "flood inundation mapping / water body extraction (binary semantic segmentation) from SAR"
model: "WDNet (WaterDetectionNet): Xception + ASPP encoder (DeepLabv3+ style) with channel & position self-attention in the decoder"
tags: [sar, sentinel-1, flood-mapping, water-extraction, semantic-segmentation, self-attention, deeplabv3plus, aspp, poyang-lake, s1water]
---

# WaterDetectionNet: A New Deep Learning Method for Flood Mapping With SAR Image Convolutional Neural Network

> **TL;DR** — The authors build WDNet, a DeepLabv3+/U-Net-derived encoder-decoder (Xception backbone + ASPP with dilation rates 1/2/4/8) that adds channel-attention (CAM) and position-attention (PAM) self-attention modules in the decoder, for binary water/non-water segmentation of Sentinel-1 SAR. They also release **S1Water**, a semi-automatically labelled dataset of **4000 SAR images of 512×512** (Sentinel-1, 2015–2023, five continents) labelled via the SDWI index plus manual correction (p. 7). On four Poyang Lake (China, 2020 flood) test regions WDNet reaches **accuracy 0.987, precision 0.994, recall 0.980, IoU 0.974, F1 0.987** (Table I, p. 10), beating U-Net, DeepLabv3+, FCN, RF, ML and K-means on precision/IoU/F1. An ablation shows removing self-attention drops IoU from 0.974 to 0.888 (Table II, p. 10).

## Problem & Motivation
Optical remote sensing is blocked by the clouds and rain that accompany the very typhoons/rainstorms that cause floods, so SAR — all-weather, day/night — is the preferred sensor for rapid flood mapping (pp. 1-2). But SAR water extraction by thresholding is subjective and fails on undulating terrain (shadows misread as water), and object-oriented methods are corrupted by speckle noise (p. 2).

Deep semantic segmentation models (U-Net, U-Net++, DeepLabv3, DeepLabv3+, FCN) exist but "are mostly used in optical remote sensing images, while there are fewer applications based on radar remote sensing images" (p. 2). Two gaps are targeted: (a) no large, diverse, public **SAR** water-body dataset for flood mapping, (b) existing decoders lose fine water information (small rivers, branches) during simple upsampling. WDNet answers (b) with attention; S1Water answers (a).

## Method / Architecture

**Overall shape** (Fig. 4, p. 5): a DeepLabv3+-style encoder-decoder, "derived from the U-Net and DeepLabv3+ framework" (p. 2), with extra conv layers in the encoder for deeper-scale small-river features and self-attention in the decoder.

**Encoder**
- Backbone: **Xception** (taken from DeepLabv3+) (p. 3).
- **ASPP** module on the backbone output: **five parallel branches** — `Conv 1×1, rate 1`; `Conv 3×3, rate 2`; `Conv 3×3, rate 4`; `Conv 3×3, rate 8`; and **image pooling** (global/spatial pyramid pooling) (Fig. 4, p. 5). ⚠️ The text on p. 4 gives the expansion rates twice and inconsistently: once as "1, 2, 4, and 8 times" and once as "four parallel zero convolutional layers with sampling rates of 1, 2, 4, and 8" where the first uses 1×1 and the other three 3×3 — matching the figure — so read the figure as authoritative.
- The parallel outputs are concatenated, then a **1×1 conv** reduces channels, then **upsample** (p. 4).

**Decoder**
- Low-level feature path: backbone output → **CAM (channel attention)** → **Conv 1×1** (Fig. 4).
- The upsampled ASPP output and the attended low-level features are **concatenated** → **CAM** → **Conv 3×3** → **PAM (position/spatial attention)** → **Upsample** → output mask (Fig. 4).
- Rationale: "the channel attention can automatically determine which channels have weights"; the spatial attention "can better focus on the information of the water region" and is used when upsampling the encoder features (p. 6).

**Self-attention module** (Fig. 6, p. 6) — the reimplementable core, a standard non-local/DANet block:
- Local feature `A ∈ R^{C×H×W}` from the backbone; three convolutions give `B, C, D ∈ R^{C×H×W}`.
- `B` reshaped to `B' ∈ R^{C×N}` with `N = H×W`, transposed to `Bᵀ`; `C` reshaped to `C'`; matrix-multiply and Softmax gives attention matrix `S ∈ R^{N×N}`:
  - `S_ij = exp(B_i · C_j) / Σ_{i=1..N} exp(B_i · C_j)`  (Eq. 1, p. 6)
- `D` reshaped, multiplied by `S`, scaled by learnable `α`, and added element-wise to `A`:
  - `E_j = α · Σ_{i=1..N} (S_ij · D_i) + A_{i,J}`  (Eq. 2, p. 6)
- α is "the scaling factor" (p. 6). ⚠️ Eq. (2)'s index `A_{i,J}` is inconsistent notation (should be `A_j`).

**Training hyperparameters** (p. 7)
- Framework: Python + **PyTorch**.
- Loss: **binary cross-entropy** (binary water/land classification).
- Optimizer: **Adam**. Learning rate **0.0001**. **100 epochs** ("trained it 100 times"); the loss "converged at this learning rate with an optimal number of iterations of 50" (p. 7).
- Training-set accuracy 97.6%, validation-set accuracy 99.4% (p. 7).
- **Batch size: not reported. Data augmentation: not reported. Train/val/test split ratio: not reported** (only "we divided the produced dataset into training set, validation set and test set", p. 3).

**Preprocessing pipeline** (Fig. 1, p. 3; p. 3 §D)
1. Sentinel-1 GRD preprocessed in **SNAP**: orbit file correction, thermal noise removal, radiometric calibration, **speckle filtering**, Doppler/terrain correction, clipping, mosaicing.
2. **PCA** on the VH band (first principal component; PCA "can suppress most of the noisy information").
3. **Decibelization** of the VH band (linear → dB log scale) to enhance contrast.
4. **Layer stacking** of three bands → **VH (raw), VH-PCA, VH-DB** → a "three-bands dataset". Each image's metadata is normalized (p. 2).
   - ⚠️ Inconsistency: Fig. 1 and p. 7 say the three bands are VH, VH-PCA, VH-DB, but p. 4 describes the ASPP input as "PCA-processed VH-band data, VH raw data, and decimated **VV**-band data". Also p. 2 says they "used the Sentinel-1 dual-polarization data", while p. 3 says the imaging polarization mode is VV. Treat the band triplet as **VH / VH-PCA / VH-dB**.

## Data

**Training dataset — S1Water** (p. 7)
- **Sentinel-1 SAR, 4000 images of 512×512**, each with a water-area label.
- Global coverage: Asia, Europe, North America, South America, Africa; period **2015–2023**.
- Deliberately heterogeneous semantics: lakes, rivers, wetlands, intertidal zones, coastal zones, ponds, reservoirs; plus confusable non-water: mountainous terrain, urban buildings, vegetated land, bare land.
- **Ground truth = semi-automatic**: the **Sentinel-1 Dual-Polarized Water Index (SDWI)** is thresholded at **0**, positive = water:
  - `SDWI = ln(10 × VV × VH) − 8`  (Eq. 8, p. 7)
  - Regions prone to misclassification (mountain shadow, vegetated land) were then **manually annotated in Labelme**, and permanent vs non-permanent water was disambiguated **using optical imagery** (p. 7).
- Availability: "The S1Water dataset and WDNet code are available from the corresponding author upon reasonable request" (p. 13) — **not a public download**.

**Case-study / test data — Poyang Lake, Jiangxi, China** (pp. 3, 7)
- **6 Sentinel-1 images, June 20 – Aug 19, 2020** (20 Jun, 02 Jul, 14 Jul, 26 Jul, 07 Aug, 19 Aug). Level-1 **GRD**, **IW** mode.
- Lake at 28°11′N–29°51′N, 115°49′E–116°46′E; area 3150 km² at flat-water level, >4125 km² at high water.
- Comparison performed on **four test regions**: predisaster, postdisaster, small water body, mountain area (p. 7) — chosen to span different semantic backgrounds.
- Spatial resolution of the SAR imagery: **not explicitly reported** (Sentinel-1 IW GRD is nominally 10 m, but the paper does not state it).
- No spectral indices in the NDVI/NDWI sense are used — SDWI (above) is the only index, and it is used for labelling, not as a model input.

## Results

**Table I — Accuracy comparison for different methods** (p. 10; four Poyang Lake regions, same training samples for all methods):

| Family | Model | Accuracy | Precision | Recall | IoU | F1 score |
|---|---|---|---|---|---|---|
| Machine learning | RF | **0.984** | 0.951 | **0.984** | **0.946** | **0.972** |
| Machine learning | ML (maximum likelihood) | 0.975 | **0.953** | 0.958 | 0.914 | 0.955 |
| Machine learning | K-means | 0.962 | 0.932 | 0.933 | 0.872 | 0.933 |
| Deep learning | U-Net | **0.989** | 0.974 | **0.988** | 0.963 | 0.981 |
| Deep learning | DeepLabv3+ | 0.972 | 0.975 | 0.970 | 0.947 | 0.973 |
| Deep learning | FCN | 0.986 | 0.974 | 0.975 | 0.950 | 0.974 |
| Deep learning | **WDNet (this study)** | 0.987 | **0.994** | 0.980 | **0.974** | **0.987** |

(Bold = paper's own bolding, which marks the best per column *within* each family.)

Margins vs the best baseline:
- **IoU**: WDNet 0.974 vs U-Net 0.963 → **+0.011**; vs FCN 0.950 → +0.024; vs DeepLabv3+ 0.947 → +0.027; vs RF 0.946 → +0.028.
- **F1**: WDNet 0.987 vs U-Net 0.981 → **+0.006**; vs FCN 0.974 → +0.013; vs RF 0.972 → +0.015.
- **Precision**: WDNet 0.994 vs DeepLabv3+ 0.975 → **+0.019**.
- WDNet does **not** win on accuracy (U-Net 0.989 > WDNet 0.987) or recall (U-Net 0.988 > WDNet 0.980); the paper concedes these "are basically consistent with the U-net" (p. 7).

**Table II — Ablation of the self-attention module** (p. 10; WDNet* = no self-attention):

| Model | Accuracy | Precision | Recall | IoU | F1 score |
|---|---|---|---|---|---|
| WDNet* (no self-attention) | 0.966 | 0.936 | 0.946 | 0.888 | 0.941 |
| WDNet | 0.987 | 0.997 | 0.980 | 0.974 | 0.987 |

"IoU shows the most significant improvement from 0.888 to 0.974" (p. 10) → **+0.086 IoU**, +0.046 F1.

**Flood monitoring result** (Fig. 12, p. 11): Poyang Lake water area 3738 km² (20 Jun) → 4454 (2 Jul) → **6173 km² (14 Jul, peak)** → 5947 (26 Jul) → 5545 (7 Aug) → 5300 km² (19 Aug). Increase of 1487 km² over 20 Jun–14 Jul (avg **+61.1 km²/day**); decrease of 873 km² over 14 Jul–19 Aug (avg **−24.25 km²/day**) (p. 12). Rapid flood mapping done by RGB composite: red = pre-flood decibelized VV, blue & green = post-flood decibelized VV (p. 12).

⚠️ **Numeric inconsistencies to flag if cited**
1. The **abstract** states "the accuracy, recall, intersection over union, and F1 score of the WDNet model are 0.986, 0.994, 0.974, and 0.987" (p. 1) — the value 0.994 is **precision** in Table I (recall is 0.980), and accuracy is 0.987 not 0.986. The abstract mislabels/miscopies. **Use Table I (p. 10).**
2. WDNet's **precision is 0.994 in Table I but 0.997 in Table II** (both p. 10) for the same model.
3. Table II's column header is typo'd "LoU" (IoU).

## Limitations
Stated by the authors (pp. 12-13):
- The S1Water labels, being semi-automatic (SDWI + partial manual fix), "still contain a small amount of incorrect information", which degrades generalization.
- Application scope is narrow: only semantic segmentation of SAR → water masks → change detection; a single model "may not be able to capture all situations".
- Sentinel-1A/B revisit of **6 days is not enough for real-time (hourly/daily) flood monitoring**.
- DeepLabv3+ (and by inheritance WDNet's backbone) misclassifies **mountain shadow** as water; WDNet "performance was superior in images with less shaded and vegetation-covered areas" (p. 9) — i.e. terrain shadow is still a failure mode.

Observed by me:
- **Single study area** for evaluation (Poyang Lake, one flood event) despite a global training set — no cross-region held-out test.
- The four "test regions" are crops of the same lake; the effective test set is very small and its ground truth was produced by the same semi-automatic SDWI+manual pipeline used for training, so labels and predictions share a bias.
- No uncertainty quantification, no confidence intervals, no repeated runs / seeds; differences of 0.006 F1 over U-Net are within plausible run-to-run noise.
- Batch size, split ratios, and augmentation are unreported → the baseline comparison is not exactly reproducible.
- Baselines are all trained on "the same training samples" but no statement that baselines were hyperparameter-tuned.
- Dataset and code are **on request only**.

## Relevance to this thesis

**Directly borrowable**
- **The architecture is the main takeaway and is reimplementable in PyTorch**: Xception/ResNet backbone → ASPP (1×1 rate1 + 3×3 rates 2/4/8 + image pooling) → decoder with a channel-attention block on the skip connection, a channel-attention block after concat, a 3×3 conv, then a position-attention block before the final upsample. Swapping the 3-channel SAR input (VH/VH-PCA/VH-dB) for a **3-channel optical/index stack (e.g. B08, NDVI, NDWI)** is a one-line change to the input conv — this is a plausible, well-motivated backbone for the thesis' PyTorch CNN.
- **The self-attention ablation (IoU 0.888 → 0.974, p. 10)** is a strong argument for including CAM/PAM in the thesis model, and gives a template for the ablation table the thesis should itself report.
- **Loss + optimizer defaults** for a binary water mask: BCE, Adam, lr 1e-4, ~50-100 epochs. Cheap and defensible starting point.
- **Metric set**: OA, Precision, Recall, IoU, F1 with the exact formulas (Eqs. 3-7, p. 6) — adopt these verbatim so the thesis is comparable to WDNet and to its baselines. IoU is the discriminating metric here (accuracy saturates >0.96 for every method, including K-means).
- **Semi-automatic labelling strategy** — index-threshold to bootstrap masks, then manual correction (Labelme) only on confusable regions. The thesis can do exactly this with **NDWI ≥ 0 → water** instead of SDWI, using optical imagery to separate permanent from seasonal water. This is the cheapest path to a labelled Peruvian training set.
- **Change-detection framing**: pre-flood mask vs post-flood mask → flooded area; plus the RGB pre/post composite (p. 12) as a visual product. Directly transferable to a Streamlit output.

**Baselines to beat / report**
- Their table is a ready-made baseline list for the thesis: **U-Net, DeepLabv3+, FCN, Random Forest, Maximum Likelihood, K-means**. Note the honest signal: **RF at F1 0.972 is within 0.015 of the best CNN** — the thesis must include a non-DL baseline or its DL results are unfalsifiable.
- WDNet's numbers themselves are **not directly comparable** (SAR, different lake, different labels) — cite as a method, not as a numeric benchmark.

**Where it differs (important caveats)**
- **SAR (Sentinel-1), not Sentinel-2 optical.** The whole preprocessing chain (speckle filter, decibelization, PCA on VH) is irrelevant to the thesis. The thesis' analogue is cloud masking + L2A + NDVI/NDWI stacking.
- **Single-date inundation mapping, not time-series prediction.** WDNet has **no temporal model** — flood evolution is obtained by running the CNN on 6 dates independently and differencing areas. The thesis' "NDVI/NDWI time series" ambition (ConvLSTM-style, or predicting *flood-prone* areas) goes beyond this paper; WDNet is a per-date segmentation component, not a susceptibility/prediction model.
- Their own limitation about **6-day revisit** applies identically (worse for S2 under clouds) — good material for the thesis' limitations chapter.
- **Cautionary example**: mountain shadow → false water. Peru's Andean terrain makes this the same trap; for optical the analogue is topographic/cloud shadow producing false-positive NDWI. Worth citing when justifying a DEM/slope mask.

**Verdict**: cite as (a) a **method/architecture source** (ASPP + dual self-attention decoder), (b) a **baseline list + metric protocol**, (c) a **labelling-strategy source** (index-threshold + manual correction). Not a dataset source (S1Water is on-request and SAR), not a numeric benchmark.

## Keywords / Tech
- **Models**: WDNet (WaterDetectionNet), DeepLabv3+, U-Net, FCN, Xception backbone, ASPP, DANet-style self-attention (CAM channel attention + PAM position attention); baselines Random Forest, Maximum Likelihood, K-means.
- **Sensors**: Sentinel-1 (GRD, IW, dual-pol VV/VH). Optical imagery used only as a labelling aid.
- **Indices**: SDWI = ln(10 × VV × VH) − 8, thresholded at 0.
- **Frameworks/tools**: Python, **PyTorch**, ESA **SNAP** (Sentinel Application Platform), **Labelme**.
- **Datasets**: **S1Water** (4000 × 512×512 Sentinel-1 tiles, 2015-2023, 5 continents; on request).
- **Techniques**: PCA on VH, decibelization, layer stacking, speckle filtering, atrous/dilated convolution, spatial pyramid pooling, binary cross-entropy, Adam, pre/post RGB flood composite.

## Notable quotes
- "In the encoding, we increased the number of convolutional layers for extracting semantic features at deeper scales, enabling the extraction of smaller rivers, and flooded waterways. In the decoding, we introduced a self-attention module to increase spatial attention and channel attention, and adaptively update the network weights" (p. 2).
- "Among the deep learning methods, the WDNet model has the highest precision (0.994), IoU (0.974), and F1-score (0.987). Meanwhile, the accuracy (0.987) and recall (0.980) are basically consistent with the U-net" (p. 7).
- "IoU shows the most significant improvement from 0.888 to 0.974. This indicates that the ability of the WDNet model to extract fine water information has been enhanced" (p. 10).
- "The Sentinel-1A/B revisit time of six days is not enough for real-time flood monitoring at the hourly or daily level, reducing the timeliness of the mapping results" (p. 13).

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/VUX9HVMK/Huang et al. - 2024 - WaterDetectionNet A New Deep Learning Method for Flood Mapping With SAR Image Convolutional Neural.pdf`
Cite as: `\cite{huang_waterdetectionnet_2024}`
