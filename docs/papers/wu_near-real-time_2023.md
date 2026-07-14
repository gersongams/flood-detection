---
title: "A Near-Real-Time Flood Detection Method Based on Deep Learning and SAR Images"
authors: "Xuan Wu, Zhijie Zhang, Shengqing Xiong, Wanchang Zhang, Jiakui Tang, Zhenghao Li, Bangsheng An, Rui Li"
year: 2023
venue: "Remote Sensing (MDPI), 15(8), 2046, 20 pp."
doi: "10.3390/rs15082046"
bibtex_key: wu_near-real-time_2023
zotero_pdf: "/Users/gersongarrido/Zotero/storage/JZWUGYF6/Wu et al. - 2023 - A Near-Real-Time Flood Detection Method Based on Deep Learning and SAR Images.pdf"
pages: 20
sensors: [Sentinel-1, ALOS-PALSAR DEM]
task: "near-real-time flood inundation detection and mapping (binary water/no-water semantic segmentation) + flood training-dataset generation"
model: "UNet (selected); also FCN-8, SegNet, DeepResUNet; global threshold baseline"
tags: [sar, sentinel-1, flood-mapping, unet, semantic-segmentation, near-real-time, weak-labels, yangtze-river-basin, dem, vh-polarization]
---

# A Near-Real-Time Flood Detection Method Based on Deep Learning and SAR Images

> **TL;DR** — The authors propose a semi-automatic pipeline to *generate* flood training datasets from Sentinel-1 SAR (coarse water-index thresholding → regional threshold refinement + DEM shadow masking + manual touch-up), producing a "strong label" dataset of 5296 tiles of 256×256 px from 16 flood events in the Yangtze River Basin (abstract, p. 1). Four CNNs (FCN-8, SegNet, UNet, DeepResUNet) were trained on it and compared against a global-threshold baseline: UNet reached OA 0.986 / F1 0.976 on test set 1 and OA 0.978 / F1 0.947 on test set 2, versus the global threshold's F1 0.877 / 0.860 (Table 3, p. 11). Band ablation shows **VH polarization alone is best**; adding VV and/or DEM does not help (Table 4, p. 12). Detections on 8 further flood events were recycled into a "weak label" dataset of 21,826 tiles (p. 14), which reproduced the same conclusions at slightly lower accuracy (UNet F1 0.920, Table 5, p. 15).

## Problem & Motivation
Flooding causes ~USD 25.5 billion in losses and 6570 fatalities per year worldwide (1970–2020) (p. 1); the 2020 southern China floods affected 30.2 million people (CNY 61.79 billion loss) (p. 2). Optical flood mapping (NDWI-based) is limited by daylight-only operation and clouds; SAR works day/night and through cloud, but the classic global-threshold approach is noise-sensitive, lacks spatial consistency, cannot handle nonlinear cases, and depends on expert knowledge (p. 2).

Deep learning could solve this but is starved of data. The authors list three explicit gaps (p. 2): (1) DL for flood detection lacks big-data support; (2) generating training labels is labor-intensive; (3) most flood methods are built for a *single* flood event and do not transfer. Their contribution is therefore primarily a **dataset-generation methodology** plus an evaluation of off-the-shelf CNNs on it, for large-scale near-real-time detection over the Yangtze River Basin (YRB).

## Method / Architecture
Three-part pipeline (Figure 2, p. 6):

**(a) Preprocessing (SNAP 8.0)** — six steps on Sentinel-1 GRD: orbit correction, thermal noise removal, radiometric calibration, **speckle filtering**, terrain correction, decibelization (dB conversion) (p. 4). DEM: mosaicked, cropped, resampled to the SAR grid.

**(b) Dataset production (semi-automatic labelling)** (pp. 5–6):
1. **Coarse segmentation** with a SAR water index applied with a threshold of **0.3–0.4**:
   `WI = ln(10 × VH × VV) − 8`  (Equation 1, p. 5) — taken from Tian et al. [38].
2. **Fine segmentation**: manual ROI selection covering varied land cover (to balance positive/negative samples); **regional** thresholds instead of one global one — threshold **0.4** for hilly areas with terrain shadow, **0.15–0.2** for farmland/aquaculture (p. 5).
3. **DEM slope mask of 10 degrees** to remove terrain-shadow false positives in steep terrain (p. 5).
4. Residual errors fixed by **manual annotation**; then batch clipping of VH, VV, DEM and label into tiles → **strong label dataset**.

**(c) Flood detection**: CNN trained on the strong labels; its predictions on 8 *other* flood events are then filtered (80% of near-fully-flooded / fully-dry tiles discarded) to build a **weak label dataset** (p. 14). Flooded area = difference between the detected water extent in the flood image and in a non-flood image (change differencing, not end-to-end change detection — the authors explicitly note this in §4.2, p. 15).

**Models compared** (§2.2.2, pp. 7–8): FCN-8, SegNet, UNet, DeepResUNet, plus the global threshold method.
- UNet (Figure 6, p. 10): symmetric encoder/decoder, 4 down-sampling + 4 up-sampling layers, 2–3 stacked conv layers each, channels 64/128/256/512/1024, 3×3 conv + ReLU, 2×2 max-pool, transposed conv, skip concatenations, final 1×1 conv → 2 classes.
- SegNet: same shape but uses max-pool indices instead of concatenation.
- DeepResUNet: UNet + ResNet residual blocks, conv channels reduced to 128 for efficiency.

**Training hyperparameters** (p. 11): Adam optimizer; batch size **10**; **60,000 iterations**; initial LR **0.0001** with exponential decay of **0.8 per 10,000 iterations**. Framework: **TensorFlow**, NVIDIA GeForce RTX 2080Ti (p. 10). Arcpy + Python for segmentation/dataset generation.
- **Loss function: not reported** (dice loss is only *suggested* as future work for change detection, p. 16).
- **Data augmentation: not reported.** Validation split: **not reported** (only train / test1 / test2).

**Metrics** (Table 2, p. 10): OA, Precision, Recall, F1 (standard confusion-matrix definitions). No IoU, no kappa.

## Data
- **Sensor**: Sentinel-1 GRD (VH + VV backscatter intensity). **Auxiliary**: 12.5 m DEM from ALOS-PALSAR, resampled to the Sentinel-1 grid (p. 3). Sentinel-1 pixel spacing itself is **not reported** (nominally 10 m for GRD-IW, but the paper does not state it) ⚠️.
- **Study area**: Yangtze River Basin, China (~1.8 million km² catchment) (p. 3). Sub-areas: Dongting Lake, Poyang Lake, Chaohu Lake, Honghu Lake, Juzhang/Huaihe/Fujiang rivers, middle/lower/upper Yangtze (Table 1, p. 4).
- **Events/images**: **16 flood events**, **32 Sentinel-1 images** (flood + non-flood pairs), 2016–2021 (Table 1, p. 4). 16 images from **8 events** → training/testing; the remaining 16 images from **8 events** → application/near-real-time detection.
- **Split**: of the 8 train/test events, **7 events for training + testing**, **1 event (Ruan Jiang, 2020) used for testing only** (p. 5, Table 1). Two test sets: **test dataset 1 = 13 images** and **test dataset 2 = 14 images**, each 3000–5000 pixels wide (p. 7). Test set 2 comes from different flood events, which is why all models score lower on it (p. 11).
- **Tiles**: training data cropped to **256 × 256** tiles; abstract reports **5296 tiles** in the strong label dataset (p. 1). Weak label dataset: **21,826 tiles** of 256×256 after discarding 80% of trivially all-water/all-dry tiles (p. 14). Test/application used a sliding-window strategy with no cropping (p. 7).
- **Ground truth**: semi-automatic — water index coarse segmentation → regional thresholding → DEM slope mask → manual annotation (see Method). No independent field validation; labels are the authors' own product ⚠️.
- **Spectral indices**: only the SAR water index `WI = ln(10 × VH × VV) − 8` (Eq. 1, p. 5). **No NDVI/NDWI** (optical indices are only mentioned in the related-work discussion).
- **Public dataset**: none released. Sen1Floods11 is cited as related work [33] but not used. Data availability statement only points to the ESA hub for the Sentinel-1 product IDs (p. 17).

## Results

### Model comparison — strong label dataset (Table 3, p. 11)
First row per model = test dataset 1; second row = test dataset 2.

| Model | OA | Precision | Recall | F1 |
|---|---|---|---|---|
| Global Threshold | 0.958 / 0.953 | 0.977 / 0.969 | 0.795 / 0.774 | 0.877 / 0.860 |
| FCN-8 | 0.974 / 0.961 | 0.943 / 0.881 | 0.970 / 0.939 | 0.956 / 0.909 |
| SegNet | 0.983 / 0.975 | **0.991** / **0.981** | 0.953 / 0.897 | 0.971 / 0.937 |
| UNet | **0.986** / 0.978 | 0.980 / 0.951 | **0.973** / **0.942** | **0.976** / 0.947 |
| DeepResUNet | 0.986 / **0.979** | 0.985 / 0.970 | 0.967 / 0.927 | 0.976 / **0.948** |

- All CNNs beat the **global threshold baseline**; the threshold's recall is "about 15% lower than the other models" (p. 11) — it misses flood along river/lake boundaries where speckle dominates.
- FCN-8 is the weakest CNN, precision "about 0.08 lower" than the others (p. 11).
- UNet and DeepResUNet are essentially tied (UNet best on test1 F1 0.976; DeepResUNet marginally best on test2, F1 0.948). ⚠️ The text says "UNet was therefore selected" (p. 11) even though DeepResUNet edges it on test2 OA/F1 — the justification is qualitative (visual maps, Figure 7).
- Qualitative map comparison on a 5812 × 4260 image from test dataset 1 (Figure 7, p. 12).

### Band ablation — UNet (Table 4, p. 12)
First row = test1, second = test2.

| Input bands | OA | Precision | Recall | F1 |
|---|---|---|---|---|
| VH | **0.986** / **0.978** | 0.980 / 0.951 | 0.973 / 0.942 | **0.976** / **0.947** |
| VV | 0.976 / 0.961 | **0.985** / **0.972** | 0.933 / 0.835 | 0.958 / 0.898 |
| VH + DEM | 0.986 / 0.978 | 0.976 / 0.941 | **0.975** / **0.952** | 0.976 / 0.947 |
| VV + DEM | 0.977 / 0.964 | 0.966 / 0.934 | 0.954 / 0.886 | 0.960 / 0.909 |
| VH + VV | 0.983 / 0.971 | 0.980 / 0.966 | 0.961 / 0.889 | 0.971 / 0.926 |
| VH + VV + DEM | 0.981 / 0.968 | 0.988 / 0.979 | 0.948 / 0.865 | 0.968 / 0.918 |

- **VH alone is the best input.** VV gives slightly higher precision but much lower recall. Adding DEM gives no improvement in the YRB (flat middle/lower reaches); adding all three *reduces* accuracy — "the signal-to-noise ratio of VV and DEM bands was low" (p. 12).
- ⚠️ VH and VH+DEM are numerically identical in F1 (0.976 / 0.947); the bolding in Table 4 splits arbitrarily between them.

### Weak label dataset (Table 5, p. 15) — two test sets combined
| Model | OA | Precision | Recall | F1 |
|---|---|---|---|---|
| FCN-8 | 0.948 | 0.897 | 0.909 | 0.903 |
| SegNet | 0.955 | 0.912 | **0.917** | 0.914 |
| UNet | **0.958** | **0.930** | 0.911 | **0.920** |
| DeepResUNet | 0.958 | 0.927 | 0.912 | 0.919 |

UNet band ablation on weak labels (same table): VH 0.920 F1 (best); VV 0.910; VH+DEM 0.919; VV+DEM 0.910; VH+VV 0.919; VH+VV+DEM 0.915. Same ordering as with strong labels, at ~0.03–0.05 lower F1 — the drop is attributed to the weak labels not being manually curated (p. 15).

**No comparison against any published external method or benchmark** (e.g. Sen1Floods11) was run — the only non-CNN baseline is their own global-threshold implementation.

## Limitations
Author-stated (§4.3, p. 16):
- Sentinel-1's revisit cycle constrains capture of flood peaks; more SAR satellites are needed for generalization.
- The change-detection step is not end-to-end; challenges listed (p. 16): selecting high-accuracy areas to avoid noise, eliminating unchanged labels, balancing positive/negative samples or using a special loss such as **dice loss**.
- DEM has limited effect in the flat middle/lower YRB, but would matter for **flash floods in mountainous terrain**.
- Weak-label generation procedure is crude and should be improved.

Observed by me:
- **Single basin (YRB), single sensor.** No cross-region transfer test; test dataset 2 (different events, same basin) already costs ~0.03 F1, hinting at limited generalization.
- **Ground truth is model-adjacent**: labels come from a threshold + manual pipeline, so the reported OA/F1 measure agreement with a semi-automatic product, not with independent survey data. Accuracies near 0.98 should be read in that light.
- **No IoU** reported — the field standard for flood segmentation — making comparison with Sen1Floods11-style papers awkward.
- **Loss function, augmentation, validation split, class balance, training-set tile count per event: not reported.** ⚠️ The 5296-tile figure appears only in the abstract and is never restated in the body.
- No uncertainty estimates, no repeated runs / error bars.

## Relevance to this thesis
**Sensor mismatch is the headline caveat: this is Sentinel-1 SAR, the thesis is Sentinel-2 optical.** None of its band findings (VH > VV, DEM useless) transfer. It is still useful, on four specific counts:

- **Borrow the label-generation pipeline, translated to optical.** Their core contribution is the recipe: *coarse index threshold → regional (land-cover-specific) thresholds → DEM/slope mask → manual touch-up → tile*. The thesis can run exactly this with **NDWI (McFeeters)** in place of their SAR `WI`, adopting their key insight that **one global threshold is wrong**: they use 0.4 in hilly/shadowed terrain and 0.15–0.2 over farmland/aquaculture (p. 5). Peru's Andes-to-Amazon gradient makes region-adaptive NDWI thresholds directly applicable, and the **10-degree slope mask from a DEM** is a cheap, reusable false-positive filter (optical has hill shadow too).
- **Borrow the weak-label bootstrap.** Train on a small curated set, predict on unlabeled events, discard the ~80% trivially all-water/all-dry tiles, keep the mixed ones as a weak-label training set (p. 14). Their result — weak labels give ~0.03–0.05 lower F1 but the *same model ranking* — is a concrete, citable justification for cheaply scaling a Peruvian NDVI/NDWI training set.
- **Borrow the architecture + hyperparameters as a starting point.** UNet, 256×256 tiles, Adam, LR 1e-4 with 0.8 decay every 10k iterations, batch 10 (p. 11). Directly portable to the PyTorch CNN. Their model comparison also says something useful: **UNet ≈ DeepResUNet > SegNet > FCN-8**, so a plain UNet is not leaving much on the table — the thesis need not chase exotic architectures.
- **Use as a cited baseline/upper bound, with care.** F1 0.976 (test1) / 0.947 (test2) for UNet on SAR (Table 3, p. 11), vs global threshold F1 0.877/0.860 — a clean "CNN beats thresholding" number to quote in the introduction. Their threshold-vs-CNN gap is exactly the argument the thesis needs against a pure NDWI-threshold baseline. But they report **no IoU** and their labels are semi-automatic, so do not present 0.976 as a target the thesis must hit.

**Differences to state explicitly:** SAR vs optical (they mention optical is *easier* for water discrimination but cloud-limited, p. 16 — this is the thesis' trade-off, in reverse); **inundation extent** vs the thesis' flood-*prone* / susceptibility framing; change detection between one flood and one non-flood image vs the thesis' multi-year NDVI/NDWI **time series** (they explicitly point to **LSTM/ConvLSTM over time series** as the future direction, §4.2 p. 15 — that is the thesis' territory and a good citation to anchor it). Also a genuine **cautionary example**: they note DEM only helps in mountainous flash-flood settings — relevant since much of Peru is exactly that, so the thesis should *not* copy their "DEM is useless" conclusion.

Classification: **method to cite + dataset-generation recipe to borrow + baseline framing**. Not a dataset source (nothing released).

## Keywords / Tech
- **Models**: UNet, DeepResUNet (UNet + ResNet blocks), SegNet, FCN-8; global threshold baseline
- **Sensors**: Sentinel-1 GRD (VH, VV); ALOS-PALSAR 12.5 m DEM
- **Index**: SAR water index `WI = ln(10 × VH × VV) − 8` (threshold 0.3–0.4)
- **Frameworks/tools**: TensorFlow, SNAP 8.0, Arcpy, Python; RTX 2080Ti
- **Techniques**: speckle filtering, terrain correction, decibelization, regional thresholding, DEM slope masking (10°), strong/weak label datasets, sliding-window inference, change differencing (flood vs non-flood water extent)
- **Metrics**: OA, Precision, Recall, F1 (no IoU, no kappa)
- **Related datasets cited (not used)**: Sen1Floods11 [33]

## Notable quotes
- "the results suggested that the efficiencies and accuracies of convolutional neural network models were obviously higher than that of the threshold method. The effects of VH polarization, VV polarization, and the involvement of auxiliary DEM on flood detection were investigated, which indicated that VH polarization was more conducive to flood detection, while the involvement of DEM has a limited effect on flood detection in the Yangtze River Basin." (p. 1, Abstract)
- "we found that for hilly areas with many terrain shadows, a single segmentation threshold of 0.4 could correct most of the misclassifications in rough segmentations, while for areas such as farmland and aquaculture, a threshold of 0.15–0.2 was more appropriate. For mountainous areas with steep terrain, the mask with a slope of 10 degrees was used to further correct the effects of terrain shadows." (p. 5)
- "The experiments indicated that the signal-to-noise ratio of VV and DEM bands was low, and that the VH polarization has the best effect on flood detection in the YRB." (p. 12)
- "The floods in the YRB are concentrated in the middle and lower reaches of the basin where the terrain is flat and DEM has limited effects. However, for flash floods, DEM are important for identifying mountainous shadows that may pose serious effects on flood detection and mapping." (p. 16)
- "The spectral information of optical data is easier to identify water bodies from than SAR data. In the future, we can find optical images with no or little cloud coverage during flooding events to expand the available datasets." (p. 16)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/JZWUGYF6/Wu et al. - 2023 - A Near-Real-Time Flood Detection Method Based on Deep Learning and SAR Images.pdf`
Cite as: `\cite{wu_near-real-time_2023}`
