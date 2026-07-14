---
title: "Flood Mapping through Sentinel-1, Sentinel-2 Imagery and U-NET Deep Learning Model"
authors: "Fernando Pech-May, Omar Álvarez-Cárdenas, Germán Ríos-Toledo"
year: 2025
venue: "Computación y Sistemas, Vol. 29, No. 1, 2025, pp. 393–407"
doi: "10.13053/CyS-29-1-5539"
bibtex_key: pech-may_flood_2025
zotero_pdf: "/Users/gersongarrido/Zotero/storage/BEJIG3FM/Pech-May et al. - 2025 - Flood Mapping through Sentinel-1, Sentinel-2 Imagery and U-NET Deep Learning Model.pdf"
pages: 15
sensors: [Sentinel-1, Sentinel-2]
task: "flood inundation mapping / water segmentation (flooded areas + permanent water)"
model: "U-Net"
tags: [flood-mapping, u-net, sentinel-1, sentinel-2, sar, ndwi, semantic-segmentation, arcgis-pro, tabasco, google-earth-engine]
---

# Flood Mapping through Sentinel-1, Sentinel-2 Imagery and U-NET Deep Learning Model

> **TL;DR** — The authors build a flood-mapping pipeline for the Ríos region of Tabasco, Mexico, combining Sentinel-1 SAR (GRD, VV+VH, IW) for flooded-area detection and Sentinel-2 L1C optical imagery + NDWI for permanent-water detection, feeding both into a standard U-Net semantic segmentation model. Labels were hand-drawn in ArcGIS Pro ("Training Samples Manager") from pre-/post-flood band comparisons and SAR coherence-histogram thresholding, exported as 256×256 image chips (Flood: 1,315 images / Permanent water: 1,215 images). The headline result is 92.14% accuracy, 86.78% recall and 89.38% F1 for the flood class at 400 epochs (with 90° rotation augmentation applied from epoch 150, expanding 1,215 → 4,860 images), and 91.11% accuracy / 90.32% recall / 90.71% F1 for permanent water at 30 epochs (p. 403). Validated against CENAPRED official flood polygons for the June 2020 Tropical Storm Cristóbal event, the model estimated 15,002.72 Ha vs CENAPRED's 21,196.15 Ha — a 29.22% margin of error (p. 402).

## Problem & Motivation
Floods are the most frequent and damaging natural disaster worldwide; the paper opens with CRED statistics (e.g. "In 2023, 414 natural disasters were recorded, of which 163 were floods, leaving more than 7,500 deaths", p. 393). Mexico — and particularly the state of Tabasco, whose Ríos zone (Tenosique, Balancán, Emiliano Zapata) is crossed by numerous water bodies and dam-regulated rivers — suffers annual inundations; the 2007 flood alone cost ~USD 3,000,000.00 and the 2020 flood affected more than 800,000 people, 200,400 homes, and 2,000 km of land (pp. 393–394).

The methodological gap the paper addresses is the complementarity of sensors: optical imagery (Sentinel-2) offers high spectral richness and is well correlated with open water surfaces via spectral indices, but is blocked by cloud cover — precisely the condition present during flood events. SAR (Sentinel-1) penetrates clouds and works day/night, but is noisy (speckle). Classical ML approaches (SVM, RF, CART) on spectral indices are widespread, but deep learning (CNNs, and specifically U-Net with skip connections) has shown better feature learning for segmentation. The paper therefore proposes an integrated pipeline that uses *each sensor for what it is good at*: SAR for the flood (temporary water) class, optical+NDWI for the permanent-water class, both segmented with the same U-Net architecture (pp. 395–396).

## Method / Architecture
Three-phase methodology (Fig. 2, p. 396): (1) input data / image acquisition → (2) preprocessing + dataset creation → (3) DL model training + validation + flood mapping.

**Architecture (Fig. 11, p. 400)**
- Standard U-Net (Ronneberger et al. 2015) encoder–decoder CNN with skip connections ("copy & concatenate") between corresponding encoder/decoder blocks.
- Input size shown in the architecture figure as **512 × 512**, with feature maps at 256×256, 128×128, 64×64, 32×32 (note: the *training chips* are stated as 256×256 in §3.3 — the paper is internally inconsistent here).
- Blocks: `Conv 3×3 + ReLU`, `Max-pool 2×2` (downsampling), `Up-conv 3×3` (upsampling), copy & concatenate skip connections, final classifier layer.
- Final layer: **soft-max** activation giving per-pixel class probability.
- Loss: **cross-entropy** ("It also employs a cross-entropy model to measure the discrepancy between the predicted output by the network and the expected segmentation mask or label", p. 400).
- No attention modules, no residual blocks, no pretrained encoder — a vanilla U-Net.

**Training hyperparameters**
- Batch size: **8 images per iteration** (p. 400).
- Epochs: swept over **25, 50, 75, 100, 150, 200, 300, 400** for the flood class; **30 epochs** for the permanent-water class (pp. 400–401).
- Validation split: **10%** (p. 401).
- Optimizer: **not reported**. Learning rate: **not reported**. Weight decay / scheduler: **not reported**. Framework (PyTorch/TF): **not reported** (training/labelling was orchestrated through ArcGIS Pro, ref. [1]).
- Augmentation: **90-degree rotations**, applied "from epoch 150 onwards", producing three additional versions per image and enlarging the dataset **from 1,215 to 4,860 images** (p. 401).

**Inference / validation pipeline (§3.5, p. 400)**
1. Load trained model (weights + architecture).
2. Preprocess: normalize pixel values, resize to model input format.
3. Classify the new image → per-pixel flood prediction.
4. Post-process: remove small groups of unwanted pixels, improve consistency of segmented areas.
5. Visualize: overlay predictions on the original image; export flood maps.

## Data
**Sensors and products (Table 1, p. 397)**
| Satellite | Characteristics |
|---|---|
| Sentinel-1 | S1-A; product **GRD** (Ground Range Detected); polarisation **VV + VH**; sensor mode **IW**; **descending** pass direction only (to avoid false positives) |
| Sentinel-2 | S2-A; product **Level 1C**; cloudiness range **< 10%**; **Bands 2, 3, 4 and 8** (Blue, Green, Red, NIR) |

Spatial resolution is **not explicitly reported** anywhere in the paper (native Sentinel-2 10 m for B2/3/4/8 is implied but never stated).

**Study area & period**
- Ríos region, Tabasco, Mexico — municipalities of Tenosique, Balancán and Emiliano Zapata (Fig. 3, p. 396).
- Study period stated as **2020–2023** in §3.1 (p. 396) but as **2019 to 2023** in the abstract (p. 393); flood maps are generated for **2019–2023** (Fig. 14, p. 403). Treat the range as 2019–2023.
- Three seasons defined: northern season (November–February), dry season (March–May), wet season (June–October).
- Sentinel-1 tiles cover Campeche, Chiapas and Tabasco. Fig. 5 (p. 397) lists 14 named S1A_IW_GRDH scenes from 2020, split into pre-flood (grey background: 01/05, 13/05, 01/06, 01/06) and post-flood (01/09, 22/09, 25/09, 04/10, 07/10, 16/10, 09/11, 12/11, 21/11, 24/11).
- Sources: **Copernicus Open Access Hub** (scihub.copernicus.eu) for SAR; **Google Earth Engine** for the optical images, clipped by a shapefile of the study area (p. 397).

**Preprocessing — Sentinel-1 (SNAP tool, §3.2.1, p. 397; Fig. 6, p. 398)**
1. Radiometric correction (digital levels → reflected radiance).
2. **Speckle filtering: 5×5 Lee filter**.
3. Geometric calibration (correct sensor-tilt/relief distortions).
4. Logarithmic scaling transformation → dB.
5. RGB layer generation (mask to detect water/vegetation/flooded pixels).
6. RGB composition.
7. Binary layer: histogram of the image texture coefficient thresholded to separate water vs land.

**Preprocessing — Sentinel-2 (§3.2.2, pp. 397–398)**
- Level 1C products with an automatic **cloud and shadow mask**; Fig. 7 (p. 398) shows optical image before/after cloud masking "with a threshold below 40%".
- Atmospheric correction to L2A is **not** performed (they stay at L1C).

**Spectral index — NDWI**, given as Eq. (1), p. 398:
```
NDWI = (NIR - SWIR) / (NIR + SWIR)
```
⚠️ **Important caveat:** this is the *Gao (1996)* formulation (the paper cites Bo-cai 1996, refs [7]/[8]), which uses SWIR — yet Table 1 says only bands 2, 3, 4 and 8 were downloaded, and Sentinel-2 SWIR (B11/B12) is 20 m. This is an internal inconsistency in the paper. NDVI is mentioned in the related-work section but **is not used** in the method. No NDVI formula is given.

**Label / dataset creation (§3.3, pp. 398–399)**
- Labelling done manually in **ArcGIS Pro** using the "Training Samples Manager" tool, drawing polygons over the imagery (Fig. 10, p. 399).
- Erroneous samples reduced by a **pre-/post-flood band comparison**: all bands from both images stacked into an RGB composite where channel **R = a pre-flood band** and channels **G and B = post-flood bands** (Fig. 8, p. 399).
- SAR **coherence-coefficient histogram thresholding** used to distinguish flooded areas from permanent water bodies (Fig. 9, p. 399).
- Exported as **256×256 pixel image chips**, in two separate datasets (Table 2, p. 399):

| Dataset name | Images | Features |
|---|---|---|
| Flood (SAR) | 1,315 | 8,309 |
| Permanent water (optical + NDWI) | 1,215 | 7,965 |

- Split: only a **10% validation** rate is reported. **No held-out test set is reported** — the reported metrics are validation-set metrics.
- Independent external check: **CENAPRED** (Mexico's National Center for Disaster Prevention) official flood-inundation maps, used for the June 2020 Tropical Storm Cristóbal event (§4.5, p. 402; Fig. 15, p. 404).

## Results
Metrics reported: **Recall, F1-Score, Accuracy** only (p. 400). **IoU, precision, kappa and overall-accuracy/confusion matrices are not reported.**

**Table 3 — evaluation metrics by epoch number (p. 403), verbatim**

*Class: Permanent water (Sentinel-2 + NDWI)*

| Metric | 30 epochs |
|---|---|
| Accuracy | 91.11% |
| Recall | 90.32% |
| F1-Score | 90.71% |

*Class: Floods (Sentinel-1 SAR)*

| Epochs | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 54.76% | 68.28% | 71.63% | 70.74% | 80.49% | 85.52% | **92.12%** | **92.14%** |
| Recall | 35.83% | 56.58% | 72.06% | 49.39% | 77.07% | 76.50% | **88.42%** | 86.78% |
| F1-Score | 43.32% | 61.88% | 71.85% | 58.17% | 78.74% | 80.76% | **90.23%** | 89.38% |

Key readings:
- Accuracy climbs monotonically-ish with epochs (54.76% → 92.14%); recall and F1 peak at **300 epochs** (recall 88.42%, F1 90.23%), so 400 epochs buys +0.02 pp accuracy while *losing* recall and F1 → 300 epochs is effectively the best model.
- The step-change between 100 and 150 epochs coincides with the introduction of the rotation augmentation ("From epoch 150 onwards, a strategy was implemented to increase the available dataset size by implementing the rotation technique", p. 401), so epoch count and dataset size are confounded — the reported curve conflates "more epochs" with "4× more data".
- Training/validation loss (Fig. 12a, p. 401) drops sharply near zero and stays flat for permanent water.

**External validation against government data (§4.5, p. 402)**

| Source | Flooded area (June 2020, TS Cristóbal) |
|---|---|
| CENAPRED (official) | 21,196.15 Ha |
| U-Net model | 15,002.72 Ha |
| Margin of error | **29.22%** (model under-estimates) |

**Territorial extension of detected floods, hectares (Table 4, p. 404)**

| Month | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|
| Sep | 60.93 | 1,223.62 | 120.94 | 2,680.17 | 115.94 |
| Oct | 84.77 | 475.12 | 313.58 | 2,946.25 | 120.47 |
| Nov | 78.41 | 8,175.09 | 66.93 | 371.07 | 2,576.57 |

**Baselines: none.** The paper does *not* run any comparative baseline (no SegNet, no BASNet, no RF/SVM, no thresholding baseline) despite discussing them in Related Works. The only external comparison is the CENAPRED area figure. So "which baselines they beat and by how much" = **not reported / no baselines run**.

## Limitations
Stated by the authors (§5, p. 404):
- "the lack of suitable hardware slows down model training. To improve the model, it requires being trained with a larger dataset and more epochs."
- Optical images are sensitive to cloud cover and require specialized techniques to obtain clear images.

Observed by me (important for how much weight to give this paper):
- **29.22% under-estimation vs the official CENAPRED flood extent** is a large error for an operational flood map; the authors present it without much critical discussion.
- **No test set.** All headline metrics are computed on a 10% validation split of the same manually-labelled chips; there is no spatially or temporally held-out evaluation. Risk of optimistic bias from spatially adjacent, autocorrelated 256×256 chips.
- **No IoU / no confusion matrix / no precision.** For a heavily class-imbalanced segmentation problem, accuracy is close to meaningless and IoU is the field-standard metric. Its absence is the paper's biggest reporting weakness.
- **Confounded ablation:** augmentation is switched on at epoch 150, so the epoch sweep is not a clean study of epochs.
- **Labels are manually drawn** by the authors in ArcGIS — no inter-annotator agreement, no independent ground truth (other than the coarse CENAPRED area total).
- **Single study area, single region**, no cross-region generalization test.
- **Internal inconsistencies**: 2019 vs 2020 start year; 256×256 chips vs 512×512 network input; NDWI defined with SWIR while only bands 2/3/4/8 are listed; the augmentation text says the dataset grew "from 1215 to 4860" but 1,215 is the *permanent water* count while the augmentation is described in the *flood* (SAR, 1,315) section.
- No uncertainty quantification, no runtime/inference-time figures, no code or data release.

## Relevance to this thesis
Concretely useful, and also a useful counter-example of what *not* to report.

**Directly borrowable**
- **Architecture baseline**: a vanilla U-Net (Conv3×3+ReLU / MaxPool2×2 / UpConv3×3 / skip concat, soft-max head, cross-entropy loss) is a perfectly reasonable first PyTorch model for the thesis's flood segmentation, and this paper is the citation that a plain U-Net on Sentinel imagery is adequate. Reproducing it in PyTorch is trivial and gives an *immediate published-number reference point* (F1 ≈ 90%, acc ≈ 92%).
- **Patch strategy**: 256×256 chips is a sane, defensible chip size for the Peru pipeline; adopt it, and cite this paper for it.
- **Two-class decomposition**: separating **permanent water** from **temporary/flood water** into two distinct models/datasets is the single most transferable idea. For Peru this matters — a Sentinel-2/NDWI-only model will happily label the Amazon/Ucayali river channels as "flood". Borrow the pre-/post-flood differencing and thresholding logic to build the permanent-water mask, then treat everything above it as flood.
- **Pre-/post-flood RGB stacking trick** (R = pre-flood band, G and B = post-flood bands) is a cheap, implementable label-QA technique for producing weak labels without a labelled dataset — very relevant since the thesis has no ground-truth flood polygons for Peru.
- **Augmentation**: 90° rotations (×4) on satellite chips — trivially implementable in a PyTorch `Dataset.__getitem__`, and this paper reports it as the cause of a large jump in F1 (58% → 79% across the augmentation boundary).
- **External-validation pattern**: compare model-predicted flooded hectares against an official government inundation layer. The Peruvian analogue is **INDECI / CENEPRED / SENAMHI** flood reports — replicate exactly the §4.5 comparison (predicted Ha vs official Ha, report the margin of error). This is a cheap and highly credible validation for a thesis with no pixel-level ground truth.
- **Time-series design**: the seasonal split (dry / wet / "northern") and the per-month hectare table (Table 4) is an excellent template for a Peru results table (e.g. hectares flooded per month across 5 years for the study area — which is exactly what the Streamlit/Celery 5-year download pipeline produces).

**Directly comparable / where it differs**
- **Optical vs SAR**: the *flood* model here is **SAR (Sentinel-1)** — the thesis is Sentinel-2-only. So the flood-class numbers (92.14% / 89.38%) are **not** a fair comparison to a Sentinel-2-only thesis model; the fair comparable is the **permanent-water Sentinel-2 + NDWI model (91.11% acc, 90.32% recall, 90.71% F1)**. Use that one as the baseline the thesis must approach, and be explicit about it.
- **NDWI definition differs.** The paper uses Gao's NDWI = (NIR − SWIR)/(NIR + SWIR); the thesis uses McFeeters' NDWI = (B03 − B08)/(B03 + B08). These are different indices with different targets (vegetation liquid water vs open water). Do **not** cite this paper as support for the thesis's NDWI formula — cite McFeeters (1996) instead. Conversely, this is a good motivating point in the thesis's related-work section: the literature conflates two NDWIs.
- **No atmospheric correction** here (L1C + a 40%-threshold cloud mask); the thesis uses L2A. The thesis setup is stricter/better here — worth stating as a methodological improvement.
- **Labelling is manual (ArcGIS Pro)** — a licensed, GUI-heavy workflow that is not reproducible in the thesis's Python/Streamlit/Celery stack. The thesis needs an automated label source: either (a) NDWI-threshold weak labels, or (b) a public benchmark. That gap points straight to **Sen1Floods11** (referenced here via Katiyar et al. [22] and Bai et al. [4]) — a strong candidate public dataset the thesis should consider, and a reason to prefer it over hand-labelling.

**Verdict for the thesis**
- **Good baseline to cite and to beat on reporting rigour**: match its U-Net + 256×256 + rotation augmentation, but *add* IoU, precision, a proper held-out test set, and a confusion matrix — the thesis can honestly claim methodological improvement over this paper on evaluation.
- **Good methods citation** for: U-Net on Sentinel imagery for flood mapping; permanent-vs-temporary water separation; validation against official government flood extents; Copernicus/GEE as a data source.
- **Not a dataset source** (no public data or code release).
- **Good related-work citation** for the SAR-vs-optical trade-off argument that the thesis must make to justify going optical-only (and the honest counter-argument: clouds during floods — this paper is the strongest source for that objection, so the thesis must pre-empt it, e.g. with the <10% cloud filter + multi-date compositing already in the pipeline).

## Keywords / Tech
- **Models**: U-Net (encoder–decoder CNN with skip connections); soft-max classifier; cross-entropy loss. (Related-work mentions: SegNet, BASNet, ResNet, CNN, RNN, SVM, RF, CART.)
- **Sensors**: Sentinel-1 (SAR, GRD, IW, VV+VH, descending); Sentinel-2 (L1C, bands 2/3/4/8, <10% cloud).
- **Indices**: NDWI = (NIR − SWIR)/(NIR + SWIR) [Gao formulation]. NDVI mentioned but not used.
- **Techniques**: 5×5 Lee speckle filter; radiometric correction; geometric calibration; log/dB scaling; texture-coefficient histogram thresholding; coherence-coefficient thresholding; cloud & shadow masking (<40% threshold); pre-/post-flood RGB band stacking; 90° rotation augmentation; 256×256 chipping; post-processing removal of small pixel groups.
- **Tools / platforms**: ESA **SNAP** (SAR preprocessing); **ArcGIS Pro** + Training Samples Manager (labelling and DL training); **Google Earth Engine** (optical acquisition); **Copernicus Open Access Hub** (SAR acquisition).
- **Reference data**: **CENAPRED** official flood inundation maps (Mexico).
- **Metrics**: Accuracy, Recall, F1-Score. (No IoU, no precision, no kappa.)

## Notable quotes
- "This paper presents a methodology for flood mapping in the Ríos zone of Tabasco State using Sentinel-1 SAR, Sentinel-2, and U-Net deep learning architecture. The study period was from 2019 to 2023. The results obtained show that with more data and training periods, accuracy in detecting floods improves." (Abstract, p. 393)
- "With a training cycle of 30 epochs and 10% validation, an excellent level of evaluation metrics was achieved: precision of 91.11%, recall of 90.32% and F1-Score of 90.71%." (§4.1, permanent water, p. 401)
- "From epoch 150 onwards, a strategy was implemented to increase the available dataset size by implementing the rotation technique. […] The new versions of each natural image were generated by applying 90-degree rotations at specific intervals, resulting in three different versions per original image. This strategy significantly enriched the training dataset (increasing from 1215 to 4860 images)." (§4.2, p. 401)
- "CENAPRED detected 21,196.15 hectares (Ha) of flooded areas, while our model estimated 15,002.72 Ha. This results in a margin of error of 29.22% for the model, indicating that our estimation was approximately 29.22% outside of CENAPRED's reported value." (§4.5, pp. 402–403)
- "It is worth noting that the lack of suitable hardware slows down model training. To improve the model, it requires being trained with a larger dataset and more epochs." (§5 Conclusions, p. 404)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/BEJIG3FM/Pech-May et al. - 2025 - Flood Mapping through Sentinel-1, Sentinel-2 Imagery and U-NET Deep Learning Model.pdf`
Cite as: `\cite{pech-may_flood_2025}`
