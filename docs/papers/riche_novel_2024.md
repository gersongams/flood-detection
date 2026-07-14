---
title: "A Novel Hybrid Deep-Learning Approach for Flood-Susceptibility Mapping"
authors: "Abdelkader Riche, Ammar Drias, Mawloud Guermoui, Tarek Gherib, Tayeb Boulmaiz, Boularbah Souissi, Farid Melgani"
year: 2024
venue: "Remote Sensing (MDPI), vol. 16, issue 19, article 3673, 41 pp."
doi: "10.3390/rs16193673"
bibtex_key: riche_novel_2024
zotero_pdf: "/Users/gersongarrido/Zotero/storage/2RSVTZQ5/Riche et al. - 2024 - A Novel Hybrid Deep-Learning Approach for Flood-Susceptibility Mapping.pdf"
pages: 41
sensors: [Landsat-5 TM, Landsat-8 OLI, SRTM DEM]
task: "flood-susceptibility mapping (FSM) as image-to-image binary segmentation"
model: "W-Res-U-Net (Weighted Residual U-Net) with dual sigmoid-focal-loss; baselines RF, 2D-CNN, U-Net, Res-U-Net, HEC-HMS/RAS"
tags: [flood-susceptibility, u-net, residual-unet, focal-loss, hec-ras, hec-hms, physical-model-hybrid, semantic-segmentation, geospatial-factors, algeria]
---

# A Novel Hybrid Deep-Learning Approach for Flood-Susceptibility Mapping

> **TL;DR** — The authors reframe flood-susceptibility mapping (FSM) as an *image-to-image* segmentation problem: 19 stacked geospatial factor rasters (30 m) go in, a binary flood/non-flood susceptibility map comes out. They introduce a "Weighted Residual U-Net" (W-Res-U-Net) trained with **two simultaneous sigmoid-focal losses** — one against a HEC-HMS/RAS hydraulic simulation target, one against 63 real ground-truth flood points — balanced by a scalar `Alpha`. On the Wadi El Harrach sub-watershed (Algiers, Algeria; >800 km²), trained on 8 sub-basins and tested on 5 others, W-Res-U-Net at Alpha = 0.1 reaches **sensitivity 71.16%, specificity 91.14%, AUC 92.95% against the physical simulation, and sensitivity 88.89%, specificity 93.07%, AUC 95.87% against ground-truth points** (abstract, p. 1) — beating HEC-RAS alone (sensitivity 72.22%, AUC 85.09%, p. 35). RF completely fails to generalize across sub-basins (sensitivity 0%, AUC 50, p. 28).

## Problem & Motivation

FSM (predicting *where floods could occur*, as opposed to detecting an ongoing inundation) is bottlenecked by three things. (1) Physical models (HEC-HMS/RAS, SWAT, LISFLOOD-FP, SOBEK) need intricate inputs, expert-defined cross-sections and riverbanks, and long compute times, and struggle beyond ~1000 km² watersheds (p. 3). (2) MCDA/AHP methods need expert-defined weights and are therefore subjective (p. 3). (3) ML/DL methods need large flood inventories, which rarely exist — and remote sensing cannot always supply them because SAR is not always acquired on the flood day and optical imagery is cloud-blocked (p. 2).

The authors' framing: prior ML/DL FSM work is almost entirely *point-based* (tabular geospatial factors → flooded/non-flooded label per point). They claim to be the first to do "images-to-image" FSM using geospatial factors as input channels and a *segmented flood map* as target, and the first to hybridize a physical-model target with sparse ground-truth points inside a single dual-loss deep network (pp. 8-9). Table 1 (pp. 5-8) is a large, genuinely useful literature table of ~35 prior FSM studies with model / inputs / location / metrics.

## Method / Architecture

**Overall pipeline (Figure 8, p. 23):**

1. Build a 19-band geospatial raster stack (30 m) for the watershed; treat it as a multi-channel image.
2. Run HEC-HMS (rainfall→runoff, SCS Curve Number) to get flow discharge hydrographs; feed them into HEC-RAS (1D/2D hydraulic) to produce simulated flooded-area rasters for **5, 10, 20, 50 and 100-year return periods**, paired with LULC/CN of 2000, 2005, 2010, 2015, 2020 respectively (p. 18). These rasters are the primary segmentation target.
3. Rasterize 63 civil-protection ground-truth flood points as pixels coded 1 (flooded) / 0 (non-flooded) — a second, extremely sparse target (p. 18).
4. Train segmentation models to map the 19-channel stack → binary flood map, in three experimental steps.

**Three-step learning strategy (Section 5, pp. 22-24):**
- **Step 1 — time-series modeling:** train on single vs. cumulative year ranges (2000; 2000-2005; …; 2000-2020) with binary cross-entropy, target = HEC-RAS only. Pick the best training period.
- **Step 2 — loss comparison:** BCE vs. Sigmoid Focal Loss (α = 0.75, γ = 0.25, p. 30) to handle the heavy 0/1 imbalance of the HEC-RAS target.
- **Step 3 — dual loss (the contribution):** `Loss_total = (1 − Alpha) × Loss₁ + Alpha × Loss₂` (Eq. 25, p. 24), where **Loss₁ = focal loss on the ground-truth points** and **Loss₂ = focal loss on the physical-model target**. Alpha swept 0 → 1 in 0.1 steps. Alpha = 0 ⇒ ground truth only; Alpha = 1 ⇒ HEC-RAS only.
  - ⚠️ Note the counter-intuitive naming: with this equation, *low* Alpha weights the ground-truth branch heavily. The paper's own prose at p. 33 ("At Alpha = 1, focusing solely on HEC-RAS") is consistent with Eq. 25, but is easy to misread. The optimum they report is **Alpha = 0.1** for all three DL models.

**W-Res-U-Net architecture (Figure 9, p. 25; described in prose p. 35):**
- U-Net encoder/decoder with skip connections **plus residual connections** inside blocks.
- Each conv layer followed by **BatchNorm + ReLU**; max-pooling for down-sampling; up-sampling + concatenate in the decoder; **sigmoid** output.
- Filter kernels of "varying sizes (e.g., 5 × 5, 3 × 3, 2 × 2)" and filter counts increasing **16, 32, 64, 128, 256, 512** with depth (p. 35).
- Residual block: `Y = F(X, Wᵢ) + X`, with `F(X,Wᵢ) = r W₂ r(W₁X)`, where the weight layers are *dilated* convolutions (Eqs. 15-16, p. 15).
- The "W" (weighted) refers to the dual weighted loss, not to a weighting layer.

**Baselines implemented:** Random Forest (Eqs. 4-5, p. 12), 2D-CNN (Figure 2, p. 13), plain U-Net, Res-U-Net, and the HEC-HMS/RAS physical simulation itself.

**Not reported:** optimizer, learning rate, number of epochs, batch size, patch size / tiling scheme, weight initialization, data augmentation, RF hyperparameters (n_trees, depth), framework/library, hardware, training time, number of runs / random seeds (no std-devs anywhere). Feature selection by RF importance thresholding was tried and **abandoned** because "the model performances did not improve significantly" — all 19 factors were kept (p. 22).

## Data

**Study area:** Wadi El Harrach sub-watershed, Algiers region, North-Central Algeria; >800 km²; 2°56′34″–3°19′39″E, 36°25′11″–36°46′40″N; elevation −1 to 1256 m; rainfall return periods 60–166 mm; river flow discharge 24–1690 m³/s; 9 rainfall stations + 1 flow-monitoring station (p. 9).

**Sensors / sources:**
- **SRTM DEM, 1 arc-second (30 m)** → DEM, slope, flow accumulation, flow direction, TWI, TRI, TPI (p. 15).
- **Landsat-5 TM** (25 Feb 2000, 21 Jan 2005, 16 Mar 2010) and **Landsat-8 OLI** (10 Feb 2015, 24 Feb 2020) → LULC (SVM classification), NDVI, NDBI, MNDWI. Dates chosen in the winter rainy season with **cloud cover ≤ 10%** (p. 16).
- Rainfall from 9 gauges, **IDW-interpolated** to rasters (p. 17).
- **No Sentinel-1 or Sentinel-2 data are used.**

**19 input factors** (p. 15-18), all rasterized at **30 m**:
- *Topographic:* DEM, slope, flow accumulation, flow direction, TWI, TRI, TPI
- *Environmental:* LULC, lithology, NDVI, NDBI, MNDWI
- *Meteorological/hydrological:* precipitation, **flow discharge** (novel factor, from HEC-HMS), curve number (CN), hydrologic soil group (HSG), SPI, STI, distance to rivers (DTR)

**Index formulas as printed in the paper** (note the missing parentheses — they are typos in the PDF):
- `NDVI = (NIR − RED / NIR + RED)` (Eq. 18, p. 17) ⚠️ should be (NIR−RED)/(NIR+RED)
- `NDBI = (SWIR − NIR / SWIR + NIR)` (Eq. 19, p. 17) ⚠️ same typo
- `MNDWI = (Green − SWIR / Green + SWIR)` (Eq. 20, p. 17) ⚠️ same typo — this is **MNDWI (Xu 2006, SWIR-based)**, *not* the McFeeters NDWI used in this thesis
- `TWI = ln(a / tan(β))` (Eq. 17, p. 16)
- `SPI = A × tan(β)` (Eq. 21, p. 17); `STI = (A/a₀)^m × (sin(β)/b₀)^n` (Eq. 22, p. 18)

**Ground truth:** **63 flood points** from civil-protection authorities — 37 river-flood, 26 flash-flood (p. 18). Split: **41 for training, 22 for testing in other sub-basins** (p. 32). Pixels coded 1 = flooded, 0 = non-flooded.

**Split:** trained on **8 sub-basins**, tested on **5 other sub-basins** (Conclusions, p. 37; Figure 7, p. 23). Pixels outside the sub-basin under consideration are treated as No-Data. Two scenarios: (1) train and test on the *same* sub-basins at different years; (2) train and test on *different* sub-basins. There is no third fully held-out set — the Alpha sweep is tuned and reported on the same test sub-basins, so the "test" is effectively a validation set. ⚠️

**Preprocessing:** not reported beyond LULC SVM classification and IDW interpolation of rainfall — no normalization, no cloud-masking procedure, no atmospheric-correction level stated for Landsat.

**Data availability:** dataset + model code stated as openly available at `https://disi.unitn.it/~melgani/datasets.html` (p. 38).

## Results

Threshold 0.5 for binarization; the authors explicitly focus on **sensitivity** (recall of flooded pixels) (p. 25).

### Headline — Step 3, dual loss, W-Res-U-Net (Table 5, p. 32; Table 6, p. 35)

| Model (Alpha = 0.1) | Val. vs HEC-RAS: Acc / Spe / Sen / AUC | Val. vs Ground Truth: Acc / Spe / Sen / AUC |
|---|---|---|
| CNN | 95.03 / 95.81 / 57.55 / 95.32 | 94.72 / 94.72 / 66.67 / 94.91 |
| U-Net | 92.72 / 93.31 / 64.41 / 93.28 | 92.14 / 92.14 / 83.33 / 95.06 |
| **W-Res-U-Net** | **90.73 / 91.14 / 71.16 / 92.95** | **93.07 / 93.07 / 88.89 / 95.13** |
| HEC-RAS simulation (physical model, Table 6 p. 35) | — | 97.96 / 97.96 / **72.22** / **85.09** |

All values %, from Table 5 (p. 32) and Table 6 (p. 35). **The DL models beat HEC-RAS on flood-point sensitivity and AUC**, while HEC-RAS keeps higher accuracy/specificity (it under-predicts flooding). W-Res-U-Net improves sensitivity over HEC-RAS by **+16.67 pp** (88.89 vs 72.22) and AUC by **+10.04 pp** (95.13 vs 85.09) on the 22 held-out ground-truth points.

⚠️ **Inconsistency (important if you cite this):** the abstract (p. 1) and the Conclusions (p. 38) state ground-truth **AUC = 95.87%** for W-Res-U-Net, but **Table 5 (p. 32) and the text on p. 35 both report AUC = 95.13%**. The 95.87 figure appears nowhere in the tables. Additionally, p. 33 §6.3.3 says "When validating against ground truth points, the best sensitivity was at Alpha = 0.1, with 83.33%" for W-Res-U-Net — but Table 5 gives **88.89%** at that cell (83.33% is the *U-Net* value; §6.3.3 appears to be a copy-paste of §6.3.2). Prefer the tables.

### Step 1 — cross-sub-basin generalization, target = HEC-RAS only, BCE loss (Table 3, p. 28; best row = train 2000-2020, test 2020)

| Model | Acc | Spe | Sen | AUC |
|---|---|---|---|---|
| RF | 97.96 | 100 | **0** | **50** |
| CNN | 94.81 | 95.52 | 61.23 | 95.02 |
| U-Net | 96.78 | 97.01 | 65.29 | 95.23 |
| Res-U-Net | 96.52 | 97.24 | **67.36** | **95.67** |

RF gets 0% sensitivity and AUC = 50 across *every* training period when tested on unseen sub-basins (p. 26, Table 3) — it predicts "no flood" everywhere and coasts on the 98% non-flood class. This is the paper's most striking negative result and a strong argument for segmentation over point/tabular ML. Within the *same* sub-basins (Table 2, p. 27), RF is less catastrophic but still weak (Sen ≈ 48-58%), while Res-U-Net reaches **Sen 97.76% / AUC 98.56%** (train 2000-2020, test 2000). Multi-year (2000→2020) training consistently beats single-year training.

### Step 2 — Sigmoid Focal Loss vs BCE (Table 4, p. 30; train 2000-2020, test 2020, other sub-basins)

| Model | Acc | Spe | Sen | AUC |
|---|---|---|---|---|
| CNN | 95.09 (+0.28) | 95.73 (+0.21) | 61.46 (+0.23) | 95.34 (+0.32) |
| U-Net | 96.92 (+0.14) | 97.42 (+0.41) | 65.39 (+0.10) | 95.38 (+0.15) |
| Res-U-Net | **97.07 (+0.55)** | **97.69 (+0.45)** | **67.63 (+0.27)** | **95.84 (+0.17)** |

Focal loss gives only a **marginal** gain over BCE (+0.10 to +0.27 pp sensitivity). ⚠️ The Conclusions (p. 38) describe this as "employing Sigmoid Focal Loss ... increased its sensitivity by 0.27 and AUC by 0.17 **compared to HEC-RAS simulations**" — that is wrong; the deltas in Table 4 are vs. the **binary cross-entropy** model, not vs. HEC-RAS.

### Alpha sweep behavior (Table 5, p. 32; Figure 15, p. 33)
At Alpha = 0 (ground-truth-only loss), all three DL models hit sensitivity ≈ 100% but specificity ≈ 53% — i.e. they flood the entire region; the authors call this "unrealistic" (p. 33). At Alpha = 1 (physical-model-only loss), sensitivity drops to 61-68%. The dual loss at Alpha = 0.1 is the sweet spot for every model.

**Baselines actually run:** RF, 2D-CNN, U-Net, Res-U-Net, HEC-HMS/RAS. **No comparison against any other published FSM method was run** — Table 1 is a literature table, not an experimental comparison, and the numbers in it come from different study areas and are not comparable.

## Limitations

Stated by the authors (pp. 37-38):
- Difficulty generalizing across sub-basins with different hydrological characteristics.
- Image-based segmentation is not fairly comparable to point-based FSM methods (different data representation) — so no head-to-head with the literature is offered.
- Class imbalance (flooded ≪ non-flooded) remains an ongoing challenge even after focal loss.
- HEC-RAS itself is limited by cross-section and riverbank definition, which is precisely why they add ground-truth points.

Observed by me:
- **Only 63 ground-truth points**, 22 of which form the entire real-world test set. Sensitivity of 88.89% is computed over a handful of positive points — 88.89% ≈ 8/9. Confidence intervals are enormous; **no uncertainty quantification, no std-dev, no repeated runs.**
- **Single study area, single watershed.** No transfer test to another basin/country.
- The training target is largely a *simulation*, so the model is partly learning to imitate HEC-RAS, including its errors — the authors acknowledge validation against HEC-RAS is "somewhat unclear" (p. 32).
- **The Alpha hyperparameter is selected on the same test sub-basins on which the headline numbers are reported** — no independent held-out set.
- Hyperparameters (LR, epochs, batch size, optimizer, patch size) are **entirely unreported**, so the work is not reproducible from the paper alone (the code link may help).
- The three metric inconsistencies flagged above (95.87 vs 95.13 AUC; 83.33 vs 88.89 sensitivity; focal-loss deltas misattributed to HEC-RAS) suggest a rushed write-up.
- Accuracy/specificity are near-useless here (98% of pixels are non-flood); the paper is right to focus on sensitivity/AUC, but Table 2 still bolds accuracy values.

## Relevance to this thesis

**High relevance methodologically, moderate relevance data-wise. This is a "method to cite" and a design template, not a baseline you can beat numerically** (different sensor, different country, different resolution, and the metrics are not transferable).

Directly borrowable:
- **The image-to-image (segmentation) framing itself.** This is the single most useful takeaway: instead of a CNN over per-pixel NDVI/NDWI vectors, stack your factors as channels and predict a full susceptibility raster with an encoder-decoder. Their RF result (**0% sensitivity, AUC 50 on unseen sub-basins**, p. 28) is a strong published justification for *why* your thesis should not stop at pixel-wise/tabular classifiers.
- **W-Res-U-Net = U-Net + residual blocks + BatchNorm/ReLU, filters 16→512, sigmoid output.** Trivially implementable in PyTorch and a fair architecture to adopt for the thesis' CNN. Res-U-Net consistently outperforms plain U-Net and plain CNN on sensitivity in every table.
- **Sigmoid Focal Loss with α = 0.75, γ = 0.25** (p. 30) as the loss for the severe flood/non-flood imbalance — directly usable (`torchvision.ops.sigmoid_focal_loss`). Expect only a small gain (+0.1 to +0.3 pp sensitivity), which is itself a useful, honest calibration of expectations.
- **The dual-loss idea generalized to your setting:** if Peruvian ground-truth flood inventories (e.g. INDECI / CENEPRED / SENAMHI records, or Sentinel-1-derived flood masks) are sparse, you can combine a weak/abundant target (e.g. an NDWI-threshold water mask, or a HAND/topographic proxy) with the sparse authoritative points via `L = (1−α)·L_gt + α·L_weak`, sweeping α. Their finding that **α ≈ 0.1 is optimal, and α = 0 collapses to "everything floods"** is an actionable prior.
- **Reporting protocol:** report **sensitivity, specificity, AUC** and be explicit that accuracy is inflated by class imbalance. Their metric equations are Eqs. 26-29 (p. 25).
- **Spatial split by sub-basin** (train on 8 basins, test on 5) instead of a random pixel split — this is the correct way to avoid spatial leakage, and a rigor point worth adopting and citing.
- **Multi-year/multi-scenario training beats single-year** (Tables 2-3) — supports the thesis' plan to use a 5-year time series rather than a single date.
- **Factor list** — the 19 factors (DEM, slope, flow accumulation, flow direction, TWI, TRI, TPI, LULC, lithology, NDVI, NDBI, MNDWI, precipitation, flow discharge, CN, HSG, SPI, STI, DTR) is a ready-made checklist if the thesis expands beyond NDVI/NDWI to true *susceptibility* (as opposed to inundation detection). Several are derivable in Peru from SRTM/Copernicus DEM + Sentinel-2.

Key differences / cautions:
- **Sensor mismatch:** they use **Landsat (30 m)**, not Sentinel-2 (10 m). Their water index is **MNDWI (Green−SWIR)/(Green+SWIR)** — Sentinel-2 has SWIR (B11/B12) at 20 m, so MNDWI is available to you but at 20 m, whereas your thesis' McFeeters NDWI `(B03−B08)/(B03+B08)` is 10 m. Worth noting in the thesis that MNDWI is generally the better water discriminator over built-up areas, which matters for a Peruvian urban/peri-urban flood study.
- **Task mismatch:** this is **susceptibility** (where *could* it flood, driven by terrain/hydrology), not **inundation mapping** (where *is* water now, driven by spectral indices). The thesis title says "identify *and predict*" flood-prone areas — so both. This paper is the reference for the "predict/susceptibility" half; it does **not** help with the "identify current inundation" half.
- **They depend on HEC-HMS/HEC-RAS**, which the thesis does not have. If you can't run a hydraulic model, you cannot replicate the dual-loss exactly — you need a substitute weak target (see above). Say so explicitly rather than pretending equivalence.
- **Not a numeric baseline.** Do not put their 88.89% sensitivity in a comparison table against your Peruvian results; different area, different target, n=22 test points.
- **Cautionary example:** the paper's four internal metric contradictions are a good reminder to keep your own tables/abstract in sync.

## Keywords / Tech

- **Models:** W-Res-U-Net (Weighted Residual U-Net), Res-U-Net, U-Net, 2D-CNN, Random Forest
- **Physical models:** HEC-HMS (SCS Curve Number rainfall-runoff), HEC-RAS (1D/2D hydraulics; continuity, momentum, Manning's equation)
- **Losses:** Binary Cross-Entropy; Sigmoid Focal Loss (α = 0.75, γ = 0.25); weighted dual loss `(1−Alpha)·L_gt + Alpha·L_physical`
- **Sensors/data:** SRTM DEM (30 m), Landsat-5 TM, Landsat-8 OLI, 9 rain gauges, 63 civil-protection flood points
- **Indices:** NDVI, NDBI, MNDWI, TWI, TRI, TPI, SPI, STI, CN, HSG, DTR, flow discharge (novel)
- **Techniques:** image-to-image semantic segmentation, residual connections, dilated convolutions, batch normalization, IDW interpolation, SVM land-cover classification, RF feature-importance selection (tried, discarded), return-period scenario simulation (5/10/20/50/100 yr), spatial cross-validation by sub-basin
- **Metrics:** accuracy, sensitivity (recall), specificity, AUC-ROC
- **Frameworks:** not reported (dataset + code released at https://disi.unitn.it/~melgani/datasets.html)

## Notable quotes

> "there are no studies considering geospatial factors as inputs and images as Target 'Images to Image'." (p. 5)

> "The RF model consistently fails to detect flooded areas in other sub-basins, maintaining a sensitivity of 0% across all training periods." (p. 26)

> "When validating with HEC-RAS, the highest sensitivity (71.16%) was at Alpha = 0.1, with a specificity of 91.14%. Focusing only on HEC-RAS (Alpha = 1) yielded a sensitivity of 67.02% while focusing only on ground truth points resulted in a sensitivity of 99.89% but with a specificity of 53.34%." (p. 33)

> "These findings indicate that the Res–U-net model excels in detecting flooding points, with the highest AUC of 95.13% ... whereas the HEC–RAS simulation has an AUC of 85.09%." (p. 35)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/2RSVTZQ5/Riche et al. - 2024 - A Novel Hybrid Deep-Learning Approach for Flood-Susceptibility Mapping.pdf`
Cite as: `\cite{riche_novel_2024}`
