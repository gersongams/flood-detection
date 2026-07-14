---
title: "Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning"
authors: "Goutam Konapala, Sujay V. Kumar, Shahryar Khalique Ahmad"
year: 2021
venue: "ISPRS Journal of Photogrammetry and Remote Sensing, vol. 180, pp. 163–173"
doi: "10.1016/j.isprsjprs.2021.08.016"
bibtex_key: konapala_exploring_2021
zotero_pdf: "/Users/gersongarrido/Zotero/storage/VZIDPI7Z/Konapala et al. - 2021 - Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning.pdf"
pages: 11
sensors: [Sentinel-1, Sentinel-2, SRTM DEM]
task: "flood inundation mapping (binary water/non-water semantic segmentation)"
model: "U-Net (fully convolutional encoder–decoder), Keras/TensorFlow"
tags: [flood-inundation-mapping, sentinel-1, sentinel-2, u-net, ndwi, mndwi, awei, hsv, sen1floods11, sar-optical-fusion, dem]
---

# Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning

> **TL;DR** — The authors benchmark **32 input combinations** of Sentinel-1 (SAR VV/VH), Sentinel-2 (optical spectral indices, raw bands, and an HSV colour-space transform) and a 10 m-resampled SRTM DEM, feeding each into the **same U-Net** for binary water segmentation on the **Sen1Floods11** dataset (446 hand-labelled 512×512 images, 11 flood events, 10 m). Optical (S2) clearly beats SAR: S1 alone reaches a **median F1 of 0.62** (precision 0.59, recall 0.88), rising to **0.73** when the DEM is added; every S2 **spectral-index** configuration lands between **0.88 and 0.90** (p. 167, Table 4) — note the two *raw-band* variants sit below that floor (rHSV 0.86, rNDWI 0.87). The best single input is the **HSV transform of S2 RGB bands (F1 = 0.90)**, statistically significantly better than the classic water indices (cNDWI/cAWEI, F1 ≈ 0.89), and **fusing S1 with S2 adds no statistically significant gain** in this (largely cloud-free) dataset — although a Ghana case shows fusion helps under heavy cloud (p. 170–171).

## Problem & Motivation

Flood mapping historically relies on either optical/multispectral (MS) sensors — which cannot see through the cloud cover that accompanies flood-producing precipitation — or SAR, which is all-weather but plagued by speckle and by confusion between water and water-like smooth surfaces (p. 163). Prior fusion studies blended S1 and S2 largely through NDWI, but "no prior study has jointly evaluated (1) the sensitivity of combination of S1 and S2 bands on the performance of flood inundation mapping in the context of deep learning and (2) the robustness of performance of the band combinations for flood inundations across a dataset with diverse land cover spanning across 5 continents" (p. 164).

The paper therefore asks two questions (p. 164): **(1) What is the optimal combination of S1 and S2 bands for flood inundation mapping through deep learning approaches? (2) Does the combination of S1 and S2 perform better than the individual performance of S1 and S2?** The intent is explicitly practical: knowing the optimal combination up front "will save time and computational resources when generating flood inundation mapping in near real time settings" (p. 164).

## Method / Architecture

- **Model**: U-Net (Ronneberger et al., 2015), encoder–decoder with skip connections that concatenate encoder feature maps into the decoder at each level (p. 166, Fig. 2). Input is **512 × 512**; the diagram shows conv 3×3 + ReLU blocks, 2×2 max-pool downsampling to a 32×32 bottleneck (channel multipliers 2x → 32x) and up-conv 3×3 upsampling back to an output mask. The standard U-Net was **modified to accept an arbitrary number of S1/S2/DEM channels instead of the traditional 3 RGB bands** (p. 172).
- **Framework**: **Keras + TensorFlow**, trained on a single **NVIDIA V100 GPU** (p. 167).
- **Optimizer**: **Adam**, base learning rate **5e-4**, weight decay coefficient **1e-2** (p. 167).
- **Epochs**: **500**, one separately trained model for each of the 32 input combinations; the model at the *end* of the last epoch is used (no early stopping / best-checkpoint selection) (p. 167).
- **Loss function**: ⚠️ **not reported** (never named in the paper).
- **Batch size**: **not reported**. Dropout / batch-norm / activation of the output layer: **not reported**.
- **No hyperparameter search**: "As our goal is to evaluate the S1 and S2 combination rather than train the best possible models, we do not perform an exhaustive hyperparameter search" (p. 167).
- **Augmentation**: the training set in each fold is "augmented fourfold by flipping the S1, S2 and ground truth images **up, down, right and left**" (p. 166).
- **Validation scheme — modified k-fold by country** (p. 166): from the 11 countries, randomly select **9 countries for training** and hold out the remaining **2 countries for testing**; repeat for **k = 10** folds with different random splits; report the **median** of per-image metrics across all test samples over all 10 folds. This deliberately avoids spatial autocorrelation: "By using a testing subset from countries which are not used in training, we avoid spatial autocorrelation" (p. 166). ⚠️ No separate validation split is described — only train/test.
- **Experiment grid**: **32 input combinations** (p. 166, Table 3), covering S1 alone, S2 feature-engineered indices (cNDWI = NDWI+MNDWI; cAWEI = AWEIsh+AWEInsh; HSV), the *raw bands* used to compute each index (rNDWI, rAWEI, rHSV), all their combinations, S1+S2 combinations, and a with/without **DEM** variant of every configuration.
- **Statistics**: non-parametric **Kruskal–Wallis** test between pairs of combinations; significance at **p < 0.05** (p. 167–168, Fig. 3).

## Data

- **Dataset**: **Sen1Floods11** (Bonafilia et al., 2020) — a georeferenced flood-label dataset. **446 human-annotated images at 10 m resolution, 512 × 512 pixels**, spanning **11 flood events across 11 countries** (Bolivia, Ghana, India, Vietnam, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA; 2016–2019; Table 1, p. 165). The 446 hand-labelled images were "selected by stratified sampling from a larger pool of **4385 images**" for those 11 events (p. 164). Each labelled event has matched S1 and S2 acquisitions.
- **Ground truth**: hand-labelled (human-annotated) pixel-level water masks supplied with Sen1Floods11 (p. 164, p. 172).
- **Sentinel-1**: **VV and VH** bands only, used **directly with no further combinations/indices** ("As S1 SAR has only two available bands with VV and VH polarizations, we use them directly in the machine learning setup without making any further combinations/indices", p. 165). VH thresholds per event are listed in Table 1 (e.g. < −20.44 dB Bolivia) but are dataset metadata, not used by the CNN. No speckle filter is mentioned — ⚠️ **speckle filtering not reported**.
- **Sentinel-2**: **all 12 bands available; resampled linearly to 10 m** for common registration (p. 164). ⚠️ Atmospheric correction level (L1C vs L2A) is **not reported** in this paper (Sen1Floods11 ships both; the paper does not say which is used). No cloud masking is applied — Sen1Floods11 "is curated to exclude majority of the satellite imagery with clouds" (p. 171).
- **DEM**: **SRTM, 30 m**, void-filled, **resampled linearly to 10 m** and used as an ancillary channel (p. 165).
- **Spectral indices used (Table 2, p. 165)** — verbatim:
  - `MNDWI = (GREEN − SWIR1) / (GREEN + SWIR1)` (Xu, 2006)
  - `NDWI = (GREEN − NIR) / (GREEN + NIR)` (McFeeters, 1996)
  - `AWEI = 4 * (GREEN − SWIR1) − (0.25 * NIR + 2.75 * SWIR2)` (Feyisa et al., 2014)
  - `AWEISH = BLUE + 2.5 * GREEN − SWIR1 − 1.5 * (NIR + SWIR1) − 0.25 * SWIR2` (Feyisa et al., 2014)
  - ⚠️ **Inconsistency**: Table 2 names the two AWEI variants "AWEI" and "AWEISH", but the body text consistently calls them "**AWEIsh and AWEInsh**" (pp. 166–167). The AWEISH row as printed also appears garbled (it contains both `− SWIR1` and `− 1.5 * (NIR + SWIR1)`); treat the AWEIsh formula as unreliable as printed and re-derive from Feyisa et al. (2014) if needed.
  - **cNDWI** = NDWI + MNDWI combined; **cAWEI** = AWEIsh + AWEInsh combined; **rNDWI / rAWEI / rHSV** = the *raw* S2 bands that go into each index (p. 166, Table 3).
- **HSV transform (Eqs. 1–3, p. 165)**: **SWIR2, NIR, RED are assigned to Red, Green, Blue** respectively and mapped into HSV via the standard colorimetric transform (Smith, 1978):
  - `V = max(R,G,B)` (1)
  - `S = V − min(R,G,B)` (2)
  - `H` = the piecewise 60°-sector formula (3), H ∈ [0°, 360°].
- **⚠️ NDVI is NOT used** in this study — it is mentioned only historically (Barton & Bathols, 1989, p. 164).
- **Normalization**: not reported.

## Results

**Table 4 — median performance for individual S1/S2 inputs (p. 167)**, medians over per-image metrics across all 10 folds:

| Input | F1 | Precision | Recall | Type |
|---|---|---|---|---|
| **S1** (VV,VH) | **0.62** | 0.59 | 0.88 | Original bands |
| **S1 + DEM** | **0.73** | 0.68 | 0.86 | Original bands |
| cAWEI | 0.89 | 0.85 | 0.93 | Feature-engineered |
| cAWEI + DEM | 0.88 | 0.86 | 0.93 | Feature-engineered |
| cNDWI | 0.89 | 0.86 | 0.93 | Feature-engineered |
| cNDWI + DEM | 0.88 | 0.86 | 0.93 | Feature-engineered |
| **HSV** | **0.90** | 0.89 | **0.94** | Feature-engineered |
| **HSV + DEM** | **0.90** | **0.89** | 0.93 | Feature-engineered |
| rNDWI | 0.87 | 0.86 | 0.92 | Original bands |
| rNDWI + DEM | 0.88 | 0.87 | 0.91 | Original bands |
| rAWEI | 0.88 | 0.87 | 0.92 | Original bands |
| rAWEI + DEM | 0.88 | 0.85 | 0.93 | Original bands |
| rHSV | 0.86 | 0.86 | 0.89 | Original bands |
| rHSV + DEM | 0.87 | 0.85 | 0.91 | Original bands |

**Table 5 — combinations *within* S2 (p. 170)**: cAWEI+cNDWI **0.86** (P 0.85 / R 0.91); cAWEI+cNDWI+DEM **0.88**; HSV+cAWEI+cNDWI **0.90** (P **0.89** / R 0.92); HSV+cAWEI+cNDWI+DEM **0.90** (P 0.88 / R **0.93**); rAWEI+rNDWI 0.87; rAWEI+rNDWI+DEM 0.87; rHSV+rAWEI+rNDWI 0.88; rHSV+rAWEI+rNDWI+DEM 0.86. **Combining S2 indices gives no statistically significant advantage over the best individual index** (p. 169–170).

**Table 6 — S1+S2 combinations (p. 170)**: S1+cAWEI **0.88**; S1+cAWEI+DEM 0.87; S1+cNDWI 0.88; S1+cNDWI+DEM 0.89; S1+cAWEI+cNDWI 0.88; S1+cAWEI+cNDWI+DEM 0.89; S1+HSV **0.90**; S1+HSV+DEM **0.90** (P 0.90 / R 0.93); S1+cAWEI+cNDWI+HSV **0.90**; S1+cAWEI+cNDWI+HSV+DEM **0.90**. **All best-performing combinations contain HSV, and none is significantly better than HSV alone** (p. 170).

Key findings:
- **SAR alone is the worst input by a wide margin** — F1 0.62 vs ~0.89 for optical, driven by **under-segmentation / low precision (0.59)** (p. 168). Adding the DEM lifts S1 to 0.73 mainly by improving precision to 0.68 (p. 168).
- **All S2 spectral-index inputs are statistically significantly better than S1** in F1, precision *and* recall (Kruskal–Wallis, Fig. 3, p. 168).
- **HSV is statistically significantly better than the water indices** (cNDWI, cAWEI), median F1 **0.90**, attributed to HSV's "superior contrast distinguishing abilities" (abstract, p. 163; p. 168).
- **The DEM adds essentially nothing to S2** ("There does not appear to be any significant advantage in combining DEM with any of spectral indices unlike the case of S1", p. 168), but is very useful for S1 (Paraguay case, Fig. 4: S1 alone fails to detect the flood; S1+DEM recovers it).
- **U-Net learns the water indices from raw bands but not HSV**: rNDWI/rAWEI ≈ cNDWI/cAWEI (0.87–0.88 vs 0.89), whereas rHSV (0.86) < HSV (0.90) — i.e. "U-net may have captured the representation of spectral indices but not that of HSV transformation" (p. 169), so **HSV must be computed as an explicit feature-engineering step before the network**.
- **Cloud caveat**: because Sen1Floods11 largely excludes cloudy scenes, fusion shows no gain — but in the cloud-covered Ghana case (Fig. 6) S2/HSV misses the river while S1 captures it, and the **S1 + HSV fusion produces the best map**: "in satellite imagery with clouds, the fusion of S1 and S2 imagery has performed significantly better than individual S2 indices" (p. 172).

**Baselines**: ⚠️ **No external baselines (no random forest, no thresholding, no other CNN architecture) were run.** The 32 input configurations are compared only against each other. The authors do informally note that prior studies report S1-based F1 between 0.65 and 0.91, so their S1 F1 of 0.62 "compares reasonably well" (p. 171).

## Limitations

Stated by the authors:
- Sen1Floods11 excludes most cloudy imagery, so the "no fusion benefit" conclusion **does not generalize to cloudy conditions** — under clouds, S1+S2 fusion is expected to help (pp. 171–172).
- A single model trained on the whole dataset performs semantic segmentation globally; **land-cover-specific ensembles or pixel-centric approaches could improve results** (p. 171).
- **U-Net's inability to form a robust threshold on VV/VH backscatter** is blamed for the poor S1 performance; modified convolutions, activations and **loss functions in U-Net** are "paving way for future research" (p. 172).
- DEM spatial resolution / source sensitivity was not tested (p. 171).
- Benchmarking **different deep learning architectures** on Sen1Floods11 is left to future work (p. 172).

Observed by me:
- **No loss function or batch size is ever reported** — the study is not fully reproducible.
- Only **446 labelled images** (11 events); the model at the final epoch (500) is used with **no validation set and no early stopping**, so overfitting is not controlled.
- Only median metrics are reported — no IoU, no OA, no kappa, no dispersion/uncertainty bounds beyond the Kruskal–Wallis significance matrix.
- The S2 processing level (L1C vs L2A) and S1 speckle filtering are unspecified.

## Relevance to this thesis

**Very high. This is a key citation and a de facto baseline.** Same sensor family, same resolution, same DL framing, and it directly answers "SAR or optical?".

Directly borrowable:
- **Architecture**: a plain U-Net with input channels widened beyond 3 is enough to hit F1 ≈ 0.90 on S2 — a strong argument for using U-Net (PyTorch) as the thesis segmentation backbone, and for **not** spending effort on exotic architectures first. Hyperparameters worth copying: **Adam, lr 5e-4, weight decay 1e-2, 512×512 patches, flip augmentation (up/down/left/right, 4×)**.
- **A cheap, high-value feature the thesis does NOT currently use: the HSV transform of S2 (SWIR2 → R, NIR → G, RED → B), Eqs. 1–3 (p. 165).** It beats NDWI/MNDWI/AWEI significantly and the network cannot learn it by itself from raw bands. **Caveat**: it requires **SWIR2 (B12)**, which is outside the thesis's current B02/B03/B04/B08 set and is only available at 20 m (needs resampling to 10 m — exactly what this paper does). Adding B11/B12 to the Sentinel Hub request is a small change with a large expected payoff.
- **NDWI is confirmed as the McFeeters form** used by the thesis: `NDWI = (GREEN − NIR)/(GREEN + NIR)` (Table 2, p. 165) — cite this paper as evidence the index works well as a CNN input channel (cNDWI F1 = 0.89).
- **NDVI is not used here** — this paper offers no support for NDVI as a flood-mapping input. If the thesis keeps NDVI in the stack, its justification must come from elsewhere (it is a vegetation/change proxy, not a water index).
- **Evaluation protocol to imitate**: report **median F1, precision and recall per image** plus a **Kruskal–Wallis significance test** between configurations, and **split train/test by geography (event/region), not randomly**, to avoid spatial autocorrelation. For a Peru-only thesis, the analogue is holding out entire river basins / events rather than random patch splits.
- **Ablation design**: the 32-combination grid (index vs raw bands, ± DEM) is a directly reusable experimental template for a thesis chapter — e.g. NDVI vs NDWI vs raw bands vs HSV, ± DEM.
- **Add a DEM channel**: SRTM 30 m resampled to 10 m. This paper shows the DEM barely helps optical inputs (⚠️ so do not expect a big gain for an S2-only thesis), but it is nearly free and is the single biggest lever for SAR.

Baselines to beat / report against:
- **Median F1 ≈ 0.89 (cNDWI) / 0.90 (HSV)** for S2-based U-Net water segmentation at 10 m — a credible target number for the thesis's inundation model.
- **Median F1 = 0.62 (S1 only)** — useful for justifying the thesis's optical-only choice: "optical outperforms SAR under cloud-free conditions."

Differences / caveats for the thesis:
- **Task mismatch**: this is *inundation mapping* (single-date, binary water segmentation), not *flood-prone-area prediction / susceptibility* over an **NDVI/NDWI time series**. Their U-Net is a single-timestep spatial segmenter; the thesis's temporal-stack CNN is a different problem, so their F1 numbers are not apples-to-apples with a susceptibility output. Say so explicitly when citing.
- **Cloud**: the thesis's 10% max-cloud filter reproduces exactly the "cloud-free" regime in which the paper finds optical superior — so the paper *supports* the thesis's design, but the Ghana example (p. 171, Fig. 6) is the honest counterargument to record in the limitations chapter: under real flood-time cloud cover, optical fails and SAR/fusion is needed.
- **Dataset source**: **Sen1Floods11 (Bonafilia et al., 2020)** is a public, hand-labelled, 10 m, global flood-label dataset (446 labelled, 4385 weakly-labelled images) — a ready-made **pre-training or transfer-learning source**, and it contains a **Bolivia** event, the closest analogue to a Peruvian setting available.

## Keywords / Tech

- **Models**: U-Net (encoder–decoder CNN, skip connections); Keras + TensorFlow; Adam; NVIDIA V100
- **Sensors**: Sentinel-1 (VV, VH, 10 m); Sentinel-2 (12 bands resampled to 10 m); SRTM DEM (30 m → 10 m)
- **Indices / transforms**: NDWI (McFeeters), MNDWI (Xu), AWEI / AWEIsh (Feyisa), HSV colour-space transform (Smith 1978; Pekel et al. 2014/2016); cNDWI, cAWEI (index combos); rNDWI, rAWEI, rHSV (raw source bands)
- **Datasets**: Sen1Floods11
- **Techniques**: binary semantic segmentation; modified (leave-two-countries-out) k-fold CV; flip augmentation; Kruskal–Wallis significance testing; SAR–optical fusion; DEM as ancillary channel
- **Metrics**: F1, Precision, Recall (medians)

## Notable quotes

- "Compared to a median F1 score of 0.62 when using only S1 bands, the combined use of S1 and elevation information led to an improved median F1 score of 0.73. Water extraction indices based on S2 bands have a statistically significant superior performance in comparison to S1." (p. 163, Abstract)
- "Among all the band combinations, HSV (Hue, Saturation, Value) transformation of S2 bands provides a median F1 score of 0.9, outperforming the commonly used water spectral indices owing to HSV's transformation's superior contrast distinguishing abilities." (p. 163, Abstract)
- "Additionally, U-Net algorithm was able to learn the relationship between raw S2 based water extraction indices and their corresponding raw S2 bands, but not for HSV owing to relatively complex computation involved in the latter." (p. 163, Abstract)
- "Since Sen1Floods11 dataset is curated to exclude majority of the satellite imagery with clouds, we could not find the significant advantage in our experiments. However, in presence of clouds, the fusion of S1 and S2 will have an added advantage." (p. 171)
- "By using a testing subset from countries which are not used in training, we avoid spatial autocorrelation which is likely in the context of geo-spatial segmentation." (p. 166)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/VZIDPI7Z/Konapala et al. - 2021 - Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning.pdf`
Cite as: `\cite{konapala_exploring_2021}`
