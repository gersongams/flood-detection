---
title: "Flood Index-Enhanced deep learning model for coastal inundation mapping in SAR imagery"
authors: "Wantai Chen, Yinfei Zhou, Xiaofeng Li"
year: 2025
venue: "International Journal of Applied Earth Observation and Geoinformation, vol. 139, art. 104550, 12 pp."
doi: "10.1016/j.jag.2025.104550"
bibtex_key: chen_flood_2025
zotero_pdf: "/Users/gersongarrido/Zotero/storage/9ZRAJPE2/Chen et al. - 2025 - Flood Index-Enhanced deep learning model for coastal inundation mapping in SAR imagery.pdf"
pages: 12
sensors: [Sentinel-1, SRTM]
task: "coastal inundation mapping (binary semantic segmentation / change detection)"
model: "FIE-Net — dual-branch U-Net with FEPPM, AFNO-SI, CSIM, FSSM modules"
tags: [sar, flood-index, ratio-image, dual-branch-unet, change-detection, inundation-mapping, sentinel-1, ablation, index-as-input]
---

# Flood Index-Enhanced deep learning model for coastal inundation mapping in SAR imagery

> **TL;DR** — Chen et al. propose **FIE-Net**, a dual-branch U-Net for coastal inundation segmentation from **bitemporal, dual-polarized Sentinel-1** imagery. The key idea: compute a **flood index (Ratio Image, RI)** from pre-/post-event SAR intensity, and feed it into the network as a **second, dedicated input branch** alongside a 5-channel "mixed data cube" (4 SAR channels + slope). Four custom modules (FEPPM, AFNO-SI, CSIM, FSSM) fuse the two branches. On 4350 patches from Madagascar cyclones AVA (2018) and Cheneso (2023), FIE-Net reaches **IOU 79.44 %, precision 90.22 %, recall 86.92 %, F1 88.54 %** (p. 8, Table 2), beating U-Net (IOU 77.27 %) and 10 other baselines; ablation shows the **flood index input alone accounts for 67.31 % of the total improvement** (p. 9).

## Problem & Motivation

Coastal inundation from tropical cyclones is a compound hazard (storm surge + riverine flooding + rainfall). SAR is preferred over optical for rapid mapping because it is all-weather / day-night, whereas optical imagery "is frequently incomplete due to cloud cover" (p. 1). Two families of SAR flood methods exist: (a) **single-date thresholding** of post-event water bodies against a reference water map — simple, but confused by mountain shadow, bare soil, and sparse vegetation which also yield low NRCS; and (b) **change-detection** methods comparing pre- and post-event images, which cancel these static confusers but historically rely on hand-tuned thresholds that "lack spatial consistency and struggle with complex, nonlinear inundation mapping tasks" (p. 2).

The gap the authors target: recent deep-learning change-detection models feed raw bitemporal SAR into a network and "rely mainly on backscattering intensity, limiting their flexibility and reliability in unstable conditions" (p. 2). Their thesis is that an explicit, physically-motivated **flood index** computed from the bitemporal pair gives the CNN a "clear, flood-specific signal that enhances segmentation precision" (abstract, p. 1), and that the network architecture should be designed to *fuse* this index stream with the raw-data stream rather than just concatenating channels.

## Method / Architecture

**FIE-Net = (1) input preprocessing module + (2) DCNN-based sub-structure** (Fig. 3, p. 4).

### (1) Input preprocessing — the flood index (the load-bearing part for this thesis)

Three SAR-derived flood indices are listed in Fig. 3 (p. 4), verbatim:

- **DII** (Difference Image Index): `DII = |I_post| − |I_pre|`
- **RI** (Ratio Image): `RI = |I_post| / |I_pre|`  ← **selected**
- **NDFI** (Normalized Difference Flood Index): `NDFI = |I_post − I_pre| / |I_post + I_pre|`

The paper formally defines only the selected one, as **Eq. (1)** (p. 5), verbatim:

> RI = |I_post| / |I_pre|   (1)
>
> "Where the left side of Eq. (1) RI represents the flood index, and the I_post and I_pre is the post-event and pre-event image intensity, respectively." (p. 5)

Note the structural parallel: **NDFI has exactly the NDVI/NDWI normalized-difference form**, but over two *dates* of the same band rather than two bands of the same date. The authors nonetheless **choose RI** (a plain ratio), citing Vanama et al. (2021) and Hamidi et al. (2023); ⚠️ they never report an empirical comparison of RI vs DII vs NDFI in the main text — the choice is justified only by citation ("recognized for its ability to reflect changing areas with sufficient stability", p. 2). Fig. 3 marks RI as "Selected".

**How the index enters the network — the transferable pattern:**
- **Branch A input — "Mixed Data Cube": `(256, 256, 5)`** = 4 SAR channels (pre-event VV, pre-event VH, post-event VV, post-event VH) + **1 slope channel** from SRTM DEM (p. 5, §4.2).
- **Branch B input — "Flood Indices": `(256, 256, 2)`** = RI computed **separately for VV and VH** polarizations (p. 5; Fig. 4 shows RI_VH and RI_VV, p. 4).
- The two cubes go into **two independent U-Net encoder branches**; after independent encoding, the features are concatenated and flow into a **single shared decoder** (p. 5, §4.2).
- Rationale given for the two-branch design instead of a single 7-channel stack: "flood indices can be noisy, leading to potential information loss; incorporating SAR imagery and slope data helps mitigate this issue by providing additional contextual information on flood distribution" (p. 5) — i.e. the index and the raw data have different noise characteristics, so they get separate encoders.

### (2) Four fusion/enhancement modules

- **FEPPM — Frequency Enhanced Pyramid Pooling Module** (Fig. 5, p. 5): applied at *each downsampling stage of the flood-index branch*. A variant of PSPNet's PPM / DAPPM. Large-kernel pooling (kernel sizes 9 and 17) is treated as **low-frequency** information with channel count `C_l`; smaller pooling (global avg, (5,2)) gives `C_o`. Constraint **Eq. (2)**: `C_l/C + C_o/C = 1`. The ratio `C_l/C_o` takes values **(1, 5/3, 3, 7)**, increasing progressively at each of the four downsampling stages — i.e. progressively more low-frequency capacity as depth increases (p. 6). Purpose: noise suppression + better localization.
- **AFNO-SI — Adaptive Fourier Neural Operator based Soft Interaction** (Fig. 6, p. 6): both branches' features go to the frequency domain via **FFT**; two MLPs (weights `w`, biases `b`) map them (Eqs. 3–4); then the learnable params are **frozen** (Eqs. 5–6) and the **two branches swap each other's filters** (cross-filtering), before IFFT back to the spatial domain plus a residual shortcut (Eqs. 7–8). Enables "mutual transfer of key information in the frequency domain" (p. 6). Best results when AFNO-SI is applied **after the first downsampling** (stated; evidence in Supplementary Material, p. 7).
- **CSIM — Cross-branch Semantic Integration Module** (Fig. 7, p. 6): placed at the **end of the encoder**, fuses the two branches via an **Adaptive Feature Fusion (AFF)** path (Eqs. 9–10, Hadamard products + 3×3 dilated conv on the concatenated features) and a **Channel Attention (CA)** path (Eqs. 11–12) that suppresses invalid channels. Removes redundant/overlapping features, keeps complementary ones.
- **FSSM — Frequency Selection Switch Module** (Fig. 8, p. 7): sits on the **skip connections**, so the decoder doesn't blindly concatenate encoder features. Builds a **contrast-aware map** `Ca = σ(Conv_{1×1,d=1}(X_en) − Conv_{3×3,d=2}(X_en))` (Eq. 13), then a switch `F_s = Ca` for high-frequency inflow, `1 − Ca` for low-frequency inflow (Eq. 14). Output re-weighted by **ECA** (Efficient Channel Attention): `X_out = ECA(concat(X_cube_en · F_s, X_FI_en · F'_s, X_in))` (Eq. 15). Policy used: **low-frequency switch for the first two upsampling stages**, **high-frequency for the last two** (detail recovery); the **flood-index branch switch stays low-frequency at all times**, "to prevent unwanted noise" (p. 7).

### Training hyperparameters (p. 8, §4.5.1)

| Item | Value |
|---|---|
| Framework | **PyTorch** |
| Optimizer | **Adam** (Kingma & Ba, 2014) |
| Epochs | **150** |
| Batch size | **16** |
| Initial LR | **0.0001**, reduced by **×0.1 after 100 epochs** |
| Loss | **cross-entropy** |
| Metrics | IOU, Precision, Recall, F1 (formulas in Supplementary) |
| Augmentation | applied **only to the state-of-the-art baselines** "to improve training efficiency" (p. 8) — ⚠️ not stated to be used for FIE-Net itself, and the type of augmentation is **not reported**. This is an odd asymmetry given the claim that "all deep learning models in this study were trained using the same hyper-parameter settings to ensure a fair comparison" (p. 8). |
| Hardware / runtime | NVIDIA **A100 80GB**; inference **0.015 s** per 256×256 patch ≈ **24 s** for a 10000×10000-pixel area; **training ≈ 10.5 h** (p. 10) |

## Data

- **Sensor**: **Sentinel-1A**, Level-1 GRD, Interferometric Wide mode, **VV + VH** dual-polarization, **10 m** (p. 4, §3.1).
- **Terrain**: **SRTM 30 m DEM → slope**, bilinearly interpolated to **10 m** in GEE (p. 4, §3.2). Slope is fed as the 5th channel of the mixed cube. Validity of slope as an input "was experimentally confirmed in the supplementary material" (p. 4).
- **Platform**: **Google Earth Engine** for preprocessing/download (p. 4). ⚠️ **Speckle filtering is never mentioned** — the paper discusses speckle noise as a problem (p. 5) and addresses it *architecturally* (FEPPM) rather than by preprocessing. Radiometric/terrain correction details are not reported beyond "GEE preprocessing".
- **Study area**: **Madagascar** (Fig. 1, p. 2). Two tropical cyclones:
  - **AVA** — landfall NE coast **2018-01-05**, 74 km/h winds. **13 ROIs**.
  - **Cheneso** — NE Sava region **2023-01-19**, winds up to 120 km/h. **3 ROIs**. Together affecting >158,000 people (p. 3).
  - **16 ROIs / 16 SAR pairs total** (Table 1, p. 3). Pre-event dates 2017-12-12 or 2017-12-26 (AVA) and 2023-01-10/15 (Cheneso); post-event 2018-01-07 or 2018-01-11 (AVA) and 2023-01-22/27 (Cheneso). Pre-event images chosen "as close to the event as possible, with low wind speeds" (p. 4).
- **Ground truth**: **Copernicus Emergency Management Service (CEMS) Rapid Mapping** products — Activation Extent Map + Delineation Vector Zip — converted to raster labels; three quality-control steps applied (Fig. 2, p. 3; §3.3 p. 4). Described as **"semi-automatic labeling"** in the abstract (p. 1).
- **Patches**: cropped to **256 × 256** slices. For the AVA ROIs, **slices without inundation were randomly excluded** to balance inundated vs non-inundated counts (Fig. 2, p. 3).
- **Split**: **train 2784 / val 696 / test 870 = 16:4:5** (total **4350** AVA slices) (p. 5). The **689 Cheneso slices are fully held out** — never seen in training — and used as an **independent generalization test** on three unseen flooded areas (p. 5, p. 8).
- **Auxiliary layers for post-processing/analysis**: **JRC Yearly Water Classification History** permanent-water mask (removes false positives from wind-roughened ocean, notably ROI 16) and **ESA WorldCover 10 m Land Cover Type (LCT)** for per-class error analysis (p. 8).
- **Spectral indices**: none — this is SAR. The only indices are the **flood indices DII / RI / NDFI** given above.

## Results

### Comparative experiments — AVA test set (Table 2, p. 8)

| # | Model | Input preproc. | IOU | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1 | Post-threshold (−18 dB VV + JRC mask) | – | 45.29 % | 73.13 % | 54.33 % | 62.34 % |
| 2 | CD-threshold (0.5716 on VV) | – | 52.29 % | 69.22 % | 68.13 % | 68.67 % |
| 3 | Random Forest | RI | 61.16 % | 78.31 % | 73.64 % | 75.90 % |
| 4 | U-Net | RI | 77.27 % | 90.11 % | 84.43 % | 87.18 % |
| 5 | Nested U-Net (UNet++) | RI | *77.33 %* | 89.64 % | *84.92 %* | *87.22 %* |
| 6 | UperNet | RI | 71.86 % | 87.14 % | 80.38 % | 83.62 % |
| 7 | MCANet | RI | 67.61 % | 86.36 % | 75.69 % | 80.67 % |
| 8 | CMGFNet | RI | 71.51 % | 86.80 % | 80.24 % | 83.39 % |
| 9 | DMINet | – | 69.42 % | 86.90 % | 77.53 % | 81.95 % |
| 10 | DSIFN | – | 72.04 % | 84.41 % | 83.09 % | 83.75 % |
| 11 | SNUNet | – | 68.52 % | 87.78 % | 75.74 % | 81.32 % |
| 12 | SwinSUNet | – | 73.62 % | 86.72 % | 82.98 % | 84.81 % |
| 13 | BIT-Former | – | 69.97 % | 84.94 % | 79.88 % | 82.33 % |
| 14 | **FIE-Net (ours)** | **RI** | **79.44 %** | **90.22 %** | **86.92 %** | **88.54 %** |

(*italic* = second best per the paper's underlining.) FIE-Net beats the best baseline (Nested U-Net, 77.33 %) by **+2.11 pp IOU**, and plain U-Net by **+2.17 pp IOU**. Thresholding baselines are 27–34 pp behind. ⚠️ Note baselines 4–8 were themselves **given RI as input** — so the +2 pp is the gain from the *architecture*, not from the index; the index's own contribution is isolated in the ablation.

### Ablation (Table 3, p. 9) — modules removed cumulatively

| Exp. | Input preproc. (RI) | FEPPM | AFNO-SI | CSIM | FSSM | IOU | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ✓ | ✓ | ✓ | ✓ | – | 78.76 % | **90.88 %** | 85.52 % | 88.12 % |
| 2 | ✓ | ✓ | ✓ | – | – | 78.48 % | 90.32 % | 85.68 % | 87.94 % |
| 3 | ✓ | ✓ | – | – | – | 78.05 % | 90.41 % | 85.09 % | 87.67 % |
| 4 | ✓ | – | – | – | – | 77.42 % | 87.24 % | **87.31 %** | 87.27 % |
| 5 | – | – | – | – | – | 73.26 % | 88.62 % | 80.87 % | 84.57 % |
| **Control (full FIE-Net)** | ✓ | ✓ | ✓ | ✓ | ✓ | **79.44 %** | 90.22 % | 86.92 % | **88.54 %** |

**Headline of the ablation (p. 9):** total gain from all optimizations = **79.44 − 73.26 = 6.18 pp IOU**. Removing the **input preprocessing (the RI flood index) alone costs 4.16 pp IOU** (77.42 → 73.26), i.e. **67.31 % of the entire improvement comes from the flood index**. Every architectural module contributes ≤ ~0.7 pp individually.

### Independent generalization — Cheneso event, 689 held-out slices (Fig. 10, p. 10; text p. 9)

| ROI | IOU | Mapped flood area | % of actual flooded area |
|---|---|---|---|
| ROI 14 | **77.90 %** | 27,936 km² | 84.72 % |
| ROI 15 | **80.42 %** | 46,375 km² | 93.93 % |
| ROI 16 | **75.24 %** | 10,239 km² | 87.86 % |

⚠️ **The mapped areas (27,936 / 46,375 / 10,239 "square kilometres", p. 9) are implausible** — Madagascar's whole land area is ~587,000 km², so 46,375 km² for one ROI-sized scene is not credible. These are almost certainly **pixel counts** or km² mislabeled/mis-scaled. Do not cite these area figures.

**Per land-cover-type IOU** (Fig. 10d, p. 10): inundated **cropland** (the dominant flooded LCT) detected at **IOU > 80 %** in all three scenes; **grassland ≈ 70 %** (stated in text; ⚠️ the Fig. 10d bars for grassland read closer to ~20–30 %, a **text/figure inconsistency**); **tree cover ≈ 60 %** (missed observations under dense canopy); **scrubland** poorly identified in ROI 16, attributed to shallow water depth not changing NRCS enough.

## Limitations

**Stated by the authors (§6.2, p. 10):**
- Training takes ~10.5 h on an A100 80 GB → dependency on high-end GPUs. Future work: pruning, knowledge distillation.
- **Detection instability in tree cover and shrubland** — minimal NRCS variation due to vegetation shading.
- **No strategy for high-wind events**: high winds roughen the surface, altering backscatter and causing false positives / missed detections in flood-prone areas. Future: incorporate texture or local wind-speed data.
- **Urban inundation applicability is limited** — double-bounce scattering increases backscatter, the opposite of the learned flood pattern (p. 10).

**Observed by me:**
- **Single country, single sensor, two events.** All 16 ROIs are in Madagascar; both events are tropical cyclones. Transfer to other geographies/flood types is untested.
- **The choice of RI over DII and NDFI is asserted, not measured** — the three formulas are shown in Fig. 3 but no ablation among them appears in the main text.
- **The 870-slice "test set" and the 2784/696/870 split are all drawn from the same 13 AVA ROIs** — spatial autocorrelation between train and test patches is likely. The Cheneso set (689 slices) is the only honest held-out generalization test, and it is *not* an official test set but a separate event. Credit to them for reporting it.
- **Speckle filtering is never described**, despite speckle being cited as the core motivation for FEPPM.
- Augmentation applied to baselines but ambiguous for FIE-Net (see table above).
- **No uncertainty quantification, no confidence intervals, no repeated runs / seeds.** All numbers are single-run point estimates; a 2.11 pp IOU margin over Nested U-Net could be within run-to-run variance.
- **Class balancing by dropping non-flooded patches** inflates all metrics relative to a real operational scene where most patches are dry.

## Relevance to this thesis

**This is a high-relevance methodological paper despite being SAR.** The sensor differs, but the *core pattern* — "compute a domain index, feed it as a dedicated input, let the ablation prove it matters" — is exactly what the thesis does with NDVI/NDWI, and this paper provides both the template and the quantitative justification.

**Directly borrowable:**
- **The two-branch input pattern.** Instead of stacking `[B02,B03,B04,B08,NDVI,NDWI]` into one 6-channel tensor, mirror their design: **Branch A = raw reflectance cube** (B02/B03/B04/B08 + a terrain channel), **Branch B = index cube** (NDVI, NDWI). Two independent encoders, concatenate at the bottleneck, one shared decoder. Their stated rationale — *indices are noisier than raw bands, so they deserve their own encoder plus contextual support from the raw stream* — applies verbatim to NDWI, which is notoriously noisy over wet soil and shadow.
- **Add slope as an input channel.** They interpolate **SRTM 30 m slope to 10 m** in GEE and put it in the data cube; supplementary experiments confirmed it helps. The thesis already works at 10 m — adding a slope channel (SRTM or ALOS PALSAR / Peruvian DEM) is a cheap, evidenced improvement for *flood-prone area* prediction, where topography is arguably more predictive than for pure inundation extent.
- **The bitemporal index framing.** Their **NDFI = |I_post − I_pre| / |I_post + I_pre|** is structurally identical to NDWI/NDVI but applied across *time*. The thesis works on **NDVI/NDWI time series** — so a directly transferable feature is the **temporal normalized difference of an index**, e.g. `ΔNDWI_norm = (NDWI_post − NDWI_pre) / (NDWI_post + NDWI_pre)` or simply the **ratio** `NDWI_post / NDWI_pre` (their RI). This turns the thesis' time series into an explicit change-detection signal rather than leaving the CNN to infer change from stacked frames.
- **Hyperparameters as a starting recipe:** PyTorch, **Adam**, **lr 1e-4** with ×0.1 decay at 100/150 epochs, **batch 16**, **cross-entropy loss**, **256×256 patches**. Directly copyable to the thesis' PyTorch CNN.
- **Class balancing**: randomly drop non-flooded patches to equalize flooded/non-flooded counts. Necessary for the thesis too (Peru scenes will be overwhelmingly dry). But **report it** — it inflates IOU.
- **Post-processing with a permanent-water mask**: they subtract the **JRC Yearly Water Classification History** layer to remove permanent rivers/lakes/ocean from the "flood" prediction. **This is essential for the thesis** — NDWI will fire on the Amazon/Marañón/Rímac channels every single date. JRC GSW is on GEE, free, global, and covers Peru.
- **Ablation design.** Their Table 3 (cumulative module removal, ending with "remove the index") is the exact experiment the thesis needs to justify NDVI/NDWI as inputs: *train once with raw bands only, once with bands+indices, report the ΔIOU.* Chen et al. give the thesis a strong prior that this delta is large (**4.16 pp IOU = 67 % of all gains**) and a citable precedent for the claim "indices as input channels matter more than architecture".

**Directly comparable / baselines to beat:**
- **U-Net (IOU 77.27 %)** and **Nested U-Net / UNet++ (77.33 %)** are the reference points; FIE-Net's 79.44 %. If the thesis reports IOU on a Sentinel-2 flood segmentation task, these are the numbers reviewers will compare against, **but they are on a different sensor and dataset — not a valid head-to-head.** Cite as "reported IOU in the literature", never as a beaten baseline.
- **Metrics to report:** IOU, Precision, Recall, F1 — the four they use. Adopt the same set.
- **Per-land-cover-type error breakdown** using **ESA WorldCover 10 m** (free, on GEE, covers Peru). This is a cheap, high-value analysis for the thesis: report IOU per LCT (cropland / grassland / tree cover / urban). Their finding that cropland is easiest (>80 %) and tree cover hardest (~60 %) gives a hypothesis to test on Peruvian scenes.

**Where it differs from the thesis:**
- **SAR (Sentinel-1) not optical (Sentinel-2)** — no cloud masking, no atmospheric correction, and the confusers are different (radar shadow, double-bounce in cities, wind-roughened water) rather than cloud/shadow/turbid water.
- **Inundation extent mapping (a supervised binary segmentation of "was this pixel flooded on date X")**, not **flood-prone area / susceptibility prediction**. The thesis' framing is predictive/prospective; this paper is retrospective/observational. The label semantics differ.
- **Ground truth from CEMS Rapid Mapping** — a European emergency-mapping product. Peru is covered by CEMS only when an activation is requested; the thesis will likely need **CEMS activations for Peru** (check `emergency.copernicus.eu/mapping/list-of-activations-rapid` — the paper gives this URL, p. 11) or INDECI/ANA/SENAMHI records, or self-labelled NDWI thresholds. **Cite this paper as the precedent for using CEMS RM as ground truth.**
- Their advantage over Sentinel-2 is precisely SAR's all-weather capability, and they say so explicitly (p. 1). **This is a cautionary note the thesis must address head-on**: floods co-occur with clouds, and a max-10 %-cloud Sentinel-2 filter will systematically drop the peak-flood images. This paper is the best citation for that limitation — and an argument for a **future-work S1/S2 fusion**.

**Verdict:** cite as a **method paper** (index-as-input-branch, ablation design, slope channel, JRC masking, CEMS ground truth) and as a **cautionary example** for the optical-vs-SAR cloud problem. Not a dataset source (Madagascar), not a valid numeric baseline.

## Keywords / Tech

- **Models**: FIE-Net (dual-branch U-Net), U-Net, Nested U-Net (UNet++), UperNet, MCANet, CMGFNet, DMINet, DSIFN, SNUNet, SwinSUNet, BIT-Former, Random Forest
- **Modules**: FEPPM (frequency-enhanced pyramid pooling), AFNO-SI (adaptive Fourier neural operator soft interaction, FFT/IFFT cross-filtering), CSIM (cross-branch semantic integration; AFF + channel attention), FSSM (frequency selection switch on skip connections), ECA (efficient channel attention)
- **Sensors**: Sentinel-1A GRD IW, VV+VH, 10 m; SRTM 30 m DEM → slope @ 10 m
- **Indices**: **RI = |I_post|/|I_pre|** (used); DII = |I_post| − |I_pre|; NDFI = |I_post − I_pre| / |I_post + I_pre|
- **Frameworks**: PyTorch, Google Earth Engine, Adam optimizer
- **Datasets / layers**: Copernicus EMS Rapid Mapping (ground truth), JRC Yearly Water Classification History (permanent water), ESA WorldCover 10 m v200 (land cover), IBTrACS (cyclone tracks)
- **Techniques**: bitemporal change detection, semantic segmentation, cross-entropy loss, 256×256 patching, class balancing by patch exclusion, cumulative ablation, cross-event generalization test

## Notable quotes

> "The index offers a clear, flood-specific signal that enhances segmentation precision." (p. 1, abstract)

> "RI = |I_post| / |I_pre| … Where the left side of Eq. (1) RI represents the flood index, and the I_post and I_pre is the post-event and pre-event image intensity, respectively." (p. 5, Eq. 1)

> "Although flood indices can be noisy, leading to potential information loss, incorporating SAR imagery and slope data helps mitigate this issue by providing additional contextual information on flood distribution." (p. 5)

> "Adding input preprocessing is the most effective among the optimizations, achieving a 4.16 % increase in IOU and accounting for 67.31 % of the total improvement. The contribution demonstrates the importance of including flood index information in the model input." (p. 9)

> "SAR operates independently of sunlight, allowing for all-weather, day-and-night imaging, whereas optical sensors rely on sunlight and are limited to clear-sky conditions. … optical imagery is frequently incomplete due to cloud cover." (p. 1)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/9ZRAJPE2/Chen et al. - 2025 - Flood Index-Enhanced deep learning model for coastal inundation mapping in SAR imagery.pdf`
Cite as: `\cite{chen_flood_2025}`
