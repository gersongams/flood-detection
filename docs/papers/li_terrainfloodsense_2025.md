---
title: "TerrainFloodSense: Improving seamless flood mapping with cloudy satellite imagery via water occurrence and terrain data fusion"
authors: "Zhiwei Li, Shaofen Xu, Qihao Weng"
year: 2025
venue: "International Journal of Applied Earth Observation and Geoinformation, vol. 144 (2025), article 104855, 13 pp."
doi: "10.1016/j.jag.2025.104855"
bibtex_key: li_terrainfloodsense_2025
zotero_pdf: "/Users/gersongarrido/Zotero/storage/3R2FEVLA/Li et al. - 2025 - TerrainFloodSense Improving seamless flood mapping with cloudy satellite imagery via water occurren.pdf"
pages: 13
sensors: [Landsat-8/9, Sentinel-2, HLS, PlanetScope (validation only), ALOS World 3D DSM]
task: "seamless flood inundation mapping under cloud cover (cloud reconstruction / gap filling of water maps)"
model: "Bayesian fusion of water occurrence + terrain indices (DSM, HAND) + adaptive local thresholding; water extraction by fine-tuned Prithvi-100M-Sen1Floods11 transformer"
tags: [cloud-reconstruction, water-occurrence, hand, dem, bayesian-fusion, hls, extreme-floods, gap-filling, jrc-global-surface-water, seamless-flood-mapping]
---

# TerrainFloodSense: Improving seamless flood mapping with cloudy satellite imagery via water occurrence and terrain data fusion

> **TL;DR** — Optical flood mapping fails exactly when it matters because floods happen under clouds. TerrainFloodSense reconstructs the cloud-covered parts of a water map instead of discarding the image: it fuses the JRC Global Surface Water (GSW) *water occurrence* layer with terrain-derived flood susceptibility (ALOS DSM + HAND) through a Bayes-theorem framework to produce an **Enhanced Water Occurrence (EWO)** prior, then fills cloud gaps by adaptively thresholding EWO under a "submaximal stability assumption". On 72 simulated cloud-reconstruction experiments across three flood events (Assam 2022, Sindh 2022, Rio Grande do Sul 2023) it reaches **94.31 % OA / 0.878 F1 overall** and, in *extreme flooding* areas (GSW occurrence < 5), **89.64 % OA / 0.867 F1** — absolute gains of **2.95 %–8.86 % OA** and **0.038–0.087 F1** over benchmarks (p. 7, Table 1). Code: https://github.com/RCAIG/TerrainFloodSense.

## Problem & Motivation

Optical satellites (Landsat, Sentinel-2, MODIS) are abundant, low-cost and high-revisit, but "their useability is notably constrained by cloud cover" (p. 10) — and flood events are driven by heavy rainfall, i.e. they occur precisely under cloudy/rainy conditions. SAR is weather-independent but has scarce open archives, long revisits (6–12 days for Sentinel-1), speckle noise and poor reliability in urban/vegetated areas (p. 2). Purely hydrodynamic simulation is computationally expensive and depends on terrain/precipitation/land-cover data quality.

The dominant fix in the literature is *cloud reconstruction* using historical **water occurrence** as an auxiliary prior (Zhao & Gao 2018; Mullen et al. 2021), or spatiotemporal interpolation from neighbouring cloud-free dates (Bai et al. 2022; Huang et al. 2023). Both break down for **extreme floods**: water occurrence is "highly biased" (p. 9) because rarely/never-inundated areas carry occurrence ≈ 0, so a WO-based prior cannot predict inundation there; and interpolation fails when water extents change dramatically over short time scales. This paper — an extension of the authors' own Li et al. (2024a) — injects a *physically grounded* terrain prior (elevation + HAND) so cloud-covered pixels that have never been observed flooded can still be reconstructed.

## Method / Architecture

Three stages (Fig. 2, p. 4). All inputs resampled to a common **30 m** grid.

**1) Data pre-processing — terrain-derived inundation index**
- HAND (Eq. 1, p. 4): `HAND = H_pixel − H_drainage` (height of pixel minus height of its nearest drainage cell along the drainage network).
- Inundation index (Eq. 2, p. 4): `Inundationindex = α · norm(DSM) + (1 − α) · norm(HAND)`, where `norm` is min–max normalization.
- `α = 0.3` chosen by sensitivity analysis (tested 0→1 in 0.1 steps), giving **weight 0.7 to HAND** because "it provides more direct information about flood susceptibility" (p. 4). α=0 and α=1 serve as ablations: either indicator alone is suboptimal, so DSM and HAND are complementary.

**2) Bayesian fusion → Enhanced Water Occurrence (EWO)**
- Fused water occurrence `P_fwo = P(wo | terrain) = P(terrain|wo) · P_wo / P_terrain` (Eq. 3, p. 5).
- `P(terrain|wo)` is approximated by `P(terrain | inundation)` estimated by statistical frequency analysis over regions indicated by water occurrence (Eq. 4): `P(terrain = h | inundation) = Σ P_wo(terrain = h) / Σ P_wo`. This is stored as a **precomputed lookup table of terrain-conditioned inundation probabilities** — no full probability distributions, so the fusion is cheap and scalable.
- Simplified result (Eq. 8, p. 5): the FWO under a given terrain condition ≈ the **average water occurrence of all pixels sharing that terrain-index value**.
- Final blend (Eq. 9, p. 5): `P_fwo^final = α·P_fwo + (1 − α)·P_wo` with **α = 0.5** ("neutral balance between model-inferred and observation-based components"). ⚠️ The symbol `α` is reused for two different coefficients (0.3 in Eq. 2, 0.5 in Eq. 9) — potential confusion when reimplementing.
- FWO is mean–variance normalized to [0,1] before the blend.
- **Fusion is applied only where GSW water occurrence < 5** (threshold of 5, following Feng et al. 2023 / Li et al. 2021a / Wang et al. 2024a) — i.e. only in low-confidence, rarely-inundated areas; stable water bodies are left untouched (p. 5).
- **Histogram matching** between FWO and the initial GSW WO keeps statistical distributions aligned and prevents instability from finer terrain-induced variability.
- EWO is **discretized into 500 levels** (vs GSW's original 100) "to allow for a more precise representation of water occurrence" (p. 5).

**3) Cloud reconstruction of the water map**
- Initial water maps are extracted from HLS images with the **fine-tuned Prithvi-100M-Sen1Floods11** model (Jakubik et al. 2023) — a transformer pretrained on HLS with masked autoencoding; used off-the-shelf, not retrained here.
- Cloud/shadow masks come from the **HLS quality band**; masked pixels are the reconstruction targets.
- Reconstruction = binary segmentation of EWO with an **adaptively determined threshold** (Eq. 10, p. 6): `W(x,y) = 1 if P_ewo(x,y) > T, else 0`, where `T` is an optimal threshold computed within a **local sliding window** centred on each cloud-covered pixel; the window is progressively expanded if there are too few valid neighbours; a **global fallback threshold** from the overall ratio distribution is used where no stable local threshold exists (noisy / sparsely observed regions).
- The threshold is derived from a **ratio curve** built from pixel-count distributions of water occurrence across cloud-free vs inundated areas, **smoothed with a moving-window average** to suppress noise from DEM/HAND artifacts.
- **Submaximal stability assumption** (from Li et al. 2024a, three observations, p. 6): (1) inundation under regular conditions usually falls within the range of maximum observed extents; (2) water dynamics occur mainly in areas with low historical occurrence; (3) cloud cover more strongly affects detection confidence of low-occurrence pixels. Operationally: *cloud-covered pixels with water occurrence above the local threshold are assumed inundated; otherwise not* (p. 6).
- **Final product has 3 classes: floodwater, pre-flood water, non-water.** Flood-inundated area = reconstructed water extent minus the pre-flood maximum water extent observed before the event (p. 6).

**Not reported**: no loss function, optimizer, learning rate, epochs, batch size, patch size for training, or augmentation — TerrainFloodSense itself contains **no trained network**; it is a probabilistic fusion + thresholding pipeline. The only deep model (Prithvi) is reused pretrained/fine-tuned by others.

## Data

- **Optical**: Harmonized Landsat and Sentinel-2 (**HLS**, Claverie et al. 2018), from https://hls.gsfc.nasa.gov/ — combines Landsat-8/9 + Sentinel-2, **2–3 day** acquisition interval, **30 m**. Bands used: not reported explicitly (water extraction is via Prithvi on HLS).
- **Water occurrence**: **Global Surface Water (GSW)** dataset (Pekel et al. 2016), https://global-surface-water.appspot.com/download — built from **37 years of Landsat (1984–2021)**; occurrence 0–100.
- **DSM**: **ALOS World 3D – 30 m (AW3D30)** (Tadono et al. 2014), via GEE (`JAXA/ALOS/AW3D30/V3_2`).
- **HAND**: **Global 30 m HAND** dataset (Donchyts et al. 2016), https://gee-community-catalog.org/projects/hand/.
- **Validation only**: **PlanetScope** 3 m imagery (same-day, Rio Grande do Sul, 2023-09-09), plus PlanetScope-derived **NDVI** used to distinguish inundated from non-water surfaces ("Although other indices like NDWI or combined metrics may enhance distinction, NDVI was selected for its simplicity and its strong spectral contrast between vegetation and water in high-resolution imagery", p. 7). ⚠️ Formulas for NDVI/NDWI are **not given** in the paper.
- **Study areas / events** (Fig. 1, p. 3): (1) Assam, India, May–Aug 2022 (monsoon, regularly flooded); (2) Sindh, Pakistan, Aug–Sep 2022 (extreme, record-breaking); (3) Rio Grande do Sul, Brazil, Sep 2023 (extreme, heavy cloud cover, exceeded historical levels in never-flooded areas).
- **Simulated experiments** (p. 6): tiles of **1000 × 1000 px (900 km²)**; **6 HLS images** (flood and non-flood periods) across the 3 sites; reference water maps **manually labelled** with high confidence via visual interpretation of the cloud-free HLS images, assisted by Sentinel-2/PlanetScope high-res imagery, and validated with local flood reports + expert hydrological knowledge. Real cloud masks (from HLS quality band of contaminated images) were resampled/cropped and overlaid: 3 cloud-cover levels — **low (<30 %), medium (30–60 %), high (>60 %)** — × **4 real cloud masks each** → **72 groups of simulated reconstruction experiments** (p. 6). No conventional train/val/test split — the method has no learned parameters (α values were tuned on the same validation labels, p. 4 ⚠️ mild tuning-on-test concern).
- **Large-area application**: full HLS tile, Rio Grande do Sul, 2023-09-29; flood-duration maps from **21 HLS images, Aug 30 – Nov 6, 2023** (Fig. 5, p. 10).
- **Preprocessing**: min–max normalization of DSM/HAND; mean–variance normalization of FWO; histogram matching; 30 m resampling; cloud/shadow masking from HLS QA band.

## Results

### Table 1 (p. 7) — mean over 72 simulated experiments

| Scenario | Method | OA | Precision | Recall | mIoU | F1 |
|---|---|---|---|---|---|---|
| **Overall** | Zhao & Gao (2018) | 89.00 % | 70.39 % | **90.58 %** | 0.656 | 0.792 |
| | Ours with WO (= Li et al. 2024a) | 93.07 % | 83.54 % | 87.24 % | 0.744 | 0.854 |
| | **Ours with EWO (TerrainFloodSense)** | **94.31 %** | **87.21 %** | 88.36 % | **0.782** | **0.878** |
| **Extreme flooding** (GSW WO < 5) | Zhao & Gao (2018) | 80.78 % | 75.86 % | 80.32 % | 0.640 | 0.780 |
| | Ours with WO | 86.69 % | 91.20 % | 75.99 % | 0.708 | 0.829 |
| | **Ours with EWO** | **89.64 %** | **95.42 %** | 79.42 % | **0.765** | **0.867** |
| **Regular inundation** | Zhao & Gao (2018) | 92.88 % | 70.16 % | **99.93 %** | 0.701 | 0.824 |
| | Ours with WO | 95.96 % | 81.81 % | 97.51 % | 0.801 | 0.890 |
| | **Ours with EWO** | **96.44 %** | **84.46 %** | 96.51 % | **0.820** | **0.901** |

- Overall scenario net gains: **+1.24 % OA / +0.024 F1** over Ours-with-WO, and **+5.31 % OA / +0.086 F1** over Zhao & Gao (2018) (p. 7).
- Extreme flooding: **absolute increases of 2.95 %–8.86 % OA and 0.038–0.087 F1** (p. 7 and abstract, p. 1).
- Regular inundation: **+0.48 %–3.57 % OA and +0.011–0.076 F1** (p. 7). ⚠️ Recomputing from Table 1 gives 96.44−92.88 = **3.56** (not 3.57) and 0.901−0.824 = **0.077** (not 0.076) — rounding inconsistency between text and table, immaterial.
- Zhao & Gao (2018) shows the **highest recall in all scenarios** (90.58 %, 80.32 %, 99.93 %) because of its global thresholding strategy, "which tends to over-include uncertain pixels" — at a large precision cost (70.39 % overall vs 87.21 %) (p. 7).
- The authors are candid that overall-scenario performance is dominated by regular inundation because extreme-flood areas are relatively small (p. 7); the extreme-flood row is where EWO pays off.
- **Baselines**: two — Zhao & Gao (2018) (GSW-WO gap filling) and Li et al. (2024a) (the authors' own prior method, = "Ours with WO"). No comparison against SAR-based flood mapping, hydrodynamic models, or a U-Net/CNN trained end-to-end.
- **Large-area application (p. 7–8)**: qualitative only — visual comparison against a same-day 3 m PlanetScope image and its NDVI. "large-area quantitative validation remains challenging due to the absence of consistent reference data" (p. 8). Fig. 5 (p. 10) shows flood-duration maps composed with vs without cloud reconstruction: without it, duration maps show spatial inconsistencies and **underestimation errors**.

## Limitations

Stated by the authors (§6.3, p. 9–10):
1. **EWO quality is capped by DSM quality.** ALOS 30 m DSM has resolution constraints and inherent errors; where elevation detail is poor, EWO fails to indicate inundation probability where satellites did observe water (Fig. 6, p. 11 — Xinxiang, Henan, 2021 flood is shown as a failure case). Mitigation requires higher-resolution/better-quality DSMs.
2. **Only two terrain indices** (DSM, HAND) are fused. The priors capture statistical inundation probability but **not hydrological/hydrodynamic processes** — exceptions such as above-ground rivers or artificial structures violate the "lower elevation + lower HAND ⇒ more flood-prone" assumption and bias EWO.
3. Terrain data uncertainties introduce noise/artifacts into the derived ratio curve (partly mitigated by smoothing).

Observed by me:
- **α = 0.3 and the EWO fusion were tuned against the same validation labels used for quantitative assessment (p. 4)** — no independent held-out tuning set.
- Reference water maps are **manually interpreted**, not independent ground truth; only **6 labelled HLS images** underpin all 72 experiments (the 72 come from resampling cloud masks, not new scenes).
- The large-area application has **no quantitative metrics at all**.
- Recall *decreases* vs Zhao & Gao in every scenario — the method trades recall for precision, which for early-warning/emergency use (where missing flooded areas is costlier than false alarms) may be the wrong trade-off.
- Everything is at **30 m** (HLS/GSW/DSM/HAND alignment), not 10 m.
- The submaximal stability assumption is, by construction, conservative — it can never reconstruct flooding in a cloud-covered pixel whose EWO is below threshold.

## Relevance to this thesis

**This is arguably the highest-leverage paper for the thesis's biggest methodological hole: the `maxcc = 10 %` cloud filter throws away the exact images that contain the flood.** Concretely:

- **Directly borrowable — reframe cloud handling.** Instead of *rejecting* images >10 % cloud, keep them, use the Sentinel-2 **SCL / cloud-probability band** to build a per-pixel cloud+shadow mask, compute NDWI where valid, and *reconstruct* the masked pixels from a water-occurrence prior. This turns a hard image-level filter into a per-pixel gap-fill and can multiply the number of usable flood-day scenes for Peru (a cloudy, Andean/Amazonian country). This is the single most actionable idea here.
- **Directly borrowable — the three ancillary layers**, all free and all available in Google Earth Engine:
  - **JRC Global Surface Water occurrence** (Pekel et al. 2016), `JRC/GSW1_4/GlobalSurfaceWater` band `occurrence`, 30 m, 1984–2021.
  - **ALOS AW3D30 DSM**, `JAXA/ALOS/AW3D30/V3_2`.
  - **Global 30 m HAND** (Donchyts et al. 2016), via the GEE community catalog.
  These can be resampled to the thesis's 10 m Sentinel-2 grid and used as **extra input channels to the PyTorch CNN** — a nearly free accuracy win, since HAND is by far the strongest single static predictor of flood susceptibility (they weight it 0.7 vs DSM's 0.3).
- **Directly borrowable — HAND as a feature/formula**: `HAND = H_pixel − H_drainage` (Eq. 1). And the terrain inundation index `0.3·norm(DSM) + 0.7·norm(HAND)` (Eq. 2) is a one-line, training-free flood-susceptibility layer that could serve as a **naive baseline** for the thesis's susceptibility/prediction component.
- **Directly borrowable — the 3-class output**: *floodwater / pre-flood water / non-water*, obtained by subtracting the pre-event **maximum** water extent from the event-day water extent. The thesis currently thinks in terms of NDWI water masks; this decomposition is what actually distinguishes a *flood* from a river, and it is cheap to implement over an NDWI time series.
- **Directly borrowable — evaluation design**: report metrics **separately for "extreme flooding areas" (GSW occurrence < 5) vs "regular inundation areas"**. The paper shows overall metrics are dominated by permanent water and hide the failure mode that matters. The thesis should adopt this stratification; it is a strong, defensible contribution in itself.
- **Directly borrowable — simulated cloud experiments**: take cloud-free scenes, overlay *real* cloud masks from other dates at low (<30 %), medium (30–60 %), high (>60 %) cover, and measure degradation. A cheap, rigorous ablation the thesis could run to *quantify* how much the 10 %-cloud filter costs.
- **Metrics to report / baseline to beat**: OA, precision, recall, mIoU, F1. If the thesis does any cloud-gap reconstruction, Zhao & Gao (2018) (global WO thresholding) is the standard weak baseline; TerrainFloodSense's numbers (94.31 % OA / 0.878 F1 overall; 89.64 % / 0.867 extreme) are the reference points — though not directly comparable, since study areas differ and resolution is 30 m vs 10 m.
- **Differences from the thesis setup** (be honest about these when citing): 30 m HLS not 10 m Sentinel-2; no trained CNN of their own (the deep model, Prithvi-100M-Sen1Floods11, is borrowed pretrained — itself a **pointer worth chasing**: a HLS-pretrained geospatial foundation model that could replace or pretrain the thesis's CNN); **inundation mapping, not susceptibility prediction**; no time-series learning; and a probabilistic/thresholding pipeline rather than end-to-end DL.
- **Cautionary example**: their failure case (Fig. 6) shows the terrain prior collapsing where the 30 m DSM lacks relief detail. Peru's coastal desert (Piura, Tumbes) and Amazon floodplains are exactly such low-relief settings — the thesis should expect HAND/DSM priors to be weakest where they'd be most needed, and consider a better DEM (e.g. Copernicus GLO-30, FABDEM).
- **Role**: primarily a **method to cite and partially port** (cloud reconstruction + HAND/GSW fusion), plus a **dataset source** (GSW, AW3D30, HAND, HLS), plus an **evaluation-protocol template**. Code is public: https://github.com/RCAIG/TerrainFloodSense.

## Keywords / Tech

- **Models/methods**: Bayesian fusion (Bayes' theorem posterior of water presence given terrain), precomputed terrain→inundation-probability lookup table, histogram matching, adaptive local sliding-window thresholding with global fallback, submaximal stability assumption, Prithvi-100M-Sen1Floods11 (transformer, masked-autoencoder pretrained on HLS).
- **Sensors/data**: HLS (Landsat-8/9 + Sentinel-2, 30 m, 2–3 d revisit), JRC Global Surface Water occurrence (Pekel 2016, 1984–2021), ALOS AW3D30 DSM 30 m, Global 30 m HAND (Donchyts 2016), PlanetScope 3 m (validation).
- **Indices**: HAND `H_pixel − H_drainage`; inundation index `0.3·norm(DSM)+0.7·norm(HAND)`; NDVI (used for visual validation on PlanetScope; formula not given).
- **Metrics**: OA, precision, recall, mIoU, F1-score.
- **Baselines**: Zhao & Gao (2018); Li et al. (2024a).
- **Platform**: Google Earth Engine data catalog links; code on GitHub (RCAIG/TerrainFloodSense).

## Notable quotes

- "While remote sensing can provide strong support for flood monitoring, optical satellite images often face significant challenges due to weather conditions and infrequent revisits, particularly in cloudy and rainy regions." (p. 1, abstract)
- "However, water occurrence data may be highly biased, as extreme floods are rarely observed historically. Moreover, water extents normally change dramatically and significantly during flood periods over short time scales, making accurate temporal interpolation of cloud-covered inundation areas based on temporally adjacent observations challenging and unreliable. As a result, conventional WO-based methods often struggle to recover inundation areas obscured by clouds during extreme flood events." (p. 9)
- "cloud-covered areas with water occurrence values above the determined threshold are assumed to be inundated; otherwise, they are not. This assumption enables the effective reconstruction of potential flooding regions obscured by clouds or lacking valid observations." (p. 6)
- "Cloud cover in optical satellite image time series leads to significant spatial inconsistencies and underestimation errors in the composed flood duration map." (p. 10, Fig. 5 caption)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/3R2FEVLA/Li et al. - 2025 - TerrainFloodSense Improving seamless flood mapping with cloudy satellite imagery via water occurren.pdf`
Cite as: `\cite{li_terrainfloodsense_2025}`
