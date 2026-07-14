---
title: "Improving pluvial flood simulations with a multi-source digital elevation model super-resolution method"
authors: "Yue Zhu, Paolo Burlando, Puay Yok Tan, Christian Geiß, Simone Fatichi"
year: 2025
venue: "Natural Hazards and Earth System Sciences (NHESS), 25, 2271–2286"
doi: "10.5194/nhess-25-2271-2025"
bibtex_key: zhu_improving_2025
zotero_pdf: "/Users/gersongarrido/Zotero/storage/9UVA3U2W/Zhu et al. - 2025 - Improving pluvial flood simulations with a multi-source digital elevation model super-resolution met.pdf"
pages: 16
sensors: [Sentinel-2A, SRTM, TanDEM-X, LiDAR]
task: "DEM super-resolution (30 m → 10 m) + pluvial flood inundation simulation"
model: "RCAN-MS (Residual Channel Attention Network with a multi-source/multi-scale input module)"
tags: [dem-super-resolution, pluvial-flood, cellular-automaton, rcan, channel-attention, sentinel-2, srtm, multi-source-fusion, iou, flood-simulation]
---

# Improving pluvial flood simulations with a multi-source digital elevation model super-resolution method

> **TL;DR** — The authors upscale freely available 30 m SRTM DEMs to 10 m using a deep CNN (RCAN) whose input module fuses the coarse DEM with 10 m four-band Sentinel-2A imagery (B02/B03/B04/B08), then feed the resulting super-resolution DEM into a cellular-automaton pluvial flood model (Caddies) to see whether better terrain yields better flood maps. On two test areas (England; Shenzhen/Hong Kong) the proposed RCAN-MS beats bicubic, SRCNN, VDSR and plain RCAN on every DEM metric — e.g. Dataset 1 MAE 2.1952 m vs 3.0078 m for bicubic, SSIM 0.6205 vs 0.4621 (Table 2, p. 2278). In the flood simulation, RCAN-MS gives the lowest floodwater-depth MAE (0.0247 m in Dataset 1, ~30 % better than bicubic, p. 2279; 0.1193 m in Dataset 2, ~13 % better, p. 2281) and the highest inundation-area IoU at every depth threshold. Key caveat: better DEM metrics do **not** automatically mean better flood maps — plain RCAN was 2nd-best on DEM accuracy but lost to SRCNN/VDSR in flood simulation because of salt-and-pepper noise (p. 2281).

## Problem & Motivation
Pluvial (rainfall-driven) flood simulation needs fine-grained topography, but globally available open DEMs are ≥ 30 m (SRTM), too coarse to resolve the micro-topography that routes surface water in built environments. High-resolution LiDAR DEMs exist only in rich, well-surveyed regions — precisely not in the data-scarce developing world that is most exposed (p. 2271–2272).

The gap the authors attack: most prior DEM super-resolution (SR) work feeds the network **only the low-resolution DEM**, making the task ill-posed — fine detail cannot be invented from coarse elevation alone. A few works added natural-colour aerial imagery; the authors argue **multispectral** imagery (especially NIR) carries land-cover information (vegetation vs bare soil vs urban vs rock) that correlates with elevation structure and can guide reconstruction (p. 2273, p. 2281). Second gap: prior SR-DEM papers evaluate with image metrics (PSNR/SSIM) and never check whether the improved DEM actually improves the downstream **flood simulation** — this paper does both.

## Method / Architecture
**Backbone**: RCAN (Zhang et al., 2018) — residual-in-residual (RIR) blocks with long/short skip connections plus per-block **channel attention** that reweights feature channels. The authors argue channel attention is especially apt for multi-source input, since it can learn to emphasise DEM vs spectral channels adaptively (p. 2273).

**The novelty — multi-source / multi-scale input module** (Fig. 1, p. 2275), inserted *before* the first RCAN layer:
- 30 m LR DEM → 2D conv, kernel 3×3, stride 1, padding 1.
- 10 m four-band multispectral image → 2D conv, kernel 3×3, **stride 3** — downsamples the spatial dims by a factor of 3 so it matches the encoded LR DEM grid; output is a 4-channel tensor.
- ReLU after the convolutions.
- The two equal-sized flows are **concatenated along the channel dimension**, then passed through another 2D conv that fuses spatial + spectral information.
- The fused tensor goes through the RIR/RCAN backbone; a final conv upscales ×3 to the 10 m HR DEM grid.
- The RCAN-MS input layer takes **five bands** (1 DEM + 4 MS); all baselines use single-band in/out (p. 2277).

**Training (identical for all methods, p. 2277)**:
- Framework: **PyTorch**, 2× NVIDIA GeForce RTX 4090 on an HPC cluster (p. 2275).
- Loss: **mean absolute error (MAE)**.
- Optimiser: **Adam**, default momentum parameters.
- Learning rate: **1×10⁻⁴**, adaptive scheduler multiplying LR by **0.8** when validation loss stops decreasing for 50 epochs.
- Batch size: **8**. Epochs: **200**; the epoch with the lowest validation MAE is selected for test evaluation.
- Upscaling factor: **×3** (30 m → 10 m).
- Augmentation: **not reported**. Normalisation scheme: **not reported**. Number of RIR blocks / channels: **not reported** (baselines "used the default parameter settings for hidden layers as specified in their original papers", p. 2275–2277).

**Baselines**: bicubic interpolation, SRCNN, VDSR, RCAN (the backbone, DEM-only input). Ablation is therefore implicit and clean: RCAN vs RCAN-MS isolates the multispectral input.

**Stage 2 — flood simulation** (Fig. 3, p. 2276):
- Model: **Caddies** cellular-automaton 2D pluvial flood model (Guidolin et al., 2016) — not a full shallow-water solver.
- Domain: a **450 × 600 px** subarea cropped from each dataset's exemplary test patch.
- Forcing: **100-year return period, 30 min duration** rainfall — **42 mm h⁻¹** for England (MIDAS IDF, Seathwaite rain gauge) and **190 mm h⁻¹** for Hong Kong SAR (Tang & Cheung, 2011) (p. 2277).
- Each DEM variant (bicubic, SRCNN, VDSR, RCAN, RCAN-MS) is fed to Caddies; the flood map produced with the **reference 10 m HR DEM** is the ground truth.
- Metrics: MAE and MSE on floodwater **depth**; **IoU** on flood **area**, thresholded at 5, 10, 20, 30, 40 cm depth.

## Data
Two datasets, deliberately different geographies (Table 1, p. 2275):

| | Dataset 1 — England, UK | Dataset 2 — Shenzhen & Hong Kong SAR, China |
|---|---|---|
| 10 m HR DEM (target) | LiDAR Composite DTM 2019, UK Environment Agency; **resampled from 2 m to 10 m** (bilinear); acquired 1 Sep 2019 | **TanDEM-X** (DLR, 12 m InSAR DSM); resampled 12 m → 10 m (bilinear); acquired 13 Jan 2016 |
| 30 m LR DEM (input) | **SRTM** (NASA JPL 2013), 1 arcsec ≈ 30 m; 23 Sep 2014 | SRTM, same, 23 Sep 2014 |
| 10 m multispectral (input) | **Sentinel-2A**, bands **B02 blue, B03 green, B04 red, B08 NIR**; 25 Nov 2022, 21 Jan 2023, 13 Feb 2023 | Sentinel-2A, same 4 bands; 25 Dec 2023 |

- **Splits** (Fig. 2, p. 2276): spatially disjoint train/val/test regions ("no spatial overlapping areas between the three subsets", p. 2274). Randomly subsampled into **2000 training / 200 validation / 300 test** patches per dataset.
- **Patch size**: LR DEM patches **80 × 80 px**; HR DEM and multispectral patches **240 × 240 px** (p. 2274).
- **Ground truth**: the 10 m HR DEM itself (LiDAR-derived for UK, TanDEM-X for China) — a *proxy* ground truth, not survey data. Note the HR targets come from **different sensor lineages** in the two datasets (LiDAR DTM vs InSAR DSM); the authors frame this as a robustness test (p. 2274).
- **Preprocessing**: deliberately **none** on the HR DEM — "we did not apply data pre-processing techniques (e.g. noise reduction) on the high-resolution DEM data" (p. 2282–2283). No cloud masking, atmospheric correction or normalisation is described. **No spectral indices are used** — NDVI/NDWI are never computed; the raw four bands go in as channels.
- **Code/data**: openly available at https://doi.org/10.5281/zenodo.15212783 (except TanDEM-X, which needs a DLR proposal). Caddies software from Univ. of Exeter (p. 2283).

## Results

### Stage 1 — DEM super-resolution (Table 2, p. 2278; bold = best)

| Method | D1 MAE (m) | D1 MSE (m²) | D1 PSNR | D1 SSIM | D2 MAE (m) | D2 MSE (m²) | D2 PSNR | D2 SSIM |
|---|---|---|---|---|---|---|---|---|
| Bicubic | 3.0078 | 19.0206 | 33.4055 | 0.4621 | 9.2924 | 163.0170 | 35.4505 | 0.6091 |
| SRCNN | 2.7665 | 15.5027 | 34.2901 | 0.5776 | 6.8153 | 94.1950 | 37.8500 | 0.6794 |
| VDSR | 2.6530 | 13.4866 | 34.8653 | 0.5737 | 6.6412 | 88.7638 | 38.1110 | 0.6811 |
| RCAN | 2.5967 | 12.9453 | 35.0460 | 0.5975 | 6.4150 | 83.5288 | 38.3950 | 0.6838 |
| **RCAN-MS** | **2.1952** | **8.7102** | **36.7605** | **0.6205** | **5.8181** | **66.6251** | **39.3543** | **0.7411** |

RCAN-MS wins on all 8 cells. Reported deltas vs bicubic: Dataset 1 MAE −26.7 %, MSE −54.2 %, PSNR +9.9 %, SSIM +34.8 % (p. 2277).

⚠️ **Inconsistency**: the text (p. 2277) says Dataset 2 MAE decreased "from 9.9 to 5.9 m (−40.4 %)" and MSE "from 186.0 to 67.6 m² (−63.7 %)", but **Table 2 gives bicubic 9.2924 m / 163.0170 m² and RCAN-MS 5.8181 m / 66.6251 m²** (which would be −37.4 % / −59.1 %). The narrative numbers do not match the table. Use the **table** values when citing.

### Stage 2 — pluvial flood simulation
Errors are against the flood map simulated on the reference HR DEM.

**Dataset 1 exemplary patch** (Fig. 8, p. 2281; text p. 2279):

| DEM used | Flood depth MAE (m) | Flood depth MSE (m²) |
|---|---|---|
| Bicubic | 0.0348 | 0.0136 |
| SRCNN | 0.0274 | 0.0108 |
| VDSR | 0.0284 | 0.0123 |
| RCAN | 0.0286 | 0.0123 |
| **RCAN-MS** | **0.0247** | **0.0095** |

"the lowest MAE of 0.0247 m and the lowest MSE of 0.0095 m², scoring an approximately 30 % improvement compared with conventional bicubic methods in both MAE and MSE" (p. 2279). IoU: RCAN-MS is highest at essentially every threshold; at the 5 cm and 10 cm thresholds it improves on bicubic by **146 % and 202 %** respectively (p. 2279). IoU values are low in absolute terms — Fig. 8b shows RCAN-MS at ≈ 0.012 (>5 cm), 0.031 (>10 cm), 0.089 (>20 cm), 0.218 (>30 cm), 0.311 (>40 cm), vs bicubic ≈ 0.0012 / 0.0032 / 0.05 / 0.072 / 0.126.

**Dataset 2 exemplary patch** (Fig. 10, p. 2282; text p. 2281): RCAN-MS flood-depth **MAE 0.1193 m, MSE 0.3009 m²**, "an improvement of approximately 13 % and 15 % in flood depth errors compared with the bicubic-based flood inundation map". IoU (Fig. 10b): RCAN-MS ≈ 0.187 / 0.19 / 0.231 / 0.264 / 0.302 across the 5→40 cm thresholds vs bicubic ≈ 0.135 / 0.144 / 0.157 / 0.178 / 0.207.

⚠️ The **abstract's headline figures** — "a reduction in the mean absolute error of floodwater depth of about 13.1 % and an increase in the intersection over union (IoU) for inundation area predictions of about 46 %" (p. 2271) — are not restated verbatim anywhere in the body. The 13.1 % appears to correspond to the Dataset 2 depth MAE ("approximately 13 %", p. 2281); I could not locate a stated "46 %" IoU figure in the results (the body reports +146 %/+202 % for Dataset 1 and per-threshold bars for Dataset 2). Treat the abstract numbers as an unlocated aggregate and cite the body/table numbers instead.

**The most interesting negative finding** (p. 2281): "better performance in DEM super-resolution methods does not necessarily guarantee an improvement in flood simulation accuracy." Plain RCAN was 2nd-best on DEM metrics but **worse than SRCNN and VDSR in flood simulation**, because it produces salt-and-pepper noise in the DEM that fragments the simulated shallow-water pixels; the over-smoothing of SRCNN/VDSR is, for the CA flood solver, a *benefit*. Adding multispectral input is what lets RCAN-MS get detail *without* the noise.

Also: the improvement from SR is **larger in the flat terrain of Dataset 1** than in the hilly Dataset 2, because hilly terrain concentrates flow into channels that all DEMs recover (p. 2282).

## Limitations
Stated by the authors (p. 2282–2283):
- No preprocessing (e.g. noise reduction) applied to the HR DEM training targets — headroom remains.
- Acquisition dates differ across sources (SRTM 2014, LiDAR 2019, Sentinel-2 2022–2023; TanDEM-X 2016, Sentinel-2 2023) — temporal inconsistency may degrade the learned DEM↔spectral correspondence.
- Trained/evaluated on only **two** geographic areas; transferability without fine-tuning is not guaranteed.
- Only four-band multispectral used as auxiliary input; slope/aspect and other terrain features were **not tested**.
- Only one rainfall scenario (1-in-100-year, 30 min) and one flood model (a cellular automaton, not a full hydrodynamic solver) — sensitivity to the flood model is untested.

Observed by me:
- The flood simulation is run on **a single 450 × 600 px exemplary patch per dataset**, not the whole test set — n = 2. All flood conclusions rest on two patches.
- "Ground truth" flood maps are **model outputs on the HR DEM**, not observed inundation. This measures *fidelity to a simulation*, not real-world flood accuracy.
- HR DEM sources are inconsistent (LiDAR **DTM** = bare earth, vs TanDEM-X **DSM** = includes buildings/canopy). The two datasets' targets mean physically different things, which is not discussed.
- No uncertainty quantification, no repeated seeds/confidence intervals.
- IoU at low depth thresholds is very small in Dataset 1 (0.012 for RCAN-MS at >5 cm); a "+146 % improvement" over 0.0012 is a large relative gain on a near-zero base.

## Relevance to this thesis
**Honest framing: this is not a Sentinel-2 flood-mapping/segmentation paper.** It never segments water in imagery, never uses NDVI or NDWI, and produces no flood labels from satellite data. It is a terrain-enhancement + hydrodynamic-simulation paper. Its relevance is to the **"predict flood-prone areas"** half of the thesis (terrain as a predictor), not the Sentinel-2 inundation-mapping half. Do **not** cite it as a flood-mapping baseline.

What is genuinely borrowable:

- **The exact same four Sentinel-2 bands.** They fuse **B02, B03, B04, B08 at 10 m** — identical to this thesis's band set. This is a direct, citable precedent that these four bands carry usable land-cover/terrain signal beyond water indices.
- **A justification for adding a DEM channel to the CNN input.** If the thesis CNN currently ingests only NDVI/NDWI time series, this paper is the strongest argument in the library for stacking a (super-resolved) **SRTM DEM / slope / HAND** channel: topography is a first-order flood predictor and Peru's terrain (Piura, Rímac, coastal alluvial plains) is exactly the "data-scarce, no LiDAR" setting they target.
- **The multi-source input module is a drop-in pattern for a heterogeneous-resolution stack.** Their trick — conv with **stride 3** on the 10 m data to bring it onto the 30 m grid, then concatenate along channels and fuse with a conv — is a clean PyTorch recipe for mixing 10 m Sentinel-2 with 30 m SRTM without naive resampling. Reusable verbatim if the thesis fuses S2 (10 m) with any coarser layer (SRTM, precipitation, land cover).
- **Channel attention (RCAN / RIR blocks)** as an architectural upgrade over a plain CNN when input channels are heterogeneous (spectral vs elevation vs index). Cheap to try in PyTorch.
- **Training recipe to copy**: Adam, LR 1e-4, batch 8, LR×0.8 on plateau (50-epoch patience), 200 epochs, best-val checkpointing, MAE loss for regression targets (p. 2277). Also: **spatially disjoint** train/val/test regions — this is the correct split discipline for geospatial DL and worth mirroring (and citing) in the thesis methodology.
- **IoU thresholded at multiple depths** (5/10/20/30/40 cm) as an evaluation idea — the thesis reports IoU on a binary flood mask; their multi-threshold sweep shows how sensitive IoU is to how "flooded" is defined, which is a good robustness check and a good caveat to cite when reporting a single IoU.
- **Cautionary example to cite explicitly**: "better performance in DEM super-resolution methods does not necessarily guarantee an improvement in flood simulation accuracy" (p. 2281). Generalise it in the discussion: a pixel-metric win (PSNR/SSIM/MAE, or IoU on a proxy label) can hide artefacts that break the downstream decision. Strong material for a "why we validate on the task, not the intermediate product" paragraph.
- **Data source pointer**: SRTM 30 m (NASA JPL 2013) as the free global DEM; Caddies CA flood model as a lightweight simulator if the thesis ever wants to *generate* physics-based flood labels for pretraining (a plausible answer to Peru's scarcity of labelled flood masks).

Where it differs / does not apply:
- **Regression, not segmentation**: their loss (MAE on elevation) and metrics (PSNR/SSIM) do not transfer to a binary flood-mask CNN.
- **Pluvial (rainfall/urban) flooding**, whereas Peru's ENSO/coastal-Niño events are largely **fluvial + pluvial**; their 100-yr/30-min urban forcing is not the thesis's hazard scenario.
- **No time series** — single-date imagery. The thesis's temporal NDVI/NDWI stack has no analogue here.
- **No real-world validation**: their "ground truth" is a simulation on a LiDAR DEM. Their numbers are **not a baseline the thesis can beat or compare against**.

Verdict: **a method-to-cite and a cautionary example**, plus a concrete architectural pattern for multi-resolution fusion. Not a baseline, not a dataset source for flood labels.

## Keywords / Tech
- **Models**: RCAN, RCAN-MS (proposed), SRCNN, VDSR, bicubic interpolation; residual-in-residual (RIR) blocks; channel attention.
- **Sensors/data**: Sentinel-2A MSI (B02/B03/B04/B08, 10 m), SRTM (30 m, 1 arcsec), TanDEM-X (12 m InSAR DSM), UK Environment Agency LiDAR Composite DTM 2019 (2 m).
- **Indices**: none (no NDVI/NDWI; raw four-band input).
- **Frameworks**: PyTorch; 2× NVIDIA RTX 4090; Caddies cellular-automaton 2D pluvial flood model.
- **Metrics**: MAE, MSE, PSNR, SSIM (DEM); MAE/MSE on flood depth; IoU on flood area at 5/10/20/30/40 cm thresholds.
- **Techniques**: multi-source/multi-scale input fusion, ×3 super-resolution, spatially disjoint splits, adaptive LR scheduling, IDF-curve rainfall forcing.

## Notable quotes
- "Compared to conventional methods (e.g. bicubic interpolation), the simulation results demonstrated that our approach significantly improved the accuracy of flood simulations, with a reduction in the mean absolute error of floodwater depth of about 13.1 % and an increase in the intersection over union (IoU) for inundation area predictions of about 46 %." (p. 2271, abstract — see ⚠️ above, the 46 % is not reproduced in the body)
- "It is important to note that, in principle, better performance in DEM super-resolution methods does not necessarily guarantee an improvement in flood simulation accuracy." (p. 2281)
- "the differentiation between vegetated areas and bare soil in multispectral data can increase the performance of the model in accurately predicting elevation changes and surface contours. The variety of spectral bands helps in distinguishing between features that may have similar elevation profiles but different spectral characteristics, such as the different inter-class variations between urban areas and rocky terrain." (p. 2281)
- "By leveraging publicly available global datasets, this approach offers a promising solution for regions with limited access to high-resolution topographic data." (p. 2283)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/9UVA3U2W/Zhu et al. - 2025 - Improving pluvial flood simulations with a multi-source digital elevation model super-resolution met.pdf`
Cite as: `\cite{zhu_improving_2025}`
