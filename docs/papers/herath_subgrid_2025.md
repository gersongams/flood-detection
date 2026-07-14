---
title: "Subgrid informed neural networks for high-resolution flood mapping"
authors: "Herath Mudiyanselage Viraj Vidura Herath, Lucy Marshall, Abhishek Saha, Sanka Rasnayaka, Sachith Seneviratne"
year: 2025
venue: "Journal of Hydrology, vol. 660 (2025), article 133329"
doi: "10.1016/j.jhydrol.2025.133329"
bibtex_key: herath_subgrid_2025
zotero_pdf: "/Users/gersongarrido/Zotero/storage/8HKBE8NE/Herath et al. - 2025 - Subgrid informed neural networks for high-resolution flood mapping.pdf"
pages: 16
sensors: [none]
task: "hydrodynamic surrogate / super-resolution (upskilling) of coarse-grid flood depth and inundation extent"
model: "U-Net with attention blocks (SGUnet), ~31M parameters"
tags: [physics-informed-ml, hybrid-model, u-net, super-resolution, hec-ras, subgrid, flood-depth, surrogate-model, csi, australia]
---

# Subgrid informed neural networks for high-resolution flood mapping

> **TL;DR** — The authors propose **SGUnet**, a hybrid theory-guided data-science model that takes the flood-depth raster produced by a *cheap coarse-grid* HEC-RAS subgrid hydrodynamic simulation plus the DEM, and uses an attention-augmented U-Net to correct it into a *fine-grid-quality* flood depth map at DEM resolution. Trained/tested on three Australian catchments (Wollombi, Chowilla, Burnett River) with HEC-RAS subgrid simulations as ground truth, SGUnet cuts the depth RMSE by a factor of **4.5–5.3** relative to the coarse-grid model (abstract, p. 1; §3.2, p. 9), reaches **CSI > 0.9** for flood extent (Table 4, p. 10), and gives a **~50× speed-up** over the fine-grid hydrodynamic model (§3.4, p. 11). It also beats the state-of-the-art LSG hybrid surrogate on the Burnett River case. **No satellite imagery is used anywhere in this paper.**

## Problem & Motivation
High-resolution ("fine-grid") 2D hydrodynamic models solving the shallow water equations are accurate but computationally prohibitive — hours to days per simulation — which makes them impractical for real-time forecasting, ensemble/probabilistic flood design, or large-scale operational modelling (p. 1). Coarse-grid models run in minutes but degrade accuracy. Purely data-driven ML surrogates are fast but are black boxes with poor generalizability (p. 2).

The paper positions itself in the **physics-informed ML / theory-guided data science (TGDS)** family: rather than predicting flooding from raw inputs, it *corrects* a physics-based coarse simulation. The specific novelty is exploiting the **subgrid formulation** (Casulli & Stelling): a coarse computational cell holds a single water-surface elevation but the fine-scale bathymetry inside the cell is retained, so water depth can be reconstructed at DEM resolution by simple subtraction. This means both coarse and fine simulations can be mapped onto the *same* DEM-resolution structured raster — no interpolation/resampling between resolutions is needed, which is the main weakness the authors attribute to prior U-Net super-resolution flood work (p. 3).

## Method / Architecture
**Predictor–corrector formulation:** coarse-grid subgrid hydrodynamic model → water depths at DEM resolution → U-Net corrects them toward fine-grid model output.

Architecture (§2.1.1, p. 3; Fig. 2, p. 4):
- U-Net (Ronneberger et al. 2015) **with attention blocks**, conditioned on external data (DEM).
- Input: `512 × 512` flood-depth raster + `512 × 512` DEM (2 channels), optionally + HAND + flow accumulation (4 channels).
- Encoder: **ten 3×3 conv layers, ReLU**, across 5 feature-map levels (512², 256², 128², 64², 32²), two convs per level; bottleneck **1024 channels**; four **2×2 max-pool** downsamplings.
- Decoder: **eight 3×3 conv layers, ReLU**, across 4 levels (64² → 512²); four **2×2 transposed convolutions**.
- **Skip connections** at every decoder level.
- **Attention blocks** at 256² and 128² in the encoder, and at 256² and 512² in the decoder; each attention block = 3 conv layers.
- Output layer: final conv reducing 64 → 1 channel.
- Total: **31 convolutional layers, ~31 million trainable parameters** (p. 3).
- Implemented in **Python / PyTorch** (p. 3).

Training (§2.1.2, p. 4; §2.3, p. 6; Table 2, p. 8):
- Supervised learning; target = fine-grid HEC-RAS flood depth maps.
- **Loss / objective: MSE** on water depth.
- **Optimizer: Adam** (Kingma & Ba 2014).
- **Learning rate: 0.00002** (all three catchments; "a low learning rate was necessary for stable model training").
- **Batch size: 48** (limited by GPU memory).
- Epochs: **Wollombi 75, Chowilla 15, Burnett River 200** (Table 2, p. 8).
- Hardware: NCI Gadi supercomputer, **Nvidia Tesla V100-SXM2 32 GB** GPUs.
- Evaluated on a test set every 5 epochs to detect overfitting.
- **Data augmentation: not reported** (none described).
- One model is trained **per catchment** (no cross-catchment generalization attempted).

Preprocessing / inference pipeline (§2.1.3, pp. 4–5; Fig. 3, p. 5):
1. Set up coarse-grid HEC-RAS model by coarsening the fine-grid mesh (same BCs, same roughness; **no recalibration**).
2. Run the coarse model (minutes).
3. Parse raw HEC-RAS result files in Python.
4. **Rasterize water-surface elevation (WSE)** from unstructured mesh cells onto the DEM grid (every DEM pixel inside a cell gets that cell's WSE).
5. **Convert WSE → water depth** by subtracting DEM elevation (the subgrid trick, Fig. 4, p. 5). Negative → dry (0).
6. **Threshold: depths < 5 cm set to 0** (Löwe et al. 2021).
7. **Clip** the DEM-resolution depth map into `512 × 512` boxes over the flood-prone area (a polygon shapefile of square boxes).
8. Save as GeoTIFF, depths stored as **integer centimetres** to reduce file size.
9. **Min–max normalization to [0, 1]** for all channels. Flow accumulation gets an **upper cutoff of 10,000** and a **cube-root transform** before scaling.
10. Inference → de-normalize → re-apply the 5 cm threshold. Overlapping clip boxes are merged by **averaging** overlapping pixels (§3.2, p. 9).

## Data
**No remote sensing imagery at all.** All "images" are rasters derived from hydrodynamic simulations and DEMs.

- **Hydrodynamic engine:** HEC-RAS 6.5 (2D, unstructured grids, subgrid technique).
- **Ground truth:** the **fine-grid HEC-RAS simulation output** — i.e., synthetic/model ground truth, not observations. Fine-grid models were *set up* using BoM gauge water levels and discharge, but "no calibration with observed data was performed for the coarse grid model simulations" and no proper calibration was needed since fine-grid output is the target (p. 5).
- **Study areas** (Table 1, p. 8):

| | Wollombi | Chowilla | Burnett River |
|---|---|---|---|
| Model area | 814 km² | 760 km² | 1197 km² |
| DEM resolution | **5 × 5 m** | **5 × 5 m** | **10 × 10 m** |
| DEM source | Geoscience Australia (2021) | Fraehr (2023) | Geoscience Australia (2021) |
| Coarse cells (domain) | 21,715 | 4,622 | 8,704 |
| Fine cells (domain) | 204,862 | 130,621 | 488,080 |
| Coarse cell size | 200×200 m | 400×400 m | 400×400 m |
| Equation set | DWE | DWE | SWE |
| No. of flood events | 4 | 6 | 4 |
| Event duration | several days | weeks to months | days to weeks |
| Mapping interval | 0.5 h | 6.0 h | 0.5 h |

- **Patches:** `512 × 512` pixels. Clip boxes: **21 (Wollombi), 100 (Chowilla), 40 (Burnett)**; clip box size 2.56 km, 2.56 km, 5.12 km respectively (Table 2, p. 8).
- **Samples (train/test):** Wollombi **35,280 / 6,048**; Chowilla **160,200 / 49,400**; Burnett River **83,520 / 15,360** (Table 2, p. 8). Split is by flood *event* (held-out testing event per catchment). **No separate validation set is described** — the "test dataset" is used to monitor overfitting every 5 epochs, so test leakage into model selection is possible ⚠️.
- **Spin-up period** excluded from train/test: 1 day (Wollombi), 12 days (Chowilla), 1 day (Burnett).
- **Spectral indices: none.** The auxiliary "spatial explanatory variables" are **DEM, HAND** (Height Above Nearest Drainage, Nobre et al. 2011) and **flow accumulation**, all derived from the DEM. No formulas given for HAND/flow accumulation beyond verbal description (p. 4).

## Results

**Single vs multiple explanatory variables (Table 3, p. 8)** — MSE in cm²:

| | Wollombi (after 75) | Chowilla (after 15) | Burnett River (after 50) |
|---|---|---|---|
| Training MSE, single (depth+DEM) | 13.44 | 7.51 | 490.30 |
| Training MSE, multiple (+HAND+flowacc) | 12.84 | 7.97 | 500.22 |
| Testing MSE, single | 17.02 | 7.16 | 770.54 |
| Testing MSE, multiple | 17.75 | 6.58 | 788.14 |

Adding HAND + flow accumulation gave **no notable improvement** and increased training time by ~20%; all subsequent results use **DEM only** (p. 8).

⚠️ **Inconsistency:** §3.1 (p. 7) and Table 3 state Burnett River was trained for **50 epochs** in this ablation, while Table 2 (p. 8) and §3.2 (p. 9) give **200 epochs** for the final Burnett model. Presumably the ablation used a shorter run, but the paper never says so explicitly.

**Main performance (Table 4, p. 10)** — coarse grid ("Low-res") vs SGUnet, both against fine-grid target:

| Metric | Wollombi Low-res | Wollombi SGUnet | Chowilla Low-res | Chowilla SGUnet | Burnett Low-res | Burnett SGUnet |
|---|---|---|---|---|---|---|
| AvgRMSE (cm) | 17.8 | **4.0** | 13.3 | **2.5** | 72.1 | **14.8** |
| AvgMAE (cm) | 5.5 | **1.1** | 7.7 | **1.0** | 24.5 | **3.7** |
| POD₅cm | 0.837 | 0.910 | 0.985 | 0.990 | 0.969 | 0.969 |
| POD₃₀cm | 0.900 | 0.961 | 0.988 | 0.995 | 0.971 | 0.965 |
| RFA₅cm | 0.118 | **0.035** | 0.068 | **0.010** | 0.165 | **0.034** |
| RFA₃₀cm | 0.107 | **0.011** | 0.078 | **0.007** | 0.173 | **0.022** |
| CSI₅cm | 0.753 | **0.881** | 0.918 | **0.981** | 0.814 | **0.937** |
| CSI₃₀cm | 0.812 | **0.950** | 0.905 | **0.988** | 0.807 | **0.945** |

- RMSE reduction factor: **4.5 (Wollombi), 5.3 (Chowilla), 4.9 (Burnett)** (p. 9).
- Only metric where SGUnet does not beat the coarse model: **POD₃₀cm on Burnett River** (0.965 vs 0.971, difference 0.006) — the coarse model over-predicts flooding, which inflates its POD (p. 10).
- Depth-difference categories (Fig. 10, p. 13): matched area (−5 to +5 cm) rises from 67.4% → **85.2%** on Wollombi ("17.8% higher", p. 10), from 50% → **98.8%** on Chowilla, and on Burnett the overestimated area drops from **23.4% → 2.2%** (p. 10).

**Baseline vs state-of-the-art (§3.5, pp. 12–13)** — LSG (Fraehr et al. 2023, low-fidelity + sparse Gaussian process) on **Burnett River only**:

| | Coarse grid | LSG | SGUnet |
|---|---|---|---|
| Event-avg RMSE (cm) | 72.1 | 17.0 | **14.8** |
| RMSE in critical 6 h near peak (cm) | 82.6 | 16.9 | **11.4** |
| Event-avg CSI (5 cm) | 0.814 | 0.937 | **0.937** (tie) |
| Event-avg CSI (30 cm) | 0.807 | 0.940 | **0.945** |
| 6 h window CSI (5 cm) | 0.814 | 0.952 | **0.967** |
| 6 h window CSI (30 cm) | 0.807 | 0.949 | **0.971** |

Note: the two models ran on **different hardware** (LSG multi-CPU, SGUnet multi-GPU), so **no direct timing comparison** is provided (p. 13).

**Computational efficiency (§3.4, p. 11)** — Wollombi:
- Coarse-grid HEC-RAS: **5 min 12 s** (Intel i5 1.90 GHz, 16 GB RAM, 12 solver cores).
- Fine-grid HEC-RAS: **7 h 46 min 18 s**.
- SGUnet preprocessing: **2 min 53 s**; inference: **1 min 12 s** (V100 GPU).
- → **~50× speed-up** vs the fine-grid model (excludes one-off training, which took **13 h 25 min** for Wollombi, 75 epochs, 35,280 samples).

## Limitations
Stated by the authors:
- One model **trained per catchment**; no generalized/pretrained model. Future work aims at spatial and temporal extrapolation (§3.4, p. 12; Conclusions, p. 15).
- Speed-up is upper-bounded by the cost of the coarse-grid run (it is a hybrid, not a pure surrogate) (p. 11).
- No DEM smoothing/sink filling and no local mesh refinement of the coarse grid; coarse model was **never calibrated**, so SGUnet's ceiling is set by a possibly poor initial guess (§3.6, p. 14).
- SGUnet operates at DEM resolution; future versions could add more spatial variables (Conclusions, p. 15).

Observed by me:
- ⚠️ **Ground truth is a model, not reality.** SGUnet is trained to mimic a fine-grid HEC-RAS simulation; its errors relative to *observed* inundation are never measured. No satellite/aerial validation of any flood extent.
- Only **3 catchments, all Australian**, and the LSG benchmark is run on **only one** of them (Burnett).
- **No validation split** distinct from the test set (early-stopping decisions made on the test data).
- Test set is a single held-out flood event per catchment.
- No uncertainty quantification.
- Data "available on request" — no public code or dataset release.

## Relevance to this thesis
**Honest verdict: LOW direct relevance.** This is a *hydrodynamic-surrogate / physics-informed ML* paper. It uses **zero satellite imagery**, no Sentinel-1/2, no spectral bands, no NDVI/NDWI, and no observed flood labels. It is not a segmentation-from-imagery paper and it cannot serve as a baseline for a Sentinel-2 CNN. Do not stretch it into the core methods chapter.

Where it *is* legitimately usable:
- **Citation for the "hybrid / physics-informed vs. purely data-driven" taxonomy** in the state-of-the-art chapter. It gives a clean statement of the trade-off you can quote: pure ML surrogates are fast but black-box and overfit; hybrid models start from physics. It also hands you a curated set of secondary citations (Karpatne et al. 2017 TGDS; Zuhairi et al. 2022 review of hybrid flood models; Bentivoglio et al. 2022 DL flood-mapping review — that last one is a genuinely useful review for the thesis).
- **Architectural detail worth borrowing:** the concrete **attention-augmented U-Net** spec (31 conv layers, ~31M params, attention blocks at 256²/128² encoder and 256²/512² decoder, skip connections at every level, 512×512 patches, Adam, LR 2e-5, batch 48, MSE loss). If the thesis CNN evolves from a plain classifier toward a U-Net segmenter, this is a citable, fully-specified configuration.
- **Metric set to adopt: POD, RFA (false-alarm rate), and CSI** — Eqs. 4–6, p. 7 — with the explicit argument (p. 7) that POD alone is gameable (a model that floods everything gets POD = 1) and CSI is the balanced summary. Reporting **CSI alongside IoU/F1** is cheap and makes the thesis comparable to the hydrology literature. Note CSI is *numerically identical to IoU* (TP/(TP+FN+FP)) — worth stating in the thesis that the hydrology community calls it CSI.
- **Threshold discipline:** they evaluate inundation at **two depth thresholds (5 cm and 30 cm)** and show every metric improves as the threshold rises. Analogous caution for the thesis: an NDWI water threshold is a free parameter and reported accuracy is a function of it — report sensitivity to it rather than a single number.
- **HAND + flow accumulation as auxiliary channels — and the negative result.** Their ablation (Table 3, p. 8) says adding HAND and flow accumulation *on top of the DEM* gave no gain and cost 20% more training time, because "the deep network layers ... may be capable of extracting this information from the DEM alone" (p. 8). This is a **useful cautionary result** if the thesis considers stacking terrain covariates onto NDVI/NDWI: it is not automatic that more channels help. Caveat: their DEM is 5–10 m and near-perfect; a Sentinel-2 pipeline typically has *no* DEM channel at all, so HAND may still add a lot in the thesis's setting. Cite it as "evidence is mixed," not as "terrain features don't help."

Where it differs from the thesis (state these explicitly if cited):
- **Regression of continuous water depth** (MSE loss), not binary water/no-water segmentation → their loss function is *not* transferable; a Sentinel-2 flood segmenter should use BCE/Dice/Focal, not MSE.
- **Inundation dynamics for a specific event**, not **susceptibility / flood-prone area prediction** — the thesis's framing (predicting where floods are likely) is closer to the susceptibility literature they cite in passing (Solaimani et al. 2023, 2024; Darvishi 2025).
- **Resolution 5–10 m from LiDAR-grade DEMs**, vs 10 m Sentinel-2 optical — coincidentally similar GSD, but the information content is completely different.
- **Ground truth is simulated**, so their >0.9 CSI is not comparable to a CSI/IoU obtained against real labelled flood extents. Do **not** put their 0.937 CSI in a comparison table next to your Sentinel-2 IoU — it would be an apples-to-oranges comparison and a reviewer will catch it.

Bottom line: **a method-to-cite for framing (hybrid ML vs pure ML) and a source for the POD/RFA/CSI metric triple and a U-Net+attention spec. Not a baseline, not a dataset source.**

## Keywords / Tech
- **Models:** U-Net + attention blocks (SGUnet, ~31M params); LSG (low-fidelity + sparse Gaussian process, comparison model); HEC-RAS 6.5 2D hydrodynamic solver
- **Physics:** shallow water equations (SWE), diffusion wave equations (DWE), **subgrid formulation** (Casulli & Stelling 2011)
- **Sensors:** none (DEM rasters only; DEMs from Geoscience Australia)
- **Indices/covariates:** DEM, **HAND** (Height Above Nearest Drainage), flow accumulation. No spectral indices.
- **Frameworks:** Python, **PyTorch**; NCI Gadi HPC, Nvidia V100 32 GB
- **Metrics:** RMSE, AvgRMSE, AvgMAE, POD, RFA, **CSI** (= IoU)
- **Techniques:** super-resolution / upskilling, theory-guided data science (TGDS), physics-informed ML, min–max normalization, cube-root transform of flow accumulation, 5 cm wet/dry threshold, 512×512 patch clipping with overlap averaging

## Notable quotes
- "SGUnet reduces root mean squared error (MSE) by a factor of 4.5–5.3 compared to coarse-grid models, achieves a critical success index exceeding 0.9 for flood extent mapping, and delivers a 50x speed-up over fine-grid hydrodynamic models." (p. 1) — ⚠️ note the abstract writes "root mean squared error (MSE)", conflating RMSE and MSE.
- "A model predicting everywhere flooded has a perfect POD value but with a high RFA value. Therefore, a model with a high POD and a low RFA is considered as a good prediction model. CSI ... measures the overall accuracy of the model in predicting inundation extents by considering both false alarms and misses." (p. 7)
- "Thus, it would be safe to assume that, when the SGUnet model is trained for each catchment individually, incorporating more explicit hydrological information adds little or no advantage ... The deep network layers of the SGUnet model may be capable of extracting this information from the DEM alone when used as the sole spatial explanatory variable." (p. 8)
- "Although a purely ML-based model might offer greater computational efficiency than the SGUnet hybrid model, the latter provides predictions that start with robust initial estimates based on physics, making it easier to understand the predictions of the model compared to the opaque nature of pure ML models." (p. 14)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/8HKBE8NE/Herath et al. - 2025 - Subgrid informed neural networks for high-resolution flood mapping.pdf`
Cite as: `\cite{herath_subgrid_2025}`
