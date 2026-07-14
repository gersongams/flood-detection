---
title: "DeepFlood for Inundated Vegetation High-Resolution Dataset for Accurate Flood Mapping and Segmentation"
authors: "Mulham Fawakherji, Jeffrey Blay, Matilda Anokye, Leila Hashemi-Beni, Jennifer Dorton"
year: 2025
venue: "Scientific Data (Nature), vol. 12, article 271 (2025), 14 pp. — Data Descriptor"
doi: "10.1038/s41597-025-04554-3"
bibtex_key: fawakherji_deepflood_2025
zotero_pdf: "/Users/gersongarrido/Zotero/storage/R7X74XGL/Fawakherji et al. - 2025 - DeepFlood for Inundated Vegetation High-Resolution Dataset for Accurate Flood Mapping and Segmentati.pdf"
pages: 14
sensors: [UAV-RGB, manned-aircraft-RGB, Sentinel-1, Sentinel-2]
task: "flood inundation mapping / semantic segmentation (4-class, incl. inundated vegetation)"
model: "UNet, UNet++, PSPNet, VNet, AttUNet (benchmarks); ArcGIS pretrained UNet for auto-labeling"
tags: [dataset, flood-mapping, inundated-vegetation, semantic-segmentation, uav, sar, sentinel-1, sentinel-2, multi-modal, north-carolina]
---

# DeepFlood for Inundated Vegetation High-Resolution Dataset for Accurate Flood Mapping and Segmentation

> **TL;DR** — A Nature *Scientific Data* data descriptor introducing **DeepFlood**, a georeferenced multi-modal post-disaster flood dataset from six North Carolina study areas hit by Hurricanes Matthew (2016) and Florence (2018). It combines very-high-resolution manned/UAV RGB orthomosaics (25 cm and 1.5/2.6 cm GSD) with Sentinel-1 SAR (10 m) and Sentinel-2 optical (10 m), plus DEM/slope (1 m), water index and SAR decomposition layers, labeled into **4 classes: Inundated Vegetation, Dry Vegetation, Open Water, Other** (p. 6). It can be tiled into up to **20,593 tiles of 256 × 256 px** (p. 3). Benchmarks of five segmentation architectures reach a best **mIoU 72.4** with UNet on RGB_HR + Sentinel-1 SAR (p. 11); label masks overlap NOAA/Nature-Conservancy reference flood extents by **87%–96%** (p. 9).

## Problem & Motivation
Flood mapping has moved from threshold/change-detection methods on SAR (expert-tuned, poorly transferable) to deep learning, whose success "crucially depends on the availability of high-quality training datasets" (p. 2). Existing flood datasets are inadequate in specific ways the authors enumerate: the European Flood Dataset is not georeferenced; WorldFloods (Sentinel-2, 10 m) suffers from inconsistent cloud labeling and is single-source; Global Flood Database is 250 m (too coarse for localized events); Sen1Floods11 and MMFlood give only **binary** flooded/non-flooded labels; FloodNet (UAV, 1.5 m) is **not geo-referenced**, hindering fusion with other sources (p. 2).

The specific gap DeepFlood targets is **inundated vegetation** — floodwater under a vegetation canopy, invisible to optical sensors, and "one of the most challenging areas for flood mapping" (p. 1). North Carolina's flood-prone areas are largely vegetated, so binary water/no-water labels systematically under-estimate flood extent there. The authors claim DeepFlood is "the first dataset that contains labeling for inundated vegetation for both high-resolution optical and SAR imagery" (p. 3).

## Method / Architecture
This is a **data descriptor**, so "method" is mostly the dataset construction pipeline; the models are benchmarks to demonstrate usability.

**Dataset construction pipeline**
- **Orthophoto generation** from raw aerial RGB frames (geometric correction to a uniform scale) (p. 4–5).
- **SAR preprocessing** (Sentinel-1): VV and VH bands filtered for pre- and post-event; **speckle filtering**; SAR decomposition from SLC images processed in **SNAP** — orbit file application, **radiometric and terrain correction** (Small, 2011), **Refined Lee filter** (Lee, 1981) (p. 5).
- **Sentinel-2 preprocessing**: **cloud masking** only (p. 5). No further atmospheric correction step described.
- **Semi-automatic mask generation (2 steps)** (Fig. 2, p. 5–6):
  1. *Auto mask*: ESRI **ArcGIS Pro high-resolution land-cover classification model** — a **UNet** trained on the 2013/2014 **NAIP** land-cover dataset (Chesapeake Conservancy) — takes 3-band high-res imagery and outputs **9 classes** with **overall accuracy 86.5%** (p. 6). The 9 classes are then **re-mapped to 4**: Wetlands → Inundated Vegetation (class 0); Tree Canopy + Shrubland + Low Vegetation → Dry Vegetation (class 1); Water → Open Water (class 2); Barren + Structures + Impervious Surfaces + Impervious Roads → Other (class 3) (p. 6, Fig. 3).
  2. *Manual correction* by experts in ArcGIS Pro (create/modify/reshape/split tools), an **iterative** process using Sentinel-1 VH/VV, SAR decomposition and a Water Index as auxiliary evidence. The Inundated Vegetation and Dry Vegetation classes were found "prone to misclassification, warranting concentrated effort in their manual annotation" (p. 8). Polygons were then **Dissolve**d and rasterized into a final mask aligned to the optical image at its native pixel resolution; each class also exported as a shapefile (p. 8).

**Benchmark models** (Table 5, p. 11): **UNet, PSPNet, VNet, UNet++, AttUNet**, each run on 4 input configurations: `RGB_HR` (high-res aerial RGB), `RGB_HR + SAR_S1`, `RGB_S2` (Sentinel-2 RGB), `RGB_S2 + SAR_S1`.

⚠️ **Training details are essentially absent**: loss function, optimizer, learning rate, epochs, batch size, augmentation, and the **train/val/test split** are all **not reported**. Framework (PyTorch/TF) is **not reported**. This is a significant reproducibility gap for a benchmark table.

## Data
**Sensors, platforms, resolution** (Table 2, p. 4):

| Event | Platform | Date | Region | GSD |
|---|---|---|---|---|
| Hurricane Matthew | Manned aircraft (NOAA, Trimble DSS) | Oct 2016 | Grifton & Kinston, NC | 25 cm |
| Hurricane Matthew | UAV (Trimble UX5, NCEM) | Oct 2016 | Princeville, NC | 1.5 cm (⚠️ text on p. 4 says **2.6 cm**, 3-band RGB, 10,816 m² coverage) |
| Hurricane Florence | Manned aircraft (NOAA NGS, King Air) | Sep 2018 | Elizabethtown & Washington, NC | 25 cm |
| Hurricane Florence | UAV (DJI M600) | Sep 2018 | Lumberton, NC | 1.5 cm (1,159 m² per image) |

⚠️ **Inconsistency**: Table 2 lists the Princeville UAV imagery at **1.5 cm**, while the Data-collection text (p. 4) states the Trimble UX5 imagery had "a remarkable spatial resolution of 2.6 cm". Table 1 lists DeepFlood's optical resolution as "1.5/2.6 cm/25 cm", consistent with the text. All rows are marked **"Openly Available"**.

- **Image counts per area** (p. 4): Grifton 14 images, Kinston 28 images, Elizabethtown 90 tiles, Washington 48 tiles. Princeville and Lumberton counts **not reported** individually.
- **Satellite companions**: **Sentinel-2** optical (10 m) and **Sentinel-1 GRD** SAR (10 m) retrieved via **Google Earth Engine** for the same events/areas; **Sentinel-1A SLC**, **VV+VH**, **Interferometric Wide swath**, 10 m, from the **Alaska Satellite Facility**, used to build SAR decomposition images (p. 4). Strict temporal rule: Sentinel-1 and coincident Sentinel-2 must be **same day or within a 2-day window** of the aerial acquisition (p. 4).
- **DEM**: USGS **3DEP Lidar Explorer**, **1 m**; **Slope** derived from DEM, also 1 m (p. 8–9).
- **Tiling**: "it is possible to create up to **20,593 tiles** with a tile size of **256 × 256 pixels**" (p. 3); Table 1 lists DeepFlood as 20,593 images, 4 classes.
- **Classes (4)** (p. 6): Open Water; Dry Vegetation; Inundated Vegetation; Other. Mask encoding in the GeoTIFF masks: **0 = inundated vegetation, 1 = dry vegetation, 2 = open water, 3 = other** (p. 8).
- **Class balance**: **not reported as pixel/class percentages.** The paper instead gives an **NLCD-based land-cover breakdown of the study areas** (Fig. 8a, p. 10): Woody Wetlands 35%, Cultivated Crops 22%, Open Water 16%, Developed/Built-up 14%, Forest 13%. Per-area (p. 10): woody wetlands ~55% in Elizabethtown, ~18% in Washington; open water ~45% in Washington; cultivated crops ~36% Grifton, ~25% Kinston; built-up ~20% Kinston; forest ~21% Elizabethtown, ~8% Washington. ⚠️ This is NLCD land cover, **not** the DeepFlood label distribution — do not cite it as class balance.
- **Spectral indices**: a **"Water Index"** raster is shipped in the SENTINEL_1 folder and used as an evidence layer during manual correction (p. 8) — **the paper never gives its formula** and never names NDWI/NDVI. ⚠️ No index formulas are provided anywhere in the paper.

**How to obtain it** (p. 8): openly downloadable from **figshare**, DOI **10.6084/m9.figshare.26791243** (ref. 34). Naming convention: `TrainingAreaName_HurricaneName_Day(00)_Month(00)_Year(00)_TileNum(00)`. Folder structure (Fig. 6): `OPTICAL`, `GEOTIFF_MASK`, `SENTINEL_1` (SAR decomposition, SAR_VH, SAR_VV, Water Index), `SENTINEL_2`, `DEM`, `SLOPE`, `SHAPEFILES` (Boundary / Final_label / Individual_labels), `FLOOD_EXTENT`, and `TILES` (subfolders: Optical, Mask, SAR VV, SAR VH).

**Licensing**: the *article* is **CC BY-NC-ND 4.0** (p. 14) — non-commercial, **no derivatives**. ⚠️ The **figshare dataset's own license is not stated in the paper**; you must check the figshare landing page before using it. Do not assume the article license applies to the data. Code: the initial-mask code is the "ArcGIS open-source high-resolution land cover classification deep neural network model"; instructions at the Geospatial and Remote Sensing Research Lab (p. 12).

## Results

**Technical validation — label quality vs. reference flood extents** (Table 4, p. 10). Labels were collapsed to binary (flooded = inundated vegetation + open water; non-flooded = dry vegetation + other) and overlaid on 2018 reference flood-extent maps from The Nature Conservancy / ASU Center for Biodiversity Outcomes. Overlay percentage coverage:

| Area | Flooded area quality | Non-flooded area quality |
|---|---|---|
| Grifton 1 | 93.98% | 98.11% |
| Grifton 2 | 96.86% | 97.73% |
| Kinston 1 | 89.05% | 89.35% |
| Kinston 2 | 91.95% | 85.96% |
| Kinston 3 | 99.24% | 85.66% |
| Elizabeth 1 | 89.20% | 89.00% |
| Elizabeth 2 | 95.39% | 90.00% |
| Washington 1 | 87.69% | 88.50% |
| Princeville | 95.34% | 89.30% |
| Lumberton | 91.91% | 93.34% |

Text summary: "the overlay percentage between the two datasets for the various study areas ranged between **87% and 96%**" (p. 9). ⚠️ This range is inconsistent with the table, which contains **99.24%** (Kinston 3 flooded) and **98.11%/97.73%** (Grifton non-flooded) — above the stated 96% ceiling, and 85.66% below the 87% floor.

**Auto-labeler (ArcGIS pretrained UNet) per-class metrics**, 9 NAIP classes, overall accuracy 86.5% (Table 3, p. 6): Water P 0.94 / R 0.93 / F1 0.93; Wetlands 0.82/0.76/**0.79**; Tree Canopy 0.90/0.93/0.92; Shrubland 0.52/0.19/**0.27**; Low Vegetation 0.86/0.87/0.86; Barren 0.67/0.51/0.58; Structures 0.81/0.85/0.83; Impervious Surfaces 0.74/0.69/0.71; Impervious Roads 0.76/0.81/0.79. Note the **Wetlands** class (→ Inundated Vegetation) is the weak one among vegetation/water, which is exactly why manual correction was needed.

**Segmentation benchmarks** (Table 5, p. 11). Metrics: mIoU, per-class IoU / Precision / Recall for IV (Inundated Vegetation), DV (Dry Vegetation), OW (Open Water), Oth (Other). Selected rows (mIoU and per-class IoU):

| Model | Input | mIoU | IoU IV | IoU DV | IoU OW | IoU Oth |
|---|---|---|---|---|---|---|
| UNet | RGB_HR | 68.3 | 71.7 | 61.2 | 61.8 | 78.6 |
| **UNet** | **RGB_HR + SAR_S1** | **72.4** | 77.7 | 67.8 | 66.2 | 77.5 |
| UNet | RGB_S2 | 48.6 | 47.8 | 52.4 | 39.1 | 47.6 |
| UNet | RGB_S2 + SAR_S1 | 63.8 | 68.4 | 60.8 | 55.2 | 70.7 |
| PSPNet | RGB_HR | 56 | 56.3 | 55.2 | 43.6 | 68.9 |
| PSPNet | RGB_HR + SAR_S1 | 58.9 | 62.6 | 58.3 | 45.1 | 69.8 |
| PSPNet | RGB_S2 | 51.4 | 59.6 | 53 | 41.4 | 51.6 |
| PSPNet | RGB_S2 + SAR_S1 | 58.1 | 61.8 | 57.6 | 51 | 62.1 |
| VNet | RGB_HR | 59.9 | 64.4 | 57.9 | 49.3 | 68.1 |
| VNet | RGB_HR + SAR_S1 | 61.1 | 63.4 | 59.1 | 50.5 | 71.4 |
| VNet | RGB_S2 | 43.2 | 43.6 | 45.8 | 39 | 47.5 |
| VNet | RGB_S2 + SAR_S1 | 53.3 | 54.7 | 52 | 46.9 | 59.4 |
| UNet++ | RGB_HR | 61.3 | 65 | 58.6 | 53.2 | 68.3 |
| UNet++ | RGB_HR + SAR_S1 | 61.1 | 71.7 | 61.1 | 44.4 | 67.5 |
| UNet++ | RGB_S2 | 48.6 | 47.8 | 52.4 | 39.1 | 47.6 |
| UNet++ | RGB_S2 + SAR_S1 | 58 | 64.2 | 55.4 | 51.2 | 62.3 |
| AttUNet | RGB_HR | 71.8 | 79.1 | 67.5 | 61.4 | 79 |
| **AttUNet** | **RGB_HR + SAR_S1** | **72.2** | **80** | 69.1 | 61.4 | 78.1 |
| AttUNet | RGB_S2 | 48.7 | 49.1 | 50.1 | 42.1 | 54 |
| AttUNet | RGB_S2 + SAR_S1 | 63.8 | 70.2 | 60.1 | 55.7 | 69.3 |

Headlines stated in text (p. 10): mIoU ranges "**from approximately 43.2 to 72.4**"; the best values come from UNet/UNet++ **with SAR added**; "IoU values for 'IV' range from approximately 43.6 to 79.1"; the **Open Water class is the weakest**, IoU "approximately 39.0 to 66.9". Precision for UNet/UNet++ ranges ~75.8–93.9, recall ~73.0–93.6 (p. 10). ⚠️ Two textual claims conflict with Table 5: (a) the text says "UNet and UNet++ consistently exhibit superior segmentation accuracy compared to PSPNet, VNet, and AttUNet" (p. 10), but **AttUNet actually beats UNet++ on every input configuration** and is second-best overall (72.2 vs UNet's 72.4); (b) the top OW IoU in Table 5 is **66.2** (UNet RGB_HR+SAR), not 66.9. The exact "highest IoU for IV = 79.1" corresponds to **AttUNet RGB_HR** — but AttUNet RGB_HR+SAR reaches **80**, so the stated 79.1 ceiling is also slightly off.

**Baselines**: no *external* dataset/method baseline was run. The "baselines" are the five architectures compared against each other on DeepFlood. There is **no cross-dataset transfer experiment** (e.g. train on DeepFlood → test on Sen1Floods11).

## Limitations
Stated by the authors (p. 12):
- **Resolution mismatch** between Sentinel-1 SAR (10 m) and the RGB imagery (25 cm) makes alignment/integration of the multi-modal sources non-trivial.
- **Mixed-class pixels**, especially transitions between dry and inundated vegetation, remain a source of error.
- **No separate evaluation** by landscape coverage type or flood severity level — "detailed information on these factors have not been available to the study."

Observed by me:
- **No train/val/test split, loss, optimizer, epochs, batch size, or augmentation reported** — Table 5 is not reproducible as published, and it is not stated whether the split is random-tile (which would leak spatially adjacent tiles) or by study area.
- **Single region (North Carolina, USA), two hurricanes, six sites** — coastal-plain temperate landscape. No claim of geographic generality.
- **Labels are semi-automatic**, seeded by a UNet whose "Wetlands" class (→ Inundated Vegetation, the headline class) has F1 = 0.79 and OA 86.5% (p. 6). Manual correction mitigates but doesn't eliminate label bias toward what that model finds.
- **Post-event only**: single-date post-disaster snapshots. It is **not** a time-series dataset, so it cannot train temporal models.
- **No uncertainty quantification**, no confidence intervals, no repeated runs on the benchmarks.
- Numeric inconsistencies flagged above (⚠️ overlay range; UAV GSD 1.5 vs 2.6 cm; text-vs-table IoU claims).

## Relevance to this thesis
Honest verdict: **useful as a citable dataset/benchmark reference and as a labeling-methodology template — but NOT a good training source for the thesis as configured.** The thesis is Sentinel-2 10 m, NDVI/NDWI time series, Peru. DeepFlood's value is centred on cm-scale aerial imagery over North Carolina.

**What can be borrowed (actionable)**
- **The 4-class scheme itself: Open Water / Inundated Vegetation / Dry Vegetation / Other** (p. 6). This is the single most transferable idea. A pure NDWI-threshold or binary water/no-water labeling in a vegetated Peruvian basin (Amazonian/Piura floodplains) will **systematically under-count flood extent** because water under canopy is optically invisible. Adopting an "inundated vegetation" class — or at minimum acknowledging it as a known failure mode of NDWI — is a direct, defensible design decision to cite.
- **The semi-automatic labeling pipeline** (pretrained model → auto mask → expert manual correction in GIS, Fig. 2, p. 5): a practical, cheap way to generate flood masks for Peruvian AOIs where no labeled ground truth exists. Cite as `\cite{fawakherji_deepflood_2025}` when justifying a semi-automatic annotation approach.
- **The validation protocol**: collapse multi-class labels to binary flooded/non-flooded and compute **overlay percentage against an independent official flood-extent map** (Table 4, p. 10). Peru has an equivalent: INDECI / CENEPRED / ANA flood-extent products and Copernicus EMS rapid-mapping activations. This gives the thesis a concrete, publishable label-quality validation number.
- **Architecture shortlist**: UNet and **AttUNet** are the two strong performers (mIoU 72.4 and 72.2, p. 11); PSPNet and VNet lag by 10-16 mIoU points. If the thesis' PyTorch CNN needs a stronger segmentation baseline than a plain CNN, **UNet / Attention-UNet is the evidence-backed choice** here.
- **Metric set to report**: mIoU + per-class IoU + precision + recall. Match this so results are comparable.

**Directly comparable numbers**
- The **`RGB_S2` rows in Table 5 are the closest analogue to the thesis setup** (Sentinel-2, 10 m): UNet mIoU **48.6**, PSPNet **51.4**, VNet **43.2**, UNet++ **48.6**, AttUNet **48.7** (p. 11). These are a legitimate, citable **reference point for what 10 m Sentinel-2 RGB alone can achieve on 4-class flood segmentation** — roughly 20 mIoU below the cm-scale aerial imagery. Useful for setting realistic expectations in the thesis and for arguing why extra inputs (indices, time series, DEM) are needed.
- **Adding Sentinel-1 SAR to Sentinel-2 RGB lifts mIoU by ~10-15 points** in every architecture (e.g. UNet 48.6 → 63.8; AttUNet 48.7 → 63.8, p. 11). This is a strong, quantified argument for a **future-work / discussion** section: if the thesis is optical-only, this paper documents the specific cost of that choice. Sentinel-1 is available on the same Copernicus Data Space the thesis already uses — a low-friction extension.
- The **Open Water class is the weakest class** across the board (IoU ~39-66, p. 10). Worth citing: it counters the naive assumption that open water is "the easy class".

**Where it differs / cautions**
- **Not a Sentinel-2 dataset.** The Sentinel-2 layer exists but is a *companion* to the aerial imagery; masks are drawn on cm-scale orthophotos and are far finer than a 10 m Sentinel-2 pixel. Naively resampling those masks to 10 m for training would produce very heavy mixed-pixel label noise.
- **No NDVI/NDWI.** The paper ships an unspecified "Water Index" and never gives any index formula. Nothing to borrow directly on the index front.
- **Not a time-series dataset** — no support for the thesis' temporal-CNN framing.
- **Inundation mapping, not susceptibility/prediction.** DeepFlood maps observed post-event extent; the thesis' "predict flood-prone areas" goal is a different task.
- **Geography**: humid subtropical US coastal plain, hurricane-driven. Peruvian El Niño / Andean-runoff floods differ in land cover and event dynamics; direct transfer of a DeepFlood-trained model is unjustified.
- **License risk**: the article is **CC BY-NC-ND**; the figshare data license is **unstated in the paper** (p. 8, p. 14). ⚠️ **Verify on figshare (10.6084/m9.figshare.26791243) before using or redistributing any tiles.** "NoDerivatives" on the article text is a warning sign to check whether derived/fine-tuned artifacts are permitted.

**Classification**: primarily a **dataset source (with caveats) + method to cite for labeling/class design + a comparable Sentinel-2 baseline number**. Not a baseline to beat, and not a cautionary example.

## Keywords / Tech
- **Models**: UNet, UNet++, PSPNet, VNet, Attention-UNet (AttUNet); ArcGIS Pro pretrained UNet land-cover model (NAIP 2013/2014, Chesapeake Conservancy)
- **Sensors**: Trimble UX5 fixed-wing UAV, DJI M600 UAV, NOAA manned aircraft (Trimble DSS, King Air), Sentinel-1 (GRD + SLC, IW, VV+VH), Sentinel-2, USGS 3DEP lidar DEM
- **Indices/derived layers**: unnamed "Water Index", SAR decomposition, slope, flood extent
- **Techniques**: orthomosaic generation, speckle filtering (Refined Lee), radiometric + terrain correction, cloud masking, semi-automatic mask generation, class aggregation (9→4), polygon dissolve, overlay accuracy assessment
- **Tools**: ESRI ArcGIS Pro, ESA SNAP, Google Earth Engine, Alaska Satellite Facility (ASF), figshare
- **Related datasets referenced**: WorldFloods, European Flood Dataset, Sen1Floods11, MMFlood, FloodNet, GF-FloodNet, SEN12-FLOOD, Global Flood Database, HISEA-1 SAR, NLCD
- **Metrics**: mIoU, per-class IoU, precision, recall, F1, overall accuracy, overlay percentage

## Notable quotes
> "DeepFlood is introduced to address the essential requirement for high-quality training datasets. This is a novel dataset comprising high-resolution manned and unmanned aerial imagery and Synthetic Aperture Radar (SAR) imagery, enriched with detailed labels including inundated vegetation, one of the most challenging areas for flood mapping." (p. 1)

> "Additionally, it is very challenging to detect floods underneath the dense vegetation canopies (inundated vegetation) due to the limitations of optical sensors. since these flooded areas are simply not visible on the imagery." (p. 3)

> "To our knowledge, this is the first dataset that contains labeling for inundated vegetation for both high-resolution optical and SAR imagery." (p. 3)

> "The dataset's efficacy is evident from the achieved mean Intersection over Union (mIoU) values, ranging from approximately 43.2 to 72.4. The highest mIoU values are consistently observed with UNet and UNet++ models incorporating SAR data, indicating the dataset's suitability for multi-modal data fusion tasks." (p. 10)

> "it is important to consider the discrepancy in spatial resolution between Sentinel-1 SAR images (10 meters) and RGB images (25 cm per pixel) when using the dataset, especially since the alignment and integration of these multi-modal and multi-resolution data sources are required." (p. 12)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/R7X74XGL/Fawakherji et al. - 2025 - DeepFlood for Inundated Vegetation High-Resolution Dataset for Accurate Flood Mapping and Segmentati.pdf`
Data: figshare DOI `10.6084/m9.figshare.26791243`
Cite as: `\cite{fawakherji_deepflood_2025}`
