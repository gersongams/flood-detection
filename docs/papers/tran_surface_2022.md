---
title: "Surface Water Mapping and Flood Monitoring in the Mekong Delta Using Sentinel-1 SAR Time Series and Otsu Threshold"
authors: "Khuong H. Tran, Massimo Menenti, Li Jia"
year: 2022
venue: "Remote Sensing (MDPI), vol. 14, no. 22, article 5721, 30 pp."
doi: "10.3390/rs14225721"
bibtex_key: tran_surface_2022
zotero_pdf: "/Users/gersongarrido/Zotero/storage/NQ72ZIE5/Tran et al. - 2022 - Surface Water Mapping and Flood Monitoring in the Mekong Delta Using Sentinel-1 SAR Time Series and.pdf"
pages: 30
sensors: [Sentinel-1, Sentinel-2]
task: "surface water mapping + near-real-time flood inundation monitoring (time series)"
model: "Dynamic (per-scene) Otsu thresholding + rule-based change-detection time series — NO machine learning, NO deep learning"
tags: [otsu, thresholding, sentinel-1, sar, ndwi, mndwi, flood-monitoring, time-series, change-detection, mekong-delta, baseline]
---

# Surface Water Mapping and Flood Monitoring in the Mekong Delta Using Sentinel-1 SAR Time Series and Otsu Threshold

> **TL;DR** — Tran et al. build a fully automatic, unsupervised pipeline for the Vietnamese Mekong Delta (An Giang province, ~3500 km²): they apply a **dynamic (per-image) Otsu threshold** to a time series of 128 Sentinel-1 GRDH **VH** images (Mar 2017–Mar 2018), producing 64 surface water maps, then chain them with a **rule-based change-detection** step to produce 56 near-real-time flood maps. Because no ground truth exists in the flooded delta, validation is *indirect*: Sentinel-1 water maps are compared against Sentinel-2 **MNDWI + Otsu** water maps, giving **R² = 0.97, RMSE = 1.18%** in river areas and **R² = 0.88, RMSE = 3.88%** in paddy fields (p. 19, p. 25). Flooded area peaked at **36% of An Giang on 8 September 2017** (p. 21). ⚠️ **No overall accuracy, F1, IoU or kappa is reported anywhere in the paper** — the only quantitative metrics are R² and RMSE on *water-percentage per subset*, not per-pixel.

## Problem & Motivation

The Vietnamese Mekong Delta (VMD) is the third-largest delta in the world, ~2 million ha of flood-prone land, ~0.8 m mean elevation, producing ~half of Vietnam's rice (p. 2). Annual monsoon floods (Aug–Nov) and increasingly erratic hydrology (upstream dams, subsidence, sea-level rise) cause major losses; the 2000 flood killed 539 people and caused ~200 M USD damage (p. 2). Optical imagery is largely unusable there: **"85% to 95% of the VMD is covered by a persistent cloud during the wet season"** (p. 3), which motivates SAR.

The specific gap: the few existing Sentinel-1 studies in the VMD apply a **single empirical/static backscatter threshold** for all scenes of a year (p. 4). But 80% of the area is paddy field with up to three rice cycles per year, so VH backscatter varies strongly with crop phenology — the minimum VH is ≈ **−20 dB in the summer-autumn season vs ≈ −24 dB in the winter-spring season** (p. 11, p. 22). A static threshold is therefore wrong for most of the year. The authors propose to (i) recompute an Otsu threshold *per acquisition*, and (ii) chain the resulting water maps in time to derive flood (as opposed to permanent-water) maps in NRT.

## Method / Architecture

**No neural network, no supervised learning at all.** The authors argue explicitly that supervised methods are "computationally costly, hard to employ for large scales, and require ground truth samples" and are thus "less effective for applications that demand swift action and in near real-time (NRT)" (pp. 3–4). This makes the paper a *classical, unsupervised, ground-truth-free* baseline.

**Pipeline (Fig. 3, p. 8):**

1. **Polarization selection (Sec. 3.1.1, p. 9).** Only **VH** is used. Rationale: VV suffers a strong double-bounce increase from the water–vegetation interaction in flooded rice; VH is much less affected. Figure 4 shows VV rising sharply in the nursery stage (~5 cm water) while VH does not.
2. **Sentinel-1 pre-processing in SNAP (Fig. 5, p. 10)**, in order:
   - Apply-Orbit-File (2 images per scene, since 2 images are needed to cover An Giang)
   - Thermal Noise Removal
   - Border Noise Removal
   - Slice Assembly (mosaic of the 2 images)
   - Radiometric Calibration (to sigma naught)
   - **Speckle filtering: Refined Lee** — chosen because it "preserve[s] linear features, scene edges, texture information, and point target" (p. 10)
   - Terrain Correction (DEM-based)
   - Subset (clip to An Giang)
   - Conversion to dB (logarithmic)
3. **Dynamic Otsu thresholding (Sec. 3.1.3, p. 11).** For *each* pre-processed VH dB image independently, the Otsu threshold *t* is found by exhaustive search over the gray-level histogram, minimizing the intra-class variance:

   `σ²(t) = P_w(t) × σ_w²(t) + P_nw(t) × σ_nw²(t)`   (Eq. 1, p. 11)

   where `P_w, σ_w` and `P_nw, σ_nw` are the probability and variance of the water (w) and non-water (nw) classes split by threshold *t*. Decision rule: **backscatter < t → water; ≥ t → non-water** (p. 11). This is a *global* threshold per scene (the authors note in the Discussion that a *local* threshold could improve results — see Limitations).
4. **Sentinel-2 reference branch (Sec. 3.2).** Sentinel-2 L1C → surface reflectance via SNAP/Sen2Cor; compute NDWI and MNDWI; then apply **the same automatic Otsu threshold** to the index image (rather than the conventional threshold of 0) so the two branches are methodologically consistent. Decision: **index value > t → water** (p. 12).
5. **Flood mapping by change detection (Sec. 3.4, Fig. 7, p. 14).** Purely rule-based, per-pixel, comparing time *t* with *t−1*:
   - all **non-water** pixels at time *t* → **non-flooded** at *t*;
   - a **water** pixel at *t* whose state at *t−1* was **non-water** → **flooded** at *t*;
   - a **water** pixel at *t* whose state at *t−1* was **water** → look at the *flood map* at *t−1*: if it was **flooded**, stay **flooded**; otherwise **non-flooded** (i.e. permanent water is excluded).
   This is O(1) per pixel per time step, hence "NRT" — it needs only the previous water map and previous flood map.
6. **Initialization fix (Sec. 4.3.1, p. 20).** The algorithm needs a seed flood map. They assume the driest date (12 March 2017) is all non-flooded, but water pixels present that day would be permanently misclassified. So they track the decay of the initial water pixels: 59% of them became non-water within two weeks (41% remaining on 24 Mar 2017), decaying to **~12% by 5 May 2017**, after which it is stable (that residual ~12% = permanent water, lakes/rivers). The **actual start time `t_start` is therefore set to 5 May 2017** (p. 20).

**Hyperparameters:** none in the ML sense — there is no loss function, optimizer, learning rate, epoch count, batch size, patch size or augmentation. All are **not applicable** (and correspondingly `not reported`).

## Data

**Sentinel-1 (primary):**
- Product: GRDH, IW swath, C-band 5.405 GHz, incidence 30.4°–46.2°, swath 250 km, **pixel size 10 m** (10 × 10 m stated in Table 1, p. 6), dual-pol VH + VV, 6-day revisit (Table 1, p. 6).
- **128 images**, March 2017 → March 2018, from ESA SciHub + Alaska Satellite Facility; **2 images per acquisition date** are required to cover An Giang → **64 dates → 64 surface water maps** (p. 6, p. 15). Full date list in Table 1 cont. (p. 7).
- **Only VH used.**

**Sentinel-2 (reference / cross-check only):**
- Level **1C (TOA)**, tile **T48PWS**, 10–60 m, **cloud cover < 50%**, and constrained to be acquired **< 1 day** from the paired Sentinel-1 image (Table 2, p. 8). Converted to surface reflectance with SNAP.
- **10 Sentinel-2 images** → **10 Sentinel-1/Sentinel-2 pairs** (dates: 2017.03.12, 2017.04.11, 2017.07.10, 2017.10.08, 2017.12.12, 2018.01.11, 2018.02.05, 2018.02.10, 2018.03.12, 2018.03.31 — Table 2, p. 8).
- SCL band (20 m) nearest-neighbour resampled to 10 m for cloud masking; SWIR B11 (20 m) also resampled to 10 m before MNDWI (p. 12).
- **Sentinel-2 FRB (Full Resolution Browse)** images from USGS EarthExplorer — a gamma-2.0 stretched SWIR/NIR/Red composite — used **only for visual comparison** (p. 13).

**Indices, verbatim formulas:**
- **NDWI** (McFeeters 1996): `NDWI = (ρ3 − ρ8) / (ρ3 + ρ8)` — Eq. 2, p. 12, with ρ3 = green (0.56 µm), ρ8 = NIR (0.842 µm). **This is exactly the thesis's NDWI (B03, B08).**
- **MNDWI** (Xu 2006): `MNDWI = (ρ3 − ρ11) / (ρ3 + ρ11)` — Eq. 3, p. 12, with ρ11 = SWIR (1.61 µm).
- **NDVI is not used in this paper.**

**Study area:** An Giang province, VMD, Vietnam — >3500 km², 10°–11° N, flat except NW hills, crossed by the Tien and Hau branches of the Mekong, heavily diked, 2–3 rice seasons/year (p. 4–5).

**Ground truth:** ***none***. Stated explicitly: "One of the most critical challenges … is the lack of ground truth data. It is very difficult or unable to collect the ground truth data in the region during the flood period" (p. 12). There is **no train/val/test split** because there is no training. Validation is a *cross-product comparison* against Sentinel-2 MNDWI+Otsu water maps, sampled as **47 cloud-free river subsets and 45 paddy-field subsets, each 0.5–12 km²**, drawn from the 10 image pairs (p. 19).

## Results

**Dynamic threshold behaviour (Sec. 4.1.1, p. 15):** the Otsu threshold on Sentinel-1 VH is *not* constant. Average threshold in the flood period (Aug–Nov) ≈ **−22 dB**, higher in the rice seasons; **the largest difference is 4 dB between the rice harvest and flood periods** (p. 15). By contrast, the Otsu thresholds on Sentinel-2 NDWI/MNDWI were **"stably very close to zero"** across the year (p. 18) — i.e. dynamic thresholding matters for SAR, much less for optical indices. Example (11 April 2017, Fig. 11, p. 18): S1 VH range −36 to 20 dB, Otsu **t = −19.9 dB**; S2 NDWI range −1 to 0.66, Otsu **t = −0.26**; S2 MNDWI range −1 to 0.92, Otsu **t = −0.02**.

**Index choice (Sec. 4.2.2, p. 18):** Otsu-on-**NDWI** "was unstable … since it overestimates surface water in the paddy field areas in all three periods"; Otsu-on-**MNDWI** "effectively captured surface water in both river and paddy field areas". MNDWI+Otsu was therefore chosen as the reference. **This is a direct, explicit caution against the thesis's NDWI in agricultural/mixed-pixel terrain** — though note it is a *qualitative/visual* verdict, no number is attached to the NDWI-vs-MNDWI comparison.

**Headline quantitative agreement (Fig. 13, p. 19–20; restated p. 25):**

| Comparison (S1-VH+Otsu vs S2-MNDWI+Otsu) | n subsets | Regression | R² | RMSE |
|---|---|---|---|---|
| **River areas** | 47 | y = 0.97x + 0.73 | **0.97** | **1.18%** |
| **Paddy fields** | 45 | y = 0.92x + 4.02 | **0.88** | **3.88%** |

(Values verbatim from Fig. 13, p. 20, and repeated in the Abstract p. 1, Discussion p. 23, and Conclusions p. 25 — internally consistent.) The metric is **water percentage per subset**, not per-pixel classification. In paddy fields the S1-derived water proportion is systematically **larger** than the S2-derived one (positive intercept 4.02), attributed to Sentinel-1's ability to detect very shallow water (<5 cm) during the sowing period (p. 20).

**Flood maps (Sec. 4.3.2, p. 21–22):** 56 flood maps generated (permanent water excluded). Flood onset early August 2017; **minimum before the flood: 16% of An Giang flooded on 22 July 2017**; **peak: 36% on 8 September 2017**; recession Oct–Nov. Two additional non-flood peaks of **36% in May 2017 and 36% in December 2017**, which the authors attribute to *irrigation*, not flooding — the algorithm cannot currently distinguish the two (p. 22).

**Baselines: NONE were run.** The paper does **not** compare against a static/empirical threshold, against SVM/RF, or against any deep-learning model, quantitatively. The 89.3% overall water-classification accuracy sometimes associated with Otsu is **cited from Bangira et al. [62]** (p. 4), *not* measured here — do not attribute it to Tran et al.

## Limitations

Author-stated:
- **No in-situ ground truth; validation is purely inter-product.** "the current validation results are purely qualitative between different products without referencing the in-situ data" (p. 23). They suggest crowdsourced geotagged smartphone photos as a future GT source.
- **Cannot separate irrigation from natural flood.** Both appear as "flooded" (p. 22, p. 24). Future work: use flood duration (irrigation < 1 month, flood > 2 months) or a supervised classifier with three classes (irrigation / flooded / non-flooded).
- **Global, not local, Otsu.** They cite Liang et al. [114] finding a local thresholding approach improves the harmonic mean of user's/producer's accuracy of water by **4–13%** over global thresholding, and recommend investigating local Otsu (p. 24).
- **Lower accuracy in paddy fields** (mixed soil/water/vegetation pixels) than in rivers (p. 23).
- **Only one year** (Mar 2017–Mar 2018); no inter-annual variability studied (p. 25).
- Sentinel-1B decommissioned (Dec 2021) → revisit degrades from 6 to 12 days for the algorithm (p. 24).

Observed by me:
- ⚠️ **No per-pixel accuracy metric at all** — no OA, kappa, F1, IoU, precision/recall, confusion matrix. R² and RMSE are computed on *areal water fraction of a subset*, which can look excellent even when pixel-level delineation is poor (compensating false positives/negatives). Treat the "R² = 0.97" as an *areal agreement*, not a segmentation accuracy.
- ⚠️ The "reference" (S2 MNDWI + Otsu) is itself an unvalidated automatic product, so this is a **method-vs-method agreement**, not a validation.
- ⚠️ **Minor internal inconsistency:** the Figure 6 caption (p. 11) says the two samples cultivate "two rice seasons a year (**summer-autumn and autumn-winter** seasons)" while the figure axis labels and the body text describe **summer-autumn and winter-spring**. Likely a caption typo.
- The initial-condition fix (`t_start` = 5 May 2017) is empirical and area-specific; it discards ~2 months of the series.
- The change-detection rule has no hysteresis/temporal smoothing, so a single misclassified scene propagates into the flood map of the next step.

## Relevance to this thesis

**High relevance as the NON-DL BASELINE.** This is the paper to beat, and to cite for the "why deep learning at all?" argument.

- **Use as the quantitative baseline to beat.** The thesis CNN should report, on the same or comparable areas, something better than the areal agreement Tran et al. achieve. But note the metric mismatch: they report only **R² / RMSE on water fraction**, no OA/F1/IoU. **Actionable:** implement dynamic Otsu on NDWI/NDWI-derived layers as a *reproduced baseline*, and evaluate *both* it and the CNN with proper per-pixel metrics (IoU, F1, OA) on your own labels — then you can honestly say "Otsu baseline gives X, our CNN gives Y", which Tran et al. themselves could not do.
- **Borrow the algorithm directly (it is ~15 lines of code).** Otsu is `skimage.filters.threshold_otsu` on the histogram of the index band. Eq. 1 (p. 11) is standard intra-class-variance Otsu. This is a cheap, zero-training baseline you can compute on every downloaded Sentinel-2 scene in the Streamlit/Celery pipeline as a fallback / sanity layer.
- **Borrow the change-detection flood rule (Fig. 7, p. 14).** Your thesis is explicitly about *time series* of NDVI/NDWI. Their `water(t) ∧ ¬water(t−1) → flooded(t)`, plus carry-forward, plus permanent-water exclusion, is a directly transplantable post-processing step that converts *water maps* into *flood maps* — a distinction the thesis needs to make (permanent rivers/lakes in Peru must not be reported as flood).
- **Borrow the `t_start` / permanent-water calibration trick (p. 20).** Their water-pixel-decay curve (Fig. 14) is a clean, unsupervised way to identify **permanent water** from a time series: pixels that stay water across the whole series (~12% here). Use this to build a permanent-water mask for the Peruvian AOI without any labels — then flood = water minus permanent water.
- **⚠️ Cautionary finding directly against the thesis's index choice.** They found **Otsu on NDWI overestimates water in paddy/agricultural areas in all periods**, while **MNDWI (green vs SWIR B11) was reliable** (p. 18, p. 23). The thesis currently uses **only B02/B03/B04/B08** and McFeeters NDWI. **Strongly consider adding B11 (SWIR, 20 m → resample to 10 m) and computing MNDWI** as an extra channel — it is free from Sentinel Hub and this paper is direct evidence it is better over mixed soil/water/vegetation, which describes Peruvian agricultural floodplains.
- **Cautionary finding on cloud.** "85% to 95% of the VMD is covered by a persistent cloud during the wet season" (p. 3). The thesis's *max 10% cloud* filter over an Amazonian/coastal Peruvian AOI during the rainy season may return **almost no usable scenes exactly when floods happen**. Cite this as motivation either to (a) relax the cloud filter and mask per-pixel with SCL, (b) add Sentinel-1 SAR as a second modality, or (c) frame the thesis as *susceptibility / pre-post* rather than *during-event* mapping. This is arguably the single most important warning in the paper for your project.
- **Differences from the thesis setup (be explicit when citing):** SAR (VH backscatter) vs optical (thesis); unsupervised thresholding vs supervised CNN; Mekong rice delta vs Peru; inundation mapping vs susceptibility *prediction*; validation against another satellite product vs (presumably) labelled data. They share: 10 m, time series, flood/disaster framing, Otsu-able water indices.
- **Cite as:** (a) a **baseline**, (b) a **method to reuse** (Otsu + change detection + permanent water mask), (c) a **cautionary example** on NDWI in agriculture and on optical cloud cover, (d) supporting evidence for the "no ground truth in flooded areas" problem, which the thesis will also face.

## Keywords / Tech

- **Models/algorithms:** Otsu automatic thresholding (dynamic, per-scene, global); rule-based change-detection time series; **no ML/DL**
- **Sensors:** Sentinel-1 C-band SAR (GRDH, IW, **VH** polarization, 10 m, 6-day); Sentinel-2 MSI (L1C→SR, 10–60 m); Sentinel-2 FRB browse composites
- **Indices:** NDWI `(ρ3−ρ8)/(ρ3+ρ8)`; MNDWI `(ρ3−ρ11)/(ρ3+ρ11)`
- **Preprocessing:** SNAP; orbit file, thermal + border noise removal, slice assembly, radiometric calibration, **Refined Lee speckle filter**, terrain correction, dB conversion; Sen2Cor; SCL cloud masking; SWIR resampled 20 m → 10 m
- **Data sources:** ESA SciHub, Alaska Satellite Facility, USGS EarthExplorer, MRC Tan Chau gauge
- **Metrics used:** R², RMSE (on areal water percentage). *No OA, F1, IoU, kappa.*

## Notable quotes

- "85% to 95% of the VMD is covered by a persistent cloud during the wet season, making it infeasible or low accuracy for various applications, such as land use land cover (LULC) classification and waterbodies extraction" (p. 3).
- "The pixels with a backscatter value less than the estimated Otsu threshold t were classified as water and greater than or equal to the estimated Otsu threshold t were classified as non-water." (p. 11)
- "the use of the Otsu threshold on the NDWI image was unstable for water delineation since it overestimates surface water in the paddy field areas in all three periods. In contrast, applying the Otsu threshold on the MNDWI image effectively captured surface water in both river and paddy field areas." (p. 18)
- "the combination of the automatic Otsu threshold and Sentinel-1 images detects surface water in the river areas efficiently (R² = 0.97 and RMSE = 1.18%). However, lower effectiveness is observed in the paddy field areas, which is predictable because most pixels on the paddy fields are mixed of soil, water, and vegetation" (p. 23).
- "the current validation results are purely qualitative between different products without referencing the in-situ data." (p. 23)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/NQ72ZIE5/Tran et al. - 2022 - Surface Water Mapping and Flood Monitoring in the Mekong Delta Using Sentinel-1 SAR Time Series and.pdf`
Cite as: `\cite{tran_surface_2022}`
