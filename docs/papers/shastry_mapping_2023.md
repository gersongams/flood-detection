---
title: "Mapping floods from remote sensing data and quantifying the effects of surface obstruction by clouds and vegetation"
authors: "Apoorva Shastry, Elizabeth Carter, Brian Coltin, Rachel Sleeter, Scott McMichael, Jack Eggleston"
year: 2023
venue: "Remote Sensing of Environment, vol. 291, article 113556, 12 pp."
doi: "10.1016/j.rse.2023.113556"
bibtex_key: shastry_mapping_2023
zotero_pdf: "/Users/gersongarrido/Zotero/storage/UAQMNXZX/Shastry et al. - 2023 - Mapping floods from remote sensing data and quantifying the effects of surface obstruction by clouds.pdf"
pages: 12
sensors: [WorldView-2, WorldView-3, "(hydraulic model: USGS FIM / HEC-RAS)"]
task: "flood inundation mapping (semantic segmentation of surface water) + quantification of optical obstruction by clouds and vegetation"
model: "CNN (dilated/Atrous-convolution 'spinal-cord' segmentation network) in the NASA DELTA framework"
tags: [flood-mapping, optical-limitations, cloud-obstruction, vegetation-obstruction, cnn, semantic-segmentation, hydraulic-model, worldview, underprediction, sar-motivation]
---

# Mapping floods from remote sensing data and quantifying the effects of surface obstruction by clouds and vegetation

> **TL;DR** — The authors trained a CNN (NASA's open-source DELTA framework) on a strategically stratified, expert-labelled dataset of ~2 m 8-band Maxar WorldView-2/3 imagery to segment surface water, and then benchmarked its flood maps against USGS Flood Inundation Mapper (FIM) hydraulic-model extents at 17 gaged locations (45 areas of interest, ~682 km²). The model is excellent as a *water detector* (98.2% precision, 93.7% recall on held-out validation, p. 7) but as a *flood-extent* product it **misses 62% of the modelled inundation** (miss rate 61.6% vs FIM, p. 7). They attribute **~79% of the underpredicted area to surface obstruction — 74% vegetation, 9% clouds, 4% both** (p. 8). Within cloud-covered pixels the CNN misses >90% of the flood (recall 9%, miss rate 91%, p. 8); within vegetation, recall 38% / miss rate 62% (p. 8). Conclusion: optical-only flood mapping structurally underestimates flood extent, and should be fused with SAR and/or hydraulic models.

## Problem & Motivation

Floods were the most frequent disaster type globally 1998–2017 (43% of recorded events, affecting >2 billion people, 45% of all people affected by disasters — p. 2, citing UNISDR 2017). Optical satellite imagery is the most abundant data source for automatic flood mapping, and CNN-based semantic segmentation now dominates the literature. But the paper identifies **two critical impediments to optical flood mapping: cloud cover and vegetation** (p. 2). Optical sensors "are unable to see through clouds, and clouds can be common during flood events" and flooded vegetation is also hard to detect (p. 2). The literature quantifies model accuracy against optical-derived labels — which are themselves blind to the same obstructions — so the *true* error is invisible in standard validation.

Second motivation: most CNN training data come from individual events, so models generalize poorly. The authors build a **strategically stratified** training set (latitude, topographic complexity, land use, day-of-year) drawn from a pool of >84,000 WorldView images (p. 2), to cover the full diversity of surface-water spectral/spatial signature across the CONUS.

The two stated contributions (p. 3): (i) an automated semantic segmentation workflow for high-res commercial multispectral imagery that generalizes across CONUS, and (ii) an estimate of how much flood inundation optical remote sensing *fails to capture* because of vegetation and clouds, obtained by comparing classified imagery to hydraulic-model (FIM) extents.

## Method / Architecture

**Framework**: NASA DELTA (Deep Earth Learning, Tools, and Analysis), open source, TensorFlow/Keras, handles tiling of very large satellite images (github.com/nasa/delta) (p. 5).

**Network** (Fig. 3, p. 5) — a modified version of the Perone et al. (2018) spinal-cord gray-matter segmentation network, built on **dilated (Atrous) convolutions**:
- Input → two 3×3 convolutions → 2D spatial dropout → two 3×3 dilated convolutions (rate 3) → 2D spatial dropout ⇒ initial feature map.
- Network then **branches into five**: branch 1 = two 1×1 convolutions; branches 2–5 = two 3×3 Atrous convolutions at **dilation rates 6, 12, 18, 24**.
- The five branches are concatenated → 2D spatial dropout → 1×1 convolution → final 1×1 convolution produces the output.
- All convolutions (except output) use **32 filters** + **batch normalization**. **Dropout rate 0.6.**
- **Modification 1**: the original 6th branch (global average pooling on the initial feature map) is **removed**, because when the image is split into tiles it produced a "distinctive tiling effect" on merge — no meaningful numeric harm, but it eroded analyst trust (p. 5).
- **Modification 2**: instead of a dropout layer after every convolution, they drop *entire feature channels* at only three key spots — found more robust to overfitting (p. 5).

**Architecture comparison** (p. 5): they investigated **U-Net** (Ronneberger et al., 2015) and **DeepLab v3** (Chen et al., 2017). The modified spinal network *outperformed U-Net* and had *similar loss performance to DeepLab*, but DeepLab has more parameters and is slower; both U-Net and DeepLab showed the visually displeasing tile-edge effects. Hence the spinal net was selected. ⚠️ No numeric table of this comparison is given — only the qualitative statement.

**Training setup** (p. 5):
- Tiles/patches: **512 × 512**, offset a random amount each epoch to prevent overfitting.
- **All image bands used** (8 WorldView bands).
- Augmentation: random vertical and horizontal flips, random rotations (**max ±4.6 degrees**), random brightness scaling (**0.75–1.15**).
- Optimizer: **Adam**.
- LR: **polynomially decaying**, initial **1e-3**, final **1e-5**, **power 0.9**.
- **100 epochs**.
- Loss: **sum of Dice loss + Focal loss** (Jadon, 2020). They note Perone et al. used only Dice; "we found adding focal loss improved convergence and helped fit to rarer test cases" (p. 5).
- **Batch size: not reported.** Hardware/training time: **not reported.**

**Evaluation metrics** (p. 6), defined explicitly because the class is imbalanced (water ≈17% of the training dataset, so a trivial all-dry model would score 83% accuracy):
- `Precision = TP / (TP + FP)` (Eq. 1) — penalizes overprediction of water.
- `Recall = TP / (TP + FN)` (Eq. 2) — hit rate; penalizes underprediction.
- `Miss rate = FN / (TP + FN)` (Eq. 3) — the fraction of true water that is missed. **This is the paper's headline statistic.**
- No IoU, no F1, no kappa reported.

**Three evaluation levels**:
1. CNN vs held-out **validation labels** (20 WV images).
2. CNN vs **reference labels** (hand/semi-automatically labelled water in the 45 AOIs).
3. CNN vs **hydraulic-model FIM polygons** (the key experiment: the model extent includes water hidden under clouds/canopy).
4. And, sliced within **NLCD vegetation** pixels and within **cloud** pixels separately (§3.3.3, p. 7).

## Data

**Imagery** — Maxar **WorldView-2 and WorldView-3**, spatial resolution **~2 m**, **8 bands**: 1-Coastal, 2-Blue, 3-Green, 4-Yellow, 5-Red, 6-Red Edge, 7-NIR1, 8-NIR2 (p. 3). WV-2 launched 2009 (revisit 1.1 days); WV-3 launched 2014 (revisit <1 day).

**Training/validation set** (p. 4): **100 satellite scenes** selected from a pool of >84,000 candidate WorldView images, strategically stratified over: % area in canopy cover, % area covered by impervious surface, day of year, latitude, day-of-year × latitude, and canopy × latitude; plus dominant land cover (2009 Landsat-derived NLCD), mean elevation and slope gradient (NED 1/9 arc-sec DEM), and hydrologic ecoregion. Split: **80 images train / 20 images validation** (p. 5). Snow and ice excluded. Labels = **two discrete classes: water / not water**, produced by a hybrid procedure — unsupervised spectral clustering, then expert manual interpretation and QA/QC peer review, by USGS geospatial data scientists (Sleeter et al., 2020 — the public dataset).

**Flood-comparison images** (§2.1.2, p. 4): from WV imagery collected **2010–2019**, selecting images that (i) overlap USGS FIM polygons, (ii) had a stage measurement above the "Action" stage at the FIM gage (i.e. a flood was occurring), (iii) **cloud cover < 35%**, (iv) manual QC to remove images with thick clouds. **39 images met the criteria** (details in Table S2). Reference water labels and **cloud labels** were hand-created for these 39 images following the Sleeter et al. (2020) methodology (§2.1.2.1, §2.1.2.3, p. 4). ⚠️ Note the authors explicitly warn the reference labels "do not overcome the problems of clouds and vegetation obscuring some pixels" (p. 4) — they are only the best available estimate of *observed* inundation.

**Vegetation mask**: **NLCD** (2011, 2016, 2019 — nearest map to each image date). Vegetation class = Forests (41 Deciduous, 42 Evergreen, 43 Mixed), Shrublands (52 Shrub/Scrub), Planted/Cultivated (81 Pasture/Hay, 82 Cultivated Crops), Wetlands (90 Woody Wetlands, 95 Emergent Herbaceous Wetlands). **71-Grassland/Herbaceous was deliberately omitted** because grasses typically don't obstruct the view of flooded regions (p. 4).

**Hydraulic model reference**: USGS **Flood Inundation Mapper (FIM)** library, >100 US locations, built on **2-D HEC-RAS steady-state** models calibrated to USGS streamgage data, using **1 m LiDAR-derived DEMs** (p. 4). For each WV image, the stage at the nearest gage at the time closest to image acquisition was pulled from NWIS, and the corresponding FIM polygon was rasterized onto the WV grid. **17 FIM locations** were used → **45 areas of interest**, total area **~680 km²** (p. 4) / **~682 km²** (p. 8). ⚠️ Minor inconsistency between the two figures; treat ~682 km² (from the obstruction analysis) as the operative number.

**Spectral indices**: **none**. The CNN consumes raw 8-band reflectance; no NDVI/NDWI/MNDWI is computed anywhere in the paper. (Indices are only mentioned in the intro as the "traditional" approach that CNNs supersede, p. 2.) No atmospheric-correction/normalization details are given beyond band usage. ⚠️ Preprocessing (radiometric normalization, pansharpening) is **not reported**.

## Results

### 1. CNN vs held-out validation labels (20 WV images) — p. 7

| Metric | Value |
|---|---|
| Precision (water) | **98.2%** (p. 7) |
| Recall (water) | **93.7%** (p. 7) |

⚠️ The abstract rounds these to "98% precision and 94% recall" (p. 1) and the conclusions say "98% precision and 94% recall" (p. 9) — consistent, just rounded. Qualitatively "the neural network struggles with narrow channels of water and in areas with thin clouds and cloud shadows" (p. 7).

### 2. CNN vs reference labels, 45 AOIs — p. 7 (Fig. 6a)

| Metric | Value |
|---|---|
| Precision | **92.7%** |
| Recall | **83.2%** |
| Miss rate | **16.8%** |
| TP px | 24,784,127 (0.832) |
| FN px | 5,011,888 (0.168) |
| FP px | 1,945,293 (0.014) |
| TN px | 138,759,437 (0.986) |

So even against optical-derived reference labels, performance drops from the training distribution (98.2/93.7 → 92.7/83.2).

### 3. CNN vs hydraulic model (FIM), 45 AOIs — p. 7–8 (Fig. 6b) ⭐ THE KEY RESULT

| Metric | Value |
|---|---|
| Precision | **76.5%** |
| Recall | **38.4%** |
| **Miss rate (underprediction)** | **61.6%** ("we estimated the underprediction of flood inundation by optical remote sensing data in our areas of interest to be **62%**", p. 1 / p. 9) |
| TP px | 20,436,230 (0.384) |
| FN px | 32,747,293 (0.616) |
| FP px | 6,293,190 (0.054) |
| TN px | 111,024,032 (0.946) |

True positives ≈ **12%** of all pixels in the 45 AOIs; false negatives ≈ **19%** of all pixels (p. 8). "The false negatives are considerably larger than true positives, indicating that DELTA misses more than it predicts correctly" (p. 8).

### 4. Sliced by obstruction type — p. 8 (Fig. 7) ⭐ THE NUMBERS TO CITE

Areas within the 45 AOIs (total ~682 km²):
- **Vegetation (NLCD: forest, shrubland, herbaceous, cultivated, wetlands): ~436 km² = 64% of total area**
- **Cloud (from reference labels): ~37 km² = 5% of total area**
- **Both cloud AND vegetation: ~20 km² = 3% of total area**
- Authors' caveat: "It should be noted that in typical flood events, the cloud cover can be much higher. The magnitude of cloud cover in our analysis is small, as we were looking at images with minimal cloud cover." (p. 8)

**CNN vs FIM, within vegetated pixels (Fig. 7a):**

| Metric | Value |
|---|---|
| Precision | **75%** ("DELTA overpredicts water by 25%") |
| Recall / hit rate | **38%** |
| Miss rate | **62%** |
| TP px | 13,095,894 (0.384) |
| FN px | 20,966,930 (0.616) |
| FP px | 4,454,948 (0.057) |
| TN px | 73,342,387 (0.943) |

→ "DELTA potentially misses the majority of flooded vegetation" (p. 8).

**CNN vs FIM, within cloud pixels (Fig. 7b):**

| Metric | Value |
|---|---|
| Precision | **93%** |
| Recall | **9%** |
| Miss rate | **91%** |
| TP px | 287,115 (0.090) |
| FN px | 2,890,487 (0.910) |
| FP px | 34,747 (0.006) |
| TN px | 6,033,918 (0.994) |

→ "**DELTA misses >90% of floods under clouds.**" (p. 8)

### 5. Attribution of the underprediction — p. 8 (Fig. 8) ⭐

Total **underpredicted area = 131 km²** (of ~682 km²). Of that underpredicted area:
- **74%** belonged to the **vegetation** class — broken down as **18% forests, 21% cultivated, 35% wetlands** (p. 8)
- **9%** were **under clouds**
- **4%** were under **both** clouds and vegetation
- ⇒ **~79% of the total underpredicted area is obscured by vegetation or cloud cover** (p. 8, p. 9, abstract p. 1)
- The residual ~21% is attributed to developed areas / grasslands and DEM error ("this is an example of the <10% underprediction that belongs to developed areas and grasslands", p. 9).

Total **overpredicted area = 32 km²** — "essentially most of the overprediction are located mainly in developed and cultivated areas. These are mainly ponded areas remaining after the flood recession, and these areas would not be correctly captured in a steady state hydraulic model" (p. 8). I.e. much of the "overprediction" is arguably a *model* error, not a CNN error (permanent lakes and ponds outside the predicted flood extent are also not in the FIM polygon but are correctly seen by the CNN).

### Baselines
⚠️ **No quantitative baseline comparison was run.** U-Net and DeepLab v3 are mentioned as alternatives that were "investigated" (p. 5) but **no numbers are given for them**. There is no comparison to an NDWI/threshold baseline, nor to another published flood CNN. The FIM hydraulic model is a *reference*, not a competing method.

## Limitations

**Stated by the authors:**
- The FIM hydraulic models themselves have errors and "may not provide good physical representation of the particular floods captured by the images" (p. 8); errors in the DEM propagate to large errors in predicted flood extent, especially in extremely flat floodplains (Fig. 9d shows FIM covering "a much larger area" than reality).
- FIM is **steady-state**, so it represents the maximum flood extent for a given stage and cannot represent instantaneous extents; it also cannot represent ponded water on the falling limb of the hydrograph (Fig. 9e–h).
- FIM does not include permanent lakes/ponds outside the modelled reach → inflates the CNN's apparent false positives.
- Cloud cover in their sample is **atypically low** (images filtered to <35% cloud, and manually QC'd); in real flood events cloud cover is much higher, so the 9% cloud-attributed underprediction is a **lower bound** (p. 8).
- The reference labels themselves cannot see through clouds/canopy, so evaluation level 2 is optimistically biased (p. 4).

**Observed by me:**
- Only **17 FIM locations / 45 AOIs / 39 images** — CONUS only, riverine floodplains only. No coastal or urban pluvial flooding in the obstruction analysis.
- Sensor is **WorldView (~2 m, 8 bands, commercial, not freely available)** — not directly transferable to Sentinel-2's 10 m / 4-band setting; obstruction percentages are likely **worse** at 10 m due to mixed pixels.
- No IoU/F1, no uncertainty quantification, no confidence intervals on the 62% / 79% figures.
- No ablation numbers for the architecture choices (branch removal, dice+focal), only qualitative claims.
- The 74/9/4% attribution is a **spatial-overlap** argument, not a causal one — an underpredicted pixel in a forest could also be a FIM DEM error.

## Relevance to this thesis

**This is the single most important paper for justifying — or honestly bounding — the thesis's optical-only Sentinel-2 design.** Use it in the Limitations / Discussion chapter, not as a method baseline.

- **The numbers to cite when defending the 10% max-cloud filter.** The paper shows the filter is *necessary but not sufficient*: they filtered to <35% cloud and *still* the CNN missed **>90% of the flood under the remaining cloud pixels** (recall 9%, miss rate 91%, p. 8). A 10% cloud threshold reduces the *area* affected but not the *blindness* within that area. Concretely: any pixel that is clouded is essentially a guaranteed flood miss (91%).
- **The bigger threat to the thesis is vegetation, not clouds.** 74% of underpredicted flood area was under vegetation vs only 9% under cloud (p. 8). A strict cloud filter does nothing about this. Under vegetation, recall is 38% / miss rate 62% (p. 8). Since the thesis relies on **NDWI (B03−B08)/(B03+B08)** — an index that is defeated by canopy over water exactly as a CNN on raw bands is — this failure mode applies directly and arguably *more* strongly, since NDWI is a single-pixel spectral test with no spatial context.
- **The headline number for the thesis abstract/discussion: optical remote sensing underpredicts flood inundation by ~62% relative to a hydraulic model, and ~79% of that gap is attributable to vegetation (74%) and cloud (9%) obstruction (4% both)** (p. 1, p. 8, p. 9).
- **Cautionary example on validation design.** The paper shows the *exact* trap the thesis is exposed to: validating against optical-derived labels gives 92.7% precision / 83.2% recall — looks great — while against a physically-based flood model the same predictions score 76.5% / 38.4%. If the thesis validates its CNN against NDWI-derived or hand-drawn optical labels, **its reported metrics will be structurally optimistic by construction.** Cite this explicitly when reporting metrics.
- **Actionable mitigation the thesis can adopt (and cite this paper for):** (a) fuse **Sentinel-1 SAR** with Sentinel-2 — the paper's own recommendation ("a merged WV and Sentinel-1 derived flood extent map has the potential to improve accuracy substantially in areas with cloud cover", p. 10); (b) fuse with a **hydraulic/hydrodynamic model** or a HAND (Height Above Nearest Drainage) product (Nobre et al., 2011) to recover flooded vegetation; (c) at minimum, **report the cloud/vegetation fraction of the study AOI** (they use NLCD; the Peruvian analogue would be a national land-cover map or an NDVI-derived canopy proxy) so the reader can bound the underestimate. Reporting "X% of my Peruvian AOI is forest/wetland, therefore my flood extent is a lower bound" is a cheap, high-value paragraph.
- **Borrowable technique — the loss function.** `Dice + Focal` for a heavily imbalanced water class (water ≈17% of pixels, p. 6). Directly reusable in PyTorch; the thesis's flood class will be far rarer than 17%.
- **Borrowable technique — the metric set.** Report **precision, recall, and explicitly "miss rate" (FN/(TP+FN))**, not accuracy. They give the reason verbatim: with 17% water, a trivial all-dry model gets 83% accuracy (p. 6). This is a strong, citable justification for the thesis's metric choice.
- **Borrowable technique — augmentation.** Flips, small rotations (±4.6°), brightness scaling 0.75–1.15 (p. 5) — mild and appropriate for satellite imagery; no elastic/heavy geometric distortion.
- **Architecture note (weak evidence).** They report the dilated-convolution "spinal" net *outperformed U-Net* on this task — but **give no numbers** (p. 5), so this cannot be used to justify avoiding U-Net. The reproducible takeaway is the practical one: U-Net and DeepLab produced **visible tiling artifacts** at tile boundaries when large scenes are chunked. If the thesis tiles Sentinel-2 scenes into patches for a PyTorch CNN, expect and handle seam artifacts (overlap tiles + blend, or remove global-pooling branches).
- **Where it differs from the thesis** — important, do not overclaim: sensor is **WorldView 2 m / 8 bands (commercial)**, not Sentinel-2 10 m / 4 bands; study area is **CONUS riverine floodplains**, not Peru; the task is **instantaneous inundation mapping**, not flood-*susceptibility* prediction; the framework is **TensorFlow/DELTA**, not PyTorch; and **no spectral indices (no NDVI/NDWI) are used at all** — the CNN eats raw bands. If the thesis is doing *susceptibility/prediction* rather than *inundation extent*, the 62% figure applies to the **label quality** (the flood masks used to train), not to the final prediction — which is arguably an even more damning framing: **the training labels themselves are missing 62% of the floods.**
- **Classification: cite as (a) a cautionary/limitation reference, (b) a source of the canonical obstruction statistics, (c) a data-source pointer (Sleeter et al., 2020 — public USGS satellite-derived flood training labels; USGS Flood Inundation Mapper; NASA DELTA at github.com/nasa/delta). It is NOT a usable baseline** (different sensor, different resolution, no comparable metric on Sentinel-2).

## Keywords / Tech

- **Models**: CNN semantic segmentation; dilated/Atrous convolutions (rates 3, 6, 12, 18, 24); modified Perone et al. (2018) spinal-cord segmentation net; compared (qualitatively) to U-Net and DeepLab v3.
- **Frameworks**: NASA **DELTA** (TensorFlow/Keras, open source, github.com/nasa/delta).
- **Sensors**: Maxar **WorldView-2 / WorldView-3** (~2 m, 8 bands). Mentioned as future/complementary: **Sentinel-1**, UAVSAR, Capella, Umbra, Iceye SAR.
- **Reference/model data**: USGS **Flood Inundation Mapper (FIM)** (2-D HEC-RAS, steady-state, 1 m LiDAR DEM); USGS **NWIS** stage; **NLCD** land cover; NED 1/9 arc-sec DEM.
- **Datasets**: Sleeter et al. (2020) — USGS satellite-derived training data for automated flood detection (public).
- **Techniques**: Dice + Focal loss; Adam; polynomial LR decay (1e-3 → 1e-5, power 0.9); 512×512 tiling with random offset; 2D spatial dropout (0.6, channel-wise); batch norm; precision / recall / **miss rate**; stratified sampling for training-set representativeness.
- **Indices**: **none used** (explicitly a CNN-on-raw-bands approach).

## Notable quotes

> "Compared to the hydraulic model, we estimated the underprediction of flood inundation by optical remote sensing data in our areas of interest to be 62%. We used land use data from National Land Cover Database (NLCD) and cloud masks to estimate that 79% of underprediction was due to these obstructions, with 74% belonging to vegetation, 9% to clouds, and 4% to both." (p. 1, abstract)

> "In cloudy areas, precision is 93%, recall is 9% and miss rate is 91%. Optical imagery is not suitable for detection of water beneath clouds, and this is apparent when DELTA and FIM flood extents are compared. DELTA misses >90% of floods under clouds." (p. 8)

> "Within vegetated areas, precision is 75%, indicating that DELTA overpredicts water by 25%. The recall, or hit rate is only 38%, while the miss rate is 62%. This means that DELTA potentially misses the majority of flooded vegetation." (p. 8)

> "It should be noted that in typical flood events, the cloud cover can be much higher. The magnitude of cloud cover in our analysis is small, as we were looking at images with minimal cloud cover." (p. 8)

> "Flood mapping using optical remote sensing alone will miss significant amounts of water under obstructions such as vegetation and clouds because of the nature of optical remote sensing. Considering data from additional sources such as SAR, flood models, and topography can produce more realistic flood extents." (pp. 9–10, Conclusions)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/UAQMNXZX/Shastry et al. - 2023 - Mapping floods from remote sensing data and quantifying the effects of surface obstruction by clouds.pdf`
Cite as: `\cite{shastry_mapping_2023}`
