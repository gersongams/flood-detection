---
title: "Deep Learning Segmentation of Satellite Imagery Identifies Aquatic Vegetation Associated with Snail Intermediate Hosts of Schistosomiasis in Senegal, Africa"
authors: "Zac Yung-Chun Liu, Andrew J. Chamberlin, Krti Tallam, Isabel J. Jones, Lance L. Lamore, John Bauer, Mariano Bresciani, Caitlin M. Wolfe, Renato Casagrandi, Lorenzo Mari, Marino Gatto, Abdou Ka Diongue, Lamine Toure, Jason R. Rohr, Gilles Riveau, Nicolas Jouanard, Chelsea L. Wood, Susanne H. Sokolow, Lisa Mandle, Gretchen Daily, Eric F. Lambin, Giulio A. De Leo"
year: 2022
venue: "Remote Sensing (MDPI), 14(6), 1345"
doi: "10.3390/rs14061345"
bibtex_key: liu_deep_2022
zotero_pdf: "/Users/gersongarrido/Zotero/storage/YSWCNIQI/Liu et al. - 2022 - Deep Learning Segmentation of Satellite Imagery Identifies Aquatic Vegetation Associated with Snail.pdf"
pages: 16
sensors: [WorldView-2, DJI Phantom IV drone (RGB)]
task: "semantic segmentation of aquatic vegetation (floating / emergent / water / land) as a proxy for snail habitat and schistosomiasis transmission risk"
model: "U-Net (classical Ronneberger encoder–decoder), benchmarked against Random Forest + GLCM"
tags: [u-net, semantic-segmentation, worldview-2, aquatic-vegetation, random-forest, glcm, schistosomiasis, keras, remote-sensing]
---

# Deep Learning Segmentation of Satellite Imagery Identifies Aquatic Vegetation Associated with Snail Intermediate Hosts of Schistosomiasis in Senegal, Africa

> **TL;DR** — The authors trained a classical U-Net on 8-band, 2 m WorldView-2 imagery of the lower Senegal River Basin to segment four classes (floating vegetation, emergent vegetation, water, land), using floating vegetation as a proxy for *Bulinus* snail habitat and thus urinary schistosomiasis transmission risk. Labels were hand-drawn on 50 satellite sub-images (256 × 256 × 8), validated against >6000 high-resolution drone photos. The U-Net reached **94.5% pixel accuracy on the test set and 82.7% on a hold-out/OOB set over all four classes** (96% / 84% for the floating class alone) (p. 11); a Random Forest + GLCM baseline scored higher on the test set (96.7%) but collapsed on the hold-out set (**67.8%**), showing the CNN generalizes far better across regions (Table 2, p. 12).

## Problem & Motivation
Schistosomiasis affects >200 million people, mostly in sub-Saharan Africa, and is tied to water-resource infrastructure (dams, canals). Control by mass drug administration does not prevent re-infection; targeting the intermediate snail host is more effective, but snail sampling is manual, labor-intensive, restricted to the near-shore area a technician can safely wade into, and cannot be scaled to the region-wide extent needed to prioritize interventions (pp. 2–3).

Prior fieldwork by the same group (ref. [7], Wood et al., PNAS) established that *Bulinus* snails are strongly associated with floating/lightly-rooted submerged vegetation (*Ceratophyllum* spp., *Potamogeton* spp.), and that the **area of suitable snail habitat is a better predictor of human infection risk than direct snail sampling**. The gap this paper fills: turn that ecological finding into a scalable mapping tool by learning the spectral signature of that vegetation from satellite imagery, with drone imagery used as the ground-truth bridge.

## Method / Architecture
Pipeline (Section 2.4, pp. 7–10): (1) preprocessing, (2) label masks, (3) U-Net training, (4) inference + colour-coded "transmission risk" heat maps.

- **Architecture**: classical U-Net [Ronneberger 2015], reproduced unchanged (Figure 4, p. 7). Contracting path (conv 3×3 + ReLU, max-pool 2×2) and symmetric expanding path (up-conv 2×2, skip connections via copy-and-crop). **23 convolutional layers total**; final 1×1 conv maps the 64-component feature vector to the **4 output classes** (p. 8).
- **Input**: 256 × 256 × 8 patches **up-scaled to 572 × 572 × 8** to fit the original U-Net input geometry (p. 8). Note: 8 input channels, not 3 — the model is fed all WorldView-2 bands directly.
- **Loss**: **binary cross-entropy** (p. 8). ⚠️ Odd for a 4-class problem; it is consistent with the fact that they produce **four independent binary masks** rather than a single mutually-exclusive label map, but the paper never says the output is sigmoid-per-class explicitly.
- **Metric**: pixel accuracy = (TP + TN) / (TP + TN + FP + FN) (Eq. 1, p. 9). No IoU/Dice/F1/kappa reported anywhere — **not reported**.
- **Optimizer**: stochastic gradient descent with "an adaptive learning rate gradient-descent back-propagation algorithm"; weights initialized randomly (p. 9). Exact optimizer name (Adam etc.) **not reported**.
- **Hyperparameter search** (grid search, p. 9): epochs ∈ {50, 100, 150}; batch size ∈ {8, 16, 32, 64}; LR coarse ∈ {1e-5, 1e-4, 1e-3, 1e-2}, then refined within the best power group in steps of 1e-5.
- **Chosen hyperparameters** (p. 11): **batch size 8, 100 epochs, LR 4 × 10⁻⁵**.
- **Augmentation**: random reversing and transposing of the first and second image dimensions (i.e. flips/transpose only); **dropout layers at the end of the contracting path** as implicit augmentation/regularization; **batch normalization applied after convolution and before activation** (p. 9).
- **Framework**: Keras on TensorFlow 2 (p. 9). Code: https://github.com/deleo-lab/schisto-vegetation
- **Baseline**: Random Forest (n = 100 trees, no max depth / leaf-node / feature limits) on **six GLCM texture features** (contrast, dissimilarity, homogeneity, energy, correlation, angular second moment) computed per band with scikit-image 0.15, same 256 × 256 × 8 inputs and same splits (p. 10). They additionally trained a **U-Net + GLCM** variant as an ablation.

## Data
- **Satellite**: **WorldView-2** (Maxar/DigitalGlobe Foundation), **8 multispectral bands** — Coastal 397–454, Blue 445–517, Green 507–586, Yellow 580–629, Red 626–696, Red Edge 698–749, NIR1 765–899, NIR2 857–1039 nm (Table 1, p. 6) — at **2.0 m** pixel resolution, acquired June–October 2016 (36 mosaics; ~90% cloudless). **10 cloud-free mosaics** covering the 32 water-access sites were used, acquired **29 June 2016** (river sites) and **4 July 2016** (lake sites). Each mosaic is **4096 × 4096 × 8** (p. 6).
- **Drone**: >6000 RGB images, DJI Phantom IV, 12.4 MP, **GSD 2.31 cm/pixel**, ~50 m AGL, covering 15 km² (p. 5). Used **only to visually validate** vegetation identity in the satellite imagery, not as model input.
- **Study area**: lower Senegal River Basin — 32 water-access sites across 16 villages, 10 sites near Saint-Louis on the river and 22 along Lac de Guiers (170 km² lake), between 15.898277–16.317924°N and 15.767996–16.431078°W. Field missions 2016–2019 (p. 4).
- **Patches**: ⚠️ **Internal inconsistency.** p. 6 says "we split each mosaic into **256 sub-images**, each of 256 × 256 × 8"; p. 7 says "we split each image mosaic into **16 sub-images** of size 256 × 256 × 8". 4096/256 = 16 per side ⇒ 256 tiles per mosaic, so the "16" on p. 7 appears to be the error (or refers to 16 selected tiles).
- **Labelled set**: technicians manually annotated **50 sub-images**, producing **4 binary masks each (50 × 4 masks)**; masks are non-overlapping (p. 8). Annotation cost: **~200 human hours** (p. 12).
- **Split**: **80% of the 50 × 4 masks as training/test, 20% as hold-out validation / out-of-bag (OOB)** (p. 8). Within training, 80% train / 20% test, with **the test set randomly re-selected at each epoch** (p. 9) — ⚠️ this makes the "test set" a rolling internal split, not a clean held-out test; the OOB set is the only true generalization estimate.
- **Class balance** (training set, p. 8): 86% of pixel area water or land, **4% floating vegetation**, **10% emergent vegetation**.
- **Preprocessing**: radiometric + geometric calibration done by DigitalGlobe (relative radiometric response, non-responsive detector fill, absolute radiometry; optical/scan/line-rate distortion, band misregistration); projected to **UTM 28N / WGS84**; pixels stored as INT2U digital numbers spanning 0–65,534 (p. 7). **No atmospheric correction to surface reflectance is described**, and **no explicit normalization/standardization step is reported**.
- **Spectral indices**: ⚠️ **NONE.** The paper uses **no NDVI, no NDWI, no index of any kind** — raw 8-band DNs are fed to the network. GLCM textures are the only engineered features, and only for the RF baseline / ablation.
- **Cloud masking**: handled by hand-selecting 10 cloud-free mosaics out of 36; no algorithmic cloud mask.

## Results
All numbers are **pixel accuracy** (Eq. 1). Table 2, p. 12:

| Model | GLCM features | Test set — 4 classes | Hold-out (OOB) — 4 classes | Test set — floating veg. | Hold-out (OOB) — floating veg. |
|---|---|---|---|---|---|
| Random forest | Added | 96.7% | **67.8%** | 97.0% | **68.0%** |
| CNN (U-Net) | Not added | 94.5% | 82.7% | 96.0% | 84.0% |
| CNN (U-Net) | Added | **96.5%** | **83.1%** | **97.0%** | 84.0% |

- Headline (p. 11): U-Net without GLCM = **94.5% test / 82.7% hold-out** over 4 classes; **96% / 84%** for the floating-vegetation class alone.
- **The comparison that matters**: on the hold-out/OOB regions not used in training, U-Net beats Random Forest by **~15 percentage points** (82.7% vs 67.8% over 4 classes; 84.0% vs 68.0% on floating vegetation). RF wins on the (leaky, re-sampled) test set — a textbook overfitting/generalization gap.
- **GLCM ablation**: adding GLCM layers to U-Net improved hold-out accuracy only marginally (82.7% → 83.1%), which the authors read as evidence that "deep learning does not heavily depend on the statistical features of the underlying image" (p. 12).
- **Error structure**: most false positives on floating vegetation lie **along patch edges and at boundaries with other vegetation** (p. 13, Figure 7c,d).
- No IoU, F1, precision/recall, kappa, or confusion matrix is reported. No comparison against any other CNN (no DeepLab, SegNet, FCN-8s, etc.) — RF+GLCM is the **only** baseline.

## Limitations
Stated by the authors:
- Pixel accuracy "can sometimes provide misleading results when the class representation is very small within the image (<1%)"; they argue this is not a concern here because the minority class reached ~97% (p. 12) — ⚠️ but the floating class is only **4%** of pixels, so a global pixel-accuracy figure is still weak evidence, and no IoU is given to check.
- Deep learning interpretability is poor relative to tree-based methods (p. 12).
- The model was trained and validated only in the lower SRB; generalization to other basins is untested (p. 13).
- The vegetation maps have **not yet been validated against actual snail counts or human infection/prevalence data** — that is explicitly left to future work (p. 13).
- Manual annotation cost ~200 human hours (p. 12).

Observed by me:
- **Very small labelled dataset**: 50 annotated sub-images total; the hold-out set is therefore ~10 images.
- **The "test set" is re-randomized every epoch** (p. 9) — it is not a held-out test set in the usual sense, which inflates the test-set column of Table 2 for both models.
- Labels come from **visual interpretation by technicians**, not from in-situ quadrats — label noise is unquantified.
- **Single date, single season** (late June / early July 2016). All the talk of "seasonally updated" risk maps is aspiration, not demonstrated.
- The abstract/conclusion mention "estimates of uncertainty" (p. 13) but **no uncertainty quantification is actually reported** — only that each pixel comes with a softmax/sigmoid probability.

## Relevance to this thesis
Honest assessment: **this paper is NOT about floods and contributes nothing on inundation, susceptibility, or hydrology.** Its value to the thesis is purely **methodological**, as a U-Net-on-multispectral-satellite-imagery template and as a rhetorical citation. Do not cite it as flood literature.

What can actually be borrowed:
- **Architecture template**: the vanilla U-Net with **an input depth equal to the number of bands** (8 here) is a direct precedent for feeding the thesis's B02/B03/B04/B08 (+ NDVI/NDWI as extra channels) as a multi-channel tensor rather than an RGB image. Cite as prior art for "classical U-Net, unmodified, works on N-band satellite input."
- **Hyperparameters as a starting point**: batch 8, 100 epochs, LR 4e-5, BN after conv/before activation, dropout at the end of the encoder, flip/transpose augmentation. Reasonable defaults to initialize the thesis's PyTorch training loop (the paper is Keras/TF2, so this is a port, not a reuse).
- **The generalization argument (the single most citable result)**: RF+GLCM ties or beats U-Net in-distribution but loses ~15 pp on unseen regions. If the thesis proposes a CNN over classical ML for flood mapping in Peru, this is a clean, quantitative citation for *why* — and it argues for evaluating on **spatially held-out areas**, not random pixel/patch splits. Worth replicating that experimental design (train on some river reaches, test on others).
- **Cautionary methodology**: (a) their "test set" is re-randomized each epoch — the thesis must not do that; (b) they report **only pixel accuracy on a 4%-minority class**. The thesis should report **IoU / F1 / precision–recall for the water/flood class**, precisely the metrics this paper is missing. Use it as a negative example in the methodology chapter.
- **Class-imbalance parallel**: floating vegetation is 4% of pixels — the same imbalance regime as flooded pixels in a Peruvian scene. They did nothing about it (no weighted/Dice/focal loss). The thesis can improve on this and say so.

Where it differs (do not paper over this):
- **Sensor**: WorldView-2, 8 bands, **2 m**, commercial — not Sentinel-2, 10 m, free. Their class boundaries are resolvable at 2 m and would likely be lost at 10 m.
- **No spectral indices at all** — no NDVI, no NDWI. The paper cannot be cited to justify the thesis's index choices.
- **No temporal dimension** — a single mosaic date. It offers nothing for the NDVI/NDWI **time-series** component of the thesis.
- **Task**: vegetation-class segmentation for disease-risk proxying, not water extent / flood extent.
- Framework is Keras/TensorFlow, not PyTorch.

Classification: **method to cite (U-Net on multispectral EO + CNN-vs-RF generalization argument)** and **cautionary example (metric choice, leaky test split)**. **Not** a baseline, **not** a dataset source, **not** flood literature.

## Keywords / Tech
- **Models**: U-Net (classical Ronneberger encoder–decoder, 23 conv layers), FCN, Random Forest (100 trees), CNN
- **Sensors**: WorldView-2 (8 bands, 2 m); DJI Phantom IV consumer drone (RGB, 2.31 cm GSD)
- **Indices**: none (no NDVI/NDWI); **GLCM texture features** (contrast, dissimilarity, homogeneity, energy, correlation, angular second moment)
- **Frameworks/libraries**: Keras, TensorFlow 2, scikit-image 0.15, Python
- **Techniques**: semantic segmentation, binary cross-entropy loss, batch normalization, dropout, flip/transpose augmentation, grid-search hyperparameter tuning, hold-out/OOB spatial validation, pixel accuracy
- **Data/code**: https://github.com/deleo-lab/schisto-vegetation
- **Domain**: schistosomiasis, *Bulinus* snails, *Ceratophyllum*/*Potamogeton*/*Typha*/*Phragmites*/*Nymphaea*, Senegal River Basin, Lac de Guiers

## Notable quotes
- "We found the optimized batch size to be 8, training epoch to be 100, and learning rate to be 4 × 10⁻⁵, which resulted in a segmentation accuracy of 94.5% for the test set and 82.7% for the hold-out validation/OOB set, over all four classes." (p. 11)
- "Our finding that the CNN (U-Net) model performed better than random forest approaches in the OOB dataset confirms previous findings that tree-based methods are less generalizable than deep learning segmentation methods." (p. 12)
- "Adding GLCM texture layers to the U-Net model only marginally improved accuracy, which shows that deep learning does not heavily depend on the statistical features of the underlying image. Consequently, the CNN model does not require prior specification of GLCM layers by a domain expert, which makes the model less prone to bias due to researcher-specified inputs." (p. 12)
- "The model evaluation metric selected here is pixel accuracy, which can sometimes provide misleading results when the class representation is very small within the image (<1%), as the measure will primarily be biased in reporting the model's ability to identify negative cases." (p. 12)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/YSWCNIQI/Liu et al. - 2022 - Deep Learning Segmentation of Satellite Imagery Identifies Aquatic Vegetation Associated with Snail.pdf`
Cite as: `\cite{liu_deep_2022}`
