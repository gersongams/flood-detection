---
title: "Estimating Rainfall Intensity Using an Image-Based Deep Learning Model"
authors: "Hang Yin, Feifei Zheng, Huan-Feng Duan, Dragan Savic, Zoran Kapelan"
year: 2023
venue: "Engineering, Vol. 21 (2023), pp. 162-174 (Research / Hydraulic Engineering—Article; published online 27 Jan 2022)"
doi: "10.1016/j.eng.2021.11.021"
bibtex_key: yin_estimating_2023
zotero_pdf: "/Users/gersongarrido/Zotero/storage/94QF4FQQ/Yin et al. - 2023 - Estimating Rainfall Intensity Using an Image-Based Deep Learning Model.pdf"
pages: 13
sensors: [none, smartphone-camera, surveillance-camera, rain-gauge]
task: "rainfall intensity estimation from ground-level images (regression)"
model: "irCNN — ResNet34-based CNN (38 layers) with a linear regression head"
tags: [rainfall-intensity, urban-flooding, cnn, resnet34, regression, crowdsourcing, transfer-learning, imagenet, ground-cameras, nowcasting-input]
---

# Estimating Rainfall Intensity Using an Image-Based Deep Learning Model

> **TL;DR** — The authors build "irCNN", a ResNet34-based CNN with a linear regression head (38 layers total, 6.3 × 10⁷ parameters, p. 165), that regresses **rainfall intensity (mm·h⁻¹) directly from a single ground-level photograph** of a rainy scene. It is pretrained on ImageNet, then trained on (a) synthetic rain images made in Photoshop by superimposing raindrop "noise layers" on 4000 background photos and (b) real rain images from smartphones and one surveillance camera on the Zhejiang University campus, labelled with rainfall intensity linearly interpolated from a 1-min tipping-bucket gauge. On real images the model reaches **MAPE 13.5%–21.9% (mean 16.5%)** (p. 172), beating the state-of-the-art decomposition-based method of Jiang et al. (21.8% MAPE) and running ~20× faster (~1 s vs 26.4 s per 100 images, p. 172). **This is not a satellite paper and not a flood-mapping paper** — it is a sensing paper about acquiring high-spatiotemporal-resolution rainfall *inputs* for urban flood warning.

## Problem & Motivation

Real-time urban flood warning systems need rainfall data at very high spatiotemporal resolution — the authors state the target as **1-min temporal resolution and ~100 m × 100 m spatial resolution across a city** (p. 163). Existing sources fall short: ground gauges are accurate but spatially sparse; weather radar has high temporal resolution but accuracy is degraded by uneven vertical rainfall distribution, anomalous propagation and tall buildings, and radar coverage is thin in many countries; satellite remote sensing gives broad spatial coverage but "its spatiotemporal resolution is often insufficient at the urban scale" (p. 163). Crowdsourcing alternatives (smart wipers on cars, acoustic umbrellas) are hard to deploy at scale.

The gap the paper attacks: rainfall images are already everywhere in cities (transportation cameras, security cameras, smartphones) and are densely distributed in space and time, so if a model could read rainfall intensity off an image, high-resolution urban rainfall fields could be obtained essentially for free. The prior state of the art is Jiang et al. (Ref. [34]), a convex-optimization / geometric-optics decomposition of rainy *videos*; the authors argue a CNN is faster, works on **still images** as well as video, and needs no hand-designed optics model. They claim (p. 163) this is "the first work in which CNNs have been adapted to model rainfall with high spatiotemporal resolution based on urban sensors."

## Method / Architecture

Core assumption (p. 164): rainfall intensity is a function of raindrop **density** and **size** in the image, `I = f(Z(d,s))` (Eq. 1), where `I` is intensity (mm·h⁻¹), `Z` the rain image, `d` raindrop density, `s` raindrop size; `f` is learned by the CNN.

- **Backbone**: ResNet34 (He et al., Ref. [42]). The irCNN has **38 layers** including input and output layers (p. 164). Layer 2 (L = 2) applies **64 kernels of 7 × 7 with stride s = 2** (combined convolution + subsampling); the remaining **29 convolutional layers use 3 × 3 kernels** with feature maps growing **64 → 128 → 256 → 512** (Fig. 3, p. 165).
- **Subsampling**: two dedicated subsampling layers, **L = 3 (max pooling, 3 × 3)** and **L = 36 (average pooling, 3 × 3)** (p. 165). Additional stride-2 convolutional layers at L = 10, 18, 30 also do subsampling.
- **Residual shortcuts**: identity shortcut connections skipping two layers, per He et al., to avoid the degradation/premature-convergence problem (p. 165).
- **Regression head**: a **linear regression layer at L = 37**, `Î = Wᵀ X + b` (Eq. 2, p. 165), replacing the usual classification softmax — this is the key adaptation from classification CNN to continuous rainfall intensity.
- **Parameters**: "a total of 6.3 × 10⁷ parameters for calibration" (p. 165).
- **Pretraining**: the model is first pretrained on **ImageNet (1000 classes, 1.28 million images)** for classification, then fine-tuned on rainfall images (p. 165). Classic transfer learning; the justification given is that low-level features are shared across object types.
- **Optimizer**: **stochastic gradient descent (SGD)** with the **cyclical learning rate (CLR)** / super-convergence approach of Smith & Topin (Ref. [53]) (p. 168).
- **Loss function**: not explicitly named. The text only says "the minimization of the training loss is the objective function, as defined by Smith and Topin [53]" (p. 169). ⚠️ For a regression task the loss is almost certainly MSE/L1 but the paper **does not state it** — do not cite a specific loss.
- **Batch size**: not reported. **Learning rate values**: not reported (only "cyclical"). **Data augmentation**: not reported (the synthetic-rain generation is arguably an augmentation scheme, but no flips/crops/rotations are described).
- **Epochs**: not fixed a priori; convergence observed at **10–50 epochs on synthetic data** and "significantly larger" numbers on real data — Fig. 8(b) shows ~100 epochs (p. 169).
- **Hardware / runtime**: Python; Intel i9-9820X @ 3.3 GHz, 32 GB RAM, **NVIDIA RTX 2080Ti 11 GB GPU** (p. 169). Each epoch on synthetic data takes 10–20 min; on real data ~3 min per epoch; all runs converge **within 3 h** (p. 169). Inference: **1–2 s for 100 images** (p. 169).

Pipeline: model setup (add regression layer) → ImageNet pretraining → rainfall data acquisition (synthetic + smartphone + surveillance) → fine-tuning with SGD/CLR → validation on held-out images and on held-out *rainfall events*.

## Data

**No satellite, no radar imagery is used.** The "sensors" are ordinary cameras plus one rain gauge for labels.

**1. Synthetic rain images (p. 165-166).**
- 4000 publicly available rain-free background images (from Ref. [51], Zhang & Patel's de-raining dataset) as background layers; **Photoshop CC2017** adds a raindrop "noise layer".
- Synthetic intensity label (Eq. 3, p. 166): `SI = 100d × (1/16)s²`, dimensionless, where `d` = raindrop density (ratio of noise pixels to total pixels, restricted to **10%–19% in steps of 1%**, giving **10 distinct SI values, SI = 10…19**) and `s` = raindrop size (ratio of raindrop-layer area to background-layer area; values 350%, 400%, 450% tried; **s = 400% is the default**, chosen as visually closest to real raindrops).
- **SD1** (background-diversity experiment): validation sub-dataset = **1000 images** (100 backgrounds × 10 SI values); training sub-datasets sweep **N = 100 → 1200 backgrounds in steps of 100** (e.g. N = 500 → 5000 training images). Validation backgrounds are disjoint from training backgrounds (p. 166).
- **SD2** (rainfall-diversity experiment): backgrounds fixed at **800**; the number of distinct SI values in training sweeps from **2 to 10**; same validation sub-dataset as SD1 (p. 166).

**2. Real smartphone images (p. 166-167).**
- Campus of **Zhejiang University, Hangzhou, China**; **11 rainfall events between May and July 2020**; **960 images** with different backgrounds; **768 (80%) train / 192 (20%) validation**, randomly split (p. 167).
- Labels: a **tipping-bucket rain gauge, 1-min resolution, 0.1 mm precision** (p. 166). Because the photo exposure (~1/200 s) is far shorter than the 1-min accumulation window, intensity at photo time `t` is obtained by **linear interpolation** between the two adjacent minute intensities (Eq. 4, p. 166): `I_t = I_L + ((t − t_L)/Δt)(I_R − I_L)`, Δt = 1 min.

**3. Real surveillance-camera images (p. 167, Table 4 p. 171).**
- One fixed in-situ camera, **six rainfall events in June–July 2020**; videos split into frames at **1 s resolution**, yielding **7117 frames**.
- **CD1**: random split, **5694 (80%) train / 1423 (20%) test**.
- **CD2**: leave-one-event-out — 5 events train, 1 event validate.
- Events (Table 4, p. 171): durations 12–69 min; average intensities 11.0–23.6 mm·h⁻¹; maxima 36.0–66.0 mm·h⁻¹. Following Ref. [14], only rainfall > 0.1 mm·min⁻¹ (= 6 mm·h⁻¹) is used.

**Preprocessing**: no atmospheric correction, no cloud masking, no speckle filtering — irrelevant here. Images are pixel matrices "approximately size normalized" (p. 164); exact input resolution is **not reported**. **No spectral indices are used** (RGB photographs only; NDVI/NDWI do not appear anywhere in this paper).

**Metrics** (Eqs. 5-9, p. 169): MAE, MAPE, R², Nash–Sutcliffe efficiency (NSE), Kling–Gupta efficiency (KGE).

## Results

**Synthetic data — SD1, effect of background diversity (Table 1, p. 167; averages over 5 runs, validation):**

| # backgrounds | MAE | MAPE (%) | R² | NSE | KGE |
|---|---|---|---|---|---|
| 100 | 1.07 | 7.67 | 0.77 | 0.75 | 0.87 |
| 300 | 0.62 | 4.47 | 0.91 | 0.91 | 0.90 |
| 600 | 0.56 | 4.01 | 0.91 | 0.91 | 0.96 |
| 800 | 0.54 | 3.81 | 0.94 | 0.94 | 0.96 |
| 1200 | 0.52 | 3.68 | 0.94 | 0.94 | 0.96 |

Performance improves sharply from 100 → 600 backgrounds then plateaus (p. 170).

**Synthetic data — SD2, effect of rainfall-intensity diversity (Table 2, p. 167):** with only **2 SI values** in training, average MAE 1.66 / MAPE 11.96% / R² 0.79 / NSE 0.49 / KGE 0.43; with **9 SI values**, average MAE 0.54 / MAPE 3.75% / R² 0.94 / NSE 0.93 / KGE 0.94 (p. 170). Conclusion: the model **interpolates but does not extrapolate** beyond the intensity range seen in training.

**Real smartphone images (Table 3, p. 170), 5 random splits, validation:** average **MAE 3.79 mm·h⁻¹, MAPE 18.53%, R² 0.96, NSE 0.95, KGE 0.91**. Per-trial MAPE ranges 16.30%–21.08%.

**Real surveillance camera, random frame split CD1 (Table 5, p. 171), 5 runs:** average **MAE 3.10 mm·h⁻¹, MAPE 16.54%, R² 0.92, NSE 0.92, KGE 0.95**.

**Real surveillance camera, held-out-event split CD2 (Table 6, p. 171):**

| Validation event | MAE (mm·h⁻¹) | MAPE (%) | R² | NSE | KGE |
|---|---|---|---|---|---|
| Event 1 | 2.40 | 18.55 | 0.93 | 0.93 | 0.94 |
| Event 4 | 4.35 | 21.90 | 0.69 | 0.60 | 0.80 |
| Average | 3.78 | 20.23 | 0.81 | 0.76 | 0.87 |

This is the honest, harder split, and performance degrades (R² drops from 0.92 to 0.81; on event 4, to 0.69) — attributed to inter-event environmental variation (brightness, wind) (p. 171).

**Post-processing improvement (p. 172):** averaging the irCNN estimates over all camera frames within each 1-min window (matching the gauge's native resolution) improves the held-out-event result to **MAE 2.55 mm·h⁻¹ and MAPE 13.5%**.

**Baseline comparison.** The only external baseline is **Jiang et al. [34]** (decomposition-based identification from rainfall video), quoted at **21.80% MAPE**; the irCNN's worst case is **21.90% MAPE** and its mean is **16.5%** (p. 172). ⚠️ Note the comparison is **not a re-implementation on the same data** — the 21.8% figure is quoted from the other paper's own reported value, so this is an indirect, non-controlled comparison. Speed: irCNN ~1 s vs Jiang et al. 26.4 s for 100 images ("about 20 times faster", p. 172). **No ablation against other CNN backbones (AlexNet, VGG16) was actually run** — ResNet34 was chosen by citing He et al.'s results, not by experiment here.

⚠️ Minor inconsistency: the abstract says "mean absolute percentage error ranging between 13.5% and 21.9%" (p. 162) — but 13.5% is the *post-processed 1-min-averaged* figure while 21.9% is a raw per-frame worst case; these come from different evaluation protocols and mixing them into one range is slightly misleading. The Section 6 summary states the range with "a mean of 16.5%" (p. 172), which is the mean of the *raw* real-image results.

## Limitations

Stated by the authors (p. 173):
- Availability of paired rainfall images + intensity labels from many sensors is the bottleneck.
- Transmission efficiency of images from distributed sensors to a processing centre in near-real-time is unaddressed.
- Image quality under varying conditions — **daytime vs night, camera position (e.g. under trees), season** — degrades performance and is not handled.
- The linear interpolation used to label images (Eq. 4) "inevitably induces errors" (p. 171).
- Uncertainty of the method is not quantified; no comprehensive controlled comparison against other rainfall-measurement models.
- Model performance was only validated on **rainfall extremes of relatively short duration** (June–July storm bursts in Hangzhou); "future research should also validate the performance of the irCNN model in estimating rainfall intensity for average rainfall events (i.e., low-intensity events with a long duration)" (p. 170).

Observed by me:
- **Single study area, single climate, single camera** for the surveillance experiments — the authors themselves note "this study uses only one camera with a fixed angle" (p. 171). Generalisation to a new camera/city is untested.
- Very small event count (6 camera events, 11 smartphone events); the leave-one-event-out test has **n = 2 held-out events**, which is a thin basis for the reported averages.
- The synthetic SI scale (Eq. 3) is **dimensionless** and never physically calibrated against mm·h⁻¹; the synthetic experiments therefore cannot transfer numerically to the real task, only qualitatively (they answer "how much diversity do I need", not "how accurate is it").
- The loss function is never stated (see Method).
- No uncertainty estimates / prediction intervals — a real problem for an input feeding a flood warning system.

## Relevance to this thesis

**Be blunt: this paper is peripheral to the core of the thesis.** It uses no satellite data, no Sentinel-2, no spectral indices, no segmentation, and it does not map flooding. It is a *sensing* paper about getting rainfall inputs for urban flood models. Do not let it into the methodological core; it belongs in the introduction/state-of-the-art as context.

What is genuinely usable:

- **Cite it in the "predict"/forecasting half of the framing.** The thesis promises to *predict* flood-prone areas, not just map them. This paper is a clean, citable example of the fact that **rainfall forcing is the dominant driver of flash flooding** (p. 162) and that acquiring it at useful resolution is the bottleneck — a good motivating citation for why a purely optical, snapshot-based flood map is insufficient for prediction. It is also a useful **citation for the limits of satellite rainfall products**: "the satellite remote sensing approach can provide rainfall prediction at a large spatial coverage, but its spatiotemporal resolution is often insufficient at the urban scale" (p. 163) — directly relevant if you ever consider GPM/IMERG as a rainfall covariate for Peru.
- **Transfer learning from ImageNet as a documented, working practice.** The thesis CNN will be trained on a small NDVI/NDWI dataset. This paper is a concrete precedent for pretraining a ResNet backbone on ImageNet and fine-tuning on a domain task, and it reports a specific quantitative lesson: **performance saturates once training diversity is sufficient (≈600 backgrounds, Table 1) and collapses when training diversity is low (2 SI values → NSE 0.49, Table 2)**. Cite this as evidence for why the thesis needs geographically/temporally diverse training tiles, not just many patches from one scene. Caveat: ImageNet pretraining transfers to 3-channel RGB; NDVI/NDWI stacks are not RGB, so the transfer is not free — this is a *cautionary* borrowing, not a drop-in.
- **Borrow the regression-head trick if the thesis ever regresses a continuous target.** If flood *susceptibility* is modelled as a continuous score rather than a binary mask, the pattern (ResNet backbone → linear layer, Eq. 2, p. 165) with MAE/R²/NSE/KGE as metrics is directly reusable. **NSE and KGE are hydrology-standard metrics** the thesis committee will recognise; the equations are given verbatim (Eqs. 7-9, p. 169).
- **Borrow the evaluation-split discipline — this is the strongest methodological takeaway.** The paper reports both a random split (R² 0.92) *and* a held-out-**event** split (R² 0.81, and 0.69 on one event) and is honest that the random split is optimistic (p. 171). The thesis has exactly the same trap: randomly splitting patches from the same Sentinel-2 scene will leak, and a **held-out-flood-event / held-out-region split is the honest evaluation**. Cite this as precedent for reporting both.
- **Cautionary example on extrapolation.** Table 2 shows the model cannot estimate intensities outside its training range (p. 170). Same warning applies to a CNN trained only on moderate flood extents in Peru: it will not extrapolate to an extreme El Niño event it never saw.
- **Not a baseline. Not a dataset source. Not directly comparable** — there is no IoU, F1, OA or kappa in this paper, and its metrics (mm·h⁻¹ MAE) have no counterpart in a flood-extent segmentation task.
- Possible future-work sentence for the thesis: coupling a satellite-derived flood-susceptibility map with high-resolution rainfall from urban cameras/crowdsourcing (as here) is the natural completion of a real-time warning system; the Peruvian context (dense cities, sparse gauge networks, e.g. Piura/Lima) makes the low-cost argument attractive. Frame it as future work, not as something the thesis does.

## Keywords / Tech

- **Models**: irCNN (ResNet34-based, 38 layers, 6.3 × 10⁷ params, linear regression head); residual/shortcut connections; ImageNet pretraining; baseline referenced (not re-run): Jiang et al. decomposition-based identification.
- **Sensors**: smartphone cameras, fixed surveillance camera, tipping-bucket rain gauge (1 min, 0.1 mm). **No satellite, no radar.**
- **Indices**: none (no NDVI/NDWI). Synthetic intensity `SI = 100d × (1/16)s²`.
- **Training**: SGD + cyclical learning rate (super-convergence, Smith & Topin); Python; NVIDIA RTX 2080Ti; < 3 h per run.
- **Datasets**: ImageNet (pretraining); Zhang & Patel de-raining backgrounds (Ref. [51]); Photoshop CC2017-generated synthetic rain (SD1, SD2); private Zhejiang University campus smartphone (960 imgs) and camera (7117 frames) sets — **the real datasets are not public**.
- **Metrics**: MAE, MAPE, R², NSE, KGE.
- **Techniques**: synthetic data generation as augmentation, transfer learning, linear temporal interpolation of gauge labels, leave-one-event-out validation, temporal averaging of frame-level predictions to gauge resolution.

## Notable quotes

> "In contrast, the satellite remote sensing approach can provide rainfall prediction at a large spatial coverage, but its spatiotemporal resolution is often insufficient at the urban scale." (p. 163)

> "To the best of our knowledge, this is the first work in which CNNs have been adapted to model rainfall with high spatiotemporal resolution based on urban sensors." (p. 163)

> "According to Table 2, for a fixed set of SI values, if the selected alternatives can cover a large span of the total options, the performance of the irCNN model improves. This finding indicates that the irCNN model may not be able to provide accurate estimates for scenarios with rainfall intensities beyond those provided in the training dataset. This limitation is typical for most machine learning methods, as they tend to perform much better at interpolating than extrapolating beyond the dataset used for their training." (p. 170)

> "The results based on real rainfall images show that the irCNN model provided rainfall estimates with an MAPE ranging between 13.5%–21.9% (with a mean of 16.5%). This average performance exceeds the corresponding accuracy (21.8% MAPE) of the decomposition-based identification algorithm [34], which is currently the state-of-the-art modeling technique." (p. 172)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/94QF4FQQ/Yin et al. - 2023 - Estimating Rainfall Intensity Using an Image-Based Deep Learning Model.pdf`
Cite as: `\cite{yin_estimating_2023}`
