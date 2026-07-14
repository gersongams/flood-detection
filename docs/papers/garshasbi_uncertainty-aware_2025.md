---
title: "Uncertainty-Aware Flood Inundation Mapping With a Bayesian Deep Learning Framework Using SAR Imagery"
authors: "Mahyar Garshasbi, Hosein Alizadeh, Barat Mojaradi, Motahareh Saadatpour, Erfan Zarei"
year: 2025
venue: "IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), vol. 18, 2025, pp. 26716–26726"
doi: "10.1109/JSTARS.2025.3610403"
bibtex_key: garshasbi_uncertainty-aware_2025
zotero_pdf: "/Users/gersongarrido/Zotero/storage/YEUB4CFM/Garshasbi et al. - 2025 - Uncertainty-Aware Flood Inundation Mapping With a Bayesian Deep Learning Framework Using SAR Imagery.pdf"
pages: 11
sensors: [Sentinel-1, Sentinel-2]
task: "flood inundation mapping (binary flood/non-flood segmentation) with pixel-wise uncertainty quantification"
model: "BHED-U-Net — Bayesian Hybrid Encoder-Decoder U-Net (Bayesian variational conv layers + standard conv layers)"
tags: [bayesian-deep-learning, variational-inference, uncertainty-quantification, sar, sentinel-1, u-net, sen1floods11, flood-mapping, bayes-by-backprop, pytorch]
---

# Uncertainty-Aware Flood Inundation Mapping With a Bayesian Deep Learning Framework Using SAR Imagery

> **TL;DR** — The authors propose **BHED-U-Net**, a U-Net where the *second* convolution of every encoder/decoder stage is replaced by a **Bayesian variational convolution** (weights are Gaussian distributions `N(μ, σ)`, trained with the reparameterization trick + Bayes-by-Backprop, loss = `α·BCE + β·KL`). Trained on Sentinel-1 VV+VH (Sen1Floods11, 446 patches of 512×512 at 10 m), it outputs a **probabilistic** flood map: at inference it draws **100 stochastic forward passes**, fits a Gaussian per pixel, and reports the **mean as flood probability** and the **standard deviation σ as the pixel-wise uncertainty**. On the Sen1Floods11 test set it reaches **Accuracy 0.9587, F1 0.8013, IoU 0.6685** vs standard U-Net's **0.9550 / 0.7672 / 0.6224** (p. 26721, Table II); on an unseen 2019 Golestan (Iran) flood it holds **Accuracy 0.925, F1 0.810, IoU 0.681** (p. 26722, Table III). Uncertainty is stratified by land cover: vegetation highest (σ ≈ 0.14), bare soil lowest (p. 26723).

## Problem & Motivation
Nearly all SAR flood-mapping methods (thresholding, change detection, supervised/unsupervised ML, deep learning) produce a **hard binary** flooded/non-flooded label, which "overlooks the inherent uncertainty in the mapping process" (p. 26717). SAR is intrinsically noisy: speckle, radar shadow/layover behind buildings, volume scattering from canopies, double-bounce from buildings, and smooth non-water surfaces (tarmac) that mimic water — all sources of **aleatoric (data) uncertainty**. Deep models additionally suffer **epistemic uncertainty** from limited knowledge of the full complexity of flood inundation.

The authors argue that flood managers and insurers need maps that state a **probability of flooding per pixel (0–1) plus a confidence estimate**, so that low-confidence regions can be flagged for verification instead of silently mis-mapped. Their gap-filling contribution is a Bayesian CNN hybridized with a standard U-Net that delivers exactly that, at 10 m, from Sentinel-1 alone (no ancillary DEM/rainfall inputs at inference).

## Method / Architecture

**Base**: standard U-Net (Ronneberger et al.) — contracting encoder / symmetric expanding decoder, each stage = **two consecutive CNNs**. Channel widths shown in Fig. 4(a): 32 → 64 → 128 → 256 → 512 (bottleneck), 2×2 max-pool down, 2×2 up-conv, copy-and-concatenate skips, final sigmoid.

**The "Hybrid" (H in BHED)**: in *each* encoder and decoder stage, the **first conv stays deterministic** (Conv2D + ReLU + BatchNorm) and the **second conv is replaced by a Bayesian variational conv layer** (Bayesian Conv2D + ReLU + BatchNorm). So Bayesian and deterministic layers alternate throughout depth.

**Bayesian layer internals (Fig. 4(b), Algorithm 1, p. 26720)** — this is the reimplementable core:
- Each filter weight is parameterized by **two learnable parameters, mean `μ` and standard deviation `σ`**, instead of one deterministic weight.
- **Prior**: standard Gaussian `N(0, 1)` on every weight (they explicitly state they "assumed a standard Gaussian prior", p. 26724).
- **Posterior**: `N(μ_i, σ_i)` per weight.
- **Forward pass = reparameterization trick** (Kingma & Welling): sample `ε ~ N(0,1)`, then `W^(i+1) = W^(i) + ε · σ^(i)`. Perform the forward pass with the sampled `W`.
- **Optimization = Bayes-by-Backprop** (Blundell et al. 2015): gradients `∇_W L` and `∇_σ L`, updates `W ← W − η·∇_W L` and `σ ← σ − η·∇_σ L`.

**Loss** (p. 26719–26720):
- `L_Total = α · L_BCE + β · L_KL`  (Eq. 1)
- `L_BCE = −(1/N) Σ_i [ y_i log P_i + (1 − y_i) log(1 − P_i) ]`  (Eq. 2), N = number of pixels.
- `D_KL(q_φ(z|x) ‖ p(z)) = ∫ q log(q/p) dx` (Eq. 3), which for Gaussian prior/posterior simplifies to the closed form (Eq. 4):
  `D_KL = 0.5 [ σ_Q²/σ_P² + (μ_Q − μ_P)²/σ_P² − 1 + log(σ_P²/σ_Q²) ]`
  (P = prior, Q = posterior). This is the ELBO / Bayesian variational inference (BVI) objective.
- **Weights chosen by iterative experimentation: `α = 1`, `β = 0.06`** (p. 26720). The heavy down-weighting of KL is important to replicate.
- ⚠️ Note Fig. 7(a): the BHED training loss curve sits far above the validation curve because **training loss includes the KL term while validation loss only includes BCE** (p. 26722) — an artifact, not overfitting-in-reverse.

**Inference / uncertainty quantification (Section III-C, p. 26720)** — the key recipe:
1. Sample from the trained posterior to generate an **ensemble of 100 stochastic forward passes** (each a sigmoid output mask). They justify 100: "the marginal utility of increasing sample outputs beyond 100 becomes negligible, while computational costs rise sharply."
2. For each pixel, **fit a Gaussian distribution** to its 100 sigmoid values.
3. **Mean of that Gaussian = flood probability**; threshold at **0.5** to obtain the binary flooded/non-flooded label.
4. **Standard deviation σ of that Gaussian = the pixel-wise uncertainty** — "the spread of the predicted probabilities for that pixel across the ensemble of output masks... the interval within which we are confident that the true probability of the pixel being flooded lies."
5. ⚠️ For the **quantitative metric tables**, they did NOT use the 100-sample ensemble — "the evaluation involved setting the weights of the BHED-U-Net to the **μ value of the posterior distribution**" (p. 26721), i.e. a deterministic mean-weight pass. The ensemble is used for the probability/uncertainty maps.

**Training hyperparameters (Section III-D, p. 26720)**:
- Framework: **Python + PyTorch**; hardware Google Colab Pro, **NVIDIA A100**.
- Input: **2 channels (VV, VH)**, tiles **512 × 512**.
- **Batch size 16**, **250 epochs**.
- **Initial LR 5 × 10⁻⁵**, **cosine annealing** decay, **Adam** optimizer.
- Params: **BHED-U-Net 11,693,185** vs deterministic U-Net **7,765,697**.
- **Augmentation: not reported.** No dropout is used (this is *not* MC dropout and *not* a deep ensemble — it is **variational inference / Bayes-by-Backprop**).

## Data

**Sensor**: Sentinel-1 A, **C-band SAR, GRD, Interferometric Wide (IW) swath**, dual-pol **VV + VH**, from Copernicus Data Space Ecosystem. Both ascending and descending orbits. **10 m** resolution. Sentinel-2 is used only to build the Golestan ground-truth mask (and Sen1Floods11's own labels derive from S2).

**Datasets (Table I, p. 26717)**:
- **Sen1Floods11** [ref 47, Bonafilia et al. 2020] — **11 global flood events**: Bolivia (2018/02/15), Ghana (2018/09/18), India (2016/08/12), Cambodia (2018/08/05), Nigeria (2018/09/21), Pakistan (2017/06/28), Paraguay (2018/10/31), Somalia (2018/05/07), Spain (2019/09/17), Sri Lanka (2017/05/30), USA (2019/05/22). **446 non-overlapping patches, 512 × 512 px at 10 m**, hand-labeled ground truth + paired post-flood S1 and S2. Split into **train / validation / test** — ⚠️ **exact split ratios / patch counts per split are not reported**.
- **Golestan flood (Iran, generalization test)** — torrential rain 17–22 March 2019, up to **315 mm in five days** (~4× March climatology); Aqqala County, Golestan Province. Inundation persisted 2–3 weeks. **Post-flood Sentinel-1: 2019/04/04 02:28:31 (descending, rel. orbit 137)**; paired **cloud-free Sentinel-2: 2019/04/05 07:11:31 (10 m)** used as the optical reference. Pre-flood S1 2019/01/05 02:20:14 and pre-flood S2 2019/01/13 07:11:31 shown for context. Ground-truth mask **manually interpreted** from the post-flood S2 image. A **land-cover layer** was produced by **Maximum Likelihood Classification (MLC)** on the pre-flood S2 imagery (classes: built-up, bare land, vegetation) for the uncertainty-by-land-cover analysis.

**Preprocessing (Section II-C, p. 26718; Fig. 3)** — standard S1 GRD workflow in **SNAP** [ref 48, Filipponi 2019]:
1. Apply orbit file
2. Thermal + border noise removal
3. Radiometric calibration
4. **Speckle filtering with the refined Lee filter** [Lee et al. 1999]
5. **Range-Doppler terrain correction with a 90 m DEM**
6. Conversion of backscatter to **decibels (logarithmic scaling)**; export GeoTIFF
7. In Python (**Rasterio**): VV and VH **clipped to (−50, 1) dB** and **normalized to [0, 1]**; tiled to 512²; stacked; split train/val/test.

**Spectral indices**: **none used** — the model is SAR-intensity-only. No NDVI/NDWI anywhere in the paper.

## Results

**Sen1Floods11 test set (Table II, p. 26721)** — BHED-U-Net vs the deterministic U-Net (the *only* baseline they actually trained):

| Metric | BHED-U-Net | Standard U-Net |
|---|---|---|
| Accuracy | **0.9587** | 0.9550 |
| Precision | 0.8646 | **0.9050** |
| Recall | **0.7467** | 0.6659 |
| F1-score | **0.8013** | 0.7672 |
| IoU | **0.6685** | 0.6224 |

BHED-U-Net wins on Accuracy (+0.0037), Recall (**+0.0808**), F1 (+0.0341), IoU (+0.0461); loses on Precision (−0.0404). Interpretation given: the Bayesian stochasticity acts as a regularizer, reducing false negatives (missed flooded regions) at the cost of some false positives (p. 26722). Per-country breakdown is only shown as spider plots (Fig. 6, p. 26722) — no per-country numbers in text.

**Golestan generalization (Table III, p. 26722)** — trained *exclusively* on Sen1Floods11, applied to Iran:

| Metric | Value |
|---|---|
| Accuracy | 0.925 |
| Precision | 0.865 |
| Recall | 0.762 |
| F1-score | 0.810 |
| IoU | 0.681 |

**Uncertainty by land cover (Fig. 9, p. 26723)** — pixel-wise σ for flooded pixels in Golestan, stratified by MLC land cover, with flooded-area extents annotated:
- **Vegetation**: highest average σ ≈ **0.14**; flooded area **4.31 km²** (heterogeneous volume scattering in VH, non-flood returns under canopy)
- **Built-up**: intermediate; flooded area **11.13 km²** (strong double-bounce helps, but layover/shadow hurts)
- **Bare land**: **lowest** average σ; flooded area **38.53 km²** (homogeneous backscatter once inundated)
- Boxplot whiskers span roughly σ ∈ [0, 0.39].

**Literature comparison (Table IV, p. 26723)** — ⚠️ **these are numbers quoted from other papers on other datasets, NOT re-runs**. Only Precision and Recall are compared:

| Model [ref] | Input | Precision | Recall |
|---|---|---|---|
| CLVAE [27] | S1 | 0.70 | 0.78 |
| U-Net-CBAM [41] | S1 | 0.83 | 0.92 |
| Urban-Aware U-Net [44] | S1 | 0.87 | 0.88 |
| WVResU-Net [16] | S1 | 0.93 | 0.70 |
| BiT [9] | S1 | 0.91 | 0.87 |
| A-SL CNN [39] | TerraSAR-X | 0.68 | 0.82 |
| **BHED-U-Net (This Study)** | S1 | **0.87** | **0.76** |

⚠️ **Inconsistency to flag**: Table IV lists BHED-U-Net at Precision 0.87 / Recall 0.76, which matches the **Golestan** numbers (0.865/0.762, Table III) rather than the Sen1Floods11 test numbers (0.8646/0.7467, Table II). The text does not say which dataset the Table IV row refers to. Cite the source table (II or III), not Table IV, for BHED numbers.

**Qualitative (Fig. 5, p. 26721)**: for 8 held-out scenes, the deterministic U-Net "almost completely omits" fragmented small water bodies (sites 2 & 3), while BHED-U-Net recovers them with **intermediate probabilities 0.4–0.7** along wet-soil fringes. In site 6 (Somalia, heavy radar layover/vegetated shadow), the U-Net suppresses nearly all output while BHED-U-Net emits a low-probability heterogeneous field flagging plausible inundation for verification.

## Limitations

**Stated by the authors (Section V-B, p. 26724)**:
- **Aleatoric and epistemic uncertainty are treated jointly, not separated** — the σ they report is a single lumped number; they name disentangling them as future work.
- **Prior is assumed standard Gaussian N(0,1)**; alternative Gaussian variants / non-Gaussian priors were not explored.
- **Ground truth is satellite-derived** (Sentinel-2 optical), so S2–S1 acquisition dates differ (here 04 Apr S1 vs 05 Apr S2) → temporal mismatch may depress reported Precision/Recall; no in-situ validation.
- SAR-only input; no DEM, impervious surface, rainfall, or interferometric coherence fusion.
- Only the U-Net backbone was Bayesianized; other architectures untested.
- C-band (S1) vs X-band (TerraSAR-X) differences confound the Table IV comparison.

**Observed by me**:
- **The only trained baseline is a vanilla U-Net.** No comparison against the many SOTA models in Table IV under matched conditions, and no comparison against other UQ approaches (MC dropout, deep ensembles) — so we cannot tell whether the gain comes from *Bayesian-ness* or simply from 1.5× more parameters (11.69 M vs 7.77 M).
- **Train/val/test split ratios of the 446 patches are never given**, and 446 patches is a small dataset. No cross-validation, no repeated seeds, no confidence intervals on metrics.
- The reported metrics use **mean-weight (μ) deterministic inference**, so the headline table does not actually evaluate the probabilistic ensemble.
- The uncertainty is **never calibrated** (no reliability diagram, no ECE, no "does σ predict error?" quantitative test) — it is only shown to correlate with land-cover class. That is a suggestive but weak validation of "well-calibrated," a word they nonetheless use (p. 26723).
- σ from a Gaussian fit to sigmoid outputs bounded in [0,1] is a slightly awkward estimator (the distribution is not Gaussian near 0 or 1); predictive entropy or the variance directly would be cleaner.

## Relevance to this thesis

**High relevance as a *method* paper, medium as a baseline, low as a data source.** It is SAR (Sentinel-1), the thesis is optical (Sentinel-2) — but the uncertainty machinery is **sensor-agnostic and directly portable**.

**Directly borrowable (this is the paper's gift to the thesis):**
- **The Bayesian variational conv layer is the whole recipe for "flood map + confidence" in PyTorch**, and it is small: for a `Conv2d`, register `weight_mu` and `weight_rho` (use `σ = softplus(ρ)` for positivity — the paper stores σ directly, which risks σ < 0; softplus is the standard fix), sample `W = μ + ε·σ` with `ε ~ N(0,1)` in `forward()`, and add the closed-form Gaussian KL (Eq. 4) against a `N(0,1)` prior to the loss. Reimplementable in <60 lines.
- **The hybrid trick**: do *not* Bayesianize every layer — only the **second conv of each U-Net stage**. This halves the extra parameters and, per their result, still yields the UQ benefit. Cheap and worth copying verbatim for a thesis-scale model.
- **Loss weighting `α = 1, β = 0.06`.** A concrete, citable starting point for the KL weight — the single hardest hyperparameter in BVI. Report it as "following \cite{garshasbi_uncertainty-aware_2025}".
- **Inference protocol**: 100 stochastic forward passes → per-pixel mean = flood probability, per-pixel std = uncertainty, threshold 0.5 for the binary map. Two rasters out of one model. Also copy their justification for N=100 (diminishing returns).
- **Optimizer/schedule**: Adam, LR 5e-5 with cosine annealing, batch 16, 250 epochs, 512×512 tiles — a sane default for a U-Net at 10 m with a small (~450-patch) dataset, which is likely the thesis's regime too.
- **The land-cover-stratified uncertainty analysis (Fig. 9)** is a cheap, high-value thesis figure: classify the pre-flood image into land-cover classes, then box-plot σ per class. For Sentinel-2 the thesis can do this even more easily (NDVI-based vegetation stratification), and the *expected story flips*: their vegetation uncertainty is high because of SAR volume scattering, whereas optical NDWI should be *most* confident over open water and least over turbid/sediment-laden flood water and cloud-shadow — a genuinely novel contrast to write about.
- **Metric set**: Precision, Recall, F1, Accuracy, IoU (Eqs. 5–9) — adopt exactly, so the thesis is numerically comparable to Table II/III.
- **Two-dataset design (train on a public global benchmark, then test generalization on one national event)** is exactly the shape the thesis should take: train on a public flood benchmark, then hold out a Peruvian event (e.g. Piura 2017 / coastal El Niño). This paper is the citation that legitimizes that protocol.

**Comparable / baseline value:**
- **Do not treat Table II as a beatable target** — different sensor and different labels. But **their U-Net-vs-Bayesian delta is the comparable quantity**: they gained **+0.034 F1 and +0.046 IoU** by going Bayesian. The thesis should report its own deterministic-vs-Bayesian delta and compare *that* against theirs.
- Their **Golestan drop** (F1 0.8013 → 0.810, IoU 0.6685 → 0.681 — essentially no drop, arguably a slight *rise*) is a useful reference point for what "good generalization to an unseen basin" looks like. ⚠️ Be a bit skeptical: the Golestan reference mask was hand-drawn by the same authors from S2, on a flat, uniform agricultural floodplain — an easier scene than the average Sen1Floods11 tile.

**Where it differs (state these explicitly in the thesis lit review):**
- **SAR vs optical**: no clouds for them; the thesis's ≤10% cloud filter + cloud masking is an extra failure mode they never face. Conversely they suffer speckle/layover/shadow that Sentinel-2 does not.
- **No spectral indices**: they feed raw dB backscatter, so NDVI/NDWI have no analogue here. The thesis's NDVI/NDWI time-series stack replaces their VV/VH 2-channel stack — the Bayesian U-Net accepts arbitrary `in_channels`, so the swap is trivial.
- **Single-date inundation mapping**, not **temporal susceptibility/prediction**. Their model answers "is this pixel flooded *now*?"; the thesis also wants "is this pixel flood-*prone*?". The UQ layer transfers, the task framing does not.
- **Ground truth from optical imagery**, which for the thesis is the *primary* imagery — so the thesis avoids their biggest stated limitation (S1/S2 temporal mismatch). Say so; it is a point in the thesis's favour.

**Verdict**: **a method-to-cite and a template to reimplement**, not a baseline to beat. It is the single strongest justification in the library for adding uncertainty output to the thesis's flood-risk map, and it supplies every hyperparameter needed to do so.

## Keywords / Tech
- **Models**: U-Net; BHED-U-Net (Bayesian Hybrid Encoder–Decoder U-Net); Bayesian CNN; Bayesian variational inference (BVI); Bayes-by-Backprop; reparameterization trick; ELBO
- **Sensors**: Sentinel-1 C-band SAR (GRD, IW, VV+VH); Sentinel-2 (ground truth only)
- **Indices**: none (raw dB backscatter)
- **Frameworks/libraries**: PyTorch; ESA SNAP; Rasterio; Google Colab Pro (NVIDIA A100)
- **Datasets**: Sen1Floods11 (446 × 512² patches, 11 events); Golestan/Aqqala Iran flood 2019 (custom)
- **Techniques**: refined Lee speckle filter; Range-Doppler terrain correction (90 m DEM); dB conversion; clip (−50, 1) dB → [0,1] normalization; BCE + KL loss (α=1, β=0.06); 100-sample posterior ensemble; per-pixel Gaussian fit → mean = probability, σ = uncertainty; Maximum Likelihood Classification for land cover
- **Metrics**: Precision, Recall, F1, Accuracy, IoU

## Notable quotes

> "Our proposed model utilizes the VV and VH polarization modes of Sentinel-1 imagery, which is less susceptible to weather and sunlight variations. This enables our model to deliver rapid, reliable flood extent maps at a 10-m spatial resolution." (p. 26716, Abstract)

> "these methods typically present a binary classification of flooded and non-flooded zones. This approach overlooks the inherent uncertainty in the mapping process. SAR data itself is noisy, introducing a source of uncertainty called aleatoric uncertainty or data uncertainty." (p. 26717)

> "To quantify the uncertainty associated with each pixel, we fit Gaussian distributions to the corresponding pixel—with a threshold of 0.5 to classify pixels as either flooded or nonflooded—across the 100 output masks... The uncertainty for each pixel is estimated by considering the standard deviation (σ) of the corresponding Gaussian distribution. This standard deviation represents the spread of the predicted probabilities for that pixel across the ensemble of output masks." (p. 26720)

> "the BHED-U-Net generates a heterogeneous probability field that flags plausible inundation with a low probability, thereby signaling to downstream users that further verification is needed." (p. 26721)

> "Vegetation exhibits the highest average σ (∼0.14), reflecting the heterogeneous volume scattering in VH polarization and the propensity for mixed flood/non-flood returns beneath forest canopies. Bare soil, in contrast, shows the lowest average uncertainty." (p. 26723)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/YEUB4CFM/Garshasbi et al. - 2025 - Uncertainty-Aware Flood Inundation Mapping With a Bayesian Deep Learning Framework Using SAR Imagery.pdf`
Cite as: `\cite{garshasbi_uncertainty-aware_2025}`
