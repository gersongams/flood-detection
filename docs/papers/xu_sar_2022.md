---
title: "SAR image water extraction using the attention U-net and multi-scale level set method: flood monitoring in South China in 2020 as a test case"
authors: "Chuan Xu, Shanshan Zhang, Bofei Zhao, Chang Liu, Haigang Sui, Wei Yang, Liye Mei"
year: 2022
venue: "Geo-spatial Information Science, Vol. 25, No. 2, pp. 155–168"
doi: "10.1080/10095020.2021.1978275"
bibtex_key: xu_sar_2022
zotero_pdf: "/Users/gersongarrido/Zotero/storage/Z58TV9IJ/Xu et al. - 2022 - SAR image water extraction using the attention U-net and multi-scale level set method flood monitor.pdf"
pages: 15
sensors: [Gaofen-3 (GF-3) SAR, COSMO-SkyMed SAR, Sentinel-2, DEM]
task: "flood inundation mapping / SAR water body extraction"
model: "Attention U-Net (CNN) + multi-scale level set (Chan-Vese / gamma model)"
tags: [sar, attention-u-net, level-set, water-extraction, flood-monitoring, dice-loss, speckle, chan-vese, dem-shadow-removal, pytorch]
---

# SAR image water extraction using the attention U-net and multi-scale level set method: flood monitoring in South China in 2020 as a test case

> **TL;DR** — The authors propose a two-stage SAR water-extraction pipeline: an **attention U-Net** produces a coarse binary water map at the *coarsest scale* of a multi-scale image pyramid, and that map is used to **initialize the zero-level set** of a gamma-model / Chan-Vese level set that is then evolved to snap to precise water edges; DEM + slope (Otsu-thresholded) are used to strip out radar shadow. The claim is that a CNN prior beats the traditional circular/rectangular level-set initialization: faster convergence, no local minima, better recovery of thin tributaries under speckle. It is demonstrated on the 2020 South China floods (Poyang Lake, Wangjiaba/Mengwa, Chongqing) with GF-3, COSMO-SkyMed and Sentinel-2 imagery. ⚠️ **The paper reports NO accuracy metrics whatsoever** — no IoU, F1, OA or kappa; all comparison to the plain U-Net baseline is purely visual (figures), and the only numbers given are flooded-area estimates (e.g. Mengwa flood storage water area grew 19.356 km² → 152.584 km², p. 162).

## Problem & Motivation
Level set methods (notably Chan-Vese, Chan and Vese 2001) are a mainstay for SAR image segmentation because they are robust to boundary location and handle topological changes during curve evolution. Their Achilles heel is **initialization**: "One of the problems that the level set method is facing is the determination of the right initial surface parameter, which implicitly affects the curve evolution and ultimately the segmentation result" (p. 155). A poor initial zero-level set (the classical circle/rectangle) makes evolution slow and prone to local minima. Meanwhile, CNN segmentation (U-Net and friends) has strong feature representation but its pooling layers destroy precise pixel localization — a known tension in semantic segmentation (p. 156).

The paper's core idea is to make the two methods complementary: the CNN does **not** need to be pixel-accurate, it only needs to "describe the general outline of the water body, rather than the accurate edges" (p. 156), acting as a *deep a priori* for the level set, which then supplies the sub-pixel edge accuracy the CNN lacks. Motivation is operational flood monitoring: SAR's all-weather / day-night capability makes it the sensor of choice during floods, but speckle noise makes crisp water edges hard (p. 155).

## Method / Architecture

Overall pipeline (Figure 1, p. 156):
1. Decompose the SAR image into a multi-scale pyramid via **four-point average down-sampling** ("block averaging algorithm"), scales S₀ (finest) … S_L (coarsest) (p. 159).
2. Run **attention U-Net** segmentation on the **coarsest scale S_L** to get a coarse water/land label map.
3. **Up-sample** the initial segmentation result.
4. Use that map to **initialize the zero-level-set function** (φ = t(x), Eq. 8, p. 159) and evolve the level set on the finer-scale reference image.
5. **Remove radar shadow** by integrating DEM + slope.

### Attention U-Net (the part to reimplement)
Reported details (§2.1.1, p. 157) — the authors explicitly defer to **Oktay et al. 2018** ("Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018) for full parameters, so the note below is what *this* paper states:

- **Encoder–decoder** (standard U-Net topology). Nodes denoted X_{i,j}.
- Every X_{i,j} node = **convolution → ReLU**. Conv layers: **kernel 3×3, stride 1, zero-padding 1** (so spatial size is preserved inside a block).
- **Down-sampling: 2×2 max-pooling.**
- **First conv layer has 32 channels**; channel count **doubles after each down-sampling** (so 32 → 64 → 128 → 256 …).
- **Decoder**: up-sampling layer at each step to restore resolution.
- **"A" nodes = attention gates (AGs)**. Placement is explicit and is the key detail:
  > "The 'A' node represents attention gates, which uses the up-sampling layer feature map and the same-level feature map … as inputs to highlight salient features and disambiguate irrelevant and noisy responses." (p. 157)
  I.e. the AG sits **on the skip connection**: its two inputs are (a) the **encoder feature map at the same level** (the skip signal *x*) and (b) the **up-sampled decoder feature map from the layer below** (the gating signal *g*). The AG outputs an attention-weighted version of the encoder feature map.
- **Concatenation** (dot arrows in Figure 2): the **attention-gated feature map is concatenated with the corresponding decoder feature map** — i.e. the raw skip is *replaced* by the gated skip before concat. This is exactly the Oktay et al. additive-attention formulation.
- **Classifier**: a **sigmoid** activation produces the confidence map. At test time threshold at **0.5** (>0.5 → 1, <0.5 → 0) (p. 157).
- ⚠️ **Not reported in this paper**: number of encoder levels/depth, the AG's internal channel count (F_int), whether attention is additive vs multiplicative, whether batch-norm is used, dropout. For a PyTorch reimplementation you must go to Oktay et al. 2018 for the AG internals: `ψ(σ₁(W_g·g + W_x·x + b_g))` → sigmoid σ₂ → resample → multiply elementwise with `x`. Practically: 1×1×1 convs W_g and W_x map g and x to F_int channels, add, ReLU, 1×1 conv to 1 channel, sigmoid → α, then `x̂ = α ⊙ x`.

### Loss
- **Dice loss** (not cross-entropy). Dice coefficient (Eq. 1, p. 157):
  `D_C(p,y) = 2·|{p=c} ∩ {y=c}| / (|{p=c}| + |{y=c}|)`
  Loss (Eq. 2): `min L_Dice = 1 − D_C(p, y)`
- Justification given: for binary {0,1} ground truth the intersection term removes all non-activated pixels; low-confidence predictions on activated pixels are penalized. Chosen because it "can directly optimize the evaluation index" (p. 157).

### Training hyperparameters (§2.1.4, p. 158)
- Input normalization: **divide by 255**.
- **Batch size 32.**
- **Adam**, initial **LR = 0.001**, **β₁ = 0.500, β₂ = 0.999**. ⚠️ Note β₁ = 0.5 is unusual (GAN-style), not the PyTorch default 0.9.
- **LR schedule**: reduced by **10×** when training loss does not decrease for **5 consecutive epochs**; **minimum LR = 1e-8**.
- **~135 epochs.**
- Framework: **PyTorch** (Paszke et al. 2017).
- Hardware: Intel Xeon E5-2680 v4 @ 2.40 GHz, 256 GB RAM, **NVIDIA 1080Ti (11 GB)**.
- ⚠️ **No data augmentation is described.** No validation split is described. No early stopping criterion beyond the LR schedule.

### Level set stage (§2.2)
- Statistical model: SAR intensity in each region R_i modeled by a **gamma distribution** with mean intensity u_i and number of looks L (Eq. 4, p. 158).
- Energy functional (Eq. 6, p. 158): `F(φ,p₁,p₂) = μ∫|∇H(φ)|dx − λ₁∫H(φ)log p₁ dx − λ₂∫(1−H(φ))log p₂ dx`, with H the Heaviside function and δ the Dirac function (Eq. 5). Log base 2.
- Evolution PDE (Eq. 7, p. 159): `∂φ/∂t = δ_ε(φ)[μ·div(∇φ/|∇φ|) + ν + λ₁ log p₁(y|θ̂¹) + λ₂ log p₂(y|θ̂²)]`.
- Gamma parameters θ = {u_i} estimated by **maximum likelihood**: `u_i = Σ_j y_j / N_i` over region Ω_i (p. 159).
- **Initialization: φ = t(x)** where t(x) is the attention U-Net segmentation output (Eq. 8, p. 159). This is the whole contribution.
- Steps: (1) init φ via Eq. 8; (2) evolve φ via Eqs. 6–7; (3) stop when evolution is stationary.
- **Multi-scale refinement** (§2.2.2, p. 159): coarse labels X_L from the CNN → candidate labels at S_{L−1} obtained by **average up-sampling** → refined using image Y_{L−1} via the level set → cascade down to S₀. Rationale: big computational saving, plus coarse scale gives global structure while fine scale gives detail.

### Shadow removal (§2.2.3, pp. 159–160)
- **Otsu** threshold on the **DEM** → mark high-elevation areas, record coordinates.
- **Otsu** threshold on **slope** derived from the DEM (mountain-shadow slope > water slope) → mark those areas.
- Areas matching **both** the elevation mask and the slope mask AND flagged as water are **shadow** → removed.
- Applied only in Chongqing (mountain city), using **2 m DEM downloaded from Google** (p. 162).

## Data

**Training set (§2.1.3, p. 158)**
- **One pair** of **3352 × 3052** SAR image + ground truth as the original dataset. (Sensor for this training image is ⚠️ **not stated** — presumably GF-3.)
- Sliced into **128 × 128 blocks** with **overlapping stride 32** → **1120 image block pairs** total.
- **Ground truth manually marked by geological experts**; black/white = river/land; pixel values **0 = river, 1 = land**. ⚠️ Inconsistent with §2.1.2, which says the sigmoid + Dice setup treats the *activated* (value-1) class as the target — under the stated 0=river/1=land convention the network would be learning *land*. Also ⚠️ the text first says "the black and white parts … represent river and land, respectively" then "pixel values of 0 and 1 are specified for rivers and land" — self-consistent internally but flipped from the usual water=1 convention. Treat with care.
- **Test set**: "three images are chosen as the test images", uniformly divided into 128×128 tiles (p. 158). ⚠️ **No validation split reported.** ⚠️ It is not stated whether the three test images are disjoint scenes from the training image.

**Case-study imagery (§3, §4)**
| Site | Sensor(s) | Date(s) | Resolution | Size |
|---|---|---|---|---|
| Poyang Lake, Jiangxi | Sentinel-2 (pre-flood, 4 scenes) | 16/26/29 Apr 2020 | 10 m | 20,976 × 20,976 px (composite) |
| Poyang Lake | GF-3 SAR, **HV** pol (dual-pol available) | 13 Jul 2020 | 10 m | 28,743 × 26,674 px |
| Wangjiaba/Mengwa, Anhui | GF-3 SAR, HV | 13, 20, 21 Jul 2020 | 10 m | 40,006×36,544; 32,592×30,001; 42,545×38,094 px |
| Wangjiaba | COSMO-SkyMed | 24 Jul 2020 | **3 m** | 16,827 × 18,030 px |
| Chongqing | Sentinel-2 | 18 May 2020 | (10 m implied) | not reported |
| Chongqing | GF-3, **spotlight** mode | 19 Aug 2020 | **1 m** | not reported |
| Chongqing | DEM (Google) | — | **2 m** | not reported |

- **Spectral indices**: ⚠️ **none defined**. For the Sentinel-2 optical extraction the paper only says "The **near-infrared image** of sentinel data is used to extract the water body" (p. 161) — no NDWI/MNDWI formula, no threshold, no method given. This is a significant gap.
- **Preprocessing**: no speckle filter is applied — the *multi-scale down-sampling* is explicitly the speckle-suppression mechanism ("To remove the influence of speckle and preserve important structural information, a multiscale level set algorithm is used", p. 158). No atmospheric correction, no cloud masking, no radiometric calibration described. Only normalization = /255.
- **Data availability**: partial, via Baidu Pan (`https://pan.baidu.com/s/14iSTbXErL44YNG55x3eQtQ`, password `slwg`) (p. 166).

## Results

⚠️ **THERE ARE NO QUANTITATIVE ACCURACY METRICS IN THIS PAPER.** No IoU, no F1, no precision/recall, no overall accuracy, no kappa, no runtime/convergence-iteration comparison — despite the abstract claiming the method "can converge to the edge of the water body **faster** and more precisely" (p. 155). There is **no table of any kind** in the paper. All method-vs-baseline comparison is **visual only** (Figures 7, 10, 11, 14).

**Baseline actually run**: plain **U-Net** (and in Fig. 14 the "attention U-Net method" alone), compared against "our method" (attention U-Net + multiscale level set + DEM). Qualitative claims:
- "Compared with Figures 10 and 11, the water extraction accuracy of our method is **significantly improved** compared with U-Net method **especially for long and thin water body and SAR images with serious speckle noise**." (p. 162) — no number attached.
- On Poyang Lake (weak speckle, water dominates the scene) "the U-Net method can also get better results … On this basis, after the optimization of the level set method, the accuracy of water extraction is also improved, which can be mainly reflected by the extraction of small tributaries." (p. 161)
- Chongqing: "the segmentation effect of U-Net method is poor for SAR images with severe speckle noise and large terrain fluctuation, resulting in a lot of mis-classification, such as roads, bare soil, and other weak scattering objects" (p. 164); DEM+slope fixes this.

**The only numbers reported are derived flood-area statistics** (which depend on the extraction but are not accuracy measures):

| Site | Quantity | Value | Page |
|---|---|---|---|
| Mengwa flood storage area | water area 13 Jul 2020 | **19.356 km²** (10.19% of total area) | p. 162 |
| Mengwa | water area 20 Jul (08:32–18:23) | **58.301 km²** (30.68%) | p. 162 |
| Mengwa | flooded area 21 Jul 17:42 | **112.452 km²** (30.68% ⚠️ *text also says* 19%) | p. 162 |
| Mengwa | inundated area 24 Jul 09:54 | **152.584 km²** (80.31%) | p. 162 |
| Chongqing central urban | Yangtze+Jialing water area, 18 May 2020 | **≈92 km²** (1.82% of Chongqing central city) | p. 164 |
| Chongqing central urban | water area, 19 Aug 2020 | **169 km²** (3.35%) | p. 164 |

⚠️ **Internal inconsistency (p. 162)**: for 21 July the text gives "112.452 square kilometers, accounting for **30.68%** of the total area, and the newly added water body … As of 09:54 on 24 July" but the same sentence earlier reads "accounting for **19%** of the total area". The percentages are mutually incompatible with the km² values and with the 20 July figure (58.301 km² is also labelled 30.68%). The percentage column is unreliable; the km² figures are internally monotonic and plausible.

## Limitations

**Stated by the authors** (p. 166):
- "the problem of water extraction in dense building areas and high mountain areas is still an international problem due to the particularity of SAR sensor imaging."
- Shadow removal uses "only a simple binary segmentation method (Otsu) … for judgment"; future work is to simulate/invert SAR with DEM for better shadow removal.

**Observed by me (important):**
- **No quantitative evaluation at all.** This is the paper's biggest weakness. Every claim of superiority is a figure caption. It cannot be used as a numeric baseline.
- **No ablation**: we cannot separate the contribution of (a) attention gates vs plain U-Net, (b) level set refinement, (c) multi-scale, (d) DEM shadow removal. Fig. 14 hints at (a)+(d) only, visually.
- **Tiny training set**: one 3352×3052 image → 1120 overlapping 128×128 patches with stride 32 means **heavy overlap** and very low effective sample diversity. High risk of overfitting; no validation set, no augmentation.
- **Test set leakage risk**: "three images are chosen as the test images" with no statement that they are geographically or temporally disjoint from the training image.
- Level set stage introduces **unspecified hyperparameters** μ, ν, λ₁, λ₂, ε and a stopping criterion — **none of their values are reported**, so the method is not reproducible as written.
- The claimed **speed** advantage over circular/rectangular initialization is never measured.
- Label convention (0=river) conflicts with the sigmoid/Dice formulation as described (see Data).

## Relevance to this thesis

**Directly borrowable:**
- **The attention U-Net recipe is the main takeaway** and is directly transferable to a Sentinel-2 PyTorch model. Concretely: 3×3 conv / stride 1 / pad 1 + ReLU blocks, 2×2 max-pool, **base width 32 doubling per level**, attention gate **on every skip connection** taking (encoder same-level map, up-sampled decoder map) → gated skip → concat with decoder map, **sigmoid head, 0.5 threshold**. The AG internals must come from Oktay et al. 2018 — **add that reference to `library.bib` too**; this paper alone is not enough to reimplement the gate.
- **Dice loss** (`1 − 2|p∩y|/(|p|+|y|)`) instead of BCE — well motivated for water segmentation, where water is a minority class in most Peruvian tiles. Worth adopting (or Dice+BCE hybrid) and citing this paper as precedent in remote sensing.
- **Training config as a sane starting point**: Adam lr=1e-3, batch 32, 128×128 patches, ReduceLROnPlateau (×0.1, patience 5, min_lr 1e-8), ~135 epochs. Note β₁=0.5 is idiosyncratic — I'd keep PyTorch's 0.9 default unless replicating.
- **Patch extraction with overlapping stride** (128×128, stride 32) as a data-multiplication trick for a small labelled area — relevant since Peruvian flood labels will be scarce. **But treat their 1120-from-one-image setup as a cautionary example**, not a model: no augmentation and no val split is exactly what we must not do.
- **DEM + slope masking** to kill false positives. In our optical case there is no radar shadow, but **terrain shadow / mountain shade in Sentinel-2 (Andes) produces low-NIR pixels that NDWI mistakes for water**. Their Otsu-on-DEM + Otsu-on-slope post-filter is a cheap, directly reusable false-positive suppressor for Peruvian mountainous terrain.
- **Change-detection framing**: pre-flood water (from optical) vs post-flood water (from SAR) → "unchanged water" (blue) vs "newly flooded" (red) map, plus area in km² and % of administrative unit (Figs. 8, 12, 15). This is a clean, cheap **product template** for the Streamlit UI and for the disaster-management framing of the thesis.

**Where it differs from the thesis setup:**
- **SAR (GF-3/COSMO-SkyMed), not Sentinel-2** for the DL stage. Their speckle problem, gamma statistics, and level set energy functional are SAR-specific and **do not transfer** to optical NDVI/NDWI. The level set / Chan-Vese half of the paper is largely irrelevant to us; the CNN half is the useful half.
- They use **NIR band alone** for the Sentinel-2 water mask with no stated method — a much weaker approach than our NDWI (McFeeters). We should not copy that; if anything it is a **cautionary example** of unrigorous optical processing.
- **Single-date inundation mapping**, not **susceptibility/prediction over time series** — no temporal model (no LSTM/ConvLSTM), so it does not inform the "predict flood-prone areas" half of the thesis.

**Classification:** a **method-to-cite** (architecture source, loss-function precedent, DEM-shadow-removal trick) and a **cautionary example** on evaluation rigor. It is **NOT usable as a quantitative baseline** — there is nothing to beat. Cite it when introducing attention U-Net and Dice loss in flood mapping; do not cite it for numbers.

## Keywords / Tech
- **Models**: Attention U-Net (Oktay et al. 2018), U-Net (baseline), UNet++ (cited only), FCN/SegNet/DeepLab (cited only)
- **Classical methods**: level set, Chan-Vese active contours without edges, Mumford-Shah functional, Heaviside/Dirac regularization, gamma distribution SAR speckle model, maximum likelihood, **Otsu** thresholding
- **Sensors**: Gaofen-3 (GF-3, HV pol, 10 m; spotlight 1 m), COSMO-SkyMed (3 m), Sentinel-2 (10 m), DEM (2 m)
- **Loss**: Dice loss
- **Frameworks**: PyTorch
- **Techniques**: four-point average down-sampling pyramid (multi-scale), overlapping patch extraction (128×128, stride 32), sigmoid + 0.5 threshold, DEM+slope shadow removal, bi-temporal water change mapping
- **Indices**: none (⚠️ no NDWI/NDVI used)

## Notable quotes

> "The prediction result is used to initialize the zero-level set, which only needs to describe the general outline of the water body, rather than the accurate edges. Compared with the traditional circular and rectangular zero level set initialization method, the U-Net level set method can converge to the edge of the water body faster and more precisely. It does not fall into the local minimum value, and can obtain accurate segmentation results." (p. 156)

> "The 'A' node represents attention gates, which uses the same-level feature map and the up-sampled feature map of the upper layer as inputs to highlight salient features and disambiguate irrelevant and noisy responses. The dot arrows indicate the concatenation connections, which merge the attention feature map and the corresponding upper feature map." (p. 157)

> "To achieve a better image SAR segmentation task, the Dice similarity coefficient (Dice) is chosen as the loss function, which can directly optimize the evaluation index and achieve a better segmentation map." (p. 157)

> "Compared with Figures 10 and 11, the water extraction accuracy of our method is significantly improved compared with U-Net method especially for long and thin water body and SAR images with serious speckle noise." (p. 162) — note: asserted without any metric.

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/Z58TV9IJ/Xu et al. - 2022 - SAR image water extraction using the attention U-net and multi-scale level set method flood monitor.pdf`
Cite as: `\cite{xu_sar_2022}`
