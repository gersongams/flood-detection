---
title: "SatViT-Seg: A Transformer-Only Lightweight Semantic Segmentation Model for Real-Time Land Cover Mapping of High-Resolution Remote Sensing Imagery on Satellites"
authors: "Daoyu Shu, Zhan Zhang, Fang Wan, Wang Ru, Bingnan Yang, Yan Zhang, Jianzhong Lu, Xiaoling Chen"
year: 2026
venue: "Remote Sensing (MDPI), 2026, 18, 1 (27 pp.); Received 15 Oct 2025, Accepted 14 Dec 2025, Published 19 Dec 2025"
doi: "10.3390/rs18010001"
bibtex_key: shu_satvit-seg_2026
zotero_pdf: "/Users/gersongarrido/Zotero/storage/EF2GH8MJ/Shu et al. - 2026 - SatViT-Seg A Transformer-Only Lightweight Semantic Segmentation Model for Real-Time Land Cover Mapp.pdf"
pages: 27
sensors: [WorldView-class RGB aerial/satellite (DeepGlobe), Wuhan-1 satellite (pan 0.5 m + MS 2 m), Gaofen series (RGB)]
task: "land cover semantic segmentation (multi-class), on-board / real-time oriented — NOT flood mapping"
model: "SatViT-Seg — pure Vision Transformer (hierarchical, window self-attention + linear-attention global tokens, LGAD + Bi-dimensional Attentive FFN, convolution-free ATM decoder)"
tags: [semantic-segmentation, vision-transformer, lightweight, land-cover, on-board-inference, linear-attention, deepglobe, efficiency]
---

# SatViT-Seg: A Transformer-Only Lightweight Semantic Segmentation Model for Real-Time Land Cover Mapping of High-Resolution Remote Sensing Imagery on Satellites

> **TL;DR** — The authors propose SatViT-Seg, a *homogeneous* (convolution-free) lightweight Vision Transformer for land-cover semantic segmentation of high-resolution RS imagery, targeted at on-board/edge deployment. Its two novel blocks are LGAD (window self-attention for local detail + dynamically pooled global tokens routed through two linear-attention steps for long-range context) and a Bi-dimensional Attentive FFN (channel gating from token-wise mean/max/var + low-rank spatial prototype gating). With **5.496 GFLOPs and 11.363 M parameters** it reports the best mIoU/mF1/OA on all three benchmarks: DeepGlobe **71.18 ± 0.24 mIoU**, Wuhan **58.56 ± 0.27**, Guangdong **78.05 ± 0.21** (p. 14), i.e. "improving mIoU by up to 1.81% over the strongest baseline" (p. 25), while having the lowest FLOPs and lowest peak activation memory (0.46 GB) of all compared methods (p. 15). **This is generic land-cover segmentation, not flood mapping.**

## Problem & Motivation
On-board processing of high-resolution remote sensing (HR-RS) imagery would let satellites downlink land-cover class maps instead of raw imagery, cutting bandwidth and latency for disaster response, environmental monitoring, and precision agriculture (p. 2). But spaceborne hardware has tight compute, power, and memory budgets, so segmentation models must be both accurate and cheap.

The authors argue two existing paradigms fall short. (1) Pure-CNN lightweight models (D-LinkNet, ESFNet, PCNeXt, LPCUNet) struggle to model long-range dependencies and cross-region semantic consistency. (2) CNN–Transformer *hybrids* (UNetFormer, LiteSeger, BAFNet, LWGANet) insert convolutions as explicit aggregation modules to compensate for attention's weak local modeling — this "operator heterogeneity" complicates cross-layer fusion, expands the operator stack, and hinders quantization/compilation for embedded inference engines (pp. 3–4). Meanwhile lightweight ViTs typically over-compress channels and use shallow contextual interaction, degrading boundary delineation and small/rare-class recall (p. 4). The gap the paper claims: a *homogeneous pure-Transformer* architecture that keeps both fine detail and global context at near-linear cost.

## Method / Architecture
**Backbone (Figure 1, p. 6).** Hierarchical pure ViT, three stages at H/16, H/32, H/64 resolution (channels C1, C2, C3; block counts L1, L2, L3 — exact per-stage depths/widths **not reported** numerically). Downsampling is by *linear* patch embedding and *linear* patch merging (no conv). Memory footprint comes only from matmul + elementwise ops, enabling graph-level operator fusion and quantization.

**LGAD — Local-Global Aggregation and Distribution (Section 2.2, pp. 7–10).**
- *Local branch*: standard softmax window self-attention with 2D relative positional bias `B_rel` inside non-overlapping windows of side `w_s` (Eq. 5, p. 8). Cost quadratic inside a window, O(N·S) overall.
- *Geometry-aware tokens*: each token is augmented by concatenating normalized 2D grid coords and second-order terms `[x_i; u_i; v_i; u_i²; v_i²]` then linearly projected (Eq. 6, p. 8), with `(u,v) ∈ [-1,1]`.
- *Dynamic global token construction*: features are re-partitioned into pooling blocks of side `g`; a 2-layer MLP produces softmax intra-block weights and a weighted sum yields **one global token per window** (Eq. 7, p. 9). T = (H/g)·(W/g) global tokens.
- *Linear attention* (Katharopoulos et al. kernel `φ(·)=ELU(·)+1`) is applied twice: image→global aggregation (Eqs. 8–9) and global→image distributing (Eqs. 10–11). No explicit N×N attention matrix ⇒ linear in token count.
- *Fusion*: `Y = Proj(Y_loc + Y_glb)` (Eq. 12, p. 10). Pre-normalization + stochastic depth are used (stochastic-depth rate **not reported**).

**Bi-dimensional Attentive FFN (Section 2.3, pp. 10–12).** A gated MLP: one shared up-projection `W_↑ ∈ R^{C×2D}` produces value `U` and gate `G`; `Ũ = φ(U) ⊙ G` with `φ = SiLU`; two light down-projections give `y_c`, `y_s` (Eq. 13).
- *Channel gate*: statistics `m = Mean_i(h_i)`, `a = Max_i(h_i)`, `v = Var_i(h_i)` across tokens → `g_c = σ(ψ([m;a;v]W_1)W_2)`, low-rank (Eqs. 14–15, pp. 10–11).
- *Spatial gate*: rank-one prototype attention — global mean gives query `q`, each token gives a low-rank key `k_i`; `s_i = qᵀk_i/(√r·T)` with `T = softplus(γ)+1`; `g_{s,i} = σ(s_i)` (Eqs. 16–17, p. 11). O(N·r) cost.
- *Fusion*: `α = σ(β)` learnable, `y = α ŷ_s + (1-α) ŷ_c`, residual `x_out = x + y` (Eq. 18, p. 12).

**Decoder.** Convolution-free ATM head from SegViT: normalized patch features are compared with learnable class prototypes to give class response maps, bilinearly upsampled to image resolution (p. 6). No heavy upsampling head, no multi-level feature fusion.

**Loss (Eqs. 1–4, p. 7).** Cosine-similarity logits with learnable temperature τ (`s_{k,i} = τ · f̂_kᵀ f̂_i`, softmax → `p_{k,i}`), then
- `L_CE` = pixelwise cross-entropy,
- `L_Dice` = class-averaged Dice regularizer,
- **`L = L_CE + λ · L_Dice`**. ⚠️ The value of λ is **not reported** (only "λ set on the training set to balance resources and accuracy", p. 7).

**Training hyperparameters (Section 3.2, p. 13).**
- Framework: first LuoJiaNET, then re-implemented in PyTorch (torch 2.4.1+cu118) for all comparisons. Rows marked `*` in tables = LuoJiaNET numbers.
- Hardware: single NVIDIA Tesla V100, 32 GB.
- Init: Gaussian, mean 0, std 0.02. Per-channel normalization using training-set mean/std.
- Optimizer: **AdamW**, LR **1e-4**, weight decay **0.01**, β1 0.9, β2 0.999.
- Schedule: **cosine annealing** with **5-epoch linear warmup**; **120 epochs**; **batch size 2**; global grad-norm clipping 1.0; mixed precision + cuDNN autotuning; fixed seeds.
- Augmentation: random horizontal/vertical flips, random 90° rotations, random scale 0.75–1.25, random brightness/contrast/saturation/hue perturbations, mild Gaussian blur or Gaussian noise. Applied synchronously to image and label.
- Inference: single-scale sliding window 1024×1024, stride = window size; **no TTA, no multi-scale ensembling**.
- Table 1 numbers = mean ± std over **five independent runs** with different seeds.
- Default local window size = **8** (chosen empirically, Section 3.6.3, p. 22).
- No pretrained weights (all models trained end-to-end from scratch in PyTorch, p. 13).

## Data
Three land-cover benchmarks (Section 3.1, p. 12). **No Sentinel-2, no SAR, no spectral indices, no flood labels.**

| Dataset | Source / sensor | Size | GSD | Classes | Split (scene-level) |
|---|---|---|---|---|---|
| **DeepGlobe** [32] | HR optical, mostly Thailand, Indonesia, India; rural/agricultural | 803 images, 2448×2448 px | ~0.5 m | 7 land-cover classes | 455 / 207 / 142 (train/val/test) |
| **Wuhan** [33] | Wuhan-1 satellite, pan–MS camera; Wuhan metro area | 20 images, ~5000×6000 px | 0.5 m pan, 2 m MS | 8 classes incl. background | 12 / 4 / 4 |
| **Guangdong** [34] | Chinese Gaofen series, UHR RGB; Guangdong Province urban/peri-urban | 15 scenes, 8192×8192 px | 0.5–2 m | 11 semantic classes | 9 / 3 / 3 |

- **Tiling**: non-overlapping sliding window, tile size **1024×1024**, stride = window size; mirror padding when dimensions are not divisible. Crops generated strictly within each subset to prevent leakage (p. 13).
- **Bands**: effectively RGB. Multispectral Wuhan data is co-registered but the paper does not enumerate bands. **No band list reported.**
- **Ground truth**: officially provided pixel-level annotations (DeepGlobe, Wuhan); Guangdong annotations were curated/quality-checked by the authors under a unified protocol.
- **Preprocessing**: per-channel mean/std normalization only. No atmospheric correction, cloud masking, or speckle filtering (not applicable — no S1/S2 data).
- **Spectral indices**: **none used.**

## Results

**Table 1 (p. 14) — main comparison, mean ± std over 5 seeds, in-distribution (train and test on same dataset).** `*` = LuoJiaNET implementation.

| Method | DeepGlobe mIoU | DG mF1 | DG OA | Wuhan mIoU | Wuhan mF1 | Wuhan OA | Guangdong mIoU | GD mF1 | GD OA |
|---|---|---|---|---|---|---|---|---|---|
| *Pure CNN* | | | | | | | | | |
| LPCUNet | 64.58 ± 0.34 | 84.61 ± 0.42 | 77.70 ± 0.28 | 52.66 ± 0.39 | 81.79 ± 0.47 | 68.28 ± 0.33 | 72.69 ± 0.31 | 87.43 ± 0.36 | 83.38 ± 0.24 |
| ESFNet | 58.92 ± 0.41 | 80.26 ± 0.49 | 74.55 ± 0.35 | 47.02 ± 0.44 | 77.56 ± 0.53 | 64.34 ± 0.37 | 64.11 ± 0.46 | 81.63 ± 0.51 | 79.70 ± 0.29 |
| D-LinkNet | 67.60 ± 0.33 | 87.05 ± 0.40 | 79.48 ± 0.27 | 54.66 ± 0.36 | 83.12 ± 0.45 | 69.81 ± 0.31 | 74.19 ± 0.34 | 88.70 ± 0.39 | 84.61 ± 0.26 |
| PCNeXt | 65.88 ± 0.37 | 85.47 ± 0.43 | 78.28 ± 0.30 | 53.35 ± 0.41 | 82.19 ± 0.48 | 68.88 ± 0.34 | 73.56 ± 0.38 | 88.12 ± 0.42 | 84.10 ± 0.27 |
| *Hybrid CNN-ViT* | | | | | | | | | |
| UNetFormer | 69.66 ± 0.28 | 88.64 ± 0.35 | 81.12 ± 0.23 | 56.75 ± 0.32 | 84.81 ± 0.40 | 71.22 ± 0.26 | 76.54 ± 0.25 | 90.82 ± 0.31 | 86.10 ± 0.19 |
| LWGANet | 67.09 ± 0.36 | 86.76 ± 0.44 | 79.12 ± 0.29 | 55.41 ± 0.38 | 83.83 ± 0.46 | 70.31 ± 0.32 | 74.70 ± 0.35 | 89.05 ± 0.41 | 84.86 ± 0.25 |
| LiteSeger | 60.87 ± 0.47 | 81.63 ± 0.55 | 75.26 ± 0.37 | 49.80 ± 0.52 | 79.53 ± 0.60 | 66.08 ± 0.40 | 66.96 ± 0.50 | 83.27 ± 0.57 | 80.68 ± 0.33 |
| BAFNet | 68.96 ± 0.34 | 88.12 ± 0.42 | 80.65 ± 0.28 | 56.25 ± 0.37 | 84.44 ± 0.45 | 71.01 ± 0.31 | 76.01 ± 0.36 | 90.39 ± 0.42 | 85.90 ± 0.27 |
| light4mars | 62.48 ± 0.45 | 82.86 ± 0.52 | 76.04 ± 0.36 | 51.24 ± 0.49 | 80.66 ± 0.58 | 67.20 ± 0.38 | 70.38 ± 0.47 | 85.60 ± 0.54 | 82.21 ± 0.32 |
| **satvit-seg (Ours)** | **71.18 ± 0.24** | **90.01 ± 0.30** | **82.09 ± 0.18** | **58.56 ± 0.27** | **86.18 ± 0.33** | **72.14 ± 0.20** | **78.05 ± 0.21** | **92.30 ± 0.26** | **86.96 ± 0.16** |
| satvit-seg * (LuoJiaNET) | 71.62 ± 0.22 | 90.31 ± 0.28 | 82.63 ± 0.17 | 59.36 ± 0.25 | 86.50 ± 0.31 | 73.80 ± 0.19 | 78.53 ± 0.19 | 92.70 ± 0.24 | 87.40 ± 0.15 |

- Strongest baseline is **UNetFormer**. SatViT-Seg (PyTorch) beats it by ~**1.5 / 1.8 / 1.5** mIoU points on DeepGlobe / Wuhan / Guangdong (p. 17). LuoJiaNET version: ~2.0 / 2.6 / 2.0 points (p. 17).
- Cross-dataset **average mIoU 69.26%** vs best baseline **67.65%**, i.e. **+1.61%** (p. 19).
- Improvement over group means: ~6.9 / 6.6 / 6.9 mIoU points vs pure CNNs; ~5.4 / 4.7 / 5.1 vs hybrids (p. 17).
- SatViT-Seg wins **all nine dataset×metric combinations** (p. 17).

**Table 5 (p. 15) — efficiency** (batch 1, 1024×1024 input, Tesla V100):

| Method | Params (M) | Peak Activ. Mem (GB) | Throughput (img/s) | FLOPs (G) |
|---|---|---|---|---|
| LPCUNet | 20.099 | 4.72 | 25.33 | 63.147 |
| ESFNet | **0.178** | 1.01 | 93.92 | 10.695 |
| D-LinkNet | 217.643 | 9.70 | 9.76 | 481.59 |
| PCNeXt | 3.626 | 1.16 | 76.94 | 12.132 |
| UNetFormer | 11.682 | 1.12 | 94.58 | 46.889 |
| LWGANet | 12.539 | 3.03 | 34.11 | 47.914 |
| LiteSeger | 0.750 | 1.30 | **123.75** | 17.168 |
| BAFNet | 5.379 | 2.01 | 51.05 | 40.239 |
| light4mars | 2.633 | 1.35 | 67.04 | 9.776 |
| **satvit-seg (Ours)** | 11.363 | 0.46 | 20.82 | **5.496** |
| satvit-seg * (LuoJiaNET) | 11.015 | **0.45** | 23.63 | 5.259 |

⚠️ Important caveat the authors state themselves: SatViT-Seg has the **lowest FLOPs but nearly the lowest throughput** (20.82 img/s vs LiteSeger 123.75, UNetFormer 94.58) on a general GPU. Their explanation is that conv-heavy baselines have highly optimized GPU kernels and that the homogeneous operator stack's advantage would only materialize on embedded inference engines (p. 22). **The "real-time" claim in the title is therefore not demonstrated** — see Limitations.

**Table 2 (p. 14) — ablation (mIoU %, PyTorch):** baseline (window attention + standard FFN) = 66.82 / 53.95 / 72.26 (DG/Wuhan/GD). +global branch only: 67.78 / 54.82 / 73.41. Local+global: 69.91 / 57.01 / 75.51. + channel branch: 70.54 / 58.03 / 76.65. Full (+spatial branch): **71.12 / 58.98 / 78.29**. Total gain over baseline: **+4.30 / +5.03 / +6.03** (p. 20), avg ~5.1%.

**Table 3 (p. 14) — global-token generation strategy (mIoU %):** Average pooling 70.08 / 58.05 / 77.42; Max pooling 69.41 / 57.11 / 76.31; Learnable static tokens 70.35 / 58.27 / 77.68; **Ours (dynamic weighted pooling) 71.12 / 58.98 / 78.29**.

**Table 4 (p. 15) — local window size:** window 1 → 70.08 / 57.86 / 77.48, 5.27 GFLOPs, 22.94 img/s; window 8 → 71.12 / 58.98 / 78.29, 5.496 G, 20.82 img/s; window 16 → 71.26 / 59.19 / 78.40, 6.184 G, 19.97 img/s. Window 16 gives only +0.15% avg mIoU for +12.52% compute → window **8** adopted.

**⚠️ Inconsistencies observed:**
1. Table 1 reports SatViT-Seg mIoU as **71.18 / 58.56 / 78.05**, but Tables 2, 3 and 4 (window 8 = the default configuration) report the *same model* as **71.12 / 58.98 / 78.29**. The ablation/window tables and the main table disagree by up to 0.42 mIoU points on Wuhan. Presumably Table 1 is a 5-seed mean and the ablations are single runs, but the paper does not say so.
2. The headline "**up to 1.81%**" (Highlights p. 1; Conclusions p. 25) does not match any single number in the body: the largest PyTorch gain over UNetFormer is 58.56 − 56.75 = **1.81** on Wuhan — so the claim is consistent with Table 1, but Section 3.5 (p. 17) describes the same gain as "about 1.8" and the abstract elsewhere says "up to 1.81%". Not a true error, but note that 1.81 is a *Wuhan-only* PyTorch figure, not a cross-dataset average (the cross-dataset gain is 1.61%, p. 19).
3. λ in the total loss (Eq. 4) is never given a value.

## Limitations
**Stated by the authors (Section 4.2, pp. 24–25):**
- **All results are in-distribution.** They explicitly did *not* run cross-dataset generalization (train on one dataset, test on another without fine-tuning): "our claims are restricted to consistent performance across multiple in-distribution HR-RS benchmarks, rather than formal domain generalization" (p. 24).
- Fixed-scale tiling + single-scale inference; no handling of inconsistent class taxonomies or severe class imbalance.
- Under cross-sensor, cross-season, open-set conditions, small-object recall and boundary consistency may degrade.
- The prototype-based ATM decoder with single-pass upsampling has limited fidelity for very fine structures → rare classes confusable, thin boundaries fragmented.
- All runtime/memory measured on a **Tesla V100**, "not as direct evidence of flight-qualified, on-board performance" (p. 23). Claims are framed as *theoretical efficiency* and *potential suitability*.

**Observed by me:**
- **The "real-time on satellites" framing is unvalidated.** No FPGA / satellite SoC / embedded benchmark was run. Actual measured throughput on GPU is the 2nd-lowest of all 10 methods. The efficiency argument rests entirely on FLOPs and peak activation memory as proxies.
- **RGB only.** No multispectral or multi-temporal experiments despite the Wuhan sensor having MS bands.
- No public code/weights link is given (Data Availability: "The original contributions of this work are fully contained in the article", p. 25).
- Batch size 2 with 120 epochs on a single V100 is a small-compute regime; baselines were retrained under the same protocol, which is fair but means none of the baselines are at their published numbers.
- Two of the three datasets (Wuhan, Guangdong) have **4 and 3 test scenes** respectively — a very small test set for a scene-level split, even if tiled into many 1024×1024 crops.

## Relevance to this thesis
**Honest assessment: low-to-moderate, and indirect.** This paper is about generic multi-class land-cover segmentation on 0.5–2 m RGB imagery. It is **not** a flood paper, **not** Sentinel-2, **not** multispectral, **not** multi-temporal, and uses **no spectral indices**. It is not a baseline you can compare against and it is not a dataset source. Where it is useful:

- **As an architecture candidate / cite for "efficient segmentation backbone".** If the thesis moves from a plain PyTorch CNN to a segmentation network, SatViT-Seg is a citable example of a lightweight ViT that outperforms UNetFormer/D-LinkNet at 5.5 GFLOPs. But the thesis works at **10 m Sentinel-2**, where scenes are far smaller in pixel count than 8192×8192 Gaofen tiles — the ultra-high-resolution motivation (quadratic attention blowup) largely evaporates. A U-Net or UNetFormer is likely sufficient; SatViT-Seg's complexity is not justified by the thesis's data scale.
- **Directly borrowable — the loss.** `L = L_CE + λ·L_Dice` with a **class-averaged Dice regularizer** (Eq. 3, p. 7) is exactly the right shape for flood segmentation, where water is a minority class. This is a cheap, well-motivated change from plain CE. ⚠️ λ is not reported, so you must tune it yourself.
- **Directly borrowable — the augmentation policy** (p. 13): random h/v flips, 90° rotations, random scale 0.75–1.25, plus brightness/contrast/saturation/hue jitter and mild Gaussian blur/noise "to accommodate radiometric variation and sensor noise". This is a sensible, citable RS-specific augmentation recipe for a Sentinel-2 pipeline. Note the color jitter is designed for RGB — applying it naively to NDVI/NDWI channels would be wrong; jitter reflectance bands *before* computing indices, or skip it.
- **Directly borrowable — the metric set and reporting protocol.** They report **mIoU (primary), mF1, OA** with formulas (Eqs. 19–22, pp. 15–16), plus **Params / FLOPs / throughput / peak activation memory** for efficiency, and — importantly — **mean ± std over 5 seeds**. Reporting seed variance is good practice the thesis should copy; a flood model's IoU difference of 1 point is meaningless without it.
- **Borrowable — tiling protocol.** Non-overlapping 1024×1024 sliding-window tiles with mirror padding, crops generated *strictly within each subset to prevent information leakage* (p. 13). The leakage point matters a lot for the thesis: if you tile a Peruvian AOI and then randomly split tiles, adjacent tiles land in train and test. Do the split at scene/region level first, as they did.
- **Cautionary example (the most valuable use).** The paper's own limitations section is a clean statement of a trap the thesis could fall into: (a) they title the paper "real-time ... on satellites" but never benchmark on satellite hardware, and their measured GPU throughput is among the worst of all methods — a reminder that FLOPs ≠ speed; (b) they explicitly disclaim cross-dataset generalization. For a flood thesis intended for Peru, a model validated only in-distribution on one AOI is not evidence it generalizes to another river basin — say so up front rather than being caught out.
- **Baselines named here are worth knowing**: UNetFormer [22] (the strongest baseline, a UNet-like transformer with a ResNet18 encoder) and D-LinkNet [18] are more realistic architecture references for a Sentinel-2 segmentation thesis than SatViT-Seg itself.
- **Where it differs from the thesis**: 0.5 m RGB vs 10 m 4-band; single-date land cover vs NDVI/NDWI *time series*; 7–11 land-cover classes vs binary/graded flood risk; no index engineering at all; disaster-response is only cited as a motivating application, never evaluated.

## Keywords / Tech
- **Models**: SatViT-Seg (pure ViT), LGAD module, Bi-dimensional Attentive FFN, ATM decoder (from SegViT), window self-attention, linear attention (Katharopoulos kernel φ = ELU+1)
- **Baselines**: LPCUNet, ESFNet, D-LinkNet, PCNeXt (CNN); UNetFormer, LWGANet, LiteSeger, BAFNet, Light4Mars (CNN-ViT hybrid)
- **Sensors / data**: DeepGlobe (0.5 m RGB), Wuhan-1 satellite (0.5 m pan / 2 m MS), Gaofen series (Guangdong, 0.5–2 m RGB)
- **Indices**: none
- **Frameworks**: PyTorch 2.4.1+cu118; LuoJiaNET (remote-sensing DL framework, LuoJiaAI platform)
- **Techniques**: dynamic weighted global-token pooling, channel gating (mean/max/var statistics), low-rank spatial prototype gating, cosine-similarity class prototypes with learnable temperature, CE + class-averaged Dice loss, AdamW + cosine annealing + 5-epoch warmup, mixed precision, stochastic depth, sliding-window tiling with mirror padding
- **Metrics**: mIoU, mF1, OA, Params, FLOPs, throughput (img/s), peak activation memory (GB)

## Notable quotes
- "With only 5.496 GFLOPs and 11.363 M of parameters, SatViT-Seg attains state-of-the-art mIoU on DeepGlobe, Wuhan, and Guangdong, improving performance by up to 1.81 over strong baselines while maintaining high throughput." (p. 1, Highlights)
- "In this work, all quantitative results are obtained in-distribution, i.e., on held-out test sets from the same dataset used for training and validation. ... we do not perform explicit cross-dataset generalization experiments ... As such, our claims are restricted to consistent performance across multiple in-distribution HR-RS benchmarks, rather than formal domain generalization." (p. 24)
- "All runtime and memory measurements in Table 5 are obtained on a Tesla V100 GPU and should therefore be interpreted as a standardized environment for relative comparison, rather than as direct evidence of flight-qualified, on-board performance." (p. 23)
- "Several baselines dominated by convolutions or containing highly optimized operators exhibit higher throughput on general GPUs, for example LiteSeger and UNetFormer, yet their FLOPs and accuracy do not dominate simultaneously." (p. 22)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/EF2GH8MJ/Shu et al. - 2026 - SatViT-Seg A Transformer-Only Lightweight Semantic Segmentation Model for Real-Time Land Cover Mapp.pdf`
Cite as: `\cite{shu_satvit-seg_2026}`
