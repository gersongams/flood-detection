---
title: "High-precision flood change detection with lightweight SAR transformer network and context-aware attention for enriched-diverse and complex flooding scenarios"
authors: "Menghao Du, Zhenfeng Shao, Xiongwu Xiao, Jindou Zhang, Duowang Zhu, Jinyang Wang, Timo Balz, Deren Li"
year: 2026
venue: "ISPRS Journal of Photogrammetry and Remote Sensing, vol. 231 (2026), pp. 507–531"
doi: "10.1016/j.isprsjprs.2025.11.011"
bibtex_key: du_high-precision_2026
zotero_pdf: "/Users/gersongarrido/Zotero/storage/MTGDAPIX/Du et al. - 2026 - High-precision flood change detection with lightweight SAR transformer network and context-aware att.pdf"
pages: 25
sensors: [Sentinel-1]
task: "flood inundation change detection (bi-temporal, pre-flood vs post-flood SAR)"
model: "AWCA-Net — Siamese PVT-v2-b2 transformer encoder + NECM / LGDM / MSAWM attention modules + multi-scale BCEDice loss"
tags: [sar, change-detection, transformer, attention, lightweight, flood-mapping, sentinel-1, benchmark-dataset, class-imbalance, siamese]
---

# High-precision flood change detection with lightweight SAR transformer network and context-aware attention

> **TL;DR** — The authors propose **AWCA-Net**, a Siamese transformer (PVT-v2-b2 backbone, weight-shared) for **bi-temporal SAR flood change detection**, combining three new modules (NECM neighborhood feature enhancement, LGDM large-kernel grouped difference attention gate, MSAWM adaptive-window multi-scale convolutional attention) and a multi-scale BCE+Dice loss. They also release **VarFloods**, a new benchmark of 2,186 Sentinel-1 pre/post-flood image pairs across 5 continents (Bolivia, France, Honduras, Vietnam, Libya) in both raw GRD and preprocessed versions. AWCA-Net reaches **IoU 94.21 % on S1GFloods, 65.60 % on ETCI-2021, 80.58 % on VarFloods-G and 84.87 % on VarFloods-P** (pp. 517–518), beating 10 SOTA CD baselines, while using only **35.21 M params / 17.53 GFLOPs** (p. 529, Table 9). Reported gains over the best competitor (BAN) are +1.42 / +1.68 / +1.31 / +1.47 IoU points respectively.

## Problem & Motivation

Flood change detection (CD) from SAR is attractive because SAR is all-weather / all-time, unlike optical imagery. But the authors identify three gaps (p. 508): (1) most deep CD work is validated on a **few specific flood scenarios**, so models do not generalize across flood causes (heavy rain, tropical storm, hurricane, dam break) or terrains; (2) existing datasets have **highly imbalanced flood-inundation ratios**, producing false positives where flooding is limited and missed detections where flooding is widespread; (3) transformer-based CD models achieve high precision but at **large parameter counts and computational cost**, hindering deployment for rapid emergency response.

Their answer is two-pronged: a lightweight-but-accurate attention transformer (AWCA-Net) and a purpose-built **enriched-diverse** benchmark (VarFloods) that deliberately spans continents, flood causes, years, land-cover types and inundation ratios (2.26 %–10.84 %), so that model strengths can be dissected per-scenario rather than reported as a single aggregate number.

## Method / Architecture

Overall (Fig. 1, Fig. 3, pp. 509–511): a **Siamese/weight-shared encoder** processes the pre-flood image A and post-flood image B independently, then bi-temporal features are fused level by level, decoded with up-convolution + attention gating, and supervised at 4 scales.

**How the temporal pair is handled** (important for the thesis):
- Encoder: `T^i_A = Encoder(A)`, `T^i_B = Encoder(B)`, i = 1..4 — **PVT-v2-b2** (Wang et al., 2022) as feature extractor, weights shared between the two branches (eqs. 1–2, p. 510).
- Each branch's multi-scale features go through **NECM** independently: `E^i_A = NECM(T^i_A)`, `E^i_B = NECM(T^i_B)` (eqs. 3–4).
- **BTFA (Bi-temporal Feature Aggregation)**: the two enhanced streams are **concatenated** then compressed: `E^i_C = Conv_{3×3}(Conv_{1×1}(Concat(E^i_A, E^i_B)))` (eq. 5, p. 510). So temporal fusion is **concat + conv**, not explicit differencing at this stage. The "difference" reasoning happens later, inside LGDM, between *high-level and low-level* decoder features (not between t1 and t2).
- Decoder: `M_4 = MSAWM(E^4_C)`; `G_i = EUCB(M_{i+1})`; `L_i = LGDM(G_i, E^i_C)`; `M_i = MSAWM(L_i + G_i)` for i = 1,2,3 (eqs. 6–9, p. 511). EUCB = efficient up-convolution block (Rahman et al., 2024).
- **FRH** (feature refinement head, a 1×1 conv) produces single-channel maps `C_i`, upsampled by ×4, ×8, ×16, ×32 to full resolution `U_i` (eqs. 10–11) → four scale outputs, all supervised.

**NECM** (neighborhood feature enhancement w/ contextual information, Fig. 4, p. 511): unify channels of T^1..T^4 with 1×1 conv, upsample all to a common size S, concat along channels, apply 3×3 conv → fused `E_f`; then residually add back to each original scale feature (`E^1_f = E_1 + T^1`, and progressively `E^k_f = E_k + T^k`, eqs. 12–18). Purpose: inject high-level semantic context into every level while keeping fine detail.

**LGDM** (large-kernel grouping attention gate on high–low layer feature difference, Fig. 5, p. 512): given high-level gating feature G and low-level feature E, apply **3×3 grouped convolutions** separately: `E' = GConv(BN(Conv_{3×3}(E)))`, `G' = GConv(BN(Conv_{3×3}(G)))`; then `Fusion = ReLU(Concat(G'+E', G'−E'))` — i.e. both the **sum** and the **difference** are used; then `φ = σ(BN(Conv_{1×1}(Fusion)))` and `LGDM(G,E) = E ⊙ φ` (eqs. 19–23). Explicitly contrasted with Attention U-Net (Oktay et al., 2018) which uses 1×1 convs — grouped 3×3 gives a bigger receptive field at lower cost.

**MSAWM** (multi-scale conv attention w/ adaptive window selection, Fig. 6, p. 513): `MSAWM = AMSCB(SAB(CAB(x)))`.
- **CAB** channel attention: max-pool + avg-pool → shared MLP with 1/16 channel bottleneck → sigmoid (eqs. 34–35), based on Woo et al. (2018).
- **SAB** spatial attention: mean + max over channels, concat, **7×7 conv**, sigmoid (eqs. 36–37).
- **AMSCB** adaptive multi-scale conv block: 1×1 conv expands channels by **factor 6**, then **depthwise convs with kernels 1×1, 3×3, 5×5** in parallel; each branch is GAP-pooled → concat → FC–ReLU–FC → **Softmax** gives weights `w ∈ R^{3×1×1}`; outputs are weighted-summed (`AMSDC(X_3) = Σ_{k∈{1,3,5}} w_k · DWConv_k(X_3)`), then 1×1 conv back to C and a residual add (eqs. 25–33). Inspired by MobileNetV2 inverted residual block + channel shuffle.

**Loss** — **multi-scale BCEDice** (eqs. 38–40, p. 514): all four scale outputs are upsampled to label resolution and `Loss = Σ_{i=1..l} (BCE_i + 1 − Dice_i)`. Motivated by class imbalance (few flood pixels).

**Training hyperparameters** (Sec. 5.2, p. 516):
- Framework **PyTorch**, single **NVIDIA RTX 3090 (24 GB)**.
- Batch size 16; initial LR **2e-4**; seed 16; num_workers 4.
- Optimizer **Adam**, betas (0.9, 0.99), weight decay **1e-4**.
- Poly LR schedule: `lr = lr^k (1 − Curiter/Maxiter)^α`, α = 0.9. Maxiter = **20 K iters (S1GFloods)**, **90 K (ETCI-2021)**, **9 K (VarFloods-G)**, **13 K (VarFloods-P)**.
- Augmentation: normalization, random cropping, flipping, rotation.
- All methods trained from scratch; for S1GFloods they use the original papers' hyperparameters, for ETCI-2021/VarFloods they use **unified** hyperparameters + loss across all compared methods "to ensure fairness" (p. 516).
- ⚠️ Number of *epochs* is **not reported** (only iterations). No early-stopping / model-selection criterion reported.

## Data

**Sensor: Sentinel-1 SAR only** (no optical, no Sentinel-2, no spectral indices — NDVI/NDWI are never used).

**(1) S1GFloods** (Saleh et al., 2024c) — 6 continents, 46 regions, **5,360 pairs** of Sentinel-1 SAR images, 2015–2022, **10 m resolution**, **256 × 256** images, split **3,216 train / 1,072 val / 1,072 test** (p. 515). Change ratio **32.32 %** (Fig. 2, p. 509).

**(2) ETCI-2021** — Sentinel-1, 5 geographic regions, **66,810 images**, 256 × 256. Authors "processed the dataset to ensure its balance" and extracted **10,745 image pairs: 7,145 train / 1,800 val / 1,800 test** (p. 515). Change ratio **1.23 %** — severely imbalanced.

**(3) VarFloods (new, contributed)** — **2,186 image pairs**, 256 × 256, random split **8:2** train/test (p. 515). Change ratio **5.74 %**. Released in two versions:
- **VarFloods-G**: original **Sentinel-1 IW GRD** products (raw).
- **VarFloods-P**: preprocessed — orbit correction, noise removal, radiometric calibration, terrain correction, conversion to **dB** scale.
- **VH polarization** selected for all regions (higher sensitivity to surface roughness and volume scattering than VV; cites Hamidi et al., 2023).

VarFloods events (Table 1, p. 516) — *note this includes Bolivia, a tropical South American analogue for Peru*:

| Continent | Location | Pre-flood | Post-flood | Cause | Inundation ratio (%) N / Y-preproc |
|---|---|---|---|---|---|
| South America | Bolivia | 02/27/2025 | 04/04/2025 | Heavy rain | 3.69 / 3.57 |
| Europe | France | 01/22/2025 | 02/03/2025 | Heavy rain | 10.39 / 9.53 |
| North America | Honduras | 11/04/2024 | 11/16/2024 | Tropical storm | 10.84 / 10.72 |
| Asia | Vietnam | 08/31/2024 | 09/12/2024 | Hurricane | 9.29 / 9.49 |
| Africa | Libya | 05/28/2023 | 09/13/2023 | Broken dam | 2.26 / 4.39 |

**Ground truth**: for the large-scale mapping experiments (Sec. 6.3, pp. 525–528) they use **Copernicus EMS Rapid Mapping products as the flood-extent ground truth**. For the patch-level VarFloods labels the annotation procedure is described only as "pixel-level annotations" (p. 528, in limitations) — ⚠️ the labelling protocol (who annotated, from what) is **not reported** in detail.

**⚠️ Inconsistencies found:**
- Table 7 (p. 523) lists "Number" = 889 for VarFloods-G and 1297 for VarFloods-P; 889 + 1297 = 2,186, which equals the *total* pair count stated on p. 515. But VarFloods-G and VarFloods-P are described as *the same scenes in two processing levels*, so each should hold ~2,186 (or ~1,093 if split). The two numbers cannot both be sample counts of parallel datasets — unclear/contradictory.
- Table 7 lists ETCI-2021 "Number" = 66,810, whereas Sec. 4 says only **10,745 pairs** were actually extracted and used. The two figures conflict.
- Table 1 gives two inundation ratios per event (raw vs preprocessed) but the header only labels one column; Libya jumps 2.26 → 4.39 % after preprocessing, which is a large change for the same event.
- Text says the Libya dam collapse "occurred on 10 September 2023" (p. 527) but Table 1's post-flood image is 09/13/2023 and pre-flood 05/28/2023 — consistent, though the pre-flood image is ~3.5 months earlier, far larger a gap than the other events (~12 days).

## Results

Metrics: **P, R, F1, IoU** (eqs. 42–45, p. 517). AWCA-Net + the 3 closest baselines were run **5 independent times** (mean ± std); other baselines are single runs.

**S1GFloods (Table 3, p. 517)** — best in bold:

| Method | P (%) | R (%) | F1 (%) | IoU (%) |
|---|---|---|---|---|
| FC-Cat | 88.63 | 91.33 | 89.96 | 81.75 |
| FC-Diff | 93.45 | 87.70 | 90.48 | 82.62 |
| BIT | 93.94 | 87.40 | 90.55 | 82.73 |
| UNet | 93.99 | 87.71 | 90.74 | 83.05 |
| HANet | 91.55 | 90.16 | 90.85 | 83.24 |
| CGNet | 92.34 | 91.39 | 91.87 | 84.96 |
| SNUNet | 92.54 | 91.23 | 91.88 | 84.98 |
| ChangFormer | 93.62 ± 1.19 | 91.09 ± 1.18 | 92.33 ± 0.28 | 85.75 ± 0.48 |
| ELGCNet | 94.84 ± 0.52 | 95.71 ± 0.31 | 95.28 ± 0.11 | 90.98 ± 0.20 |
| BAN | 96.18 ± 0.36 | 96.35 ± 0.32 | 96.26 ± 0.10 | 92.79 ± 0.19 |
| **AWCA-Net** | **96.91 ± 0.02** | **97.12 ± 0.03** | **97.02 ± 0.02** | **94.21 ± 0.05** |

**ETCI-2021 (Table 4, p. 518)**: AWCA-Net P **82.85 ± 0.75**, R **75.92 ± 0.67**, F1 **79.23 ± 0.04**, IoU **65.60 ± 0.06**. Best competitor BAN: P 86.08 ± 1.27 (BAN wins precision), R 71.31, F1 75.42, IoU 60.55. AWCA-Net improves **+1.68 IoU** over BAN, **+37.19 IoU** over FC-Diff, **+5.05 IoU** over BIT.

**VarFloods-G (Table 5, p. 518)**: AWCA-Net P **89.53 ± 0.35**, R **88.97 ± 0.45**, F1 **89.25 ± 0.09**, IoU **80.58 ± 0.14**. BAN 79.27 IoU, BIT 78.13, UNet only 33.68. → **+1.31 IoU** over BAN, **+40.1 IoU** over FC-Diff.

**VarFloods-P (Table 6, p. 518)**: AWCA-Net P **92.27 ± 0.19**, R **91.37 ± 0.30**, F1 **91.82 ± 0.06**, IoU **84.87 ± 0.11**. BAN 83.40, BIT 82.71. → **+1.47 IoU** over BAN, **+2.16** over BIT. **Preprocessing (VarFloods-P) beats raw GRD (VarFloods-G) by ~4.3 IoU points for AWCA-Net** — a clean, direct measurement of the value of the radiometric/terrain correction pipeline.

**Efficiency (Table 9, p. 529)**, 256×256 inputs:

| Method | Params (M) | FLOPs (G) |
|---|---|---|
| UNet | 1.35 | 3.57 |
| FC-Diff | 1.35 | 4.72 |
| FC-Cat | 1.55 | 5.32 |
| BIT | 3.50 | 10.61 |
| HANet | 3.03 | 20.82 |
| ELGCNet | 10.57 | 187.98 |
| SNUNet | 12.03 | 54.82 |
| CGNet | 38.99 | 87.55 |
| ChangeFormer | 41.02 | 202.62 |
| BAN | 68.88 | 29.21 |
| **AWCA-Net** | **35.21** | **17.53** |

AWCA-Net FLOPs = 8.7 % of ChangeFormer and 60 % of BAN; params = 51 % of BAN. Note it is **not** the smallest model — UNet/FC-Diff/BIT are far smaller; "lightweight" is relative to the transformer SOTA class.

**Ablation (Table 8, p. 529)** — A=NECM, B=LGDM, C=MSAWM, D=multi-scale BCEDice loss. IoU on Baseline+ → full:

| Config | S1GFloods IoU | ETCI-2021 IoU | VarFloods-G IoU | VarFloods-P IoU |
|---|---|---|---|---|
| Baseline+ | 90.53 | 51.88 | 71.76 | 78.48 |
| +A | 90.73 | 52.24 | 72.47 | 78.61 |
| +A +D | 89.80 | 57.15 | 72.79 | 78.58 |
| +A +B | 94.02 | 59.29 | 78.81 | 83.37 |
| +A +B +D | 93.76 | 64.91 | 79.95 | 84.43 |
| +A +B +C | 94.05 | 56.62 | 79.90 | 84.41 |
| **+A +B +C +D** | **94.21** | **65.60** | **80.58** | **84.87** |

⚠️ Note the ablation is **not monotone**: adding D to A *drops* S1GFloods IoU (90.73 → 89.80), and A+B+C is *worse* than A+B+D on ETCI-2021 (56.62 vs 64.91). **LGDM (B) is by far the biggest contributor** (+3.49 / +7.41 / +7.05 / +4.89 IoU across the four datasets, p. 527). NECM alone contributes only +0.2 / +0.36 / +0.71 / +0.13 IoU (p. 526) — nearly negligible.

**Per-scenario analysis (Figs. 12–13, pp. 523–524)** — VarFloods-G IoU by region: France 85.13, Libya 84.57, Vietnam 84.81, Honduras 79.73, **Bolivia 78.24** (lowest). By cause: broken dam 84.57, hurricane 84.81, tropical storm 79.73, **heavy rain 78.96** (lowest). By inundation ratio: 3.7 % → 78.39 IoU vs 10.1 % → 82.14 IoU. Same pattern on VarFloods-P (Bolivia 80.62 = lowest; heavy rain 82.72 = lowest; 3.6 % ratio → 80.97 vs 10.0 % → 87.56).

## Limitations

Author-stated (p. 528):
- "extreme class imbalance and noisy data, particularly in small-scale and rare flood events, can affect performance."
- The framework "relies on pixel-level annotations, which may introduce bias"; they suggest semi-/weakly-/unsupervised alternatives as future work.
- VarFloods "may have limitations in capturing certain flood dynamics, particularly in areas where flooding is driven by heavy rainfall (e.g., Bolivia) or tropical storms (e.g., Honduras)."
- "its robustness boundaries are still to be fully explored."

Observed by me:
- **No epochs / convergence criterion reported**; only iteration budgets, which differ 10× across datasets.
- **VarFloods is small** (2,186 pairs, 5 regions/events) and the test split is a **random 8:2 over patches**, not a held-out region or held-out event → risk of spatial autocorrelation leakage inflating the reported IoU. There is no cross-region generalization experiment (train on 4 regions, test on the 5th), which is the experiment their "diverse scenarios" framing most invites.
- Sample-count contradictions in Table 7 (see Data section).
- The three closest baselines got 5 runs; the other 7 got 1 run each — so most "improvements" are against single-run numbers.
- No uncertainty quantification / calibration; no timing (inference latency) despite the lightweight framing — only Params/FLOPs.

## Relevance to this thesis

Honest framing: this is **SAR (Sentinel-1), not Sentinel-2 optical**, and it is **inundation change detection (bi-temporal)**, not multi-year flood-susceptibility prediction. So it is **not a direct baseline** for the thesis. But it is a **strong method-to-cite** and the single best template available for the *bi-temporal pair* framing. Specifically:

- **Borrow the temporal-pair architecture pattern.** The Siamese weight-shared encoder + **BTFA fusion by `Conv3×3(Conv1×1(Concat(E_A, E_B)))`** (eq. 5, p. 510) transposes directly to a Sentinel-2 setting: feed pre-flood and post-flood NDVI/NDWI stacks through one shared CNN encoder and fuse by concat+conv. This is a cheaper and more defensible starting point than a ConvLSTM over the full 5-year series, and it gives a clean baseline to compare a time-series model against.
- **Borrow the multi-scale BCE+Dice loss** (eqs. 38–40, p. 514) verbatim: `Loss = Σ_i (BCE_i + 1 − Dice_i)` with deep supervision at 4 upsampled scales. Flood pixels in Peru will be a small minority class exactly as in ETCI-2021 (1.23 % change); BCE alone will collapse to the majority class. This is a **directly transferable, PyTorch-trivial** change.
- **Borrow the attention gate (LGDM).** The ablation shows it is the module that actually matters (+3.5 to +7.4 IoU, p. 527), and it is a small, self-contained block: grouped 3×3 conv on skip (E) and gate (G), then `Concat(G+E, G−E)` → 1×1 conv → sigmoid → multiply E. Drop it into the skip connections of a U-Net over NDVI/NDWI and cite this paper.
- **Borrow the evaluation protocol.** Report **P, R, F1, IoU** (their eqs. 42–45) — IoU is the headline. Also copy their **per-scenario breakdown** idea: report IoU by flood cause and **by inundation ratio**, because they demonstrate performance is strongly ratio-dependent (78.39 IoU @ 3.7 % vs 82.14 @ 10.1 %, Fig. 12d). A single aggregate IoU on a Peruvian dataset will hide exactly this.
- **Bolivia is the closest analogue to Peru in the literature I've seen so far** — tropical South America, heavy-rain-driven flooding, Feb–Apr 2025 event, 3.69 % inundation ratio. It is also **AWCA-Net's worst region** (78.24 IoU on GRD, 80.62 on preprocessed). This is a useful, quotable **calibration of expectations**: even a 2026 SOTA SAR model does worst on precisely the Andean/Amazonian heavy-rain regime the thesis targets. Cite this as motivation.
- **Cautionary example on preprocessing.** VarFloods-P beats VarFloods-G by ~4.3 IoU using the *same* scenes and model (Tables 5–6). The optical analogue is L2A atmospheric correction + cloud masking + normalization — this paper is quantitative evidence that the preprocessing step is worth several IoU points and should not be skipped or hand-waved.
- **Dataset source, indirectly.** VarFloods, S1GFloods and ETCI-2021 are all Sentinel-1; the thesis cannot use them as data. But the **Copernicus EMS Rapid Mapping products** they use as flood-extent ground truth (Sec. 6.3) *are* a viable label source for a Peruvian Sentinel-2 study — EMS activations exist for Peruvian floods. This is the most actionable data lead in the paper.
- **What NOT to borrow**: their backbone (PVT-v2-b2, 35 M params) is heavier than what a Streamlit/Celery demo on modest hardware needs; the thesis' CNN can stay small. Also, their random 8:2 patch split is a methodological weakness — the thesis should prefer a **spatially disjoint or event-disjoint split** and can say so, citing this as the thing being improved on.

## Keywords / Tech

- **Models**: AWCA-Net (proposed); baselines FC-Diff, FC-Cat, UNet, BIT, ChangeFormer, SNUNet, HANet, CGNet, ELGCNet, BAN, DAM-Net, LiST-Net (discussed).
- **Backbone**: PVT-v2-b2 (Pyramid Vision Transformer v2), Siamese/weight-shared.
- **Modules**: NECM (neighborhood feature enhancement + context), LGDM (large-kernel grouped attention gate on high–low difference), MSAWM (adaptive-window multi-scale conv attention = CAB → SAB → AMSCB), BTFA (bi-temporal feature aggregation), EUCB (efficient up-conv block), FRH (feature refinement head).
- **Loss**: multi-scale BCEDice (deep supervision at 4 scales).
- **Sensors**: Sentinel-1 IW GRD, **VH polarization**. No optical sensor, no NDVI/NDWI.
- **Datasets**: **VarFloods** (new; VarFloods-G raw / VarFloods-P preprocessed), S1GFloods, ETCI-2021; Copernicus EMS Rapid Mapping as flood-extent ground truth.
- **Preprocessing**: orbit correction, noise removal, radiometric calibration, terrain correction, dB conversion.
- **Frameworks**: PyTorch; RTX 3090; Adam; poly LR.
- **Code/data**: https://github.com/Dumh1998/AWCA-Net (stated "will be released").

## Notable quotes

> "Floods are highly destructive natural disasters that threaten both society and the environment. Given the all-weather, all-time imaging capability of synthetic aperture radar (SAR), analyzing flood events using SAR imagery across diverse scenarios is essential for developing high-precision and robust detection models." (p. 507, Abstract)

> "AWCA-Net improves the IoU by 11.59 % to 42.57 % over the basic model, 2.16 % to 11.48 % over an advanced transformer-based model, and 1.31 % to 1.68 % over the best comparative model, while maintaining a computational cost of only 17.53G, which is just 8.7 % to 60 % of existing SOTA models." (p. 507, Abstract)

> "Current models struggle with datasets featuring imbalanced flood-inundation ratios, often generating false detections in regions with limited flood extent while missing detections in areas with widespread flooding. This imbalance significantly affects model reliability, leading to inconsistent performance across different flood events." (p. 508)

> "AWCA-Net demonstrates strong performance across various flood scenarios but still faces challenges. Despite its overall effectiveness, extreme class imbalance and noisy data, particularly in small-scale and rare flood events, can affect performance. The current framework relies on pixel-level annotations, which may introduce bias." (p. 528)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/MTGDAPIX/Du et al. - 2026 - High-precision flood change detection with lightweight SAR transformer network and context-aware att.pdf`
Cite as: `\cite{du_high-precision_2026}`
