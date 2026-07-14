---
title: "High-precision flood detection and mapping via multi-temporal SAR change analysis with semantic token-based transformer"
authors: "Tamer Saleh, Shimaa Holail, Xiongwu Xiao, Gui-Song Xia"
year: 2024
venue: "International Journal of Applied Earth Observation and Geoinformation, vol. 131, article 103991, 13 pp."
doi: "10.1016/j.jag.2024.103991"
bibtex_key: saleh_high-precision_2024
zotero_pdf: "/Users/gersongarrido/Zotero/storage/GCP6C4H4/Saleh et al. - 2024 - High-precision flood detection and mapping via multi-temporal SAR change analysis with semantic toke.pdf"
pages: 13
sensors: [Sentinel-1]
task: "flood inundation mapping via bi-temporal SAR change detection (binary change detection)"
model: "SemT-Former (Siamese ResNet-50 encoder + semantic-token transformer + CSwin/EfficientNetv2 decoder)"
tags: [sar, change-detection, transformer, semantic-token, flood-mapping, sentinel-1, khartoum, siamese, dice-loss]
---

# High-precision flood detection and mapping via multi-temporal SAR change analysis with semantic token-based transformer

> **TL;DR** — Saleh et al. propose **SemT-Former**, a Siamese semantic-token transformer for bi-temporal Sentinel-1 SAR change detection, applied to the September 2020 Khartoum (Sudan) flood and the 2022 Lokoja (Nigeria) flood. It combines a shared-weight ResNet-50 backbone (TDFR), a criss-cross-style *temporal-sensitive change attention* (TSCA), a *cross-time change enhancement* (CTCE) module that injects a learnable **class/semantic token**, and a *differential feature fusion* (DFF) head, trained with BCE + Dice loss. On the Khartoum test set with a CSwin decoder it reaches **F1 = 90.6 ± 1.5 %, IoU = 88.5 %, OA = 98.8 %, P = 98.6 %, R = 83.8 %** (p. 9, Table 3), beating BiT-Former (F1 88.4, IoU 86.3) and FC-Siam-Diff (F1 81.5, IoU 80.1). Estimated inundation: **~351.178 sq km** in Khartoum and **~301.572 sq km** in Lokoja (p. 10).

⚠️ **Abstract wording is misleading.** The abstract says SemT-Former "exhibit[s] a 90.6% improvement in F1-score and an 88.5% enhancement in IoU" (p. 1). Those are the **absolute** F1 and IoU values, not improvements. The actual gains over the best baseline (BiT-Former, CSwin) are ~+2.2 F1 points and ~+2.2 IoU points. Do **not** cite "90.6% improvement".

## Problem & Motivation
Optical imagery cannot see through clouds — exactly the condition during floods (Fig. 1 shows a Sentinel-2 image over Nebraska rendered useless by cloud, vs a clear Sentinel-1 SAR view, p. 2). SAR is therefore the sensor of choice, but SAR has "limited visual information, pervasive speckle noise, and analogous backscatter signals" (p. 1) which makes water/non-water discrimination and change extraction hard. Classical thresholding (e.g. NDFI-style automatic thresholds) is noise-sensitive and lacks spatial consistency; CNN change-detection models are limited by receptive-field size and miss long-range/global context, which matters because floods "typically span a large spatial range" unlike building change (p. 3).

The authors argue that ViT-based change detection has been developed almost exclusively for **optical** imagery and that "there is currently no literature available specifically examining the use of ViTs for flood detection in SAR images" (p. 4). SemT-Former is their answer: prioritize *changes of interest* rather than fully modelling the whole scene, using a semantic token that encodes high-level segmentation of water-body change to suppress spurious changes from speckle/similar backscatter.

## Method / Architecture
Overall: sharing-weights **dual-branch (Siamese)** network on pre-flood (T1) and post-flood (T2) images, plus a CSwin transformer decoder per branch, a Differential Feature Fusion block, and a prediction head with 4× upsampling (Fig. 2, p. 4). Output M ∈ {0,1}^{H×W} (1 = change/flooded, 0 = non-change).

Four modules:
- **TDFR — Temporal-Driven Feature Representation** (p. 4, Fig. 3): two parallel pre-trained **ResNet-50** backbones (He et al., 2016), shared weights. Input goes through a 7×7 conv + max-pool, then B^n blocks (n ∈ {1,2,3,4}) of Conv+BN+ReLU, with down-sampling producing multi-scale hierarchical features; first block output X^4 ∈ R^{(H/8)×(W/8)×8C}. Residual connection of the input scene added element-wise to the final block output → R^i.
- **TSCA — Temporal-Sensitive Change Attention** (p. 5, Fig. 4, Eq. 1): a criss-cross-style attention. Q, K, V produced by three conv layers from R_pre; attention map A_C ∈ R^{(H+W−1)×W×H} computed as an affinity between each pixel and the pixels in **its row and column only** (H+W−1 positions), i.e. contextual information over the full image at reduced self-attention complexity. F^i_pre = Σ_{i=0}^{H+W−1} A^i_C φ^i_A + R_pre.
- **CTCE — Cross-Time Change Enhancement** (p. 5, Fig. 5, Eq. 2): MHSA where **R_pre is the query** and F_pre is key and value, then FFN, each with residual: T^i = MHSA_i(R^i_pre, F^i_pre) + R^i_pre ; F^{e,i}_pre = FFN_i(T^i) + T^i. A learnable **class token Sem_T** is concatenated to R^i_pre and fed into the MHSA as part of the query, so the token accumulates high-level semantic information about water-body change. Built on the **CSwin Transformer** (Dong et al., 2022).
- **DFF — Differential Feature Fusion** (p. 5, Fig. 6, Eq. 3): absolute difference |F^{e,i}_pre − F^{e,i}_post| at 4 scales (i = 1..4) → per-scale Conv → concatenation → Conv_{1×1} → element-wise multiplication with mlp(Sem_T) → sigmoid: F_fused = Cat[Conv_i(|F^{e,i}_pre − F^{e,i}_post|)], F_enhanced = Conv_{1×1}(F_fused)·mlp(Sem_T), M = σ(g(F_enhanced)), where g(·) is the prediction head.

**Loss** (p. 6, Eqs. 4–6): weighted **binary cross-entropy** (class weights ω_c, ω_u to counter the change/non-change imbalance) **+ Dice loss**: L = L_bce + L_dice. Rationale: Dice fluctuates badly when the change region is small, BCE stabilizes gradients.

**Training details** (p. 8, §4.2.2):
- PyTorch 1.7.2, cuDNN 7.6.1, CUDA 10.1; NVIDIA RTX8000-8Q (32 GB), Xeon E5-2687W; Windows 10 Pro.
- **100 epochs**, **Adam**, **LR = 1e-5**, **batch size 8** for both decoders.
- Input size **512×512 with the EfficientNetv2 decoder**, **448×448 with the CSwin decoder**.
- **Augmentation** (because the Sudan dataset is small): random flip, random rotation, **histogram matching**, Gaussian blur, clipping.
- Validation after each training epoch; best model kept and used on the test set.
- Code promised at `https://github.com/Tamer-Saleh/SemT-Former` (p. 2).

## Data
- **Sensor**: Sentinel-1B **SAR only** (no optical used for the model; Sentinel-2 appears only as an illustration of cloud occlusion in Fig. 1). Level-1 **GRD**, **IW** mode, **swath 250 km**, **range and azimuth resolution 10 m**, **VV polarization only** (VV = vertical transmit / vertical receive) (p. 6, §4.1.2).
- **Table 1 (p. 6) — Khartoum 2020**: Pre-flood **July 2020**, orbit 22 447, 2 images, descending, size **13558 × 18260 px**; Post-flood **Sept. 2020**, orbit 23 497, 2 images, descending, size **13560 × 18260 px**. DEM: **SRTM 3Sec** (ancillary, for geometric correction).
  - ⚠️ Minor inconsistency: text says Sudan SAR data cover **13 July – 23 September 2020**, while Table 1 lists "July, 2020" / "Sept., 2020". Also Fig. 8 caption says the pre-event image was acquired in July 2020 and the post-event on 23 September 2020.
- **Study areas** (p. 6): **Khartoum, Sudan** (confluence of the Blue and White Nile; 14°30′N–15°30′N, 32°00′E–32°30′E; 386 m a.s.l.), flooded September 2020 after heavy rain. **Lokoja, Nigeria** (Niger river confluence), flood of 13 Oct 2022; Nigeria SAR data span **26 August – 13 October 2022**. Lokoja is used **only for inference/transfer**, not for training or quantitative evaluation.
- **Preprocessing** (p. 6, §4.1.3) — all in ESA **SNAP / python-snappy**: thermal noise removal → radiometric calibration → terrain correction → **Lee filter** for speckle → normalization of all SAR images to **grey scale 0–255** → projection to **WGS84**. For water–land segmentation they apply "a consistent thresholding approach, setting the segmentation threshold of **σ₀ = −16 dB** across all image pairs" to help separate flooded/non-flooded areas.
- **Ground truth**: images **manually annotated** into flooded / non-flooded; non-flooded pixel value 0, flooded 255 (p. 7).
- **Patches / split**: crops of **512 × 512 px**, divided **70 % train / 15 % validation / 15 % test** (p. 7).
- **Number of patches: not reported.** Total number of training patches, per-class pixel counts, and the exact Khartoum/Lokoja tile counts are not given.
- **Spectral indices: none.** No NDVI/NDWI/NDFI is computed by the model (NDFI is only mentioned when describing Haile et al., 2023 in the intro). The only radiometric rule is the −16 dB σ₀ threshold above.

## Results
Metrics defined in Eq. 7 (p. 7): Recall, Precision, OA, IoU, F1.

**Table 3 (p. 9) — Khartoum test set, CSwin decoder** (best in bold in the paper):

| Method | P (%) | R (%) | OA (%) | F1 ± σ (%) | IoU (%) |
|---|---|---|---|---|---|
| FC-Siam-Diff (Daudt et al., 2018) | 97.3 | 70.1 | 98.5 | 81.5 ± 2.3 | 80.1 |
| DASNet (Chen et al., 2020) | **98.7** | 74.5 | 98.2 | 84.9 ± 2.5 | 82.7 |
| AFDE-Net (Holail et al., 2023) | 98.5 | 77.6 | **98.9** | 86.8 ± 2.1 | 84.4 |
| SwinSUnet (Zhang et al., 2022b) | 97.4 | 78.1 | 98.5 | 86.7 ± 1.8 | 85.1 |
| BiT-Former (Dong et al., 2023) | 97.9 | 80.6 | 97.8 | 88.4 ± 1.7 | 86.3 |
| **SemT-Former (Ours)** | 98.6 | **83.8** | 98.8 | **90.6 ± 1.5** | **88.5** |

**Table 2 (p. 8) — same test set, EfficientNetv2 decoder**:

| Method | P (%) | R (%) | OA (%) | F1 ± σ (%) | IoU (%) |
|---|---|---|---|---|---|
| FC-Siam-Diff | **97.7** | 65.2 | 98.1 | 78.2 ± 2.5 | 80.2 |
| DASNet | 97.3 | 68.9 | 98.4 | 80.7 ± 2.1 | 79.9 |
| AFDE-Net | 97.1 | 70.2 | 97.9 | 81.5 ± 2.2 | 80.1 |
| SwinSUnet | 96.8 | 69.5 | 98.1 | 80.9 ± 2.4 | 81.2 |
| BiT-Former | 96.6 | 73.2 | **98.8** | 83.3 ± 1.8 | 81.7 |
| **SemT-Former (Ours)** | 97.5 | **77.1** | 98.5 | **86.1 ± 1.7** | **82.4** |

- ⚠️ **Text/table mismatch**: §4.4 (p. 8) says "SemT-Former achieved the highest OA, F1-score, and IoU of 98.8%, 90.6%, and 88.5%… when using a Resnet-50 encoder and CSwin decoder", but Table 3 shows AFDE-Net has the highest OA (98.9 %) and DASNet the highest precision (98.7 %). The same paragraph also claims "the BiT-Former with the CSwin decoder … had the second-best performance with an IoU of 86.3%" — consistent — but then says "FC-Siam-Diff … performed the worst, with F1-score (81.5%) and IoU (80.1%)", while Table 2's worst F1 is FC-Siam-Diff at 78.2 %. So "beats all methods in all measures" is **not** true; it wins on F1, IoU and Recall only.
- **Gains over best baseline** (BiT-Former, CSwin): +2.2 F1 pts, +2.2 IoU pts. The conclusion states "SemT-Former also demonstrated a 2.2% improvement in terms of IoU score when employing an EfficientNetv2 decoder" (p. 11) — note EfficientNetv2 IoU gain in Table 2 is 82.4 − 81.7 = **0.7 pts**, not 2.2. ⚠️ Another inconsistency; 2.2 matches the **CSwin** IoU gain.
- **CNN vs transformer**: "CNN-based change detection methods exhibit relatively lower performance on the Khartoum dataset compared to competitive vision transformers … (with an 8.4% reduction in terms of IoU score)" (p. 11).
- **ROC/AUC** (p. 11, Fig. 13): SemT-Former has the highest **AUC = 97.7 %**; lowest is FC-Siam-Diff at **95.3 %** (a 2.4 pt spread).
- **Convergence** (p. 9, Fig. 10): most models converge quickly with CSwin; SemT-Former's F1 curve fluctuates notably in the **first 40 epochs** ("initial instability").
- **Area estimates** (p. 10): Khartoum inundation **≈ 351.178 sq km**; Lokoja **≈ 301.572 sq km**. Lokoja has **no ground truth**, so "the lack of ground truth data for the test area impedes a quantitative evaluation of the model" (p. 10).
- **No ablation study** of TSCA / CTCE / DFF / the semantic token, and **no comparison against the authors' own PDCA-Former**, despite it being cited.

### Relationship to `saleh_pdca-former_2023`
Same first author, same study area (Khartoum, Sudan), same sensor (Sentinel-1 SAR), same task (bi-temporal flood change detection). PDCA-Former is cited in the related-work list of transformer-based flood methods as "PDCA-Former (Saleh et al., 2023b)" (p. 2), alongside the authors' DAM-Net (Saleh et al., 2023a). **SemT-Former is a successor/sibling architecture, not an extension of PDCA-Former** — the modules are new (TDFR/TSCA/CTCE/DFF, semantic token, CSwin decoder) and PDCA-Former is **not included in the baseline tables**, so the paper does **not** demonstrate that SemT-Former supersedes it numerically. Treat this as "same group, later paper, no head-to-head". Their AFDE-Net (Holail et al., 2023, with Saleh as co-author) *is* in the baselines and is beaten (F1 86.8 → 90.6).

## Limitations
Stated by the authors:
- Dataset is small ("Due to the Sudan dataset's small size, we employed data augmentation", p. 8).
- No ground truth for the Lokoja test area → no quantitative validation of transfer (p. 10).
- Initial training instability / F1 fluctuation in first ~40 epochs; "further enhancement of the generalization capability of the SemT-Former method is needed" (p. 9).
- Future work: "development of a more diverse dataset encompassing various flood categories, including open and urban floods" (p. 11) — i.e. the model has only been shown on **riverine/agricultural** flooding.

Observed by me:
- **Single training site** (Khartoum). All quantitative results come from one flood event, one river confluence, one polarization (VV), one orbit direction (descending). Train/val/test are random 512×512 crops from the **same scene pair** → strong spatial autocorrelation between train and test; the 88.5 % IoU is almost certainly optimistic for a truly held-out event.
- **No ablation**, so the contribution of the headline "semantic token" is unquantified.
- Manual annotation of ground truth by the authors, on the same SAR images the model sees → label noise and circularity risks; no independent validation (no field data, no optical-derived reference).
- Recall (83.8 %) is far below precision (98.6 %) — the model **misses ~16 % of flooded pixels**. For a disaster-management use case, under-detection is the costly error.
- No uncertainty quantification, no per-class breakdown, no computational cost / inference-time table.
- ⚠️ Several numeric inconsistencies between text, abstract and tables (flagged above).

## Relevance to this thesis
**Overall: MEDIUM relevance — SAR, not optical; inundation mapping, not susceptibility/prediction.** It is a *method-and-framing* citation, not a baseline you can run against.

Directly borrowable:
- **Loss function**: `L = weighted BCE + Dice` (Eqs. 4–6, p. 6). This is the single most transferable item. Flood pixels are a small minority class in NDVI/NDWI patches too, and the paper's reasoning (Dice unstable on small change regions, BCE stabilizes) applies verbatim to a PyTorch CNN on Sentinel-2. Easy to implement, easy to cite as justification.
- **Bi-temporal / Siamese framing**: pre-event vs post-event pair with a shared-weight encoder and an **absolute-difference feature fusion at multiple scales** (DFF, Eq. 3). If the thesis moves from a per-date NDWI classifier to a change-detection formulation over a Sentinel-2 time series, this is the reference architecture. The DFF pattern (|F_pre − F_post| per scale → conv → concat) is a ~20-line PyTorch block.
- **Augmentation set** for small flood datasets: random flip, random rotation, **histogram matching**, Gaussian blur, clipping (p. 8). Histogram matching is especially relevant to Sentinel-2 time series where radiometry drifts between dates.
- **Metric suite to report**: Precision, Recall, OA, **IoU**, F1 (± std over runs), plus **ROC/AUC**. Reporting σ over runs (they do: ±1.5 etc.) is good practice worth copying.
- **Split & patching convention**: 512×512 patches, 70/15/15 train/val/test — a defensible, citable convention. But see the caution below.
- **Adam, LR 1e-5, batch 8, 100 epochs** — a reasonable starting hyperparameter box for a transformer/CNN flood model.

Directly comparable / baseline value:
- **Not a usable baseline** — different sensor (Sentinel-1 VV SAR), different region (Sudan/Nigeria), non-public dataset ("Data will be made available on request", p. 12). The 90.6 F1 / 88.5 IoU numbers should **not** be compared to Sentinel-2 results in the thesis.
- The baselines *they* use (FC-Siam-Diff, DASNet, SwinSUnet, BiT, U-Net-family) are however a good shortlist of **change-detection architectures to consider or cite** for a Sentinel-2 bi-temporal setup.

Where it differs / cautions:
- **SAR vs optical**: their whole motivation (Fig. 1, p. 2) is that Sentinel-2 is unusable under flood-time cloud. This is a **cautionary citation the thesis must address head-on**: a Sentinel-2 + 10 %-cloud-max pipeline will often have *no* usable image at peak flood in Peru. Use this paper to justify either (a) framing the thesis as *susceptibility / flood-prone area prediction* from cloud-free time-series statistics rather than same-day inundation mapping, or (b) adding Sentinel-1 as a complementary source. Cite p. 1–2 for the cloud argument.
- **No spectral indices** here — NDVI/NDWI play no role; there is nothing to borrow on the index side. The only radiometric decision (σ₀ = −16 dB water threshold) has no optical analogue beyond "pick an NDWI threshold".
- **Cautionary example on evaluation**: train/val/test crops from the *same* scene pair. The thesis should avoid this and hold out a **separate event or a spatially disjoint region**, otherwise its IoU will be inflated in exactly the same way.
- **Cautionary example on reporting**: the abstract's "90.6% improvement in F1" is an absolute value dressed up as a gain. Good example of why the thesis should report deltas explicitly.
- The Lokoja section is a nice template for the **"apply the trained model to a new region and report inundated area in km²"** deliverable — which is essentially what a Peru case study would do (Khartoum ≈ 351.178 km², Lokoja ≈ 301.572 km², p. 10) — while also showing the honest admission that without ground truth you cannot quantify it.

## Keywords / Tech
- **Models**: SemT-Former (proposed); ResNet-50 backbone; CSwin Transformer decoder; EfficientNetv2 decoder; baselines FC-Siam-Diff, DASNet, AFDE-Net, SwinSUnet, BiT-Former.
- **Modules**: TDFR (temporal-driven feature representation), TSCA (temporal-sensitive change attention, criss-cross row+column affinity), CTCE (cross-time change enhancement, MHSA + class/semantic token), DFF (differential feature fusion), semantic/class token `Sem_T`.
- **Sensors**: Sentinel-1B GRD, IW mode, VV polarization, 10 m; SRTM 3Sec DEM (ancillary). Sentinel-2 mentioned only as a cloud-limitation illustration.
- **Indices**: none (σ₀ = −16 dB water threshold; NDFI mentioned only in related work).
- **Frameworks**: PyTorch 1.7.2, CUDA 10.1, cuDNN 7.6.1; ESA SNAP via python-snappy.
- **Techniques**: thermal noise removal, radiometric calibration, terrain correction, **Lee speckle filter**, WGS84 reprojection, 0–255 normalization; weighted BCE + Dice loss; Adam; histogram matching augmentation.
- **Datasets**: custom Khartoum (Sudan, 2020) and Lokoja (Nigeria, 2022) Sentinel-1 pairs from Alaska Satellite Facility; **not public** (on request).

## Notable quotes
- "SemT-Former operates by prioritizing changes of interest rather than fully comprehending the entire image scene. This is achieved through the integration of temporal-wise feature representation and the introduction of a class token to capture high-level segmentation associated with changes in water bodies." (p. 1)
- "To our best understanding, there is currently no literature available specifically examining the use of ViTs for flood detection in SAR images and demonstrating their superiority in SAR image segmentation." (p. 4)
- "Because optical remote sensing imagery has limitations, such as its inability to penetrate clouds and reliance on daylight operation, synthetic aperture radar (SAR) imagery is the ideal choice for flood detection due to its ability to operate effectively in adverse weather conditions." (p. 1)
- "While the model has shown promising results in detecting flooded areas in other study regions, the lack of ground truth data for the test area impedes a quantitative evaluation of the model, highlighting an area for future research." (p. 10)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/GCP6C4H4/Saleh et al. - 2024 - High-precision flood detection and mapping via multi-temporal SAR change analysis with semantic toke.pdf`
Cite as: `\cite{saleh_high-precision_2024}`
