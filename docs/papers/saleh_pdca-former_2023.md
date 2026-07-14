---
title: "PDCA-Former: Prior-Diagonal Cross Attention-Guided Transformer for Flood Mapping from SAR Imagery: A Case in Khartoum"
authors: "Tamer Saleh, Mohamed Zahran, Shimaa Holail, Gui-Song Xia"
year: 2023
venue: "ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, Vol. X-1/W1-2023 (ISPRS Geospatial Week 2023, Cairo, Egypt), pp. 723-730"
doi: "10.5194/isprs-annals-X-1-W1-2023-723-2023"
bibtex_key: saleh_pdca-former_2023
zotero_pdf: "/Users/gersongarrido/Zotero/storage/EFT6MMIR/Saleh et al. - 2023 - PDCA-FORMER PRIOR-DIAGONAL CROSS ATTENTION-GUIDED TRANSFORMER FOR FLOOD MAPPING FROM SAR IMAGERY A.pdf"
pages: 8
sensors: [Sentinel-1, SRTM]
task: "bi-temporal flood inundation mapping (change detection) from SAR"
model: "Siamese ResNet-50 encoder + Diagonal-Cross Attention Module (DCAM) + CSWin Transformer decoder"
tags: [flood-mapping, sar, sentinel-1, change-detection, transformer, cross-attention, siamese, khartoum, dice-loss, isprs]
---

# PDCA-Former: Prior-Diagonal Cross Attention-Guided Transformer for Flood Mapping from SAR Imagery: A Case in Khartoum

> **TL;DR** — The authors propose PDCA-Former, a bi-temporal (pre-/post-flood) change-detection network for SAR flood mapping: a Siamese ResNet-50 "Prior Siamese Feature Extraction" (PSFE) backbone, a novel Diagonal-Cross Attention Module (DCAM) that combines a cross-attention module (CAM, horizontal+vertical context) with a diagonal attention module (DAM), a CSWin Transformer decoder block, and a prediction head trained with weighted BCE + Dice loss. Evaluated on a self-built Khartoum (Sudan) Sentinel-1 dataset from the September 2020 Nile flood (145/42/39 train/val/test pairs of 512×512), it reaches **OA 98.2%, F1 88.9% ± 0.018, IoU 85.7%** (p. 727, Table 2 p. 729), beating BiT-Former (IoU 82.7%) and a CNN baseline (IoU 79.9%). The method also estimates a total inundated area of **~348.253 sq-km** (p. 728).

## Problem & Motivation
Floods cause loss of life, infrastructure damage and economic disruption; Sudan's 2020 flood affected 17 states, >650,000 people and >100 fatalities, with Khartoum and Omdurman badly hit (p. 723). SAR is preferred over optical imagery for rapid flood response because it penetrates clouds and works day/night (p. 723). Prior Khartoum-area studies relied on DEM/hydraulic modeling plus optical data, or on thresholding of SAR, but thresholding is "extremely challenging due to the complex characteristics of SAR images... susceptible to noise interference, lack spatial consistency, and are unable to handle complex nonlinear issues" (p. 723).

Deep learning (CNNs such as CMCD-Net, CNN-Unet, FWENet, attention U-Net) improves on this but "still struggle to connect distant spatio-temporal concepts due to the limitation of the size of the receptive field" (p. 724). Existing flood transformers (FloodTransformer, BiT-STANet-SNUNet, FI-Former, DA-Transformer, Trans-SANet+, DAM-Net) are criticised as "effective in capturing temporal information by ignoring the spatial interactions of neighboring pixels" (p. 724). PDCA-Former is proposed to collect contextual information from multiple receptive fields simultaneously while reducing the complexity of full self-attention.

**Relation to `saleh_high-precision_2024`**: same first author (Tamer Saleh, LIESMARS Wuhan / Benha University) and the same research line. This 8-page ISPRS Annals conference paper is the *earlier / precursor* work; it also cites the authors' own DAM-Net (Saleh et al., 2023, arXiv:2306.00704) as related work. Expect the 2024 paper to be the extended, larger-scale journal version of the same attention-based SAR flood-mapping agenda.

## Method / Architecture
Three components (Fig. 2, p. 725):

- **Prior Siamese Feature Extraction (PSFE)** — two parallel **shared-weight ResNet-50** backbones (He et al., 2016) receive pre-flood image `t1` and post-flood image `t2`. Input X ∈ R^(H×W×C) passes a 7×7 conv + MaxPool, then 4 blocks (Conv→BN→ReLU ×2 each) producing multi-scale features; final block output X⁴ ∈ R^(H/8 × W/8 × 8C). A 1×1 conv on each block yields deep feature F. Feature maps in the figure: 512×512×C → 256×256×2C → 128×128×4C → 64×64×8C (p. 725). Outputs F1 (from t1) and F2 (from t2).
- **Diagonal-Cross Attention Module (DCAM)** — F1 goes to the **CAM** (cross-attention: contextual info of pixels in *horizontal and vertical* directions, criss-cross style, "greatly reduces the number of weights") → H1; F2 goes to the **DAM** (contextual info in *diagonal* directions) → H2 (p. 725, Fig. 3 p. 726). In CAM, Q, K ∈ R^(H×W×C′) (C′ < C, dimensionality reduced) and V ∈ R^(H×W×C) come from three convs on F1; the affinity operation gives attention map A_C ∈ R^((H+W−1)×W×H); φ1 ∈ R^((H+W−1)×C) collects the V vectors in the same row/column. Aggregation (Eq. 1, p. 725):

  `H1 = Σ_{i=0}^{H+W−1} A_C^i φ1^i + F1`

  H1 and H2 are each refined by a 3×3 conv → O1, O2, concatenated into the feature encoder **O12**, then a conv layer + split into embedding tokens **E12**.
- **Transformer Decoder Block (TDB)** — a **CSWin Transformer** block (Dong et al., 2022) with **shifting-window multi-head cross-attention (SW-MHCA)**: query Q from E12, key K and value V from H (Eq. 2, p. 726). LayerNorm before SW-MHCA and MLP, residual connections, MLP with three linear layers + GeLU. Output E′12.
- **Prediction Head (PH)** — E′12 → fully connected layer → **4× bi-linear upsampling** → Sigmoid → flood map (p. 726).

**Loss** (p. 726): class-imbalance-aware **weighted binary cross-entropy** (Eq. 3, with ω_c weights for changed pixels and ω_uc for unchanged pixels — the actual weight values are **not reported**) **plus Dice loss** (Eq. 4):
`L_T(Pr, GT) = L_bce(Pr, GT) + L_dice(Pr, GT)` (Eq. 5).

**Training hyperparameters** (p. 727): PyTorch 1.7.2, CUDA 10.1, cuDNN 7.6.1, Windows 10 Pro; NVIDIA RTX8000-8Q GPU with 8 GB (32 GB GPU memory mentioned — see ⚠️ below), Intel Xeon E5-2687W v4 @ 3.00 GHz. **300 epochs**, **Adam optimizer**, initial **learning rate 10e-5**, **batch size 4** for both decoder variants. Test images sized **512×512 with the EfficientNetv2 decoder** and **448×448 with the CSWin decoder**. Validation after each epoch; best validation model used for testing.

**Augmentation** (p. 727, "due to the Sudan dataset's small size"): random flip, random rotation, histogram matching, Gaussian blur, clipping.

## Data
- **Sensor**: Sentinel-1B GRD, Level-1, **IW mode**, 250 km swath, **VV polarization only**, range & azimuth resolution 10 m, descending pass. Ancillary: **SRTM 3 arc-second DEM** for geometric correction (Table 1, p. 727).
- **Pre-flood**: 13-Jul-2020, orbit 22447, 1 image, size 13558 × 18260 px. **Post-flood**: 23-Sep-2020, orbit 23497, 1 image, size 13560 × 18260 px (Table 1, p. 727). ⚠️ The two scene widths differ by 2 px (13558 vs 13560) — trivial but worth noting.
- **Downloaded from the Alaska Satellite Facility (ASF)** for 13 Jul – 23 Sep 2020 (p. 727).
- **Study area**: Khartoum, Sudan — confluence of the Blue Nile and White Nile, lat 15–16° N, lon 32–33° E, ~386 m elevation; rainfall season June–September, ~135 mm average 2017–2020 (p. 727).
- **Patches**: **512 × 512** px, projected to WGS84 at **10 m** ground resolution. **145 pairs for training, 42 for validation, 39 for testing** (p. 727). Two label classes: **no-flood** and **flood**.
- **Ground truth**: the paper says only "The label chunk contains two types of pixels: no-flood and flood" (p. 727) — **how the ground truth was produced (manual annotation? thresholding? which analyst?) is not reported.** This is a significant gap.
- **Preprocessing**: geometric correction using SRTM DEM. **No speckle filtering, radiometric calibration, or normalization procedure is described** — not reported.
- **Spectral indices**: **none** — SAR-only, no NDVI/NDWI (the study uses VV backscatter directly).
- **Public dataset name**: none; the "Khartoum dataset" is self-built and no availability statement is given.

## Results
All numbers from **Table 2, p. 729** (test set; σ = standard deviation). ⚠️ The table caption states "All values are reported as percentages (%)" but the cells are printed as decimals (0.955, 0.889…) — read them as fractions/percentages accordingly (0.889 = 88.9%). The abstract and §3.4 restate F1 = 88.9% and IoU = 85.7%, consistent with the table.

| Method | Encoder | Decoder | P | R | OA | F1 ± σ | IoU |
|---|---|---|---|---|---|---|---|
| CNN (Zhang & Xia, 2022) | ResNet-50 | EfficientNetv2 | **0.955** | 0.646 | 0.979 | 0.771 ± 0.026 | 0.799 |
| CNN (Zhang & Xia, 2022) | ResNet-50 | CSWin | 0.943 | 0.627 | 0.964 | 0.753 ± 0.028 | 0.783 |
| BiT-Former (Dong et al., 2023) | ResNet-50 | EfficientNetv2 | 0.905 | 0.663 | 0.963 | 0.765 ± 0.029 | 0.801 |
| BiT-Former (Dong et al., 2023) | ResNet-50 | CSWin | 0.910 | 0.782 | 0.976 | 0.841 ± 0.024 | 0.827 |
| **Ours (PDCA-Former)** | ResNet-50 | EfficientNetv2 | 0.944 | 0.778 | 0.968 | 0.853 ± 0.022 | 0.810 |
| **Ours (PDCA-Former)** | ResNet-50 | **CSWin** | 0.952 | **0.834** | **0.982** | **0.889 ± 0.018** | **0.857** |

- Headline: PDCA-Former + CSWin achieves **OA 98.2%, F1 88.9%, IoU 85.7%** (p. 727).
- **Margin over the best baseline**: +3.0 IoU points and +4.8 F1 points over BiT-Former + CSWin (IoU 82.7%, F1 84.1%); the CNN + ResNet-50 + CSWin variant is worst (F1 77.1% ⚠️ — §3.4 text on p. 728 says "an F1-score of 77.1% and an IoU of 79.9%", but Table 2 lists the ResNet-50 + CSWin CNN row as F1 **75.3%** / IoU **78.3%** and the ResNet-50 + EfficientNetv2 CNN row as F1 **77.1%** / IoU **79.9%**. The text mislabels which CNN variant is "worst"; the 77.1/79.9 pair belongs to the *EfficientNetv2* CNN row).
- **Precision is the one metric PDCA-Former does not win**: CNN + EfficientNetv2 has the highest precision (0.955 vs 0.952) — PDCA-Former wins by having far higher recall (0.834 vs 0.646).
- Qualitative (Fig. 7, p. 728): fewer false positives (magenta) and fewer missed detections (cyan) than CNN and BiT; correctly detects a newly emerged island in the river and performs well over complex farmland.
- **Flood inundation extraction** (p. 728, Fig. 8 p. 729): total inundation estimated at **~348.253 sq-km**.
- **Baselines run**: yes — two (a CNN from Zhang & Xia 2022, and BiT-Former from Dong et al. 2023), each with two decoders. **No ablation study of DCAM/CAM/DAM in isolation is reported** — the encoder/decoder swap is the only controlled comparison, so the individual contribution of the diagonal attention branch is unverified.

## Limitations
Stated by the authors:
- Only one flood type/scene studied; "Future work will include developing a more diverse dataset that includes different categories of floods, such as open and urban floods, to evaluate the PDCA-Former network's performance on different flood types" (p. 729).
- The Sudan dataset is explicitly "small" (p. 727), requiring heavy augmentation.

Observed by me:
- **Single study area, single event, single date pair** — no cross-region or cross-event generalization test, despite the abstract's claim that the method "can be quickly generalized to other regions" (p. 723). That claim is unsupported by any experiment.
- **Test set is only 39 image pairs** from the same scene as training — spatially adjacent tiles from the same Sentinel-1 acquisition means train/test are almost certainly not spatially independent; leakage is plausible.
- **Ground-truth generation is never described**, which makes the 98.2% OA hard to interpret.
- **No ablation** isolating DCAM, CAM vs DAM, or the prior-Siamese design.
- **No preprocessing details** (speckle filtering, calibration) — hard to reproduce.
- **Class imbalance**: OA is dominated by the huge unchanged/no-flood class; IoU/F1 are the meaningful metrics here.
- ⚠️ Contradictory hardware description (p. 727): "an NVIDIA RTX8000-8Q with 8GB memory ... and 32.0 GB of GPU memory".
- ⚠️ Two different test patch sizes (512×512 vs 448×448) depending on decoder, which makes the decoder comparison not perfectly controlled.
- Dataset is not released (no link, no availability statement).

## Relevance to this thesis
Honest framing: **this is a SAR-only, bi-temporal change-detection paper — a method paper to cite and an architectural donor, not a baseline you can reproduce or a dataset you can use.** Your thesis is Sentinel-2 optical, NDVI/NDWI time series, 10 m, Peru. Nothing here transfers at the data level.

What *is* actionable:
- **Loss function — borrow directly.** `L = weighted BCE + Dice` (Eqs. 3–5, p. 726) is exactly the right recipe for your setting: flooded pixels are a small minority of any Sentinel-2 tile, and plain BCE will collapse to predicting "no flood". This is a cheap, high-value import into your PyTorch CNN. Cite this paper (and note the paper does **not** report the ω_c / ω_uc values — you will have to tune them, e.g. inverse class frequency).
- **Siamese pre/post bi-temporal framing — a design pattern to consider.** Instead of feeding a single-date NDVI/NDWI stack, feed a *pair* (pre-flood, post-flood) through shared-weight encoders and let the network do change detection. Your Sentinel Hub pipeline already downloads multi-date imagery, so a Siamese pre/post NDWI pair is a low-cost architectural variant worth trying against your ConvLSTM/CNN-over-time-series baseline.
- **Metric set to mirror**: report **Precision, Recall, OA, IoU, F1** (Eq. 6, p. 726) — and report **± standard deviation** across runs, as they do. Do **not** lead with OA; their own table shows OA 96.4–98.2% across methods whose IoU spans 78–86%, a perfect illustration of why OA is near-useless under class imbalance. Use this as your justification for foregrounding IoU/F1 in the thesis.
- **A number to compare against (carefully)**: IoU 85.7 / F1 88.9 on SAR flood extent. If your Sentinel-2 model lands well below this, the honest explanation is data (cloud-limited optical, no post-event acquisition guaranteed), not architecture. Use it as a *reference point in the literature table*, not as a baseline you claim to beat.
- **Cautionary example**: the paper claims cross-region generalization while testing on 39 tiles from a single scene. Your thesis should explicitly do what this paper did not — hold out a **spatially disjoint** region/event. Cite this as motivation for your evaluation protocol.
- **Attention mechanism (DCAM)** — the criss-cross + diagonal attention trick is a cheap way to get near-global context at sub-quadratic cost. Relevant if you go beyond a plain CNN, but it is probably over-engineering for a thesis-scale Sentinel-2 model; mention in related work rather than implement.
- **Argument for SAR in the related-work chapter**: the paper's clean statement of why SAR beats optical for flood *response* (clouds, night) is exactly the counter-argument your Sentinel-2 thesis must address. Use it to justify why your framing is **flood-prone-area prediction/susceptibility over time** (where optical time series and vegetation/water indices matter) rather than **rapid post-event inundation mapping** (where SAR wins).
- Also cross-reference with `saleh_high-precision_2024` (same first author): treat this as the precursor/short version and cite the pair together when discussing attention-based SAR flood mapping.

## Keywords / Tech
- **Models**: PDCA-Former; Siamese ResNet-50 (PSFE); Diagonal-Cross Attention Module (DCAM) = CAM (criss-cross) + DAM (diagonal); CSWin Transformer decoder; EfficientNetv2 decoder; SW-MHCA. Baselines: CNN (Zhang & Xia 2022), BiT-Former (Dong et al. 2023).
- **Sensors**: Sentinel-1B GRD (IW, VV, 10 m, descending); SRTM 3-arcsec DEM.
- **Indices**: none (SAR backscatter; no NDVI/NDWI).
- **Loss**: weighted binary cross-entropy + Dice.
- **Frameworks**: PyTorch 1.7.2, CUDA 10.1, cuDNN 7.6.1.
- **Datasets**: self-built "Khartoum dataset" (145/42/39 pairs of 512×512, not public); source data from Alaska Satellite Facility (ASF).
- **Techniques**: bi-temporal change detection; shared-weight Siamese encoders; criss-cross/diagonal attention; shifted-window cross-attention; augmentation (random flip, rotation, histogram matching, Gaussian blur, clipping); Adam, 300 epochs, LR 10e-5, batch 4.

## Notable quotes
- "PDCA-Former adopts Prior Siamese Feature Extraction (PSFE) to extract multi-scale deep features from the input SAR images. Additionally, we propose a novel Diagonal Cross-Attention Module (DCAM) to capture relational information of all pixel positions on the entire image." (p. 723, Abstract)
- "The experimental results show that PDCA-Former outperforms the latest comparator methods in terms of F1 by 88.9% and IoU by 85.7%." (p. 723, Abstract) — note the phrasing is sloppy: these are the *absolute* F1/IoU values, not the margin of improvement.
- "Since the distribution of change and non-change classes is often highly unbalanced in the detection task, the loss function may be biased towards the class with larger samples during training, resulting in lower class recognition accuracy for the class with smaller samples. Therefore, we use the weighted entropy function to train the network... Furthermore, the dice loss is employed to measure the similarity between the predicted change map Pr and the ground truth GT." (p. 726)
- "The total area of inundation was estimated to be approximately 348.253 sq-km. The proposed method provides a reliable means of quantifying the extent of inundation, allowing for informed decision-making by relevant authorities." (p. 728)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/EFT6MMIR/Saleh et al. - 2023 - PDCA-FORMER PRIOR-DIAGONAL CROSS ATTENTION-GUIDED TRANSFORMER FOR FLOOD MAPPING FROM SAR IMAGERY A.pdf`
Cite as: `\cite{saleh_pdca-former_2023}`
