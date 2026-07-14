# Literature Index — Flood Mapping with Satellite Imagery & Deep Learning

Memory notes for the 23 papers in the Zotero `tesis` collection. **Read this file first**; open the individual note for detail; open the PDF only if the note is insufficient.

- Notes: `docs/papers/<bibtex_key>.md` — each has a `zotero_pdf:` pointer to the original.
- Cite with `\cite{<bibtex_key>}` — all 23 keys already exist in `overleaf-thesis/3_3_BIBLIOGRAFIA/library.bib`.
- Regenerate the paper→PDF map: `./scripts/zotero_index.sh tesis`
- Writing conventions for these notes: `_INSTRUCTIONS.md`

> ⚠️ **Every metric below was copied verbatim from its paper, with page cites in the notes.** Where a paper does not report a metric, the note says `not reported` rather than guessing. Numbers marked ⚠️ are ones where the *paper contradicts itself* — check the note before citing.

---

## The comparison table

| Paper | Yr | Sensor | Model | Task | Headline metric | Relevance |
|---|---|---|---|---|---|---|
| [konapala_exploring_2021](konapala_exploring_2021.md) | 2021 | **S1 + S2** | U-Net | Inundation | **S2 F1 0.90** vs S1 F1 0.62 | 🔴 **critical** |
| [shastry_mapping_2023](shastry_mapping_2023.md) | 2023 | WorldView (optical) | CNN | Inundation | **62% underprediction**; 74% veg / 9% cloud | 🔴 **critical** |
| [li_terrainfloodsense_2025](li_terrainfloodsense_2025.md) | 2025 | HLS 30 m + DEM | Bayesian fusion | Inundation thru cloud | OA 94.31%, F1 0.878 | 🔴 **critical** |
| [ulloa_sentinel-1_2022](ulloa_sentinel-1_2022.md) | 2022 | S1 time series | **ConvLSTM** | Spatiotemporal | Kappa 0.93 (AUS) | 🔴 **critical** (temporal) |
| [tran_surface_2022](tran_surface_2022.md) | 2022 | S1 + S2 | **Otsu** (no DL) | Water mapping | R² 0.97 river / 0.88 paddy | 🟠 non-DL baseline |
| [pech-may_flood_2025](pech-may_flood_2025.md) | 2025 | S1 + S2 | U-Net | Inundation | Acc 92.14%, F1 89.38% | 🟠 closest setup |
| [garshasbi_uncertainty-aware_2025](garshasbi_uncertainty-aware_2025.md) | 2025 | S1 | **Bayesian U-Net** | Inundation + UQ | F1 0.8013, IoU 0.6685 | 🟠 uncertainty |
| [xu_sar_2022](xu_sar_2022.md) | 2022 | SAR | **Attention U-Net** | Water extraction | ⚠️ **no metrics reported** | 🟠 architecture |
| [huang_waterdetectionnet_2024](huang_waterdetectionnet_2024.md) | 2024 | S1 | WDNet (Xception+ASPP+attn) | Inundation | IoU 0.974, F1 0.987 | 🟠 architecture |
| [chen_flood_2025](chen_flood_2025.md) | 2025 | S1 | FIE-Net | Coastal inundation | IoU 79.44% | 🟠 **index-as-input** |
| [du_high-precision_2026](du_high-precision_2026.md) | 2026 | S1 | AWCA-Net (PVT-v2) | Change detection | IoU 94.21% (S1GFloods) | 🟡 method |
| [saleh_high-precision_2024](saleh_high-precision_2024.md) | 2024 | S1 | SemT-Former | Change detection | F1 90.6%, IoU 88.5% | 🟡 method |
| [saleh_pdca-former_2023](saleh_pdca-former_2023.md) | 2023 | S1 | PDCA-Former | Change detection | F1 88.9%, IoU 85.7% | 🟡 precursor of ↑ |
| [wu_near-real-time_2023](wu_near-real-time_2023.md) | 2023 | S1 | U-Net | Near-real-time | OA 0.986, F1 0.976 | 🟡 method |
| [fawakherji_deepflood_2025](fawakherji_deepflood_2025.md) | 2025 | Aerial + S1/S2 | U-Net/UNet++ | **Dataset** | mIoU 72.4 (S2-only: 43–51) | 🟡 dataset |
| [riche_novel_2024](riche_novel_2024.md) | 2024 | Landsat + SRTM | W-Res-U-Net | **Susceptibility** | AUC 95.13% ⚠️ | 🟡 susceptibility |
| [shu_satvit-seg_2026](shu_satvit-seg_2026.md) | 2026 | Aerial RGB | SatViT-Seg (ViT) | Land cover | mIoU 71.18 | 🟡 architecture |
| [liu_deep_2022](liu_deep_2022.md) | 2022 | WorldView-2 | U-Net | Aquatic vegetation | Acc 94.5% | ⚪ method only |
| [herath_subgrid_2025](herath_subgrid_2025.md) | 2025 | none (HEC-RAS) | SGUnet | Hydro surrogate | CSI 0.881–0.988 | ⚪ low |
| [zhu_improving_2025](zhu_improving_2025.md) | 2025 | S2 + SRTM | RCAN (super-res) | DEM SR → sim | DEM MAE 2.20 m | ⚪ low |
| [kapoor_qdeepgr4j_2026](kapoor_qdeepgr4j_2026.md) | 2026 | none (rainfall) | LSTM + GR4J | Runoff + UQ | NSE 0.7313 (VIC) | ⚪ low (UQ recipe) |
| [yin_estimating_2023](yin_estimating_2023.md) | 2023 | none (photos) | ResNet34 | Rainfall intensity | MAPE 16.5% | ⚪ low |
| [gau_using_2024](gau_using_2024.md) | 2024 | none (tabular) | SVM/XGBoost | Socio-econ risk | Acc 0.60 ⚠️ | ⚪ framing only |

🔴 critical · 🟠 directly useful · 🟡 method to cite · ⚪ peripheral

---

## Cross-cutting findings

These emerged only from reading all 23 together — they are the backbone of a literature review.

### 1. Optical vs SAR: the evidence is split, and cloud cover is why

The single most important question for this thesis (which uses **optical Sentinel-2**) gets *opposite* answers depending on the dataset:

- **`konapala_exploring_2021`** — on Sen1Floods11, **Sentinel-2 optical F1 0.88–0.90** crushes **Sentinel-1 SAR F1 0.62**. Fusion adds nothing. Best input is an **HSV transform** of S2, beating NDWI.
- **`fawakherji_deepflood_2025`** — adding SAR to optical gains **+10–15 mIoU**; S2-only scores just **43–51 mIoU**.

**The reconciliation:** Sen1Floods11 is *near-cloud-free*. Real floods arrive with storms, i.e. with clouds. Konapala's own Ghana case shows fusion helping under heavy cloud. **So the optical advantage is real but conditional on visibility** — which is precisely what the next finding attacks.

### 2. ⚠️ The biggest threat to this thesis is vegetation, not clouds

`shastry_mapping_2023` is the paper to take most seriously. An optical CNN with excellent validation scores (98.2% precision / 93.7% recall) still **underpredicted real flood extent by 62%**. The breakdown of what it missed:

| Obstruction | Share of underprediction |
|---|---|
| **Vegetation** | **74%** |
| Clouds | 9% |
| Both | 4% |
| **Total obstructed** | **~79%** |

Within clouds the model missed **>90%** of flooding; within vegetation, **62%**.

**Why this matters:** the thesis config filters to **max 10% cloud**. That filter addresses the *9%* problem and does **nothing** about the *74%* problem. Flooded vegetation is invisible to optical sensors regardless of how clear the sky is. This must be confronted explicitly in the thesis — as a stated limitation, a justification for adding SAR, or a reason to adopt the mitigations below.

### 3. Three concrete mitigations exist in this library

- **`li_terrainfloodsense_2025`** — map floods *through* cloud by fusing **JRC Global Surface Water occurrence + DEM + HAND** into a Bayesian prior (OA 94.31%). The most directly actionable cloud fix.
- **`ulloa_sentinel-1_2022`** — **ConvLSTM** over an image time series. The closest architectural match to this thesis's multi-year NDVI/NDWI framing (Kappa 0.93). Full hyperparameters captured.
- **`chen_flood_2025`** — feed the **index as its own encoder branch** rather than as an extra channel. Their index alone contributed **67% of all IoU gains**. Directly transferable to NDVI/NDWI.

### 4. ⚠️ NDWI choice needs care — two papers warn about it

- **`tran_surface_2022`**: Otsu on **NDWI overestimates water in paddy/agricultural areas**; **MNDWI** (needs SWIR **B11**) was reliable. The thesis currently uses NDWI only and does not download B11.
- **`pech-may_flood_2025`**: uses **Gao's NDWI (NIR−SWIR)**, *not* the McFeeters NDWI `(B03−B08)/(B03+B08)` in this thesis — so **it cannot be cited as support for the thesis's formula**.
- **`konapala_exploring_2021`**: an **HSV transform beat NDWI** (0.90 vs 0.89) — worth testing.

### 5. Baselines available to compare against

| Type | Paper | Number to beat |
|---|---|---|
| Non-DL threshold | `tran_surface_2022` | Otsu (no per-pixel metrics ⚠️) |
| Classical ML | `liu_deep_2022` | Random Forest + GLCM (U-Net beat it by ~15pp) |
| Vanilla U-Net | `konapala_exploring_2021` | **F1 0.90 on Sen1Floods11 with S2** ← the number to target |
| Sentinel-2 only | `fawakherji_deepflood_2025` | mIoU 43.2–51.4 |
| Bayesian U-Net | `garshasbi_uncertainty-aware_2025` | F1 0.8013 / IoU 0.6685 |

**`Sen1Floods11`** (used by `konapala_exploring_2021` and `garshasbi_uncertainty-aware_2025`) is the obvious public benchmark: 446 hand-labelled 512×512 images, 11 flood events, 10 m — same resolution as the thesis.

### 6. Regional expectation-setter

`du_high-precision_2026` reports **Bolivia** (heavy-rain flooding, 3.69% inundation) as its **worst-performing region**. For a Peru-focused thesis in similar terrain and rainfall regimes, that is a realistic prior — do not expect Sen1Floods11-level scores.

---

## ⚠️ Data-quality warnings

Reading these papers closely surfaced a striking number of **internal contradictions** — cases where a paper's abstract disagrees with its own tables. **Always cite the table, not the abstract.** Flagged in detail in each note:

- **`saleh_high-precision_2024`** — the abstract's "90.6% *improvement* in F1" is actually the **absolute value**; the real gain is ≈ +2.2 pts. Do not repeat the abstract's framing.
- **`riche_novel_2024`** — 4 contradictions; abstract says AUC 95.87%, table says 95.13%.
- **`gau_using_2024`** — abstract's "0.62 accuracy" is actually *precision*; accuracy is 0.59.
- **`xu_sar_2022`** — reports **no accuracy metrics at all**; its U-Net comparison is purely visual. Cannot be used as a numeric baseline.
- **`chen_flood_2025`** — its Madagascar flood areas (27,936 km² etc.) are implausible; do not cite them.
- **`fawakherji_deepflood_2025`** — article is CC BY-NC-**ND**; the figshare dataset licence is unstated. **Verify before using the data.**
- **`wu_near-real-time_2023`**, **`liu_deep_2022`** — labels are semi-automatic / re-randomized each epoch, so their ~0.98 accuracies are agreement with their own product, not independent ground truth.
