---
title: "QDeepGR4J: Quantile-based ensemble of deep learning and GR4J hybrid rainfall-runoff models for extreme flow prediction with uncertainty quantification"
authors: "Arpit Kapoor, Rohitash Chandra"
year: 2026
venue: "Journal of Hydrology, vol. 664 (2026), article 134434, 12 pp. (Received 20 July 2025; accepted 15 October 2025; available online 23 October 2025)"
doi: "10.1016/j.jhydrol.2025.134434"
bibtex_key: kapoor_qdeepgr4j_2026
zotero_pdf: "/Users/gersongarrido/Zotero/storage/7RTEBXBK/Kapoor and Chandra - 2026 - QDeepGR4J Quantile-based ensemble of deep learning and GR4J hybrid rainfall-runoff models for extre.pdf"
pages: 12
sensors: [none]
task: "rainfall-runoff / streamflow prediction (multi-step ahead) with uncertainty quantification + qualitative flood risk indicator"
model: "GR4J conceptual model hybridised with quantile-regression LSTM/CNN/RNN/MLP ensemble (QDeepGR4J)"
tags: [rainfall-runoff, gr4j, hybrid-model, quantile-regression, uncertainty-quantification, lstm, gev, flood-early-warning, camels-aus, time-series]
---

# QDeepGR4J: Quantile-based ensemble of deep learning and GR4J hybrid rainfall-runoff models for extreme flow prediction with uncertainty quantification

> **TL;DR** — The authors extend their earlier hybrid model DeepGR4J (GR4J conceptual rainfall-runoff model whose routing component is replaced by a deep learning surrogate) into **QDeepGR4J**: an ensemble of three quantile-regression deep networks trained with the tilted (pinball) loss at τ = {0.05, 0.50, 0.95}, giving a 90% prediction interval on streamflow for 3-day-ahead multi-step forecasts. Trained on 222 Australian catchments (CAMELS-AUS, 1980–2014, 60/40 train/test split), the LSTM-based QDeepGR4J beats a pure quantile-LSTM baseline in RMSE, NSE and interval score in **all seven states** (e.g. VIC test NSE 0.7313 vs 0.6699 for LSTM, p. 7). A GEV-based flood threshold turns the predicted quantiles into a 4-level "flood risk indicator"; TPR for 3-year recurrence floods reaches 0.75–1.00 for QDeepGR4J-LSTM across six Queensland stations, but collapses toward 0.000 at 7- and 10-year recurrence intervals (Table 3, p. 9).
> **NOTE: this is hydrological time-series modelling. There is NO satellite imagery, no Sentinel-2, no spectral indices, no image segmentation anywhere in this paper.**

## Problem & Motivation

Conceptual rainfall-runoff models (GR4J, AWBM, Sacramento) are parsimonious and interpretable but perform poorly on extreme events; purely data-driven deep learning models are accurate but data-hungry, lack physical consistency and "generalise poorly outside the training regime or during extremes" (p. 2). The authors' own prior model, DeepGR4J (Kapoor et al., 2023), hybridised the two but "suffers from poor performance in extreme flow regions" and "does not address any uncertainty in model predictions, which is crucial for increasing the adoption and reliability of data-driven/hybrid modelling" (p. 2).

The gap addressed: (i) make the hybrid model good at the **tails** (high flows → floods) rather than at the conditional mean, and (ii) attach **uncertainty bounds** to the forecast so it can drive an early-warning decision. Their answer is to swap the mean-regression objective for a quantile-regression objective and to reinterpret the resulting upper/lower quantiles as an operational flood alert.

## Method / Architecture

Four-stage hierarchical pipeline (Algorithm 1, p. 4; Fig. 2, p. 4):

- **Stage 1 — GR4J skeleton.** GR4J (Perrin et al., 2003) with four parameters δ = {X1 maximal capacity of production store, X2 catchment water-exchange coefficient, X3 maximal routing reservoir capacity, X4 unit-hydrograph time base}. Inputs precipitation P⁽ᵗ⁾ and evapotranspiration E⁽ᵗ⁾. Equations for net precipitation, production-store update and percolation are given as Eqs. (1)–(6), pp. 3.
- **Stage 2 — Calibrate GR4J with Differential Evolution** (gradient-free) to obtain optimal δ̂. Only the **production storage** is retained; the **routing storage is replaced by the deep learning surrogate**.
- **Stage 3 — Hybrid feature generation.** The calibrated production store emits internal states which are concatenated with meteorological drivers to give the feature vector (Eq. 9, p. 5):
  `x̃⁽ᵗ⁾ = [ P⁽ᵗ⁾, E⁽ᵗ⁾, T_min⁽ᵗ⁾, T_max⁽ᵗ⁾, vprp⁽ᵗ⁾, Pₙ⁽ᵗ⁾, Pₛ⁽ᵗ⁾, Perc⁽ᵗ⁾ ]`
  i.e. 5 meteorological + 3 GR4J-internal variables. These are windowed via state-space (Takens') reconstruction with **window size α = 7** days (p. 5) into sequences X̄⁽ᵗ⁾ (Eq. 10).
- **Stage 4 — Quantile deep-learning ensemble.** Three independent networks, one per quantile τ ∈ {0.05, 0.50, 0.95}, trained with the **tilted / pinball loss** (Eq. 7, p. 3):
  `L_τ(θ) = (τ − 1) Σ_{Q_i < Q̂_i} (Q_i − Q̂_i) + τ Σ_{Q_i ≥ Q̂_i} (Q_i − Q̂_i)`
  The 0.05 and 0.95 predictions bracket a **90% confidence interval**.
- **Architectures compared:** MLP, vanilla RNN, CNN, LSTM (an **encoder–decoder LSTM**, explicitly different from their earlier work, p. 9). Layer counts, widths, kernel sizes, batch size, and number of epochs are **not reported** in the paper text.
- **Optimiser:** Adam, **learning rate 0.001**, β₁ = 0.89, β₂ = 0.97 (p. 5; repeated on p. 10 as "β₁ = 0.89 and β₁ = 0.97" — ⚠️ typo, the second should be β₂).
- **Forecast horizon:** multi-step, **3 days ahead** (motivated by Patel et al., 2024, and by dam pre-release / community-warning lead times in Australia, p. 5).
- **Flood risk indicator (FRI).** A flood threshold γ is derived by fitting a **Generalised Extreme Value (GEV)** distribution to the observed annual maximum streamflow, γ = F⁻¹(p; ζ, μ, σ) with p = 1 − 1/k for a once-in-k-year flood (p = 0.80 for a 5-year flood). FRI is then a 4-level label (Eq. 12, p. 6):
  - **High** if max(Q̂₀.₀₅) > γ; **Moderate** if max(Q̂₀.₅₀) > γ; **Low** if max(Q̂₀.₉₅) > γ; **Unlikely** otherwise.
- **Metrics.** RMSE and NSE on the median (τ = 0.50) prediction (Eqs. 13–14, p. 6); **Interval Score (IS)** for interval quality (Eq. 15, p. 6, with δ = 0.1 for the 90% interval — lower is better; it penalises both interval width and observations falling outside it). For flood detection, **True Positive Rate (TPR)** only; the authors argue accuracy is "unreliable" for such an imbalanced problem (p. 10).
- No augmentation. No dropout / MC-dropout / Bayesian layers — uncertainty comes *only* from the quantile heads (they explicitly note this captures aleatoric but **not epistemic** uncertainty, p. 10).

## Data

- **Sensors: none.** Purely in-situ / gridded hydro-meteorological time series. No satellite imagery.
- **Dataset:** **CAMELS-AUS** (Fowler et al., 2021) — 222 unregulated Australian catchments, streamflow + 12 climate variables + 134 catchment attributes (geology, soil, topography) (p. 5).
- **Period:** **1980–2014**, split **60% training / 40% testing** "as used in previous studies (Kapoor et al., 2023)" (p. 5). ⚠️ It is a **temporal split with no separate validation set**; hyperparameters appear to have been chosen on the test set.
- **Preprocessing** (p. 5):
  - Processed with the `camels-aus-py` Python package (CSIRO).
  - **Stations with >10% missing time steps for any variable of interest were discarded**; remaining missing values **imputed by linear interpolation**.
  - **z-score standardisation** applied to **both inputs and targets** (p. 5, restated p. 10).
  - Windowing with α = 7 (Takens' theorem style state-space reconstruction).
- **Spectral indices: none — N/A.** (No optical bands, no NDVI/NDWI.)
- **Ground truth:** observed gauged daily streamflow. Flood "labels" are *derived*, not observed: a binary flood label is assigned via the indicator `f = 1(Q > γ)` on observed streamflow, with γ from the fitted GEV (p. 8). The authors admit "the lack of availability of a flooding indicator in the observation data make it challenging to compute and evaluate a qualitative flood risk" (p. 10).
- **Evaluation subsets:** Section 4.2 uses **all stations in South Australia (SA)** — only **nine** stations (p. 9). Section 4.3 uses **five stations per state with the greatest runoff ratios**, for seven states (ACT excluded, only three stations available) (p. 7). The flood risk indicator is evaluated on **six stations on the eastern coast of Queensland** (Table 3, p. 9).
- **Code/data:** https://github.com/DARE-ML/DeepGR4J-Extremes (p. 5, p. 10).

## Results

### Architecture selection — all SA stations (Table 1, p. 6)

| Model | RMSE train | RMSE test | NSE train | NSE test | IS train | IS test |
|---|---|---|---|---|---|---|
| MLP | 0.5616 | 0.4131 | 0.6499 | 0.6160 | 1.4220 | 0.9338 |
| RNN | 0.6684 | 0.4716 | 0.3386 | 0.3551 | 0.9272 | 0.6244 |
| CNN | 0.5451 | 0.3979 | 0.6129 | 0.5892 | 0.4830 | 0.4247 |
| **LSTM** | **0.4792** | **0.3829** | **0.7775** | **0.6650** | **0.3679** | **0.4198** |

LSTM wins on RMSE, NSE and interval score. RNN is worst on median value; **MLP is worst on interval score** despite decent RMSE/NSE — i.e. good point accuracy does not imply good uncertainty (p. 6). This *reverses* their earlier finding that CNN beat LSTM for single-step mean prediction; they attribute the flip to quantile regression emphasising "tail behaviour and multi-step temporal dependencies, which are better captured by the LSTM" (p. 9).

### Hybrid vs pure DL baseline — 5 stations × 7 states (Table 2, p. 7)

Test-set figures (τ = 0.5), hybrid vs its own non-hybrid counterpart:

| State | LSTM test RMSE / NSE / IS | **DeepGR4J-LSTM** test RMSE / NSE / IS |
|---|---|---|
| NSW | 2.8321 / 0.3674 / 4.0582 | **2.4840 / 0.4557 / 3.6984** |
| NT | 3.1549 / 0.5358 / 4.3802 | **2.7486 / 0.6568 / 3.8564** |
| QLD | 6.8828 / 0.5165 / 11.9032 | **6.3671 / 0.5907 / 12.1210** (IS worse; CNN hybrid best IS at 11.4625) |
| SA | 0.5301 / 0.4579 / 0.6982 | **0.4829 / 0.5649 / 0.6394** |
| TAS | 2.2221 / 0.6528 / 5.1561 | **2.0702 / 0.7159 / 4.9004** |
| VIC | 1.2191 / 0.6699 / 1.9660 | **1.1106 / 0.7313 / 1.7884** |
| WA | 1.7810 / 0.6265 / 1.5947 | **1.6344 / 0.6900 / 1.5692** (CNN hybrid best IS at 1.4832) |

Headline: "in all seven states, the LSTM-based QDeepGR4J ensemble demonstrates the best performance in median value prediction, followed by the CNN-based QDeepGR4J ensemble" (p. 7). Hybridisation lifts test NSE by roughly **+0.06 to +0.12** over the corresponding pure DL model. On the **test set**, however, CNN-QDeepGR4J beats LSTM-QDeepGR4J on RMSE for **QLD and WA** (p. 7), and CNN gives the better interval score for QLD and WA — the authors flag this as "potential for regional variation in optimal architecture" (p. 10).

⚠️ NT shows a higher test RMSE than train RMSE alongside a *higher* test NSE; the authors explain this is because the test split contains larger flow variability and NSE is variance-normalised while RMSE is scale-dependent (p. 7). Worth noting for anyone reading the table naively.

### Flood risk indicator — TPR, 6 Queensland stations (Table 3, p. 9)

| Station | Model | 3-yr | 5-yr | 7-yr | 10-yr |
|---|---|---|---|---|---|
| 116006B Herbert R. at Abergowrie | LSTM | 0.926 | 0.000 | 0.000 | 0.000 |
| | **DeepGR4J-LSTM** | **0.963** | **1.000** | **0.750** | **1.000** |
| 121001A Don R. at Ida Creek | LSTM | 0.000 | 0.000 | 0.000 | 0.000 |
| | **DeepGR4J-LSTM** | **0.923** | **0.750** | 0.000 | 0.000 |
| 122004A Gregory R. at Lower Gregory | LSTM | 0.000 | 0.000 | 0.000 | 0.000 |
| | **DeepGR4J-LSTM** | **0.750** | **0.500** | **0.600** | 0.000 |
| 126003A Carmila Creek at Carmila | LSTM | 0.000 | 0.000 | 0.000 | 0.000 |
| | **DeepGR4J-LSTM** | **0.800** | **0.625** | **0.333** | (blank) |
| 136202D Barambah Creek at Litzows | LSTM | 0.962 | 0.000 | 0.000 | 0.000 |
| | **DeepGR4J-LSTM** | **1.000** | **0.875** | 0.000 | 0.000 |
| 137201A Isis R. at Bruce Highway | LSTM | 0.000 | 0.000 | 0.000 | 0.000 |
| | **DeepGR4J-LSTM** | **1.000** | **1.000** | **1.000** | **0.600** |

The pure LSTM ensemble "is unable to identify extreme events, particularly at higher flood recurrence intervals" (p. 8) — most of its cells are 0.000. The hybrid captures "almost all of the flooding events for a 3-year flood recurrence interval and close to half of the flooding events for a 5-year flood interval. However, for 7-year and 10-year floods, the TPR values are much lower" (p. 8).

⚠️ **Inconsistencies to flag:** (a) Table 3 has an **empty cell** for the 10-year interval of station 126003A (no value, not "0.000"). (b) Many 0.000 entries are ambiguous — with a 25-year record there may simply be **zero observed events** at 7- and 10-year recurrence, in which case TPR is undefined rather than zero; the authors themselves note "no observed events were available for validation at these higher recurrence levels" for intervals beyond 10 years (p. 8), but the table does not distinguish "missed" from "none existed". (c) **False positives are never quantified** — only TPR is reported, so precision/FAR is unknown even though the text repeatedly says the model "overestimates the streamflow… leading to false positive alerts" (p. 9).

**No comparison against MCMC/DREAM/GLUE Bayesian UQ was run** — it is only discussed as a "natural point of comparison" in future work (p. 10). No comparison against plain (non-quantile) DeepGR4J on the same tables.

## Limitations

Stated by the authors (pp. 9–10):
- **Uncertainty bounds widen excessively over multi-step horizons** → "with a high number of time-steps in the prediction horizon, we found a higher chance of false positives."
- Hybrid predictions are **partially dependent on the GR4J calibration**; errors there propagate into the DL inputs.
- Quantile regression captures **aleatoric** uncertainty only; **epistemic** (model/parameter) uncertainty is not quantified. Bayesian MCMC/DREAM/GLUE would, but is computationally infeasible at this scale.
- **Upper quantiles are overestimated** for some peaks (visible in Fig. 5) → false alarms. Fixes proposed but not implemented: multi-quantile training with non-crossing constraints, post-hoc quantile matching, CRPS-style proper scoring rules.
- **Training record ≈ 25 years**, which "restricts our ability to evaluate very rare events such as 50 or 100-year floods."
- Flood thresholds are subjective; no observed flood-indicator ground truth exists.
- Computational limits forced evaluation on **only 9 SA stations** (architecture study) and **5 stations per state** (regional study) — a small fraction of the 222 catchments.

Observed by me (not stated):
- **No validation split.** 60/40 train/test only; architecture and hyperparameters were selected by looking at test metrics (Table 1), so the reported test numbers are optimistically biased.
- **Quantile crossing** is acknowledged as a risk but never measured (how often is Q̂₀.₀₅ > Q̂₀.₅₀?).
- **PICP (prediction interval coverage probability) is not reported** — only interval score. We never learn the empirical coverage of the nominal 90% interval.
- No repeated runs / no error bars: single-seed results throughout.
- Network depth, width, epochs, batch size **not reported** → the paper is not reproducible from text alone (only via the GitHub repo).

## Relevance to this thesis

Be honest: **this paper is not about satellite imagery and contributes nothing to the Sentinel-2 mapping side of the thesis.** No optical bands, no NDVI/NDWI, no CNN-over-images, no segmentation, no IoU/F1. Anyone hoping to lift an architecture or a preprocessing step for Sentinel-2 will find nothing here. Its value is confined to two narrow but genuinely useful areas:

**1. It justifies (and gives methodology for) the "predict" half of the thesis title.**
- The thesis promises to *identify AND predict* flood-prone areas. Identification is the Sentinel-2/NDWI part; **prediction is the part that currently has no method**. This paper is a defensible citation for how the prediction is normally done in the literature (rainfall-runoff modelling), and lets you argue *why* your approach differs (you predict susceptibility from imagery time-series, not discharge from rainfall).
- **Borrowable directly: the pinball / tilted loss (Eq. 7, p. 3)** with τ = {0.05, 0.50, 0.95}. If the thesis CNN currently outputs a single flood-probability map, adding two extra quantile heads trained with pinball loss is a cheap way to get a **90% uncertainty band per pixel** — a well-cited contribution that most Sentinel-2 flood-mapping theses lack. Trivially implementable in PyTorch (three output channels, three tilted losses).
- **Borrowable: the Interval Score (Eq. 15, p. 6, δ=0.1)** as the metric for reporting that uncertainty. It is a single number that penalises both over-wide intervals and misses.

**2. The GEV flood-threshold → 4-level risk label (Eq. 11–12, p. 6) is a directly transferable idea.**
- The thesis needs a way to turn a continuous model output into an actionable **risk class** for disaster-management framing. QDeepGR4J's `High / Moderate / Low / Unlikely` rule based on which quantile crosses the GEV-derived threshold is an elegant, citable pattern. For the Peru case you would swap "streamflow > γ" for something like "predicted water-fraction / NDWI-derived inundation > threshold", and you could still fit a GEV to the historical annual maxima of a nearby gauge (SENAMHI / ANA data) to set the threshold objectively rather than arbitrarily.
- Their honest conclusion that TPR is high only for **3- and 5-year** floods and collapses at 7- and 10-year intervals is a **cautionary result** worth citing: rare events are precisely what a ~25-year record cannot validate. The thesis has the same problem (Sentinel-2 only starts in 2015 → ~10 years of imagery). Cite this as prior evidence that short records limit rare-event validation, and scope your claims accordingly.

**3. Cautionary / methodological warnings to import into the thesis design:**
- **Good point accuracy ≠ good uncertainty.** Their MLP had fine RMSE/NSE but the worst interval score (Table 1, p. 6). If the thesis adds UQ, it must report an interval/calibration metric separately, not assume accuracy implies calibration.
- They used **z-score normalisation on both inputs and targets** (p. 5) — an explicit, citable precedent for a preprocessing choice you'll also make.
- Their evaluation has **no validation split** and single-seed runs. **Do better in the thesis** — this is an example of a published-in-J.Hydrol paper with a protocol weakness you can consciously avoid and, if useful, contrast against in the methodology chapter.
- The authors report **TPR only** for flood detection and never quantify false alarms, despite admitting overestimation causes them. **Do not repeat this**: for the thesis's flood classes, report precision/FAR alongside recall.

**4. What it is NOT for this thesis:** not a baseline (nothing to beat — different task, different output space, different data), not a dataset source (CAMELS-AUS is Australian gauge data, useless for Peru), not an architecture to reuse. Its ideal home is the **related-work chapter** (as the "hybrid physics+DL / discharge-forecasting" strand distinct from the remote-sensing strand) and, more concretely, the **uncertainty-quantification subsection of the methodology** if you decide to add quantile heads.

## Keywords / Tech

- **Models:** GR4J (conceptual rainfall-runoff), DeepGR4J, QDeepGR4J; LSTM (encoder–decoder), CNN, vanilla RNN, MLP
- **Sensors:** none (in-situ gauged streamflow + gridded meteorology)
- **Indices:** none (no spectral indices; hydrological only)
- **Techniques:** quantile regression, tilted/pinball loss, ensemble UQ, Differential Evolution (GR4J calibration), Adam (lr 0.001, β₁=0.89, β₂=0.97), Takens' state-space reconstruction / windowing (α = 7 days), z-score normalisation, linear interpolation imputation, Generalised Extreme Value (GEV) distribution, flood frequency analysis
- **Metrics:** RMSE, NSE (Nash–Sutcliffe), Interval Score, True Positive Rate
- **Datasets:** CAMELS-AUS (222 Australian catchments, 1980–2014)
- **Libraries:** `camels-aus-py` (CSIRO); code at github.com/DARE-ML/DeepGR4J-Extremes
- **Framing:** early warning system, flood risk indicator, aleatoric vs epistemic uncertainty

## Notable quotes

> "DeepGR4J showed considerable improvement in accuracy when compared to the baseline conceptual and machine learning models trained independently. However, the results also show that the DeepGR4J suffers from poor performance in extreme flow regions. Therefore, DeepGR4J needs to be adapted to make it suitable for predicting extremely high flows such as floods. Lastly, DeepGR4J does not address any uncertainty in model predictions, which is crucial for increasing the adoption and reliability of data-driven/hybrid modelling." (p. 2)

> "In this study, we train deep learning models for quantiles τ = {0.05, 0.50, 0.95}, which together define a 90% confidence interval." (p. 3)

> "The DeepGR4J-LSTM ensemble can capture almost all of the flooding events for a 3-year flood recurrence interval and close to half of the flooding events for a 5-year flood interval. However, for 7-year and 10-year floods, the TPR values are much lower, highlighting the limitations of this approach." (p. 8)

> "Although the quantile regression approach can quantify the aleatoric uncertainty arising from the data, it cannot quantify the epistemic uncertainty relating to the model architecture/parameters." (p. 10)

> "We note that a key limitation of our study is the relatively short training period length (25 years), which restricts our ability to evaluate very rare events such as 50 or 100-year floods." (p. 10)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/7RTEBXBK/Kapoor and Chandra - 2026 - QDeepGR4J Quantile-based ensemble of deep learning and GR4J hybrid rainfall-runoff models for extre.pdf`
Cite as: `\cite{kapoor_qdeepgr4j_2026}`
