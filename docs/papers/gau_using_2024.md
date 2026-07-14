---
title: "Using Machine Learning to Determine the Efficacy of Socio-Economic Indicators as Predictors for Flood Risk in London"
authors: "Grace Gau, Minerva Singh"
year: 2024
venue: "Revue Internationale de Géomatique (RIG), 2024, vol. 33, pp. 427–443"
doi: "10.32604/rig.2024.055752"
bibtex_key: gau_using_2024
zotero_pdf: "/Users/gersongarrido/Zotero/storage/GB9HH9FG/Gau and Singh - 2024 - Using Machine Learning to Determine the Efficacy of Socio-Economic Indicators as Predictors for Floo.pdf"
pages: 17
sensors: [none]
task: "flood risk classification (High/Medium/Low risk labels per census area) from socio-economic tabular data"
model: "SVM, Random Forest, XGBoost, Stacked Generalization (no deep learning, no imagery)"
tags: [socio-economic, flood-risk, vulnerability, machine-learning, tabular-data, london, smote, stacked-generalization, class-imbalance, spatial-overfitting]
---

# Using Machine Learning to Determine the Efficacy of Socio-Economic Indicators as Predictors for Flood Risk in London

> **TL;DR** — The authors ask whether an area's socio-economic profile alone (race/BAME shares, mean house price, free school meals, crime, jobless households, income assistance) can predict its official flood-risk label, with **no environmental/climatic predictors at all**. Using UK DEFRA flood-risk polygons joined to 2011 census LSOA data for London (final dataset: 1150 distinct LSOA codes, p. 431), they train SVM, RF, XGBoost and a stacked-generalization ensemble. Best binary result: **SVM, accuracy 0.60, AUC 0.62** (Table 1, p. 434); best multiclass result: **XGBoost, validation accuracy 0.59 / precision 0.62** (Table 1, p. 434 — the abstract's "0.62 accuracy" claim does not match the table, see ⚠️). Conclusion: socio-economic variables carry a real but weak signal (barely above chance), suggesting they should *augment*, not replace, environmental flood models.

## Problem & Motivation

Most ML flood-risk work uses climatic/geomorphic predictors (rainfall, elevation, topography) and reaches AUC > 0.9 (p. 428, citing [10–13]); socio-economic research on floods is mostly *reactive*, studying how communities recover, not whether socio-economic status can *predict* risk (p. 428). The authors position their gap explicitly: "there has been limited exploration of the predictive capabilities of socioeconomic indicators, which could be crucial in implementing proactive flood mitigation strategies in vulnerable communities" (p. 428).

The framing is equity/disaster-relief: flood defenses protect higher-value riverside property; Black communities and low-income populations are disproportionately exposed and least able to recover (p. 428). London is chosen for its socio-economic heterogeneity. Research question (p. 429): "Can an area's socioeconomic status be used as a dependable predictor for flood risk?"

## Method / Architecture

**No deep learning, no imagery, no spectral indices.** Classical tabular ML on scikit-learn-style models.

- **Targets**: official flood-risk labels ('High', 'Medium', 'Low'; 'Very Low' deliberately dropped, see Data). Two problem settings:
  - *Multiclass*: High / Medium / Low.
  - *Binary*: High vs Low (Table 1 header confusingly repeats "(High, Medium, and Low risk)" for the binary block — ⚠️ a typo).
- **Models**:
  - SVM (RBF kernel) — "impressive performance on small datasets… maximizing the distance between these classes" (p. 433).
  - Random Forest.
  - XGBoost.
  - **Stacked Generalization (SG)**: for *binary*, RF = base model and SVM = meta-model; for *multiclass*, RF + SVM = base models and XGBoost = meta-model (abstract, p. 427; p. 433).
- **Spatial-overfitting control** (the most transferable idea in the paper): random train/val splits leak, because neighbouring LSOA points are near-duplicates. They instead **partition by randomly chosen ward codes**, "guaranteeing that no data with the same ward code is distributed throughout the datasets" (p. 432). This is block/group splitting.
- **Feature selection**: **Sequential Forward Selection (SFS)** [25] to reduce spatial overfitting; applied to SVM, RF, SG. **Excluded from XGBoost** because XGBoost's greedy split choice + built-in L2 regularization made SFS unhelpful — "results were not improved, and the decision to exclude SFS in the final model was made" (p. 433).
- **Class imbalance**: **SMOTE** variants — Borderline-SMOTE for RF and SG (multiclass), SVM-SMOTE for SVM (p. 433, Table 2 p. 435–436). Imbalance: 'Low' > 53% of the final dataset, 'High' only 11% (p. 438).
- **Hyperparameter tuning**: Grid Search. Reported final settings (Table 2, pp. 435–436), e.g.:
  - SVM (multi): C=1, kernel 'rbf', gamma 'scale'.
  - RF (multi): max_depth 8, max_features 'sqrt', min_samples_leaf 6, min_samples_split 2, n_estimators 60.
  - XGBoost (multi): gamma 6, learning_rate 0.4, max_depth 5, min_child_weight 7, n_estimators 5.
  - SG (multi): rf_n_estimators 75, rf_max_depth 6, svm_C 2, xgb_learning_rate 0.4, xgb_max_depth 7, xgb_n_estimators 10.
  - SVM (binary): default params (C=1, rbf, degree 3, gamma 'scale').
  - RF (binary): max_depth 18, n_estimators 87.
  - XGBoost (binary): gamma 6, **learning_rate 2**, max_depth 4, min_child_weight 1 (⚠️ a learning rate of 2 is implausible for XGBoost and contradicts the text on p. 432, which says raising LR "from 0.3 to 0.4" gave a six percent training-accuracy improvement).
- **Statistics**: Student's t-test (two groups) / one-way ANOVA (multi-group), significance p < 0.05 (p. 435).
- **Not reported**: number of grid-search folds, cross-validation fold count k, random seeds, software/library versions, runtime, any held-out *test* set (only train / validation / cross-validation are reported).

## Data

**Sensors: none.** This is a purely tabular geospatial study — no satellite imagery, no SAR, no optical bands, no NDVI/NDWI.

- **Study area**: London, England — 607 square miles, >8.8 million inhabitants, 32 boroughs, 17 adjacent to the Thames (p. 429). Analysis unit = **LSOA** (Lower Super Output Area), average population 1722 (p. 430).
- **Flood-risk labels**: DEFRA (Department for Environment, Food & Rural Affairs), **17,223 polygons**; mean polygon area **13,054 m² before anomaly removal, 6217 m² after** (p. 430). Labels cover **only flood risk from rivers and sea** — no groundwater/surface-water/pluvial flooding (p. 429, Fig. 1 caption; restated as a limitation p. 439).
- **Geography join**: Ordnance Survey code-point postcode data — 33 boroughs, 654 wards, 5835 LSOAs (p. 430). Census 2011 LSOA atlas from the London Datastore: 4766 distinct LSOA codes (p. 430).
- **Final dataset: 1150 distinct LSOA codes** (p. 431) — i.e. only 1150 of 4766 socio-economic LSOAs overlap the flood data (p. 438).
- **Join method**: `geopandas` `sjoin` was rejected — >7,000,000 crossing points, boundary points ambiguously assigned to neighbouring polygons of different labels (p. 430). They wrote a custom **'nearest'** function joining by proximity instead.
- **'Very Low' class removed** for three reasons (p. 430): its mean polygon area exceeds 82,000 m² (vs <8000 m² for all other labels) → outliers; those regions sit around the Thames and are protected by the Thames Barrier; and Zone 1 (<1-in-1000/yr probability) sites need no flood-specific planning studies.
- **Socio-economic predictors** (p. 431, Fig. 2; feature keys in Table 2): racial distribution percentages (BAME, White, Mixed, Asian, Black, Other), **mean house price (Price)**, **free school meals (FSM, banded 5–10 and 11–15)**, **crime / notifiable offences (Crime)**, **households with no adults in employment (NAE)**, **child poverty (CP)**, **children out of work benefits (COW)**.
- **Preprocessing**: anomaly/outlier polygon removal, 'Very Low' class deletion, custom nearest-neighbour spatial join, SMOTE oversampling, ward-code-based split. Normalization/scaling: **not reported** (notable, since SVM with RBF is scale-sensitive).
- **Spectral indices**: none — N/A for this paper.

## Results

All figures below are **validation** metrics from **Table 1 (p. 434)**. There is **no independent held-out test set** and **no environmental-predictor baseline model was trained** — the "comparison" to environmental models is only a citation of other papers' AUC ≈ 0.9 (p. 428), not an experiment run here.

### Multiclass (High / Medium / Low), Table 1, p. 434

| Metric | SVM | RF | XGBoost | Stack Gen. |
|---|---|---|---|---|
| Precision | 0.60 | 0.61 | **0.62** | 0.61 |
| Recall | 0.58 | 0.58 | **0.59** | 0.59 |
| F1 | 0.52 | 0.53 | **0.54** | 0.51 |
| Train accuracy | 0.59 | 0.79 | 0.61 | 0.63 |
| Validation accuracy | 0.58 | 0.58 | **0.59** | 0.59 |
| Cross validation | 0.55 | **0.58** | 0.57 | 0.55 |

### Binary (High vs Low), Table 1, p. 434

| Metric | SVM | RF | XGBoost | Stack Gen. |
|---|---|---|---|---|
| Precision | **0.60** | 0.58 | 0.57 | 0.58 |
| Recall | **0.60** | 0.57 | 0.56 | 0.57 |
| F1 | **0.60** | 0.57 | 0.56 | 0.57 |
| Train accuracy | 0.67 | 0.89 | 0.93 | 0.85 |
| Validation accuracy | **0.60** | 0.57 | 0.56 | 0.57 |
| Cross validation | 0.60 | 0.61 | 0.57 | **0.65** |
| AUC | **0.62** | 0.61 | 0.57 | 0.59 |

- **Headline**: SVM binary — accuracy 0.60, AUC 0.62 (p. 434). SVM also shows the least train/validation gap (train 0.67 vs val 0.60), i.e. least overfitting; RF (0.89) and XGBoost (0.93) train accuracies show heavy overfitting (p. 434).
- **RF multiclass** regularization worked: grid search + SFS cut training accuracy "from 1 to 0.78" and raised recall "from 0.44 to 0.58" (p. 435). ⚠️ Table 1 lists RF multiclass train accuracy as **0.79**, not 0.78.
- **Feature importance (SFS, Table 2 + §3.3, p. 435)**: racial distribution (BAME, White, Mixed, Asian, Other) selected by **all** algorithms; **mean property price selected by all models**; free school meals second-most-frequent; only 2 of 8 final models exclude crime and no-adults-in-employment. Notably SFS keeps **child-poverty** indicators (FSM, child poverty, children out-of-work benefits) but drops **adult-poverty** indicators (income assistance rate, unemployment rate) (p. 437).

### ⚠️ Inconsistencies found

1. **Abstract vs Table 1**: the abstract states "XGBoost outperforms other multiclass classification methods with 0.62 accuracy" (p. 427); Table 1 gives XGBoost multiclass **validation accuracy 0.59** — 0.62 is its *precision*. §3.1 (p. 435) likewise says XGBoost has "validation accuracy score of 0.6". The conclusion (p. 440) again claims "an accuracy of up to 0.62". **Do not cite 0.62 as an accuracy.**
2. §3.1 (p. 435): "SVM has impressive accuracy and recall scores of 0.60 and 0.58" — 0.60 is SVM's multiclass *precision*; its validation accuracy is 0.58.
3. RF training accuracy 0.78 (text, p. 435) vs 0.79 (Table 1, p. 434).
4. XGBoost (binary) `learning_rate: 2` (Table 2, p. 436) contradicts the text's 0.3→0.4 discussion (p. 432).
5. Table 1's binary block is headed "(High, Medium, and Low risk)" — copy-paste error.

## Limitations

**Stated by the authors** (§4.3, pp. 438–439):
- 2011 census data used because 2021 data was unavailable at the time; LSOA boundary revisions not accounted for.
- Flood dataset covers only river/sea inundation → excludes areas far from watercourses; **only 1150 of 4766 LSOAs overlap**.
- Class imbalance: 'Low' >53%, 'High' 11% of the final dataset.
- London's urban density, historic infrastructure and defense levels may not generalize to other cities.
- Explicitly (§4.4, p. 439): "Using just socioeconomic data to evaluate flood risk is insufficient in creating a flawless model, since some places are not prone to floods owing to natural factors such as elevation and geography."

**Observed by me**:
- **Performance is barely above chance.** For binary with a roughly balanced set, 0.60 accuracy / 0.62 AUC is a weak classifier. The paper's rhetoric ("dependable feature", "extremely significant") substantially over-reads the numbers.
- **No held-out test set.** Everything reported is validation/CV; hyperparameters were grid-searched — so validation metrics are optimistically biased.
- **No environmental baseline was trained.** The claim that socio-economic data "can enhance" environmental models is *never tested* — no combined socio-economic + environmental model was run.
- **Confounding by geography**: house price and BAME share are spatially structured in London and correlate with distance-from-Thames; the model may simply be recovering "where the river is", not a causal socio-economic effect. The ward-block split mitigates but does not remove this.
- **Target leakage of a policy artifact**: labels are DEFRA *planning* zones, partly a function of built defenses — which themselves were sited by property value. The paper hints at this (p. 430, Thames Barrier / 'Very Low') but doesn't control for it.
- **No uncertainty quantification**, no confidence intervals on any metric despite claiming statistical significance.
- Data are available only "from the corresponding author upon reasonable request" (p. 440) — not a reusable public dataset.

## Relevance to this thesis

**Honest verdict: LOW methodological relevance, MODERATE framing relevance.** This is tabular socio-economic ML with **zero remote sensing** — no Sentinel-2, no bands, no NDVI/NDWI, no CNN, no segmentation, no pixels. Nothing in its architecture, loss, augmentation or preprocessing transfers to a PyTorch CNN over NDVI/NDWI time series. Do not present it as a methods precedent, and do not use it as a baseline (its 0.60 accuracy is not comparable to any pixel-level inundation metric).

Where it *is* usable:

- **Motivation / Chapter 1 citation.** Best single use: a citable, peer-reviewed source for the disaster-management and social-equity framing — vulnerable, low-income and minority populations are disproportionately exposed and least able to recover; flood-risk products should direct relief resources to them (pp. 428, 439–440). Good for the "why flood-prone-area mapping matters" argument in a Peruvian context.
- **Terminological discipline: risk ≠ inundation.** The paper predicts an official *flood-risk / susceptibility label* per census area, not observed water extent. The thesis maps/predicts *flood-prone areas* from imagery. Cite this as an example of the **vulnerability/exposure axis** of risk, distinct from the **hazard axis** the thesis addresses — useful for a "scope of this thesis" paragraph that explicitly says socio-economic vulnerability is out of scope.
- **A concrete methodological borrow: block/group spatial splitting.** Their ward-code-based train/val partition (p. 432) is the *one* directly reusable idea. The thesis will face the same leakage problem: adjacent Sentinel-2 patches from the same scene in train and validation inflate scores. Adopt an analogous **spatially-disjoint tile split** (split by region/scene, not by random patch) and cite this paper for the rationale.
- **A second borrow: recall-first metric justification.** Their argument (p. 434) that recall matters most because "false negatives, where high-risk areas are mislabeled as low risk, can have severe consequences" is exactly the argument the thesis needs for prioritizing recall/IoU on the flood class over overall accuracy. Directly quotable.
- **Class-imbalance handling (weak borrow).** SMOTE/Borderline-SMOTE is a tabular technique; for a CNN the equivalent is class-weighted / focal / Dice loss, not SMOTE. Cite only to note the imbalance *problem* (flooded pixels are a small minority), not the solution.
- **Cautionary example.** Useful as an explicit demonstration that socio-economic data alone yields near-chance performance (0.60 / AUC 0.62) whereas environmental predictors reach AUC > 0.9 (their own p. 428). That is a good argument in the thesis's favour: **physical/spectral predictors from Sentinel-2 are where the signal is.**
- **Also a cautionary example of reporting hygiene** — the abstract's "0.62 accuracy" is actually precision. Worth internalizing: report metrics that match the table.

## Keywords / Tech

- **Models**: SVM (RBF), Random Forest, XGBoost, Stacked Generalization (stacking ensemble)
- **Sensors**: none (tabular geospatial; no remote sensing)
- **Indices**: none
- **Techniques**: Sequential Forward Selection (SFS), Grid Search, SMOTE / Borderline-SMOTE / SVM-SMOTE, ward-code block splitting for spatial overfitting, Student's t-test, one-way ANOVA
- **Libraries**: geopandas (`sjoin` rejected; custom 'nearest' join used); ML library not named (implied scikit-learn / xgboost)
- **Datasets**: DEFRA flood-risk polygons (rivers & sea, 17,223 polygons), Ordnance Survey Code-Point postcodes, London Datastore LSOA Atlas (2011 census)
- **Metrics**: precision, recall, F1, train/validation/cross-validation accuracy, AUC

## Notable quotes

- "Can an area's socioeconomic status be used as a dependable predictor for flood risk?" (p. 429)
- "By using conventional flood indicators like rainfall, elevation, topography, among others, machine learning models may achieve Area Under the Curve (AUC) values above 0.9 in certain regions." (p. 428) — the implicit benchmark their 0.62 falls far short of.
- "Recall is prioritized because it measures the proportion of actual high-risk flood areas correctly identified, which is crucial to minimize false negatives. False negatives, where high-risk areas are mislabeled as low risk, can have severe consequences, such as leaving properties uninsured or unprotected." (p. 434)
- "Geospatial information is susceptible to overfitting due to the dense clustering of data points. It is necessary to apply measures to prevent models from memorizing closely similar points. One way to do this is by dividing the training and validation data based on distinct ward numbers." (p. 438)
- "Using just socioeconomic data to evaluate flood risk is insufficient in creating a flawless model, since some places are not prone to floods owing to natural factors such as elevation and geography." (p. 439)

## Pointer
Full PDF: `/Users/gersongarrido/Zotero/storage/GB9HH9FG/Gau and Singh - 2024 - Using Machine Learning to Determine the Efficacy of Socio-Economic Indicators as Predictors for Floo.pdf`
Cite as: `\cite{gau_using_2024}`
