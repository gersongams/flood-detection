# Instructions: writing a paper memory note

You are building a literature-review memory note for a master's thesis. You summarize **ONE** paper into **ONE** Markdown file. This note is meant to *replace re-reading the paper*, so it must be long, rich, and above all **accurate**.

## CRITICAL RULES — accuracy over completeness

1. **READ THE ENTIRE PDF.** The Read tool caps PDFs at **20 pages per call**. If the paper is longer, you MUST call Read again with a `pages` range (e.g. `pages: "21-40"`) until you have seen the conclusion and references. A note built from only the first window is a silent failure that *looks* complete. Verify you reached the conclusions.
2. **NEVER invent numbers.** Every metric (IoU, F1, OA, accuracy, kappa, dataset size, resolution, #images, patch size) must be copied **verbatim** from the paper. If something is not reported, write exactly `not reported`. **A fabricated metric is worse than a blank** — it will be cited in a thesis and propagate.
3. **Cite page numbers** for every metric and quote, e.g. `(p. 8)`.
4. **Flag inconsistencies.** If the paper contradicts itself (different numbers in text vs table, mismatched dates, a formula that doesn't match the bands listed), say so explicitly with a ⚠️.
5. **No baselines? Say so.** Do not imply a comparison that wasn't run.

## Thesis context — for the "Relevance" section

The thesis: **satellite image analysis to identify and predict flood-prone areas** (Peru).
- Sensor: **Sentinel-2** (Sentinel Hub / Copernicus Data Space), L2A, 10 m resolution, max 10% cloud
- Bands: B02 blue, B03 green, B04 red, B08 NIR
- Indices: **NDVI** `(B08-B04)/(B08+B04)`, **NDWI (McFeeters)** `(B03-B08)/(B03+B08)`
- Method: **deep learning (PyTorch CNN)** over NDVI/NDWI time series; disaster management framing
- Stack: Python, Streamlit, Celery/Redis

In **Relevance to this thesis**, be SPECIFIC and actionable. Not "this is relevant to flood mapping." Instead: what exactly can be **borrowed** (architecture, preprocessing step, loss function, index, data split, metric, augmentation)? What is **directly comparable** (a baseline to beat, a metric to report)? Where does it **differ** from the thesis setup (SAR vs optical, susceptibility vs inundation, different resolution)? Is it a **method to cite**, a **baseline**, a **dataset source**, or a **cautionary example**? If the paper is only tangentially related, say that honestly — a note that admits low relevance is more useful than one that manufactures it.

## Required output structure

Write the file to `docs/papers/<bibtex_key>.md` using EXACTLY this structure:

```
---
title: "<full title>"
authors: "<all authors>"
year: <year>
venue: "<journal/conference + volume/issue/pages>"
doi: "<doi or 'not reported'>"
bibtex_key: <key>
zotero_pdf: "<absolute path to the PDF>"
pages: <n>
sensors: [<e.g. Sentinel-1, Sentinel-2, none>]
task: "<e.g. flood inundation mapping / susceptibility mapping / rainfall estimation>"
model: "<e.g. U-Net, ConvLSTM, transformer>"
tags: [<5-10 lowercase keywords>]
---

# <Title>

> **TL;DR** — 3-4 sentences: what they did, how, and the headline result (with numbers).

## Problem & Motivation
<Why this work exists; the gap it addresses. 1-2 paragraphs.>

## Method / Architecture
<Detailed. Architecture (encoder/decoder, layers, attention), loss function, optimizer, LR,
epochs, batch size, augmentation, hyperparameters. Bullet-breakdown of the pipeline.
Anything not reported → say so explicitly.>

## Data
<Sensors + bands. Study area/region + dates. Spatial resolution. Number of images/patches/tiles,
patch size. Train/val/test split. How ground truth was produced. Public dataset names.
Preprocessing (speckle filtering, atmospheric correction, normalization, cloud masking).
Spectral indices used, WITH formulas as given in the paper.>

## Results
<Key quantitative results WITH page cites. Use a Markdown table for baseline comparisons.
State which baselines they beat and by how much. If no baselines were run, say so.>

## Limitations
<Stated by the authors AND any you observe (single study area, small test set, no uncertainty,
no held-out test, etc.).>

## Relevance to this thesis
<THE KEY SECTION. Bullets. Specific and actionable per the thesis context above.>

## Keywords / Tech
<Bulleted: models, sensors, indices, frameworks/libraries, datasets, techniques.>

## Notable quotes
<2-4 verbatim quotes worth citing directly, each with a page number.>

## Pointer
Full PDF: `<absolute path>`
Cite as: `\cite{<bibtex_key>}`
```

## What to return

Your final message must be **one line only**: the file path written, the headline metric, and any problems (e.g. "PDF is a scan, no text layer", "no metrics reported", "paper is about X not floods"). **Do not paste the note back** — it is already on disk.
