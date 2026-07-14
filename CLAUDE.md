# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a master's thesis project for satellite image analysis to identify and predict flood-prone areas. It uses Sentinel-2 satellite imagery from Sentinel Hub (Copernicus Data Space) and applies deep learning techniques for disaster management.

## Zotero Reference Library (local PDFs)

The user's papers live in a **local Zotero** library at `~/Zotero/` — no MCP server or plugin is used. Access is direct and read-only:

- **Catalog**: `~/Zotero/zotero.sqlite` (SQLite). Always open it READ-ONLY via `file:$HOME/Zotero/zotero.sqlite?immutable=1` so it is never locked/modified, even while Zotero is running.
- **PDFs**: `~/Zotero/storage/<KEY>/<filename>.pdf`
- **Main collection**: `tesis` (the papers for this thesis).

### Working rules

1. **Resolve papers to PDFs** with the helper script — do not hand-write SQL each time:
   `./scripts/zotero_index.sh tesis`  (omit `tesis` for the whole library)
   Output is tab-separated: `YEAR <TAB> AUTHORS <TAB> TITLE <TAB> ABSOLUTE_PDF_PATH`.
2. **Read a paper** by passing its resolved absolute path to the Read tool (PDFs are read directly).
3. **Never write to `zotero.sqlite`** or the `storage/` folder — the library is source-controlled by the Zotero app, not by us.
4. When citing a paper in the thesis, cross-reference `overleaf-thesis/3_3_BIBLIOGRAFIA/library.bib` — add the BibTeX entry there if missing.

## Paper Memory Notes (`docs/papers/`)

All 23 papers of the `tesis` collection have been read and distilled into long Markdown notes — a literature-review knowledge base.

**Recall order — do NOT re-read PDFs by default:**
1. Read **`docs/papers/INDEX.md`** first — comparison table + cross-cutting findings.
2. Then the individual note **`docs/papers/<bibtex_key>.md`** (~3k words each: TL;DR, Problem, Method/Architecture, Data, Results, Limitations, **Relevance to this thesis**, Keywords, Notable quotes).
3. Only open the PDF (via the note's `zotero_pdf:` frontmatter pointer) if the note is genuinely insufficient.

**Rules when writing or updating a note** (full spec in `docs/papers/_INSTRUCTIONS.md`):
- **Never invent a number.** Metrics are copied verbatim with page cites; if a paper omits one, write `not reported`. A fabricated metric in a thesis lit review is worse than a blank.
- Read the **whole** PDF — the Read tool caps at **20 pages per call**, so longer papers need multiple `pages:` ranges.
- Flag a paper's **internal contradictions** with ⚠️. Many of these papers' abstracts disagree with their own tables — **cite the table, not the abstract**.
- File name = BibTeX key, so `docs/papers/<key>.md` ↔ `\cite{<key>}` ↔ `library.bib` line up 1:1.
- `docs/papers/_manifest.json` maps bibtex_key → zotero_key → title → PDF path.

## Overleaf Thesis Document (LaTeX)

The written thesis lives in Overleaf and is cloned locally at `overleaf-thesis/` (git-ignored — it is a separate git repo with its own `origin` pointing back to Overleaf).

- **Root document**: `overleaf-thesis/TESIS_UNI_main.tex` (UNI thesis template, class `TesisUNI.cls`)
- **Structure**: content is split by folder — preamble (`0_0_PREAMBULO`), front matter (`1_*`), chapters (`2_CAPITULO1` … `2_CAPITULO6`), closing sections (`3_*`), bibliography at `3_3_BIBLIOGRAFIA/library.bib`, images under `E_IMAGENES/`.

### Working rules

1. **Always `git pull` before editing** the thesis, in case it was changed in the Overleaf web editor:
   `cd overleaf-thesis && git pull`
2. **Edit `.tex` files directly** with normal file tools.
3. **Push back to Overleaf** only when the user approves — changes go live in the editor:
   `cd overleaf-thesis && git add -A && git commit -m "..." && git push`
4. **Never commit `overleaf-thesis/` into this repo** (it is in `.gitignore`).
5. **Do not hardcode the Overleaf Git token.** Auth is via the remote URL / git credential helper; keep secrets out of files and out of committed content.

## Technology Stack

- **Language**: Python 3.11+
- **Package Manager**: UV
- **Web Framework**: Streamlit
- **Background Jobs**: Celery + Redis
- **Satellite Data**: Sentinel Hub API (Copernicus Data Space)
- **Deep Learning**: PyTorch
- **Key Libraries**: sentinelhub, numpy, pandas, folium, streamlit-folium

## Running the Application

```bash
# Install dependencies
uv sync

# Start Redis (using Docker)
docker compose up -d

# Start Celery worker (in one terminal)
uv run celery -A app.celery_app worker -l info

# Start Streamlit app (in another terminal)
uv run streamlit run app/main.py
```

## Running the Original Notebook

```bash
# Run notebook
uv run jupyter notebook analysis.ipynb
```

## Key Concepts

### Spectral Indices Used
- **NDVI** (Normalized Difference Vegetation Index): `(B08 - B04) / (B08 + B04)` - vegetation health assessment
- **NDWI** (Normalized Difference Water Index): `(B03 - B08) / (B03 + B08)` - water body detection

### Sentinel-2 Bands
- B02: Blue
- B03: Green
- B04: Red
- B08: Near Infrared (NIR)

### Data Configuration
- Resolution: 10 meters
- Max cloud coverage: 10%
- Data collection: Sentinel-2 L2A (atmospherically corrected)
- Study area: Configurable via map selection

## Project Structure

```
.
├── app/
│   ├── main.py           # Streamlit application
│   ├── celery_app.py     # Celery configuration
│   ├── tasks.py          # Background tasks
│   ├── config.py         # Application configuration
│   ├── models/
│   │   └── flood_model.py    # PyTorch CNN model
│   └── utils/
│       └── satellite.py      # Satellite data fetching
├── data/
│   ├── images/           # Downloaded satellite imagery
│   └── models/           # Trained models
├── analysis.ipynb        # Original analysis notebook
├── docker-compose.yml    # Redis container
├── pyproject.toml        # UV dependencies
└── .env                  # Environment variables
```

## Application Workflow

1. **Select Area**: Draw a rectangle on the interactive map
2. **Download Images**: Celery task downloads 5 years of satellite imagery
3. **Train Model**: CNN learns flood patterns from NDVI/NDWI time series
4. **View Results**: Interactive flood risk map with statistics

## API Configuration

The project uses Sentinel Hub from Copernicus Data Space. Configure in `.env`:
- `SH_CLIENT_ID`
- `SH_CLIENT_SECRET`
- `REDIS_URL` (default: redis://localhost:6379/0)
