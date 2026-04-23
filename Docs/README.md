# New Rebellion Paper

Quantitative analysis of the Pilgrimage of Grace (1536), examining the role of monastic land tenure in parish-level rebellion.

---

## Pipeline — Run Order

The pipeline has two stages. **Stage 1 must complete before Stage 2.** Within each stage, notebooks can be run in any order unless otherwise noted.

### Stage 1: Data Processing (Python / Jupyter)

Run these notebooks in order — each depends on outputs from the previous:

| Order | Notebook | Key output |
|-------|----------|------------|
| 1 | `Code/jn_00_parish_processing_consolidated.ipynb` | `Data/Processed/northParishFlows.shp` (master parish shapefile) |
| 2 | `Code/jn_01_rebel_var_mods.ipynb` | Updates `northParishFlows.shp` with rebel proximity vars |
| 3 | `Code/jn_02_news_day.ipynb` | Adds `news_day` to `northParishFlows.shp` |
| 4 | `Code/jn_03_drought_processing.ipynb` | `Data/Processed/drought_intensity_bng.csv` |
| 5 | `Code/jn_04_drought_parish_join.ipynb` | Adds drought vars to `northParishFlows.shp` |
| 6 | `Code/jn_05_gentlemen_parish_join.ipynb` | Adds gentleman proximity dummies to `northParishFlows.shp` |

> **Important:** All Stage 2 R scripts read `northParishFlows.shp`. If any Stage 1 notebook is re-run, all Stage 2 scripts must be re-run to regenerate outputs.

### Stage 2: Analysis (R)

All scripts in `Code/` read `Data/Processed/northParishFlows.shp` and are independent of each other (any run order):

| Script | Outputs |
|--------|---------|
| `parish_logits.R` | `Output/Tables/muster_monastic.tex`, `primary_monastic.tex`, `seat_monastic.tex`, `muster_all.tex`, `primary_all.tex`, `seat_all.tex`, `full_monastic.tex`, `dag.tex`, and distance-weighted tables |
| `regression_plots.R` | 15 coefficient PNGs in `Output/Images/Graphs/` |
| `survival_analysis.R` | `Output/Tables/survival.tex` + 3 Cox coefficient PNGs |
| `AIPW.R` | `Output/Tables/IPW.tex` + 8 IPW coefficient PNGs |
| `grid_regs.R` | `Output/Tables/grid_muster.tex`, `grid_primary.tex`, `grid_seats.tex` |
| `elite_vs_commons.R` | Console output only (McFadden R² comparison) |

---

## Key Variables

- **Outcomes:** `muster` (any rebellion muster), `primary` (primary muster), `seats` (muster count)
- **Monastic land (per arable km²):** `lsm_arak` (small houses), `lbg_arak` (large houses), `lti_arak` (tithes), `lal_arak` (alms), `lni_arak` (net income), `llo_arak` (total owned), `lown_arak` (on-site), `loth_arak` (off-site)
- **House-size dummies:** `smHouse` (net income ≤ £200/yr), `bigHouse` (> £200/yr)
- **Geographic control:** `distScot` (distance to Scottish border) — used consistently across all parish-level models
- **News travel:** `news_day` (days for news to travel from Louth Park Abbey, at 40 km/day default)

---

## Data

- `Data/Raw/` — never modified; original source files
- `Data/Processed/` — derived datasets; see `data_description.md` for full provenance
- Main analysis dataset: `Data/Processed/northParishFlows.shp`
