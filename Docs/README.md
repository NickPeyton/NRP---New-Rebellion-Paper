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
| 7 | `Code/jn_06_monastic_opposition_vars.ipynb` | Adds monastic opposition variables to `northParishFlows.shp` |
| 8 | `Code/jn_07_dissolved_monks_transfer.ipynb` | Adds dissolved monks transfer exposure variables to `northParishFlows.shp` |

*Note: `Code/jn_03_rebellion_animation.ipynb` (generated from `Code/create_animation_notebook.py`) can be run after Stage 1 is complete to visualize the spread of the rebellion.*

> **Important:** All Stage 2 R notebooks read `northParishFlows.shp`. If any Stage 1 notebook is re-run, the Stage 2 notebooks must be re-run to regenerate the final tables and plots.

### Stage 2: Analysis (R Notebooks)

All notebooks in `Code/` read `Data/Processed/northParishFlows.shp` and use an R kernel. They are independent of each other and can be run in any order:

| Notebook | Outputs / Key Analyses |
|----------|------------------------|
| `Code/jn_r_00_moran_diagnostics.ipynb` | Moran's I spatial autocorrelation diagnostics. Writes `Output/Tables/morans_nb01.tex`. |
| `Code/jn_r_01_logits.ipynb` | Logit and Poisson models (economic variables vs. rebellion outcomes) + distance-weighted interactions + Conley spatial HAC standard errors. Writes `Output/Tables/muster_monastic*.tex`, `primary_monastic*.tex`, `seat_monastic*.tex`, `muster_all*.tex`, `primary_all*.tex`, `seat_all*.tex`, `full_monastic*.tex`, `dag.tex`, distance-weighted/Conley table variants, and coefficient plots in `Output/Images/Graphs/`. |
| `Code/jn_r_02_IPW.ipynb` | Inverse-Probability-Weighted (IPW/CBPS) logit models, IPW Cox survival models, and Conley SEs. Writes `Output/Tables/IPW*.tex`, `conley_ipw_*.tex`, and coefficient plots to `Output/Images/Graphs/`. |
| `Code/jn_r_03_survival_analysis.ipynb` | Cox Proportional Hazards models and Schoenfeld residual tests. Writes `Output/Tables/survival*.tex` and Schoenfeld plots in `Output/Images/Graphs/`. |
| `Code/jn_r_04_shapley_owen.ipynb` | Shapley-Owen value variance decomposition partitioning McFadden R² into Elite, Commons, and Control groups. Writes LaTeX tables to `Output/Tables/`. |
| `Code/jn_r_05_entropy_balancing.ipynb` | Robustness check replacing CBPS weighting with entropy balancing (EB). Writes tables to `Output/Tables/EB_*.tex` and coefficient plots to `Output/Images/Graphs/eb_*.png`. |
| `Code/jn_r_06_interaction_weather.ipynb` | Tests interaction effects between intensity of monastic land and weather shocks (PDSI moisture shocks). |
| `Code/jn_r_07_interaction_contagion.ipynb` | Tests whether monastic land presence amplifies spatial spillovers (rebel proximity/contagion). |
| `Code/jn_r_08_dr_total_effect.ipynb` | DoubleML PLR estimating the **total causal effect** of monastic land. Writes `Output/Tables/dr_total_*.tex` and coefficient plots in `Output/Images/Graphs/`. |
| `Code/jn_r_09_dr_controlled_effect.ipynb` | DoubleML PLR estimating the **direct causal effect** of monastic land, controlling for population, wealth, tithes, and alms. Writes `Output/Tables/dr_ctrl_*.tex` and coefficient plots in `Output/Images/Graphs/`. |

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
