# Code Descriptions

This document provides a brief description of each script found in the `Code/` directory of the Rebellion Paper repository.

## Python Scripts

### `00_parish_processing_consolidated.py`
The primary data processing pipeline. It loads and cleans the *Valor Ecclesiasticus* dataset, creates geographic "flow" lines for monastic income and expenditure, and performs spatial joins with ancient parish shapefiles. It aggregates economic data, rebellion muster points, and gentry seats to the parish level, preparing the final dataset for analysis. New variables added: `bigHouse` (dummy for houses with net income > £200/yr), `smHouse` (dummy ≤ £200/yr), distance-decay weighted land and tithe variables (`lo_dw`, `sm_dw`, `ti_dw`) using a 12.5 km threshold, per capita and per sq km denominator versions of all main monastic variables, `distScot` (distance from parish centroid to the Scottish border, approximated as the northern boundary of the north English counties), and terrain land-use shares: `pct_arable`, `pct_pastoral`, `pct_ag_land` (fraction of parish area classified as arable, pastoral, or agricultural respectively, derived from an area-weighted intersection with `TerrainZones.shp`).

### `01_rebel_var_mods.py`
Extends the rebellion-related variables. It calculates parish proximity to rebel muster points (within 10km) and assigns parishes to specific "hosts" (gentlemen involved in the rebellion) based on convex hulls and distance (within 50km). It also creates dummy variables for influential surnames like Darcy and Neville.

### `02_news_day.py`
Calculates the travel time for news to reach different parishes. It constructs a cost surface based on transport networks (roads, shipping routes, and historical evidence) and uses a Least Cost Path algorithm (MCP) to determine the number of days it would take for news to travel from Louth Park Abbey to all northern parishes.

### `03_drought_processing.py`
Processes scPDSI drought data from the Old World Drought Atlas. It calculates average drought intensity for 1, 2, 3, 5, and 10-year windows leading up to and including 1536. The script reprojects the grid cell coordinates from WGS84 (EPSG:4326) to the British National Grid (EPSG:27700) and exports the results to `Data/Processed/drought_intensity_bng.csv`.

### `04_drought_parish_join.py`
Creates a shapefile of drought grid cells (`Data/Processed/drought_cells.shp`) and samples these cells at the centroid of each parish in `northParishFlows`. It attaches the 1, 2, 3, 5, and 10-year drought intensity averages to the parish shapefile as new variables (`drought_1` through `drought_10`).

### `jn_survival_analysis.ipynb`
Python translation of `survival_analysis.R`. Loads `northParishFlows.shp`, applies identical data processing and standardisation steps, fits three nested Cox Proportional Hazards models (via `lifelines.CoxPHFitter`), prints model summaries, and writes a stargazer-style LaTeX table to `Output/Tables/survival.tex`. Updated to include an events-vs-parameters convergence warning and a mild ridge penalizer (`penalizer=0.01`) for model 3 to prevent Newton-step NaN failures under sparse events.

### `jn_05_gentlemen_parish_join.ipynb`
Joins `main_gentlemen.csv` to the parish shapefile using the same 20 km proximity logic applied to the Percy/disgruntled-gentlemen variables in `jn_01`. Converts gentleman seat coordinates from WGS84 (EPSG:4326) to British National Grid (EPSG:27700), then for each role category buffers the relevant points by 20 000 m and flags parishes whose centroid falls within the union. Adds eight binary columns to `northParishFlows.shp`: `mg_any` (any gentleman), `mg_rebel` (Rebel_Participant or Active_Rebel), `mg_act_reb` (Active_Rebel only), `mg_part` (Rebel_Participant only), `mg_loyal` (Active_Loyalist or Reluctant_Loyalist), `mg_act_loy` (Active_Loyalist only), `mg_neutral` (Neutral), `mg_rel_reb` (Reluctant_Rebel only).

### `DAG_maker.py`
Uses the `networkx` and `matplotlib` libraries to generate Directed Acyclic Graphs (DAGs). These visualizations represent the hypothesized causal relationships between variables such as population, wealth, monastic land tenure, and the probability of rebellion. Outputs to `Output/Images/Graphs/LittleRebellionDAG.png` and `Output/Images/Graphs/BigRebellionDAG.png`.

### `functions.py`
A utility module containing the `prettyReg` function. This function streamlines the process of running regressions (OLS, Negative Binomial, Poisson, Logit) using `statsmodels`, handling variable renaming and producing formatted summary tables.

### `gentry_fee_analysis.py`
Uses a Cross-Encoder machine learning model to match the names of rebellious gentlemen with the counter-parties of monastic fee payments in the *Valor Ecclesiasticus*. This analysis explores the economic ties between the monastic system and the secular gentry who led the rebellion.

### `gridRegs.py`
Performs grid-based Poisson regressions at various resolutions (2km, 5km, 10km, and 20km). It analyzes the relationship between monastic presence and rebellion indicators (musters and seats) within standardized geographic units rather than irregular parishes.

### `prettyRegs.py`
The main execution script for parish-level statistical models. It utilizes the functions in `functions.py` to run a battery of Logit and Poisson regressions, testing the impact of monastic economic variables on rebellion outcomes. It automatically exports the results to LaTeX tables.

## R Scripts

### `check_data.R` *(root directory)*
Performs data quality checks on the main processed dataset — missing value counts, value ranges, and summary statistics for land variables (logged and raw). Outputs to console only; does not write files.

### `AIPW.R`
Unified IPW/CBPS script (merged from former `AIPW.R`, `AIPW_plots.R`, `AIPW_plots_ownOther.R`). Two sections: (1) main specification using small/large monastery land per arable km² (`lsm_arak`, `lbg_arak`, `lti_arak`, `lal_arak`, `lni_arak`), and (2) ownOther specification using on-site vs. off-site land per arable km² (`lown_arak`, `loth_arak`). Each section produces a stargazer table and four coefficient plots. Geographic control standardised to `distScot` (was `Y_COORD`) for consistency with `parish_logits.R`. Outputs `Output/Tables/IPW.tex` and eight PNGs to `Output/Images/Graphs/` (`ipw_logit_*_coefficients[_ownOther].png`, `ipw_cox_coefficients[_ownOther].png`).

### `conley_regs.R`
Runs Logit regressions with Conley standard errors. This approach accounts for spatial autocorrelation in the data by adjusting standard errors based on specified distance cutoffs (20km to 200km). Outputs `Output/Tables/conley.tex`.

### `diagnostics.R`
Performs diagnostic checks on the main processed dataset: missing value summaries, correlation matrix, and variance inflation factors (VIF) from a logit model. Outputs to console only; does not write files.

### `grid_regs.R`
The R implementation of grid-based regressions. Poisson models on muster, primary, and seat counts across grid sizes (2km, 5km, 10km, 20km). Updated to use `lni_arak` (net income per arable km²) in place of `lnetInc`. Outputs `Output/Tables/grid_muster.tex`, `grid_primary.tex`, `grid_seats.tex`.

### `parish_logits.R`
Unified parish-level logit script (merged from former `parish_logits.R` and `parish_logits_interactions.R`). Section 1: individual and progressive Logit/Poisson regressions for muster, primary, and seats using monastic variables per arable km² (`lsm_arak`, `lbg_arak`, `lti_arak`, `lal_arak`, `lni_arak`, plus house-size dummies and friary). Includes full-model robustness check, DAG regressions, and VIF analysis. Section 2: distance-weighted interaction models (raw, per-capita, per-sq-km). Outputs 15+ LaTeX tables to `Output/Tables/`.

### `regression_plots.R`
Unified coefficient-plot script (merged from former `regression_plots.R`, `regression_plots_landOwned.R`, `progressive_regression_plots.R`, `progressive_regression_plots_ownOther.R`, `progressive_regression_plots_landOwned.R`). Five sections producing coefficient plots for muster, primary, and seats: (1) full-model sm/bg/ti/al/ni per arable km², (2) full-model land owned per arable km², (3) progressive sm/bg per arable km², (4) progressive own/oth per arable km², (5) progressive land owned per arable km². All monastic variables now use `_arak` (per arable km²). Geographic control standardised to `distScot` (was `Y_COORD`) to match `parish_logits.R` table specifications. Outputs 15 PNGs to `Output/Images/Graphs/`.

### `survival_analysis.R`
Unified survival analysis script (merged from former `survival_analysis.R` and `survival_plots.R`). Section 1: three nested Cox PH models (monastic → + taxation/population → + geography) with a stargazer table. Section 2: coefficient plots for each Cox model. All monastic variables use `_arak` (per arable km²); adds `lni_arak`. Geographic control standardised to `distScot` (was `Y_COORD`) for consistency with `parish_logits.R`. Outputs `Output/Tables/survival.tex` and three PNGs to `Output/Images/Graphs/`.

### `old_vs_new_grievances.R`
Tests whether pre-1536 structural variables ("old" grievances: taxation, population, weather, geography, drought) or Dissolution-specific variables ("new" grievances: monastic land, tithes, alms per arable km², house dummies, friary) better explain rebellion. Fits old-only, new-only, and combined models for muster, primary, and seats. Reports McFadden pseudo-R² for each; runs a likelihood-ratio test of H₀ = old-only vs. H₁ = old + new to test the joint significance of the new variables. Outputs `Output/Tables/old_vs_new_{muster,primary,seats}.tex` with R² and LR test rows.

### `interaction_house_size.R`
Formal test of the large-house vs. small-house differential effect. Three sections: (1) full models with both `lsm_arak` and `lbg_arak` simultaneously for muster, primary, and seats; (2) Wald tests (HC3 robust) of H₀: coef(lbg_arak) = coef(lsm_arak) for each outcome; (3) stratum regressions splitting on median `lbg_arak`. Outputs `Output/Tables/interaction_house_size.tex` with Wald p-value rows, and Wald test results + stratum regression summaries to console.

### `sensitivity_thresholds.R`
Tests coefficient stability for the two key threshold choices: (1) gentleman proximity buffer — recomputes `mg_loyal`/`mg_rebel` dummies from raw coordinates (`main_gentlemen.csv`) at 10, 15, 20, 25, and 30 km and refits the primary logit at each threshold; (2) distance-decay threshold (12.5 km) — notes that this is baked into the Python notebooks and cannot be varied in R. Outputs `Output/Tables/sensitivity_gent_buffer.tex` and a console coefficient-stability table.

### `elite_vs_commons.R`
Compares "elite" (gentleman proximity dummies) vs. "commons" (monastic economic footprint) explanatory frameworks. Updated to use monastic variables per arable km² (`lsm_arak`, `lbg_arak`, `lti_arak`, `lal_arak`). Fits logit (muster, primary) and Poisson (seats) regressions for each framework plus a combined model. Prints HC3-robust coefficient tables and McFadden pseudo-R² comparison to console; also writes stargazer tables with R² rows to `Output/Tables/elite_vs_commons_{muster,primary,seats}.tex`.
