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
Estimates inverse-probability-weighted (IPW) causal effects of monastic land ownership on rebellion outcomes. Uses Covariate Balancing Propensity Score (CBPS) weighting, weighted logit regression (`svyglm`), and a weighted Cox proportional hazards survival model. Outputs `Output/Tables/IPW.tex`.

### `AIPW_plots.R`
Creates coefficient plots visualizing IPW-weighted logit and Cox PH survival model results across three rebellion outcomes (primary, muster, seats) using CBPS-weighted regressions with 90% confidence intervals. Outputs to `Output/Images/Graphs/`: `ipw_logit_primary_coefficients.png`, `ipw_logit_muster_coefficients.png`, `ipw_logit_seats_coefficients.png`, `ipw_cox_coefficients.png`.

### `AIPW_plots_ownOther.R`
Variant of `AIPW_plots.R` that splits monastic land into site-owned (`lownLand`) vs. offsite (`lotherLand`) categories. Uses CBPS weighting on offsite land. Outputs four PNG plots with `_ownOther` suffix to `Output/Images/Graphs/`.

### `conley_regs.R`
Runs Logit regressions with Conley standard errors. This approach accounts for spatial autocorrelation in the data by adjusting standard errors based on specified distance cutoffs (20km to 200km). Outputs `Output/Tables/conley.tex`.

### `diagnostics.R`
Performs diagnostic checks on the main processed dataset: missing value summaries, correlation matrix, and variance inflation factors (VIF) from a logit model. Outputs to console only; does not write files.

### `grid_regs.R`
The R implementation of grid-based regressions. It performs Poisson models on muster and seat counts across different grid sizes (2km, 5km, 10km, 20km), exploring the robustness of the results to different geographic aggregations. Outputs `Output/Tables/grid_muster.tex`, `grid_primary.tex`, `grid_seats.tex`.

### `parish_logits.R`
Executes parish-level Logit and Poisson regressions. It focuses on the relationship between specific monastic variables (land, tithes, alms, and house-size dummies) and rebellion outcomes, producing LaTeX tables for the paper. Updated: `lnetInc` replaced by `smHouse`/`bigHouse` dummies; `dg_percy` replaced by `disg_gnt` (all disgruntled gentry surnames); `distScot` added to controls; full robustness model (all monastic variables simultaneously) added, outputting to `Output/Tables/full_monastic.tex`.

### `parish_logits_interactions.R`
Runs the same parish-level regressions as `parish_logits.R` but uses distance-decay weighted versions of monastic land and tithe variables. Tests three specifications: raw distance-weighted values, per-capita distance-weighted values, and per-sq-km distance-weighted values. Outputs 9 LaTeX tables to `Output/Tables/` with prefix `muster_dw_*`, `primary_dw_*`, `seat_dw_*`.

### `progressive_regression_plots.R`
Generates coefficient evolution plots showing how key coefficients change across four nested model specifications (land only → land + monastic → + tax/population/weather → + geography). Runs logit (muster, primary) and Poisson (seats) regressions with 90% CIs, colour-coded by specification. Outputs `Output/Images/Graphs/muster_progressive_coefficients.png`, `primary_progressive_coefficients.png`, `seats_progressive_coefficients.png`.

### `progressive_regression_plots_landOwned.R`
Variant of `progressive_regression_plots.R` using a single monolithic land ownership variable (`llandOwned`). Same four-model nesting structure. Outputs three PNG plots with `_landOwned` suffix to `Output/Images/Graphs/`.

### `progressive_regression_plots_ownOther.R`
Variant of `progressive_regression_plots.R` splitting land into site-owned (`lownLand`) vs. offsite (`lotherLand`). Same nesting structure. Outputs three PNG plots with `_ownOther` suffix to `Output/Images/Graphs/`.

### `regression_plots.R`
Generates coefficient plots from the full logit (muster, primary) and Poisson (seats) models. Extracts 90% confidence intervals and visualises them using `ggplot2`. Outputs `Output/Images/Graphs/muster_coefficients.png`, `primary_coefficients.png`, `seats_coefficients.png`.

### `regression_plots_landOwned.R`
Variant of `regression_plots.R` using the monolithic `llandOwned` variable. Outputs three PNG plots with `_landOwned` suffix to `Output/Images/Graphs/`.

### `survival_analysis.R`
Conducts survival analysis using Cox Proportional Hazards models. It treats the timing of the rebellion as a "risk" process, estimating how monastic presence influenced the speed and likelihood of a parish joining the rebellion after the initial outbreak. Outputs `Output/Tables/survival.tex`.

### `elite_vs_commons.R`
Compares two explanatory frameworks for rebellion participation: an "elite" model using proximity dummies for loyalist and rebel gentlemen seats (`mg_loyal`, `mg_rebel`), and a "commons" model using monastic land, tithe, and alms variables per sq km (`lsm_sk`, `lbg_sk`, `lti_sk`, `lal_sk`). Both models include the same tax, population, and geographic controls. Fits logit (muster, primary) and Poisson (seats) regressions for each framework plus a combined model, prints HC3-robust coefficient tables, and reports McFadden's pseudo-R² for all nine models to allow direct framework comparison. Outputs to console only.

### `survival_plots.R`
Creates coefficient plots for Cox Proportional Hazards models across three nested model specifications for the primary monastery outcome, visualising log hazard ratios with 90% confidence intervals. Outputs `Output/Images/Graphs/cox1_coefficients.png`, `cox2_coefficients.png`, `cox3_coefficients.png`.
