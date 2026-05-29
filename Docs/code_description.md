# Code Descriptions

This document provides a description of each script and notebook found in the `Code/` directory of the Rebellion Paper repository.

## Python Notebooks (.ipynb)

### `jn_00_parish_processing_consolidated.ipynb`
The primary data processing pipeline. It loads and cleans the *Valor Ecclesiasticus* dataset, creates geographic "flow" lines for monastic income and expenditure, and performs spatial joins with ancient parish shapefiles. It aggregates economic data, rebellion muster points, and gentry seats to the parish level.

### `jn_01_rebel_var_mods.ipynb`
Extends the rebellion-related variables. It calculates parish proximity to rebel muster points (within 10km) and assigns parishes to specific "hosts" based on convex hulls and distance.

### `jn_02_news_day.ipynb`
Calculates the travel time for news to reach different parishes. It constructs a cost surface based on transport networks and uses a Least Cost Path algorithm to determine travel time from Louth Park Abbey.

### `jn_03_rebellion_animation.ipynb`
Generates a frame-by-frame animation illustrating the spread of news and actual rebel musters across the North during October and November 1536. Saves outputs as an MP4 and GIF.

### `jn_03_drought_processing.ipynb`
Processes scPDSI drought data from the Old World Drought Atlas. Reprojects coordinates to BNG and calculates multi-year intensity averages.

### `jn_04_drought_parish_join.ipynb`
Joins the processed drought intensity data to the parish shapefile (`northParishFlows`) by sampling at parish centroids.

### `jn_05_gentlemen_parish_join.ipynb`
Joins the `main_gentlemen.csv` dataset to parishes using 20km proximity and IDW logic. Adds binary and weighted indicators for various gentleman roles (Loyalist, Rebel, etc.).

### `jn_06_monastic_opposition_vars.ipynb`
Processes monastic opposition data (Crown interference, etc.) and joins it to the parish shapefile using spatial proximity and IDW weighting.
### `jn_07_dissolved_monks_transfer.ipynb`
Calculates `dissolved_L`, the total net income of dissolved small monasteries assigned to their nearest large monastery of the same order, and joins this exposure to parishes via IDW and buffers.

## R Notebooks (.ipynb)

### `jn_r_00_moran_diagnostics.ipynb`
R-kernel implementation to analyze spatial autocorrelation (Moran's I) in key outcomes and explanatory variables across distance bands (20km to 200km). Helps inform spatial Conley standard error distance cutoff choices. Writes `Output/Tables/morans_nb01.tex`.

### `jn_r_01_logits.ipynb`
R-kernel implementation of logit and distance-weighted interaction models. Fits monastic variables against rebellion outcomes (muster, primary, seats). Fits standard and Conley spatial HAC standard errors (100km Bartlett). Converted from `Code/r_01_logits.R`.

### `jn_r_02_IPW.ipynb`
R-kernel implementation of Inverse-Probability-Weighted (IPW/CBPS) logit and survival models, estimating treatment effects using Conley SEs. Converted from `Code/r_02_AIPW.R`.

### `jn_r_03_survival_analysis.ipynb`
R-kernel implementation of Cox Proportional Hazards models and Schoenfeld residual tests on the timing of the rebellion's spread. Converted from `Code/r_03_survival_analysis.R`.

### `jn_r_04_shapley_owen.ipynb`
R-kernel implementation of exact Owen-value decomposition of McFadden R² partitioning variance among Elite (proximity to snubbed/courtier families), Commons (monastic economic variables), and Controls.

### `jn_r_05_entropy_balancing.ipynb`
R-kernel robustness check for `jn_r_02_IPW.ipynb`. Replaces CBPS inverse-probability weighting with entropy balancing (`WeightIt::weightit(..., method = "ebal", moments = 1)`). EB directly imposes exact mean-balance of covariates with the continuous treatment variable without modelling a propensity score, providing a stronger identification claim. Produces the same table and plot structure as the IPW notebook, with outputs prefixed `EB_` (tables) and `eb_` (graphs).

### `jn_r_06_interaction_weather.ipynb`
R-kernel interaction analysis between intensity of monastic land and weather shocks (PDSI moisture shocks in 1535 and 1536). Tests whether the impact of monastic land is amplified by adverse weather conditions.

### `jn_r_07_interaction_contagion.ipynb`
R-kernel interaction analysis checking whether the presence of monastic land amplifies the spatial spillover effect (contagion) from nearby rebelling muster points.

### `jn_r_08_dr_total_effect.ipynb`
DoubleML Partially Linear Regression (Robinson 1988; Chernozhukov et al. 2018) estimating the **total causal effect** of monastic land on rebellion outcomes. Treatment nuisance (`E[T|X_T]`) and outcome nuisance (`E[Y|X_Y]`) use geographic/proximity covariates only — omitting tithes, alms, population, and wealth — so the estimate flows through all causal channels. Runs 9 specifications (3 treatment types × 3 normalizations: raw, per km², per arable km²) against 3 outcomes (muster, primary, seats). Applies 5-fold cross-fitting; OLS for treatment nuisance; logistic GLM for outcome nuisance; Conley spatial HAC SEs (100 km Bartlett) on the final stage. Outputs tables to `Output/Tables/dr_total_*.tex` and coefficient plots to `Output/Images/Graphs/dr_total_*_coefs.png`.

### `jn_r_09_dr_controlled_effect.ipynb`
DoubleML PLR estimating the **direct effect** of monastic land controlling for other monastic economic variables. Treatment nuisance adds population and lay subsidy; outcome nuisance further adds normalization-matched tithes and alms. Isolates the direct effect of monastic land holding economic channels constant. Same 9 × 3 specification structure as `jn_r_08`. Outputs tables to `Output/Tables/dr_ctrl_*.tex` and plots to `Output/Images/Graphs/dr_ctrl_*_coefs.png`.

## Python Scripts (.py)

### `create_animation_notebook.py`
A utility script that programmatically generates `jn_03_rebellion_animation.ipynb` for illustrating the spread of news and musters.

## R Scripts (.R)

### `debug_diagnostics.R`
A diagnostics script running DoubleML Partially Linear Regression (PLR) using random forests and cross-validated lasso to report residual treatment and outcome standard deviations and correlations.

## Utility Files

### `pretty_dict.json`
A mapping dictionary between raw variable names and "pretty" labels used for LaTeX tables and plots.
