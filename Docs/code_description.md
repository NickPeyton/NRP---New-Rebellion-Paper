# Code Descriptions

This document provides a description of each script and notebook found in the `Code/` directory of the Rebellion Paper repository.

## Python Notebooks (.ipynb)

### `jn_00_parish_processing_consolidated.ipynb`
The primary data processing pipeline. It loads and cleans the *Valor Ecclesiasticus* dataset, creates geographic "flow" lines for monastic income and expenditure, and performs spatial joins with ancient parish shapefiles. It aggregates economic data, rebellion muster points, and gentry seats to the parish level.

### `jn_01_rebel_var_mods.ipynb`
Extends the rebellion-related variables. It calculates parish proximity to rebel muster points (within 10km) and assigns parishes to specific "hosts" based on convex hulls and distance.

### `jn_02_news_day.ipynb`
Calculates the travel time for news to reach different parishes. It constructs a cost surface based on transport networks and uses a Least Cost Path algorithm to determine travel time from Louth Park Abbey.

### `jn_03_drought_processing.ipynb`
Processes scPDSI drought data from the Old World Drought Atlas. Reprojects coordinates to BNG and calculates multi-year intensity averages.

### `jn_04_drought_parish_join.ipynb`
Joins the processed drought intensity data to the parish shapefile (`northParishFlows`) by sampling at parish centroids.

### `jn_05_gentlemen_parish_join.ipynb`
Joins the `main_gentlemen.csv` dataset to parishes using 20km proximity and IDW logic. Adds binary and weighted indicators for various gentleman roles (Loyalist, Rebel, etc.).

### `jn_06_monastic_opposition_vars.ipynb`
Processes monastic opposition data (Crown interference, etc.) and joins it to the parish shapefile using spatial proximity and IDW weighting.

### `01_jn_clark_land_values.ipynb`
Processes the Clark land-use dataset to derive arable, pastoral, and agricultural land shares for each parish, used as normalization denominators.

### `jn_10_AIPW.ipynb` to `jn_18_elite_vs_commons.ipynb`
A series of notebooks that translate the core R statistical analyses (AIPW, Grid Regressions, Interaction Models, Old vs. New Grievances, Parish Logits, etc.) into Python for cross-validation and specialized plotting.

### `jn_DAG_maker.ipynb` / `jn_survival_analysis.ipynb` / `jn_gentry_fee_analysis.ipynb`
Notebook versions of the specialized causal, survival, and machine learning name-matching analyses.

## R Notebooks (.ipynb)

### `jn_r_01_logits.ipynb`
R-kernel implementation of logit and distance-weighted interaction models. Fits monastic variables against rebellion outcomes.

### `jn_r_02_AIPW.ipynb`
R-kernel implementation of Inverse-Probability-Weighted (IPW/CBPS) logit and survival models. 
*Note: Fixed label mapping to handle list-based dictionaries from jsonlite.*

### `jn_r_03_survival_analysis.ipynb`
R-kernel implementation of Cox Proportional Hazards models and Schoenfeld residual tests.

### `jn_r_04_elite_vs_commons.ipynb`
R-kernel implementation comparing elite (gentleman) vs. commons (monastic economic) frameworks. Includes DoubleML PLR and Shapley-Owen decomposition.
*Note: Fixed stargazer compatibility issue with coxph objects.*

### `jn_r_05_old_vs_new_grievances.ipynb`
R-kernel implementation of nested model comparisons between structural "old" grievances and Dissolution-specific "new" grievances.

## Python Scripts (.py)

### `DAG_maker.py`
Generates Directed Acyclic Graphs (DAGs) representing hypothesized causal relationships. Outputs to `Output/Images/Graphs/`.

### `gentry_fee_analysis.py`
Uses a Cross-Encoder machine learning model to match rebellious gentlemen's names with monastic fee recipients in the *Valor Ecclesiasticus*.

## R Scripts (.R)

### `parish_logits.R`
The primary statistical engine for the paper. Fits various Logit and Poisson models testing the impact of monastic economic variables on rebellion.

### `AIPW.R`
Unified script for Augmented Inverse Probability Weighting (AIPW) and Covariate Balancing Propensity Score (CBPS) models.

### `elite_vs_commons.R`
Fits models comparing the explanatory power of elite-driven (gentleman) vs. commons-driven (monastic economic) variables.

### `interaction_house_size.R`
Conducts Wald tests and interaction models to test differential effects between small and large monastic houses.

### `old_vs_new_grievances.R`
Compares pre-1536 structural grievances (taxation, weather) against Dissolution-specific economic variables using LR tests.

### `regression_plots.R`
Generates coefficient plots for all major regression specifications in the paper.

### `sensitivity_thresholds.R`
Tests the stability of results across alternative distance thresholds (10km–30km) for proximity variables.

### `survival_analysis.R`
Fits Cox Proportional Hazards models to analyze the timing of the rebellion's spread.

## Utility Files

### `pretty_dict.json`
A mapping dictionary between raw variable names and "pretty" labels used for LaTeX tables and plots.
