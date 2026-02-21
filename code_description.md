# Code Descriptions

This document provides a brief description of each script found in the `Code/` directory of the Rebellion Paper repository.

## Python Scripts

### `00_parish_processing_consolidated.py`
The primary data processing pipeline. It loads and cleans the *Valor Ecclesiasticus* dataset, creates geographic "flow" lines for monastic income and expenditure, and performs spatial joins with ancient parish shapefiles. It aggregates economic data, rebellion muster points, and gentry seats to the parish level, preparing the final dataset for analysis.

### `01_rebel_var_mods.py`
Extends the rebellion-related variables. It calculates parish proximity to rebel muster points (within 10km) and assigns parishes to specific "hosts" (gentlemen involved in the rebellion) based on convex hulls and distance (within 50km). It also creates dummy variables for influential surnames like Darcy, Percy, and Neville.

### `02_news_day.py`
Calculates the travel time for news to reach different parishes. It constructs a cost surface based on transport networks (roads, shipping routes, and historical evidence) and uses a Least Cost Path algorithm (MCP) to determine the number of days it would take for news to travel from Louth Park Abbey to all northern parishes.

### `03_drought_processing.py`
Processes scPDSI drought data from the Old World Drought Atlas. It calculates average drought intensity for 1, 2, 3, 5, and 10-year windows leading up to and including 1536. The script reprojects the grid cell coordinates from WGS84 (EPSG:4326) to the British National Grid (EPSG:27700) and exports the results to `Data/Processed/drought_intensity_bng.csv`.

### `04_drought_parish_join.py`
Creates a shapefile of drought grid cells (`Data/Processed/drought_cells.shp`) and samples these cells at the centroid of each parish in `northParishFlows`. It attaches the 1, 2, 3, 5, and 10-year drought intensity averages to the parish shapefile as new variables (`drought_1` through `drought_10`).

### `DAG_maker.py`
Uses the `networkx` and `matplotlib` libraries to generate Directed Acyclic Graphs (DAGs). These visualizations represent the hypothesized causal relationships between variables such as population, wealth, monastic land tenure, and the probability of rebellion.

### `functions.py`
A utility module containing the `prettyReg` function. This function streamlines the process of running regressions (OLS, Negative Binomial, Poisson, Logit) using `statsmodels`, handling variable renaming and producing formatted summary tables.

### `gentry_fee_analysis.py`
Uses a Cross-Encoder machine learning model to match the names of rebellious gentlemen with the counter-parties of monastic fee payments in the *Valor Ecclesiasticus*. This analysis explores the economic ties between the monastic system and the secular gentry who led the rebellion.

### `gridRegs.py`
Performs grid-based Poisson regressions at various resolutions (2km, 5km, 10km, and 20km). It analyzes the relationship between monastic presence and rebellion indicators (musters and seats) within standardized geographic units rather than irregular parishes.

### `prettyRegs.py`
The main execution script for parish-level statistical models. It utilizes the functions in `functions.py` to run a battery of Logit and Poisson regressions, testing the impact of monastic economic variables on rebellion outcomes. It automatically exports the results to LaTeX tables.

### `surname_id_assigner_from_land_paper.py`
A utility script that uses a Cross-Encoder ML model to match and assign unique IDs to surnames across different datasets. This ensures consistent identification of families and individuals in the various records used in the study.

### `jn_01_playground.ipynb`
A Jupyter notebook used for exploratory data analysis, testing new models, and preliminary visualizations.

### `jn_99_income_counties.ipynb`
A Jupyter notebook used for descriptive analysis of monastic income. It compares total net income and per capita net income of religious houses between the North of England and the rest of the country.

## R Scripts

### `conley_regs.R`
Runs Logit regressions with Conley standard errors. This approach accounts for spatial autocorrelation in the data by adjusting standard errors based on specified distance cutoffs (20km to 200km).

### `grid_regs.R`
The R implementation of grid-based regressions. It performs Poisson models on muster and seat counts across different grid sizes, exploring the robustness of the results to different geographic aggregations.

### `parish_logits.R`
Executes parish-level Logit and Poisson regressions. It focuses on the relationship between specific monastic variables (land, tithes, alms, net income) and rebellion outcomes, producing LaTeX tables for the paper.

### `PSM.R`
Implements Propensity Score Matching (PSM) and Inverse-Probability-Weighting (IPW). It estimates the treatment effect of monastic land ownership on rebellion using weighted Logit and Cox Proportional Hazards models to control for selection bias.

### `survival_analysis.R`
Conducts survival analysis using Cox Proportional Hazards models. It treats the timing of the rebellion as a "risk" process, estimating how monastic presence influenced the speed and likelihood of a parish joining the rebellion after the initial outbreak.
