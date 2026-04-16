# Variable Description

This file documents the key outcome, explanatory, and control variables that appear in the regression tables in `Output/Tables/`. Variable construction details (sources, transformations, joins) are described in `paper.tex` §Data and `data_description.md`; this file is a quick reference for what each variable *is* in the analyses.

The unit of observation throughout is the **ancient parish** (circa 1851 boundaries from the Cambridge Group), restricted to the North of England plus Lincolnshire. N varies between 974 and 1,755 across tables depending on which controls require non-missing values.

---

## Outcome variables

### `muster` — Rebel Muster Indicator
Binary. Equals 1 if any rebel muster (primary or non-primary) recorded in M. L. Bush, *The Pilgrimage of Grace* (1996) falls within the parish polygon. Used as the dependent variable in muster logits.

### `primary` — Primary Rebellion Participation Indicator
Binary. Equals 1 if a *primary* muster — one Bush identifies as the first organic gathering at which local men took up arms — falls within the parish. The most informative outcome for "did the commons of this parish rebel," because non-primary musters were often gathering points along the rebel army's line of march and may not reflect local sympathy.

### `seats` — Gentleman Seats Count
Non-negative integer. Number of country seats of gentlemen identified as participants in the rebellion (per Bush) located within the parish. Modeled with Poisson. A noisier outcome than `primary` because gentlemen's involvement was often coerced and runs through different causal channels than commoner mobilization.

### `primary_survival` — Time-to-Primary-Muster
Numeric (days). Days from the moment news of the Louth Park Abbey rising "exposed" each parish (estimated using a multimodal least-cost path over the late-medieval Gough road network, river shipping routes, and maritime routes — 20 km/day overland, 50 km/day by road or river, 100 km/day by ship) to the date of the first primary muster in that parish. Censored at the date of the rebel army's dispersal at Doncaster for parishes that never mustered. Used as the survival outcome in Cox proportional-hazards models (`survival.tex`, AIPW Cox).

---

## Monastic explanatory variables

All monastic stocks are normalized by the parish's arable land area (from the Clark land-use dataset) rather than by total parish area. The rescaling matters: it asks whether monastic activity is *intense* per unit of cultivable land in the parish, not whether the parish happens to be large. All are then logged.

### `lsm_arak` — ln(Small Monastery Land / Arable km²)
Annual income from land owned by religious houses with net income below £200/year (the 1536 dissolution threshold), summed across all parcels in the parish, divided by the parish's arable area, then logged. The "small" houses are those legally subject to dissolution in 1536. Surprisingly, this variable is null or negative across most specifications — see `claude_interpretation.md`.

### `lbg_arak` — ln(Large Monastery Land / Arable km²)
Same construction for religious houses with net income £200/year or above. These houses were *not* yet legally targeted in 1536 but were under increasing political pressure. This is the headline monastic predictor in the paper: positive and significant for primary musters across nearly every specification.

### `lti_arak` — ln(Tithe / Arable km²)
Annual tithe income paid by the parish to monastic houses through "appropriated" parish churches (and a small amount of glebe income), per arable km², logged. In the paper's framing this measures the *extractive* relationship — tithe payments traveled farther and involved less back-and-forth than tenant rents. Coefficient is consistently negative across specifications.

### `lal_arak` — ln(Alms / Arable km²)
Annual recorded almsgiving by monasteries located in the parish, per arable km², logged. Caveat: the *Valor Ecclesiasticus* only records perpetual alms given from named benefactors' bequests, so this dramatically undercounts actual monastic charity. Despite this limitation, the variable contributes meaningfully in the Shapley–Owen decomposition for muster.

### `lni_arak` — ln(Net Income / Arable km²)
Net income (gross income less land rents, fees to secular officials, perpetual alms, transfers, corrodies) of monasteries located in the parish, per arable km², logged. Behaves erratically: positive and significant in single-variable specs, often negative once the large-house dummy and large land variables are also in the regression. Reflects collinearity between "presence of a big house" and "net income of that house."

### `smHouse` — Small House Dummy
Binary. Equals 1 if any monastic house with net income below £200/year is located in the parish.

### `bigHouse` — Large House Dummy
Binary. Equals 1 if any monastic house with net income £200/year or above is located in the parish.

### `friary` — Friary Presence
Binary. Equals 1 if any friary (mendicant order house) is located in the parish. Friaries are conceptually distinct from the contemplative orders that owned most of the dissolved property.

### `lown_arak` / `loth_arak` — On-site vs. Off-site Land
ln(monastic land at the site of the religious house / arable km²) and ln(monastic land elsewhere in the same parish / arable km²). On-site land was disproportionately demesne worked by hired labor; off-site land was more likely tenanted. The paper uses the split (Table `primary_split`) to argue that the rebellion was driven by tenant land specifically, not by the monastic economy in general.

---

## Elite-grievance explanatory variables

These are inverse-distance-weighted (IDW) scores: for each parish centroid, the contribution of each gentleman within ~112 km is weighted by 1/distance (with a 11.2 km floor to prevent infinities), and summed.

### `mg_fsnub_w` — Snubbed Family IDW Score
IDW count of gentlemen from families that had been politically snubbed at court (denied office, overruled in succession disputes, etc.) in the years preceding the rising.

### `mg_court_w` — Court Officer IDW Score
IDW count of gentlemen who held formal positions at court. The hypothesis is that court officers, being closer to royal power, would be *less* likely to support rebellion in their neighborhoods. Empirically this is the strongest single predictor of *gentleman seats* (positive — i.e., seats cluster near each other) but null for muster and primary.

### Other gentleman proximity variables
The `pretty_dict.json` contains a wider set of IDW and 20km-buffer gentleman variables (`mg_rebel_w`, `mg_loyal_w`, `mg_areb_w`, `mg_aloy_w`, `mg_part_w`, `mg_neut_w`, `mg_rreb_w`, etc.) that classify gentlemen by behavior (rebel, loyalist, neutral, active rebel, reluctant rebel, etc.). These appear in the Shapley–Owen decomposition and in sensitivity tables but are not headline regressors.

### Loyalist / Rebel Gentleman buffer variables
Used in `sensitivity_gent_buffer.tex`. Binary indicators that equal 1 if any gentleman of the relevant type lies within X km of the parish centroid, swept across X = 10, 15, 20, 25, 30 km. The paper's main models use the 20 km cutoff.

---

## Tax / fiscal variables

### `lLStax_pc` — ln(Lay Subsidy per Capita)
Logged per-capita 1525 lay subsidy assessment, constructed by dividing tax revenue density (shillings/sq mi) by taxpayer density (taxpayers/sq mi) from the Sheail (1972) maps and combining with population. The headline test of Hypothesis 1 (tax-burden-as-cause). Null across every muster and primary specification; positive and significant for seats only (mechanically: gentlemen lived in wealthier places).

### `LS_pc_ch` — Lay Subsidy per Capita Change
Percentage change in the per-capita lay subsidy assessment between 1332 and 1525, drawn from Heldring, Robinson, and Vollmer (2021). Used in the "old vs. new grievances" tables to test whether *increases* in tax (rather than levels) predict revolt. Null across all outcomes.

---

## Climate / weather variables

### `wet_1535` — Wet 1535 Weather Shock
Palmer Drought Severity Index value for 1535 in the 0.5° × 0.5° grid cell containing the parish (Old World Drought Atlas, NOAA). Higher = wetter. The wet 1535 caused the harvest failure that produced the Cravendale grain riots. Null in regressions.

### `wet_1536` — Wet 1536 Weather Shock
Same construction for 1536. Positive and significant for primary musters and seats in most specifications — the immediately-preceding harvest mattered.

### `dwx_1536` — Wet 1535 × Wet 1536 Interaction
Product term for back-to-back wet years.

### `drought_5` — 5-year Drought Index
Rolling 5-year drought index from the same source. Used in the "old vs. new grievances" tables.

---

## Population and demographic controls

### `lpopC` — ln(Population)
Logged population in towns within the parish, primarily from Wallis & Udale (combining the 1563 diocesan returns and Jan De Vries's estimates), with Buringh's updated Bairoch dataset filling in gaps. The strongest and most robust predictor of muster and primary across every specification — more people, more rebellion. Essentially zero for seats. The paper notes that switching between Wallis/Udale, Buringh, and 1500 estimates does not change the monastic results.

---

## Geographic controls

These enter as a block via the "Geographic Controls" indicator in most tables.

### `area` — Parish Area (km²)
Polygon area of the parish.

### `Y_COORD` — Latitude
Centroid latitude. Strong positive predictor in IPW models (further north → more rebellion).

### `X_COORD` — Longitude
Centroid longitude.

### `mean_slope` — Mean Slope
Mean terrain slope within the parish polygon. One of the largest single Shapley–Owen contributors among controls.

### `mean_elev` — Mean Elevation
Mean elevation within the parish polygon.

### `uplands` / `lowlands` — Terrain Type Dummies
Binary indicators from the *Atlas of Rural Settlement of England and Wales* assigning each parish to "Lowland," "Upland," or "Intermediate" based on the centroid. Intermediate is the omitted category.

### `distScot` — Distance to Scotland
Geodesic distance from the parish centroid to the Anglo-Scottish border. Negative and significant across nearly every specification — closer to the border, more rebellion. The paper's "Why the North?" section anticipates this; the data confirm it strongly enough that it should arguably be promoted from "control" to main-text predictor.

### Distance from Louth Park Abbey
Distance from the parish centroid to the rising's outbreak site. Used in some specifications to control for the spatial spread of the rebellion (with the survival framework as a more rigorous alternative).

---

## Notes on transformations

- **Standardization.** All continuous variables are standardized (mean 0, SD 1) before entering the regressions, so logit/Cox/Poisson coefficients reflect the effect of a one-standard-deviation move in the underlying variable.
- **Logging.** Monetary stocks (land, tithe, alms, net income, lay subsidy) are logged because their distributions are heavily right-skewed. The `arak` suffix denotes per-arable-km² normalization applied *before* logging.
- **Inverse-distance weighting.** Several specifications and robustness checks use IDW versions of monastic and gentleman variables (with 11.2 km floor and 112 km ceiling) to soften the impact of georeferencing errors and parish boundary changes between 1536 and 1851.
- **Hundreds.** The newer elite-vs.-commons tables cluster standard errors at the hundred level; parishes with missing hundred assignments enter as singleton clusters.

---

## Variables not used as headline regressors

A number of additional variables exist in `pretty_dict.json` and the underlying data — copyhold counts, mills, Catholic share in 1800, industrial/agricultural shares, distance to coal/market/river/London, wheat suitability, Domesday and 1332 population baselines, etc. — but do not appear as primary explanatory variables in the regression tables currently in `Output/Tables/`. They are available for further robustness work.
