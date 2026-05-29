# Variable Description

This file documents the key outcome, explanatory, and control variables appearing in the analytical tables and plots.

---

## Rebellion Outcome Variables

### `muster`
Binary. Any rebel muster (per M. L. Bush, 1996) within the parish polygon.

### `primary`
Binary. First organic gatherings where local men took up arms (per Bush). The most informative indicator of local commoner rebellion.

### `seats`
Non-negative integer. Count of documented rebel gentry country seats in the parish.

### `primary_survival`
Days from news exposure (Louth Park Abbey rising) to first primary muster in the parish.

---

## Monastic Explanatory Variables
*(All monetary stocks are standardized and normalized by arable land area: `_arak`)*

- `lsm_arak`: ln(Small House Land / Arable Area).
- `lbg_arak`: ln(Large House Land / Arable Area).
- `lown_arak`: ln(On-site Land / Arable Area).
- `loth_arak`: ln(Off-site Land / Arable Area).
- `llo_arak`: ln(Total Owned Land / Arable Area).
- `lti_arak`: ln(Tithe / Arable Area).
- `lal_arak`: ln(Alms / Arable Area).
- `lni_arak`: ln(Net Income / Arable Area).
- `smHouse` / `bigHouse`: Indicators for presence of small or large monastic houses.
- `friary`: Presence of a mendicant friary.

---

## Elite & Gentry Variables
*(Calculated as 20km buffers or Inverse-Distance-Weighted (IDW) scores)*

- `mg_any_w`: Total IDW score for any gentleman proximity.
- `mg_rebel_w`: IDW score for rebellious gentleman proximity.
- `mg_loyal_w`: IDW score for loyalist gentleman proximity.
- `mg_fsnub_w`: IDW score for gentlemen from politically snubbed families.
- `mg_court_w`: IDW score for gentlemen holding court offices.
- `mg_rebel`: Binary indicator for rebellious gentleman within 20km.
- `mg_loyal`: Binary indicator for loyalist gentleman within 20km.

---

## Monastic Opposition Variables
*(Measures of institutional threat and Crown interference)*

- `mo_ci1_w`: IDW score for "Strong" Crown interference (executions, direct displacement).
- `mo_ci05_w`: IDW score for "Mild" Crown interference (official harassment, forced elections).
- `mo_anyop_w`: IDW score for any recorded monastic opposition.

---

## Monastic Dispersal Variables
*(Measures of the demographic/economic shock of small house dissolution on large houses of the same order)*

- `dis_L_w`: IDW exposure score to the transferred net income of dissolved small monasteries.
- `dis_L_20`: Binary indicator for a receiving large monastery within 20km.
- `dis_L_g_w`: IDW exposure score to the net income of dissolved monasteries assigned via gravity model.
- `dis_L_g_20`: Binary indicator for a gravity-assigned large monastery within 20km.

---

## Controls & Geographic Variables

- `lpopC`: ln(Population) based on Wallis & Udale (1563/1600) estimates.
- `lLStax_pc`: ln(Lay Subsidy per Capita) from the 1525 assessment.
- `distScot`: Distance from the parish centroid to the Scottish border.
- `wet_1535` / `wet_1536`: PDSI moisture values (Old World Drought Atlas).
- `mean_slope` / `mean_elev`: Parish terrain characteristics.
- `uplands` / `lowlands`: Terrain classification dummies.
- `area`: Total parish area in km².

---

## Notes
- **Normalization:** Variables with the `_arak` suffix are divided by the parish's arable land area (km²) before being logged to capture the *intensity* of monastic activity.
- **IDW Weighting:** IDW scores use a 11.2 km floor and a 112 km ceiling (1/distance decay).
- **Standardization:** In regression tables, continuous variables are typically standardized (mean 0, SD 1).
