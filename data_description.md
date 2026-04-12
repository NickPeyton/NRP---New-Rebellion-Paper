# Data Descriptions
*New Rebellion Paper — Pilgrimage of Grace (1536)*

This document describes all active data files in the repository. Files not listed here are considered orphaned (not referenced by any current script).

**Last updated:** 2026-03-05

---

## Raw Data — `Data/Raw/CSV/`

### `ValorLineItems.csv`
Individual line-item records from the *Valor Ecclesiasticus* (1535), Henry VIII's survey of ecclesiastical income. Contains monastic house identifiers, categories of income/expenditure (land, tithes, alms, fees, etc.), counterparty names, and values in pounds.

- **Source:** Transcribed from the *Valor Ecclesiasticus* (primary source)
- **Used by:** `00_parish_processing_consolidated.py`, `gentry_fee_analysis.py`

---

### `NationalArchivesData.csv`
Supplementary records from the National Archives relating to monastic houses and associated individuals. Used to extend or cross-reference *Valor* data.

- **Source:** National Archives (primary source transcription)
- **Used by:** `00_parish_processing_consolidated.py`, `gentry_fee_analysis.py`

---

### `LincsRebels.csv`
List of rebels from Lincolnshire, the first county to rise in October 1536. Used to code rebellion participation at the parish level.

- **Source:** Historical records of rebel participants
- **Used by:** `00_parish_processing_consolidated.py`

---

### `drought_coords.csv`
Grid cell coordinates (latitude/longitude, WGS84) for the Old World Drought Atlas scPDSI grid, covering the British Isles.

- **Source:** Old World Drought Atlas (OWDA)
- **Used by:** `03_drought_processing.py`

---

### `drought_values.csv`
scPDSI (Palmer Drought Severity Index) values for each OWDA grid cell across relevant years, used to construct drought intensity measures for windows of 1, 2, 3, 5, and 10 years leading up to 1536.

- **Source:** Old World Drought Atlas (OWDA)
- **Used by:** `03_drought_processing.py`

---

### `main_gentlemen.csv`
Individual records for major secular gentlemen and nobles active during the Pilgrimage of Grace (1536), including their country seat coordinates and role classification. Each row is one individual; columns record name, title, rank, family, country seat, county, latitude/longitude (WGS84), and six mutually exclusive role dummies: `Active_Loyalist`, `Reluctant_Loyalist`, `Neutral`, `Reluctant_Rebel`, `Rebel_Participant`, `Active_Rebel`.

- **Source:** Constructed from historical records
- **Used by:** `jn_05_gentlemen_parish_join.ipynb`

---

### `HRV/` — Historical Rural Values tables (22 files)
`HRVtable2a.csv` through `HRVtableA7b.csv`. Parish-level economic data on agricultural values and taxable wealth from Broadberry et al. Used to impute or cross-reference monastic income against rural economic baselines.

- **Source:** Broadberry et al. historical rural values dataset
- **Used by:** `00_parish_processing_consolidated.py`

---

## Raw Data — `Data/Raw/GIS/BNG Projections/`

> All shapefiles below are projected in the British National Grid (EPSG:27700).

### `AncientParishesBNG.shp`
Polygon shapefile of ancient (pre-modern) parish boundaries in northern England. The primary geographic unit for all parish-level analysis.

- **Source:** Historical GIS data
- **Used by:** `00_parish_processing_consolidated.py`, `02_news_day.py`

---

### `SuburbanBufferBNG.shp`
Buffer zones around historical urban centres (combined suburban buffer). Used to control for urban proximity in the parish-level regressions.

- **Source:** Derived from Bairoch urban data
- **Used by:** `00_parish_processing_consolidated.py`

---

### `UrbanBufferBNG1.shp`
Buffer zone around historical urban centres (variant 1 — smaller radius). Used alongside `SuburbanBufferBNG.shp` to capture different definitions of urban influence.

- **Source:** Derived from Bairoch urban data
- **Used by:** `00_parish_processing_consolidated.py`

---

### `rebPoints.shp`
Point shapefile of rebel muster locations during the Pilgrimage of Grace (1536). Each point represents a documented gathering or march of rebel forces.

- **Source:** Historical records of rebel musters
- **Used by:** `00_parish_processing_consolidated.py`, `01_rebel_var_mods.py`

---

### `gentlemenInvolved.shp`
Point shapefile of gentry seats belonging to gentlemen who participated in the rebellion. Used to assign parishes to rebel "hosts" and to test surname-based gentry networks.

- **Source:** Historical records of rebel leaders
- **Used by:** `00_parish_processing_consolidated.py`, `01_rebel_var_mods.py`, `gentry_fee_analysis.py`

---

### `TerrainZones.shp`
Polygon shapefile classifying northern England into terrain zones (e.g., upland, lowland). Used as a geographic control variable.

- **Source:** Derived from digital elevation and land-use data
- **Used by:** `00_parish_processing_consolidated.py`

---

### `CombinedPop.shp`
Polygon shapefile containing combined population estimates for parishes, used to create per-capita versions of monastic economic variables.

- **Source:** Composite of Sheail and other historical population sources
- **Used by:** `00_parish_processing_consolidated.py`

---

### `SheailParishPops1525ND.shp`
Polygon shapefile of parish-level population estimates from the 1525 lay subsidy (Sheail), with no-data (ND) parishes omitted or handled. Used to create per-capita monastic variables.

- **Source:** Sheail lay subsidy data
- **Used by:** `00_parish_processing_consolidated.py`

---

### `SheailParishShillings1525ND.shp`
Polygon shapefile of parish-level taxable wealth (in shillings) from the 1525 lay subsidy (Sheail), with ND parishes handled. Used as a wealth control variable.

- **Source:** Sheail lay subsidy data
- **Used by:** `00_parish_processing_consolidated.py`

---

### `friarPoints.shp`
Point shapefile of friary locations in northern England. Used to create a variable capturing proximity to mendicant (friar) religious houses, which were distinct from the wealthier monasteries dissolved under the 1536 act.

- **Source:** Historical records of religious houses
- **Used by:** `00_parish_processing_consolidated.py`

---

### `CountiesConsolidatedBNG.shp`
Polygon shapefile of consolidated county boundaries in northern England. Used to derive the Scottish border distance variable and to aggregate parish-level data to county level.

- **Source:** Historical county boundary data
- **Used by:** `00_parish_processing_consolidated.py`

---

### `countiesBNG.shp`
Polygon shapefile of county boundaries used for cost-surface construction and clipping in the news travel time analysis.

- **Source:** Historical county boundary data
- **Used by:** `02_news_day.py`

---

### `LouthParkAbbey.shp`
Point shapefile marking the location of Louth Park Abbey (Lincolnshire), the origin point for the news travel time / Least Cost Path analysis.

- **Source:** Georeferenced from historical maps
- **Used by:** `02_news_day.py`

---

### `direct_evidence.shp`
Line or point shapefile representing historically attested news transmission routes with direct documentary evidence. Used to calibrate the cost surface in the news travel time analysis.

- **Source:** Historical records of communication routes
- **Used by:** `02_news_day.py`

---

### `indirect_evidence.shp`
Line or point shapefile representing historically inferred news transmission routes based on indirect evidence. Used to supplement `direct_evidence.shp` in calibrating the cost surface.

- **Source:** Inferred from historical records
- **Used by:** `02_news_day.py`

---

### `gough_routes.shp`
Line shapefile of medieval road routes from the Gough Map (c. 1360), a medieval road map of Britain. Used to construct the transport network component of the cost surface.

- **Source:** Gough Map (primary source), georeferenced
- **Used by:** `02_news_day.py`

---

### `shippingDissolved.shp`
Line shapefile of historical coastal and river shipping routes, dissolved into a single network layer. Used to represent water-based transport in the cost surface for the news travel time analysis.

- **Source:** Historical shipping records and coastal geography
- **Used by:** `02_news_day.py`

---

### `grid2.shp` / `grid5.shp` / `grid10.shp` / `grid20.shp`
Square grid polygon shapefiles at 2 km, 5 km, 10 km, and 20 km resolutions, covering the study area. Used as geographic aggregation units in the grid-based Poisson regression robustness checks.

- **Source:** Constructed in QGIS
- **Used by:** `grid_regs.R`

---

## Processed Data — `Data/Processed/`

### `northParishFlows.shp` *(+ companion `.dbf`, `.prj`, `.shx`, `.cpg` files)*
**The primary analytical dataset for the paper.** A polygon shapefile of ancient northern parishes containing all constructed variables: monastic economic variables (land, tithes, alms, house-size dummies, distance-weighted variants, per-capita and per-sq-km versions), rebellion outcomes (muster dummy, primary dummy, gentry seat count), geographic controls (terrain zone, urban proximity, distance to Scottish border, news travel days), population estimates, taxable wealth, drought indices, and gentleman proximity dummies. This file is the sole input to all R analysis scripts.

- **Source:** Output of the Python processing pipeline (scripts `00`–`05`)
- **Transformation:** Spatial joins and aggregation of all raw inputs to ancient parish polygons. Built by running scripts `00` through `05` in sequence. Script `05` adds eight binary proximity variables derived from `main_gentlemen.csv`: `mg_any`, `mg_rebel`, `mg_act_reb`, `mg_part`, `mg_loyal`, `mg_act_loy`, `mg_neutral`, `mg_rel_reb` — each flagging parishes within 20 km of gentleman seats in the relevant role category. Script `05` also adds eight inverse-distance-weighted (IDW) counterparts: `mg_any_w`, `mg_rebel_w`, `mg_areb_w`, `mg_part_w`, `mg_loyal_w`, `mg_aloy_w`, `mg_neut_w`, `mg_rreb_w`. Each is the sum of weights across gentlemen in the group, where w(d) = 1 for d ≤ 10 km and w(d) = 10/d_km for d > 10 km (so 0.5 at 20 km, 0.33 at 30 km, etc.).
- **Date created:** Updated through 2025–2026 analysis revisions

---

### `drought_intensity_bng.csv`
Average scPDSI drought intensity for each Old World Drought Atlas grid cell, reprojected to British National Grid coordinates. Contains columns for 1, 2, 3, 5, and 10-year average drought intensity preceding 1536.

- **Source:** `Data/Raw/CSV/drought_coords.csv`, `Data/Raw/CSV/drought_values.csv`
- **Transformation:** Averages computed over specified windows; coordinates reprojected from WGS84 (EPSG:4326) to BNG (EPSG:27700)
- **Date created:** 2025
- **Used by:** `04_drought_parish_join.py`

---

### `drought_cells.shp` *(+ companion `.dbf`, `.prj`, `.shx`, `.cpg` files)*
Polygon shapefile of OWDA drought grid cells covering the study area, with drought intensity values (1, 2, 3, 5, 10-year windows) attached as attributes. Created as an intermediate step before sampling at parish centroids.

- **Source:** `Data/Processed/drought_intensity_bng.csv`
- **Transformation:** Grid cells constructed as polygons from centroid coordinates and raster resolution; drought values joined as attributes
- **Date created:** 2025
- **Note:** Output of `04_drought_parish_join.py`; not directly referenced after the parish join step
