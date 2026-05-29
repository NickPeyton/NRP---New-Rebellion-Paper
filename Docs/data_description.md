# Data Descriptions
*New Rebellion Paper — Pilgrimage of Grace (1536)*

This document describes all active data files in the repository.

**Last updated:** 2026-04-18

---

## Raw Data — `Data/Raw/CSV/`

### `clark_land_vals.csv`
Parish-level estimates of land-use areas (arable, pasture, meadow, etc.) and annual rental values derived from historical tithe files and land surveys. Used to derive normalization denominators for monastic stocks.
- **Source:** Clark, Gregory (2010), *Historical Tithe Surveys*
- **Used by:** `01_jn_clark_land_values.ipynb`

### `monastic_opposition.csv`
Records of monastic houses categorized by their degree of opposition to the Henrician Reformation (e.g., Crown interference, execution of superiors).
- **Source:** Historical records of monastic suppression and state trials
- **Used by:** `jn_06_monastic_opposition_vars.ipynb`

### `Merged Early Modern Towns Full Data.xlsx`
Comprehensive town-level demographic and economic dataset. Used for population control variables.
- **Source:** Wallis, P., & Udale, G. (Early Modern Towns project)
- **Used by:** `jn_00_parish_processing_consolidated.ipynb`

### `ValorLineItems.csv`
Individual line-item records from the *Valor Ecclesiasticus* (1535). Contains monastic income/expenditure and counterparty names.
- **Source:** Transcribed from the *Valor Ecclesiasticus* (Record Commission edition)
- **Used by:** `jn_00_parish_processing_consolidated.ipynb`, `gentry_fee_analysis.py`

### `NationalArchivesData.csv`
Supplementary monastic records from the National Archives, used for cross-referencing *Valor* data.
- **Source:** National Archives (TNA)
- **Used by:** `jn_00_parish_processing_consolidated.ipynb`

### `LincsRebels.csv`
Historical list of rebel participants from the Lincolnshire rising (October 1536).
- **Used by:** `jn_00_parish_processing_consolidated.ipynb`

### `drought_coords.csv` / `drought_values.csv`
Coordinates and Palmer Drought Severity Index (scPDSI) values from the Old World Drought Atlas grid.
- **Source:** Cook et al. (2015), Old World Drought Atlas (OWDA)
- **Used by:** `jn_03_drought_processing.ipynb`

### `main_gentlemen.csv`
Individual records for secular gentlemen active in 1536, including seat coordinates and role dummies (Active_Rebel, Neutral, etc.).
- **Source:** Constructed from historical biographical records
- **Used by:** `jn_05_gentlemen_parish_join.ipynb`

### `HRV/` — Historical Rural Values
`HRVtable2a.csv` through `HRVtableA7b.csv`. Parish-level agricultural values and taxable wealth.
- **Source:** Broadberry et al. (2015)
- **Used by:** `jn_00_parish_processing_consolidated.ipynb`

---

## Raw Data — `Data/Raw/GIS/BNG Projections/`
*(Projected in British National Grid, EPSG:27700)*

- `AncientParishesBNG.shp`: Polygon shapefile of pre-modern parish boundaries.
- `rebPoints.shp`: Point locations of rebel muster sites.
- `gentlemenInvolved.shp`: Point locations of rebellious gentry seats.
- `TerrainZones.shp`: Polygon shapefile of terrain classifications.
- `CombinedPop.shp` / `SheailParishPops1525ND.shp`: Population estimate shapefiles.
- `friarPoints.shp`: Locations of mendicant friaries.
- `LouthParkAbbey.shp`: Origin point for news transmission analysis.
- `gough_routes.shp`: Medieval road routes from the Gough Map.
- `shippingDissolved.shp`: Historical coastal and river shipping routes.

---

## Processed Data — `Data/Processed/`

### `northParishFlows.shp`
**The primary analytical dataset.** Polygon shapefile of ancient northern parishes containing all constructed variables (monastic econ, rebellion outcomes, geographic controls, drought, and elite/opposition dummies).
- **Source:** Output of the `jn_00` to `jn_07` processing pipeline.

### `northParishFlows_pre_jn06.shp`
A versioned backup of the primary dataset taken before the monastic opposition variables from `jn_06` were added. Maintained for diagnostic comparison and regression stability tests.

### `northParishFlows_pre_jn07.shp`
A versioned backup of the primary dataset taken before the dissolved small monasteries transfer variables from `jn_07` were added. Maintained for diagnostic comparison and regression stability tests.

### `drought_intensity_bng.csv`
Multi-year drought intensity averages reprojected to BNG.
- **Used by:** `jn_04_drought_parish_join.ipynb`.

### `drought_cells.shp`
Intermediate spatial representation of the OWDA drought grid cells.
