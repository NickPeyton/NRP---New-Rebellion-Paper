# sensitivity_thresholds.R
#
# Tests coefficient stability across alternative threshold choices:
#
#   1. Gentleman proximity buffer (20 km default in jn_05)
#      Tests: 10, 15, 20, 25, 30 km
#      Method: recomputes proximity dummies from raw gentleman coordinates
#              (Data/Raw/CSV/main_gentlemen.csv) at each threshold.
#
#   2. Distance-decay threshold (12.5 km default in jn_00)
#      NOTE: the _dw variables are precomputed in Python with a hard 12.5 km
#      cutoff. Re-testing this threshold requires modifying jn_00 and
#      regenerating northParishFlows.shp. This script cannot test it in R.
#      A note is printed; see jn_00_parish_processing_consolidated.ipynb.
#
# Outcome: primary (logit) — main outcome throughout the paper.
# Controls: same as parish_logits.R Section 1.
# Outputs:
#   Output/Tables/sensitivity_gent_buffer.tex
#   Console: coefficient and p-value table across thresholds

pacman::p_load(
  sf, tidyverse, dplyr, stargazer,
  lmtest, sandwich
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Standardize continuous variables (same as parish_logits.R)
for (v in c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
            "lLStax_pc", "distScot", "lpopC",
            "area", "mean_slope", "wet_1535", "wet_1536")) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

controls <- c("lLStax_pc", "wet_1535", "wet_1536", "lpopC",
              "uplands", "lowlands", "area", "mean_slope", "distScot")

# ============================================================================
# Section 1: Gentleman proximity buffer sensitivity
# ============================================================================

gent_csv <- read.csv("Data/Raw/CSV/main_gentlemen.csv")

# Convert to sf in WGS84, reproject to British National Grid (EPSG:27700)
gent_sf <- st_as_sf(gent_csv, coords = c("Longitude", "Latitude"), crs = 4326)
gent_sf <- st_transform(gent_sf, crs = 27700)

# Parish centroids in BNG
parish_centroids <- st_centroid(st_transform(pdf, crs = 27700))

# Role groupings matching jn_05 logic
loyal_roles  <- gent_sf[gent_sf$Active_Loyalist == 1 | gent_sf$Reluctant_Loyalist == 1, ]
rebel_roles  <- gent_sf[gent_sf$Rebel_Participant == 1 | gent_sf$Active_Rebel == 1, ]

# Compute proximity dummy at a given buffer distance (metres)
compute_proximity <- function(parishes, gent_subset, buffer_m) {
  if (nrow(gent_subset) == 0) return(rep(0L, nrow(parishes)))
  union_buf <- st_union(st_buffer(gent_subset, dist = buffer_m))
  as.integer(lengths(st_intersects(parishes, union_buf)) > 0)
}

buffer_thresholds_km <- c(10, 15, 20, 25, 30)

results_list <- list()
coef_table   <- data.frame()

for (km in buffer_thresholds_km) {
  buf_m <- km * 1000
  parish_centroids$mg_loyal_sens <- compute_proximity(parish_centroids, loyal_roles, buf_m)
  parish_centroids$mg_rebel_sens <- compute_proximity(parish_centroids, rebel_roles, buf_m)

  df <- as.data.frame(pdf)
  df$mg_loyal_sens <- parish_centroids$mg_loyal_sens
  df$mg_rebel_sens <- parish_centroids$mg_rebel_sens

  formula <- as.formula(paste(
    "primary ~ mg_loyal_sens + mg_rebel_sens +",
    paste(controls, collapse = " + ")
  ))
  model <- glm(formula, data = df, family = binomial(link = "logit"))
  results_list[[as.character(km)]] <- model

  ct <- coeftest(model, vcov = vcovHC(model, type = "HC3"))
  coef_table <- rbind(coef_table, data.frame(
    buffer_km   = km,
    variable    = c("mg_loyal_sens", "mg_rebel_sens"),
    coefficient = ct[c("mg_loyal_sens", "mg_rebel_sens"), "Estimate"],
    se          = ct[c("mg_loyal_sens", "mg_rebel_sens"), "Std. Error"],
    p_value     = ct[c("mg_loyal_sens", "mg_rebel_sens"), "Pr(>|z|)"]
  ))
}

cat("\n========== GENTLEMAN PROXIMITY BUFFER SENSITIVITY ==========\n")
cat("Outcome: primary (logit). HC3 robust SEs.\n\n")
cat(sprintf("%-12s %-20s %10s %8s %8s\n",
            "Buffer (km)", "Variable", "Coef", "SE", "p-value"))
cat(strrep("-", 62), "\n")
for (i in seq_len(nrow(coef_table))) {
  cat(sprintf("%-12s %-20s %10.4f %8.4f %8.4f\n",
              coef_table$buffer_km[i],
              coef_table$variable[i],
              coef_table$coefficient[i],
              coef_table$se[i],
              coef_table$p_value[i]))
}
cat("\n")

# Stargazer table: one column per buffer threshold
buffer_cov_labels <- c(
  "Loyalist Gentleman (buffer)",
  "Rebel Gentleman (buffer)",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "ln(Population)", "Distance to Scotland"
)
hide_geo <- c("Constant", "uplands", "lowlands", "area", "mean_slope", "distScot")

stargazer(
  results_list,
  type             = "latex",
  title            = "Sensitivity: Gentleman Proximity Buffer Threshold (Primary Logit)",
  label            = "tab:sensitivity_gent",
  column.labels    = paste0(buffer_thresholds_km, " km"),
  omit             = hide_geo,
  covariate.labels = buffer_cov_labels,
  add.lines        = list(c("Geographic Controls", rep("Y", length(buffer_thresholds_km)))),
  align            = TRUE,
  column.sep.width = ".5pt",
  omit.stat        = c("aic"),
  table.placement  = "H",
  out              = "Output/Tables/sensitivity_gent_buffer.tex"
)

cat("Table written: Output/Tables/sensitivity_gent_buffer.tex\n\n")

# ============================================================================
# Section 2: Distance-decay threshold — not testable in R
# ============================================================================

cat("========== DISTANCE-DECAY THRESHOLD (12.5 km) ==========\n")
cat("The _dw variables (llo_dw, lsl_dw, lbl_dw, lti_dw and _dwpc/_dwsk variants)\n")
cat("are precomputed in jn_00_parish_processing_consolidated.ipynb with a hard\n")
cat("12.5 km inverse-distance cutoff. Testing alternative thresholds requires\n")
cat("modifying the DECAY_KM parameter in that notebook and regenerating\n")
cat("northParishFlows.shp. Re-run parish_logits.R Section 2 after each rebuild.\n")
cat("\nSuggested alternatives to test: 8 km, 10 km, 15 km, 20 km.\n")
