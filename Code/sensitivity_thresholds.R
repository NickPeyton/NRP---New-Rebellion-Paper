# sensitivity_thresholds.R
#
# Tests coefficient stability across alternative threshold choices for the
# elite-grievance gentleman variables used in elite_vs_commons.R.
#
#   Section 1: Buffer proximity sensitivity (binary dummies)
#     Tests 10, 15, 20, 25, 30 km buffers for:
#       - Snubbed family (Family_Snub == 1)
#       - Court officer  (Court_Office == 1)
#       - Any gentleman  (all 39)
#     Recomputes dummies from raw gentleman coordinates at each threshold.
#
#   Section 2: IDW specification sensitivity (continuous scores)
#     Tests flat-zone radii 5, 10, 15, 20, 25 km for the same three categories.
#     IDW weight: w(d) = 1 if d <= flat_radius, else flat_radius / d.
#     Recomputes IDW scores from raw gentleman coordinates at each radius.
#
# Outcome: primary (logit) -- main outcome throughout the paper.
# Controls: same as elite_vs_commons.R / parish_logits.R Section 1.
# Outputs:
#   Output/Tables/sensitivity_gent_buffer.tex
#   Output/Tables/sensitivity_gent_idw.tex
#   Console: coefficient and p-value tables

pacman::p_load(
  sf, tidyverse, dplyr, stargazer,
  lmtest, sandwich
)

PROJECT_ROOT <- tryCatch(
  normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), "..")),
  error = function(e) normalizePath(getwd())
)
setwd(PROJECT_ROOT)

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Standardize continuous variables (same as parish_logits.R / elite_vs_commons.R)
for (v in c(
  "lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
  "lLStax_pc", "distScot", "lpopC",
  "area", "mean_slope", "wet_1535", "wet_1536"
)) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

controls <- c(
  "lLStax_pc", "wet_1535", "wet_1536", "lpopC",
  "uplands", "lowlands", "area", "mean_slope", "distScot"
)

# ---------------------------------------------------------------------------
# Load gentleman data and reproject to BNG
# ---------------------------------------------------------------------------
gent_csv <- read.csv("Data/Raw/CSV/main_gentlemen.csv")
gent_sf <- st_as_sf(gent_csv, coords = c("Longitude", "Latitude"), crs = 4326)
gent_sf <- st_transform(gent_sf, crs = 27700)

# Parish centroids in BNG
parish_centroids <- st_centroid(st_transform(pdf, crs = 27700))

# Role groupings: snubbed family, court officer, any gentleman
gent_subsets <- list(
  fsnub = gent_sf[gent_sf$Family_Snub == 1, ],
  court = gent_sf[gent_sf$Court_Office == 1, ]
)

cat(sprintf(
  "Gentleman counts: fsnub=%d, court=%d\n",
  nrow(gent_subsets$fsnub),
  nrow(gent_subsets$court)
))

# ============================================================================
# Section 1: Buffer proximity sensitivity (binary dummies)
# ============================================================================

# Compute binary proximity dummy at a given buffer distance (metres)
compute_proximity <- function(parishes, gent_subset, buffer_m) {
  if (nrow(gent_subset) == 0) {
    return(rep(0L, nrow(parishes)))
  }
  union_buf <- st_union(st_buffer(gent_subset, dist = buffer_m))
  as.integer(lengths(st_intersects(parishes, union_buf)) > 0)
}

buffer_thresholds_km <- c(10, 15, 20, 25, 30)

buffer_results <- list()
buffer_coef_table <- data.frame()

for (km in buffer_thresholds_km) {
  buf_m <- km * 1000

  # Compute binary dummies for each category at this buffer
  parish_centroids$sens_fsnub <- compute_proximity(parish_centroids, gent_subsets$fsnub, buf_m)
  parish_centroids$sens_court <- compute_proximity(parish_centroids, gent_subsets$court, buf_m)

  df <- as.data.frame(pdf)
  df$sens_fsnub <- parish_centroids$sens_fsnub
  df$sens_court <- parish_centroids$sens_court

  formula <- as.formula(paste(
    "primary ~ sens_fsnub + sens_court +",
    paste(controls, collapse = " + ")
  ))
  model <- glm(formula, data = df, family = binomial(link = "logit"))
  buffer_results[[as.character(km)]] <- model

  ct <- coeftest(model, vcov = vcovHC(model, type = "HC3"))
  sens_vars <- c("sens_fsnub", "sens_court")
  buffer_coef_table <- rbind(buffer_coef_table, data.frame(
    buffer_km   = km,
    variable    = sens_vars,
    coefficient = ct[sens_vars, "Estimate"],
    se          = ct[sens_vars, "Std. Error"],
    p_value     = ct[sens_vars, "Pr(>|z|)"]
  ))
}

cat("\n========== BUFFER PROXIMITY SENSITIVITY ==========\n")
cat("Outcome: primary (logit). HC3 robust SEs.\n")
cat("Variables: Snubbed Family, Court Officer (binary dummies)\n\n")
cat(sprintf(
  "%-12s %-20s %10s %8s %8s\n",
  "Buffer (km)", "Variable", "Coef", "SE", "p-value"
))
cat(strrep("-", 62), "\n")
for (i in seq_len(nrow(buffer_coef_table))) {
  cat(sprintf(
    "%-12s %-20s %10.4f %8.4f %8.4f\n",
    buffer_coef_table$buffer_km[i],
    buffer_coef_table$variable[i],
    buffer_coef_table$coefficient[i],
    buffer_coef_table$se[i],
    buffer_coef_table$p_value[i]
  ))
}
cat("\n")

# Stargazer table: one column per buffer threshold
buffer_cov_labels <- c(
  "Snubbed Family (buffer)",
  "Court Officer (buffer)",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "ln(Population)"
)
hide_geo <- c("Constant", "uplands", "lowlands", "area", "mean_slope", "distScot")

stargazer(
  buffer_results,
  type             = "latex",
  title            = "Sensitivity: Gentleman Buffer Threshold (Primary Logit)",
  label            = "tab:sensitivity_gent_buffer",
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
# Section 2: IDW specification sensitivity (continuous scores)
# ============================================================================

# Extract BNG coordinates as matrices for vectorized distance computation
parish_coords <- st_coordinates(parish_centroids) # n_parishes x 2
n_parishes <- nrow(parish_coords)

# Precompute gentleman coordinate matrices for each subset
gent_coords <- lapply(gent_subsets, function(g) st_coordinates(g))

# Compute IDW scores for a given flat-zone radius
# w(d) = 1 if d <= flat_m, else flat_m / d
compute_idw <- function(parish_xy, gent_xy, flat_m) {
  n_g <- nrow(gent_xy)
  if (n_g == 0) {
    return(rep(0, nrow(parish_xy)))
  }

  # Pairwise distances: n_parishes x n_gentlemen
  dx <- outer(parish_xy[, 1], gent_xy[, 1], "-")
  dy <- outer(parish_xy[, 2], gent_xy[, 2], "-")
  dist_m <- sqrt(dx^2 + dy^2)

  # Weight: capped at 1 inside flat zone
  weights <- ifelse(dist_m <= flat_m, 1.0, flat_m / dist_m)
  rowSums(weights)
}

flat_zone_km <- c(5, 10, 15, 20, 25)

idw_results <- list()
idw_coef_table <- data.frame()

for (fz_km in flat_zone_km) {
  flat_m <- fz_km * 1000

  df <- as.data.frame(pdf)
  # Compute IDW scores for each category at this flat-zone radius
  df$idw_fsnub <- compute_idw(parish_coords, gent_coords$fsnub, flat_m)
  df$idw_court <- compute_idw(parish_coords, gent_coords$court, flat_m)

  # Standardize the IDW scores (same z-score convention as main regressions)
  for (v in c("idw_fsnub", "idw_court")) {
    df[[v]] <- scale(df[[v]], center = TRUE, scale = TRUE)[, 1]
  }

  formula <- as.formula(paste(
    "primary ~ idw_fsnub + idw_court +",
    paste(controls, collapse = " + ")
  ))
  model <- glm(formula, data = df, family = binomial(link = "logit"))
  idw_results[[as.character(fz_km)]] <- model

  ct <- coeftest(model, vcov = vcovHC(model, type = "HC3"))
  idw_vars <- c("idw_fsnub", "idw_court")
  idw_coef_table <- rbind(idw_coef_table, data.frame(
    flat_zone_km = fz_km,
    variable     = idw_vars,
    coefficient  = ct[idw_vars, "Estimate"],
    se           = ct[idw_vars, "Std. Error"],
    p_value      = ct[idw_vars, "Pr(>|z|)"]
  ))
}

cat("\n========== IDW SPECIFICATION SENSITIVITY ==========\n")
cat("Outcome: primary (logit). HC3 robust SEs.\n")
cat("Variables: Snubbed Family, Court Officer (IDW scores)\n")
cat("IDW: w(d) = 1 if d <= flat zone, else flat_zone / d. Standardized.\n\n")
cat(sprintf(
  "%-15s %-20s %10s %8s %8s\n",
  "Flat zone (km)", "Variable", "Coef", "SE", "p-value"
))
cat(strrep("-", 67), "\n")
for (i in seq_len(nrow(idw_coef_table))) {
  cat(sprintf(
    "%-15s %-20s %10.4f %8.4f %8.4f\n",
    idw_coef_table$flat_zone_km[i],
    idw_coef_table$variable[i],
    idw_coef_table$coefficient[i],
    idw_coef_table$se[i],
    idw_coef_table$p_value[i]
  ))
}
cat("\n")

# Stargazer table: one column per flat-zone radius
idw_cov_labels <- c(
  "Snubbed Family IDW",
  "Court Officer IDW",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "ln(Population)"
)

stargazer(
  idw_results,
  type             = "latex",
  title            = "Sensitivity: Gentleman IDW Flat-Zone Radius (Primary Logit)",
  label            = "tab:sensitivity_gent_idw",
  column.labels    = paste0(flat_zone_km, " km"),
  omit             = hide_geo,
  covariate.labels = idw_cov_labels,
  add.lines        = list(c("Geographic Controls", rep("Y", length(flat_zone_km)))),
  align            = TRUE,
  column.sep.width = ".5pt",
  omit.stat        = c("aic"),
  table.placement  = "H",
  out              = "Output/Tables/sensitivity_gent_idw.tex"
)

cat("Table written: Output/Tables/sensitivity_gent_idw.tex\n\n")

# ============================================================================
# Section 3: Distance-decay threshold — not testable in R
# ============================================================================

cat("========== DISTANCE-DECAY THRESHOLD (12.5 km) ==========\n")
cat("The _dw variables (llo_dw, lsl_dw, lbl_dw, lti_dw and _dwpc/_dwsk variants)\n")
cat("are precomputed in jn_00_parish_processing_consolidated.ipynb with a hard\n")
cat("12.5 km inverse-distance cutoff. Testing alternative thresholds requires\n")
cat("modifying the DECAY_KM parameter in that notebook and regenerating\n")
cat("northParishFlows.shp. Re-run parish_logits.R Section 2 after each rebuild.\n")
cat("\nSuggested alternatives to test: 8 km, 10 km, 15 km, 20 km.\n")
