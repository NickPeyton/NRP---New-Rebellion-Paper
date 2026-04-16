# elite_vs_commons.R
#
# Compares two explanatory frameworks for rebellion participation:
#   - "Elite" model:   loyalist and rebel gentleman proximity dummies (mg_loyal, mg_rebel)
#   - "Commons" model: monastic land, tithes, and alms per arable km² (lsm_arak, lbg_arak, lti_arak, lal_arak)
#
# Both models include the same tax, population, and geographic controls.
# McFadden's pseudo-R² is reported for each model and outcome to facilitate comparison.
#
# Outcomes: muster (logit), primary (logit), seats (Poisson)
# Outputs:
#   Output/Tables/elite_vs_commons_muster.tex
#   Output/Tables/elite_vs_commons_primary.tex
#   Output/Tables/elite_vs_commons_seats.tex
#   Console: McFadden R² comparison

pacman::p_load(
  sf, tidyverse, dplyr,
  lmtest, sandwich, stargazer,
  survival, jsonlite,
  car, AER, DoubleML, mlr3, mlr3learners, glmnet
)
pacman::p_load_gh("elbersb/shapley")

PROJECT_ROOT <- tryCatch(
  normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), "..")),
  error = function(e) normalizePath(getwd())
)
setwd(PROJECT_ROOT)

# Load pretty dictionary for labels
pretty_dict <- fromJSON("Code/pretty_dict.json")

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Aggregate IDW score across rebel, neutral, and loyal gentlemen.
# Distinct from existing `mg_any_w`, which sums over all 39 gentlemen
# (including reluctant rebels etc.); `mg_rnl_w` covers the 33 with a
# clean rebel/neutral/loyal stance.
pdf$mg_rnl_w <- pdf$mg_rebel_w + pdf$mg_neut_w + pdf$mg_loyal_w

# ---------------------------------------------------------------------------
# RECOMPUTE GENTLEMAN VARIABLES: 5km IDW for main models, 10km binary for Shapley-Owen
# ---------------------------------------------------------------------------
cat("Loading raw gentleman data and recomputing specs...\n")

gent_csv <- read.csv("Data/Raw/CSV/main_gentlemen.csv")
gent_sf <- st_as_sf(gent_csv, coords = c("Longitude", "Latitude"), crs = 4326)
gent_sf <- st_transform(gent_sf, crs = 27700)

# Parish centroids in BNG
parish_centroids <- st_centroid(st_transform(pdf, crs = 27700))
parish_coords <- st_coordinates(parish_centroids) # n_parishes x 2

# Role groupings: snubbed family, court officer
gent_subsets <- list(
  fsnub = gent_sf[gent_sf$Family_Snub == 1, ],
  court = gent_sf[gent_sf$Court_Office == 1, ]
)
gent_coords <- lapply(gent_subsets, function(g) st_coordinates(g))

cat(sprintf(
  "Gentleman counts: fsnub=%d, court=%d\n",
  nrow(gent_subsets$fsnub),
  nrow(gent_subsets$court)
))

# Helper function: Compute IDW scores with specified flat-zone radius
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

# Helper function: Compute binary proximity dummy
compute_proximity <- function(parishes, gent_subset, buffer_m) {
  if (nrow(gent_subset) == 0) {
    return(rep(0L, nrow(parishes)))
  }
  union_buf <- st_union(st_buffer(gent_subset, dist = buffer_m))
  as.integer(lengths(st_intersects(parishes, union_buf)) > 0)
}

# Compute 5km flat-zone IDW (main models)
cat("Computing 5km flat-zone IDW variables...\n")
pdf$mg_fsnub_w <- compute_idw(parish_coords, gent_coords$fsnub, 5000)
pdf$mg_court_w <- compute_idw(parish_coords, gent_coords$court, 5000)

# Aggregate across rebel/neutral/loyal (5km)
# Note: We don't have gent_subsets for rnl, so recompute from full gent_sf
gent_rebel <- gent_sf[gent_sf$Stance == "rebel", ]
gent_neut <- gent_sf[gent_sf$Stance == "neutral", ]
gent_loyal <- gent_sf[gent_sf$Stance == "loyal", ]

gent_coords_rnl <- list(
  rebel = st_coordinates(gent_rebel),
  neut = st_coordinates(gent_neut),
  loyal = st_coordinates(gent_loyal)
)

pdf$mg_rebel_w <- compute_idw(parish_coords, gent_coords_rnl$rebel, 5000)
pdf$mg_neut_w <- compute_idw(parish_coords, gent_coords_rnl$neut, 5000)
pdf$mg_loyal_w <- compute_idw(parish_coords, gent_coords_rnl$loyal, 5000)
pdf$mg_rnl_w <- pdf$mg_rebel_w + pdf$mg_neut_w + pdf$mg_loyal_w

# Compute 10km binary buffer (Shapley-Owen only)
cat("Computing 10km binary buffer variables for Shapley-Owen...\n")
pdf$mg_fsnub_bin_10km <- compute_proximity(parish_centroids, gent_subsets$fsnub, 10000)
pdf$mg_court_bin_10km <- compute_proximity(parish_centroids, gent_subsets$court, 10000)

# Verify recomputation
cat("IDW (5km) summary stats:\n")
cat(sprintf(
  "  mg_fsnub_w: min=%.3f, max=%.3f, mean=%.3f\n",
  min(pdf$mg_fsnub_w), max(pdf$mg_fsnub_w), mean(pdf$mg_fsnub_w)
))
cat(sprintf(
  "  mg_court_w: min=%.3f, max=%.3f, mean=%.3f\n",
  min(pdf$mg_court_w), max(pdf$mg_court_w), mean(pdf$mg_court_w)
))
cat(sprintf("\nBinary (10km) counts:\n"))
cat(sprintf("  mg_fsnub_bin_10km: %d parishes within 10km\n", sum(pdf$mg_fsnub_bin_10km)))
cat(sprintf("  mg_court_bin_10km: %d parishes within 10km\n", sum(pdf$mg_court_bin_10km)))

# ---------------------------------------------------------------------------
# Standardize continuous variables (z-score, same as parish_logits.R)
# ---------------------------------------------------------------------------
continuous_vars <- c(
  "lsm_arak", "lbg_arak", "lti_arak", "lal_arak",
  "llo_arak",
  "mg_fsnub_w", "mg_court_w", "mg_rnl_w",
  "mg_fsnub_bin_10km", "mg_court_bin_10km",
  "lLStax_pc", "wet_1535", "wet_1536", "lpopC",
  "area", "mean_slope", "distScot"
)
for (v in continuous_vars) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

# ---------------------------------------------------------------------------
# Variable sets
# ---------------------------------------------------------------------------

# "Elite": IDW intensity of proximity to gentlemen with Crown interactions:
#   - mg_fsnub_w: families snubbed by the Crown (grievance indicator)
#   - mg_court_w: held court office (favor indicator)
# Summed weights across gentlemen with w(d)=1 for d≤10km, w(d)=10/d_km beyond.
# Captures both grievance and favor channels of Crown-elite interaction.
elite_vars <- c("mg_fsnub_w", "mg_court_w")
elite_bin_vars <- c("mg_fsnub_bin_10km", "mg_court_bin_10km")

# "Commons": monastic economic footprint at the parish level (per arable km²)
#   lsm_arak = ln(small-house land / arable km²)
#   lbg_arak = ln(large-house land / arable km²)
#   lti_arak = ln(tithe income / arable km²)
#   lal_arak = ln(alms income / arable km²)
commons_vars <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak")

# Shared controls: taxes, population, weather shocks, geography
controls <- c(
  "lLStax_pc", "wet_1535", "wet_1536", "lpopC",
  "uplands", "lowlands", "area", "mean_slope", "distScot"
)

# Outcome specifications
#   `seats` stays as poisson() so that McFadden R² / Shapley-Owen remain
#   likelihood-based (quasi-families have no logLik). The Poisson point
#   estimate is consistent under mis-specification of the variance (Gourieroux
#   et al. 1984); dispersion is handled via cluster-robust SEs in the
#   inferential tables and tested explicitly below.
outcomes <- list(
  muster  = list(dep = "muster", family = binomial(link = "logit")),
  primary = list(dep = "primary", family = binomial(link = "logit")),
  seats   = list(dep = "seats", family = poisson())
)

# ---------------------------------------------------------------------------
# Helper: McFadden's pseudo-R²
# ---------------------------------------------------------------------------
mcfadden_r2 <- function(model, data) {
  dep_var <- as.character(formula(model)[[2]])
  null_model <- glm(
    as.formula(paste(dep_var, "~ 1")),
    data   = data,
    family = model$family
  )
  1 - as.numeric(logLik(model)) / as.numeric(logLik(null_model))
}

# ---------------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------------
df <- as.data.frame(sf::st_drop_geometry(pdf)) # drop sf geometry for glm / DoubleML

# Cluster id for cluster-robust SEs: parishes missing `hundred` get a
# singleton cluster id so they contribute no within-cluster correlation
# rather than being dropped. Roughly 11% of parishes have hundred == NA.
df$cluster_id <- ifelse(
  is.na(df$hundred),
  paste0("_singleton_", seq_len(nrow(df))),
  as.character(df$hundred)
)

# ---------------------------------------------------------------------------
# Survival variables (for Cox PH specifications)
#   primary_survival: days from news arrival to rebellion event, floored at 1
#   and right-censored at cox_horizon. Built from a local copy of `day` so
#   that the original column is not mutated.
# ---------------------------------------------------------------------------
cox_horizon <- 40
event_day <- df$day
event_day[is.na(event_day) | event_day < 1] <- cox_horizon
primary_day <- ifelse(df$primary == 1, event_day, cox_horizon)
df$primary_survival <- pmax(primary_day - df$news_day, 1)
df$primary_survival <- ifelse(is.na(df$primary_survival), cox_horizon, df$primary_survival)
df$primary_survival <- pmin(df$primary_survival, cox_horizon)

fit_models <- function(dep, family) {
  elite_f <- as.formula(paste(dep, "~", paste(c(elite_vars, controls), collapse = " + ")))
  elite_bin_f <- as.formula(paste(dep, "~", paste(c(elite_bin_vars, controls), collapse = " + ")))
  commons_f <- as.formula(paste(dep, "~", paste(c(commons_vars, controls), collapse = " + ")))
  combined_f <- as.formula(paste(dep, "~", paste(c(elite_vars, commons_vars, controls), collapse = " + ")))

  list(
    elite     = glm(elite_f, data = df, family = family),
    elite_bin = glm(elite_bin_f, data = df, family = family),
    commons   = glm(commons_f, data = df, family = family),
    combined  = glm(combined_f, data = df, family = family)
  )
}

results <- lapply(outcomes, function(o) fit_models(o$dep, o$family))

# ---------------------------------------------------------------------------
# Cluster-robust SE helper (cluster = hundred; singleton clusters for NAs)
# ---------------------------------------------------------------------------
cluster_vcov <- function(model) {
  vcovCL(model, cluster = df$cluster_id, type = "HC1")
}
cluster_se <- function(model) sqrt(diag(cluster_vcov(model)))

# ---------------------------------------------------------------------------
# Print coefficients with cluster-robust SEs (clustered on `hundred`)
# ---------------------------------------------------------------------------
cat("\n========== COEFFICIENT ESTIMATES (cluster-robust SEs, hundred) ==========\n")
for (outcome_name in names(results)) {
  for (model_name in names(results[[outcome_name]])) {
    cat(sprintf("\n--- %s | %s model ---\n", toupper(outcome_name), model_name))
    print(coeftest(
      results[[outcome_name]][[model_name]],
      vcov = cluster_vcov(results[[outcome_name]][[model_name]])
    ))
  }
}

# ---------------------------------------------------------------------------
# Overdispersion test for the seats (Poisson) combined model
# ---------------------------------------------------------------------------
cat("\n========== POISSON OVERDISPERSION TEST (seats) ==========\n")
disp_test <- tryCatch(
  AER::dispersiontest(results$seats$combined, trafo = 1),
  error = function(e) {
    cat("dispersiontest failed:", conditionMessage(e), "\n")
    NULL
  }
)
if (!is.null(disp_test)) {
  print(disp_test)
  cat(sprintf(
    "Dispersion estimate (alpha) = %.3f; p = %.4g\n",
    disp_test$estimate, disp_test$p.value
  ))
  cat("Interpretation: alpha>0 & p<0.05 indicates overdispersion; ",
    "cluster-robust SEs are used in tables to adjust inference.\n",
    sep = ""
  )
}

# ---------------------------------------------------------------------------
# VIF check on each combined model (flag VIF > 10)
# ---------------------------------------------------------------------------
cat("\n========== VIF (combined models) ==========\n")
for (outcome_name in names(results)) {
  v <- tryCatch(car::vif(results[[outcome_name]]$combined),
    error = function(e) NULL
  )
  if (is.null(v)) next
  cat(sprintf("\n--- %s | combined ---\n", toupper(outcome_name)))
  print(round(v, 2))
  high <- names(v)[v > 10]
  if (length(high) > 0) {
    cat("  WARNING: VIF > 10 for: ", paste(high, collapse = ", "), "\n", sep = "")
  }
}

# ---------------------------------------------------------------------------
# McFadden pseudo-R² comparison table
# ---------------------------------------------------------------------------
cat("\n========== McFadden PSEUDO-R² COMPARISON ==========\n\n")
cat(sprintf("%-10s  %-12s  %-12s  %-10s  %-10s\n", "Outcome", "Elite(IDW)", "Elite(Bin)", "Commons", "Combined"))
cat(strrep("-", 60), "\n")

r2_table <- lapply(names(results), function(outcome_name) {
  models <- results[[outcome_name]]
  r2 <- sapply(models, function(m) round(mcfadden_r2(m, df), 4))
  cat(sprintf(
    "%-10s  %-12.4f  %-12.4f  %-10.4f  %-10.4f\n",
    outcome_name, r2["elite"], r2["elite_bin"], r2["commons"], r2["combined"]
  ))
  r2
})
names(r2_table) <- names(results)

cat("\n")
cat("Elite vars:    mg_fsnub_w, mg_court_w\n")
cat("Commons vars:  lsm_arak, lbg_arak, lti_arak, lal_arak\n")
cat("Controls:      lLStax_pc, wet_1535, wet_1536, lpopC, uplands, lowlands, area, mean_slope, distScot\n")
cat("\nNote: McFadden's pseudo-R² = 1 - logL(model) / logL(null). Higher = better fit.\n")

# ---------------------------------------------------------------------------
# Stargazer tables — one per outcome
# ---------------------------------------------------------------------------

hide_geo <- c("Constant", "uplands", "lowlands", "area", "mean_slope", "distScot")

# Variable display order and labels (must match `order` regex below)
ec_var_order <- c(elite_vars, elite_bin_vars, commons_vars, "lLStax_pc", "wet_1535", "wet_1536", "lpopC")
ec_cov_labels <- unlist(pretty_dict[ec_var_order])

for (outcome_name in names(results)) {
  models <- results[[outcome_name]]
  r2 <- sapply(models, function(m) round(mcfadden_r2(m, df), 4))
  ses <- lapply(models, cluster_se)

  stargazer(
    models$elite, models$elite_bin, models$commons, models$combined,
    type = "latex",
    se = ses,
    title = paste0(
      "Elite vs. Commons Frameworks — ",
      toupper(outcome_name)
    ),
    label = paste0("tab:ec_", outcome_name),
    column.labels = c("Elite (IDW)", "Elite (Binary)", "Commons", "Combined"),
    omit = hide_geo,
    order = paste0("^", ec_var_order, "$"),
    covariate.labels = ec_cov_labels,
    add.lines = list(
      c("Geographic Controls", "Y", "Y", "Y", "Y"),
      c("SE cluster", "hundred", "hundred", "hundred", "hundred"),
      c(
        "McFadden R\\textsuperscript{2}",
        sprintf("%.4f", r2["elite"]),
        sprintf("%.4f", r2["elite_bin"]),
        sprintf("%.4f", r2["commons"]),
        sprintf("%.4f", r2["combined"])
      )
    ),
    align = TRUE,
    column.sep.width = ".5pt",
    omit.stat = c("aic"),
    table.placement = "H",
    out = paste0("Output/Tables/elite_vs_commons_", outcome_name, ".tex")
  )
}

cat("\nTables written to Output/Tables/elite_vs_commons_{muster,primary,seats}.tex\n")

# ===========================================================================
# AIPW via DoubleML PLR (partially linear regression with cross-fitting)
# ===========================================================================
#
# Replaces the prior svyglm+CBPS "AIPW" specification. That specification was
# IPW in substance — it reweighted an outcome regression but lacked the
# outcome-model augmentation that gives AIPW its double-robustness — and its
# CBPS weights were also internally inconsistent (weights balanced on a
# single treatment variable while the outcome model treated several
# variables as jointly causal).
#
# DoubleML::DoubleMLPLR with cross-fitting is the AIPW/Robinson-style
# partialling-out estimator for the partially linear model:
#     Y = D theta + g(X) + U,    D = m(X) + V
# The nuisance functions g, m are estimated by `cv_glmnet` on K-1 folds and
# theta is solved on the held-out fold. Under the usual Neyman-orthogonality
# and rate conditions theta is root-n consistent and doubly robust. Cluster-
# robust inference uses `hundred` via DoubleMLClusterData.
#
# Scale note: PLR estimates a LINEAR partial effect on the raw outcome
# (LPM-equivalent for the binary outcomes muster/primary; linear for the
# count outcome seats). Coefficients are NOT directly comparable to the
# logit / Poisson coefficients in the base GLM tables above.

lgr::get_logger("mlr3")$set_threshold("warn")

# DoubleML's mlr3 learners reject rows with NA in any modelling column, so
# build a complete-case subset matching the base GLM sample (N = 1391).
dml_cols <- c(
  unlist(lapply(outcomes, `[[`, "dep")),
  elite_vars, elite_bin_vars, commons_vars, controls, "cluster_id"
)
df_dml <- df[complete.cases(df[, dml_cols]), dml_cols]
cat(sprintf(
  "DoubleML complete-case sample: %d / %d parishes (%d dropped)\n",
  nrow(df_dml), nrow(df), nrow(df) - nrow(df_dml)
))

dml_learner <- function() lrn("regr.cv_glmnet", s = "lambda.min")

fit_dml_plr <- function(dep, treat_cols, x_cols, n_folds = 5) {
  dml_data <- DoubleMLClusterData$new(
    data         = df_dml,
    y_col        = dep,
    d_cols       = treat_cols,
    x_cols       = x_cols,
    cluster_cols = "cluster_id"
  )
  dml <- DoubleMLPLR$new(
    data    = dml_data,
    ml_l    = dml_learner(),
    ml_m    = dml_learner(),
    n_folds = n_folds,
    score   = "partialling out"
  )
  set.seed(42)
  dml$fit()
  dml
}

dml_specs <- list(
  elite     = elite_vars,
  elite_bin = elite_bin_vars,
  commons   = commons_vars,
  combined  = c(elite_vars, commons_vars)
)

dml_results <- list()
for (outcome_name in names(outcomes)) {
  dml_results[[outcome_name]] <- list()
  for (spec_name in names(dml_specs)) {
    cat(sprintf("Fitting DoubleML PLR: %s | %s\n", outcome_name, spec_name))
    dml_results[[outcome_name]][[spec_name]] <- fit_dml_plr(
      dep        = outcomes[[outcome_name]]$dep,
      treat_cols = dml_specs[[spec_name]],
      x_cols     = controls
    )
  }
}

cat("\n========== DoubleML PLR ESTIMATES (cluster-robust) ==========\n")
for (outcome_name in names(dml_results)) {
  for (spec_name in names(dml_results[[outcome_name]])) {
    cat(sprintf("\n--- DML %s | %s ---\n", toupper(outcome_name), spec_name))
    print(dml_results[[outcome_name]][[spec_name]]$summary())
  }
}

# --- Custom LaTeX table per outcome ---------------------------------------
#  (stargazer does not support DoubleML objects; we build a minimal table by
#  hand with point estimates, cluster-robust SEs, and significance stars.)
write_dml_table <- function(dml_models, out_path, title, label) {
  all_treats <- unique(unlist(lapply(dml_models, function(d) d$data$d_cols)))
  pretty <- unlist(pretty_dict[all_treats])

  lines <- c(
    "\\begin{table}[H]",
    "\\centering",
    paste0("\\caption{", title, "}"),
    paste0("\\label{", label, "}"),
    "\\begin{tabular}{lcccc}",
    "\\hline\\hline",
    " & Elite (IDW) & Elite (Binary) & Commons & Combined \\\\",
    "\\hline"
  )

  for (t in all_treats) {
    est_row <- pretty[t]
    se_row <- ""
    for (spec in c("elite", "elite_bin", "commons", "combined")) {
      d <- dml_models[[spec]]
      if (t %in% d$data$d_cols) {
        est <- as.numeric(d$coef[t])
        se <- as.numeric(d$se[t])
        p <- 2 * (1 - pnorm(abs(est / se)))
        star <- ifelse(p < 0.01, "$^{***}$",
          ifelse(p < 0.05, "$^{**}$",
            ifelse(p < 0.10, "$^{*}$", "")
          )
        )
        est_row <- paste0(est_row, " & ", sprintf("%.4f%s", est, star))
        se_row <- paste0(se_row, " & ", sprintf("(%.4f)", se))
      } else {
        est_row <- paste0(est_row, " & ")
        se_row <- paste0(se_row, " & ")
      }
    }
    lines <- c(lines, paste0(est_row, " \\\\"), paste0(se_row, " \\\\"))
  }

  n_obs <- nrow(df_dml)
  n_clust <- length(unique(df_dml$cluster_id))
  lines <- c(
    lines,
    "\\hline",
    paste0("Observations & ", n_obs, " & ", n_obs, " & ", n_obs, " & ", n_obs, " \\\\"),
    paste0("Clusters (hundred$^\\dagger$) & ", n_clust, " & ", n_clust, " & ", n_clust, " & ", n_clust, " \\\\"),
    "Geographic Controls & Y & Y & Y & Y \\\\",
    "Cross-fit folds & 5 & 5 & 5 & 5 \\\\",
    "\\hline\\hline",
    "\\multicolumn{5}{l}{\\footnotesize DoubleML partially linear regression; \\texttt{cv\\_glmnet} nuisance.} \\\\",
    "\\multicolumn{5}{l}{\\footnotesize $^\\dagger$NA hundreds enter as singleton clusters. $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$.} \\\\",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(lines, out_path)
}

for (outcome_name in names(dml_results)) {
  write_dml_table(
    dml_results[[outcome_name]],
    out_path = paste0("Output/Tables/elite_vs_commons_aipw_", outcome_name, ".tex"),
    title = paste0(
      "Elite vs.\\ Commons Frameworks (DoubleML PLR) --- ",
      toupper(outcome_name)
    ),
    label = paste0("tab:aipw_", outcome_name)
  )
}

cat("\nDoubleML tables written to Output/Tables/elite_vs_commons_aipw_{muster,primary,seats}.tex\n")

# ===========================================================================
# Cox PH — primary participation only
#
# Cluster-robust SEs use the `cluster` argument (Lin-Wei sandwich on
# `cluster_id`). The CBPS-weighted "AIPW Cox" spec from earlier revisions
# was removed: the weights had the same inconsistencies as the svyglm IPW
# block, and a proper AIPW estimator for Cox PH is not a drop-in. The
# DoubleML PLR table on `primary` above already provides a doubly-robust
# sensitivity check on the same outcome (at the cost of a linear-scale
# estimand instead of a hazard ratio).
# ===========================================================================

cox_elite_f <- as.formula(paste(
  "Surv(primary_survival, primary) ~",
  paste(c(elite_vars, controls), collapse = " + ")
))
cox_elite_bin_f <- as.formula(paste(
  "Surv(primary_survival, primary) ~",
  paste(c(elite_bin_vars, controls), collapse = " + ")
))
cox_commons_f <- as.formula(paste(
  "Surv(primary_survival, primary) ~",
  paste(c(commons_vars, controls), collapse = " + ")
))
cox_combined_f <- as.formula(paste(
  "Surv(primary_survival, primary) ~",
  paste(c(elite_vars, commons_vars, controls), collapse = " + ")
))

cox_results <- list(
  elite     = coxph(cox_elite_f, data = df, cluster = cluster_id, robust = TRUE),
  elite_bin = coxph(cox_elite_bin_f, data = df, cluster = cluster_id, robust = TRUE),
  commons   = coxph(cox_commons_f, data = df, cluster = cluster_id, robust = TRUE),
  combined  = coxph(cox_combined_f, data = df, cluster = cluster_id, robust = TRUE)
)

cat("\n========== COX PH COEFFICIENT ESTIMATES (cluster-robust) ==========\n")
for (model_name in names(cox_results)) {
  cat(sprintf("\n--- Cox PH | %s model ---\n", model_name))
  print(summary(cox_results[[model_name]])$coefficients)
}

cat("\n========== COX PH PROPORTIONAL-HAZARDS TEST (cox.zph) ==========\n")
for (model_name in names(cox_results)) {
  cat(sprintf("\n--- cox.zph | %s ---\n", model_name))
  zph <- tryCatch(cox.zph(cox_results[[model_name]]),
    error = function(e) {
      cat("cox.zph failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (!is.null(zph)) {
    print(zph)
    offenders <- rownames(zph$table)[zph$table[, "p"] < 0.05]
    offenders <- setdiff(offenders, "GLOBAL")
    if (length(offenders) > 0) {
      cat("  PH violation (p<0.05) for: ",
        paste(offenders, collapse = ", "), "\n",
        sep = ""
      )
    }
  }
}

# Robust SEs from coxph(robust=TRUE) are stored in model$var
cox_robust_se <- function(m) sqrt(diag(m$var))
cox_ses <- lapply(cox_results, cox_robust_se)

stargazer(
  cox_results$elite, cox_results$elite_bin, cox_results$commons, cox_results$combined,
  type = "latex",
  se = cox_ses,
  title = "Elite vs.\\ Commons Frameworks --- Cox PH (Primary Participation)",
  label = "tab:cox_primary",
  column.labels = c("Elite (IDW)", "Elite (Binary)", "Commons", "Combined"),
  omit = hide_geo,
  order = paste0("^", ec_var_order, "$"),
  covariate.labels = ec_cov_labels,
  add.lines = list(
    c("Geographic Controls", "Y", "Y", "Y", "Y"),
    c("SE cluster", "hundred", "hundred", "hundred", "hundred")
  ),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("rsq", "max.rsq", "logrank"),
  table.placement = "H",
  out = "Output/Tables/elite_vs_commons_cox.tex"
)

cat("\nCox PH table written to Output/Tables/elite_vs_commons_cox.tex\n")

# ===========================================================================
# Shapley-Owen decomposition of McFadden R²
#   Exact Owen-value decomposition via the elbersb/shapley package, with a
#   McFadden R² value function. Restricted to the most predictive variables
#   across the broader specifications, partitioned into:
#     elite    = mg_fsnub_w, mg_court_w, mg_rnl_w  (Crown grievance channels
#                                                   + overall gentleman presence)
#     commons  = lbg_arak, lti_arak, lal_arak      (big-house land, tithe,
#                                                   alms per arable km²)
#     controls = lpopC, lLStax_pc, mean_slope
#   Variable values sum to the full-model McFadden R²; group values are the
#   sum of variable values within each group.
# ===========================================================================

owen_decomposition <- function(data, outcome, groups,
                               family = binomial(link = "logit")) {
  null_ll <- as.numeric(logLik(suppressWarnings(
    glm(as.formula(paste(outcome, "~ 1")), data = data, family = family)
  )))

  value_fun <- function(factors) {
    if (length(factors) == 0) {
      return(0)
    }
    f <- as.formula(paste(outcome, "~", paste(factors, collapse = " + ")))
    m <- suppressWarnings(glm(f, data = data, family = family))
    1 - as.numeric(logLik(m)) / null_ll
  }

  owen_df <- shapley::owen(value_fun, groups, silent = TRUE)
  var_vals <- setNames(owen_df$value, owen_df$factor)

  list(
    variable = var_vals,
    group    = sapply(groups, function(g) sum(var_vals[g])),
    total_r2 = value_fun(unlist(groups, use.names = FALSE))
  )
}

# Two specifications:
#   Binary (10km buffer):  mg_fsnub_bin_10km, mg_court_bin_10km
#   IDW (5km flat-zone):   mg_fsnub_w, mg_court_w
so_groups_bin <- list(
  elite    = c("mg_fsnub_bin_10km", "mg_court_bin_10km"),
  commons  = c("lbg_arak", "lti_arak", "lal_arak"),
  controls = c("lpopC", "lLStax_pc", "mean_slope")
)
so_groups_idw <- list(
  elite    = c("mg_fsnub_w", "mg_court_w"),
  commons  = c("lbg_arak", "lti_arak", "lal_arak"),
  controls = c("lpopC", "lLStax_pc", "mean_slope")
)

cat("\n========== SHAPLEY-OWEN DECOMPOSITION ==========\n")
cat("(Exact Owen values via `shapley` package; may take a few minutes)\n")
cat("Binary spec:  10km buffer (mg_fsnub_bin_10km, mg_court_bin_10km)\n")
cat("IDW spec:     5km flat-zone (mg_fsnub_w, mg_court_w)\n\n")

# --- Binary (10km) ---
cat("Running binary (10km) decompositions...\n")
so_bin_muster <- owen_decomposition(df, "muster", so_groups_bin, family = binomial(link = "logit"))
so_bin_primary <- owen_decomposition(df, "primary", so_groups_bin, family = binomial(link = "logit"))
so_bin_seats <- owen_decomposition(df, "seats", so_groups_bin, family = poisson())

# --- IDW (5km) ---
cat("Running IDW (5km) decompositions...\n")
so_idw_muster <- owen_decomposition(df, "muster", so_groups_idw, family = binomial(link = "logit"))
so_idw_primary <- owen_decomposition(df, "primary", so_groups_idw, family = binomial(link = "logit"))
so_idw_seats <- owen_decomposition(df, "seats", so_groups_idw, family = poisson())

so_print <- function(tag, d, groups) {
  cat(sprintf("\n--- %s (total R\u00b2 = %.4f) ---\n", tag, d$total_r2))
  for (gname in names(groups)) {
    cat(sprintf("  [%s]  group = %.4f\n", gname, d$group[gname]))
    for (v in groups[[gname]]) {
      cat(sprintf("    %-25s %.4f\n", v, d$variable[v]))
    }
  }
}

cat("\n--- BINARY (10km) ---")
so_print("MUSTER", so_bin_muster, so_groups_bin)
so_print("PRIMARY", so_bin_primary, so_groups_bin)
so_print("SEATS", so_bin_seats, so_groups_bin)

cat("\n--- IDW (5km) ---")
so_print("MUSTER", so_idw_muster, so_groups_idw)
so_print("PRIMARY", so_idw_primary, so_groups_idw)
so_print("SEATS", so_idw_seats, so_groups_idw)

# --- LaTeX table helper --------------------------------------------------
group_titles <- c(elite = "Elite", commons = "Commons", controls = "Controls")

write_owen_table <- function(decompositions, groups, labels, caption, label, out_path) {
  outcome_cols <- toupper(names(decompositions))
  n_out <- length(decompositions)

  lines <- c(
    "\\begin{table}[H]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    paste0("\\begin{tabular}{l", strrep("r", n_out), "}"),
    "\\hline\\hline",
    paste0("Variable & ", paste(outcome_cols, collapse = " & "), " \\\\"),
    "\\hline"
  )

  for (gname in names(groups)) {
    group_vals <- sapply(decompositions, function(d) d$group[gname])
    lines <- c(
      lines,
      paste0(
        "\\textbf{", group_titles[gname], "} & ",
        paste(sprintf("\\textbf{%.4f}", group_vals), collapse = " & "),
        " \\\\"
      )
    )
    for (v in groups[[gname]]) {
      var_vals <- sapply(decompositions, function(d) d$variable[v])
      lines <- c(
        lines,
        paste0(
          "\\quad ", labels[v], " & ",
          paste(sprintf("%.4f", var_vals), collapse = " & "),
          " \\\\"
        )
      )
    }
  }

  total_vals <- sapply(decompositions, function(d) d$total_r2)
  lines <- c(
    lines,
    "\\hline",
    paste0(
      "Total McFadden $R^2$ & ",
      paste(sprintf("%.4f", total_vals), collapse = " & "), " \\\\"
    ),
    "\\hline\\hline",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(lines, out_path)
}

so_labels_bin <- unlist(pretty_dict[unlist(so_groups_bin)])
so_labels_idw <- unlist(pretty_dict[unlist(so_groups_idw)])

write_owen_table(
  list(muster = so_bin_muster, primary = so_bin_primary, seats = so_bin_seats),
  so_groups_bin,
  so_labels_bin,
  caption = "Shapley--Owen Decomposition of McFadden $R^2$ (Binary, 10km Buffer)",
  label = "tab:shapley_owen_bin",
  out_path = "Output/Tables/elite_vs_commons_shapley_owen_bin.tex"
)

write_owen_table(
  list(muster = so_idw_muster, primary = so_idw_primary, seats = so_idw_seats),
  so_groups_idw,
  so_labels_idw,
  caption = "Shapley--Owen Decomposition of McFadden $R^2$ (IDW, 5km Flat-Zone)",
  label = "tab:shapley_owen_idw",
  out_path = "Output/Tables/elite_vs_commons_shapley_owen_idw.tex"
)

cat("\nShapley-Owen tables written to:\n")
cat("  Output/Tables/elite_vs_commons_shapley_owen_bin.tex\n")
cat("  Output/Tables/elite_vs_commons_shapley_owen_idw.tex\n")
