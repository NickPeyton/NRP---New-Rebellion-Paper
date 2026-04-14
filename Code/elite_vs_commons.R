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
  lmtest, sandwich, stargazer
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# ---------------------------------------------------------------------------
# Standardize continuous variables (z-score, same as parish_logits.R)
# ---------------------------------------------------------------------------
continuous_vars <- c(
  "lsm_arak", "lbg_arak", "lti_arak", "lal_arak",
  "lLStax_pc", "wet_1535", "wet_1536", "lpopC",
  "area", "mean_slope", "distScot"
)
for (v in continuous_vars) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

# ---------------------------------------------------------------------------
# Variable sets
# ---------------------------------------------------------------------------

# "Elite": proximity (within 20km) to any loyalist or any rebel gentleman seat
#   mg_loyal = Active_Loyalist OR Reluctant_Loyalist (from main_gentlemen.csv)
#   mg_rebel = Rebel_Participant OR Active_Rebel
elite_vars <- c("mg_loyal", "mg_rebel")

# "Commons": monastic economic footprint at the parish level (per arable km²)
#   lsm_arak = ln(small-house land / arable km²)
#   lbg_arak = ln(large-house land / arable km²)
#   lti_arak = ln(tithe income / arable km²)
#   lal_arak = ln(alms income / arable km²)
commons_vars <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak")

# Shared controls: taxes, population, weather shocks, geography
controls <- c("lLStax_pc", "wet_1535", "wet_1536", "lpopC",
              "uplands", "lowlands", "area", "mean_slope", "distScot")

# Outcome specifications
outcomes <- list(
  muster  = list(dep = "muster",  family = binomial(link = "logit")),
  primary = list(dep = "primary", family = binomial(link = "logit")),
  seats   = list(dep = "seats",   family = poisson())
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
df <- as.data.frame(pdf)   # drop sf geometry for glm

fit_models <- function(dep, family) {
  elite_f    <- as.formula(paste(dep, "~", paste(c(elite_vars,   controls), collapse = " + ")))
  commons_f  <- as.formula(paste(dep, "~", paste(c(commons_vars, controls), collapse = " + ")))
  combined_f <- as.formula(paste(dep, "~", paste(c(elite_vars, commons_vars, controls), collapse = " + ")))

  list(
    elite    = glm(elite_f,    data = df, family = family),
    commons  = glm(commons_f,  data = df, family = family),
    combined = glm(combined_f, data = df, family = family)
  )
}

results <- lapply(outcomes, function(o) fit_models(o$dep, o$family))

# ---------------------------------------------------------------------------
# Print coefficients with HC3 robust standard errors
# ---------------------------------------------------------------------------
cat("\n========== COEFFICIENT ESTIMATES (HC3 robust SEs) ==========\n")
for (outcome_name in names(results)) {
  for (model_name in names(results[[outcome_name]])) {
    cat(sprintf("\n--- %s | %s model ---\n", toupper(outcome_name), model_name))
    print(coeftest(results[[outcome_name]][[model_name]],
                   vcov = vcovHC(results[[outcome_name]][[model_name]], type = "HC3")))
  }
}

# ---------------------------------------------------------------------------
# McFadden pseudo-R² comparison table
# ---------------------------------------------------------------------------
cat("\n========== McFadden PSEUDO-R² COMPARISON ==========\n\n")
cat(sprintf("%-10s  %-10s  %-10s  %-10s\n", "Outcome", "Elite", "Commons", "Combined"))
cat(strrep("-", 46), "\n")

r2_table <- lapply(names(results), function(outcome_name) {
  models <- results[[outcome_name]]
  r2 <- sapply(models, function(m) round(mcfadden_r2(m, df), 4))
  cat(sprintf("%-10s  %-10.4f  %-10.4f  %-10.4f\n",
              outcome_name, r2["elite"], r2["commons"], r2["combined"]))
  r2
})
names(r2_table) <- names(results)

cat("\n")
cat("Elite vars:    mg_loyal, mg_rebel\n")
cat("Commons vars:  lsm_arak, lbg_arak, lti_arak, lal_arak\n")
cat("Controls:      lLStax_pc, wet_1535, wet_1536, lpopC, uplands, lowlands, area, mean_slope, distScot\n")
cat("\nNote: McFadden's pseudo-R² = 1 - logL(model) / logL(null). Higher = better fit.\n")

# ---------------------------------------------------------------------------
# Stargazer tables — one per outcome
# ---------------------------------------------------------------------------

hide_geo <- c("Constant", "uplands", "lowlands", "area", "mean_slope", "distScot")

ec_cov_labels <- c(
  "Loyalist Gentleman (20 km)",
  "Rebel Gentleman (20 km)",
  "ln(Small Monastery Land / Arable km\\textsuperscript{2})",
  "ln(Large Monastery Land / Arable km\\textsuperscript{2})",
  "ln(Tithe / Arable km\\textsuperscript{2})",
  "ln(Alms / Arable km\\textsuperscript{2})",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "ln(Population)"
)

for (outcome_name in names(results)) {
  models <- results[[outcome_name]]
  r2     <- sapply(models, function(m) round(mcfadden_r2(m, df), 4))

  stargazer(
    models$elite, models$commons, models$combined,
    type             = "latex",
    title            = paste0("Elite vs. Commons Frameworks — ",
                              toupper(outcome_name)),
    label            = paste0("tab:ec_", outcome_name),
    column.labels    = c("Elite", "Commons", "Combined"),
    omit             = hide_geo,
    covariate.labels = ec_cov_labels,
    add.lines        = list(
      c("Geographic Controls", "Y", "Y", "Y"),
      c("McFadden R\\textsuperscript{2}",
        sprintf("%.4f", r2["elite"]),
        sprintf("%.4f", r2["commons"]),
        sprintf("%.4f", r2["combined"]))
    ),
    align            = TRUE,
    column.sep.width = ".5pt",
    omit.stat        = c("aic"),
    table.placement  = "H",
    out              = paste0("Output/Tables/elite_vs_commons_", outcome_name, ".tex")
  )
}

cat("\nTables written to Output/Tables/elite_vs_commons_{muster,primary,seats}.tex\n")
