# old_vs_new_grievances.R
#
# Compares two explanatory frameworks using a nested model approach:
#
#   "Old" grievances: pre-existing structural variables present before the
#   Dissolution (taxation, population, weather shocks, geography, medieval
#   structural conditions). These would predict rebellion regardless of
#   what happened to the monasteries.
#
#   "New" grievances: Dissolution-specific variables (monastic land, tithes,
#   alms per arable km², house-size dummies, friary presence). These capture
#   the economic threat from the suppression of the smaller houses in 1536.
#
# Method:
#   (1) Fit old-only, new-only, and combined models for each outcome.
#   (2) Report McFadden pseudo-R² for each.
#   (3) Likelihood-ratio (LR) test of the new variables' joint contribution
#       given the old variables (i.e., does adding "new" improve fit?).
#   (4) Stargazer table with R² rows.
#
# Outcomes: muster (logit), primary (logit), seats (Poisson).
# Outputs:
#   Output/Tables/old_vs_new_muster.tex
#   Output/Tables/old_vs_new_primary.tex
#   Output/Tables/old_vs_new_seats.tex
#   Console: McFadden R² comparison + LR tests

pacman::p_load(
  sf, tidyverse, dplyr, stargazer,
  lmtest, sandwich
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# ---------------------------------------------------------------------------
# Standardize continuous variables
# ---------------------------------------------------------------------------
continuous_vars <- c(
  "lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
  "lLStax_pc", "lpopC", "distScot", "area", "mean_slope",
  "wet_1535", "wet_1536", "drought_5", "LS_pc_ch"
)
for (v in continuous_vars) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

# ---------------------------------------------------------------------------
# Variable sets
# ---------------------------------------------------------------------------

# "Old" grievances: pre-existing structural conditions.
#   - lLStax_pc:  lay subsidy per capita (wealth in 1535)
#   - LS_pc_ch:   lay subsidy per capita change (economic trajectory)
#   - lpopC:      log parish population
#   - wet_1535/6: wet-year weather shocks (harvest disruption)
#   - drought_5:  5-year drought index (longer-run climatic stress)
#   - distScot:   proximity to Scotland (geopolitical exposure)
#   - uplands/lowlands/area/mean_slope: terrain and geography
old_vars <- c(
  "lLStax_pc", "LS_pc_ch", "lpopC",
  "wet_1535", "wet_1536", "drought_5",
  "distScot", "uplands", "lowlands", "area", "mean_slope"
)

# "New" grievances: Dissolution-specific economic threat.
#   - lsm_arak / lbg_arak: small/large monastery land per arable km²
#   - lti_arak / lal_arak / lni_arak: tithe, alms, net income per arable km²
#   - smHouse / bigHouse: house-size dummies (presence)
#   - friary: mendicant order presence
new_vars <- c(
  "lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
  "smHouse", "bigHouse", "friary"
)

outcomes <- list(
  muster  = list(dep = "muster",  family = binomial(link = "logit")),
  primary = list(dep = "primary", family = binomial(link = "logit")),
  seats   = list(dep = "seats",   family = poisson())
)

# ---------------------------------------------------------------------------
# McFadden's pseudo-R²
# ---------------------------------------------------------------------------
mcfadden_r2 <- function(model, data) {
  dep_var    <- as.character(formula(model)[[2]])
  null_model <- glm(as.formula(paste(dep_var, "~ 1")),
                    data = data, family = model$family)
  1 - as.numeric(logLik(model)) / as.numeric(logLik(null_model))
}

df <- as.data.frame(pdf)

# ---------------------------------------------------------------------------
# Fit models and run tests
# ---------------------------------------------------------------------------
cat("\n========== OLD vs. NEW GRIEVANCES ==========\n\n")
cat(sprintf("%-10s  %-10s  %-10s  %-10s  %-12s  %-8s\n",
            "Outcome", "Old R²", "New R²", "Both R²", "LR stat", "LR p"))
cat(strrep("-", 66), "\n")

all_results  <- list()
all_lr_tests <- list()

for (outcome_name in names(outcomes)) {
  dep    <- outcomes[[outcome_name]]$dep
  family <- outcomes[[outcome_name]]$family

  m_old  <- glm(as.formula(paste(dep, "~", paste(old_vars,  collapse = " + "))),
                data = df, family = family)
  m_new  <- glm(as.formula(paste(dep, "~", paste(new_vars,  collapse = " + "))),
                data = df, family = family)
  m_both <- glm(as.formula(paste(dep, "~",
                  paste(c(old_vars, new_vars), collapse = " + "))),
                data = df, family = family)

  all_results[[outcome_name]] <- list(old = m_old, new = m_new, both = m_both)

  # LR test: does adding new vars significantly improve over old-only?
  lr <- lrtest(m_old, m_both)
  all_lr_tests[[outcome_name]] <- lr

  r2_old  <- mcfadden_r2(m_old,  df)
  r2_new  <- mcfadden_r2(m_new,  df)
  r2_both <- mcfadden_r2(m_both, df)

  cat(sprintf("%-10s  %-10.4f  %-10.4f  %-10.4f  %-12.3f  %-8.4f\n",
              outcome_name, r2_old, r2_new, r2_both,
              lr$Chisq[2], lr[2, "Pr(>Chisq)"]))
}
cat("\nLR test: H0 = old-only model; H1 = old + new model.\n")
cat("Significant p means new (Dissolution-specific) vars jointly improve fit\n")
cat("beyond what old (structural) vars alone explain.\n\n")

# Print full LR test details
for (nm in names(all_lr_tests)) {
  cat(sprintf("--- %s ---\n", toupper(nm)))
  print(all_lr_tests[[nm]])
  cat("\n")
}

# ---------------------------------------------------------------------------
# Stargazer tables — one per outcome
# ---------------------------------------------------------------------------

hide_geo <- c("Constant", "uplands", "lowlands", "area", "mean_slope", "distScot")

old_labels <- c(
  "ln(Lay Subsidy)", "Lay Subsidy Change", "ln(Population)",
  "Wet 1535", "Wet 1536", "5-yr Drought Index"
)
new_labels <- c(
  "ln(Small Monastery Land / Arable km\\textsuperscript{2})",
  "ln(Large Monastery Land / Arable km\\textsuperscript{2})",
  "ln(Tithe / Arable km\\textsuperscript{2})",
  "ln(Alms / Arable km\\textsuperscript{2})",
  "ln(Net Income / Arable km\\textsuperscript{2})",
  "Small House Dummy", "Large House Dummy", "Friary"
)
both_labels <- c(old_labels, new_labels)

for (outcome_name in names(all_results)) {
  models  <- all_results[[outcome_name]]
  lr      <- all_lr_tests[[outcome_name]]
  r2_old  <- round(mcfadden_r2(models$old,  df), 4)
  r2_new  <- round(mcfadden_r2(models$new,  df), 4)
  r2_both <- round(mcfadden_r2(models$both, df), 4)
  lr_stat <- sprintf("%.3f", lr$Chisq[2])
  lr_p    <- sprintf("%.4f", lr[2, "Pr(>Chisq)"])

  stargazer(
    models$old, models$new, models$both,
    type             = "latex",
    title            = paste0("Old vs. New Grievances — ", toupper(outcome_name)),
    label            = paste0("tab:old_new_", outcome_name),
    column.labels    = c("Old", "New", "Combined"),
    omit             = hide_geo,
    covariate.labels = both_labels,
    add.lines        = list(
      c("Geographic Controls", "Y", "N", "Y"),
      c("McFadden R\\textsuperscript{2}",
        sprintf("%.4f", r2_old),
        sprintf("%.4f", r2_new),
        sprintf("%.4f", r2_both)),
      c("LR $\\chi^2$ (New $|$ Old)", "---", "---", lr_stat),
      c("LR $p$-value",               "---", "---", lr_p)
    ),
    align            = TRUE,
    column.sep.width = ".5pt",
    omit.stat        = c("aic"),
    table.placement  = "H",
    out              = paste0("Output/Tables/old_vs_new_", outcome_name, ".tex")
  )
}

cat("Tables written to Output/Tables/old_vs_new_{muster,primary,seats}.tex\n")
