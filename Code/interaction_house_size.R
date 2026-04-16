# interaction_house_size.R
#
# Formal test of the large-house vs. small-house differential effect.
# The paper claims large-house land predicts rebellion more strongly than
# small-house land. This script tests that directly via:
#
#   (1) Side-by-side coefficients: lsm_arak vs lbg_arak in the same model
#       (already in parish_logits.R full models — reproduced here for focus).
#
#   (2) Wald test: H0: coef(lbg_arak) = coef(lsm_arak).
#       A significant result means the large-house effect is statistically
#       distinguishable from the small-house effect.
#
#   (3) Stratum regressions: separate samples for parishes with above-median
#       lbg_arak vs. below-median (robustness: does the full-model result
#       hold within each stratum?).
#
# Outcomes: muster (logit), primary (logit), seats (Poisson).
# Outputs:
#   Output/Tables/interaction_house_size.tex   (coefficients + Wald test rows)
#   Console: Wald test results for each outcome

pacman::p_load(
  sf, tidyverse, dplyr, stargazer,
  lmtest, sandwich, car, jsonlite
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)

# Load pretty dictionary for labels
pretty_dict <- fromJSON("Code/pretty_dict.json")

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Standardize continuous variables (same as parish_logits.R Section 1)
for (v in c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
            "lLStax_pc", "distScot", "lpopC",
            "area", "mean_slope", "wet_1535", "wet_1536")) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

controls <- c("lLStax_pc", "wet_1535", "wet_1536", "lpopC",
              "uplands", "lowlands", "area", "mean_slope", "distScot")

# Both house-size land variables enter simultaneously so their coefficients
# are directly comparable under the same controls.
monastic_both <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
                   "smHouse", "bigHouse", "friary")

df <- as.data.frame(pdf)

# ============================================================================
# Section 1: Full models with both sm and bg land — coefficients
# ============================================================================

fit_full <- function(outcome, family) {
  formula <- as.formula(paste(
    outcome, "~", paste(c(monastic_both, controls), collapse = " + ")
  ))
  glm(formula, data = df, family = family)
}

m_muster  <- fit_full("muster",  binomial(link = "logit"))
m_primary <- fit_full("primary", binomial(link = "logit"))
m_seats   <- fit_full("seats",   poisson())

# ============================================================================
# Section 2: Wald tests — H0: coef(lbg_arak) = coef(lsm_arak)
# ============================================================================

cat("\n========== WALD TESTS: coef(lbg_arak) = coef(lsm_arak) ==========\n")
cat("H0: large-house land and small-house land have equal effects.\n")
cat("A significant result supports the large-house differential claim.\n\n")

wald_results <- list()
for (nm in c("muster", "primary", "seats")) {
  model <- get(paste0("m_", nm))
  wt    <- linearHypothesis(model, "lbg_arak = lsm_arak",
                             vcov. = vcovHC(model, type = "HC3"))
  wald_results[[nm]] <- wt
  cat(sprintf("--- %s ---\n", toupper(nm)))
  print(wt)
  cat("\n")
}

# ============================================================================
# Section 3: Stargazer table with manual Wald p-value rows
# ============================================================================

hide_vars <- c("Constant", "uplands", "lowlands", "area", "mean_slope", "distScot")
house_vars_labels <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak", "smHouse", "bigHouse", "friary", "lLStax_pc", "wet_1535", "wet_1536", "lpopC")
cov_labels <- unlist(pretty_dict[house_vars_labels])

# Extract Wald p-values for add.lines
wald_p <- sapply(wald_results, function(wt) sprintf("%.3f", wt[2, "Pr(>F)"]))

stargazer(
  m_muster, m_primary, m_seats,
  type             = "latex",
  title            = "Large vs. Small Monastery Land: Differential Effect Test",
  label            = "tab:house_size_interaction",
  column.labels    = c("Muster", "Primary", "Seats"),
  omit             = hide_vars,
  covariate.labels = cov_labels,
  add.lines        = list(
    c("Geographic Controls", "Y", "Y", "Y"),
    c("Wald $p$: lbg = lsm", wald_p["muster"], wald_p["primary"], wald_p["seats"])
  ),
  align            = TRUE,
  column.sep.width = ".5pt",
  omit.stat        = c("aic"),
  table.placement  = "H",
  out              = "Output/Tables/interaction_house_size.tex"
)

cat("Table written: Output/Tables/interaction_house_size.tex\n\n")

# ============================================================================
# Section 4: Stratum regressions (above vs. below median lbg_arak)
# ============================================================================

cat("========== STRATUM REGRESSIONS (lbg_arak above vs. below median) ==========\n")
cat("Tests whether the full-model result holds within each sub-sample.\n\n")

# Use the pre-standardised lbg_arak from df (median = 0 after z-scoring)
df_high <- df[df$lbg_arak >= 0, ]
df_low  <- df[df$lbg_arak <  0, ]

cat(sprintf("High-lbg parishes: %d   Low-lbg parishes: %d\n\n",
            nrow(df_high), nrow(df_low)))

for (nm in c("muster", "primary")) {
  formula <- as.formula(paste(
    nm, "~",
    paste(c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
            "smHouse", "friary", controls), collapse = " + ")
    # bigHouse omitted in high stratum (no within-group variation)
  ))
  m_high <- glm(formula, data = df_high, family = binomial(link = "logit"))
  m_low  <- glm(formula, data = df_low,  family = binomial(link = "logit"))

  cat(sprintf("--- %s | High-lbg stratum ---\n", toupper(nm)))
  print(coeftest(m_high, vcov = vcovHC(m_high, type = "HC3"))[
    c("lsm_arak", "lbg_arak"), ])
  cat(sprintf("--- %s | Low-lbg stratum ---\n", toupper(nm)))
  print(coeftest(m_low, vcov = vcovHC(m_low, type = "HC3"))[
    c("lsm_arak", "lbg_arak"), ])
  cat("\n")
}
