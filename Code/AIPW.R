# AIPW.R — Inverse-Probability-Weighted (IPW/CBPS) logit and survival models
# Sections:
#   1. Main specification: small/large monastery land per arable km²
#   2. OwnOther specification: on-site vs off-site land per arable km²
# Each section produces stargazer tables and coefficient plots.

pacman::p_load(
  sf, tidyverse, stargazer, sp, dplyr,
  cem, MatchIt, WeightIt, marginaleffects, ipw,
  survey, optmatch, conflicted, cobalt, twang,
  survival, ggplot2, broom, jsonlite
)
conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)

# Load pretty dictionary for labels
pretty_dict <- fromJSON("Code/pretty_dict.json")

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Replace NAs in terrainTyp with 'Other'
pdf$terrainTyp <- ifelse(is.na(pdf$terrainTyp), "Other", pdf$terrainTyp)
pdf$uplands    <- ifelse(pdf$terrainTyp == "Uplands",  1, 0)
pdf$lowlands   <- ifelse(pdf$terrainTyp == "Lowlands", 1, 0)
pdf$otherlands <- ifelse(pdf$terrainTyp == "Other",    1, 0)

rdf <- data.frame(pdf)
day <- 40
rdf$day <- replace(rdf$day, rdf$day < 1, day)
rdf$day <- ifelse(is.na(rdf$day), day, rdf$day)
rdf$primary_day <- rdf$day * rdf$primary
rdf$primary_day <- replace(rdf$primary_day, rdf$primary_day < 1, day)
rdf$survival <- rdf$day - rdf$news_day
rdf$primary_survival <- rdf$primary_day - rdf$news_day
rdf$primary_survival <- ifelse(is.na(rdf$primary_survival), day, rdf$primary_survival)

# Convert seats to binary (1 if seats > 1, 0 otherwise)
rdf$seats <- ifelse(rdf$seats > 1, 1, rdf$seats)

# Standardize and center continuous variables
for (v in c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
            "lown_arak", "loth_arak",
            "lLStax_pc", "lpopC", "distScot", "area", "mean_slope",
            "wet_1535", "wet_1536", "dwx_1536")) {
  rdf[[v]] <- scale(rdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

# Shared geographic / socioeconomic covariates
covar_rhs <- c(
  "lLStax_pc", "lpopC", "distScot", "area",
  "uplands", "lowlands", "mean_slope", "wet_1535", "wet_1536", "dwx_1536"
)

# ============================================================================
# Helper: coefficient extraction and plotting
# ============================================================================

extract_coefs_svyglm <- function(model, var_name) {
  coef_summary <- summary(model)$coefficients
  coef_val <- coef_summary[var_name, "Estimate"]
  se_val   <- coef_summary[var_name, "Std. Error"]
  z_crit   <- qnorm(0.95)  # 90% CI
  ci_lower <- coef_val - z_crit * se_val
  ci_upper <- coef_val + z_crit * se_val
  z_stat   <- coef_val / se_val
  p_val    <- 2 * pnorm(abs(z_stat), lower.tail = FALSE)
  data.frame(variable = var_name, coefficient = coef_val, se = se_val,
             ci_lower = ci_lower, ci_upper = ci_upper, p_value = p_val)
}

extract_coefs_coxph <- function(model, var_name) {
  coef_summary <- summary(model)$coefficients
  coef_val <- coef_summary[var_name, "coef"]
  se_val   <- coef_summary[var_name, "se(coef)"]
  z_crit   <- qnorm(0.95)
  ci_lower <- coef_val - z_crit * se_val
  ci_upper <- coef_val + z_crit * se_val
  z_stat   <- coef_val / se_val
  p_val    <- 2 * pnorm(abs(z_stat), lower.tail = FALSE)
  data.frame(variable = var_name, coefficient = coef_val, se = se_val,
             ci_lower = ci_lower, ci_upper = ci_upper, p_value = p_val)
}

make_coef_df_ipw <- function(model, vars, extract_fn) {
  coefs <- bind_rows(lapply(vars, function(v) extract_fn(model, v)))
  coefs$significant <- ifelse(is.na(coefs$p_value), FALSE, coefs$p_value < 0.10)
  coefs$order       <- match(coefs$variable, vars)
  coefs
}

make_ipw_plot <- function(coef_df, var_labels, x_label = "Coefficient (Log Odds)") {
  coef_df$variable_label <- var_labels[coef_df$variable]
  ggplot(coef_df, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper),
                  width = 0.2, color = "gray30", orientation = "y") +
    geom_point(aes(color = significant), size = 3) +
    scale_color_manual(
      values = c("FALSE" = "gray60", "TRUE" = "#0072B2"),
      labels = c("FALSE" = "Not Significant", "TRUE" = "p < 0.10")
    ) +
    labs(x = x_label, y = "", color = "Significance") +
    theme_minimal() +
    theme(
      axis.text.x  = element_text(size = 16),
      axis.text.y  = element_text(size = 16),
      axis.title.x = element_text(size = 16),
      legend.text  = element_text(size = 15),
      legend.title = element_text(size = 15),
      legend.position = "bottom"
    )
}

# ============================================================================
# Section 1: Main specification (sm/bg monastery land per arable km²)
# ============================================================================

main_treat  <- "lsm_arak"
main_covars <- c("lbg_arak", "lti_arak", "lal_arak", "lni_arak", "friary",
                 covar_rhs)
main_all    <- c(main_treat, main_covars)
main_formula_rhs <- paste(main_all, collapse = " + ")

# CBPS weights (treatment = small-house land per arable km²)
wt_main <- weightit(
  as.formula(paste(main_treat, "~", paste(main_covars, collapse = " + "))),
  data = rdf, method = "cbps", over = FALSE
)
weights_main <- wt_main$weights
design_main  <- svydesign(~1, weights = weights_main, data = rdf)

# Weighted logit models
wlm_primary_main <- svyglm(as.formula(paste("primary ~", main_formula_rhs)),
                            data = rdf, weights = weights_main,
                            design = design_main, family = quasibinomial())
wlm_muster_main  <- svyglm(as.formula(paste("muster ~",  main_formula_rhs)),
                            data = rdf, weights = weights_main,
                            design = design_main, family = quasibinomial())
wlm_seats_main   <- svyglm(as.formula(paste("seats ~",   main_formula_rhs)),
                            data = rdf, weights = weights_main,
                            design = design_main, family = quasibinomial())

# Weighted Cox survival model
wsurv_main <- coxph(as.formula(paste("Surv(primary_survival, primary) ~", main_formula_rhs)),
                    data = rdf, weights = weights_main, robust = TRUE)

print(summary(wlm_primary_main))
print(summary(wsurv_main))

stargazer(wlm_primary_main, wsurv_main,
  type = "latex",
  title = "Inverse-Probability-Weighted Logit and Survival Models",
  align = TRUE,
  table.placement = "H",
  column.labels = c("Logit", "Cox PH"),
  add.lines = list(
    c("Population", "Y", "Y"),
    c("Geographic Controls", "Y", "Y")
  ),
  covariate.labels = unlist(pretty_dict[main_treat]),
  column.sep.width = ".5pt",
  omit.stat = c("aic", "lr", "wald", "logrank"),
  omit = "Constant",
  out = "Output/Tables/IPW.tex"
)

# Coefficient plots — main specification
vars_to_plot_main <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
                       "friary", "lLStax_pc", "wet_1535", "wet_1536", "lpopC")

ggsave("Output/Images/Graphs/ipw_logit_primary_coefficients.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wlm_primary_main, vars_to_plot_main, extract_coefs_svyglm),
         pretty_dict),
       width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/ipw_logit_muster_coefficients.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wlm_muster_main, vars_to_plot_main, extract_coefs_svyglm),
         pretty_dict),
       width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/ipw_logit_seats_coefficients.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wlm_seats_main, vars_to_plot_main, extract_coefs_svyglm),
         pretty_dict),
       width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/ipw_cox_coefficients.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wsurv_main, vars_to_plot_main, extract_coefs_coxph),
         pretty_dict, x_label = "Coefficient (Log Hazard Ratio)"),
       width = 10, height = 6, dpi = 300)

# ============================================================================
# Section 2: OwnOther specification (on-site vs off-site land per arable km²)
# ============================================================================

oo_treat  <- "loth_arak"
oo_covars <- c("lti_arak", "lal_arak", "lni_arak", "friary", covar_rhs)
oo_all    <- c(oo_treat, oo_covars)
oo_formula_rhs <- paste(oo_all, collapse = " + ")

# CBPS weights (treatment = off-site land per arable km²)
wt_oo <- weightit(
  as.formula(paste(oo_treat, "~", paste(oo_covars, collapse = " + "))),
  data = rdf, method = "cbps", over = FALSE
)
weights_oo <- wt_oo$weights
design_oo  <- svydesign(~1, weights = weights_oo, data = rdf)

# Weighted logit models
wlm_primary_oo <- svyglm(as.formula(paste("primary ~", oo_formula_rhs)),
                          data = rdf, weights = weights_oo,
                          design = design_oo, family = quasibinomial())
wlm_muster_oo  <- svyglm(as.formula(paste("muster ~",  oo_formula_rhs)),
                          data = rdf, weights = weights_oo,
                          design = design_oo, family = quasibinomial())
wlm_seats_oo   <- svyglm(as.formula(paste("seats ~",   oo_formula_rhs)),
                          data = rdf, weights = weights_oo,
                          design = design_oo, family = quasibinomial())

# Weighted Cox survival model
wsurv_oo <- coxph(as.formula(paste("Surv(primary_survival, primary) ~", oo_formula_rhs)),
                  data = rdf, weights = weights_oo, robust = TRUE)

# Coefficient plots — ownOther specification
vars_to_plot_oo <- c("loth_arak", "lti_arak", "lal_arak", "lni_arak",
                     "friary", "lLStax_pc", "wet_1535", "wet_1536", "lpopC")

ggsave("Output/Images/Graphs/ipw_logit_primary_coefficients_ownOther.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wlm_primary_oo, vars_to_plot_oo, extract_coefs_svyglm),
         pretty_dict),
       width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/ipw_logit_muster_coefficients_ownOther.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wlm_muster_oo, vars_to_plot_oo, extract_coefs_svyglm),
         pretty_dict),
       width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/ipw_logit_seats_coefficients_ownOther.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wlm_seats_oo, vars_to_plot_oo, extract_coefs_svyglm),
         pretty_dict),
       width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/ipw_cox_coefficients_ownOther.png",
       plot = make_ipw_plot(
         make_coef_df_ipw(wsurv_oo, vars_to_plot_oo, extract_coefs_coxph),
         pretty_dict, x_label = "Coefficient (Log Hazard Ratio)"),
       width = 10, height = 6, dpi = 300)

cat("\nAIPW outputs created successfully!\n")
cat("Tables:\n")
cat("  Output/Tables/IPW.tex\n")
cat("Plots (main):\n")
cat("  Output/Images/Graphs/ipw_logit_primary_coefficients.png\n")
cat("  Output/Images/Graphs/ipw_logit_muster_coefficients.png\n")
cat("  Output/Images/Graphs/ipw_logit_seats_coefficients.png\n")
cat("  Output/Images/Graphs/ipw_cox_coefficients.png\n")
cat("Plots (ownOther):\n")
cat("  Output/Images/Graphs/ipw_logit_primary_coefficients_ownOther.png\n")
cat("  Output/Images/Graphs/ipw_logit_muster_coefficients_ownOther.png\n")
cat("  Output/Images/Graphs/ipw_logit_seats_coefficients_ownOther.png\n")
cat("  Output/Images/Graphs/ipw_cox_coefficients_ownOther.png\n")
