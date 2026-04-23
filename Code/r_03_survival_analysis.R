pacman::p_load(
  sf, tidyverse, stargazer, spatialreg, spatstat, sp,
  raster, spdep, conleyreg, dplyr, survival, survminer,
  ggplot2, broom, jsonlite
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)

# Load pretty dictionary for labels
pretty_dict <- fromJSON("Code/pretty_dict.json")

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

day <- 40
rdf <- data.frame(pdf)
rdf$day <- replace(rdf$day, rdf$day < 1, day)
rdf$day <- ifelse(is.na(rdf$day), day, rdf$day)
rdf$primary <- ifelse(is.na(rdf$primary), 0, rdf$primary)
rdf$primary_day <- rdf$day * rdf$primary
rdf$primary_day <- replace(rdf$primary_day, rdf$primary_day < 1, day)

rdf$survival <- rdf$day - rdf$news_day
rdf$primary_survival <- rdf$primary_day - rdf$news_day
rdf$primary_survival <- ifelse(is.na(rdf$primary_survival), day, rdf$primary_survival)

# Standardize and center continuous variables
for (v in c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
            "mo_ci1_w", "mo_ci05_w", "mo_anyop_w",
            "lLStax_pc", "lpopC", "distScot", "area", "mean_slope",
            "wet_1535", "wet_1536", "dwx_1536")) {
  rdf[[v]] <- scale(rdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

# ============================================================================
# Section 1: Cox Proportional Hazards Models (stargazer tables)
# ============================================================================

cox1 <- coxph(
  Surv(primary_survival, primary) ~
    lsm_arak + lbg_arak + lti_arak + lal_arak + lni_arak +
    friary + mo_ci1_w + mo_ci05_w + mo_anyop_w +
    wet_1535 + wet_1536,
  data = rdf
)
cox2 <- coxph(
  Surv(primary_survival, primary) ~
    lsm_arak + lbg_arak + lti_arak + lal_arak + lni_arak +
    friary + mo_ci1_w + mo_ci05_w + mo_anyop_w +
    wet_1535 + wet_1536 + lLStax_pc + lpopC,
  data = rdf
)
cox3 <- coxph(
  Surv(primary_survival, primary) ~
    lsm_arak + lbg_arak + lti_arak + lal_arak + lni_arak +
    friary + mo_ci1_w + mo_ci05_w + mo_anyop_w +
    wet_1535 + wet_1536 + lLStax_pc + lpopC +
    distScot + area + uplands + lowlands + mean_slope,
  data = rdf
)

# ============================================================================
# Proportional Hazards Assumption — Schoenfeld Residual Tests
# Null hypothesis: log hazard ratio is constant over time (PH holds).
# A significant p-value for a term indicates a PH violation for that variable.
# Global test p-value is the omnibus test across all terms.
# If the global test is significant, consider: stratification, time-varying
# coefficients (tt() in coxph), or reporting with a caveat.
# ============================================================================

cat("\n========== PROPORTIONAL HAZARDS TESTS (Schoenfeld residuals) ==========\n\n")

zph1 <- cox.zph(cox1)
zph2 <- cox.zph(cox2)
zph3 <- cox.zph(cox3)

cat("--- Model 1 (monastic + weather) ---\n")
print(zph1)
cat("\n--- Model 2 (+ taxation/population) ---\n")
print(zph2)
cat("\n--- Model 3 (+ geographic controls) ---\n")
print(zph3)

# Save Schoenfeld residual plots for Model 3 (most complete specification)
png("Output/Images/Graphs/cox3_schoenfeld.png", width = 1200, height = 900, res = 120)
par(mfrow = c(3, 4))
plot(zph3)
dev.off()

cat("\nSchoenfeld plot saved: Output/Images/Graphs/cox3_schoenfeld.png\n\n")

hideVars <- c("Constant", "distScot", "area", "uplands", "lowlands", "mean_slope")
cox_vars_labels <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak", "friary", "mo_ci1_w", "mo_ci05_w", "mo_anyop_w", "wet_1535", "wet_1536", "lLStax_pc", "lpopC")
cox_cov_labels <- unlist(pretty_dict[cox_vars_labels])

stargazer(cox1, cox2, cox3,
  type = "latex",
  title = "Risk of Rebellion - Cox Proportional Hazards Model",
  omit = hideVars,
  align = TRUE,
  table.placement = "H",
  column.labels = c("Land", "Taxation and Population", "Geographic Controls"),
  add.lines = list(
    c("Population", "N", "Y", "Y"),
    c("Geographic Controls", "N", "N", "Y")
  ),
  covariate.labels = cox_cov_labels,
  omit.stat = c("wald", "lr", "logrank"),
  out = "Output/Tables/survival.tex"
)

# ============================================================================
# Section 2: Cox Model Coefficient Plots
# ============================================================================

# Variable sets per model
vars_cox1 <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
               "friary", "mo_ci1_w", "mo_ci05_w", "mo_anyop_w",
               "wet_1535", "wet_1536")
vars_cox2 <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
               "friary", "mo_ci1_w", "mo_ci05_w", "mo_anyop_w",
               "wet_1535", "wet_1536", "lLStax_pc", "lpopC")
vars_cox3 <- vars_cox2  # same variables plotted for model 3

# Function to extract coefficients with 90% CI from coxph
extract_coefs_coxph <- function(model, var_name) {
  coef_summary <- summary(model)$coefficients
  coef_val <- coef_summary[var_name, "coef"]
  se_val   <- coef_summary[var_name, "se(coef)"]
  ci_lower <- coef_val - 1.645 * se_val  # 90% CI
  ci_upper <- coef_val + 1.645 * se_val
  p_val    <- coef_summary[var_name, "Pr(>|z|)"]
  data.frame(variable = var_name, coefficient = coef_val, se = se_val,
             ci_lower = ci_lower, ci_upper = ci_upper, p_value = p_val)
}

make_coef_df <- function(model, vars) {
  coefs <- bind_rows(lapply(vars, function(v) extract_coefs_coxph(model, v)))
  coefs$variable_label <- unlist(pretty_dict[coefs$variable])
  coefs$significant    <- coefs$p_value < 0.10
  coefs$order          <- match(coefs$variable, vars)
  coefs
}

make_cox_plot <- function(coef_df, x_label = "Coefficient (Log Hazard Ratio)") {
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

cox1_coefs <- make_coef_df(cox1, vars_cox1)
cox2_coefs <- make_coef_df(cox2, vars_cox2)
cox3_coefs <- make_coef_df(cox3, vars_cox3)

ggsave("Output/Images/Graphs/cox1_coefficients.png",
       plot = make_cox_plot(cox1_coefs), width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/cox2_coefficients.png",
       plot = make_cox_plot(cox2_coefs), width = 10, height = 6, dpi = 300)
ggsave("Output/Images/Graphs/cox3_coefficients.png",
       plot = make_cox_plot(cox3_coefs), width = 10, height = 6, dpi = 300)

cat("\nSurvival analysis outputs created successfully!\n")
cat("Tables:\n")
cat("  Output/Tables/survival.tex\n")
cat("Plots:\n")
cat("  Output/Images/Graphs/cox1_coefficients.png\n")
cat("  Output/Images/Graphs/cox2_coefficients.png\n")
cat("  Output/Images/Graphs/cox3_coefficients.png\n")
