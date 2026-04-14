# regression_plots.R — Coefficient plots for logit/Poisson regressions
# Sections:
#   1. Full-model single plots (sm/bg/ti/al/ni per arable km²)
#   2. Full-model single plots (land owned per arable km²)
#   3. Progressive plots (sm/bg per arable km²)
#   4. Progressive plots (own/oth per arable km²)
#   5. Progressive plots (land owned per arable km²)

pacman::p_load(
  sf, tidyverse, dplyr, ggplot2, broom
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Standardize and center continuous variables
for (v in c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
            "llo_arak", "lown_arak", "loth_arak",
            "lLStax_pc", "lpopC", "distScot", "area", "mean_slope",
            "wet_1535", "wet_1536")) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

# Use distScot (not Y_COORD) to match parish_logits.R table specification
controls <- c("lLStax_pc", "wet_1535", "wet_1536", "lpopC",
              "uplands", "lowlands", "area", "mean_slope", "distScot")

# ============================================================================
# Shared helpers
# ============================================================================

# Extract coefficients + 90% CI from a glm
extract_coefs <- function(model, var_name) {
  coef_summary <- summary(model)$coefficients
  coef_val <- coef_summary[var_name, "Estimate"]
  se_val   <- coef_summary[var_name, "Std. Error"]
  ci_lower <- coef_val - 1.645 * se_val
  ci_upper <- coef_val + 1.645 * se_val
  p_val    <- coef_summary[var_name, "Pr(>|z|)"]
  data.frame(variable = var_name, coefficient = coef_val, se = se_val,
             ci_lower = ci_lower, ci_upper = ci_upper, p_value = p_val)
}

# Same but with model number (for progressive plots)
extract_coefs_glm <- function(model, var_name, model_num) {
  if (!var_name %in% names(coef(model))) return(NULL)
  df <- extract_coefs(model, var_name)
  df$model <- paste("Model", model_num)
  df
}

# Single-spec coefficient plot
make_single_plot <- function(coef_df, var_labels, x_label = "Coefficient (Log Odds)") {
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

# Progressive coefficient plot (4 overlaid models)
make_prog_plot <- function(coef_df, var_labels, x_label = "Coefficient (Log Odds)") {
  coef_df$variable_label <- var_labels[coef_df$variable]
  coef_df$model <- factor(coef_df$model,
                          levels = c("Model 4", "Model 3", "Model 2", "Model 1"))
  prog_labels <- c(
    "Model 1" = "Model 1: Land",
    "Model 2" = "Model 2: + Monastic",
    "Model 3" = "Model 3: + Tax, Population, Weather",
    "Model 4" = "Model 4: + Geography"
  )
  ggplot(coef_df, aes(x = coefficient, y = reorder(variable_label, -order),
                      color = model, shape = model)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(size = 3, position = position_dodge(width = 0.5)) +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
                   height = 0, position = position_dodge(width = 0.5), alpha = 0.7) +
    scale_color_manual(values = c(
      "Model 1" = "#D55E00", "Model 2" = "#009E73",
      "Model 3" = "#0072B2", "Model 4" = "#CC79A7"
    ), labels = prog_labels, breaks = c("Model 1", "Model 2", "Model 3", "Model 4")) +
    scale_shape_manual(values = c(
      "Model 1" = 16, "Model 2" = 17, "Model 3" = 15, "Model 4" = 18
    ), labels = prog_labels, breaks = c("Model 1", "Model 2", "Model 3", "Model 4")) +
    labs(x = x_label, y = "", color = "Specification", shape = "Specification") +
    theme_minimal() +
    theme(
      axis.text.x  = element_text(size = 16),
      axis.text.y  = element_text(size = 16),
      axis.title.x = element_text(size = 16),
      legend.text  = element_text(size = 14),
      legend.title = element_text(size = 14),
      legend.position = "bottom"
    ) +
    guides(color = guide_legend(ncol = 2), shape = guide_legend(ncol = 2))
}

# Run a single full model and extract coefficients for a set of variables
run_full_model <- function(outcome, monastic_vars, controls, family, vars_to_plot) {
  formula <- paste(outcome, "~", paste(c(monastic_vars, controls), collapse = " + "))
  model   <- glm(formula, data = pdf, family = family)
  coefs   <- bind_rows(lapply(vars_to_plot, function(v) extract_coefs(model, v)))
  coefs$significant <- coefs$p_value < 0.10
  coefs$order       <- match(coefs$variable, vars_to_plot)
  coefs
}

# Run progressive models and extract all coefficients
run_progressive <- function(outcome, var_list_list, family, vars_to_plot, n_models = 4) {
  var_list <- c()
  models   <- list()
  for (i in seq_along(var_list_list)) {
    var_list  <- c(var_list, var_list_list[[i]])
    formula   <- paste(outcome, "~", paste(var_list, collapse = " + "))
    models[[i]] <- glm(formula, data = pdf, family = family)
  }
  coefs_all <- list()
  for (i in seq_len(n_models)) {
    for (v in vars_to_plot) {
      cd <- extract_coefs_glm(models[[i]], v, i)
      if (!is.null(cd)) coefs_all[[length(coefs_all) + 1]] <- cd
    }
  }
  coefs <- bind_rows(coefs_all)
  coefs$significant <- coefs$p_value < 0.10
  coefs$order       <- match(coefs$variable, vars_to_plot)
  coefs
}

outcomes_list <- list(
  muster  = list(label = "muster",  family = binomial(link = "logit"),
                 x_label = "Coefficient (Log Odds)"),
  primary = list(label = "primary", family = binomial(link = "logit"),
                 x_label = "Coefficient (Log Odds)"),
  seats   = list(label = "seats",   family = poisson(),
                 x_label = "Coefficient (Log Count)")
)

# ============================================================================
# Section 1: Full-model single plots — sm/bg/ti/al/ni per arable km²
# ============================================================================

monastic_vars_main <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak",
                        "lni_arak", "smHouse", "bigHouse", "friary")
vars_to_plot_main  <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak",
                        "lni_arak", "smHouse", "bigHouse", "friary",
                        "lLStax_pc", "wet_1535", "wet_1536")
var_labels_main <- c(
  "lsm_arak"  = "Small Monastery Land / Arable km\u00b2",
  "lbg_arak"  = "Large Monastery Land / Arable km\u00b2",
  "lti_arak"  = "Tithe / Arable km\u00b2",
  "lal_arak"  = "Alms / Arable km\u00b2",
  "lni_arak"  = "Net Income / Arable km\u00b2",
  "smHouse"   = "Small House Dummy",
  "bigHouse"  = "Large House Dummy",
  "friary"    = "Friary",
  "lLStax_pc" = "Lay Subsidy per Capita",
  "wet_1535"  = "Wet 1535",
  "wet_1536"  = "Wet 1536"
)

for (o in names(outcomes_list)) {
  coefs <- run_full_model(o, monastic_vars_main, controls,
                          outcomes_list[[o]]$family, vars_to_plot_main)
  ggsave(paste0("Output/Images/Graphs/", o, "_coefficients.png"),
         plot = make_single_plot(coefs, var_labels_main, outcomes_list[[o]]$x_label),
         width = 10, height = 6, dpi = 300)
}

# ============================================================================
# Section 2: Full-model single plots — land owned per arable km²
# ============================================================================

monastic_vars_lo   <- c("llo_arak", "lti_arak", "lal_arak", "lni_arak",
                        "smHouse", "bigHouse", "friary")
vars_to_plot_lo    <- c("llo_arak", "lti_arak", "lal_arak", "lni_arak",
                        "smHouse", "bigHouse", "friary", "lLStax_pc",
                        "wet_1535", "wet_1536")
var_labels_lo <- c(
  "llo_arak"  = "Land Owned / Arable km\u00b2",
  "lti_arak"  = "Tithe / Arable km\u00b2",
  "lal_arak"  = "Alms / Arable km\u00b2",
  "lni_arak"  = "Net Income / Arable km\u00b2",
  "smHouse"   = "Small House Dummy",
  "bigHouse"  = "Large House Dummy",
  "friary"    = "Friary",
  "lLStax_pc" = "Lay Subsidy per Capita",
  "wet_1535"  = "Wet 1535",
  "wet_1536"  = "Wet 1536"
)

for (o in names(outcomes_list)) {
  coefs <- run_full_model(o, monastic_vars_lo, controls,
                          outcomes_list[[o]]$family, vars_to_plot_lo)
  ggsave(paste0("Output/Images/Graphs/", o, "_coefficients_landOwned.png"),
         plot = make_single_plot(coefs, var_labels_lo, outcomes_list[[o]]$x_label),
         width = 10, height = 6, dpi = 300)
}

# ============================================================================
# Section 3: Progressive plots — sm/bg per arable km²
# ============================================================================

var_list_list_main <- list(
  c("lsm_arak", "lbg_arak"),
  c("lti_arak", "lal_arak", "lni_arak", "smHouse", "bigHouse", "friary"),
  c("lLStax_pc", "lpopC", "wet_1535", "wet_1536"),
  c("uplands", "lowlands", "area", "mean_slope", "distScot")
)
vars_to_plot_prog <- c(
  "lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
  "smHouse", "bigHouse", "friary", "lLStax_pc", "lpopC", "wet_1535", "wet_1536"
)
var_labels_prog <- c(
  "lsm_arak"  = "Small Monastery Land / Arable km\u00b2",
  "lbg_arak"  = "Large Monastery Land / Arable km\u00b2",
  "lti_arak"  = "Tithe / Arable km\u00b2",
  "lal_arak"  = "Alms / Arable km\u00b2",
  "lni_arak"  = "Net Income / Arable km\u00b2",
  "smHouse"   = "Small House Dummy",
  "bigHouse"  = "Large House Dummy",
  "friary"    = "Friary",
  "lLStax_pc" = "Lay Subsidy per Capita",
  "lpopC"     = "Population",
  "wet_1535"  = "Wet 1535",
  "wet_1536"  = "Wet 1536"
)

for (o in names(outcomes_list)) {
  coefs <- run_progressive(o, var_list_list_main, outcomes_list[[o]]$family,
                           vars_to_plot_prog)
  ggsave(paste0("Output/Images/Graphs/", o, "_progressive_coefficients.png"),
         plot = make_prog_plot(coefs, var_labels_prog, outcomes_list[[o]]$x_label),
         width = 12, height = 8, dpi = 300)
}

# ============================================================================
# Section 4: Progressive plots — own/oth per arable km²
# ============================================================================

var_list_list_oo <- list(
  c("lown_arak", "loth_arak"),
  c("lti_arak", "lal_arak", "lni_arak", "smHouse", "bigHouse", "friary"),
  c("lLStax_pc", "lpopC", "wet_1535", "wet_1536"),
  c("uplands", "lowlands", "area", "mean_slope", "distScot")
)
vars_to_plot_oo <- c(
  "lown_arak", "loth_arak", "lti_arak", "lal_arak", "lni_arak",
  "smHouse", "bigHouse", "friary", "lLStax_pc", "lpopC", "wet_1535", "wet_1536"
)
var_labels_oo <- c(
  "lown_arak"  = "Monastic Site Land / Arable km\u00b2",
  "loth_arak"  = "Offsite Land / Arable km\u00b2",
  "lti_arak"   = "Tithe / Arable km\u00b2",
  "lal_arak"   = "Alms / Arable km\u00b2",
  "lni_arak"   = "Net Income / Arable km\u00b2",
  "smHouse"    = "Small House Dummy",
  "bigHouse"   = "Large House Dummy",
  "friary"     = "Friary",
  "lLStax_pc"  = "Lay Subsidy per Capita",
  "lpopC"      = "Population",
  "wet_1535"   = "Wet 1535",
  "wet_1536"   = "Wet 1536"
)

for (o in names(outcomes_list)) {
  coefs <- run_progressive(o, var_list_list_oo, outcomes_list[[o]]$family,
                           vars_to_plot_oo)
  ggsave(paste0("Output/Images/Graphs/", o, "_progressive_coefficients_ownOther.png"),
         plot = make_prog_plot(coefs, var_labels_oo, outcomes_list[[o]]$x_label),
         width = 12, height = 8, dpi = 300)
}

# ============================================================================
# Section 5: Progressive plots — land owned per arable km²
# ============================================================================

var_list_list_lo <- list(
  c("llo_arak"),
  c("lti_arak", "lal_arak", "lni_arak", "smHouse", "bigHouse", "friary"),
  c("lLStax_pc", "lpopC", "wet_1535", "wet_1536"),
  c("uplands", "lowlands", "area", "mean_slope", "distScot")
)
vars_to_plot_lo_prog <- c(
  "llo_arak", "lti_arak", "lal_arak", "lni_arak",
  "smHouse", "bigHouse", "friary", "lLStax_pc", "lpopC", "wet_1535", "wet_1536"
)
var_labels_lo_prog <- c(
  "llo_arak"  = "Land Owned / Arable km\u00b2",
  "lti_arak"  = "Tithe / Arable km\u00b2",
  "lal_arak"  = "Alms / Arable km\u00b2",
  "lni_arak"  = "Net Income / Arable km\u00b2",
  "smHouse"   = "Small House Dummy",
  "bigHouse"  = "Large House Dummy",
  "friary"    = "Friary",
  "lLStax_pc" = "Lay Subsidy per Capita",
  "lpopC"     = "Population",
  "wet_1535"  = "Wet 1535",
  "wet_1536"  = "Wet 1536"
)

for (o in names(outcomes_list)) {
  coefs <- run_progressive(o, var_list_list_lo, outcomes_list[[o]]$family,
                           vars_to_plot_lo_prog)
  ggsave(paste0("Output/Images/Graphs/", o, "_progressive_coefficients_landOwned.png"),
         plot = make_prog_plot(coefs, var_labels_lo_prog, outcomes_list[[o]]$x_label),
         width = 12, height = 8, dpi = 300)
}

cat("\nRegression plots created successfully!\n")
cat("Section 1 (main, single): muster/primary/seats_coefficients.png\n")
cat("Section 2 (landOwned, single): *_coefficients_landOwned.png\n")
cat("Section 3 (progressive, sm/bg): *_progressive_coefficients.png\n")
cat("Section 4 (progressive, own/oth): *_progressive_coefficients_ownOther.png\n")
cat("Section 5 (progressive, landOwned): *_progressive_coefficients_landOwned.png\n")
