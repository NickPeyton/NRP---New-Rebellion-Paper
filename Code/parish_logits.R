pacman::p_load(
  sf, tidyverse, stargazer, dplyr,
  raster, spdep, sp, ggplot2, robust,
  lmtest, sandwich, car
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Standardize and center continuous variables
pdf$llo_arak <- scale(pdf$llo_arak, center = TRUE, scale = TRUE)[, 1]
pdf$lsm_sk <- scale(pdf$lsm_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lbg_sk <- scale(pdf$lbg_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lti_sk <- scale(pdf$lti_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lal_sk <- scale(pdf$lal_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lLStax_pc <- scale(pdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
pdf$distScot <- scale(pdf$distScot, center = TRUE, scale = TRUE)[, 1]
pdf$lpopC <- scale(pdf$lpopC, center = TRUE, scale = TRUE)[, 1]
pdf$area <- scale(pdf$area, center = TRUE, scale = TRUE)[, 1]
pdf$mean_slope <- scale(pdf$mean_slope, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1535 <- scale(pdf$wet_1535, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1536 <- scale(pdf$wet_1536, center = TRUE, scale = TRUE)[, 1]

monastic_vars <- c("lsm_sk", "lbg_sk", "lti_sk", "lal_sk", "smHouse", "bigHouse", "friary")
# Removed X_COORD (VIF=24.6), news_day (VIF=35.8), and LS_pc_ch (680 NAs)
# Removed disg_gnt (causally downstream from rebellion)
# Removed Y_COORD (multicollinearity with distScot)
controls <- c("lLStax_pc", "wet_1535", "wet_1536", "lpopC", "uplands", "lowlands", "area", "mean_slope", "distScot")

muster_results_list <- list()
for (var in monastic_vars) {
  muster_formula <- paste("muster ~", var, "+", paste(controls, collapse = " + "))
  result <- glm(muster_formula,
    data = pdf,
    family = binomial(link = "logit")
  )
  muster_results_list[[var]] <- result
}

primary_results_list <- list()
for (var in monastic_vars) {
  primary_formula <- paste("primary ~", var, "+", paste(controls, collapse = " + "))
  result <- glm(primary_formula, data = pdf, family = binomial(link = "logit"))
  primary_results_list[[var]] <- result
  print(coeftest(result, vcov = vcovHC(result, type = "HC3")))
}

seat_results_list <- list()
for (var in monastic_vars) {
  seat_formula <- paste("seats ~", var, "+", paste(controls, collapse = " + "))
  result <- glm(seat_formula, data = pdf, family = "poisson")
  seat_results_list[[var]] <- result
}

hide_vars <- c("Constant", "uplands", "lowlands", "area", "mean_slope")
cov_labels <- c(
  "ln(Small Monastery Land / km\\textsuperscript{2})", "ln(Large Monastery Land / km\\textsuperscript{2})", "ln(Tithe / km\\textsuperscript{2})", "ln(Alms / km\\textsuperscript{2})",
  "Small House Dummy", "Large House Dummy", "Friary",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "ln(Population)", "Distance to Scotland"
)
stargazer(muster_results_list,
  type = "latex",
  title = "Muster Results: Monastic Variables",
  label = "tab:muster_monastic",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/muster_monastic.tex"
)

stargazer(primary_results_list,
  type = "latex",
  title = "Primary Results: Monastic Variables",
  label = "tab:primary_monastic",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/primary_monastic.tex"
)

stargazer(seat_results_list,
  type = "latex",
  title = "Seat Results: Monastic Variables",
  label = "tab:seat_monastic",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/seat_monastic.tex"
)

var_list_list <- list(
  c("lsm_sk", "lbg_sk", "smHouse", "bigHouse"),
  c("lti_sk", "lal_sk", "friary"),
  c("lLStax_pc", "lpopC", "wet_1535", "wet_1536", "distScot"),
  c("uplands", "lowlands", "area", "mean_slope")
)

var_list <- c()
muster_results_list <- list()
i <- 1
for (vars in var_list_list) {
  var_list <- c(var_list, vars)
  formula <- paste("muster ~", paste(var_list, collapse = " + "))
  result <- glm(formula, data = pdf, family = binomial(link = "logit"))
  muster_results_list[[i]] <- result
  i <- i + 1
}


var_list <- c()
primary_results_list <- list()
i <- 1
for (vars in var_list_list) {
  var_list <- c(var_list, vars)
  formula <- paste("primary ~", paste(var_list, collapse = " + "))
  result <- glm(formula, data = pdf, family = binomial(link = "logit"))
  primary_results_list[[i]] <- result
  i <- i + 1
}

var_list <- c()
seat_results_list <- list()
i <- 1
for (vars in var_list_list) {
  var_list <- c(var_list, vars)
  formula <- paste("seats ~", paste(var_list, collapse = " + "))
  result <- glm(formula, data = pdf, family = "poisson")
  seat_results_list[[i]] <- result
  i <- i + 1
}

cov_labels <- c(
  "ln(Small Monastery Land / km\\textsuperscript{2})", "ln(Large Monastery Land / km\\textsuperscript{2})", "Small House Dummy", "Large House Dummy",
  "ln(Tithe / km\\textsuperscript{2})", "ln(Alms / km\\textsuperscript{2})", "Friary",
  "ln(Lay Subsidy)", "ln(Population)", "Wet 1535", "Wet 1536", "Distance to Scotland"
)

stargazer(muster_results_list,
  type = "latex",
  title = "Muster Results: All Variables",
  label = "tab:muster_all",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geographic Controls", "N", "N", "N", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/muster_all.tex"
)

stargazer(primary_results_list,
  type = "latex",
  title = "Primary Results: All Variables",
  label = "tab:primary_all",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geographic Controls", "N", "N", "N", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/primary_all.tex"
)

stargazer(seat_results_list,
  type = "latex",
  title = "Seat Results: All Variables",
  label = "tab:seat_all",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geographic Controls", "N", "N", "N", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/seat_all.tex"
)

# Full model: all monastic variables simultaneously (robustness check)
all_monastic_formula <- paste(monastic_vars, collapse = " + ")
full_controls_formula <- paste(controls, collapse = " + ")

full_muster <- glm(
  paste("muster ~", all_monastic_formula, "+", full_controls_formula),
  data = pdf, family = binomial(link = "logit")
)
full_primary <- glm(
  paste("primary ~", all_monastic_formula, "+", full_controls_formula),
  data = pdf, family = binomial(link = "logit")
)
full_seat <- glm(
  paste("seats ~", all_monastic_formula, "+", full_controls_formula),
  data = pdf, family = "poisson"
)

full_cov_labels <- c(
  "ln(Small Monastery Land / km\\textsuperscript{2})", "ln(Large Monastery Land / km\\textsuperscript{2})", "ln(Tithe / km\\textsuperscript{2})", "ln(Alms / km\\textsuperscript{2})",
  "Small House Dummy", "Large House Dummy", "Friary",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "ln(Population)", "Distance to Scotland"
)

stargazer(full_muster, full_primary, full_seat,
  type = "latex",
  title = "Full Model Results: All Monastic Variables (Robustness)",
  label = "tab:full_monastic",
  omit = hide_vars,
  covariate.labels = full_cov_labels,
  column.labels = c("Muster", "Primary", "Seats"),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/full_monastic.tex"
)

# DAG Regs
dag_cov_labels <- c("ln(Land Owned / km\\textsuperscript{2})", "ln(Population)", "ln(Lay Subsidy Per Capita)")

dag_muster <- glm(muster ~ llo_arak + lpopC + lLStax_pc,
  data = pdf,
  family = binomial(link = "logit")
)

dag_primary <- glm(primary ~ llo_arak + lpopC + lLStax_pc,
  data = pdf,
  family = binomial(link = "logit")
)

dag_seat <- glm(seats ~ llo_arak + lpopC + lLStax_pc,
  data = pdf,
  family = "poisson"
)

stargazer(dag_muster, dag_primary, dag_seat,
  type = "latex",
  title = "DAG Results",
  label = "tab:dag",
  align = TRUE,
  column.sep.width = ".5pt",
  covariate.labels = dag_cov_labels,
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/dag.tex"
)
# ============================================================================
# VIF Analysis for Multicollinearity Detection
# ============================================================================

cat("\n========== VIF ANALYSIS ==========\n\n")

# Function to calculate and display VIF
vif_table <- function(model, model_name) {
  cat(paste0("Model: ", model_name, "\n"))
  cat("-----------------------------------\n")
  vif_values <- vif(model)
  vif_df <- data.frame(
    Variable = names(vif_values),
    VIF = round(vif_values, 3)
  )
  vif_df <- vif_df[order(-vif_df$VIF), ]
  print(vif_df)
  cat("\n")
  return(vif_df)
}

# Individual monastic vars (using the full control set)
cat("=== INDIVIDUAL MONASTIC VARIABLE MODELS ===\n\n")
for (i in seq_along(monastic_vars)) {
  vif_table(muster_results_list[[i]], 
            paste("Muster -", monastic_vars[[i]]))
}

# Progressive model - final fully specified model
cat("\n=== PROGRESSIVE MODELS (Final) ===\n\n")
vif_table(muster_results_list[[4]], "Muster - Full Progressive")
vif_table(primary_results_list[[4]], "Primary - Full Progressive")
vif_table(seat_results_list[[4]], "Seats - Full Progressive")

# Full model with all monastic variables
cat("\n=== FULL MODELS (All Monastic Vars) ===\n\n")
vif_table(full_muster, "Muster - Full Model")
vif_table(full_primary, "Primary - Full Model")
vif_table(full_seat, "Seats - Full Model")

# DAG models
cat("\n=== DAG MODELS ===\n\n")
vif_table(dag_muster, "Muster - DAG Model")
vif_table(dag_primary, "Primary - DAG Model")
vif_table(dag_seat, "Seats - DAG Model")

# Summary: Identify high-VIF predictors (VIF > 10 indicates problematic multicollinearity)
cat("\n========== VIF SUMMARY & RECOMMENDATIONS ==========\n")
cat("VIF Interpretation:\n")
cat("  VIF < 5:    Low multicollinearity (generally acceptable)\n")
cat("  VIF 5-10:   Moderate multicollinearity (use with caution)\n")
cat("  VIF > 10:   High multicollinearity (problematic)\n\n")

# Extract and summarize all VIF values
all_vif_results <- list()
all_vif_results[["muster_model1"]] <- vif(muster_results_list[[1]])
all_vif_results[["muster_full_prog"]] <- vif(muster_results_list[[4]])
all_vif_results[["full_muster"]] <- vif(full_muster)

high_vif <- sapply(all_vif_results, function(x) any(x > 10))
if (any(high_vif)) {
  cat("WARNING: High VIF (>10) detected in:\n")
  for (name in names(high_vif)[high_vif]) {
    high_vars <- names(all_vif_results[[name]][all_vif_results[[name]] > 10])
    cat(paste("  -", name, ":", paste(high_vars, collapse=", "), "\n"))
  }
} else {
  cat("✓ No high VIF (>10) detected. Multicollinearity appears manageable.\n")
}

cat("\n")