pacman::p_load(
  sf, tidyverse, stargazer, dplyr,
  raster, spdep, sp, ggplot2, robust,
  lmtest, sandwich, car, jsonlite
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)

# Load pretty dictionary for labels
pretty_dict <- fromJSON("Code/pretty_dict.json")

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# ============================================================================
# Section 1: Main Parish-Level Logit Models
# Monastic variables per arable km²
# ============================================================================

# Standardize and center continuous variables
for (v in c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
            "llo_arak", "lLStax_pc", "distScot", "lpopC",
            "area", "mean_slope", "wet_1535", "wet_1536")) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

monastic_vars <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak",
                   "smHouse", "bigHouse", "friary")
# Removed X_COORD (VIF=24.6), news_day (VIF=35.8), and LS_pc_ch (680 NAs)
# Removed disg_gnt (causally downstream from rebellion)
# Removed Y_COORD (multicollinearity with distScot)
controls <- c("lLStax_pc", "wet_1535", "wet_1536", "lpopC",
              "uplands", "lowlands", "area", "mean_slope", "distScot")

muster_results_list <- list()
for (var in monastic_vars) {
  muster_formula <- paste("muster ~", var, "+", paste(controls, collapse = " + "))
  result <- glm(muster_formula, data = pdf, family = binomial(link = "logit"))
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
main_vars_labels <- c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak", "smHouse", "bigHouse", "friary", "lLStax_pc", "wet_1535", "wet_1536", "lpopC", "distScot")
cov_labels <- unlist(pretty_dict[main_vars_labels])
stargazer(muster_results_list,
  type = "latex",
  title = "Muster Results: Monastic Variables",
  label = "tab:muster_monastic",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y")),
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
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y")),
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
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/seat_monastic.tex"
)

# --- Progressive model building ---
var_list_list <- list(
  c("lsm_arak", "lbg_arak", "smHouse", "bigHouse"),
  c("lti_arak", "lal_arak", "lni_arak", "friary"),
  c("lLStax_pc", "lpopC", "wet_1535", "wet_1536"),
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

cov_labels_prog <- unlist(pretty_dict[c("lsm_arak", "lbg_arak", "smHouse", "bigHouse", "lti_arak", "lal_arak", "lni_arak", "friary", "lLStax_pc", "lpopC", "wet_1535", "wet_1536")])

stargazer(muster_results_list,
  type = "latex",
  title = "Muster Results: All Variables",
  label = "tab:muster_all",
  omit = hide_vars,
  covariate.labels = cov_labels_prog,
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
  covariate.labels = cov_labels_prog,
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
  covariate.labels = cov_labels_prog,
  add.lines = list(c("Geographic Controls", "N", "N", "N", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/seat_all.tex"
)

# --- Full model: all monastic variables simultaneously (robustness check) ---
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

full_cov_labels <- unlist(pretty_dict[c("lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lni_arak", "smHouse", "bigHouse", "friary", "lLStax_pc", "wet_1535", "wet_1536", "lpopC", "distScot")])

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

# --- DAG Regressions ---
dag_vars_labels <- c("llo_arak", "lpopC", "lLStax_pc")
dag_cov_labels <- unlist(pretty_dict[dag_vars_labels])

dag_muster  <- glm(muster  ~ llo_arak + lpopC + lLStax_pc, data = pdf,
                   family = binomial(link = "logit"))
dag_primary <- glm(primary ~ llo_arak + lpopC + lLStax_pc, data = pdf,
                   family = binomial(link = "logit"))
dag_seat    <- glm(seats   ~ llo_arak + lpopC + lLStax_pc, data = pdf,
                   family = "poisson")

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

vif_table <- function(model, model_name) {
  cat(paste0("Model: ", model_name, "\n"))
  cat("-----------------------------------\n")
  vif_values <- vif(model)
  vif_df <- data.frame(Variable = names(vif_values), VIF = round(vif_values, 3))
  vif_df <- vif_df[order(-vif_df$VIF), ]
  print(vif_df)
  cat("\n")
  return(vif_df)
}

cat("=== INDIVIDUAL MONASTIC VARIABLE MODELS ===\n\n")
for (i in seq_along(monastic_vars)) {
  vif_table(muster_results_list[[i]], paste("Muster -", monastic_vars[[i]]))
}

cat("\n=== PROGRESSIVE MODELS (Final) ===\n\n")
vif_table(muster_results_list[[4]],  "Muster - Full Progressive")
vif_table(primary_results_list[[4]], "Primary - Full Progressive")
vif_table(seat_results_list[[4]],    "Seats - Full Progressive")

cat("\n=== FULL MODELS (All Monastic Vars) ===\n\n")
vif_table(full_muster,  "Muster - Full Model")
vif_table(full_primary, "Primary - Full Model")
vif_table(full_seat,    "Seats - Full Model")

cat("\n=== DAG MODELS ===\n\n")
vif_table(dag_muster,  "Muster - DAG Model")
vif_table(dag_primary, "Primary - DAG Model")
vif_table(dag_seat,    "Seats - DAG Model")

cat("\n========== VIF SUMMARY & RECOMMENDATIONS ==========\n")
cat("VIF Interpretation:\n")
cat("  VIF < 5:    Low multicollinearity (generally acceptable)\n")
cat("  VIF 5-10:   Moderate multicollinearity (use with caution)\n")
cat("  VIF > 10:   High multicollinearity (problematic)\n\n")

all_vif_results <- list()
all_vif_results[["muster_model1"]]    <- vif(muster_results_list[[1]])
all_vif_results[["muster_full_prog"]] <- vif(muster_results_list[[4]])
all_vif_results[["full_muster"]]      <- vif(full_muster)

high_vif <- sapply(all_vif_results, function(x) any(x > 10))
if (any(high_vif)) {
  cat("WARNING: High VIF (>10) detected in:\n")
  for (name in names(high_vif)[high_vif]) {
    high_vars <- names(all_vif_results[[name]][all_vif_results[[name]] > 10])
    cat(paste("  -", name, ":", paste(high_vars, collapse = ", "), "\n"))
  }
} else {
  cat("No high VIF (>10) detected. Multicollinearity appears manageable.\n")
}
cat("\n")

# ============================================================================
# Section 2: Distance-Weighted Interaction Models (Robustness)
# Spatially-weighted monastic variables (raw, per-capita, per-sq-km)
# ============================================================================

# Standardize distance-weighted variables
dw_vars_all <- c(
  "llo_dw",   "lsl_dw",   "lbl_dw",   "lti_dw",
  "llo_dwpc", "lsl_dwpc", "lbl_dwpc", "lti_dwpc",
  "llo_dwsk", "lsl_dwsk", "lbl_dwsk", "lti_dwsk"
)
for (v in dw_vars_all) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}
pdf$Y_COORD <- scale(pdf$Y_COORD, center = TRUE, scale = TRUE)[, 1]

controls_dw <- c(
  "lLStax_pc", "wet_1535", "wet_1536", "lpopC",
  "Y_COORD", "uplands", "lowlands", "area", "mean_slope", "distScot"
)
hide_vars_dw <- c("Constant", "Y_COORD", "uplands", "lowlands", "area", "mean_slope")

run_models <- function(monastic_vars, pdf, controls) {
  muster_list  <- list()
  primary_list <- list()
  seat_list    <- list()
  for (var in monastic_vars) {
    f_base <- paste(controls, collapse = " + ")
    muster_list[[var]]  <- glm(paste("muster ~",  var, "+", f_base), data = pdf,
                                family = binomial(link = "logit"))
    primary_list[[var]] <- glm(paste("primary ~", var, "+", f_base), data = pdf,
                                family = binomial(link = "logit"))
    seat_list[[var]]    <- glm(paste("seats ~",   var, "+", f_base), data = pdf,
                                family = "poisson")
  }
  list(muster = muster_list, primary = primary_list, seat = seat_list)
}

# --- Set 1: Raw distance-weighted ---
dw_raw <- c("llo_dw", "lsl_dw", "lbl_dw", "lti_dw")
res_raw <- run_models(dw_raw, pdf, controls_dw)

raw_labels <- unlist(pretty_dict[c(dw_raw, "lLStax_pc", "wet_1535", "wet_1536", "lpopC", "distScot")])

stargazer(res_raw$muster,
  type = "latex", title = "Muster: Raw Distance-Weighted Monastic Variables",
  label = "tab:muster_dw_raw", omit = hide_vars_dw, covariate.labels = raw_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/muster_dw_raw.tex"
)
stargazer(res_raw$primary,
  type = "latex", title = "Primary: Raw Distance-Weighted Monastic Variables",
  label = "tab:primary_dw_raw", omit = hide_vars_dw, covariate.labels = raw_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/primary_dw_raw.tex"
)
stargazer(res_raw$seat,
  type = "latex", title = "Seats: Raw Distance-Weighted Monastic Variables",
  label = "tab:seat_dw_raw", omit = hide_vars_dw, covariate.labels = raw_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/seat_dw_raw.tex"
)

# --- Set 2: Per capita distance-weighted ---
dw_pc <- c("llo_dwpc", "lsl_dwpc", "lbl_dwpc", "lti_dwpc")
res_pc <- run_models(dw_pc, pdf, controls_dw)

pc_labels <- unlist(pretty_dict[c(dw_pc, "lLStax_pc", "wet_1535", "wet_1536", "lpopC", "distScot")])

stargazer(res_pc$muster,
  type = "latex", title = "Muster: Per Capita Distance-Weighted Monastic Variables",
  label = "tab:muster_dw_pc", omit = hide_vars_dw, covariate.labels = pc_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/muster_dw_pc.tex"
)
stargazer(res_pc$primary,
  type = "latex", title = "Primary: Per Capita Distance-Weighted Monastic Variables",
  label = "tab:primary_dw_pc", omit = hide_vars_dw, covariate.labels = pc_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/primary_dw_pc.tex"
)
stargazer(res_pc$seat,
  type = "latex", title = "Seats: Per Capita Distance-Weighted Monastic Variables",
  label = "tab:seat_dw_pc", omit = hide_vars_dw, covariate.labels = pc_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/seat_dw_pc.tex"
)

# --- Set 3: Per sq km distance-weighted ---
dw_sk <- c("llo_dwsk", "lsl_dwsk", "lbl_dwsk", "lti_dwsk")
res_sk <- run_models(dw_sk, pdf, controls_dw)

sk_labels <- unlist(pretty_dict[c(dw_sk, "lLStax_pc", "wet_1535", "wet_1536", "lpopC", "distScot")])

stargazer(res_sk$muster,
  type = "latex", title = "Muster: Per Sq Km Distance-Weighted Monastic Variables",
  label = "tab:muster_dw_sk", omit = hide_vars_dw, covariate.labels = sk_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/muster_dw_sk.tex"
)
stargazer(res_sk$primary,
  type = "latex", title = "Primary: Per Sq Km Distance-Weighted Monastic Variables",
  label = "tab:primary_dw_sk", omit = hide_vars_dw, covariate.labels = sk_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/primary_dw_sk.tex"
)
stargazer(res_sk$seat,
  type = "latex", title = "Seats: Per Sq Km Distance-Weighted Monastic Variables",
  label = "tab:seat_dw_sk", omit = hide_vars_dw, covariate.labels = sk_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/seat_dw_sk.tex"
)
