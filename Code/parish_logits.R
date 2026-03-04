pacman::p_load(
  sf, tidyverse, stargazer, dplyr,
  raster, spdep, sp, ggplot2, robust,
  lmtest, sandwich
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")
pdf$lbigLand <- log(pdf$bigLand + 1)

# Standardize and center continuous variables
pdf$llandOwned <- scale(pdf$llandOwned, center = TRUE, scale = TRUE)[, 1]
pdf$lsmLand <- scale(pdf$lsmLand, center = TRUE, scale = TRUE)[, 1]
pdf$lbigLand <- scale(pdf$lbigLand, center = TRUE, scale = TRUE)[, 1]
pdf$ltitheOutT <- scale(pdf$ltitheOutT, center = TRUE, scale = TRUE)[, 1]
pdf$lalmsInTot <- scale(pdf$lalmsInTot, center = TRUE, scale = TRUE)[, 1]
pdf$lLStax_pc <- scale(pdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
pdf$distScot <- scale(pdf$distScot, center = TRUE, scale = TRUE)[, 1]
pdf$lpopC <- scale(pdf$lpopC, center = TRUE, scale = TRUE)[, 1]
pdf$Y_COORD <- scale(pdf$Y_COORD, center = TRUE, scale = TRUE)[, 1]
pdf$area <- scale(pdf$area, center = TRUE, scale = TRUE)[, 1]
pdf$mean_slope <- scale(pdf$mean_slope, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1535 <- scale(pdf$wet_1535, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1536 <- scale(pdf$wet_1536, center = TRUE, scale = TRUE)[, 1]

monastic_vars <- c("lsmLand", "lbigLand", "ltitheOutT", "lalmsInTot", "smHouse", "bigHouse", "friary")
# Removed X_COORD (VIF=24.6), news_day (VIF=35.8), and LS_pc_ch (680 NAs)
# dg_percy replaced by disg_gnt (all disgruntled gentry); distScot added
controls <- c("lLStax_pc", "wet_1535", "wet_1536", "disg_gnt", "lpopC", "Y_COORD", "uplands", "lowlands", "area", "mean_slope", "distScot")

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

hide_vars <- c("Constant", "Y_COORD", "uplands", "lowlands", "area", "mean_slope")
cov_labels <- c(
  "ln(Small Monastery Land)", "ln(Large Monastery Land)", "ln(Tithe)", "ln(Alms)",
  "Small House Dummy", "Large House Dummy", "Friary",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "Disgruntled Gentry", "ln(Population)", "Distance to Scotland"
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
  c("lsmLand", "lbigLand", "smHouse", "bigHouse"),
  c("ltitheOutT", "lalmsInTot", "friary"),
  c("lLStax_pc", "lpopC", "wet_1535", "wet_1536", "disg_gnt", "distScot"),
  c("Y_COORD", "uplands", "lowlands", "area", "mean_slope")
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
  "ln(Small Monastery Land)", "ln(Large Monastery Land)", "Small House Dummy", "Large House Dummy",
  "ln(Tithe)", "ln(Alms)", "Friary",
  "ln(Lay Subsidy)", "ln(Population)", "Wet 1535", "Wet 1536", "Disgruntled Gentry", "Distance to Scotland"
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
  "ln(Small Monastery Land)", "ln(Large Monastery Land)", "ln(Tithe)", "ln(Alms)",
  "Small House Dummy", "Large House Dummy", "Friary",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "Disgruntled Gentry", "ln(Population)", "Distance to Scotland"
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
dag_cov_labels <- c("ln(Land Owned)", "ln(Population)", "ln(Lay Subsidy Per Capita)")

dag_muster <- glm(muster ~ llandOwned + lpopC + lLStax_pc,
  data = pdf,
  family = binomial(link = "logit")
)

dag_primary <- glm(primary ~ llandOwned + lpopC + lLStax_pc,
  data = pdf,
  family = binomial(link = "logit")
)

dag_seat <- glm(seats ~ llandOwned + lpopC + lLStax_pc,
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
