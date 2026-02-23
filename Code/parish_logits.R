pacman::p_load(
  sf, tidyverse, stargazer, dplyr,
  raster, spdep, sp, ggplot2, robust,
  lmtest, sandwich
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

monastic_vars <- c("llandOwned", "ltitheOutT", "lalmsInTot", "lnetInc", "friary")
# Removed X_COORD (VIF=24.6), news_day (VIF=35.8), and LS_pc_ch (680 NAs)
controls <- c("lLStax_pc", "wet_1535", "wet_1536", "dg_percy", "lpopC", "Y_COORD", "uplands", "lowlands", "area", "mean_slope")

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
  "ln(Land Owned)", "ln(Tithe)", "ln(Alms)", "ln(Net Income)", "Friary",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "Percy", "ln(Population)"
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
  c("llandOwned"),
  c("ltitheOutT", "lalmsInTot", "lnetInc", "friary"),
  c("lLStax_pc", "lpopC", "wet_1535", "wet_1536", "dg_percy"),
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
  "ln(Land Owned)", "ln(Tithe)", "ln(Alms)", "ln(Net Income)", "Friary",
  "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "Percy", "ln(Population)"
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
