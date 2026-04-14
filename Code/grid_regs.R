pacman::p_load(
  sf, tidyverse, stargazer, dplyr,
  raster, spdep, sp, ggplot2
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)

g2df <- read_sf(dsn = "Data/Raw/grid2.shp")
g5df <- read_sf(dsn = "Data/Raw/grid5.shp")
g10df <- read_sf(dsn = "Data/Raw/grid10.shp")
g20df <- read_sf(dsn = "Data/Raw/grid20.shp")

df_list <- list(g2df, g5df, g10df, g20df)
muster_results_list <- list()
for (df in df_list) {
  df$lLStax_pc <- log(df$LStax_pc + 1)
  formula <- muster ~ llandOwned +
    lti_arak + lal_arak + lni_arak + friary +
    lLStax_pc + LS_pc_ch + lpopC +
    X_COORD + Y_COORD + mean_slope + mean_elev
  result <- glm(formula, data = df, family = poisson)
  muster_results_list[[length(muster_results_list) + 1]] <- result
}
primary_results_list <- list()
for (df in df_list) {
  df$lLStax_pc <- log(df$LStax_pc + 1)
  formula <- primary ~ llandOwned +
    lti_arak + lal_arak + lni_arak + friary +
    lLStax_pc + LS_pc_ch + lpopC +
    X_COORD + Y_COORD + mean_slope + mean_elev
  result <- glm(formula, data = df, family = poisson)
  primary_results_list[[length(primary_results_list) + 1]] <- result
}
seats_results_list <- list()
for (df in df_list) {
  df$lLStax_pc <- log(df$LStax_pc + 1)
  formula <- seats ~ llandOwned +
    lti_arak + lal_arak + lni_arak + friary +
    lLStax_pc + LS_pc_ch + lpopC +
    X_COORD + Y_COORD + mean_slope + mean_elev
  result <- glm(formula, data = df, family = poisson)
  seats_results_list[[length(seats_results_list) + 1]] <- result
}

hide_vars <- c("Constant", "X_COORD", "Y_COORD", "mean_slope")
cov_labels <- c(
  "ln(Land Owned)", "ln(Tithe / Arable km\\textsuperscript{2})",
  "ln(Alms / Arable km\\textsuperscript{2})",
  "ln(Net Income / Arable km\\textsuperscript{2})", "Friary",
  "ln(Lay Subsidy)", "Lay Subsidy Change", "ln(Population)"
)

stargazer(muster_results_list,
  type = "latex",
  title = "Grid Regression Results",
  label = "tab:grid",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geography", "Y", "Y", "Y", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/grid_muster.tex"
)

stargazer(primary_results_list,
  type = "latex",
  title = "Grid Regression Results",
  label = "tab:grid",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geography", "Y", "Y", "Y", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/grid_primary.tex"
)

stargazer(seats_results_list,
  type = "latex",
  title = "Grid Regression Results",
  label = "tab:grid",
  omit = hide_vars,
  covariate.labels = cov_labels,
  add.lines = list(c("Geography", "Y", "Y", "Y", "Y")),
  align = TRUE,
  column.sep.width = ".5pt",
  omit.stat = c("aic"),
  table.placement = "H",
  out = "Output/Tables/grid_seats.tex"
)
