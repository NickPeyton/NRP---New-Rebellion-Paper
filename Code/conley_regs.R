pacman::p_load(
  sf, tidyverse, stargazer, spatialreg, spatstat, sp,
  raster, spdep, conleyreg, dplyr
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")



conleyModel <- primary ~ llandOwned +
  ltitheOutT + lalmsInTot + lnetInc + friary +
  lLStax_pc + LS_pc_ch + lpopC +
  X_COORD + Y_COORD + uplands + lowlands + area + mean_slope + mean_elev
conley20 <- conleyreg(formula = conleyModel, pdf, dist_cutoff = 20, model = "logit")
conley50 <- conleyreg(formula = conleyModel, pdf, dist_cutoff = 50, model = "logit")
conley100 <- conleyreg(formula = conleyModel, pdf, dist_cutoff = 100, model = "logit")
conley200 <- conleyreg(formula = conleyModel, pdf, dist_cutoff = 200, model = "logit")


hide_vars <- c("Constant", "X_COORD", "Y_COORD", "area", "uplands", "lowlands", "mean_slope")
covLabels <- c(
  "ln(Land Owned)", "ln(Tithe)", "ln(Alms)", "ln(Net Income)", "Friary",
  "ln(Lay Subsidy)", "Lay Subsidy Change", "ln(Population)"
)
stargazer(conley20,
  conley50,
  conley100,
  conley200,
  type = "latex",
  title = "Conley Standard Errors",
  label = "tab:conley",
  omit = hide_vars,
  add.lines = list(
    c("Population", "Y", "Y", "Y", "Y"),
    c("Geographic Controls", "Y", "Y", "Y", "Y")
  ),
  covariate.labels = covLabels,
  align = TRUE,
  table.placement = "H",
  column.labels = c("20km", "50km", "100km", "200km"),
  out = paste("Output/Tables/conley.tex", sep = "")
)
