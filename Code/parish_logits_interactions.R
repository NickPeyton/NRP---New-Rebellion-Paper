pacman::p_load(
  sf, tidyverse, stargazer, dplyr,
  raster, spdep, sp, ggplot2, robust,
  lmtest, sandwich
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Standardize and center continuous variables (same controls as parish_logits.R)
pdf$lLStax_pc <- scale(pdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
pdf$lpopC     <- scale(pdf$lpopC,     center = TRUE, scale = TRUE)[, 1]
pdf$Y_COORD   <- scale(pdf$Y_COORD,   center = TRUE, scale = TRUE)[, 1]
pdf$area      <- scale(pdf$area,      center = TRUE, scale = TRUE)[, 1]
pdf$mean_slope<- scale(pdf$mean_slope,center = TRUE, scale = TRUE)[, 1]
pdf$wet_1535  <- scale(pdf$wet_1535,  center = TRUE, scale = TRUE)[, 1]
pdf$wet_1536  <- scale(pdf$wet_1536,  center = TRUE, scale = TRUE)[, 1]
pdf$distScot  <- scale(pdf$distScot,  center = TRUE, scale = TRUE)[, 1]

# Standardize distance-weighted variables
dw_vars_all <- c(
  "llo_dw",   "lsl_dw",   "lbl_dw",   "lti_dw",     # raw distance-weighted
  "llo_dwpc", "lsl_dwpc", "lbl_dwpc", "lti_dwpc",    # per capita distance-weighted
  "llo_dwsk", "lsl_dwsk", "lbl_dwsk", "lti_dwsk"     # per sq km distance-weighted
)
for (v in dw_vars_all) {
  pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1]
}

controls <- c("lLStax_pc", "wet_1535", "wet_1536", "disg_gnt", "lpopC",
              "Y_COORD", "uplands", "lowlands", "area", "mean_slope", "distScot")
hide_vars <- c("Constant", "Y_COORD", "uplands", "lowlands", "area", "mean_slope")

run_models <- function(monastic_vars, pdf, controls) {
  muster_list  <- list()
  primary_list <- list()
  seat_list    <- list()
  for (var in monastic_vars) {
    f_base <- paste(controls, collapse = " + ")
    muster_list[[var]]  <- glm(paste("muster ~",  var, "+", f_base), data = pdf, family = binomial(link = "logit"))
    primary_list[[var]] <- glm(paste("primary ~", var, "+", f_base), data = pdf, family = binomial(link = "logit"))
    seat_list[[var]]    <- glm(paste("seats ~",   var, "+", f_base), data = pdf, family = "poisson")
  }
  list(muster = muster_list, primary = primary_list, seat = seat_list)
}

# --- Set 1: Raw distance-weighted (land owned, small land, large land, tithe) ---
dw_raw   <- c("llo_dw", "lsl_dw", "lbl_dw", "lti_dw")
res_raw  <- run_models(dw_raw, pdf, controls)

raw_labels <- c("ln(Land × InvDist)", "ln(Small Land × InvDist)", "ln(Large Land × InvDist)", "ln(Tithe × InvDist)",
                "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "Disgruntled Gentry",
                "ln(Population)", "Distance to Scotland")

stargazer(res_raw$muster,  type = "latex", title = "Muster: Raw Distance-Weighted Monastic Variables",
  label = "tab:muster_dw_raw", omit = hide_vars, covariate.labels = raw_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/muster_dw_raw.tex")

stargazer(res_raw$primary, type = "latex", title = "Primary: Raw Distance-Weighted Monastic Variables",
  label = "tab:primary_dw_raw", omit = hide_vars, covariate.labels = raw_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/primary_dw_raw.tex")

stargazer(res_raw$seat, type = "latex", title = "Seats: Raw Distance-Weighted Monastic Variables",
  label = "tab:seat_dw_raw", omit = hide_vars, covariate.labels = raw_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/seat_dw_raw.tex")

# --- Set 2: Per capita distance-weighted ---
dw_pc    <- c("llo_dwpc", "lsl_dwpc", "lbl_dwpc", "lti_dwpc")
res_pc   <- run_models(dw_pc, pdf, controls)

pc_labels <- c("ln(Land × InvDist / Pop)", "ln(Small Land × InvDist / Pop)", "ln(Large Land × InvDist / Pop)", "ln(Tithe × InvDist / Pop)",
               "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "Disgruntled Gentry",
               "ln(Population)", "Distance to Scotland")

stargazer(res_pc$muster,  type = "latex", title = "Muster: Per Capita Distance-Weighted Monastic Variables",
  label = "tab:muster_dw_pc", omit = hide_vars, covariate.labels = pc_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/muster_dw_pc.tex")

stargazer(res_pc$primary, type = "latex", title = "Primary: Per Capita Distance-Weighted Monastic Variables",
  label = "tab:primary_dw_pc", omit = hide_vars, covariate.labels = pc_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/primary_dw_pc.tex")

stargazer(res_pc$seat, type = "latex", title = "Seats: Per Capita Distance-Weighted Monastic Variables",
  label = "tab:seat_dw_pc", omit = hide_vars, covariate.labels = pc_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/seat_dw_pc.tex")

# --- Set 3: Per sq km distance-weighted ---
dw_sk    <- c("llo_dwsk", "lsl_dwsk", "lbl_dwsk", "lti_dwsk")
res_sk   <- run_models(dw_sk, pdf, controls)

sk_labels <- c("ln(Land × InvDist / km²)", "ln(Small Land × InvDist / km²)", "ln(Large Land × InvDist / km²)", "ln(Tithe × InvDist / km²)",
               "ln(Lay Subsidy)", "Wet 1535", "Wet 1536", "Disgruntled Gentry",
               "ln(Population)", "Distance to Scotland")

stargazer(res_sk$muster,  type = "latex", title = "Muster: Per Sq Km Distance-Weighted Monastic Variables",
  label = "tab:muster_dw_sk", omit = hide_vars, covariate.labels = sk_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/muster_dw_sk.tex")

stargazer(res_sk$primary, type = "latex", title = "Primary: Per Sq Km Distance-Weighted Monastic Variables",
  label = "tab:primary_dw_sk", omit = hide_vars, covariate.labels = sk_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/primary_dw_sk.tex")

stargazer(res_sk$seat, type = "latex", title = "Seats: Per Sq Km Distance-Weighted Monastic Variables",
  label = "tab:seat_dw_sk", omit = hide_vars, covariate.labels = sk_labels,
  add.lines = list(c("Geographic Controls", "Y", "Y", "Y", "Y")),
  align = TRUE, column.sep.width = ".5pt", omit.stat = c("aic"), table.placement = "H",
  out = "Output/Tables/seat_dw_sk.tex")
