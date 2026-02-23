pacman::p_load(
    sf, tidyverse, stargazer, spatialreg, spatstat, sp,
    raster, spdep, conleyreg, dplyr, survival, survminer,
    ggplot2
)

setwd("C:/PhD//DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

day <- 40
rdf <- data.frame(pdf)
rdf$day <- replace(rdf$day, rdf$day < 1, day)
# Replace day NAs with 40 (indicating no event observed within the study period)
rdf$day <- ifelse(is.na(rdf$day), day, rdf$day)
# Replace primary with 0 if NA, indicating no event observed within the study period
rdf$primary <- ifelse(is.na(rdf$primary), 0, rdf$primary)
rdf$primary_day <- rdf$day * rdf$primary
rdf$primary_day <- replace(rdf$primary_day, rdf$primary_day < 1, day)

rdf$survival <- rdf$day - rdf$news_day
rdf$primary_survival <- rdf$primary_day - rdf$news_day
rdf$llandOwned <- log(rdf$landOwned + 1)
# Replace primary_survival NAs with 40 (indicating no event observed within the study period)
rdf$primary_survival <- ifelse(is.na(rdf$primary_survival), day, rdf$primary_survival)

# Standardize and center continuous variables
# Removed LS_pc_ch (680 NAs) and X_COORD (VIF=24.6) to match parish_logits.R
rdf$llandOwned <- scale(rdf$llandOwned, center = TRUE, scale = TRUE)[, 1]
rdf$ltitheOutT <- scale(rdf$ltitheOutT, center = TRUE, scale = TRUE)[, 1]
rdf$lalmsInTot <- scale(rdf$lalmsInTot, center = TRUE, scale = TRUE)[, 1]
rdf$lnetInc <- scale(rdf$lnetInc, center = TRUE, scale = TRUE)[, 1]
rdf$lLStax_pc <- scale(rdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
rdf$lpopC <- scale(rdf$lpopC, center = TRUE, scale = TRUE)[, 1]
rdf$Y_COORD <- scale(rdf$Y_COORD, center = TRUE, scale = TRUE)[, 1]
rdf$area <- scale(rdf$area, center = TRUE, scale = TRUE)[, 1]
rdf$mean_slope <- scale(rdf$mean_slope, center = TRUE, scale = TRUE)[, 1]
rdf$wet_1535 <- scale(rdf$wet_1535, center = TRUE, scale = TRUE)[, 1]
rdf$wet_1536 <- scale(rdf$wet_1536, center = TRUE, scale = TRUE)[, 1]
rdf$dwx_1536 <- scale(rdf$dwx_1536, center = TRUE, scale = TRUE)[, 1]

cox1 <- coxph(
    Surv(primary_survival, primary) ~
        llandOwned +
        ltitheOutT +
        lalmsInTot +
        lnetInc +
        friary +
        wet_1535 + wet_1536 +
        dg_percy,
    data = rdf
)
cox2 <- coxph(
    Surv(primary_survival, primary) ~
        llandOwned +
        ltitheOutT +
        lalmsInTot +
        lnetInc +
        friary +
        wet_1535 + wet_1536 +
        dg_percy +
        lLStax_pc +
        lpopC,
    data = rdf
)
cox3 <- coxph(Surv(primary_survival, primary) ~
    llandOwned +
    ltitheOutT +
    lalmsInTot +
    lnetInc +
    friary +
    wet_1535 + wet_1536 +
    dg_percy +
    lLStax_pc +
    lpopC +
    Y_COORD +
    area +
    uplands +
    lowlands +
    mean_slope, data = rdf)
hideVars <- c("Constant", "Y_COORD", "area", "uplands", "lowlands", "mean_slope")
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
    covariate.labels = c(
        "ln(Land Owned)", "ln(Tithe)", "ln(Alms)", "ln(Monastic Net Income)", "Friary", "Wet 1535", "Wet 1536", "Percy", "ln(1535 Lay Subsidy Amount)", "ln(Population)"
    ),
    omit.stat = c("wald", "lr", "logrank"),
    out = "Output/Tables/survival.tex"
)
