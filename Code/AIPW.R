# Let's Match some Propensity Scores!
pacman::p_load(
    sf, tidyverse, stargazer, sp, dplyr,
    cem, MatchIt, WeightIt, marginaleffects, ipw,
    survey, optmatch, conflicted, cobalt, twang
)
conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")

setwd("C:/PhD/DissolutionProgramming/REB---Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")




# Replace NAs in terrainTyp with 'Other'
pdf$terrainTyp <- ifelse(is.na(pdf$terrainTyp), "Other", pdf$terrainTyp)
pdf$uplands <- ifelse(pdf$terrainTyp == "Uplands", 1, 0)
pdf$lowlands <- ifelse(pdf$terrainTyp == "Lowlands", 1, 0)
pdf$otherlands <- ifelse(pdf$terrainTyp == "Other", 1, 0)


rdf <- data.frame(pdf)
day <- 40
rdf$day <- replace(rdf$day, rdf$day < 1, day)
# Replace day NAs with 40 (indicating no event observed within the study period)
rdf$day <- ifelse(is.na(rdf$day), day, rdf$day)
rdf$primary_day <- rdf$day * rdf$primary
rdf$primary_day <- replace(rdf$primary_day, rdf$primary_day < 1, day)
rdf$survival <- rdf$day - rdf$news_day
rdf$primary_survival <- rdf$primary_day - rdf$news_day
# Replace primary_survival NAs with 40 (indicating no event observed within the study period)
rdf$primary_survival <- ifelse(is.na(rdf$primary_survival), day, rdf$primary_survival)

# Standardize and center continuous variables
rdf$llandOwned <- scale(rdf$llandOwned, center = TRUE, scale = TRUE)[, 1]
rdf$ltitheOutT <- scale(rdf$ltitheOutT, center = TRUE, scale = TRUE)[, 1]
rdf$lalmsInTot <- scale(rdf$lalmsInTot, center = TRUE, scale = TRUE)[, 1]
rdf$lnetInc <- scale(rdf$lnetInc, center = TRUE, scale = TRUE)[, 1]
rdf$lLStax_pc <- scale(rdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
rdf$lpopC <- scale(rdf$lpopC, center = TRUE, scale = TRUE)[, 1]
rdf$X_COORD <- scale(rdf$X_COORD, center = TRUE, scale = TRUE)[, 1]
rdf$Y_COORD <- scale(rdf$Y_COORD, center = TRUE, scale = TRUE)[, 1]
rdf$area <- scale(rdf$area, center = TRUE, scale = TRUE)[, 1]
rdf$mean_slope <- scale(rdf$mean_slope, center = TRUE, scale = TRUE)[, 1]
rdf$dry_1535 <- scale(rdf$dry_1535, center = TRUE, scale = TRUE)[, 1]
rdf$wet_1536 <- scale(rdf$wet_1536, center = TRUE, scale = TRUE)[, 1]
rdf$dwx_1536 <- scale(rdf$dwx_1536, center = TRUE, scale = TRUE)[, 1]

weightitmodel <- weightit(
    llandOwned ~
        ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + X_COORD + Y_COORD + area + uplands + lowlands + mean_slope + dry_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    method = "cbps",
    over = FALSE
)
weights <- weightitmodel$weights
weighted_lm <- svyglm(
    primary ~ llandOwned + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + X_COORD + Y_COORD + area + uplands + lowlands + mean_slope + dry_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)
print(summary(weighted_lm))

weighted_survival <- coxph(
    Surv(primary_survival, primary) ~ llandOwned + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + X_COORD + Y_COORD + area + uplands + lowlands + mean_slope + dry_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    weights = weights,
    robust = TRUE
)
print(summary(weighted_survival))

stargazer(weighted_lm, weighted_survival,
    type = "latex",
    title = "Inverse-Probability-Weighted Logit and Survival Models",
    align = TRUE,
    table.placement = "H",
    column.labels = c("Logit", "Cox PH"),
    add.lines = list(
        c("Population", "Y", "Y"),
        c("Geographic Controls", "Y", "Y")
    ),
    covariate.labels = c("ln(Land Owned)"),
    column.sep.width = ".5pt",
    omit.stat = c("aic", "lr", "wald", "logrank"),
    omit = "Constant",
    out = "Output/Tables/IPW.tex"
)
