library(sf)
library(tidyverse)
library(WeightIt)
library(survey)
library(survival)

pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")
rdf <- data.frame(pdf)

# Create dummy for own land
rdf$dummyOwnLand <- ifelse(rdf$ownLand > 0, 1, 0)
cat("dummyOwnLand distribution:")
print(table(rdf$dummyOwnLand))

# Standardize variables
rdf$lotherLand <- log(rdf$otherLand + 1)
rdf$lotherLand <- scale(rdf$lotherLand, center = T, scale = T)[, 1]
rdf$ltitheOutT <- scale(rdf$ltitheOutT, center = T, scale = T)[, 1]
rdf$lalmsInTot <- scale(rdf$lalmsInTot, center = T, scale = T)[, 1]
rdf$lnetInc <- scale(rdf$lnetInc, center = T, scale = T)[, 1]
rdf$lLStax_pc <- scale(rdf$lLStax_pc, center = T, scale = T)[, 1]
rdf$lpopC <- scale(rdf$lpopC, center = T, scale = T)[, 1]
rdf$Y_COORD <- scale(rdf$Y_COORD, center = T, scale = T)[, 1]
rdf$area <- scale(rdf$area, center = T, scale = T)[, 1]
rdf$mean_slope <- scale(rdf$mean_slope, center = T, scale = T)[, 1]
rdf$wet_1535 <- scale(rdf$wet_1535, center = T, scale = T)[, 1]
rdf$wet_1536 <- scale(rdf$wet_1536, center = T, scale = T)[, 1]
rdf$dwx_1536 <- scale(rdf$dwx_1536, center = T, scale = T)[, 1]

cat("\nFitting CBPS weights...\n")
wm <- weightit(
    dummyOwnLand ~ lotherLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf, method = "cbps", over = FALSE
)
cat("Weights generated successfully!\n")
print(summary(wm$weights))

cat("\nFitting logit model...\n")
weighted_lm_primary <- svyglm(
    primary ~ dummyOwnLand + lotherLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf, weights = wm$weights,
    design = svydesign(~1, weights = wm$weights, data = rdf),
    family = quasibinomial()
)
cat("Logit model fitted!\n")
print(summary(weighted_lm_primary)$coefficients["dummyOwnLand", ])
