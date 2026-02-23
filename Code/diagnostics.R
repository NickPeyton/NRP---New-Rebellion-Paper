library(sf)
library(dplyr)
library(car)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf("Data/Processed/northParishFlows.shp")

# Check for missing values
cat("=== Missing Values ===\n")
cat("llandOwned:", sum(is.na(pdf$llandOwned)), "\n")
cat("primary:", sum(is.na(pdf$primary)), "\n")

# Summary statistics
cat("\n=== Summary of llandOwned ===\n")
print(summary(pdf$llandOwned))

# Check standard deviation
cat("\nSD of llandOwned:", sd(pdf$llandOwned, na.rm = TRUE), "\n")

# Correlations
cat("\n=== Correlations between monastic variables ===\n")
monastic_df <- data.frame(
    llandOwned = pdf$llandOwned,
    ltitheOutT = pdf$ltitheOutT,
    lalmsInTot = pdf$lalmsInTot,
    lnetInc = pdf$lnetInc
)
cor_matrix <- cor(monastic_df, use = "pairwise.complete.obs")
print(round(cor_matrix, 3))

# Check VIF
cat("\n=== Variance Inflation Factors ===\n")
model <- glm(
    primary ~ llandOwned + lLStax_pc + LS_pc_ch + wet_1535 + wet_1536 +
        dg_percy + lpopC + X_COORD + Y_COORD + uplands + lowlands + area +
        mean_slope + news_day,
    data = pdf, family = binomial
)
print(vif(model))

# Try without robust SE to see basic model
cat("\n=== Basic Model Results (without robust SE) ===\n")
print(summary(model))

# Check if scaling helps
cat("\n=== Model with standardized llandOwned ===\n")
pdf$llandOwned_std <- scale(pdf$llandOwned)[, 1]
model_std <- glm(
    primary ~ llandOwned_std + lLStax_pc + LS_pc_ch + wet_1535 + wet_1536 +
        dg_percy + lpopC + X_COORD + Y_COORD + uplands + lowlands + area +
        mean_slope + news_day,
    data = pdf, family = binomial
)
print(summary(model_std))
