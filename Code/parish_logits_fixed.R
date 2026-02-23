pacman::p_load(
    sf, tidyverse, stargazer, dplyr,
    raster, spdep, sp, ggplot2, robust,
    lmtest, sandwich
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

monastic_vars <- c("llandOwned", "ltitheOutT", "lalmsInTot", "lnetInc", "friary")

# OPTION 1: Drop problematic variables with many missing values
# Use only complete cases for key variables
controls_reduced <- c("dg_percy", "lpopC", "Y_COORD", "uplands", "lowlands", "area", "mean_slope")

cat("=== OPTION 1: Reduced Controls (Drops LS_pc_ch, wet variables, news_day) ===\n\n")

muster_results_list <- list()
for (var in monastic_vars) {
    muster_formula <- paste("muster ~", var, "+", paste(controls_reduced, collapse = " + "))
    result <- glm(muster_formula,
        data = pdf,
        family = binomial(link = "logit")
    )
    muster_results_list[[var]] <- result
}

primary_results_list <- list()
for (var in monastic_vars) {
    primary_formula <- paste("primary ~", var, "+", paste(controls_reduced, collapse = " + "))
    result <- glm(primary_formula, data = pdf, family = binomial(link = "logit"))
    primary_results_list[[var]] <- result
    cat("\n", var, ":\n")
    print(coeftest(result, vcov = vcovHC(result, type = "HC3")))
}

seat_results_list <- list()
for (var in monastic_vars) {
    seat_formula <- paste("seats ~", var, "+", paste(controls_reduced, collapse = " + "))
    result <- glm(seat_formula, data = pdf, family = "poisson")
    seat_results_list[[var]] <- result
}

# OPTION 2: Use only parishes with complete data on ALL original variables
cat("\n\n=== OPTION 2: Complete Cases Only (Original Controls) ===\n\n")
controls_full <- c("lLStax_pc", "LS_pc_ch", "wet_1535", "wet_1536", "dg_percy", "lpopC", "Y_COORD", "uplands", "lowlands", "area", "mean_slope")

pdf_complete <- pdf %>%
    filter(!is.na(lLStax_pc) & !is.na(LS_pc_ch) & !is.na(wet_1535) & !is.na(wet_1536))

cat("Sample size with complete data:", nrow(pdf_complete), "\n\n")

primary_results_complete <- list()
for (var in monastic_vars) {
    primary_formula <- paste("primary ~", var, "+", paste(controls_full, collapse = " + "))
    result <- glm(primary_formula, data = pdf_complete, family = binomial(link = "logit"))
    primary_results_complete[[var]] <- result
    cat("\n", var, "(complete cases):\n")
    print(coeftest(result, vcov = vcovHC(result, type = "HC3")))
}

# OPTION 3: Use DAG-implied minimal specification
cat("\n\n=== OPTION 3: DAG Minimal Specification ===\n\n")

dag_primary <- glm(primary ~ llandOwned + lpopC + lLStax_pc,
    data = pdf,
    family = binomial(link = "logit")
)

cat("DAG specification (llandOwned):\n")
print(coeftest(dag_primary, vcov = vcovHC(dag_primary, type = "HC3")))
print(summary(dag_primary))

# OPTION 4: Use interaction with population
cat("\n\n=== OPTION 4: Interaction with Population ===\n\n")

interact_primary <- glm(primary ~ llandOwned * lpopC + dg_percy + Y_COORD + uplands + lowlands + area + mean_slope,
    data = pdf,
    family = binomial(link = "logit")
)

cat("Interaction model:\n")
print(coeftest(interact_primary, vcov = vcovHC(interact_primary, type = "HC3")))
