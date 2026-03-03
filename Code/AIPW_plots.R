pacman::p_load(
    sf, tidyverse, dplyr, ggplot2, broom,
    WeightIt, survey, survival
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Replace NAs in terrainTyp with 'Other'
pdf$terrainTyp <- ifelse(is.na(pdf$terrainTyp), "Other", pdf$terrainTyp)
pdf$uplands <- ifelse(pdf$terrainTyp == "Uplands", 1, 0)
pdf$lowlands <- ifelse(pdf$terrainTyp == "Lowlands", 1, 0)
pdf$otherlands <- ifelse(pdf$terrainTyp == "Other", 1, 0)

rdf <- data.frame(pdf)
day <- 40
rdf$day <- replace(rdf$day, rdf$day < 1, day)
rdf$day <- ifelse(is.na(rdf$day), day, rdf$day)
rdf$primary_day <- rdf$day * rdf$primary
rdf$primary_day <- replace(rdf$primary_day, rdf$primary_day < 1, day)
rdf$survival <- rdf$day - rdf$news_day
rdf$primary_survival <- rdf$primary_day - rdf$news_day
rdf$primary_survival <- ifelse(is.na(rdf$primary_survival), day, rdf$primary_survival)

# Convert seats to binary (1 if seats > 1, 0 otherwise)
rdf$seats <- ifelse(rdf$seats > 1, 1, rdf$seats)

# Create and standardize lbigLand
rdf$lbigLand <- log(rdf$bigLand + 1)

# Standardize and center continuous variables
rdf$lsmLand <- scale(rdf$lsmLand, center = TRUE, scale = TRUE)[, 1]
rdf$lbigLand <- scale(rdf$lbigLand, center = TRUE, scale = TRUE)[, 1]
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

# Calculate weights
weightitmodel <- weightit(
    lsmLand ~
        lbigLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    method = "cbps",
    over = FALSE
)
weights <- weightitmodel$weights

# Weighted logit model for primary
weighted_lm_primary <- svyglm(
    primary ~ lsmLand + lbigLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)

# Weighted logit model for muster
weighted_lm_muster <- svyglm(
    muster ~ lsmLand + lbigLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)

# Weighted logit model for seats
weighted_lm_seats <- svyglm(
    seats ~ lsmLand + lbigLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)

# Weighted survival model
weighted_survival <- coxph(
    Surv(primary_survival, primary) ~ lsmLand + lbigLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    weights = weights,
    robust = TRUE
)

# Variables to plot (excluding geographic controls to match the parish plots)
vars_to_plot <- c(
    "lsmLand", "lbigLand", "ltitheOutT", "lalmsInTot", "lnetInc", "friary",
    "lLStax_pc", "wet_1535", "wet_1536", "dg_percy", "lpopC"
)

# Variables for seats model (excluding Percy)
vars_to_plot_seats <- c(
    "lsmLand", "lbigLand", "ltitheOutT", "lalmsInTot", "lnetInc", "friary",
    "lLStax_pc", "wet_1535", "wet_1536", "lpopC"
)

# Variable labels
var_labels <- c(
    "lsmLand" = "Small Monastery Land",
    "lbigLand" = "Large Monastery Land",
    "ltitheOutT" = "Tithe",
    "lalmsInTot" = "Alms",
    "lnetInc" = "Net Income",
    "friary" = "Friary",
    "lLStax_pc" = "Lay Subsidy per Capita",
    "wet_1535" = "Wet 1535",
    "wet_1536" = "Wet 1536",
    "dg_percy" = "Percy",
    "lpopC" = "Population"
)

# Function to extract coefficients with 90% CI from svyglm
extract_coefs_svyglm <- function(model, var_name) {
    coef_summary <- summary(model)$coefficients
    coef_val <- coef_summary[var_name, "Estimate"]
    se_val <- coef_summary[var_name, "Std. Error"]
    ci_lower <- coef_val - 1.645 * se_val # 90% CI
    ci_upper <- coef_val + 1.645 * se_val # 90% CI
    p_val <- coef_summary[var_name, "Pr(>|t|)"]

    return(data.frame(
        variable = var_name,
        coefficient = coef_val,
        se = se_val,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        p_value = p_val
    ))
}

# Function to extract coefficients with 90% CI from coxph
extract_coefs_coxph <- function(model, var_name) {
    coef_summary <- summary(model)$coefficients
    coef_val <- coef_summary[var_name, "coef"]
    se_val <- coef_summary[var_name, "se(coef)"]
    ci_lower <- coef_val - 1.645 * se_val # 90% CI
    ci_upper <- coef_val + 1.645 * se_val # 90% CI
    p_val <- coef_summary[var_name, "Pr(>|z|)"]

    return(data.frame(
        variable = var_name,
        coefficient = coef_val,
        se = se_val,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        p_value = p_val
    ))
}

# Extract coefficients for weighted logit (primary)
logit_primary_coefs_list <- list()
for (var in vars_to_plot) {
    logit_primary_coefs_list[[var]] <- extract_coefs_svyglm(weighted_lm_primary, var)
}
logit_primary_coefs <- bind_rows(logit_primary_coefs_list)
logit_primary_coefs$variable_label <- var_labels[logit_primary_coefs$variable]
logit_primary_coefs$significant <- logit_primary_coefs$p_value < 0.10
logit_primary_coefs$order <- match(logit_primary_coefs$variable, vars_to_plot)

# Extract coefficients for weighted logit (muster)
logit_muster_coefs_list <- list()
for (var in vars_to_plot) {
    logit_muster_coefs_list[[var]] <- extract_coefs_svyglm(weighted_lm_muster, var)
}
logit_muster_coefs <- bind_rows(logit_muster_coefs_list)
logit_muster_coefs$variable_label <- var_labels[logit_muster_coefs$variable]
logit_muster_coefs$significant <- logit_muster_coefs$p_value < 0.10
logit_muster_coefs$order <- match(logit_muster_coefs$variable, vars_to_plot)

# Extract coefficients for weighted logit (seats)
logit_seats_coefs_list <- list()
for (var in vars_to_plot_seats) {
    logit_seats_coefs_list[[var]] <- extract_coefs_svyglm(weighted_lm_seats, var)
}
logit_seats_coefs <- bind_rows(logit_seats_coefs_list)
logit_seats_coefs$variable_label <- var_labels[logit_seats_coefs$variable]
logit_seats_coefs$significant <- logit_seats_coefs$p_value < 0.10
logit_seats_coefs$order <- match(logit_seats_coefs$variable, vars_to_plot_seats)

# Extract coefficients for weighted survival
survival_coefs_list <- list()
for (var in vars_to_plot) {
    survival_coefs_list[[var]] <- extract_coefs_coxph(weighted_survival, var)
}
survival_coefs <- bind_rows(survival_coefs_list)
survival_coefs$variable_label <- var_labels[survival_coefs$variable]
survival_coefs$significant <- survival_coefs$p_value < 0.10
survival_coefs$order <- match(survival_coefs$variable, vars_to_plot)

# Create coefficient plot for Weighted Logit model (primary)
logit_primary_plot <- ggplot(logit_primary_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), width = 0.2, color = "gray30", orientation = "y") +
    geom_point(aes(color = significant), size = 3) +
    scale_color_manual(
        values = c("FALSE" = "gray60", "TRUE" = "#0072B2"),
        labels = c("FALSE" = "Not Significant", "TRUE" = "p < 0.10")
    ) +
    labs(
        x = "Coefficient (Log Odds)",
        y = "",
        color = "Significance"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15),
        legend.position = "bottom"
    )

# Save logit primary plot
ggsave("Output/Images/Graphs/ipw_logit_primary_coefficients.png",
    plot = logit_primary_plot,
    width = 10, height = 6, dpi = 300
)

# Create coefficient plot for Weighted Logit model (muster)
logit_muster_plot <- ggplot(logit_muster_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), width = 0.2, color = "gray30", orientation = "y") +
    geom_point(aes(color = significant), size = 3) +
    scale_color_manual(
        values = c("FALSE" = "gray60", "TRUE" = "#0072B2"),
        labels = c("FALSE" = "Not Significant", "TRUE" = "p < 0.10")
    ) +
    labs(
        x = "Coefficient (Log Odds)",
        y = "",
        color = "Significance"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15),
        legend.position = "bottom"
    )

# Save logit muster plot
ggsave("Output/Images/Graphs/ipw_logit_muster_coefficients.png",
    plot = logit_muster_plot,
    width = 10, height = 6, dpi = 300
)

# Create coefficient plot for Weighted Logit model (seats)
logit_seats_plot <- ggplot(logit_seats_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), width = 0.2, color = "gray30", orientation = "y") +
    geom_point(aes(color = significant), size = 3) +
    scale_color_manual(
        values = c("FALSE" = "gray60", "TRUE" = "#0072B2"),
        labels = c("FALSE" = "Not Significant", "TRUE" = "p < 0.10")
    ) +
    labs(
        x = "Coefficient (Log Odds)",
        y = "",
        color = "Significance"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15),
        legend.position = "bottom"
    )

# Save logit seats plot
ggsave("Output/Images/Graphs/ipw_logit_seats_coefficients.png",
    plot = logit_seats_plot,
    width = 10, height = 6, dpi = 300
)

# Create coefficient plot for Weighted Survival model (hazard ratios)
survival_plot <- ggplot(survival_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), width = 0.2, color = "gray30", orientation = "y") +
    geom_point(aes(color = significant), size = 3) +
    scale_color_manual(
        values = c("FALSE" = "gray60", "TRUE" = "#0072B2"),
        labels = c("FALSE" = "Not Significant", "TRUE" = "p < 0.10")
    ) +
    labs(
        x = "Coefficient (Log Hazard Ratio)",
        y = "",
        color = "Significance"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 15),
        legend.position = "bottom"
    )

# Save survival plot
ggsave("Output/Images/Graphs/ipw_cox_coefficients.png",
    plot = survival_plot,
    width = 10, height = 6, dpi = 300
)

cat("\nAIPW regression plots created successfully!\n")
cat("- Output/Images/Graphs/ipw_logit_primary_coefficients.png\n")
cat("- Output/Images/Graphs/ipw_logit_muster_coefficients.png\n")
cat("- Output/Images/Graphs/ipw_logit_seats_coefficients.png\n")
cat("- Output/Images/Graphs/ipw_cox_coefficients.png\n")
