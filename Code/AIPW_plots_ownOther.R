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

# Create and standardize lotherLand
rdf$lotherLand <- log(rdf$otherLand + 1)

# Standardize and center continuous variables
rdf$lotherLand <- scale(rdf$lotherLand, center = TRUE, scale = TRUE)[, 1]
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
    lotherLand ~
        ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536,
    data = rdf,
    method = "cbps",
    over = FALSE
)
weights <- weightitmodel$weights

# Weighted logit model for primary
weighted_lm_primary <- svyglm(
    primary ~ lotherLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)

# Weighted logit model for muster
weighted_lm_muster <- svyglm(
    muster ~ lotherLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)

# Weighted logit model for seats
weighted_lm_seats <- svyglm(
    seats ~ lotherLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)

# Weighted survival model
weighted_survival <- coxph(
    Surv(primary_survival, primary) ~ lotherLand + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + Y_COORD + area + uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536,
    data = rdf,
    weights = weights,
    robust = TRUE
)

# Variables to plot (excluding geographic controls to match the parish plots)
vars_to_plot <- c(
    "lotherLand", "ltitheOutT", "lalmsInTot", "lnetInc", "friary",
    "lLStax_pc", "wet_1535", "wet_1536", "lpopC"
)

# Variable labels
var_labels <- c(
    "lotherLand" = "Offsite Land",
    "ltitheOutT" = "Tithe",
    "lalmsInTot" = "Alms",
    "lnetInc" = "Net Income",
    "friary" = "Friary",
    "lLStax_pc" = "Lay Subsidy per Capita",
    "wet_1535" = "Wet 1535",
    "wet_1536" = "Wet 1536",
    "lpopC" = "Population"
)

# Function to extract coefficients with 90% CI from svyglm
extract_coefs_svyglm <- function(model, var_name) {
    coef_summary <- summary(model)$coefficients
    coef_val <- coef_summary[var_name, "Estimate"]
    se_val <- coef_summary[var_name, "Std. Error"]

    # For svyglm, we should use the model's actual p-value but recalculate CI consistently
    # Check if p-value exists in summary
    p_val_from_summary <- coef_summary[var_name, "Pr(>|t|)"]

    # Calculate t-statistic
    t_stat <- coef_val / se_val

    # Use a large df (svyglm may not have residual df in traditional sense)
    # Instead, calculate p-value directly from t-stat and use qt for CI
    p_val <- 2 * pt(abs(t_stat), df = Inf, lower.tail = FALSE) # Use normal distribution (df=Inf)
    t_crit <- qnorm(0.95) # 90% CI uses z-critical value
    ci_lower <- coef_val - t_crit * se_val
    ci_upper <- coef_val + t_crit * se_val

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
    # Use z critical value for 90% CI
    z_crit <- qnorm(0.95)
    ci_lower <- coef_val - z_crit * se_val
    ci_upper <- coef_val + z_crit * se_val
    # Calculate p-value from z-statistic to ensure consistency with CI
    z_stat <- coef_val / se_val
    p_val <- 2 * pnorm(abs(z_stat), lower.tail = FALSE)

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
logit_primary_coefs$significant <- ifelse(is.na(logit_primary_coefs$p_value), FALSE, logit_primary_coefs$p_value < 0.10)
logit_primary_coefs$order <- match(logit_primary_coefs$variable, vars_to_plot)

# Extract coefficients for weighted logit (muster)
logit_muster_coefs_list <- list()
for (var in vars_to_plot) {
    logit_muster_coefs_list[[var]] <- extract_coefs_svyglm(weighted_lm_muster, var)
}
logit_muster_coefs <- bind_rows(logit_muster_coefs_list)
logit_muster_coefs$variable_label <- var_labels[logit_muster_coefs$variable]
logit_muster_coefs$significant <- ifelse(is.na(logit_muster_coefs$p_value), FALSE, logit_muster_coefs$p_value < 0.10)
logit_muster_coefs$order <- match(logit_muster_coefs$variable, vars_to_plot)

# Extract coefficients for weighted logit (seats)
logit_seats_coefs_list <- list()
for (var in vars_to_plot) {
    logit_seats_coefs_list[[var]] <- extract_coefs_svyglm(weighted_lm_seats, var)
}
logit_seats_coefs <- bind_rows(logit_seats_coefs_list)
logit_seats_coefs$variable_label <- var_labels[logit_seats_coefs$variable]
logit_seats_coefs$significant <- ifelse(is.na(logit_seats_coefs$p_value), FALSE, logit_seats_coefs$p_value < 0.10)
logit_seats_coefs$order <- match(logit_seats_coefs$variable, vars_to_plot)

# Extract coefficients for weighted survival
survival_coefs_list <- list()
for (var in vars_to_plot) {
    survival_coefs_list[[var]] <- extract_coefs_coxph(weighted_survival, var)
}
survival_coefs <- bind_rows(survival_coefs_list)

survival_coefs$variable_label <- var_labels[survival_coefs$variable]
survival_coefs$significant <- ifelse(is.na(survival_coefs$p_value), FALSE, survival_coefs$p_value < 0.10)
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
ggsave("Output/Images/Graphs/ipw_logit_primary_coefficients_ownOther.png",
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
ggsave("Output/Images/Graphs/ipw_logit_muster_coefficients_ownOther.png",
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
ggsave("Output/Images/Graphs/ipw_logit_seats_coefficients_ownOther.png",
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
ggsave("Output/Images/Graphs/ipw_cox_coefficients_ownOther.png",
    plot = survival_plot,
    width = 10, height = 6, dpi = 300
)

# Display the survival plot
print(survival_plot)

cat("\nAIPW regression plots (ownOther) created successfully!\n")
cat("- Output/Images/Graphs/ipw_logit_primary_coefficients_ownOther.png\n")
cat("- Output/Images/Graphs/ipw_logit_muster_coefficients_ownOther.png\n")
cat("- Output/Images/Graphs/ipw_logit_seats_coefficients_ownOther.png\n")
cat("- Output/Images/Graphs/ipw_cox_coefficients_ownOther.png\n")
