pacman::p_load(
    sf, tidyverse, dplyr, ggplot2, broom
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

# Standardize and center continuous variables
pdf$lsm_sk <- scale(pdf$lsm_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lbg_sk <- scale(pdf$lbg_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lti_sk <- scale(pdf$lti_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lal_sk <- scale(pdf$lal_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lni_sk <- scale(pdf$lni_sk, center = TRUE, scale = TRUE)[, 1]
pdf$lLStax_pc <- scale(pdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
pdf$lpopC <- scale(pdf$lpopC, center = TRUE, scale = TRUE)[, 1]
pdf$Y_COORD <- scale(pdf$Y_COORD, center = TRUE, scale = TRUE)[, 1]
pdf$area <- scale(pdf$area, center = TRUE, scale = TRUE)[, 1]
pdf$mean_slope <- scale(pdf$mean_slope, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1535 <- scale(pdf$wet_1535, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1536 <- scale(pdf$wet_1536, center = TRUE, scale = TRUE)[, 1]

monastic_vars <- c("lsm_sk", "lbg_sk", "lti_sk", "lal_sk", "lni_sk", "friary")
controls <- c("lLStax_pc", "wet_1535", "wet_1536", "lpopC", "Y_COORD", "uplands", "lowlands", "area", "mean_slope")

# Variables to plot (in order of appearance in regression)
vars_to_plot <- c(
    "lsm_sk", "lbg_sk", "lti_sk", "lal_sk", "lni_sk", "friary",
    "lLStax_pc", "wet_1535", "wet_1536"
)

# Variable labels for plotting
var_labels <- c(
    "lsm_sk" = "Small Monastery Land / km\u00b2",
    "lbg_sk" = "Large Monastery Land / km\u00b2",
    "lti_sk" = "Tithe / km\u00b2",
    "lal_sk" = "Alms / km\u00b2",
    "lni_sk" = "Net Income / km\u00b2",
    "friary" = "Friary",
    "lLStax_pc" = "Lay Subsidy per Capita",
    "wet_1535" = "Wet 1535",
    "wet_1536" = "Wet 1536"
)

# Function to extract coefficients with CIs (90% CI)
extract_coefs <- function(model, var_name) {
    coef_summary <- summary(model)$coefficients
    coef_val <- coef_summary[var_name, "Estimate"]
    se_val <- coef_summary[var_name, "Std. Error"]
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

# Run one full muster model with all variables
muster_formula <- paste("muster ~", paste(c(monastic_vars, controls), collapse = " + "))
muster_model <- glm(muster_formula, data = pdf, family = binomial(link = "logit"))

# Extract coefficients for variables to plot
muster_coefs_list <- list()
for (var in vars_to_plot) {
    muster_coefs_list[[var]] <- extract_coefs(muster_model, var)
}
muster_coefs <- bind_rows(muster_coefs_list)
muster_coefs$variable_label <- var_labels[muster_coefs$variable]
muster_coefs$significant <- muster_coefs$p_value < 0.10
muster_coefs$order <- match(muster_coefs$variable, vars_to_plot)

# Run one full primary model with all variables
primary_formula <- paste("primary ~", paste(c(monastic_vars, controls), collapse = " + "))
primary_model <- glm(primary_formula, data = pdf, family = binomial(link = "logit"))

# Extract coefficients for variables to plot
primary_coefs_list <- list()
for (var in vars_to_plot) {
    primary_coefs_list[[var]] <- extract_coefs(primary_model, var)
}
primary_coefs <- bind_rows(primary_coefs_list)
primary_coefs$variable_label <- var_labels[primary_coefs$variable]
primary_coefs$significant <- primary_coefs$p_value < 0.10
primary_coefs$order <- match(primary_coefs$variable, vars_to_plot)

# Run one full seat model with all variables
seat_formula <- paste("seats ~", paste(c(monastic_vars, controls), collapse = " + "))
seat_model <- glm(seat_formula, data = pdf, family = "poisson")

# Extract coefficients for variables to plot
seat_coefs_list <- list()
for (var in vars_to_plot) {
    seat_coefs_list[[var]] <- extract_coefs(seat_model, var)
}
seat_coefs <- bind_rows(seat_coefs_list)
seat_coefs$variable_label <- var_labels[seat_coefs$variable]
seat_coefs$significant <- seat_coefs$p_value < 0.10
seat_coefs$order <- match(seat_coefs$variable, vars_to_plot)

# Create coefficient plot for Muster models
muster_plot <- ggplot(muster_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.2, color = "gray30") +
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

# Save muster plot
ggsave("Output/Images/Graphs/muster_coefficients.png",
    plot = muster_plot,
    width = 10, height = 6, dpi = 300
)

# Create coefficient plot for Primary models
primary_plot <- ggplot(primary_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.2, color = "gray30") +
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

# Save primary plot
ggsave("Output/Images/Graphs/primary_coefficients.png",
    plot = primary_plot,
    width = 10, height = 6, dpi = 300
)

# Create coefficient plot for Seats models
seat_plot <- ggplot(seat_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.2, color = "gray30") +
    geom_point(aes(color = significant), size = 3) +
    scale_color_manual(
        values = c("FALSE" = "gray60", "TRUE" = "#0072B2"),
        labels = c("FALSE" = "Not Significant", "TRUE" = "p < 0.10")
    ) +
    labs(
        x = "Coefficient (Log Count)",
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

# Save seat plot
ggsave("Output/Images/Graphs/seats_coefficients.png",
    plot = seat_plot,
    width = 10, height = 6, dpi = 300
)

cat("\nRegression plots created successfully!\n")
cat("- Output/Images/Graphs/muster_coefficients.png\n")
cat("- Output/Images/Graphs/primary_coefficients.png\n")
cat("- Output/Images/Graphs/seats_coefficients.png\n")
