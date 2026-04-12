pacman::p_load(
    sf, tidyverse, dplyr, ggplot2, survival, broom
)

PROJECT_ROOT <- normalizePath(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), ".."))
setwd(PROJECT_ROOT)
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")

day <- 40
rdf <- data.frame(pdf)
rdf$day <- replace(rdf$day, rdf$day < 1, day)
rdf$day <- ifelse(is.na(rdf$day), day, rdf$day)
rdf$primary <- ifelse(is.na(rdf$primary), 0, rdf$primary)
rdf$primary_day <- rdf$day * rdf$primary
rdf$primary_day <- replace(rdf$primary_day, rdf$primary_day < 1, day)

rdf$survival <- rdf$day - rdf$news_day
rdf$primary_survival <- rdf$primary_day - rdf$news_day
rdf$primary_survival <- ifelse(is.na(rdf$primary_survival), day, rdf$primary_survival)

# Standardize and center continuous variables
rdf$lsm_arak <- scale(rdf$lsm_arak, center = TRUE, scale = TRUE)[, 1]
rdf$lbg_arak <- scale(rdf$lbg_arak, center = TRUE, scale = TRUE)[, 1]
rdf$lti_arak <- scale(rdf$lti_arak, center = TRUE, scale = TRUE)[, 1]
rdf$lal_arak <- scale(rdf$lal_arak, center = TRUE, scale = TRUE)[, 1]
rdf$lnetInc <- scale(rdf$lnetInc, center = TRUE, scale = TRUE)[, 1]
rdf$lLStax_pc <- scale(rdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
rdf$lpopC <- scale(rdf$lpopC, center = TRUE, scale = TRUE)[, 1]
rdf$Y_COORD <- scale(rdf$Y_COORD, center = TRUE, scale = TRUE)[, 1]
rdf$area <- scale(rdf$area, center = TRUE, scale = TRUE)[, 1]
rdf$mean_slope <- scale(rdf$mean_slope, center = TRUE, scale = TRUE)[, 1]
rdf$wet_1535 <- scale(rdf$wet_1535, center = TRUE, scale = TRUE)[, 1]
rdf$wet_1536 <- scale(rdf$wet_1536, center = TRUE, scale = TRUE)[, 1]
rdf$dwx_1536 <- scale(rdf$dwx_1536, center = TRUE, scale = TRUE)[, 1]

# Run the three Cox models
cox1 <- coxph(
    Surv(primary_survival, primary) ~
        lsm_arak +
        lbg_arak +
        lti_arak +
        lal_arak +
        lnetInc +
        friary +
        wet_1535 + wet_1536,
    data = rdf
)

cox2 <- coxph(
    Surv(primary_survival, primary) ~
        lsm_arak +
        lbg_arak +
        lti_arak +
        lal_arak +
        lnetInc +
        friary +
        wet_1535 + wet_1536 +
        lLStax_pc +
        lpopC,
    data = rdf
)

cox3 <- coxph(Surv(primary_survival, primary) ~
    lsm_arak +
    lbg_arak +
    lti_arak +
    lal_arak +
    lnetInc +
    friary +
    wet_1535 + wet_1536 +
    lLStax_pc +
    lpopC +
    Y_COORD +
    area +
    uplands +
    lowlands +
    mean_slope, data = rdf)

# Variables to plot for each model
vars_cox1 <- c(
    "lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lnetInc", "friary",
    "wet_1535", "wet_1536"
)

vars_cox2 <- c(
    "lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lnetInc", "friary",
    "wet_1535", "wet_1536", "lLStax_pc", "lpopC"
)

vars_cox3 <- c(
    "lsm_arak", "lbg_arak", "lti_arak", "lal_arak", "lnetInc", "friary",
    "wet_1535", "wet_1536", "lLStax_pc", "lpopC"
)

# Variable labels
var_labels <- c(
    "lsm_arak" = "Small Monastery Land / Arable km²",
    "lbg_arak" = "Large Monastery Land / Arable km²",
    "lti_arak" = "Tithe / Arable km²",
    "lal_arak" = "Alms / Arable km²",
    "lnetInc" = "Net Income",
    "friary" = "Friary",
    "wet_1535" = "Wet 1535",
    "wet_1536" = "Wet 1536",
    "lLStax_pc" = "Lay Subsidy per Capita",
    "lpopC" = "Population"
)

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

# Extract coefficients for Cox Model 1
cox1_coefs_list <- list()
for (var in vars_cox1) {
    cox1_coefs_list[[var]] <- extract_coefs_coxph(cox1, var)
}
cox1_coefs <- bind_rows(cox1_coefs_list)
cox1_coefs$variable_label <- var_labels[cox1_coefs$variable]
cox1_coefs$significant <- cox1_coefs$p_value < 0.10
cox1_coefs$order <- match(cox1_coefs$variable, vars_cox1)

# Extract coefficients for Cox Model 2
cox2_coefs_list <- list()
for (var in vars_cox2) {
    cox2_coefs_list[[var]] <- extract_coefs_coxph(cox2, var)
}
cox2_coefs <- bind_rows(cox2_coefs_list)
cox2_coefs$variable_label <- var_labels[cox2_coefs$variable]
cox2_coefs$significant <- cox2_coefs$p_value < 0.10
cox2_coefs$order <- match(cox2_coefs$variable, vars_cox2)

# Extract coefficients for Cox Model 3 (only plotting same vars as cox2)
cox3_coefs_list <- list()
for (var in vars_cox3) {
    cox3_coefs_list[[var]] <- extract_coefs_coxph(cox3, var)
}
cox3_coefs <- bind_rows(cox3_coefs_list)
cox3_coefs$variable_label <- var_labels[cox3_coefs$variable]
cox3_coefs$significant <- cox3_coefs$p_value < 0.10
cox3_coefs$order <- match(cox3_coefs$variable, vars_cox3)

# Create coefficient plot for Cox Model 1
cox1_plot <- ggplot(cox1_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
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

# Save cox1 plot
ggsave("Output/Images/Graphs/cox1_coefficients.png",
    plot = cox1_plot,
    width = 10, height = 6, dpi = 300
)

# Create coefficient plot for Cox Model 2
cox2_plot <- ggplot(cox2_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
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

# Save cox2 plot
ggsave("Output/Images/Graphs/cox2_coefficients.png",
    plot = cox2_plot,
    width = 10, height = 6, dpi = 300
)

# Create coefficient plot for Cox Model 3
cox3_plot <- ggplot(cox3_coefs, aes(x = coefficient, y = reorder(variable_label, -order))) +
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

# Save cox3 plot
ggsave("Output/Images/Graphs/cox3_coefficients.png",
    plot = cox3_plot,
    width = 10, height = 6, dpi = 300
)

cat("\nSurvival analysis regression plots created successfully!\n")
cat("- Output/Images/Graphs/cox1_coefficients.png\n")
cat("- Output/Images/Graphs/cox2_coefficients.png\n")
cat("- Output/Images/Graphs/cox3_coefficients.png\n")
