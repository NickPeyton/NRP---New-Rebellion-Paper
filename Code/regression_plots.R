# Regression Plots for AIPW and Survival Analysis Models
# Using 90% Confidence Intervals with Monastic Land Variable Highlighted
pacman::p_load(
    sf, tidyverse, ggplot2, survival, survminer,
    WeightIt, survey, cobalt, broom, patchwork,
    gridExtra, ggfortify
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")

# Custom function to calculate 90% CI for svyglm objects
confint90_svyglm <- function(model) {
    ci <- confint(model, level = 0.90)
    return(ci)
}

# Load and prepare data (same as in both scripts)
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
rdf$primary <- ifelse(is.na(rdf$primary), 0, rdf$primary)
rdf$primary_day <- rdf$day * rdf$primary
rdf$primary_day <- replace(rdf$primary_day, rdf$primary_day < 1, day)
rdf$survival <- rdf$day - rdf$news_day
rdf$primary_survival <- rdf$primary_day - rdf$news_day
rdf$primary_survival <- ifelse(is.na(rdf$primary_survival), day, rdf$primary_survival)
rdf$llandOwned <- log(rdf$landOwned + 1)

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
rdf$wet_1535 <- scale(rdf$wet_1535, center = TRUE, scale = TRUE)[, 1]
rdf$wet_1536 <- scale(rdf$wet_1536, center = TRUE, scale = TRUE)[, 1]
rdf$dwx_1536 <- scale(rdf$dwx_1536, center = TRUE, scale = TRUE)[, 1]

# ============================================================================
# PART 1: AIPW MODELS AND PLOTS
# ============================================================================

cat("Running AIPW models...\n")

# Fit propensity score model
weightitmodel <- weightit(
    llandOwned ~
        ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + X_COORD + Y_COORD + area + uplands + lowlands +
        mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    method = "cbps",
    over = FALSE
)
weights <- weightitmodel$weights

# Weighted logit model
weighted_lm <- svyglm(
    primary ~ llandOwned + ltitheOutT + lalmsInTot + lnetInc + friary +
        lLStax_pc + lpopC + X_COORD + Y_COORD + area + uplands + lowlands +
        mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    weights = weights,
    design = svydesign(~1, weights = weights, data = rdf),
    family = quasibinomial()
)

# Weighted Cox model
weighted_survival <- coxph(
    Surv(primary_survival, primary) ~ llandOwned + ltitheOutT + lalmsInTot +
        lnetInc + friary + lLStax_pc + lpopC + X_COORD + Y_COORD + area +
        uplands + lowlands + mean_slope + wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf,
    weights = weights,
    robust = TRUE
)

# Extract coefficients for plotting
coef_labels <- c(
    "llandOwned" = "ln(Land Owned)",
    "ltitheOutT" = "ln(Tithe)",
    "lalmsInTot" = "ln(Alms)",
    "lnetInc" = "ln(Net Income)",
    "friary" = "Friary",
    "lLStax_pc" = "ln(Lay Subsidy)",
    "lpopC" = "ln(Population)",
    "wet_1535" = "Wet 1535",
    "wet_1536" = "Wet 1536",
    "dwx_1536" = "Dry x Wet",
    "dg_percy" = "Percy",
    "X_COORD" = "X Coordinate",
    "Y_COORD" = "Y Coordinate",
    "area" = "Area",
    "uplands" = "Uplands",
    "lowlands" = "Lowlands",
    "mean_slope" = "Mean Slope"
)

# Plot 1: Coefficient plot for weighted logit
cat("Creating weighted logit coefficient plot...\n")
# Calculate 90% CI manually for svyglm
logit_ci <- confint90_svyglm(weighted_lm)
logit_coefs <- tidy(weighted_lm) %>%
    filter(term != "(Intercept)") %>%
    mutate(
        conf.low = logit_ci[term, 1],
        conf.high = logit_ci[term, 2],
        label = ifelse(term %in% names(coef_labels),
            coef_labels[term], term
        ),
        significant = p.value < 0.10,
        is_land = term == "llandOwned"
    )

p1 <- ggplot(logit_coefs, aes(x = estimate, y = reorder(label, estimate))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(
        aes(
            xmin = conf.low, xmax = conf.high,
            color = is_land, linewidth = is_land
        ),
        height = 0.3
    ) +
    geom_point(aes(color = is_land, size = is_land, shape = significant)) +
    scale_color_manual(values = c("FALSE" = "gray50", "TRUE" = "#8B0000")) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 4.5)) +
    scale_linewidth_manual(values = c("FALSE" = 0.5, "TRUE" = 1.2)) +
    scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16)) +
    labs(
        title = "IPW Logit Model: Coefficient Estimates",
        subtitle = "90% Confidence Intervals (Monastic Land Highlighted in Red)",
        x = "Coefficient Estimate",
        y = ""
    ) +
    theme_minimal() +
    theme(
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(
            size = 10,
            face = ifelse(logit_coefs$is_land[order(logit_coefs$estimate)],
                "bold", "plain"
            )
        )
    )

ggsave("Output/Images/Graphs/ipw_logit_coefficients.png", p1,
    width = 8, height = 10, dpi = 300
)

# Plot 2: Hazard ratios for weighted Cox model
cat("Creating weighted Cox hazard ratio plot...\n")
cox_coefs <- tidy(weighted_survival, conf.int = TRUE, conf.level = 0.90, exponentiate = TRUE)
cox_coefs <- cox_coefs %>%
    mutate(
        label = ifelse(term %in% names(coef_labels),
            coef_labels[term], term
        ),
        significant = p.value < 0.10,
        is_land = term == "llandOwned"
    )

p2 <- ggplot(cox_coefs, aes(x = estimate, y = reorder(label, estimate))) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_errorbarh(
        aes(
            xmin = conf.low, xmax = conf.high,
            color = is_land, linewidth = is_land
        ),
        height = 0.3
    ) +
    geom_point(aes(color = is_land, size = is_land, shape = significant)) +
    scale_color_manual(values = c("FALSE" = "gray50", "TRUE" = "#8B0000")) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 4.5)) +
    scale_linewidth_manual(values = c("FALSE" = 0.5, "TRUE" = 1.2)) +
    scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16)) +
    labs(
        title = "IPW Cox Model: Hazard Ratios",
        subtitle = "90% Confidence Intervals (Monastic Land Highlighted in Red)",
        x = "Hazard Ratio",
        y = ""
    ) +
    theme_minimal() +
    theme(
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(
            size = 10,
            face = ifelse(cox_coefs$is_land[order(cox_coefs$estimate)],
                "bold", "plain"
            )
        )
    )

ggsave("Output/Images/Graphs/ipw_cox_hazard_ratios.png", p2,
    width = 8, height = 10, dpi = 300
)

# Plot 3: Balance plot for propensity scores
cat("Creating balance plot...\n")
tryCatch(
    {
        p3_balance <- love.plot(weightitmodel,
            stats = c("mean.diffs"),
            thresholds = c(m = .1),
            abs = TRUE,
            var.order = "unadjusted",
            title = "Covariate Balance: Before and After Weighting"
        )
        ggsave("Output/Images/Graphs/ipw_balance_plot.png", p3_balance,
            width = 10, height = 8, dpi = 300
        )
    },
    error = function(e) {
        cat("Warning: Could not create balance plot. Error:", e$message, "\n")
        cat("Skipping balance plot...\n")
    }
)

# ============================================================================
# PART 2: SURVIVAL ANALYSIS MODELS AND PLOTS
# ============================================================================

cat("Running survival analysis models...\n")

# Fit Cox models
cox1 <- coxph(
    Surv(primary_survival, primary) ~
        llandOwned + ltitheOutT + lalmsInTot + lnetInc + friary +
        wet_1535 + wet_1536 + dwx_1536 + dg_percy,
    data = rdf
)

cox2 <- coxph(
    Surv(primary_survival, primary) ~
        llandOwned + ltitheOutT + lalmsInTot + lnetInc + friary +
        wet_1535 + wet_1536 + dwx_1536 + dg_percy +
        lLStax_pc + lpopC,
    data = rdf
)

cox3 <- coxph(
    Surv(primary_survival, primary) ~
        llandOwned + ltitheOutT + lalmsInTot + lnetInc + friary +
        wet_1535 + wet_1536 + dwx_1536 + dg_percy +
        lLStax_pc + lpopC + X_COORD + Y_COORD + area +
        uplands + lowlands + mean_slope,
    data = rdf
)

# Extract coefficients from all models with 90% CI
vars_of_interest <- c(
    "llandOwned", "ltitheOutT", "lalmsInTot", "lnetInc",
    "friary", "wet_1535", "wet_1536", "dwx_1536", "dg_percy"
)

cox1_tidy <- tidy(cox1, conf.int = TRUE, conf.level = 0.90, exponentiate = TRUE) %>%
    mutate(model = "Model 1: Land")

cox2_tidy <- tidy(cox2, conf.int = TRUE, conf.level = 0.90, exponentiate = TRUE) %>%
    mutate(model = "Model 2: + Pop/Tax")

cox3_tidy <- tidy(cox3, conf.int = TRUE, conf.level = 0.90, exponentiate = TRUE) %>%
    mutate(model = "Model 3: + Geography")

# Plot 4a: Individual Cox Model 1 plot
cat("Creating Cox Model 1 plot...\n")
cox1_plot_data <- cox1_tidy %>%
    mutate(
        label = ifelse(term %in% names(coef_labels),
            coef_labels[term], term
        ),
        significant = p.value < 0.10,
        is_land = term == "llandOwned"
    )

p4a <- ggplot(cox1_plot_data, aes(x = estimate, y = reorder(label, estimate))) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_errorbarh(
        aes(
            xmin = conf.low, xmax = conf.high,
            color = is_land, linewidth = is_land
        ),
        height = 0.3
    ) +
    geom_point(aes(color = is_land, size = is_land, shape = significant)) +
    scale_color_manual(values = c("FALSE" = "gray50", "TRUE" = "#8B0000")) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 4.5)) +
    scale_linewidth_manual(values = c("FALSE" = 0.5, "TRUE" = 1.2)) +
    scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16)) +
    labs(
        title = "Cox Model 1: Hazard Ratios (Land Variables)",
        subtitle = "90% Confidence Intervals (Monastic Land Highlighted in Red)",
        x = "Hazard Ratio",
        y = ""
    ) +
    theme_minimal() +
    theme(
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(
            size = 10,
            face = ifelse(cox1_plot_data$is_land[order(cox1_plot_data$estimate)],
                "bold", "plain"
            )
        )
    )

ggsave("Output/Images/Graphs/cox_model1_hazard_ratios.png", p4a,
    width = 8, height = 8, dpi = 300
)

# Plot 4b: Individual Cox Model 2 plot
cat("Creating Cox Model 2 plot...\n")
cox2_plot_data <- cox2_tidy %>%
    mutate(
        label = ifelse(term %in% names(coef_labels),
            coef_labels[term], term
        ),
        significant = p.value < 0.10,
        is_land = term == "llandOwned"
    )

p4b <- ggplot(cox2_plot_data, aes(x = estimate, y = reorder(label, estimate))) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_errorbarh(
        aes(
            xmin = conf.low, xmax = conf.high,
            color = is_land, linewidth = is_land
        ),
        height = 0.3
    ) +
    geom_point(aes(color = is_land, size = is_land, shape = significant)) +
    scale_color_manual(values = c("FALSE" = "gray50", "TRUE" = "#8B0000")) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 4.5)) +
    scale_linewidth_manual(values = c("FALSE" = 0.5, "TRUE" = 1.2)) +
    scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16)) +
    labs(
        title = "Cox Model 2: Hazard Ratios (+ Population/Tax)",
        subtitle = "90% Confidence Intervals (Monastic Land Highlighted in Red)",
        x = "Hazard Ratio",
        y = ""
    ) +
    theme_minimal() +
    theme(
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(
            size = 10,
            face = ifelse(cox2_plot_data$is_land[order(cox2_plot_data$estimate)],
                "bold", "plain"
            )
        )
    )

ggsave("Output/Images/Graphs/cox_model2_hazard_ratios.png", p4b,
    width = 8, height = 9, dpi = 300
)

# Plot 4c: Forest plot comparing all three Cox models
cat("Creating forest plot for Cox models comparison...\n")

all_cox <- bind_rows(
    cox1_tidy %>% filter(term %in% vars_of_interest),
    cox2_tidy %>% filter(term %in% vars_of_interest),
    cox3_tidy %>% filter(term %in% vars_of_interest)
) %>%
    mutate(
        label = ifelse(term %in% names(coef_labels),
            coef_labels[term], term
        ),
        significant = p.value < 0.10,
        is_land = term == "llandOwned"
    )

p4c <- ggplot(all_cox, aes(
    x = estimate, y = reorder(label, estimate),
    color = interaction(model, is_land), shape = model
)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
        height = 0, position = position_dodge(width = 0.5)
    ) +
    geom_point(aes(size = is_land), position = position_dodge(width = 0.5)) +
    scale_color_manual(values = c(
        "Model 1: Land.FALSE" = "darkblue",
        "Model 1: Land.TRUE" = "#8B0000",
        "Model 2: + Pop/Tax.FALSE" = "darkgreen",
        "Model 2: + Pop/Tax.TRUE" = "#8B0000",
        "Model 3: + Geography.FALSE" = "darkorange",
        "Model 3: + Geography.TRUE" = "#8B0000"
    )) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 4.5)) +
    guides(color = "none", size = "none") +
    labs(
        title = "Cox Proportional Hazards Models: Hazard Ratios Comparison",
        subtitle = "90% Confidence Intervals (Monastic Land Highlighted in Red)",
        x = "Hazard Ratio",
        y = "",
        shape = "Model Specification"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(size = 10),
        legend.position = "bottom"
    )

ggsave("Output/Images/Graphs/cox_models_comparison.png", p4c,
    width = 10, height = 8, dpi = 300
)

# Plot 5: Full coefficient plot for Model 3 (with geography)
cat("Creating full Model 3 coefficient plot...\n")
cox3_full <- tidy(cox3, conf.int = TRUE, conf.level = 0.90, exponentiate = TRUE) %>%
    mutate(
        label = ifelse(term %in% names(coef_labels),
            coef_labels[term], term
        ),
        significant = p.value < 0.10,
        is_land = term == "llandOwned"
    )

p5 <- ggplot(cox3_full, aes(x = estimate, y = reorder(label, estimate))) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_errorbarh(
        aes(
            xmin = conf.low, xmax = conf.high,
            color = is_land, linewidth = is_land
        ),
        height = 0.3
    ) +
    geom_point(aes(color = is_land, size = is_land, shape = significant)) +
    scale_color_manual(values = c("FALSE" = "gray50", "TRUE" = "#8B0000")) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 4.5)) +
    scale_linewidth_manual(values = c("FALSE" = 0.5, "TRUE" = 1.2)) +
    scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16)) +
    labs(
        title = "Full Cox Model (Model 3): Hazard Ratios",
        subtitle = "With Geographic Controls - 90% Confidence Intervals (Monastic Land in Red)",
        x = "Hazard Ratio",
        y = ""
    ) +
    theme_minimal() +
    theme(
        legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(
            size = 10,
            face = ifelse(cox3_full$is_land[order(cox3_full$estimate)],
                "bold", "plain"
            )
        )
    )

ggsave("Output/Images/Graphs/cox_model3_full.png", p5,
    width = 8, height = 10, dpi = 300
)

# Plot 6: Survival curves
cat("Creating survival curves...\n")
fit_surv <- survfit(Surv(primary_survival, primary) ~ 1, data = rdf, conf.int = 0.90)

p6 <- ggsurvplot(
    fit_surv,
    data = rdf,
    conf.int = TRUE,
    risk.table = TRUE,
    risk.table.height = 0.25,
    xlab = "Days Since News Arrival",
    ylab = "Probability of No Rebellion",
    title = "Kaplan-Meier Survival Curve (90% CI)",
    ggtheme = theme_minimal()
)

ggsave("Output/Images/Graphs/survival_curve.png",
    print(p6),
    width = 10, height = 8, dpi = 300
)

# Plot 7: Stratified survival curves by land ownership quartiles
cat("Creating stratified survival curves...\n")
# Use unique() to handle duplicate breaks in quantiles
breaks <- unique(quantile(rdf$llandOwned, probs = seq(0, 1, 0.25)))
n_groups <- length(breaks) - 1

if (n_groups >= 2) {
    # Create labels based on actual number of groups
    if (n_groups == 4) {
        group_labels <- c("Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)")
    } else if (n_groups == 3) {
        group_labels <- c("Lower Tertile", "Middle Tertile", "Upper Tertile")
    } else {
        group_labels <- c("Lower Half", "Upper Half")
    }

    rdf$land_quartile <- cut(rdf$llandOwned,
        breaks = breaks,
        labels = group_labels,
        include.lowest = TRUE
    )
} else {
    # Fallback: create binary split if not enough unique values
    rdf$land_quartile <- cut(rdf$llandOwned,
        breaks = 2,
        labels = c("Lower Half", "Upper Half"),
        include.lowest = TRUE
    )
    group_labels <- c("Lower Half", "Upper Half")
    n_groups <- 2
}

fit_surv_strat <- survfit(Surv(primary_survival, primary) ~ land_quartile,
    data = rdf, conf.int = 0.90
)

# Create palette based on number of groups
palette_colors <- c("#1B9E77", "#D95F02", "#7570B3", "#8B0000")[1:n_groups]

p7 <- ggsurvplot(
    fit_surv_strat,
    data = rdf,
    conf.int = TRUE,
    pval = TRUE,
    risk.table = FALSE,
    xlab = "Days Since News Arrival",
    ylab = "Probability of No Rebellion",
    title = "Survival by Monastic Land Ownership Quartile (90% CI)",
    legend.title = "Land Owned",
    legend.labs = group_labels,
    ggtheme = theme_minimal(),
    palette = palette_colors
)

ggsave("Output/Images/Graphs/survival_by_land.png",
    p7$plot,
    width = 10, height = 8, dpi = 300
)

# Plot 8: Combined coefficient comparison plot (AIPW vs regular Cox)
cat("Creating IPW vs unweighted comparison...\n")

ipw_cox <- tidy(weighted_survival, conf.int = TRUE, conf.level = 0.90, exponentiate = TRUE) %>%
    filter(term %in% vars_of_interest) %>%
    mutate(model = "IPW Cox")

cox3_subset <- tidy(cox3, conf.int = TRUE, conf.level = 0.90, exponentiate = TRUE) %>%
    filter(term %in% vars_of_interest) %>%
    mutate(model = "Unweighted Cox")

comparison <- bind_rows(ipw_cox, cox3_subset) %>%
    mutate(
        label = ifelse(term %in% names(coef_labels),
            coef_labels[term], term
        ),
        significant = p.value < 0.10,
        is_land = term == "llandOwned"
    )

p8 <- ggplot(comparison, aes(
    x = estimate, y = reorder(label, estimate),
    color = interaction(model, is_land), shape = model
)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
        height = 0, position = position_dodge(width = 0.5)
    ) +
    geom_point(aes(size = is_land), position = position_dodge(width = 0.5)) +
    scale_color_manual(values = c(
        "IPW Cox.FALSE" = "darkblue",
        "IPW Cox.TRUE" = "#8B0000",
        "Unweighted Cox.FALSE" = "darkorange",
        "Unweighted Cox.TRUE" = "#8B0000"
    )) +
    scale_size_manual(values = c("FALSE" = 3, "TRUE" = 4.5)) +
    guides(color = "none", size = "none") +
    labs(
        title = "IPW vs Unweighted Cox Models: Hazard Ratios",
        subtitle = "Comparing causal estimates - 90% CI (Monastic Land in Red)",
        x = "Hazard Ratio",
        y = "",
        shape = "Model Type"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(face = "bold", size = 14),
        axis.text.y = element_text(size = 10),
        legend.position = "bottom"
    )

ggsave("Output/Images/Graphs/ipw_vs_unweighted.png", p8,
    width = 10, height = 8, dpi = 300
)

cat("\n========================================\n")
cat("All plots saved successfully!\n")
cat("Output directory: Output/Images/Graphs/\n")
cat("========================================\n")
cat("\nGenerated plots (all with 90% CI and monastic land highlighted):\n")
cat("1. ipw_logit_coefficients.png - IPW Logit coefficients\n")
cat("2. ipw_cox_hazard_ratios.png - IPW Cox hazard ratios\n")
cat("3. ipw_balance_plot.png - Covariate balance plot\n")
cat("4a. cox_model1_hazard_ratios.png - Cox Model 1 (Land only)\n")
cat("4b. cox_model2_hazard_ratios.png - Cox Model 2 (+ Pop/Tax)\n")
cat("4c. cox_models_comparison.png - All three Cox models comparison\n")
cat("5. cox_model3_full.png - Full Model 3 with geography\n")
cat("6. survival_curve.png - Overall survival curve\n")
cat("7. survival_by_land.png - Survival by land ownership quartile\n")
cat("8. ipw_vs_unweighted.png - IPW vs unweighted comparison\n")
cat("\nNote: Monastic land variable (ln(Land Owned)) is highlighted in dark red\n")
cat("      across all relevant plots with bold axis labels.\n")
