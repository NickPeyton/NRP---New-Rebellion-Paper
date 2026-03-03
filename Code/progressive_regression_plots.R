pacman::p_load(
    sf, tidyverse, dplyr, ggplot2, broom
)

setwd("C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper/")
pdf <- read_sf(dsn = "Data/Processed/northParishFlows.shp")
pdf$lbigLand <- log(pdf$bigLand + 1)

# Standardize and center continuous variables
pdf$llandOwned <- scale(pdf$llandOwned, center = TRUE, scale = TRUE)[, 1]
pdf$lsmLand <- scale(pdf$lsmLand, center = TRUE, scale = TRUE)[, 1]
pdf$lbigLand <- scale(pdf$lbigLand, center = TRUE, scale = TRUE)[, 1]
pdf$ltitheOutT <- scale(pdf$ltitheOutT, center = TRUE, scale = TRUE)[, 1]
pdf$lalmsInTot <- scale(pdf$lalmsInTot, center = TRUE, scale = TRUE)[, 1]
pdf$lnetInc <- scale(pdf$lnetInc, center = TRUE, scale = TRUE)[, 1]
pdf$lLStax_pc <- scale(pdf$lLStax_pc, center = TRUE, scale = TRUE)[, 1]
pdf$lpopC <- scale(pdf$lpopC, center = TRUE, scale = TRUE)[, 1]
pdf$Y_COORD <- scale(pdf$Y_COORD, center = TRUE, scale = TRUE)[, 1]
pdf$area <- scale(pdf$area, center = TRUE, scale = TRUE)[, 1]
pdf$mean_slope <- scale(pdf$mean_slope, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1535 <- scale(pdf$wet_1535, center = TRUE, scale = TRUE)[, 1]
pdf$wet_1536 <- scale(pdf$wet_1536, center = TRUE, scale = TRUE)[, 1]

# Variable groups for progressive model building
var_list_list <- list(
    c("lsmLand", "lbigLand"),
    c("ltitheOutT", "lalmsInTot", "lnetInc", "friary"),
    c("lLStax_pc", "lpopC", "wet_1535", "wet_1536", "dg_percy"),
    c("Y_COORD", "uplands", "lowlands", "area", "mean_slope")
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
    "lpopC" = "Population",
    "wet_1535" = "Wet 1535",
    "wet_1536" = "Wet 1536",
    "dg_percy" = "Percy"
)

# Variables to plot (focus on main variables, not geographic controls)
vars_to_plot <- c(
    "lsmLand", "lbigLand", "ltitheOutT", "lalmsInTot", "lnetInc", "friary",
    "lLStax_pc", "lpopC", "wet_1535", "wet_1536", "dg_percy"
)

# Function to extract coefficients with 90% CI from glm
extract_coefs_glm <- function(model, var_name, model_num) {
    if (!var_name %in% names(coef(model))) {
        return(NULL)
    }
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
        p_value = p_val,
        model = paste("Model", model_num)
    ))
}

# Run muster models with progressive controls
var_list <- c()
muster_results_list <- list()
i <- 1
for (vars in var_list_list) {
    var_list <- c(var_list, vars)
    formula <- paste("muster ~", paste(var_list, collapse = " + "))
    result <- glm(formula, data = pdf, family = binomial(link = "logit"))
    muster_results_list[[i]] <- result
    i <- i + 1
}

# Run primary models with progressive controls
var_list <- c()
primary_results_list <- list()
i <- 1
for (vars in var_list_list) {
    var_list <- c(var_list, vars)
    formula <- paste("primary ~", paste(var_list, collapse = " + "))
    result <- glm(formula, data = pdf, family = binomial(link = "logit"))
    primary_results_list[[i]] <- result
    i <- i + 1
}

# Run seat models with progressive controls (excluding Percy from final models)
var_list <- c()
seat_results_list <- list()
i <- 1
for (vars in var_list_list) {
    var_list <- c(var_list, vars)
    # Remove dg_percy from seats models if present
    var_list_seats <- var_list[var_list != "dg_percy"]
    formula <- paste("seats ~", paste(var_list_seats, collapse = " + "))
    result <- glm(formula, data = pdf, family = "poisson")
    seat_results_list[[i]] <- result
    i <- i + 1
}

# Extract coefficients for muster models
muster_coefs_all <- list()
for (i in 1:4) {
    for (var in vars_to_plot) {
        coef_data <- extract_coefs_glm(muster_results_list[[i]], var, i)
        if (!is.null(coef_data)) {
            muster_coefs_all[[length(muster_coefs_all) + 1]] <- coef_data
        }
    }
}
muster_coefs <- bind_rows(muster_coefs_all)
muster_coefs$variable_label <- var_labels[muster_coefs$variable]
muster_coefs$significant <- muster_coefs$p_value < 0.10
muster_coefs$order <- match(muster_coefs$variable, vars_to_plot)
muster_coefs$model <- factor(muster_coefs$model, levels = c("Model 4", "Model 3", "Model 2", "Model 1"))

# Extract coefficients for primary models
primary_coefs_all <- list()
for (i in 1:4) {
    for (var in vars_to_plot) {
        coef_data <- extract_coefs_glm(primary_results_list[[i]], var, i)
        if (!is.null(coef_data)) {
            primary_coefs_all[[length(primary_coefs_all) + 1]] <- coef_data
        }
    }
}
primary_coefs <- bind_rows(primary_coefs_all)
primary_coefs$variable_label <- var_labels[primary_coefs$variable]
primary_coefs$significant <- primary_coefs$p_value < 0.10
primary_coefs$order <- match(primary_coefs$variable, vars_to_plot)
primary_coefs$model <- factor(primary_coefs$model, levels = c("Model 4", "Model 3", "Model 2", "Model 1"))

# Extract coefficients for seat models (excluding Percy)
vars_to_plot_seats <- vars_to_plot[vars_to_plot != "dg_percy"]
seat_coefs_all <- list()
for (i in 1:4) {
    for (var in vars_to_plot_seats) {
        coef_data <- extract_coefs_glm(seat_results_list[[i]], var, i)
        if (!is.null(coef_data)) {
            seat_coefs_all[[length(seat_coefs_all) + 1]] <- coef_data
        }
    }
}
seat_coefs <- bind_rows(seat_coefs_all)
seat_coefs$variable_label <- var_labels[seat_coefs$variable]
seat_coefs$significant <- seat_coefs$p_value < 0.10
seat_coefs$order <- match(seat_coefs$variable, vars_to_plot_seats)
seat_coefs$model <- factor(seat_coefs$model, levels = c("Model 4", "Model 3", "Model 2", "Model 1"))

# Create plot for Muster models (all 4 models on same plot)
muster_progressive_plot <- ggplot(muster_coefs, aes(
    x = coefficient, y = reorder(variable_label, -order),
    color = model, shape = model
)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(size = 3, position = position_dodge(width = 0.5)) +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
        height = 0,
        position = position_dodge(width = 0.5),
        alpha = 0.7
    ) +
    scale_color_manual(
        values = c(
            "Model 1" = "#D55E00", "Model 2" = "#009E73",
            "Model 3" = "#0072B2", "Model 4" = "#CC79A7"
        ),
        labels = c(
            "Model 1" = "Model 1: Land",
            "Model 2" = "Model 2: + Monastic",
            "Model 3" = "Model 3: + Tax, Population, Weather, Percy",
            "Model 4" = "Model 4: + Geography"
        ),
        breaks = c("Model 1", "Model 2", "Model 3", "Model 4")
    ) +
    scale_shape_manual(
        values = c("Model 1" = 16, "Model 2" = 17, "Model 3" = 15, "Model 4" = 18),
        labels = c(
            "Model 1" = "Model 1: Land",
            "Model 2" = "Model 2: + Monastic",
            "Model 3" = "Model 3: + Tax, Population, Weather, Percy",
            "Model 4" = "Model 4: + Geography"
        ),
        breaks = c("Model 1", "Model 2", "Model 3", "Model 4")
    ) +
    labs(
        x = "Coefficient (Log Odds)",
        y = "",
        color = "Specification",
        shape = "Specification"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.position = "bottom"
    ) +
    guides(color = guide_legend(ncol = 2), shape = guide_legend(ncol = 2))

ggsave("Output/Images/Graphs/muster_progressive_coefficients.png",
    plot = muster_progressive_plot,
    width = 12, height = 8, dpi = 300
)

# Create plot for Primary models (all 4 models on same plot)
primary_progressive_plot <- ggplot(primary_coefs, aes(
    x = coefficient, y = reorder(variable_label, -order),
    color = model, shape = model
)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(size = 3, position = position_dodge(width = 0.5)) +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
        height = 0,
        position = position_dodge(width = 0.5),
        alpha = 0.7
    ) +
    scale_color_manual(
        values = c(
            "Model 1" = "#D55E00", "Model 2" = "#009E73",
            "Model 3" = "#0072B2", "Model 4" = "#CC79A7"
        ),
        labels = c(
            "Model 1" = "Model 1: Land",
            "Model 2" = "Model 2: + Monastic",
            "Model 3" = "Model 3: + Tax, Population, Weather, Percy",
            "Model 4" = "Model 4: + Geography"
        ),
        breaks = c("Model 1", "Model 2", "Model 3", "Model 4")
    ) +
    scale_shape_manual(
        values = c("Model 1" = 16, "Model 2" = 17, "Model 3" = 15, "Model 4" = 18),
        labels = c(
            "Model 1" = "Model 1: Land",
            "Model 2" = "Model 2: + Monastic",
            "Model 3" = "Model 3: + Tax, Population, Weather, Percy",
            "Model 4" = "Model 4: + Geography"
        ),
        breaks = c("Model 1", "Model 2", "Model 3", "Model 4")
    ) +
    labs(
        x = "Coefficient (Log Odds)",
        y = "",
        color = "Specification",
        shape = "Specification"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.position = "bottom"
    ) +
    guides(color = guide_legend(ncol = 2), shape = guide_legend(ncol = 2))

ggsave("Output/Images/Graphs/primary_progressive_coefficients.png",
    plot = primary_progressive_plot,
    width = 12, height = 8, dpi = 300
)

# Create plot for Seats models (all 4 models on same plot)
seat_progressive_plot <- ggplot(seat_coefs, aes(
    x = coefficient, y = reorder(variable_label, -order),
    color = model, shape = model
)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(size = 3, position = position_dodge(width = 0.5)) +
    geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
        height = 0,
        position = position_dodge(width = 0.5),
        alpha = 0.7
    ) +
    scale_color_manual(
        values = c(
            "Model 1" = "#D55E00", "Model 2" = "#009E73",
            "Model 3" = "#0072B2", "Model 4" = "#CC79A7"
        ),
        labels = c(
            "Model 1" = "Model 1: Land",
            "Model 2" = "Model 2: + Monastic",
            "Model 3" = "Model 3: + Tax, Population, Weather, Percy",
            "Model 4" = "Model 4: + Geography"
        ),
        breaks = c("Model 1", "Model 2", "Model 3", "Model 4")
    ) +
    scale_shape_manual(
        values = c("Model 1" = 16, "Model 2" = 17, "Model 3" = 15, "Model 4" = 18),
        labels = c(
            "Model 1" = "Model 1: Land",
            "Model 2" = "Model 2: + Monastic",
            "Model 3" = "Model 3: + Tax, Population, Weather, Percy",
            "Model 4" = "Model 4: + Geography"
        ),
        breaks = c("Model 1", "Model 2", "Model 3", "Model 4")
    ) +
    labs(
        x = "Coefficient (Log Count)",
        y = "",
        color = "Specification",
        shape = "Specification"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.position = "bottom"
    ) +
    guides(color = guide_legend(ncol = 2), shape = guide_legend(ncol = 2))

ggsave("Output/Images/Graphs/seats_progressive_coefficients.png",
    plot = seat_progressive_plot,
    width = 12, height = 8, dpi = 300
)

cat("\nProgressive control regression plots created successfully!\n")
cat("- Output/Images/Graphs/muster_progressive_coefficients.png\n")
cat("- Output/Images/Graphs/primary_progressive_coefficients.png\n")
cat("- Output/Images/Graphs/seats_progressive_coefficients.png\n")
