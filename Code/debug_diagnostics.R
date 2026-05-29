pacman::p_load(DoubleML, mlr3, mlr3learners, ranger, glmnet, sf, jsonlite)
lgr::get_logger('mlr3')$set_threshold('warn')
pdf <- read_sf('Data/Processed/northParishFlows.shp')
continuous_vars <- c('llandOwned', 'llo_sk',  'llo_arak', 'lsmLand', 'lbigLand', 'lsm_sk', 'lbg_sk', 'lsm_arak', 'lbg_arak', 'lotherLand', 'lownLand', 'loth_sk', 'lown_sk', 'loth_arak', 'lown_arak', 'ltitheOutT', 'lti_sk', 'lti_arak', 'lalmsInTot', 'lal_sk', 'lal_arak', 'lni_sk', 'lni_arak', 'lLStax_pc', 'wet_1535', 'wet_1536', 'lpopC', 'area', 'mean_slope', 'distScot')
for (v in continuous_vars) { pdf[[v]] <- scale(pdf[[v]], center = TRUE, scale = TRUE)[, 1] }
df <- as.data.frame(sf::st_drop_geometry(pdf))
df$cluster_id <- ifelse(is.na(df$hundred), paste0('_singleton_', seq_len(nrow(df))), as.character(df$hundred))

geo_controls <- c('mg_fsnub', 'mg_court', 'wet_1535', 'wet_1536', 'lLStax_pc', 'lpopC', 'distScot', 'mean_slope', 'area')
get_x_cols <- function(comp) c(comp, geo_controls)

test_diagnostics <- function(treat, comp, dep) {
  x_cols <- get_x_cols(comp)
  df_dml <- df[complete.cases(df[, c(treat, x_cols, dep)]), ]
  dml_data <- DoubleMLClusterData$new(data=df_dml, y_col=dep, d_cols=treat, x_cols=x_cols, cluster_cols='cluster_id')
  dml <- DoubleMLPLR$new(data=dml_data, ml_l=lrn('regr.ranger', num.trees=200), ml_m=lrn('regr.cv_glmnet', s='lambda.min'), n_folds=5, score='partialling out')
  set.seed(42)
  dml$fit(store_predictions = TRUE)
  
  y_hat <- dml$predictions$ml_l[, 1, 1]
  d_hat <- dml$predictions$ml_m[, 1, 1]
  res_y <- dml$data$data[[dml$data$y_col]] - y_hat
  res_d <- dml$data$data[[dml$data$d_cols[1]]] - d_hat
  
  cat(sprintf('\n=== Diagnostics for %s on %s ===\n', treat, dep))
  cat(sprintf('Original Treatment SD: %.4f\n', sd(dml$data$data[[dml$data$d_cols[1]]])))
  cat(sprintf('Residual Treatment SD: %.4f\n', sd(res_d)))
  cat(sprintf('Original Outcome SD: %.4f\n', sd(dml$data$data[[dml$data$y_col]])))
  cat(sprintf('Residual Outcome SD: %.4f\n', sd(res_y)))
  cat(sprintf('Correlation (res_d, res_y): %.4f\n', cor(res_d, res_y)))
}

test_diagnostics('lbg_arak', 'lsm_arak', 'primary')
test_diagnostics('llo_arak', NULL, 'primary')
