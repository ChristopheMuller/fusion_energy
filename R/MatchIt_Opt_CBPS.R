# if (file.exists("renv/activate.R")) suppressMessages(source("renv/activate.R"))
suppressPackageStartupMessages(library(MatchIt))
suppressPackageStartupMessages(library(CBPS))

estimate_matchit_Opt_CBPS <- function(X_treat, Y_treat, X_control_int, Y_control_int, X_external, Y_external, true_sate, n_external) {
  
  # 1. Pool all controls (Internal + External)
  X_pool <- rbind(X_control_int, X_external)
  Y_pool <- c(Y_control_int, Y_external)
  
  # Create source indicator to track which controls are selected
  source_vec <- c(rep("Internal", nrow(X_control_int)), rep("External", nrow(X_external)))
  
  # Combine with Treated
  X_combined <- rbind(X_treat, X_pool)
  Y_combined <- c(Y_treat, Y_pool)
  treat_vec <- c(rep(1, nrow(X_treat)), rep(0, nrow(X_pool)))
  
  df_combined <- as.data.frame(X_combined)
  df_combined$T_assign <- treat_vec
  
  # Formula for Propensity Score
  formula_vars <- colnames(df_combined)[1:(ncol(df_combined) - 1)]
  formula_str <- paste("T_assign ~", paste(formula_vars, collapse = " + "))
  
  start_time <- Sys.time()
  
  result <- tryCatch({
    # 2. Run MatchIt with distance = "cbps"
    # This uses CBPS to estimate the scores and then performs 1:1 Nearest Neighbor matching
    capture.output(
      m.out <- matchit(as.formula(formula_str), 
                       data = df_combined,
                       method = "optimal",
                       distance = "cbps",
                       ratio = 1,
                       replace = FALSE)
    )
    
    # 3. Extract the matched dataset
    # This dataset contains only the Treated and the Selected Controls
    matched_data <- match.data(m.out)
    
    # 4. Estimate ATE on the matched sample
    y_tr_matched <- Y_combined[as.numeric(rownames(matched_data[matched_data$T_assign == 1, ]))]
    y_ct_matched <- Y_combined[as.numeric(rownames(matched_data[matched_data$T_assign == 0, ]))]
    
    ate_est <- mean(y_tr_matched) - mean(y_ct_matched)
    
    all_weights <- m.out$weights
    
    n_tr <- nrow(X_treat)
    n_int <- nrow(X_control_int)
    n_total_ctrl <- nrow(X_pool)
    
    weights_pool <- all_weights[(n_tr + 1):(n_tr + n_total_ctrl)]
    w_int <- weights_pool[(n_int + 1):n_total_ctrl]
    
    list(ate = ate_est, w_int = w_int, object = m.out)
    
  }, error = function(e) {
    warning(paste("MatchIt CBPS failed:", e$message))
    list(ate = NaN, w_int = rep(0, nrow(X_external)), object = NULL)
  })
    
  list(
    ate_est = result$ate,
    bias = if(is.nan(result$ate)) NaN else (result$ate - true_sate),
    weights_continuous = result$w_int,
    weights_external = result$w_int,
    energy_distance = 0
  )
}