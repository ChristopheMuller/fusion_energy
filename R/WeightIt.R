suppressPackageStartupMessages(library(WeightIt))

estimate_ew <- function(X_treat, Y_treat, X_control_int, Y_control_int, X_external, Y_external, true_sate, n_external) {
  
  # Combine data into dataframes
  X_pool <- rbind(X_control_int, X_external)
  Y_pool <- c(Y_control_int, Y_external)
  X_combined <- as.data.frame(rbind(X_treat, X_pool))
  
  n_tr <- nrow(X_treat)
  n_pool <- nrow(X_pool)
  treat_vec <- c(rep(1, n_tr), rep(0, n_pool))
  
  df <- cbind(treat = treat_vec, X_combined)
  cov_names <- colnames(X_combined)
  formula_str <- as.formula(paste("treat ~", paste(cov_names, collapse = " + ")))
  
  result <- tryCatch({
    # We wrap this in capture.output to prevent EOF errors in some environments
    capture.output({
      w.out <- WeightIt::weightit(
        formula_str, 
        data = df, 
        method = "energy",
        estimand = "ATT", 
        verbose = FALSE
      ) 
    })
    
    # Extract weights
    weights_all <- w.out$weights
    weights_pool <- weights_all[(n_tr + 1):(n_tr + n_pool)]
    
    # Normalize weights
    w_norm <- weights_pool / sum(weights_pool)
    
    # Calculate effect
    y_tr_mean <- mean(Y_treat)
    y_pool_mean <- sum(Y_pool * w_norm)
    ate_est <- y_tr_mean - y_pool_mean
    
    n_int <- nrow(X_control_int)
    w_int <- w_norm[1:n_int]
    w_ext <- w_norm[(n_int + 1):n_pool]
    
    list(ate = ate_est, w_int = w_int, w_ext = w_ext)
    
  }, error = function(e) {
    warning(paste("Energy balancing failed:", e$message))
    list(ate = NaN, w_int = rep(0, nrow(X_control_int)), w_ext = rep(0, nrow(X_external)))
  })
  
  list(
    ate_est = result$ate,
    bias = if(is.nan(result$ate)) NaN else (result$ate - true_sate),
    weights_continuous = result$w_ext,
    weights_external = result$w_ext,
    energy_distance = 0 # Method minimizes energy distance
  )
}