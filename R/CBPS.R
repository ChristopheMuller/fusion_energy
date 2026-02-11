# if (file.exists("renv/activate.R")) source("renv/activate.R")
suppressPackageStartupMessages(library(CBPS))


estimate_ate_cbps <- function(X_treat, Y_treat, X_control_int, Y_control_int, X_external, Y_external, true_sate, n_external) {
  
  X_pool <- rbind(X_control_int, X_external)
  Y_pool <- c(Y_control_int, Y_external)
  
  X_combined <- rbind(X_treat, X_pool)
  Y_combined <- c(Y_treat, Y_pool)
  
  n_tr <- nrow(X_treat)
  n_pool <- nrow(X_pool)
  n_int <- nrow(X_control_int)
  
  treat_vec <- c(rep(1, n_tr), rep(0, n_pool))
  
  df_combined <- as.data.frame(X_combined)
  df_combined$T_assign <- treat_vec
  
  formula_str <- paste("T_assign ~", paste(colnames(df_combined)[1:(ncol(df_combined)-1)], collapse = " + "))
  
  start_time <- Sys.time()
  
  result <- tryCatch({
    cbps_fit <- CBPS(as.formula(formula_str), 
                     data = df_combined, 
                     ATT = 1,
                     standardize = TRUE)
    
    all_weights <- cbps_fit$weights
    
    weights_pool <- all_weights[(n_tr + 1):(n_tr + n_pool)]
    
    w_norm <- weights_pool / sum(weights_pool)
    
    y_tr_mean <- mean(Y_treat)
    y_pool_mean <- sum(Y_pool * w_norm)
    
    ate_est <- y_tr_mean - y_pool_mean
    
    w_int <- w_norm[1:n_int]
    w_ext <- w_norm[(n_int + 1):n_pool]
    
    list(ate = ate_est, w_int = w_int, w_ext = w_ext)
    
  }, error = function(e) {
    warning(paste("CBPS balancing failed:", e$message))
    list(ate = NaN, w_int = rep(0, n_int), w_ext = rep(0, nrow(X_external)))
  })
  
  end_time <- Sys.time()
  
  list(
    ate_est = result$ate,
    bias = if(is.nan(result$ate)) NaN else (result$ate - true_sate),
    weights_continuous = result$w_ext,
    weights_external = result$w_ext,
    energy_distance = 0
  )
}