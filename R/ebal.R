# if (file.exists("renv/activate.R")) source("renv/activate.R")
suppressPackageStartupMessages(library(ebal))

estimate_ebal <- function(X_treat, Y_treat, X_control_int, Y_control_int, X_external, Y_external, true_sate, n_external) {
  
  X_pool <- rbind(X_control_int, X_external)
  Y_pool <- c(Y_control_int, Y_external)
  X_combined <- rbind(X_treat, X_pool)
  
  n_tr <- nrow(X_treat)
  n_pool <- nrow(X_pool)
  treat_vec <- c(rep(1, n_tr), rep(0, n_pool))
  
  result <- tryCatch({
    X_enhanced <- ebal::matrixmaker(as.matrix(X_combined))
    capture.output(
      eb.out <- ebal::ebalance(Treatment = treat_vec, X = X_enhanced, print.level = 0)
    )    
    weights_pool <- eb.out$w
    
    w_norm <- weights_pool / sum(weights_pool)

    y_tr_mean <- mean(Y_treat)
    y_pool_mean <- sum(Y_pool * w_norm)
    
    ate_est <- y_tr_mean - y_pool_mean
    
    n_int <- nrow(X_control_int)
    
    w_int <- w_norm[1:n_int]
    w_ext <- w_norm[(n_int + 1):n_pool]
    
    list(ate = ate_est, w_int = w_int, w_ext = w_ext)
    
  }, error = function(e) {
    warning(paste("Entropy balancing failed:", e$message))
    list(ate = NaN, w_int = rep(0, nrow(X_control_int)), w_ext = rep(0, nrow(X_external)))
  })
  
  list(
    ate_est = result$ate,
    bias = if(is.nan(result$ate)) NaN else (result$ate - true_sate),
    weights_continuous = result$w_ext,
    weights_external = result$w_ext,
    energy_distance = 0
  )
}