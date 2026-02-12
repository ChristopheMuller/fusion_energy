# if (file.exists("renv/activate.R")) suppressMessages(source("renv/activate.R"))
suppressPackageStartupMessages(library(MatchIt))
suppressPackageStartupMessages(library(CBPS))

if (!requireNamespace("optmatch", quietly = TRUE)) {
  stop("Package 'optmatch' is required for full matching. Please install it.")
}

estimate_matchit_Full_CBPS <- function(X_treat, Y_treat, X_control_int, Y_control_int, X_external, Y_external, true_sate, n_external) {
  
  optmatch::setMaxProblemSize(Inf)

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
    
  result <- tryCatch({
    # Capture output to silence the optimizer
    capture.output(
      m.out <- matchit(as.formula(formula_str), 
                       data = df_combined,
                       method = "full",
                       distance = "cbps")
    )
    
    # Full matching produces weights for everyone
    all_weights <- m.out$weights
    
    # Extract weights
    w_tr <- all_weights[1:nrow(X_treat)]
    w_pool <- all_weights[(nrow(X_treat)+1):length(all_weights)]
    
    # Weighted Means
    y_tr_mean <- sum(Y_treat * w_tr) / sum(w_tr)
    y_pool_mean <- sum(Y_pool * w_pool) / sum(w_pool)
    
    ate_est <- y_tr_mean - y_pool_mean
    
    # Extract external specific weights for return
    n_int <- nrow(X_control_int)
    w_ext <- w_pool[(n_int + 1):length(w_pool)]
    
    list(ate = ate_est, w_ext = w_ext, object = m.out)
    
  }, error = function(e) {
    warning(paste("MatchIt Full failed:", e$message))
    list(ate = NaN, w_ext = rep(0, nrow(X_external)), object = NULL)
  })

  list(
    ate_est = result$ate,
    bias = if(is.nan(result$ate)) NaN else (result$ate - true_sate),
    weights_continuous = result$w_ext,
    weights_external = result$w_ext,
    energy_distance = 0
  )
}