if (file.exists("renv/activate.R")) source("renv/activate.R")

estimate_ate_r <- function(X_treat, Y_treat, X_control_int, Y_control_int, X_external, Y_external, true_sate, n_external) {
  
  ate_est <- mean(Y_treat) - mean(Y_control_int)
  
  w_cont <- rep(1 / length(Y_external), length(Y_external))
  w_ext <- rep(0, length(Y_external))
  
  list(
    ate_est = ate_est,
    bias = ate_est - true_sate,
    weights_continuous = w_cont,
    weights_external = w_ext,
    energy_distance = 0
  )
}