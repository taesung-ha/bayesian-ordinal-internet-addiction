data {
  int<lower=1> N;
  int<lower=1> G;
  int<lower=1> K_shared;
  int<lower=1> K_act;
  array[N] int<lower=1, upper=G> g;
  // SII categories encoded as 0..3; use (y+1) in likelihood
  array[N] int<lower=0, upper=3> y;
  vector[N] z_paq;
  matrix[N, K_shared] X_shared;
  matrix[N, K_act] X_act;
}
parameters {
  real mu_alpha;
  real<lower=0> tau_alpha;
  vector[G] alpha_raw;
  real mu_paq;
  real<lower=0> tau_paq;
  vector[G] beta_paq_raw;
  vector[K_shared] beta_shared;
  vector[K_act] beta_act;
  ordered[3] c;
}
transformed parameters {
  vector[G] alpha = mu_alpha + tau_alpha * alpha_raw;
  vector[G] beta_paq = mu_paq + tau_paq * beta_paq_raw;
}
model {
  alpha_raw ~ normal(0,1);
  beta_paq_raw ~ normal(0,1);
  mu_alpha ~ normal(0,1);
  mu_paq ~ normal(0,1);
  tau_alpha ~ normal(0,0.5);
  tau_paq ~ normal(0,0.5);
  beta_shared ~ normal(0,0.5);
  beta_act ~ normal(0,0.5);
  c ~ normal([-1, 0, 1], 1);
  for (n in 1:N) {
    real eta = alpha[g[n]] + beta_paq[g[n]] * z_paq[n]
               + dot_product(X_shared[n], beta_shared)
               + dot_product(X_act[n], beta_act);
    (y[n] + 1) ~ ordered_logistic(eta, c);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real eta = alpha[g[n]] + beta_paq[g[n]] * z_paq[n]
               + dot_product(X_shared[n], beta_shared)
               + dot_product(X_act[n], beta_act);
    log_lik[n] = ordered_logistic_lpmf(y[n] + 1 | eta, c);
  }
}
