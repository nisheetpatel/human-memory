############################################################
# STAN MODELS
############################################################

LOGIT_DIFF_BIAS = """
    data {
        int<lower=1> N; // number of training samples
        int<lower=0> K; // number of predictors (4 SMs)
        int<lower=1> L; // number of subjects
        
        int<lower=1, upper=L> ll[N]; // subject id {1,...,L}
        row_vector<lower=0, upper=1>[K] ss[N]; // slot machine id indicator

        row_vector[K] X[N];         // predictors
        int<lower=0, upper=1> y[N]; // response
    }

    parameters {
        vector[K] beta[L];  // individual slope
        vector[K] mu_beta;  // Hierarchical mean for slope
        vector<lower=0>[K] sigma_beta; // h std for slope
        
        vector[K] alpha[L]; // individual intercept
        vector[K] mu_alpha;   // Hierarchical mean for intercept
        vector<lower=0>[K] sigma_alpha; // h std for intercept

        vector[L] alpha_0; // common intercept
        real mu_alpha_0;   // hierarchical mean for common intercept
        real<lower=0> sigma_alpha_0; // h std for common intercept
    }

    model {
        mu_beta ~ normal(0.25, 2);
        sigma_beta ~ cauchy(0, 1);
            
        mu_alpha ~ normal(0, 0.5);
        sigma_alpha ~ cauchy(0, 0.5);
        
        mu_alpha_0 ~ normal(0, 0.5);
        sigma_alpha_0 ~ cauchy(0, 0.5);

        for (l in 1:L) {
            beta[l] ~ normal(mu_beta, sigma_beta);
            alpha[l] ~ normal(mu_alpha, sigma_alpha);
            alpha_0[l] ~ normal(mu_alpha_0, sigma_alpha_0);
        }
        
        {
        vector[N] x_beta_ll;

        for (n in 1:N)
            x_beta_ll[n] = X[n] * beta[ll[n]] + ss[n] * alpha[ll[n]] + alpha_0[ll[n]];
        
        y ~ bernoulli_logit(x_beta_ll);
        }
    }
"""

LOGIT_COMMON_BIAS = """
    data {
        int<lower=1> N; // number of training samples
        int<lower=0> K; // number of predictors (4 SMs)
        int<lower=1> L; // number of subjects
        
        int<lower=1, upper=L> ll[N]; // subject id {1,...,L}
        row_vector<lower=0, upper=1>[K] ss[N]; // slot machine id indicator

        row_vector[K] X[N];         // predictors
        int<lower=0, upper=1> y[N]; // response
    }

    parameters {
        vector[K] beta[L];  // individual slope
        vector[K] mu_beta;  // Hierarchical mean for slope
        vector<lower=0>[K] sigma_beta; // h std for slope

        vector[L] alpha_0; // common intercept
        real mu_alpha_0;   // hierarchical mean for common intercept
        real<lower=0> sigma_alpha_0; // h std for common intercept
    }

    model {
        mu_beta ~ normal(0.25, 2);
        sigma_beta ~ cauchy(0, 1);
        
        mu_alpha_0 ~ normal(0, 0.25);
        sigma_alpha_0 ~ cauchy(0, 0.5);

        for (l in 1:L) {
            beta[l] ~ normal(mu_beta, sigma_beta);
            alpha_0[l] ~ normal(mu_alpha_0, sigma_alpha_0);
        }
        
        {
        vector[N] x_beta_ll;

        for (n in 1:N)
            x_beta_ll[n] = X[n] * beta[ll[n]] + alpha_0[ll[n]];
        
        y ~ bernoulli_logit(x_beta_ll);
        }
    }
"""
