data {
    int<lower=0> N; // number of samples
    int<lower=0> F; // number of attributes
    int<lower=0> B; // number of samples in a comparsion batch
    int<lower=0> M; // number of comparsions

    matrix[N, F] X; // all samples
    matrix[M, B, F] C; // Comparisons  
    
    matrix[M] D_l; // first comparison dimension
    matrix[M] D_k; // second comparison dimension    

    real sigma; 
    int C_y[M*(B-1)]; // outcome
}

parameters {
    vector[F] beta; // parameter of Utility
    vector[M, B-1] eta; // noise per comparison
}

transformed parameters {
    vector[M, B-1] Udiff;
    vector[N] U;
    int d_l;
    int d_k; 
    
    for(i in 1:M) {
        d_l = D_l[i]
        d_k = D_k[i]
        x_best = C[i, y[i]] 
        for(j in 1:(B-1)) {
            if (j != y[i]) {
                x_other = C[i, j]
                Udiff[i, j] = (x_best[d_l] - x_other[d_l]) * beta[d_l] + (x_best[d_k] - x_other[d_k]) * beta[d_k] + sigma * eta;
            }
        }

    U = X * beta;
}

model {
    beta ~ normal(0, 1);
    for(i in 1:M) {
        C_y[i] ~ bernoulli(Phi(Udiff[i]));
    }
}
