data {
    int<lower=0> N; // number of samples
    int<lower=0> F; // number of attributes
    int<lower=0> B; // number of samples in a comparsion batch
    int<lower=0> M; // number of comparsions

    matrix[N, F] X; // all samples
    real C[M, B, 2, F]; // Comparisons  
    int D[M, 2]; // comparison dimension

    real sigma; 
    int C_y[M, B]; // outcome
}

parameters {
    vector[F] beta; // parameter of Utility
    real eta[M, B]; // noise per comparison
}

transformed parameters {
    matrix[M, B] dU;
    vector[N] U;
    
    for(i in 1:M) {
        for(j in 1:B) {
            dU[i, j] = (C[i,j,1,D[i,1]] - C[i,j,2,D[i,1]]) * beta[D[i,1]] + (C[i,j,1,D[i,2]] - C[i,j,2,D[i,2]]) * beta[D[i,2]] + sigma * eta[i,j];
        }
    }
    U = X * beta;
}

model {
    for (i in 1:M) {
        for (j in 1:B) {
            eta[i][j] ~ normal(0, 1);
        }   
    }
    beta ~ normal(0, 1); 
    for (i in 1:M) {
        for (j in 1:B) {
            C_y[i][j] ~ bernoulli(Phi_approx(dU[i][j]));
        }   
    }
}
