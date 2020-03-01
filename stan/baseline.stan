data {
    int<lower=0> N; // number of samples
    int<lower=0> F; // number of attributes
    int<lower=0> M; // number of comparsions
    matrix[N, F] X; // all samples
    matrix[M, F] C_x1; // first element of the comprisons 
    matrix[M, F] C_x2; // second element of the comprisons
    real sigma; 
    int C_y[M]; // outcome
}
parameters {
    vector[F] beta; // parameter of Utility
    vector[M] eta; // noise per comparison
}
transformed parameters {
    vector[M] dU;
    vector[N] U;
    dU = (C_x1 - C_x2) * beta + sigma * eta;
    U = X * beta;
}
model {
    eta ~ normal(0, 1);
    beta ~ normal(0, 1);
    C_y ~ bernoulli(Phi_approx(dU));
}
