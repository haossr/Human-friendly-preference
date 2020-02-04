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
    vector[F] theta; // parameter of Utility
    vector[M] eta; // noise per comparison
}
transformed parameters {
    vector[M] Udiff;
    vector[N] U;
    Udiff = (C_x1 - C_x2) * theta + sigma * eta;
    U = X * theta;
}
model {
    theta ~ normal(0, 1);
    for(i in 1:M) {
        C_y[i] ~ bernoulli(Phi(Udiff[i]));
    }
}