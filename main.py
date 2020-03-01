import numpy as np
import pystan
import fire

from data import LinearRandomDataset
from exp import Experiment


# iteration 250, sigma 0.01, 1-100
def main(iteration=1000, 
         N=100,
         M=None,
         F=8,
         B=50,
         sigma=0.01):
    if M is None:
        M = [5]

    dataset = LinearRandomDataset(N=N,
                                  sigma=sigma,
                                  F=F)
    exp = Experiment(dataset,
                     stan_file_full="stan/baseline.stan",
                     stan_file_partial="stan/partial.stan",
                     M=M,
                     B=B,
                     iteration=iteration)
    results_partial = exp.experiment_partial()
    results_full = exp.experiment_full()
    results_partial.to_csv("outputs/partial.csv", index=None)
    results_full.to_csv("outputs/full.csv", index=None)
    print("===========\tResult of the [Partial Comparisons]\t==========")
    print(results_partial)
    print("===========\tResult of the [Full Comparisons]\t==========")
    print(results_full)


if __name__ == "__main__":
    fire.Fire(main)
