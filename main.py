import numpy as np
import pystan
import fire

from data import LinearRandomDataset
from exp import Experiment


# iteration 250, sigma 0.01, 1-100
def main(iteration=1000, M=10):
    dataset = LinearRandomDataset(sigma=0.01)
    exp = Experiment(dataset,
                     stan_file_full="stan/baseline.stan",
                     stan_file_partial="stan/partial.stan",
                     M=[5 * i for i in range(1, 20)],
                     iteration=iteration)
    results_partial = exp.experiment_partial()
    results_full = exp.experiment_full()
    results_partial.to_csv("outputs/partial.csv")
    results_full.to_csv("outputs/full.csv")
    print(results_partial)
    print(results_full)


if __name__ == "__main__":
    fire.Fire(main)
