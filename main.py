import numpy as np
import pystan
import fire

from exp import LinearRandomDataset, Experiment


# iteration 250, sigma 0.01, 1-100
def main(iteration=1000, M=10):
    dataset = LinearRandomDataset(sigma=0.01)
    #print(dataset.sample_partial(50, 12))
    exp = Experiment(dataset, 
                     stan_file_full="stan/baseline.stan", 
                     stan_file_partial="stan/partial.stan",
                     M = [5*i for i in range(1,20)],
                     iteration=iteration)
    results_partial = exp.experiment_partial()
    results_full = exp.experiment_full()
    print(results_partial)
    print(results_full)
    
    #print(f"Top-1 Recall: Mean {top1.mean():.4f}, SD {top1.std():.4f}")
    #print(f"Top-5 Recall: Mean {top5.mean():.4f}, SD {top5.std():.4f}")
    #print(f"Correlation: Mean {cor.mean():.4f}, SD {cor.std():.4f}")
    #print(f"Regret: Mean {reg.mean():.4f}, SD {reg.std():.4f}")


if __name__ == "__main__":
    fire.Fire(main)
    
