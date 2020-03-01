import numpy as np
import pystan
import fire

from exp import LinearRandomDataset, Experiment


def main(iteration=15, M=10):
    dataset = LinearRandomDataset(sigma=0.01)
    exp = Experiment(dataset, 
                     stan_file_full="stan/baseline.stan", 
                     stan_file_partial="stan/partial.stan",
                     iteration=iteration)
    results = exp.experiment_full()
    print(results)
    
    #print(f"Top-1 Recall: Mean {top1.mean():.4f}, SD {top1.std():.4f}")
    #print(f"Top-5 Recall: Mean {top5.mean():.4f}, SD {top5.std():.4f}")
    #print(f"Correlation: Mean {cor.mean():.4f}, SD {cor.std():.4f}")
    #print(f"Regret: Mean {reg.mean():.4f}, SD {reg.std():.4f}")


if __name__ == "__main__":
    fire.Fire(main)
    
