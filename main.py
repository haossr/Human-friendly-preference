from exp import LinearRandomDataset 
import numpy as np
import pystan
import fire

def main(iteration=1000, M=10):
    exp = LinearRandomDataset()
    
    data = exp.sample_full(M)
    groundtruth = exp.get_groundtruth()
    sm = pystan.StanModel(file="stan/baseline.stan")
    
    fit = sm.sampling(data=data, iter=iteration, chains=4)

    Umax_ind_hat = fit.extract()['U'].argmax(1)
    Umax_ind = groundtruth['U'].argmax()

    correct = np.sum(Umax_ind_hat == Umax_ind)
    print(correct.shape)
    print(f"Out model hit the maximum [{correct} / {N}] = {correct/N:.2f}") 


if __name__ == "__main__":
    fire.Fire(main)
    
