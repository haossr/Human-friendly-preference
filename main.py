from exp import Exp
import numpy as np
import pystan

if __name__ == "__main__":
    N = 1000
    exp = Exp(M=10)
    data = exp.gen_data()
    gt = exp.ground_truth()
    sm = pystan.StanModel(file="stan/baseline.stan")
    fit = sm.sampling(data=data, iter=N, chains=4)

    Umax_ind_hat = fit.extract()['U'].argmax(1)
    Umax_ind = gt['umax_ind']

    correct = np.sum(Umax_ind_hat == Umax_ind)
    print(f"Out model hit the maximum [{correct} / {N}] = {correct/N:.2f}") 

