import pystan
import pandas as pd

from eval import top_k_recall, correlation, regret, all_metrics


class Experiment:
    def __init__(self, dataset,
                 stan_file_full,
                 stan_file_partial,
                 M=None,
                 B=50,
                 R=10,
                 iteration=200,
                 **kwargs):
        self._dataset = dataset
        self._sm_full = pystan.StanModel(file=stan_file_full)
        self._sm_partial = pystan.StanModel(file=stan_file_partial)
        self._groundtruth = self._dataset.get_groundtruth()
        if M is None:
            M = [2, 5, 10, 20, 40, 100, 200, 400, 1000]
        self._M = M
        self._B = B
        self._R = R
        self._iteration = iteration

    def experiment_full(self):
        results = None 
        for m in self._M:
            for _ in range(self._R):
                comparisons = self._dataset.sample_full(m)
                fit = self._sm_full.sampling(data=comparisons,
                                             iter=self._iteration,
                                             pars=['U'],
                                             sample_file="outputs/baseline.out")
                U = self._groundtruth['U']
                Uhat = fit.extract()['U']
                result = all_metrics(Uhat, U)
                result['M'] = m
                results = pd.concat([results, result])
        return results

    def experiment_partial(self):
        results = None 
        for m in self._M:
            for _ in range(self._R):
                result = {"M": m}
                comparisons = self._dataset.sample_partial(m, self._B)
                fit = self._sm_partial.sampling(data=comparisons,
                                                iter=self._iteration,
                                                pars=['U'],
                                                sample_file="outputs/partial.out")
                U = self._groundtruth['U']
                Uhat = fit.extract()['U']
                result = all_metrics(Uhat, U)
                result['M'] = m
                results = pd.concat([results, result])
        return results
