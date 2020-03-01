import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr


"""
yhat: N-d array of (iteration, number of samples)
y:    array of (number of samples)
"""


def top_k_recall(yhat, y, k=5):
    top_index = y.argmax()
    top_indices_predict = np.argpartition(yhat, -k, axis=1)[:, -k:]
    return np.max((top_indices_predict == top_index), 1)


def correlation(yhat, y):
    return np.array([pearsonr(yh, y)[0] for yh in yhat]).reshape(-1)


def regret(yhat, y):
    y_best = np.max(y)
    ind_predict = np.argmax(yhat, axis=1)
    y_best_predict = y[ind_predict]
    return y_best - y_best_predict


def all_metrics(Uhat, U):
    #top1 = top_k_recall(Uhat, U, 1)
    #top5 = top_k_recall(Uhat, U, 5)
    #cor = correlation(Uhat, U)
    #reg = regret(Uhat, U)
    #print(top1.shape, top5.shape, cor.shape, reg.shape)
    return pd.DataFrame({
        "top1": top_k_recall(Uhat, U, 1),
        "top5": top_k_recall(Uhat, U, 5),
        "correlation": correlation(Uhat, U),
        "regret": regret(Uhat, U)})

