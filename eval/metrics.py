import numpy as np
from scipy.stats.stats import pearsonr


"""
yhat: N-d array of (iteration, number of samples)
y:    array of (number of samples)
"""
def top_k_recall(yhat, y, k=5):
    top_index = y.argmax()
    top_indices_predict = np.argpartition(yhat, -k, axis=1)[:,-k:]
    return np.max((top_indices_predict==top_index), 1)

def correlation(yhat, y):
    return np.array([pearsonr(yh, y) for yh in yhat])

def regret(yhat, y):
    y_best = np.max(y)
    ind_predict = np.argmax(yhat, axis=1)
    y_best_predict = y[ind_predict]
    return y_best - y_best_predict
    
def all_metrics(Uhat, U):
    result = {}
    top1 = top_k_recall(Uhat, U, 1)
    top5 = top_k_recall(Uhat, U, 5)
    cor = correlation(Uhat, U)
    reg = regret(Uhat, U)

    result['top1_mean'] = top1.mean()
    result['top1_sd'] = top1.std()
    result['top5_mean'] = top5.mean()
    result['top5_sd'] = top5.std()
   
    result['cor_mean'] = cor.mean()
    result['cor_sd'] = cor.std()
    result['reg_mean'] = reg.mean()
    result['reg_sd'] = reg.std()
    
    result['sample_size'] = Uhat.shape[0]
    
    return result
