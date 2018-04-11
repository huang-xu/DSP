import numpy as np


def metrics(y, yhat, epsilon=0.001):
    if type(y) is not np.ndarray:
        y = np.array(y)
    if type(yhat) is not np.ndarray:
        yhat = np.array(yhat)
    rmse = np.sqrt(np.nanmean(np.square(y - yhat)))
    mape = np.nanmean(np.abs((yhat - y)/(y+epsilon)))
    return rmse, mape