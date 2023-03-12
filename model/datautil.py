import pandas as pd
import numpy as np


def read_met(file_id):
    df = pd.read_csv(f'Data/{file_id}.csv', index_col=0, parse_dates=True)
    return df.interpolate()


def read_swat(file_id):
    df = pd.read_csv(f'Data/{file_id}.csv', index_col=0, parse_dates=True)
    # update select paras
    return df


def read_cod(file_id):
    df = pd.read_csv(f'Data/{file_id}.csv', index_col=0, parse_dates=True)
    ser = df['mean']
    return ser


def transform(df, log=False):
    if log:
        arr = np.log(df.to_numpy())
    else:
        arr = df.to_numpy()
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, ddof=1, axis=0)
    return arr, means, stds


def trans_obs(ser):
    arr = np.array([ser.to_numpy()]).T
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, ddof=1, axis=0)
    return arr, means, stds


def reshape(x, y, seq_len):
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_len + 1, seq_len, num_features))
    y_new = np.zeros((num_samples - seq_len + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_len, :]
        y_new[i, :] = y[i + seq_len - 1, 0]

    return x_new, y_new
