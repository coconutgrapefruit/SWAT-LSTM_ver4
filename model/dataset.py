import pandas as pd
import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset
import model.datautil as datautil


class Combined(Dataset):
    def __init__(self,
                 met_id: str,
                 swat_id: str,
                 obs_id: str,
                 seq_len: int,
                 dates: List,
                 period: str
                 ):
        self.path = 'Data/'
        self.met_id = met_id
        self.swat_id = swat_id
        self.obs_id = obs_id
        self.seq_len = seq_len
        self.dates = dates
        self.period = period

        self.means = None
        self.stds = None

        self.x, self.y = self.load()
        self.num_samples = self.y.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def load(self):
        met = datautil.read_met(self.met_id)
        swat = datautil.read_swat(self.swat_id)
        obs = datautil.read_cod(self.obs_id)


        start_date = self.dates[0] - pd.DateOffset(days=self.seq_len)
        met = met[start_date:self.dates[1]]
        swat = swat[start_date:self.dates[1]]
        obs = obs[start_date:self.dates[1]]

        # to np
        x_met, means_met, stds_met = datautil.transform(met, log=False)
        x_swat, means_swat, stds_swat = datautil.transform(swat, log=True)
        y, means_y, stds_y = datautil.trans_obs(obs)

        x = np.concatenate((x_met, x_swat), axis=1)

        self.means = np.concatenate((means_met, means_swat)), means_y
        self.stds = np.concatenate((stds_met, stds_swat)), stds_y

        # to numpy for reshaping
        x = self.normalization(x, variable='inputs')
        x, y = datautil.reshape(x, y, self.seq_len)

        # delete nan
        if self.period == "train" and np.sum(np.isnan(y)) > 0:
            x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
            y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

        y = self.normalization(y, variable='outputs')

        # to tensor
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        return x, y

    def normalization(self, feature, variable):
        if variable == 'inputs':
            feature = (feature - self.means[0]) / self.stds[0]
        elif variable == 'outputs':
            feature = (feature - self.means[1]) / self.stds[1]
        return feature

    def rescale(self, feature, variable):
        if variable == 'inputs':
            feature = feature * self.stds[0] + self.means[0]
        elif variable == 'outputs':
            feature = feature * self.stds[1] + self.means[1]
        return feature

