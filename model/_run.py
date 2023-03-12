import torch
import pandas as pd
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from model.dataset import Combined
from model.lstm import Model1
from model.train import train_epoch, eval_model
from model.nseloss import NSELoss
from model.calc_nse import calc_nse


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

met_id = 'meteo_hokota'
swat_id = 'swatout'
obs_id = 'day_mean_COD'
seq_len = 365
learning_rate = 1e-3

tr_start = pd.to_datetime("2012-06-01", format="%Y-%m-%d")
tr_end = pd.to_datetime("2016-05-31", format="%Y-%m-%d")
test_start = pd.to_datetime("2016-06-01", format="%Y-%m-%d")
test_end = pd.to_datetime("2019-05-31", format="%Y-%m-%d")

ds = Combined(met_id, swat_id, obs_id, seq_len=seq_len, dates=[tr_start, tr_end], period='train')
ds_test = Combined(met_id, swat_id, obs_id, seq_len=seq_len, dates=[test_start, test_end], period='test')

loss_func = NSELoss()

epochs = 10
split = KFold(n_splits=5, shuffle=True, random_state=1)

best_NSE = -100

for fold, (tr_idx, val_idx) in enumerate(split.split(ds)):
    tqdm.tqdm.write(f"Fold: {fold + 1}")

    tr_loader = DataLoader(ds, batch_size=256, sampler=SubsetRandomSampler(tr_idx))
    val_loader = DataLoader(ds, batch_size=256, sampler=SubsetRandomSampler(val_idx))

    model = Model1().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = NSELoss().to(DEVICE)

    for i in range(epochs):
        train_epoch(model, optimizer, tr_loader, loss_func, DEVICE, i + 1)

        obs, preds = eval_model(model, val_loader, DEVICE)

        preds = ds.rescale(preds.cpu().numpy(), variable='outputs')
        obs = ds.rescale(obs.cpu().numpy(), variable='outputs')
        nse = calc_nse(obs, preds)
        # Todo: save models
        if nse >= best_NSE:
            best = model.state_dict()
            best_NSE = nse

        tqdm.tqdm.write(f"Validation NSE: {nse:.2f}")


import matplotlib.pyplot as plt

test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)
model = Model1().to(DEVICE)
model.load_state_dict(best)

obs, preds = eval_model(model, test_loader, DEVICE)
preds = ds_test.rescale(preds.cpu().numpy(), variable='outputs')
obs = ds_test.rescale(obs.cpu().numpy(), variable='outputs')
nse = calc_nse(obs, preds)

#plotting
start_date = ds_test.dates[0]
end_date = ds_test.dates[1] + pd.DateOffset(days=1)
date_range = pd.date_range(start_date, end_date)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(date_range, obs, label="observation")
ax.plot(date_range, preds, label="prediction")
ax.legend()
ax.set_title(f"NSE: {nse:.3f}")
ax.xaxis.set_tick_params(rotation=90)
ax.set_xlabel("Date")
_ = ax.set_ylabel("COD")