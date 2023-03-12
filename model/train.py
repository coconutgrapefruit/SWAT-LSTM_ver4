from typing import Tuple
import torch
import tqdm


def train_epoch(model, optimizer, loader, loss_func, device, epoch):
    model.train()
    pbar = tqdm.notebook.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    for xs, ys in pbar:
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_hat = model(xs)
        loss = loss_func(y_hat, ys)
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def eval_model(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    obs = []
    preds = []
    with torch.inference_mode():
        for xs, ys in loader:
            xs = xs.to(device)
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)

    return torch.cat(obs), torch.cat(preds)

