import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from model import Net, SinglePred, SinglePrey
import numpy as np
from torch.optim.lr_scheduler import MultiplicativeLR


def get_val_loss(net_pred, net_prey, val, device="cpu"):
    net_pred.eval()
    net_prey.eval()
    losses_pred = []
    losses_prey = []
    with torch.no_grad():
        for X, y in tqdm(val, leave=False, desc="val"):
            X_ = X.to(device)
            y_ = y.to(device)
            for i in range(2):
                idxes = list(range(i * 4, (i + 1) * 4)) \
                    + list(range(2 * 4, 2 * 4 + 5 * 5))
                X_agent = X_[:, idxes]
                y_pred = net_pred(X_agent).squeeze(-1)
                
                assert y_pred.shape == y_[:, i].shape
                
                losses_pred.append(F.mse_loss(y_pred, y_[:, i]).cpu().numpy().item())
                
            for i in range(5):
                preds = list(range(0, 2 * 4))
                prey = list(range(2 * 4 + i * 5, 2 * 4 + (i + 1) * 5))
                idxes = preds + prey
                X_agent = X_[:, idxes]
                y_prey = net_prey(X_agent).squeeze(-1)
                
                assert y_prey.shape == y_[:, 2 + i].shape
                
                losses_prey.append(F.mse_loss(y_prey, y_[:, 2 + i]).cpu().numpy().item())
                
    net_pred.train()
    net_prey.train()
    return np.mean(losses_pred), np.mean(losses_prey)


def train(ds, num_epoch=1000, batch_size=256, saverate=10, device="cpu", pred_path="",
          prey_path=""):
    # net = Net()
    # if path:
    #     net.load_state_dict(torch.load(path, map_location=device))
    #     net.train()
    # net.to(device)
    # optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # net_pred = Net(2)
    # net_prey = Net(5)
    
        
    
    net_pred = SinglePred()
    net_prey = SinglePrey()    
    if pred_path:
        net_pred.load_state_dict(torch.load(pred_path, map_location=device))
        net_pred.train()
    if prey_path:
        net_prey.load_state_dict(torch.load(prey_path, map_location=device))
        net_prey.train()
    
    net_pred.to(device)
    net_prey.to(device)
    optim_pred = torch.optim.Adam(net_pred.parameters(), lr=1e-3)
    optim_prey = torch.optim.Adam(net_prey.parameters(), lr=1e-3)
    
    pred_scheduler = MultiplicativeLR(optim_pred, lambda x: 0.95)
    prey_scheduler = MultiplicativeLR(optim_prey, lambda x: 0.95)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True)

    for t in trange(num_epoch):
        for X, y in tqdm(dl, leave=False, desc="train"):
            pred_losses = []
            prey_losses = []
            
            X_ = X.to(device)
            y_ = y.to(device)
            
            # y_hat = net(X_)
            # loss = F.mse_loss(y_hat, y_)
            # optim.zero_grad()
            # loss.backward()
            # optim.step()
            for i in range(2):
                idxes = list(range(i * 4, (i + 1) * 4)) \
                    + list(range(2 * 4, 2 * 4 + 5 * 5))
                X_agent = X_[:, idxes]
                y_pred = net_pred(X_agent).squeeze(-1)
                
                assert y_pred.shape == y_[:, i].shape
                
                loss_pred = F.mse_loss(y_pred, y_[:, i])
                optim_pred.zero_grad()
                loss_pred.backward()
                optim_pred.step()
                
                with torch.no_grad():
                    pred_losses.append(loss_pred.cpu().numpy().item())
                
            for i in range(5):
                preds = list(range(0, 2 * 4))
                prey = list(range(2 * 4 + i * 5, 2 * 4 + (i + 1) * 5))
                idxes = preds + prey
                X_agent = X_[:, idxes]
                y_prey = net_prey(X_agent).squeeze(-1)
                
                assert y_prey.shape == y_[:, 2 + i].shape
                
                loss_prey = F.mse_loss(y_prey, y_[:, 2 + i])
                optim_prey.zero_grad()
                loss_prey.backward()
                optim_prey.step()
                
                with torch.no_grad():
                    prey_losses.append(loss_prey.cpu().numpy().item())
        
        # print("=" * 30)
        # print("Current pred lr:", pred_scheduler.get_lr())
        # print("Current prey lr:", prey_scheduler.get_lr())
        # print("=" * 30)
        
        pred_scheduler.step()
        prey_scheduler.step()
        
        if (t + 1) % saverate == 0:
            with torch.no_grad():
                # pred_val_loss, prey_val_loss = get_val_loss(net_pred, net_prey, val, device=device)
                pred_val_loss, prey_val_loss = 0, 0
                info = f"Losses @ {t + 1:>5d}: {np.mean(pred_losses):0.5f} ~ {pred_val_loss:0.5f}" \
                    f" | {np.mean(prey_losses):0.5f} ~ {prey_val_loss:0.5f}"
                tqdm.write(info)
                torch.save(net_pred.state_dict(), f"nn/oleg_pred_bn_{t + 1}.pt")
                torch.save(net_prey.state_dict(), f"nn/oleg_prey_bn_{t + 1}.pt")


if __name__ == "__main__":
    ds = torch.load("dataset/oleg_10081009.pkl")
    ds_val = torch.load("dataset/val_3368245.pkl")
    
    print("=== Check correctness of target action values ===")
    print("TRAIN: min = ", ds[:][-1].min(), "max = ", ds[:][-1].max())
    print("VAL: min = ", ds_val[:][-1].min(), "max = ", ds_val[:][-1].max())
    print()
    
    train(ds, device="cuda", batch_size=65536 * 8, saverate=1)
