import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm


n_preds = 2
n_preys = 5
pred_state_dim = 4
prey_state_dim = 5


class Net(nn.Module):
    def __init__(self, outdim=7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_preds * pred_state_dim + n_preys * prey_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, outdim),
        )
    
    def forward(self, state):
        return torch.tanh(self.model(state))
    

def train(ds, num_epoch=1000, batch_size=256, saverate=10, device="cpu", path=""):
    net = Net()
    if path:
        net.load_state_dict(torch.load(path, map_location=device))
        net.train()
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-2)
    
    # net_pred = Net(2)
    # net_prey = Net(5)
    # net_pred.to(device)
    # net_prey.to(device)
    # optim_pred = torch.optim.Adam(net_pred.parameters(), lr=4e-3)
    # optim_prey = torch.optim.Adam(net_prey.parameters(), lr=4e-3)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

    for t in trange(num_epoch):
        for X, y in dl:
            X_ = X.to(device)
            y_ = y.to(device)
            
            loss = F.mse_loss(net(X_), y_)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # loss_pred = F.mse_loss(net_pred(X_), y_[:, :2])
            # loss_prey = F.mse_loss(net_prey(X_), y_[:, 2:])
            # optim_pred.zero_grad()
            # optim_prey.zero_grad()
            # loss_pred.backward()
            # loss_prey.backward()
            # optim_pred.step()
            # optim_prey.step()
            
        if (t + 1) % saverate == 0:
            with torch.no_grad():
                tqdm.write(f"Epoch {t + 1:>5d} | Loss {loss:0.5f}")
                torch.save(net.state_dict(), f"nn/{t + 1}.pt")
                
                # tqdm.write(f"Epoch {t + 1:>5d} | Pred loss {loss_pred:0.5f} | Prey loss {loss_prey:0.5f}")
                # torch.save(net_pred.state_dict(), f"nn/pred_{t + 1}.pt")
                # torch.save(net_prey.state_dict(), f"nn/prey_{t + 1}.pt")


if __name__ == "__main__":
    ds = torch.load("dataset/14136792.pkl")
    train(ds, device="cuda", batch_size=65536 * 8, saverate=1)
