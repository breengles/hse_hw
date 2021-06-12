import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np

n_preds = 2
n_preys = 5
pred_state_dim = 4
prey_state_dim = 5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_preds * pred_state_dim + n_preys * prey_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )
    
    def forward(self, state):
        return torch.tanh(self.model(state))
    

def train(ds, num_epoch=1000, batch_size=256, saverate=10, device="cpu"):
    net = Net()
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-2)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # X = torch.tensor(ds[:][0], dtype=torch.float, device=device)
    # y = torch.tensor(ds[:][1], dtype=torch.float, device=device)

    for t in trange(num_epoch):
        for X, y in tqdm(dl, leave=False):
            X_ = X.clone().detach().type(torch.float).to(device)
            y_ = y.clone().detach().type(torch.float).to(device)
            loss = F.mse_loss(net(X_), y_)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        if (t + 1) % saverate == 0:
            with torch.no_grad():
                tqdm.write(f"loss @ {t + 1} epoch: {loss}")
                torch.save(net.state_dict(), f"nn/{t + 1}.pt")


if __name__ == "__main__":
    ds = torch.load("dataset/60000000.pkl")
    train(ds, device="cuda", batch_size=65536 * 16, saverate=1)
