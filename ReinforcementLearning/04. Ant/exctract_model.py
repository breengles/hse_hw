import torch
import sys


name = sys.argv[1]
model = torch.load(name, map_location="cpu").model
torch.save(model, f"{name}.model.pkl")
