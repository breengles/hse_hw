import torch
from  torch.utils.data import DataLoader, Dataset
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np


ds = torch.load("dataset.pkl")
dl = DataLoader(ds, batch_size=2, shuffle=True)

X = ds[:][0].numpy()
y = ds[:][1].numpy()

# for b in dl:
#     print(b[:][1])
#     quit()

reg = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', verbose=1), n_jobs=1)

reg.fit(X, y)

print(np.mean((reg.predict(X) - y) ** 2, axis=0))
