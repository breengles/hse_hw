import torch
from  torch.utils.data import DataLoader, Dataset
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
import pickle


ds = torch.load("dataset/40000000.pkl")

X = ds[:][0].numpy()
y = ds[:][1].numpy()

# dl = DataLoader(ds, batch_size=2, shuffle=True)
# for b in dl:
#     print(b[:][1])
#     quit()

reg = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=500, max_depth=5, verbose=1), n_jobs=1)

reg.fit(X, y)

print(np.mean((reg.predict(X) - y) ** 2, axis=0))

with open("model.pkl", "wb") as m:
    pickle.dump(reg, m)
