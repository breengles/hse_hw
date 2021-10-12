import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(R, model):
    auc = 0

    for user, row in enumerate(R):
        if row.indices.shape[0] == 0:
            continue

        y_pred = model.predict_user(user)

        y_true = np.zeros(R.shape[1])
        y_true[row.indices] = 1

        auc += roc_auc_score(y_true, y_pred)
    return auc / R.shape[0]
