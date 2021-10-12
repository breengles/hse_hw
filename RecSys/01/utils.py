import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(R, model):
    R = R.toarray()
    auc1 = 0
    auc2 = 0

    for user, row in enumerate(R):
        if np.count_nonzero(row != 0) == 0:
            continue

        y_pred = model.predict_user(user)

        y_true = np.zeros(R.shape[1])
        y_true[row > 0] = 1

        auc1 += roc_auc_score(y_true, y_pred)

        pos_msk = row > 0
        neg_msk = ~row

        preds = model.I @ model.U[user]
        comp_matrix = preds[pos_msk].reshape(-1, 1) > preds[neg_msk]
        u_auc = np.count_nonzero(comp_matrix) / (comp_matrix.shape[0] * comp_matrix.shape[1])
        auc2 += u_auc

    return auc1 / R.shape[0], auc2 / R.shape[0]


def roc_auc_2(R, model):
    R = R.toarray()
    auc = 0
    for u_id, u in enumerate(R):
        if np.count_nonzero(u != 0) == 0:
            continue

        pos_mask = u > 0
        neg_mask = ~u

        preds = model.I @ model.U[u_id]
        comp_matrix = preds[pos_mask].reshape(-1, 1) > preds[neg_mask]
        u_auc = np.count_nonzero(comp_matrix) / (comp_matrix.shape[0] * comp_matrix.shape[1])

        auc += u_auc

    return auc / R.shape[0]
