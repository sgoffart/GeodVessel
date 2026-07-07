import numpy as np

def sample(centerline, vessel):
    pos = np.argwhere(centerline>0)
    neg = np.argwhere((vessel>0)&(centerline==0))

    n = len(pos)
    idx = np.random.choice(len(neg), size=min(len(neg),4*n), replace=False)
    neg = neg[idx]

    pts = np.vstack([pos,neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    return pts, labels
