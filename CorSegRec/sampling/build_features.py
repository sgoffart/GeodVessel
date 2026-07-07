import numpy as np
from utils.patch import build_feature

def build(volume, pts):
    X=[]
    for p in pts:
        f = build_feature(volume, tuple(p))
        if f is not None:
            X.append(f)
    return np.array(X)
