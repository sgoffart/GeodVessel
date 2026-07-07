import numpy as np
from scipy.ndimage import zoom

def extract_patch(volume, center, size):
    z,y,x = center
    r = size//2
    z0,z1 = z-r, z+r+1
    y0,y1 = y-r, y+r+1
    x0,x1 = x-r, x+r+1

    if z0<0 or y0<0 or x0<0: return None
    if z1>volume.shape[0] or y1>volume.shape[1] or x1>volume.shape[2]: return None

    return volume[z0:z1,y0:y1,x0:x1]

def build_feature(volume, center, sizes=(15,7), out=11):
    feats = []
    for s in sizes:
        p = extract_patch(volume, center, s)
        if p is None: return None
        p = zoom(p, [out/s]*3, order=1)
        feats.append(p.flatten())
    return np.concatenate(feats)
