import numpy as np

def euclidean_distance(p1, p2, spacing=(1,1,1)):
    p1 = np.array(p1)
    p2 = np.array(p2)
    spacing = np.array(spacing)
    return float(np.linalg.norm((p1 - p2) * spacing))

def cosine_similarity(a, b, eps=1e-8):
    a = np.array(a)
    b = np.array(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
