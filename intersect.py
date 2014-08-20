import numpy as np

def intersect(pos, dir, triangle):
    v0, v1, v2 = triangle
    m = np.vstack((v1-v0,v2-v0,-dir)).T
    det = np.linalg.det(m)

    if det == 0.0:
        return None

    b = pos-v0

    u1 = (np.cross(*m[:,[1,2]].T)*b).sum()/det
    if u1 < 0 or u1 > 1:
        return None

    u2 = (np.cross(*m[:,[2,0]].T)*b).sum()/det
    if u2 < 0 or u2 > 1:
        return None

    u3 = (np.cross(*m[:,[0,1]].T)*b).sum()/det
    if u3 <= 0 or (1-u1-u2) < 0:
        return None

    return u3
