import numpy as np
from sklearn.linear_model import LinearRegression
def samedirection(v1, v2):
    angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if np.isclose(angle, 1, atol=0.01):
        return True
    elif np.isclose(angle, -1, atol=0.01):
        return False
    else:
        raise ValueError("Vectors are not orthogonal, value:{}".format(angle))
def qr_tempadjust(a, t, td=1.):
    l = np.linalg.norm(t)
    c = np.concatenate([t[:, None], a], 1)
    q, r = np.linalg.qr(c)
    if not samedirection(t, q[:,0]):
        q = -q
    q[:,0] += td/l
    at = np.dot(q, r)
    return at[:, 0], at[:, 1:]

def adj_r2(X, y):
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    return 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1), r2