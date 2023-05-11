import numpy as np
from sklearn.linear_model import LinearRegression

def samedirection(v1, v2, atol=0.05):
    angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if np.isclose(angle, 1, atol=atol):
        return True
    elif np.isclose(angle, -1, atol=atol):
        return False
    else:
        raise ValueError("Vectors are not orthogonal, value:{}".format(angle))


def align(src, ref):
    for i in range(src.shape[1]):
        if not samedirection(src[:, i], ref[:, i], 1.):
            src[:, i] = -src[:, i]
    return src
def qr_tempadjust(a, t, td=1.):
    l = np.linalg.norm(t)
    c = np.concatenate([t[:, None], a], 1)
    q, r = np.linalg.qr(c)
    if samedirection(t, q[:,0]):
        q[:,0] = q[:,0] + td/l
    else:
        q[:,0] = q[:,0] - td/l

    at = np.dot(q, r)
    at = align(at, c)
    # if not samedirection(t, at[:,0]):
    #     at = -at
    return at[:, 0], at[:, 1:]

# def qr_tempadjust_tp(a, t, p, dt, dp):
#     t_norm = np.linalg.norm(t)
#     p_norm = np.linalg.norm(p)
#     tp = np.concatenate([t[:, None], p[:, None]], 1)
#     q, r = np.linalg.qr(tp)
#     if not samedirection(t, q[:,0]):
#         q = -q
#     c = np.concatenate([q, a], 1)
#     q2, r2 = np.linalg.qr(c)
#     if not samedirection(q[:,0], q2[:,0]):
#         q2 = -q2
#     assert samedirection(q[:,1], q2[:,1])
#     q2[:,0] += dt/t_norm
#     q2[:,1] += dp/p_norm
#     at2 = np.dot(q2, r2)
#     at = np.dot(at2[:, :2], r)
#     pass


def qr_tempadjust_tp(a, t, p, dt, dp):
    t_norm = np.linalg.norm(t)
    p_norm = np.linalg.norm(p)
    tp = np.concatenate([t[:, None], p[:, None]], 1)
    q, r = np.linalg.qr(tp)
    if not samedirection(t, q[:,0]):
        q = -q

    q[:,0] += dt/t_norm
    at = np.dot(q, r)
    if not samedirection(t, at[:,0]):
        at = -at
    dp_resi = dp - (at[:,1] - p).mean()


    c = np.concatenate([q, a], 1)
    q2, r2 = np.linalg.qr(c)
    if not samedirection(q[:,0], q2[:,0]):
        q2 = -q2


    # assert samedirection(q[:,1], q2[:,1])
    q2[:,0] += dp/t_norm
    # q2[:,1] += dp/p_norm
    at2 = np.dot(q2, r2)
    at = np.dot(at2[:, :2], r)
    pass


def adj_r2(X, y):
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    return 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1), r2

if __name__ == "__main__":
    a = np.random.randint(0, 10, (10, 3)).astype(float)
    t = np.random.randint(0, 10, 10).astype(float)
    p = t + np.random.randint(0, 10, 10).astype(float)
    qr_tempadjust_tp(a, t, p, 1., 2.)
    adj_r2(a, t)