import numpy as np

def flatten(s):
    if isinstance(s, int):
        return (s,)
    if s == ():
        return s
    if isinstance(s[0], tuple):
        return(flatten(s[0]) + flatten(s[1:]))
    return(s[:1] + flatten(s[1:]))


def ultra_fill(ds, k, queue, ultra):
    for i in flatten(queue[0]):
        for j in flatten(queue[1]):
            ultra[i][j] = ultra[j][i] = ds[k]


def unpack(s):
    way = []
    def wrapper(s):
        if isinstance(s, tuple):
            way.append(s)
            if isinstance(s[0], tuple):
                wrapper(s[0])
            if isinstance(s[1], tuple):
                wrapper(s[1])
        return way
    return wrapper(s)


def count_clusters(q):
    if isinstance(flatten(q[0]), tuple):
        left = len(flatten(q[0]))
    else:
        left = 1

    if isinstance(flatten(q[1]), tuple):
        right = len(flatten(q[1]))
    else:
        right = 1

    full = len(flatten(q))

    return left, right, full


def ultramatrix(clusters, ds):
    dim = len(flatten(clusters))
    ultra = np.zeros((dim, dim))

    for k, clsr in enumerate(unpack(clusters)):
        ultra_fill(ds, len(ds) - 1 - k, clsr, ultra)

    return ultra