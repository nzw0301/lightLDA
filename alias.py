import numpy as np


class AliasSampler(object):
    """
    alias table for arange topics.
    p: store probability np.ndarray,  [0.1, 0.5, 0.4]
    """
    def __init__(self, p: np.ndarray):
        self.build_table(p)

    def sample(self):
        u, k = np.modf(np.random.rand()*self._K)
        k = int(k)
        if u < self.v[k]:
            return k
        else:
            return self.a[k]

    def build_table(self, p: np.ndarray):
        self._K = len(p)
        p /= np.sum(p)
        self.a = np.zeros(self._K, dtype=np.uint32)
        self.v = np.array(self._K*p)

        L, S = [], []
        for k, vk in enumerate(self.v):
            if 1. <= vk:
                L.append(k)
            else:
                S.append(k)

        while len(L) > 0 and len(S) > 0:
            l = L.pop()
            s = S.pop()
            self.a[s] = l
            self.v[l] -= (1. - self.v[s])
            if 1. > self.v[l]:
                S.append(l)
            else:
                L.append(l)


class SparseAliasSampler(AliasSampler):
    """
    alias class for not arange topics (sparse)
    p: store probability np.ndarray,  [0.1, 0.5, 0.4]
    topics: np.array or list. [100, 20, 1000]
    """
    def __init__(self, p: np.ndarray, topics: np.ndarray):
        self.build_table(p, topics)

    def sample(self):
        u, k = np.modf(np.random.rand()*self._K)
        k = int(k)

        if u < self.v[k]:
            return self.topics[k]
        else:
            return self.a[k]

    def build_table(self, p: np.ndarray, topics: np.ndarray):
        self._K = len(p)
        p /= np.sum(p)

        self.a = np.zeros(self._K, dtype=np.uint32)
        self.v = np.array(self._K*p)
        self.topics = topics

        L, S = [], []

        for k, vk in enumerate(self.v):
            if 1. <= vk:
                L.append(k)
            else:
                S.append(k)

        while len(L) > 0 and len(S) > 0:
            l = L.pop()
            s = S.pop()
            self.a[s] = self.topics[l]
            self.v[l] -= (1. - self.v[s])
            if 1. > self.v[l]:
                S.append(l)
            else:
                L.append(l)
