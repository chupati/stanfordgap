import numpy
from sklearn.cluster import KMeans


class StanfordGap(object):
    def __init__(self, B=10, random_state=42):
        self.B = B
        self.random_state = random_state
        self.K = None
        self.s = None
        self.gap = None

    def w_k(self, x, labels, cluster_centers):
        """
        Calculate the within-dispersion measures
        :param x:
        :param labels:
        :param cluster_centers:
        :return:
        """
        d = numpy.square(numpy.linalg.norm(x - cluster_centers[labels], 2, 1))
        w = numpy.zeros((self.K, 1))
        for k in range(0, self.K):
            n_k = labels[labels == k].shape[0]
            w[k] = d[labels == k].sum() / n_k
        return w

    def w_star(self, x):
        """
        Calculate the within dispersion measures of B reference sets
        :param x:
        :return:
        """
        w_star = numpy.zeros((self.K, self.B))
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        for b in range(0, self.B):
            reference_set = numpy.random.uniform(low=x_min, high=x_max, size=x.shape)
            reference_cluster = KMeans(n_clusters=self.K)
            reference_cluster.fit(reference_set)
            w_star[:, b] = self.w_k(reference_set, reference_cluster.labels_, reference_cluster.cluster_centers_).ravel()
        return w_star

    def fit(self, X, labels, cluster_centers):
        """
        Calculate the Stanford Gap Statistic of the cluster
        :param X:
        :param labels:
        :param cluster_centers:
        :return:
        """
        x = numpy.array(X)
        self.K = numpy.unique(labels).shape[0]
        w = self.w_k(x, labels, cluster_centers)
        w_star = self.w_star(x)

        self.gap = numpy.log2(w_star.sum(axis=0)).mean() - numpy.log2(w.sum())
        l_bar = numpy.log2(w_star.sum(axis=0)).mean()
        sd_k = numpy.sqrt(numpy.square(numpy.log2(w_star.sum(axis=0)) - l_bar).mean())
        self.s = sd_k * numpy.sqrt((1 + 1 / self.B))

