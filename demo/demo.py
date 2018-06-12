from sg.StanfordGap import StanfordGap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


class StanfordGapDemo(object):
    def run(self):
        """
        Run the Stanford Gap Statistic Analysis on the iris data set presented in
        http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
        :return:
        """
        np.random.seed(42)
        iris = datasets.load_iris()
        X = iris.data

        gaps = np.zeros((20, 1))
        s = np.zeros((20, 1))
        for k in range(0, 20):
            est = KMeans(n_clusters=(k + 1))
            est.fit(X)
            sg = StanfordGap(B=10)
            sg.fit(X, est.labels_, est.cluster_centers_)
            gaps[k] = sg.gap
            s[k] = sg.s

        # Plot Gap(k)
        # Choose the smallest k such that Gap(k)>=Gap(k+1) - s_(k+1)
        plt.plot(gaps[0:18])
        plt.plot(gaps[1:19] - s[1:19])
        plt.legend(['Gap(k)', 'Gap(k+1) - s_k+1'])
        plt.xticks(np.arange(20), np.arange(1, 20))
        plt.xlabel('K')
        plt.show()
