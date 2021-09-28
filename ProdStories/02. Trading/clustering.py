import tslearn
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.preprocessing import TimeSeriesResampler
import matplotlib.pyplot as plt
import numpy as np


def KMeans(X_train, n_clusters, metric, n_init=1, seed=None):
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=seed, n_init=n_init)
    y_pred = km.fit_predict(X_train)
    return km, y_pred


def MyKernelKMeans(X_train, n_clusters, kernel, n_init=1, seed=None):
    km = KernelKMeans(n_clusters=n_clusters, kernel=kernel, n_init=n_init, random_state=seed)
    y_pred = km.fit_predict(X_train)
    return km, y_pred


def visualize_kernel_kmeans(km, X_train, y_pred=None, title=None):
    if y_pred is None:
        y_pred = km.predict(X_train)

    plt.figure(figsize=(20, 20))

    n = km.n_clusters

    n_subplots = int(np.ceil(np.sqrt(n)))

    for yi in range(n):
        plt.subplot(n_subplots, n_subplots, 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=0.2)
        plt.text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title(title)

    plt.show()


def visualize_clusters(km, X_train, y_pred=None, title=None):
    if y_pred is None:
        y_pred = km.predict(X_train)

    plt.figure(figsize=(20, 20))

    n = km.n_clusters

    n_subplots = int(np.ceil(np.sqrt(n)))

    for yi in range(n):
        plt.subplot(n_subplots, n_subplots, 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=0.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title(title)

    plt.show()
