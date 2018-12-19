# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: reduce_function.py 
@time: 2018/12/19 
"""
import numpy as np
from sklearn import manifold
from typing import List

__all__ = ["ae", "tsne", "lle", "ltsa", "le", "mds", "isomap"]

cpu_num = 4


def ae(features, embedding_matrix):
    average_embeddings = np.dot(features, embedding_matrix)
    return average_embeddings


def pca(features: List[List], dim):
    pass


def tsne(featrues: List[List], dim):
    featrues = np.asarray(featrues, dtype=np.float32)
    Y = manifold.TSNE(n_components=dim, init="pca", random_state=0, method="exact", verbose=2).fit_transform(
        featrues)
    return Y


def isomap(featrues: List[List], dim):
    featrues = np.asarray(featrues, dtype=np.float32)
    Y = manifold.Isomap(n_components=dim, n_neighbors=3, n_jobs=cpu_num).fit_transform(
        featrues)
    return Y


def lle(featrues: List[List], dim, method="standard"):
    featrues = np.asarray(featrues, dtype=np.float32)
    Y = manifold.LocallyLinearEmbedding(n_neighbors=3, n_components=dim, method=method,
                                        eigen_solver="auto", n_jobs=cpu_num).fit_transform(
        featrues)
    return Y


def mds(featrues: List[List], dim):
    featrues = np.asarray(featrues, dtype=np.float32)
    Y = manifold.MDS(n_components=dim, max_iter=100, n_init=1, n_jobs=cpu_num).fit_transform(
        featrues)
    return Y


def le(featrues: List[List], dim):
    featrues = np.asarray(featrues, dtype=np.float32)
    Y = manifold.SpectralEmbedding(n_components=dim, n_jobs=cpu_num).fit_transform(
        featrues)
    return Y


def ltsa(featrues: List[List], dim):
    method = "ltsa"
    featrues = np.asarray(featrues, dtype=np.float32)
    Y = manifold.LocallyLinearEmbedding(n_components=dim, method=method,
                                        eigen_solver="auto", n_jobs=cpu_num).fit_transform(
        featrues)
    return Y


if __name__ == "__main__":
    import pickle
    import sys
    import time

    sys.path.append("..")
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    with open("../temp/embedding.npy", "rb") as f:
        embedding_matrix = pickle.load(f)

    with open("../temp/features.npy", "rb") as f:
        features = pickle.load(f)

    dim = 2
    n_clusters = 10
    fig = plt.figure(figsize=(15, 8))

    # y_ae = ae(features, embedding_matrix)

    t0 = time.time()
    y_le = le(features, dim)
    t1 = time.time()
    ax = fig.add_subplot(2, 3, 1)
    plt.scatter(y_le[:, 0], y_le[:, 1])
    plt.title("le (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time.time()
    y_mds = mds(features, dim)
    t1 = time.time()
    ax = fig.add_subplot(2, 3, 2)
    plt.scatter(y_mds[:, 0], y_mds[:, 1])
    plt.title("mds (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time.time()
    y_isomap = isomap(features, dim)
    t1 = time.time()
    ax = fig.add_subplot(2, 3, 3)
    plt.scatter(y_isomap[:, 0], y_isomap[:, 1])
    plt.title("isomap (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time.time()
    y_lle = lle(features, dim)
    t1 = time.time()
    ax = fig.add_subplot(2, 3, 4)
    plt.scatter(y_lle[:, 0], y_lle[:, 1])
    plt.title("lle (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time.time()
    y_tsne = tsne(features, dim)
    t1 = time.time()
    ax = fig.add_subplot(2, 3, 5)
    plt.scatter(y_tsne[:, 0], y_tsne[:, 1])
    plt.title("tsne (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()

    km = KMeans(n_clusters=n_clusters, n_jobs=1)
    # km.fit_predict()
