# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: kmeans.py 
@time: 2018/12/17 
"""
from sklearn.cluster import KMeans
from sklearn import metrics


def kmeans(features, n_cluster=2000, n_jobs=2):
    km = KMeans(n_clusters=n_cluster, n_jobs=n_jobs)
    f_labels = km.fit_predict(features)
    ch_score = metrics.calinski_harabaz_score((features, f_labels))
    return f_labels, ch_score
