# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: stc.py 
@time: 2018/12/17 
"""
import pickle
import re
import gensim
import pyhanlp
from tqdm import tqdm
from sklearn.cluster import KMeans

sen_without_labels = pickle.load(open("without_label_sens.npy", "rb"))

pattern_num_comma = r"[^\u4E00-\u9FA5]+"
sen_without_labels_reg = []
for sen in tqdm(sen_without_labels, total=len(sen_without_labels)):
    sen = re.sub(pattern_num_comma, "", sen)
    if len(sen) == 0:
        continue
    sen_without_labels_reg.append(sen)
sen_without_labels = list(set(sen_without_labels_reg))
print(len(sen_without_labels))

hanlp = pyhanlp.HanLP
sen_seg_words = []
for sen in tqdm(sen_without_labels, total=len(sen_without_labels)):
    terms = hanlp.segment(sen)
    words = [term.word for term in terms]
    sen_seg_words.append(words)

dictionary = gensim.corpora.Dictionary(sen_seg_words)

ids = [tokenid for tokenid, freq in dictionary.dfs.items() if freq == 1]
dictionary.filter_tokens(ids)
dictionary.compactify()
corpus = [dictionary.doc2bow(sen) for sen in sen_seg_words]

tfidf = gensim.models.TfidfModel(corpus)
lda = gensim.models.LdaModel(corpus)

