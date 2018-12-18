# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: labeled_data_stat.py 
@time: 2018/12/14 
"""
import json
import re
import pickle
from typing import List
from tqdm import tqdm
from misc import sort_dict

his_count = 0  # 病例数量
sen_count = 0  # 句子数量
sen_with_labels_count = 0  # 有标签的句子数量
sen_without_labels_count = 0  # 没有标签的句子数量
label_stat = {}  # 标签统计
label_content_stat_tags = ["ZZ", "YXZZ", "BSZZ", "ZJY", "BSSM", "FRBSSM", "GMYW", "YB"]
label_content_stat = {}
[label_content_stat.update({key: {}}) for key in label_content_stat_tags]  # 标签内容统计

without_label_sens = []

# with open(r"D:\dr_easy_predata\temp\label_data.npy", "rb") as f:
#     result_labeled = pickle.load(f)
#     for json_index, json_object in tqdm(enumerate(result_labeled), total=len(result_labeled)):
#         ori_sen = json_object["xianbingshi_biaojihou"]
#         labels = json_object["xianbingshi_biaojihou_labeled"]
#         his_count += 1
#         sen_count += len(labels)
#         sen_with_labels_count += len([label for label in labels if len(label[1]) != 0])
#         with_labels_sen = [label[1] for label in labels if len(label[1]) != 0]  # type: List[List[str]]
#         for sen in with_labels_sen:
#             for index, label in enumerate(sen):
#                 if "#_#" in label:
#                     print(ori_sen)
#                     print(sen)
#                     break
#                 label_content = label.split("##")
#                 if len(label_content) == 2:
#                     label_stat_key, label_content_stat_key = label_content
#                 else:
#                     # print(json_index)
#                     # print(ori_sen)
#                     continue
#                 # 统计标签频率
#                 label_stat_key_value = label_stat.get(label_stat_key, 0)
#                 label_stat_key_value += 1
#                 label_stat[label_stat_key] = label_stat_key_value
#
#                 # 统计标签内容频率
#                 label_content_stat_key_value = label_content_stat[label_stat_key].get(label_content_stat_key, 0)
#                 label_content_stat_key_value += 1
#                 label_content_stat[label_stat_key][label_content_stat_key] = label_content_stat_key_value
#
#         without_labels_sen = [label[0] for label in labels if len(label[1]) == 0]  # type: List[List[str]]
#         for sen in without_labels_sen:
#             start, end = sen
#             # print(ori_sen[start:end])
#             without_label_sens.append(ori_sen[start:end])
#
#         sen_without_labels_count += len([label for label in labels if len(label[1]) == 0])
#
# print("病例数量：%s" % his_count)
# print("句子数量：%s" % sen_count)
# print("有标签句子数量%s" % sen_with_labels_count)
# print("没有标签句子数量%s" % sen_without_labels_count)
# print("标签数量统计%s" % label_stat)
# print("标签数量统计%s" % label_content_stat["ZJY"])
# pickle.dump(without_label_sens, open("without_label_sens.npy", "wb"))

# TODO unlabeld sen parse
# result_sens = []
# first_char_stat = {}
# for sen in without_label_sens:
#     sen = re.sub(
#         "([\d]+[天年月日分钟次]+)|(就诊)|(我院)|(患儿)|(患者)|(医院)|([，。!；！：？“”a-zA-Z（）%#+\\/]+)|(收入)|（我）|([\d]+.[\d]+)|([\d]+)|([()\"*,-\^、×_~℃])",
#         "", sen)
#     startswith_label = ["随后", "遂", "一直未", "不适", "与其家属", "与患者及其家属", "与患者及家属", "为求", "与家属沟通", "为进一步", "今入", "征得患者", "排除",
#                         "经", "现", "本次", "术中", "建议", "病", "自", "未", "此次", "末次", "家长", "家属", "家人"]
#     legal = True
#     for start_label in startswith_label:
#         if sen.startswith(start_label):
#             legal = False
#             break
#     if not legal:
#         continue
#
#     pattern_labels = [".*进一步.*", ".*与.*家属", ".*门诊.*"]
#     for pattern in pattern_labels:
#         if re.match(pattern, sen):
#             legal = False
#             break
#     if not legal:
#         continue
#
#     if len(sen) != 0:
#         result_sens.append(sen)
#         char_count = first_char_stat.get(sen[0], 0)
#         char_count += 1
#         first_char_stat[sen[0]] = char_count
#
# without_label_sens = sorted(list(set(result_sens)))  # type:List[str]
# with open(r"D:\dr_easy_predata\temp\sens_without_labels.txt", "w") as f:
#     for sen in without_label_sens:
#         print(sen)
#         f.writelines(sen + "\n")
# first_char_stat = sort_dict(first_char_stat)
# print(first_char_stat)


