# -*- coding:utf-8 -*-
# @Time： 2020-07-10 10:58
# @Author: Joshua_yi
# @FileName: make_datafiles.py
# @Software: PyCharm
# @Project: MSMO
# @Description:
import ujson
import os
img_cap = {}
img_cap['dataset'] = 'mmae'

img_dir = '../data/data_preview/img'
cap_dir = '../data/data_preview/caption'

img_list = os.listdir(img_dir)
cap_list = os.listdir(cap_dir)

images = []
for i, img_name in enumerate(img_list):
    if i == 50:
        break
    img = {}

    img['sentids'] = [i]
    img['imgid'] = i
    img['filename'] = img_name
    img['split'] = 'train'
    sentences = []
    cap_name = cap_list[i]
    with open(cap_dir+'/'+cap_name, 'r', encoding='utf-8') as cap_f:
        cap_content = cap_f.readline()
    cap_dict = {}
    cap_dict['tokens'] = cap_content.split()
    cap_dict['raw'] = cap_content
    cap_dict['imgid'] = i
    cap_dict['sentid'] = i

    sentences.append(cap_dict)

    img['sentences'] = sentences

    images.append(img)

img_cap['images'] = images
with open('../data/data_preview/mmae_data/dataset.json', 'w', encoding='utf-8') as f:
    ujson.dump(img_cap, f, indent=4)
