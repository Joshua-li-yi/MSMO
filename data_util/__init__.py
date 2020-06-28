# -*- coding:utf-8 -*-
# @Time： 2020-06-28 22:24
# @Author: Joshua_yi
# @FileName: __init__.py
# @Software: PyCharm
# @Project: MSMO
 
from data_util import config
from data_util.data import example_generator

example_generator(config.train_article_path, single_pass=True)

