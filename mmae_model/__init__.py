# -*- coding:utf-8 -*-
# @Time： 2020-07-09 17:23
# @Author: Joshua_yi
# @FileName: __init__.py.py
# @Software: PyCharm
# @Project: MSMO
# @Description: 

from mmae_model import mmae
if __name__ == '__main__':
    a = mmae.MMAE()
    b = a.model_test()
    print(b)