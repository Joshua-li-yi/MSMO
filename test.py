# -*- coding:utf-8 -*-
# @Time： 2020-06-22 14:57
# @Author: Joshua_yi
# @FileName: test.py
# @Software: PyCharm
# @Project: MSMO
# Owed by: http://blog.csdn.net/chunyexiyu
# direct get the input name from called function code


def f1():
    return 1,2,3,(4,5)

a, b,c,_= f1()
print(a,b,c,_,_[0],_[1])