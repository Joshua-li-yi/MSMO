# -*- coding:utf-8 -*-
# @Time： 2020-06-22 14:57
# @Author: Joshua_yi
# @FileName: test.py
# @Software: PyCharm
# @Project: MSMO
#Owed by: http://blog.csdn.net/chunyexiyu
#direct get the input name from called function code

import inspect
def retrieve_name_ex(var):
    stacks = inspect.stack()
    try:
        callFunc = stacks[1].function
        code = stacks[2].code_context[0]
        startIndex = code.index(callFunc)
        startIndex = code.index("(", startIndex + len(callFunc)) + 1
        endIndex = code.index(")", startIndex)
        return code[startIndex:endIndex].strip()
    except:
        return ""


def tensor_shape(value):
    print("{} shape:  {}".format(retrieve_name_ex(value), value.shape))
    pass

import torch
a = torch.Tensor(3, 3)
tensor_shape(a)
