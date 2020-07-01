# -*- coding:utf-8 -*-
# @Time： 2020-06-22 14:57
# @Author: Joshua_yi
# @FileName: test.py
# @Software: PyCharm
# @Project: MSMO
import json


class Employee(object):
    def __init__(self):
        super(Employee, self).__init__()
        self.emp_no = input('employee number: ')
        self.emp_name = input('employee name: ')
        self.emp_s = input('employee salary: ')
        self.write_e()

    def write_e(self):
        with open('employees.txt', 'a') as ef:
            ef.write(self.emp_no + '\n')
            ef.write(self.emp_name + '\n')
            ef.write(self.emp_s)


def read_e(filename):
    with open(filename, 'r') as ef:
        lines = ef.readlines()
        print(lines)
    pass

i = 0
while True:
    i += 1
    e = Employee()
    if i == 2:
        break

read_e('employees.txt')
# 接下来要执行的代码

 