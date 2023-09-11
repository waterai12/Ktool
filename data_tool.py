#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 10:32
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : data_tool.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
def list_count(l):
    return pd.value_counts(l)


def print_ac(a, b, r=False):
    # a = pred b = label
    ac = a/b
    print(f'{ac:.2f}({a}/{b})')
    if r:
        return ac


def show_result(y_true1, y_pred1):
    emo_class7 = ["happy", "sad", "normal", "angry", "surprise", "disgust", "fair"]
    y_pred = [emo_class7.index(i) for i in y_pred1]
    y_true = [emo_class7.index(i) for i in y_true1]
    t = classification_report(y_true, y_pred, target_names=["happy", "sad", "normal", "angry", "surprise", "disgust", "fair"])
    print(t)
    cmatrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6])
    plt.matshow(cmatrix, cmap=plt.cm.Reds, alpha=0.7)
    plt.show()
    print('\n-----------------confusion matrix-----------------')
    tag = 8
    print('class'.ljust(tag,' ')+ '\t'+'r_sum'.ljust(5,' ')+ '\t'+'recall'.ljust(5, ' '), end='\t')
    for i in emo_class7:
        print((i).ljust(tag, ' '), end= '\t')
    print()
    for i in range(len(cmatrix)):
        if sum(cmatrix[i]) <= 0:
            print(emo_class7[i].ljust(tag, ' ') + '\t' + str(sum(cmatrix[i])).ljust(5, ' ') + '\t'+ f'0.00'.ljust(5, ' '), end='\t')
        else:
            print(emo_class7[i].ljust(tag, ' ') + '\t' + str(sum(cmatrix[i])).ljust(5, ' ') + '\t'+ f'{cmatrix[i][i]/sum(cmatrix[i]):.2f}'.ljust(5, ' '), end='\t')
        for j in range(len(cmatrix[i])):
            print(str(cmatrix[i][j]).ljust(tag, " "), end='\t')
        print()
    ac = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            ac += 1
    ss, recall = '', ''
    for i in range(len(emo_class7)):
        ss += str(y_pred.count(i)).ljust(tag, ' ')+'\t'
        if y_pred.count(i) <= 0:
            recall += f'0.0'.ljust(tag, ' ')+'\t'
        else:
            recall += f'{cmatrix[i][i] / y_pred.count(i):.2f}'.ljust(tag, ' ')+'\t'
    print('AC'.ljust(tag, ' ')+'\t'+''.ljust(5, ' ')+'\t'+f'{ac/len(y_pred):.2f}'+'\t'+recall)
    print('p_sum'.ljust(tag, ' ')+'\t'+str(len(y_pred)).ljust(5, ' ')+'\t'+''.ljust(5, ' ') +'\t'+ ss)


def datetime2timestamp(t):
    return datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timestamp() if type(t) == str else t.timestamp()


def timestamp2datetime(timeStamp):
    d = datetime.fromtimestamp(timeStamp)
    # str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
    return d.strftime("%Y-%m-%d %H:%M:%S.%f")


def calculate_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length.")
    errors = []
    errors_calc = []
    for val1, val2 in zip(list1, list2):
        error = abs(val1 - val2)
        if error<30:
            errors_calc.append(error)
        errors.append(error)
    strs = f"{np.mean(errors_calc):.2f}({len(errors_calc)}/{len(errors)})"
    # plt.title(f"{subject}:{np.mean(errors_calc):.2f}({len(errors_calc)}/{len(errors)})")
    return errors, errors_calc, strs


def split_list(list1, length, file_name):
    '''
        a = [i for i in range(1033)]
        for k, v in split_list(a, 200, 'out_{}.txt').items():
            print(k, v)
    :param list1:
    :param length:
    :param file_name:
    :return:
    '''
    a = {}
    part_id = 0
    last_part_id = 0
    idx = 0
    for i in range(len(list1)):
        part_id += 1
        if part_id%length==0:
            idx+=1
            a[file_name.format(idx)]=list1[last_part_id:part_id]
            last_part_id = part_id

    if part_id%length!=0:
        a[file_name.format(idx+1)] = list1[idx*length:]
    return a


