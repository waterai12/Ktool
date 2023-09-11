#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 14:39
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : exp2.1 pulse_unsensible.py             

from Ktool.video_tool import read_capture_show, read_video_save
from Ktool.GUI.exp1_pulse_display import Front_Pulse
from Ktool.algorithm.get_pulse_org import Solution


frame_idx_list = [1, 29, 59, 89, 119, 148, 179, 209, 239, 269, 299, 329, 359, 389, 419, 449,
                  479, 509, 538, 569, 599, 629, 659, 689, 719, 749, 779, 809, 838, 868, 899,
                  929, 959, 989, 1019, 1049, 1079, 1109, 1139, 1169, 1199, 1229, 1259, 1289,
                  1319, 1349, 1379, 1409, 1439, 1469, 1499, 1529, 1559, 1589, 1619, 1649, 1679,
                  1708, 1739, 1769, 1799, 1829, 1859, 1889, 1919, 1949, 1979, 2008, 2039]
# 69s
if __name__ == "__main__":

    # read_capture_show(r'D:\data\pulse_test3\leijiang8\kuai\test.mp4', Front_Pulse(Solution).forward)
    # read_video_save(Solution().forward, r'D:\data\pulse_test3\leijiang8\kuai\test.mp4', './result/leijiang2_fast.json', frame_idx_list)
    read_capture_show('0', Front_Pulse(Solution).forward)
