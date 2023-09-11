#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/7 10:28
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : main.py
import os

print('aa')
import cv2
import time
import dlib
import json
from datetime import datetime
import numpy as np
import configparser
from scipy import signal


class Front_end():
    def __init__(self):
        '''
        :description: 组合显示窗口
        :param select_face_box: 人脸预选框
        :param select_forehead_box: 额头预选框
        :param select_nose_box: 鼻子预选框
        :return: 组合的显示图
        '''
        # self.select_face_box = select_face_box
        # self.select_forehead_box, self.select_nose_box = select_forehead_box, select_nose_box

    def show_face_box(self, img, face):
        # 原图上加人脸box
        f_x, f_y, f_w, f_h = face
        cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 255, 0), 2)
        cv2.putText(img, "Face", (f_x, f_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        return img

    def show_select_box(self, img, select_face_box, select_forehead_box, select_nose_box):
        # 加 select_face_box
        s_x, s_y, s_w, s_h = select_face_box
        s_x, s_y, s_w, s_h = int(s_x), int(s_y), int(s_w), int(s_h)
        cv2.rectangle(img, (s_x, s_y), (s_x + s_w, s_y + s_h), (255, 0, 255), 2)
        cv2.putText(img, "select_face_box", (s_x, s_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
        # 加 select_forehead_box
        fs_x, fs_y, fs_w, fs_h = select_forehead_box
        fs_x, fs_y, fs_w, fs_h = int(fs_x), int(fs_y), int(fs_w), int(fs_h)
        cv2.rectangle(img, (fs_x, fs_y), (fs_x + fs_w, fs_y + fs_h), (255, 0, 255), 2)
        cv2.putText(img, "select_forehead_box", (fs_x, fs_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
        # 加 select_nose_box
        ns_x, ns_y, ns_w, ns_h = select_nose_box
        ns_x, ns_y, ns_w, ns_h = int(ns_x), int(ns_y), int(ns_w), int(ns_h)
        cv2.rectangle(img, (ns_x, ns_y), (ns_x + ns_w, ns_y + ns_h), (255, 0, 255), 2)
        cv2.putText(img, "select_nose_box", (ns_x, ns_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
        return img

    def show_pulse(self, img, pulse, face):
        f_x, f_y, f_w, f_h = face
        text = "(Estimate: %0.1f bpm)" % (pulse)
        cv2.putText(img, text, (f_x,f_y+f_h), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        return img

    def show_graph(self, img, bpm_display, graph, graph_raw, graph_detrend, graph_demean):
        graph = np.vstack((graph_demean, graph))
        # graph = np.vstack((graph_detrend, graph))
        graph = np.vstack((graph_raw, graph))
        img = np.vstack((img, bpm_display))
        h, w, _ = img.shape
        img = np.hstack((img, graph))
        # 将中文字符转换为Unicode编码
        color = (233, 233, 233)
        cv2.putText(img, 'RAW PPG', (w, int(h/3)-5), cv2.FONT_HERSHEY_PLAIN, 1, color)
        cv2.line(img, (w, int(h/3)), (w*2, int(h/3)), color, thickness=2, lineType=cv2.LINE_8, shift=0)
        # cv2.putText(img, 'Detrend PPG', (w, int(h/2)-5), cv2.FONT_HERSHEY_PLAIN, 1, color)
        # cv2.line(img, (w, int(h/2)), (w*2, int(h/2)), color, thickness=2, lineType=cv2.LINE_8, shift=0)
        cv2.putText(img, 'BWF PPG', (w, int(h*2/3)-5), cv2.FONT_HERSHEY_PLAIN, 1, color)
        cv2.line(img, (w, int(h*2/3)), (w*2, int(h*2/3)), color, thickness=2, lineType=cv2.LINE_8, shift=0)
        cv2.putText(img, f'FFT PPG', (w, int(h)-8), cv2.FONT_HERSHEY_PLAIN, 1, color)
        cv2.line(img, (w, int(h)), (w*2, int(h)-8), color, thickness=2, lineType=cv2.LINE_8, shift=0)
        return img


class GetPulse:
    def __init__(self) -> None:
        self.update_idx = 0

        self.have_face = []  # 存储当前脸经过iou计算是否有效
        self.values = []  # 存储三通道均值
        self.times = []
        self.BUFFER_MAX_SIZE = 220
        self.MIN_FRAMES = 200
        self.DEBUG_MODE = False
        self.MIN_HZ = 0.83  # 50 BPM - minimum allowed heart rate
        self.MAX_HZ = 3.33  # 200 BPM - maximum allowed heart rate
        self.last_bpm = 0
        self.delay = 4

        self.graph_width = 0
        self.bpm_display_width = 0
        self.graph_values_raw = []
        self.graph_values_detrend = []
        self.graph_values_demean = []
        self.graph_values = []
        self.MAX_VALUES_TO_GRAPH = 50
        self.graph_height = 220
        self.bpm_display_height = 180

    def find_forehead(self, box, t_x=0.5, t_y=0.1, t_w=0.25, t_h=0.15):
        '''
        description: 根据人脸box，获取额头box。
        :param box: 人脸box坐标
        :param t_x, t_y, t_w, t_h: 计算坐标所需的常数偏置
        return: 额头box坐标
        '''
        fh_x = int(box[0] + box[2] * t_x - (box[2] * t_w / 2.0))
        fh_y = int(box[1] + box[3] * t_y - (box[3] * t_h / 2.0))
        fh_w = int(box[2] * t_w)
        fh_h = int(box[3] * t_h)
        forehead = [fh_x, fh_y, fh_w, fh_h]
        return forehead

    def get_subface_means(self, img, box):
        '''
        description: 获取box中图像的RGB均值，是计算心率的数据依据。
        param: img-图像，box-box坐标。
        return: v1是B通道，v2是G通道，v3是R通道。
        '''
        x, y, w, h = box
        img_box = img[y:y + h, x:x + w, :]
        v1 = np.mean(img_box[:, :, 0])
        v2 = np.mean(img_box[:, :, 1])
        v3 = np.mean(img_box[:, :, 2])
        # return (v1 + v2 + v3) / 3.
        return v2

    def sliding_window_demean(self, signal_values, num_windows):
        '''
        :description: 滑动平均
        :param signal_values: 帧数据
        :param num_windows: 滑动窗口个数
        :return: 返回滑动平均后的帧数据
        '''
        window_size = int(round(len(signal_values) / num_windows))
        demeaned = np.zeros(signal_values.shape)
        for i in range(0, len(signal_values), window_size):
            if i + window_size > len(signal_values):
                window_size = len(signal_values) - i
            curr_slice = signal_values[i: i + window_size]
            if self.DEBUG_MODE and curr_slice.size == 0:
                print('Empty Slice: size={0}, i={1}, window_size={2}'.format(signal_values.size, i, window_size))
                print(curr_slice)
            demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
        return demeaned

    def butterworth_filter(self, data, low, high, fps, order=5):
        '''
        :description: 巴特沃兹滤波
        :param data: 帧数据
        :param low: 带通最低频率
        :param high: 带通最高频率
        :param fps: 每秒帧数
        :param order: 滤波器阶数
        :return: 返回巴特沃兹滤波后的帧数据
        '''
        nyquist_rate = fps * 0.5  # 奈奎斯特速率
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low, high], btype='band')  # 'band'带通
        return signal.lfilter(b, a, data)

    def filter_signal_data(self, values, fps):
        '''
        :description: 平滑数据、使用滤波器过滤帧数据
        :param values: 原始帧数据
        :param fps: 每秒帧数
        :return: 返回过滤后的帧数据
        '''
        # Ensure that array doesn't have infinite or NaN values
        values = np.array(values)
        np.nan_to_num(values, copy=False)
        average = sum(values) / len(values)
        # Smooth the signal by detrending and demeaning
        detrended = signal.detrend(values, type='linear')  # 去掉序列的趋势，去趋势后的数据均值为零。
        demeaned = self.sliding_window_demean(detrended, 15)  # 滑动平均
        # Filter signal with Butterworth bandpass filter 过滤
        filtered = self.butterworth_filter(demeaned, self.MIN_HZ, self.MAX_HZ, fps, order=5)
        return values - average, detrended, demeaned, filtered

    def compute_bpm(self, filtered_values, fps, buffer_size, last_bpm):
        '''
        :description: 根据过滤后的帧数据、每秒帧数、数据帧总数，上一帧的心率，计算当前心率。
        :param filtered_values: 根据过滤后的数据
        :param fps: 每秒帧数
        :param buffer_size: 数据帧总数
        :param last_bpm: 上一帧的心率
        :return: 返回当前心率值
        '''
        # Compute FFT
        fft = np.abs(np.fft.rfft(filtered_values))

        # Generate list of frequencies that correspond to the FFT values生成与FFT值对应的频率列表
        freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)
        # Filter dialog_node any peaks person_dialog the FFT that are not within our range of [MIN_HZ, MAX_HZ]
        # because they correspond to impossible BPM values.
        # 过滤掉FFT中不在[MIN_HZ, MAX_HZ]频率范围内的任何峰值，因为它们对应于不可能的BPM值。
        while True:
            max_idx = fft.argmax()
            bps = freqs[max_idx]
            if bps < self.MIN_HZ or bps > self.MAX_HZ:
                fft[max_idx] = 0
            else:
                bpm = bps * 60.0
                break
                # It's impossible for the heart rate to change more than 10% between samples,
        # so use a weighted average to smooth the BPM with the last BPM.
        # 样本之间心率的变化不可能超过10%，所以使用上一个BPM进行加权平均、
        if last_bpm == 0:
            bpm = bpm
        if last_bpm > 0:
            if abs(bpm-last_bpm)>17 or bpm <50:
                print(f'>17 bpm:{bpm:.3f} | last_bpm:{last_bpm:.3f}')
                self.delay=4
                bpm = 0
            elif 5<=abs(bpm-last_bpm)<17:
                bpm = (last_bpm * 0.4) + (bpm * 0.6)
                print(f'>5 bpm:{bpm:.3f} | last_bpm:{last_bpm:.3f}')
            elif 3<=abs(bpm-last_bpm)<5:
                bpm = (last_bpm * 0.6) + (bpm * 0.4)
                print(f'>5 bpm:{bpm:.3f} | last_bpm:{last_bpm:.3f}')
            elif abs(bpm - last_bpm) < 3:
                bpm = bpm
                # print(f'>5 bpm:{bpm:.3f} | last_bpm:{last_bpm:.3f}')
        # elif abs(bpm-last_bpm)>3:
            #     print('>10 bpm:', bpm, ' | last_bpm:', last_bpm)
            #     bpm = (last_bpm * 0.5) + (bpm * 0.5)
            # else:
            #     print('bpm:', bpm, ' | last_bpm:', last_bpm)
            #     bpm = (last_bpm  * 0.8) + (bpm * 0.2)
        return bpm

    def find_effective_face(self, nums):
        '''
        :description: 找have_face有效段的起止下标。
        :param num: have_face数组
        :return: 数组中最长连续1的起始位置和结束位置
        '''
        max_len = 0  # 最长连续1的长度
        start, end = 0, 0  # 最长连续1的起始位置和结束位置
        cur_start = 0  # 当前连续1的起始位置

        for i in range(len(nums)):
            if nums[i] == 0:
                # 更新最长连续1的长度和起始位置
                if i - cur_start > max_len:
                    max_len = i - cur_start
                    start, end = cur_start, i - 1
                cur_start = i + 1  # 更新当前连续1的起始位置

        # 处理最后一个连续的1
        if len(nums) - cur_start > max_len:
            max_len = len(nums) - cur_start
            start, end = cur_start, len(nums) - 1

        return start, end

    def calculate(self, img, box1, box2, have_face):  # (img, select_forhead_box, self.have_face)
        '''
        :description: 根据图像、额头box坐标、鼻子box坐标、人脸检测状态数组，计算心率、绘制显示窗口。
        :param img: 一帧图像
        :param box1: 额头box坐标
        :param box2: 鼻子box坐标
        :param have_face: 人脸检测状态数组
        :return: 返回心率值、波形图、心率显示图
        '''
        enable = {'pulse': 0,
                  'face': 0,
                  'loading': 0}
        # 用于显示波形图 BPM值
        view = np.array(img)
        # Heart rate graph gets 75% of window width.  BPM gets 25%.
        if self.graph_width == 0:
            self.graph_width = int(view.shape[1] * 0.75)
        if self.bpm_display_width == 0:
            self.bpm_display_width = view.shape[1]
            # self.bpm_display_width = view.shape[1] - self.graph_width

        # 用于记录当前帧是否有脸
        self.have_face = have_face

        # 计算额头box范围内，原始照片的RGB通道均值，
        val1 = self.get_subface_means(img, box1)
        val2 = self.get_subface_means(img, box2)
        val = (val1 + val2) / 2.
        # 记录到values_buffer中
        self.values.append(val)
        self.times.append(time.time())

        # 数据帧Buffer满, pop出之前的数据，保留新的
        if len(self.times) > self.BUFFER_MAX_SIZE:
            self.values.pop(0)
            self.have_face.pop(0)
            self.times.pop(0)

            # 找have_face有效段的起止下标
        idx_start, idx_end = self.find_effective_face(self.have_face)
        curr_buffer_size = idx_end - idx_start + 1  # 当前有效数据帧数

        if (self.have_face[-1] == 1):  # and self.have_face[-2]==1 and self.have_face[-3]==1):
            # 使用have_face满足条件的部分, 帧数要大于一定长度才能计算心率
            if (curr_buffer_size > self.MIN_FRAMES):
                # 计算这一段数据帧一共花费时间 和 每秒帧数
                time_elapsed = self.times[idx_end] - self.times[idx_start]
                fps = (curr_buffer_size) / time_elapsed
                # 实际参与的数据 self.values[idx_start:idx_end+1]
                # 过滤得到纯净的数据帧
                raw, detrended, demeaned, filtered = self.filter_signal_data(self.values[idx_start:idx_end + 1], fps)

                # 记录信号,（四种波形）
                self.graph_values_raw.append(raw[-1])
                self.graph_values_detrend.append(detrended[-1])
                self.graph_values_demean.append(demeaned[-1])
                self.graph_values.append(filtered[-1])
                if len(self.graph_values) > self.MAX_VALUES_TO_GRAPH:
                    self.graph_values_raw.pop(0)
                    self.graph_values_detrend.pop(0)
                    self.graph_values_demean.pop(0)
                    self.graph_values.pop(0)
                    # 绘制波形图
                graph_raw = self.draw_graph(self.graph_values_raw, self.graph_width, self.graph_height)
                graph_detrend = self.draw_graph(self.graph_values_detrend, self.graph_width, self.graph_height)
                graph_demean = self.draw_graph(self.graph_values_demean, self.graph_width, self.graph_height)
                graph = self.draw_graph(self.graph_values, self.graph_width, self.graph_height)

                # 计算心率
                self.update_idx += 1
                if self.update_idx > 30:
                    if self.delay>0:
                        self.delay-=1
                    self.update_idx = 0
                    bpm = self.compute_bpm(filtered, fps, curr_buffer_size, self.last_bpm)
                    # 绘制BPM显示
                    if bpm==0 or self.delay>0:
                        bpm_display = self.draw_bpm('--', self.bpm_display_width, self.bpm_display_height)
                        enable['pulse'], enable['face'], enable['loading'] = 0, 1, 100
                    else:
                        bpm_display = self.draw_bpm(str(int(round(bpm))), self.bpm_display_width, self.bpm_display_height)
                        enable['pulse'], enable['face'], enable['loading'] = bpm, 1, 100
                    # 更新last_bpm的值
                    self.last_bpm = bpm
                    return enable, bpm, graph, bpm_display, graph_raw, graph_detrend, graph_demean
                else:
                    bpm = self.last_bpm
                    if bpm == 0 or self.delay>0:
                        enable['pulse'], enable['face'], enable['loading'] = 0, 1, 100
                        bpm_display = self.draw_bpm('--', self.bpm_display_width, self.bpm_display_height)
                    else:
                        enable['pulse'], enable['face'], enable['loading'] = bpm, 1, 100
                        bpm_display = self.draw_bpm(str(int(round(bpm))), self.bpm_display_width, self.bpm_display_height)
                    return enable, bpm, graph, bpm_display, graph_raw, graph_detrend, graph_demean

            else:
                # have_face数据量不足
                bpm = 0
                pct = int(round(float(curr_buffer_size) / self.MIN_FRAMES * 100.0))
                loading_text = 'Load face data...' + str(pct) + '%'
                graph = self.draw_graph_text(loading_text, (255, 0, 255), self.graph_width, self.graph_height)
                graph_raw = self.draw_graph_text(loading_text, (255, 0, 255), self.graph_width, self.graph_height)
                graph_detrend = self.draw_graph_text(loading_text, (255, 0, 255), self.graph_width, self.graph_height)
                graph_demean = self.draw_graph_text(loading_text, (255, 0, 255), self.graph_width, self.graph_height)
                bpm_display = self.draw_bpm('--', self.bpm_display_width, self.bpm_display_height)
                enable['pulse'], enable['face'], enable['loading'] = 0, 1, pct
                return enable, bpm, graph, bpm_display, graph_raw, graph_detrend, graph_demean

        else:  # self.have_face[-1]!=1
            bpm = 0
            self.delay=4
            loading_text = 'No face detected'
            graph = self.draw_graph_text(loading_text, (0, 0, 255), self.graph_width, self.graph_height)
            graph_raw = self.draw_graph_text(loading_text, (0, 0, 255), self.graph_width, self.graph_height)
            graph_detrend = self.draw_graph_text(loading_text, (0, 0, 255), self.graph_width, self.graph_height)
            graph_demean = self.draw_graph_text(loading_text, (0, 0, 255), self.graph_width, self.graph_height)
            bpm_display = self.draw_bpm('--', self.bpm_display_width, self.bpm_display_height)
            enable['pulse'], enable['face'], enable['loading'] = 0, 0, 0
            return enable, bpm, graph, bpm_display, graph_raw, graph_detrend, graph_demean

    def img_pulse(self, img, pulse, face, graph, bpm_display, graph_raw, graph_detrend, graph_demean, select_face_box,
                  select_forehead_box, select_nose_box):
        '''
        :description: 组合显示窗口
        :param img: 一帧照片
        :param pulse: 心率值
        :param face: 检测到的人脸box
        :param graph: 波形图
        :param bpm_display: 心率显示图
        :param select_face_box: 人脸预选框
        :param select_forehead_box: 额头预选框
        :param select_nose_box: 鼻子预选框
        :return: 组合的显示图
        '''
        # 原图上加 人脸box
        f_x, f_y, f_w, f_h = face
        cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 255, 0), 2)
        cv2.putText(img, "Face", (f_x, f_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        # # 再加 额头box
        # forehead = self.find_forehead(face)
        # fh_x, fh_y, fh_w, fh_h = forehead
        # cv2.rectangle(img, (fh_x,fh_y), (fh_x+fh_w, fh_y+fh_h), (0, 255, 0), 2)
        # cv2.putText(img, "Forehead", (fh_x,fh_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        # box附近显示心率
        text = "(Estimate: %0.1f bpm)" % (pulse)
        cv2.putText(img, text, (f_x, f_y + f_h), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        # 拼接波形图 和 BPM显示
        # graph = np.hstack((graph, bpm_display))
        # img = np.vstack((img, graph))
        graph = np.vstack((graph_demean, graph))
        graph = np.vstack((graph_detrend, graph))
        graph = np.vstack((graph_raw, graph))
        img = np.vstack((img, bpm_display))
        img = np.hstack((img, graph))
        # 加 select_face_box
        s_x, s_y, s_w, s_h = select_face_box
        s_x, s_y, s_w, s_h = int(s_x), int(s_y), int(s_w), int(s_h)
        cv2.rectangle(img, (s_x, s_y), (s_x + s_w, s_y + s_h), (255, 0, 255), 2)
        cv2.putText(img, "select_face_box", (s_x, s_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
        # 加 select_forehead_box
        fs_x, fs_y, fs_w, fs_h = select_forehead_box
        fs_x, fs_y, fs_w, fs_h = int(fs_x), int(fs_y), int(fs_w), int(fs_h)
        cv2.rectangle(img, (fs_x, fs_y), (fs_x + fs_w, fs_y + fs_h), (255, 0, 255), 2)
        cv2.putText(img, "select_forehead_box", (fs_x, fs_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
        # 加 select_nose_box
        ns_x, ns_y, ns_w, ns_h = select_nose_box
        ns_x, ns_y, ns_w, ns_h = int(ns_x), int(ns_y), int(ns_w), int(ns_h)
        cv2.rectangle(img, (ns_x, ns_y), (ns_x + ns_w, ns_y + ns_h), (255, 0, 255), 2)
        cv2.putText(img, "select_nose_box", (ns_x, ns_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))

        return img

    def draw_graph(self, signal_values, graph_width, graph_height):
        '''
        :description: 根据滤波后的数据，绘制波形图。
        :param signal_values: 滤波后的数据
        :param graph_width: 波形图宽度
        :param graph_height: 波形图高度
        :return: 波形图
        '''
        graph = np.zeros((graph_height, graph_width, 3), np.uint8)
        scale_factor_x = float(graph_width) / self.MAX_VALUES_TO_GRAPH
        # Automatically rescale vertically based on the value with largest absolute value
        max_abs = max(max(signal_values), -min(signal_values))
        scale_factor_y = (float(graph_height) / 2.4) / max_abs
        midpoint_y = graph_height / 2
        for i in range(0, len(signal_values) - 1):
            curr_x = int(i * scale_factor_x)
            curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
            next_x = int((i + 1) * scale_factor_x)
            next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
            cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=2)
        cv2.putText(graph, f'sacle:{scale_factor_y:.1f}', (0, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        return graph

    def draw_bpm(self, bpm_str, bpm_width, bpm_height):
        '''
        :description: 根据心率值，绘制心率值显示图。
        :param bpm_str: str(心率值)
        :param bpm_width: 心率值显示图宽度
        :param bpm_height: 心率值显示图高度
        :return: 心率值显示图
        '''
        bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
        bpm_text_size, bpm_text_base = cv2.getTextSize(bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                       thickness=2)
        bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
        bpm_text_y = int(bpm_height / 2 + bpm_text_base)
        cv2.putText(bpm_display, bpm_str, (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=2.7, color=(0, 255, 0), thickness=2)
        bpm_label_size, bpm_label_base = cv2.getTextSize('BPM', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                         thickness=1)
        bpm_label_x = int((bpm_width - bpm_label_size[0]) / 2)
        bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
        cv2.putText(bpm_display, 'BPM', (bpm_label_x, bpm_label_y),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 255, 0), thickness=1)
        return bpm_display

    def draw_graph_text(self, text, color, graph_width, graph_height):
        '''
        :description: 提示显示
        :param text: 提示句 or 进度条
        :param color: 字体颜色
        :param graph_width: 显示宽度
        :param graph_height: 显示高度
        :return: 显示图
        '''
        graph = np.zeros((graph_height, graph_width, 3), np.uint8)
        text_size, text_base = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
        text_x = int((graph_width - text_size[0]) / 2)
        text_y = int((graph_height / 2 + text_base))
        cv2.putText(graph, text, (text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color,
                    thickness=1)
        return graph
class FaceDetector:
    def __init__(self, detection_type='box'):
        '''
        一个基于OpenCV的人脸检测类, 根据需要选择返回人脸框、人脸图像或绘制人脸框的原始图像
        :param detection_type: ['crop', 'box', 'draw']
        '''
        self.output_video_size = (224, 224)

        # self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        # self.openface_align = openface.AlignDlib(r"C:\Users\ykjiang2\PycharmProjects\FER_iflytek\corecode\ktool\shape_predictor_68_face_landmarks.dat")
        self.face_detector_dilb = dlib.get_frontal_face_detector()

        self.detection_type = detection_type

        self.face_box_buffer = []
        self.FACE_BUFFER_MAX_SIZE = 3

    def get_box_cv2(self, image):
        '''
        description: 输入一张图像，获取该图像中所有的人脸box坐标
        :param image: 一帧图像
        return: 所有的人脸box的坐标
        '''
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = self.face_detector.detectMultiScale(gray,
                                                    scaleFactor=1.3,
                                                    minNeighbors=1,
                                                    minSize=(120, 120),
                                                    maxSize=(300, 300),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        return faces  # x, y , w, h

    def get_box_dilb(self, image):
        # 将图像缩小为原来的1/4
        image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

        # 将图像转换为灰度图
        gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = []
        rects = self.face_detector_dilb(gray, 1)
        for rect in rects:
            faces.append([rect.left()*4, rect.top()*4, rect.width()*4, rect.height()*4])
        return faces

    # def get_box_openface(self, image):
    #     # 将图像转换为灰度图像
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # 检测人脸
    #     faces = []
    #     rects = self.openface_align.getAllFaceBoundingBoxes(gray)
    #     for rect person_dialog rects:
    #         faces.append([rect.left(), rect.top(), rect.width(), rect.height()])
    #     return faces

    def detect_face(self, image):
        '''
        description: 输入一张图像，在"box"模式下，获取该图像中人脸box坐标。
                     如果有多个人脸，保留最大的人脸box；
                     如果没有人脸，令box的坐标值都为0。
        :param image: 图像
        return: 人脸box坐标
        '''
        faces = self.get_box_dilb(image)

        if len(faces) == 1:
            # 只检测到一个人脸
            face = faces[0]
        elif len(faces) > 1:
            # 检测到多个人脸
            max_size = 0
            for (x, y, w, h) in faces:
                if w * h > max_size:
                    max_size = w * h
                    max_face_coords = (x, y, w, h)
            face = max_face_coords
        else:
            print("没检测到人脸")
            face = [0, 0, 0, 0]
        # 根据检测类型返回结果
        if self.detection_type == 'box':
            return face
        elif self.detection_type == 'crop':
            x, y, w, h = face
            image = image[y:y + h, x:x + w]
            image = cv2.resize(image, self.output_video_size)
            return image
        elif self.detection_type == 'draw':
            x, y, w, h = face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return image
        else:
            raise ValueError('Invalid detection type: {}'.format(self.detection_type))
class Solution:
    def __init__(self) -> None:
        '''
        心率检测类
        self.FD: 人脸检测类
        self.have_face: 人脸检测状态记录
        self.P: 心率计算类
        '''
        self.FD = FaceDetector('box')
        self.Front_end = Front_end()
        self.GP = GetPulse()
        self.have_face = []
        self.en = 0
        self.face = [0,0,0,0]
    def xywh2xyxy(self, face):

        # :description: 把x\y\w\h形式的box坐标，转换为x\y\x+w\y+h的坐标
        # :param face: box坐标
        # :return: 转换后的box坐标

        x, y, w, h = face
        return x, y, x + w, y + h

    def iou(self, box1, box2):
        """
        Calculate IOU between two bounding boxes
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        """
        box1 = self.xywh2xyxy(box1)
        box2 = self.xywh2xyxy(box2)

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area_box1 + area_box2 - intersection
        iou = intersection / union
        return iou

    def adaption_box(self, image):
        '''
        :description: 根据一帧照片的尺寸，确定人脸预选框、额头预选框、鼻子预选框的位置和尺寸
        :param image: 一帧照片
        :return: 三个预选框坐标
        '''
        H, W, _ = image.shape  # (480, 640)
        x1, y1, w1, h1 = int(W / 2) - 100, int(H / 2) - 150, 200, 300
        x2, y2, w2, h2 = x1 + 60, y1 + 40, w1 - 120, h1 - 260
        x3, y3, w3, h3 = x1 + 50, y1 + 150, w1 - 100, h1 - 250
        return [x1, y1, w1, h1], [x2, y2, w2, h2], [x3, y3, w3, h3]

    def is_face(self, face, target, iou_theshold, have_face):
        '''
        :description: 检测人脸是否在合适的区域,
                      若在合适区域 则记录为1 否则为0;
                      返回状态队列;
        :param face: 检测到的人脸box
        :param target: 人脸预选框
        :param iou_theshold: iou阈值
        :param have_face: 人脸检测状态记录
        :return: 更新人脸检测状态记录
        '''
        if self.iou(face, target) > iou_theshold:
            have_face.append(1)
        else:
            have_face.append(0)
            have_face = [0] * len(have_face)

        return have_face

    def forward(self, img):
        # 确定预选框, 脸+额头
        select_face_box, select_forhead_box, select_nose_box = self.adaption_box(img)
        # img一帧照片，检测人脸box
        self.en+=1
        if self.en>30:
            self.en=0
            self.face = self.FD.detect_face(img)  # face [x, y, w, h]
        # 判断预选框内人脸是否有效
        self.have_face = self.is_face(self.face, select_face_box, 0.3, self.have_face)

        # # 根据img和box，计算心率，返回心率、波形图、心率显示图
        enable, pulse, graph, bpm_display, graph_raw, graph_detrend, graph_demean = self.GP.calculate(img, select_forhead_box,
                                                                                             select_nose_box,
                                                                                             self.have_face)
        # 组合show图
        img = self.Front_end.show_select_box(img, select_face_box, select_forhead_box, select_nose_box)
        img = self.Front_end.show_face_box(img, self.face)
        img = self.Front_end.show_pulse(img, pulse, self.face)
        img = self.Front_end.show_graph(img, bpm_display, graph, graph_raw, graph_detrend, graph_demean)
        return img, enable
def write_json(json_path, data_dict):
    '''
    字典类型格式化保持json文件，可提供自动换行
    :param json_path: 要保持的json路径
    :param data_dict: 待保持的字典数据
    :return:
    '''
    json_str = json.dumps(data_dict, indent=4, ensure_ascii=False)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)
def read_capture_write(funtion, output_path):
    capture = cv2.VideoCapture(0)
    # capture.set(3, 480)
    # 获取视频的帧速率和尺寸
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置视频编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    output_predict = {}
    # 初始化计时器
    total_time, seconds = 0.0, 0
    frame_idx = 0
    # 循环读取视频帧并处理
    while True:
        # 读取一帧
        ret, frame = capture.read()

        # 如果读取到帧，则进行处理
        if ret:
            frame_idx+=1
            # 初始化计时器
            start_time = time.time()
            output_video.write(frame)
            frame, enable = funtion(frame)
            # 将处理后的帧写入新视频文件
            cv2.imshow('video', frame)
            if datetime.now().second != seconds:
                seconds = datetime.now().second
                enable['frame_idx'] = frame_idx
                output_predict[datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-7]] = enable

            # 停止计时器并计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            if processing_time>0.033:
                print("\n{:.3f} seconds (有点卡顿，心率准确度受影响)\n".format(processing_time))
            else:
                print("{:.3f} seconds (Processing time)".format(processing_time))
            total_time += processing_time
            # 等待按键按下
            if cv2.waitKey(1) == ord('q'):
                break
    # 释放资源
    write_json(output_path.replace('mp4', 'json'), output_predict)
    capture.release()
    output_video.release()
def read_json(json_path):
    return json.load(open(json_path, 'r', encoding='utf-8'))
    # for -, v person_dialog json.load(open(json_path,'r',encoding='utf-8')).items():
    #       print(_， v)
def show_pulse_list(save_path):
    # 解析心率数据
    heart_rate = []
    timestamps = []
    for key, value in read_json(save_path).items():
        timestamp = datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
        if value['pulse'] > 0:
            timestamps.append(timestamp)
            heart_rate.append(value['pulse'])

    # 创建一个黑色图像
    img = np.zeros((512, 512, 3), np.uint8)
    # 计算x轴和y轴的比例尺
    x_scale = 512 / 60
    y_scale = 512 / 150

    # 将时间对象列表转换为时间戳列表，并缩小x轴范围
    x = [(t - timestamps[0]).total_seconds() *10 for t in timestamps]

    # 将时间戳和心率列表转换为OpenCV的点坐标列表
    pts = [(int(x[i]), int(512 - (heart_rate[i] * y_scale))) for i in range(len(x))]

    # 将点坐标列表转换为包含单个曲线的列表
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))

    # 绘制曲线
    cv2.polylines(img, [pts], False, (0, 255, 0), 2)

    # 添加x轴标签和刻度
    cv2.putText(img, "Time (s)", (250, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(0, 61, 10):
        cv2.putText(img, str(i), (int(i * x_scale), 505), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # 添加y轴标签和刻度
    for i in range(0, 151, 10):
        cv2.putText(img, str(i), (0, int(512 - (i * y_scale))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # 添加标题
    cv2.putText(img, "Heart Rate vs Time", (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 显示图像
    cv2.imshow("Heart Rate vs Time", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read("config.ini",encoding='utf-8')
    save_path = conf.get("data", "save_path")
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    # input("enter exit")
    read_capture_write(Solution().forward, save_path+'test.mp4')
    show_pulse_list(save_path+'test.json')

