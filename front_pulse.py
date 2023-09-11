#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 16:31
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : exp1_pulse_display.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Front_Pulse():
    def __init__(self, Solution, realtime=True):
        self.Solution = Solution(show_display=True, realtime=realtime)
        self.graph_values = []
        # self.graph_values_demeaned = []
        # self.graph_values_detrended = []
        # self.graph_values_raw = []
        self.graph_height = 480
        self.graph_width = 0
        self.bpm_display_height = 480
        self.bpm_display_width = 0
        self.MAX_VALUES_TO_GRAPH = 50
        self.fig, self.ax = plt.subplots()

        # 设置子图的初始x和y轴范围
        self.ax.set_xlim(0, 121)
        self.ax.set_ylim(80, 150)
        list1, list2 = [0]*120, [0]*120
        self.line1, = self.ax.plot(list1, label='signal')
        self.line2, = self.ax.plot(list2, label='values')

    def forward(self, img):

        # 用于显示波形图 BPM值
        view = np.array(img)
        # Heart rate graph gets 75% of window width.  BPM gets 25%.
        if self.graph_width == 0:
            self.graph_width = int(view.shape[1] * 0.75)
        if self.bpm_display_width == 0:
            self.bpm_display_width = view.shape[1]
        # 计算心率
        result = self.Solution.forward(img)
        # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4], result['pulse'])

        # 实时显示信号通道的值
        # if len(result['signal_list'])>=120:
        #     # my_list = result['signal_list'][-120:]
        #     # # 显示原始采集信号
        #     # self.line1.set_ydata(my_list)
        #     # # 显示有效信号段
        #     # if result['filtered']!=[]:
        #     #     self.line2.set_ydata(result['values'][-120:])
        #     # else:
        #     #     self.line2.set_ydata([90]*120)
        #     # plt.legend()  # 显示图例
        #     # self.fig.canvas.draw()
        #     print(f"bpm:{result['pulse']:.2f} fps:{result['fps']:.1f}")
        #     # plt.pause(0.01)

        # 采集graph显示信号
        if result['filtered']!=[]:
            self.graph_values.append(result['filtered'][-1])
        if len(self.graph_values) > self.MAX_VALUES_TO_GRAPH:
            self.graph_values.pop(0)
        # 若检测到人脸，且信号采样集满足条件
        if result['state']==1 and len(result['filtered']) > 0:
            graph0 = self.draw_k_graph(result['signal_list'], [130, 170],self.graph_width, 120)
            graph1 = self.draw_k_graph(result['values'], [130, 170], self.graph_width, 120)
            graph2 = self.draw_k_graph(result['demeaned'], [-2, 2], self.graph_width, 120)
            # graph3 = self.draw_k_graph(self.graph_values_demeaned, self.graph_width, 100)
            graph4 = self.draw_graph(self.graph_values, self.graph_width, 120)

            graph = np.vstack((graph0, graph1, graph2, graph4))
            bpm_display = self.draw_bpm(str(int(round(result['pulse']))), self.bpm_display_width,
                                        self.bpm_display_height, result['fps'])
            face_outline = self.get_face_outline(result['face_landmarks'])
            segmented_face = self.draw_face_outline(img, face_outline, result['filtered'])
            for i in result['roi_box']:
                img = self.show_face_box(img, i)

        # 若检测到人脸，且信号采样集未满足条件，则显示百分比
        elif result['state']==1 and len(result['filtered']) <= 0:
            graph0 = self.draw_k_graph(result['signal_list'], [130, 170],self.graph_width, 120)
            graph1 = self.draw_k_graph(result['values'], [130, 170], self.graph_width, 120)
            graph2 = self.draw_k_graph(result['demeaned'], [-2, 2], self.graph_width, 120)
            # graph3 = self.draw_k_graph(self.graph_values_demeaned, self.graph_width, 100)
            graph4 = self.draw_graph(self.graph_values, self.graph_width, 120)
            pct = len(result['values']) / 150 * 100.0
            loading_text = 'Load face data...' + str(pct) + '%'
            graph = np.vstack((graph0, graph1, graph2, graph4))
            bpm_display = self.draw_bpm('--', self.bpm_display_width, self.bpm_display_height, result['fps'])
            segmented_face = np.zeros_like(img)
            for i in result['roi_box']:
                img = self.show_face_box(img, i)
        # 若人脸运动过速，则显示人脸过速
        elif result['state']==0:
            graph0 = self.draw_k_graph(result['signal_list'], [130, 170],self.graph_width, 120)
            graph1 = self.draw_k_graph(result['values'], [130, 170], self.graph_width, 120)
            graph2 = self.draw_k_graph(result['demeaned'], [-2, 2], self.graph_width, 120)
            # graph3 = self.draw_k_graph(self.graph_values_demeaned, self.graph_width, 100)
            graph4 = self.draw_graph(self.graph_values, self.graph_width, 120)
            pct = len(result['values']) / 150 * 100.0
            loading_text = 'face speed...' + str(pct) + '%'
            graph = np.vstack((graph0, graph1, graph2, graph4))
            # loading_text = 'face speed'
            # graph = self.draw_graph_text(loading_text, (255, 0, 255), self.graph_width, self.graph_height)
            bpm_display = self.draw_bpm('--', self.bpm_display_width, self.bpm_display_height, result['fps'])
            segmented_face = np.zeros_like(img)
            for i in result['roi_box']:
                img = self.show_face_box(img, i)
        # 若未检测到人脸，则显示No face
        else:
            loading_text = 'No face'
            graph = self.draw_graph_text(loading_text, (255, 0, 255), self.graph_width, self.graph_height)
            bpm_display = self.draw_bpm('--', self.bpm_display_width, self.bpm_display_height, result['fps'])
            segmented_face = np.zeros_like(img)
        img = self.draw_landmarks(img, face_landmarks=result['face_landmarks'])
        segmented_face = self.draw_line(segmented_face, result['state_list'])
        bpm_display = self.draw_landmarks(bpm_display, face_emotion=f"last:{result['last_bpm']:.2f}")
        cv2.putText(bpm_display, f"displace:{result['face_displacement']:.1f}", (100, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.putText(graph, f"std:{result['std']:.1f}", (250, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        img = self.show_total(img, graph, bpm_display, segmented_face)
        return img

    def draw_k_graph(self, signal_values, value_range, graph_width, graph_height):
        '''
        :description: 根据滤波后的数据，绘制波形图。
        :param signal_values: 滤波后的数据
        :param graph_width: 波形图宽度
        :param graph_height: 波形图高度
        :return: 波形图
        '''
        value_range = [np.mean(signal_values)-15, np.mean(signal_values)+15]

        graph = np.zeros((graph_height, graph_width, 3), np.uint8)
        d_y = value_range[0] # 最低均值
        s_y = graph_height/(value_range[1]-value_range[0])  # 40 最大间隔 [90, 130]
        scale_factor_x = float(graph_width) / 320
        for i in range(0, len(signal_values) - 1):
            curr_x = int(i* scale_factor_x)
            curr_y = int(graph_height-s_y*(signal_values[i]-d_y))
            next_x = int((i + 1)* scale_factor_x)
            next_y = int(graph_height-s_y*(signal_values[i + 1]-d_y))
            cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=2)
        # cv2.putText(graph, f'-{(graph_height/s_y)+d_y}', (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        # cv2.putText(graph, f'-{d_y}', (0, graph_height-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.putText(graph, f'v:{signal_values[-1]:.2f}', (0, int(graph_height/2)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        return graph

    def draw_line(self, img, signal_values):
        graph_width, graph_height, _ = img.shape
        graph_height = graph_height/4
        scale_factor_x = float(graph_width) / 300
        # Automatically rescale vertically based on the value with largest absolute value
        max_abs = 3
        scale_factor_y = (float(graph_height) / 2.4) / max_abs
        midpoint_y = graph_height / 2
        for i in range(0, len(signal_values) - 1):
            curr_x = int(i * scale_factor_x)
            curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
            next_x = int((i + 1) * scale_factor_x)
            next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
            cv2.line(img, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=2)
        cv2.putText(img, f'scale:{scale_factor_y:.1f}', (0, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
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
        cv2.putText(graph, f'scale:{scale_factor_y:.1f}', (0, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        return graph

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

    def draw_bpm(self, bpm_str, bpm_width, bpm_height, fps):
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
        cv2.putText(bpm_display, f'fps:{fps:.1f}', (0, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        return bpm_display

    def draw_face_outline(self, img, face_outline, RPG):
        if face_outline is None:
            return img
        else:
            # 创建一个与输入图像大小相同的黑色图像，用于绘制多边形
            mask = np.zeros_like(img)
            image = img.copy()
            max_abs = max(max(RPG), -min(RPG))
            scale_factor_y = (float(180) / 2.4) / max_abs
            value = RPG[-1]*scale_factor_y+90
            # print(f'scale:{scale_factor_y:.2f}RPG: {value:.3f}')
            image[:,:,[0,2]]=0
            image[:,:,1]+=int(value)
            # 根据面部轮廓的点绘制多边形
            mask = cv2.fillPoly(mask, [face_outline], (255, 255, 255))
            segmented_face = cv2.bitwise_and(image, mask)
            return segmented_face

    def show_total(self, img, graph, bpm_display, segmented_face):
        segmented_face = segmented_face[:, 80:560, :]
        img = np.vstack((img[:, ::-1], bpm_display))
        graph = np.vstack((segmented_face, graph))
        img = np.hstack((img, graph))
        return img

    def show_face_box(self, img, face):
        # 原图上加人脸box
        f_x, f_y, f_w, f_h = face
        cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 255, 0), 2)
        cv2.putText(img, "xod", (f_x, f_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        return img

    def get_face_outline(self, landmarks):
        if landmarks is None:
            return landmarks
        # 从landmarks中获取面部轮廓的点
        list_face = []
        for p in landmarks.parts()[:17]:
            list_face.append((p.x, p.y))
        list_face.append((landmarks.parts()[28].x, landmarks.parts()[28].y))
        face_outline = np.array(list_face, dtype=np.int32)
        # 计算多边形的中心点
        center = np.mean(face_outline, axis=0)
        # 计算每个顶点与中心点之间的向量
        vectors = face_outline - center
        # 将向量缩小1/6
        scaled_vectors = vectors * 5 / 6
        scaled_face_outline = center + scaled_vectors
        return scaled_face_outline.astype(np.int32)

    def draw_landmarks(self, frame, face_box=None, face_landmarks=None, face_emotion=None):
        if face_landmarks is not None:
            # for id, pt person_dialog enumerate(face_landmarks):
            #     red = 255 if id == 21 or id == 22 or id == 27 or id == 39 or id == 42 else 0
            #     cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, [red, 0, 255], thickness=-1)
            for i in range(68):
                x, y = face_landmarks.part(i).x, face_landmarks.part(i).y
                red = 255 if id == 21 or id == 22 or id == 27 or id == 39 or id == 42 else 0
                cv2.circle(frame, (x, y), 2, [red, 0, 255], thickness=-1)
        if face_box:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if face_emotion:
            if face_box:
                x, y, w, h = face_box
            else:
                x, y = 100, 100
            cv2.putText(frame, face_emotion, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0, 0, 255], 2)
        return frame
