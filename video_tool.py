#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 10:48
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : video_tool.py
import cv2
import time
from datetime import datetime
from Ktool.file_tool import write_json
def read_write_video(input_path, output_path, funtion, frame_idx_list, timestamp_list, video_size=None):
# def read_write_video(input_path, output_path, funtion, video_size=None):
    # 读取视频文件
    video = cv2.VideoCapture(input_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 初始化计时器
    total_time = 0.0

    # 获取视频的帧速率和尺寸
    fps = int(video.get(cv2.CAP_PROP_FPS))
    # fps = 30
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置视频编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建一个新的视频文件，用于保存处理后的视频
    if video_size:
        output_video = cv2.VideoWriter(output_path, fourcc, fps, video_size)
    else:
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_idx = 0
    timestamp = 0
    # 循环读取视频帧并处理
    while True:
        # 读取一帧
        ret, frame = video.read()
        # 如果读取到帧，则进行处理
        if ret:
            frame_idx+=1
            # 初始化计时器
            start_time = time.time()
            if frame_idx in frame_idx_list:
                timestamp = timestamp_list[frame_idx_list.index(frame_idx)]
            frame = funtion(frame, timestamp)
            # frame = funtion(frame)
            # cv2.imshow('video', frame)
            # 停止计时器并计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            # print("Processing time: {:.2f} seconds".format(processing_time))
            total_time += processing_time
            # # 等待按键按下
            # if cv2.waitKey(1) == ord('q'):
            #     break
            # 将处理后的帧写入新视频文件
            output_video.write(frame)
        else:
            print("Average frame processing time: ", total_time / total_frames)
            # 如果读取到视频结尾，则退出循环
            break

    # 释放资源
    video.release()
    output_video.release()


def read_video_show(input_path, funtion, video_size=None):
    # 读取视频文件
    video = cv2.VideoCapture(input_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 初始化计时器
    total_time = 0.0
    # 循环读取视频帧并处理
    while True:
        # 读取一帧
        ret, frame = video.read()

        # 如果读取到帧，则进行处理
        if ret:
            # 初始化计时器
            start_time = time.time()
            frame = funtion(frame)
            # 停止计时器并计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            # print("Processing time: {:.2f} seconds".format(processing_time))
            total_time += processing_time
            # 将处理后的帧写入新视频文件
            cv2.imshow('video', frame)

            # 等待按键按下
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            # 如果读取到视频结尾，则退出循环
            # 计算平均每一帧的处理时间
            print("Average frame processing time: ", total_time / total_frames)
            break


def read_capture_show(input_path, funtion):
    capture = cv2.VideoCapture(0)
    # capture.set(3, 480)
    # 获取视频的帧速率和尺寸
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            frame = funtion(frame)
            # 将处理后的帧写入新视频文件
            cv2.imshow('video', frame)
            # 停止计时器并计算处理时间
            end_time = time.time()
            processing_time = end_time - start_time
            # print("{:.3f} seconds Processing time".format(processing_time))
            total_time += processing_time
            # 等待按键按下
            if cv2.waitKey(1) == ord('q'):
                break
    # 释放资源
    capture.release()


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
            print("Processing time: {:.3f} seconds".format(processing_time))
            total_time += processing_time
            # 等待按键按下
            if cv2.waitKey(1) == ord('q'):
                break
    # 释放资源
    write_json(output_path.replace('mp4', 'json'), output_predict)
    capture.release()
    output_video.release()


def read_capture_save(funtion, output_path):
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
            frame_idx += 1
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
            if processing_time > 0.033:
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


def read_video_save(funtion, input_path, output_path, a):
    capture = cv2.VideoCapture(input_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # capture.set(3, 480)
    # 获取视频的帧速率和尺寸

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_predict = {}
    # 初始化计时器
    total_time, seconds = 0.0, 0
    frame_idx = 0
    # 循环读取视频帧并处理
    while True:
        # 读取一帧
        ret, frame = capture.read()
        desired_fps = 80
        # 如果读取到帧，则进行处理
        if ret:
            frame_idx += 1
            # 初始化计时器
            start_time = time.time()
            enable = funtion(frame)
            # 将处理后的帧写入新视频文件
            # cv2.imshow('video', frame)
            if frame_idx in a:
            # if datetime.now().second != seconds:
                seconds = datetime.now().second
                enable['frame_idx'] = frame_idx
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4], enable['pulse'])
                output_predict[datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]] = enable

            # 停止计时器并计算处理时间
            time.sleep(1 / desired_fps)
            end_time = time.time()
            processing_time = end_time - start_time
            # 按照设定的帧率显示视频
            # if processing_time > 0.033:
            #     print("\n{:.3f} seconds (有点卡顿，心率准确度受影响)\n".format(processing_time))
            # else:
            #     print("{:.3f} seconds (Processing time)".format(processing_time))
            total_time += processing_time
            # 等待按键按下
            # if cv2.waitKey(1) == ord('q') :
            #     break
        else:
            # 如果读取到视频结尾，则退出循环
            # 计算平均每一帧的处理时间
            print("Average frame processing time: ", total_time / total_frames)
            break
    # 释放资源
    write_json(output_path, output_predict)
    capture.release()
