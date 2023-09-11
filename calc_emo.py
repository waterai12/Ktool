#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 11:22
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : calc_emo.py
import numpy as np
import cv2
import math


def time_conut(face_emotion, face_emotion_list):
    face_emotion_list.append(face_emotion)
    if len(face_emotion_list)>5:
        face_emotion_list.pop(0)
    print(face_emotion_list)
    if face_emotion_list.count('head speed up')>2:
        return face_emotion_list, 'head speed up'
    if face_emotion_list.count('anxious')>2:
        return face_emotion_list, 'anxious'
    if face_emotion_list.count('happy')>2:
        return face_emotion_list, 'happy'
    if face_emotion_list.count('surprised')>2:
        return face_emotion_list, 'surprised'
    return face_emotion_list, 'normal'


def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    # 单位转换：将弧度转换为度
    Y = int((pitch / math.pi) * 180)
    X = int((yaw / math.pi) * 180)
    Z = int((roll / math.pi) * 180)
    if Y > 0:
        Y = 180 - Y
    else:
        Y = - (180 + Y)
    return Y, X, Z


def get_pose_estimation(img_size, landmark_shape):
    image_points = np.array([
        (landmark_shape[30][0], landmark_shape[30][1]),  # Nose tip
        (landmark_shape[8][0], landmark_shape[8][1]),  # Chin
        (landmark_shape[36][0], landmark_shape[36][1]),  # Left eye left corner
        (landmark_shape[45][0], landmark_shape[45][1]),  # Right eye right corne
        (landmark_shape[48][0], landmark_shape[48][1]),  # Left Mouth corner
        (landmark_shape[54][0], landmark_shape[54][1])  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals

    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    pitch, yaw, roll = get_euler_angle(rotation_vector)
    return pitch, yaw, roll


def cal_static_face_au(face_landmarks, face_boxes):
    face_landmarks = list(map(lambda p: (p.x, p.y), face_landmarks.parts()))
    face_landmarks = np.array(face_landmarks, dtype=np.float)

    ##############################
    # 头部特征
    ##############################
    head_angle = get_pose_estimation((480, 640, 3), face_landmarks)

    ##############################
    # 关键点归一化
    ##############################
    x, y, w, h = face_boxes
    face_landmarks[:, 0] = (face_landmarks[:, 0] - x - w / 2) / w
    face_landmarks[:, 1] = -(face_landmarks[:, 1] - y - h / 2) / h

    ##############################
    # 眼部特征
    ##############################
    # 计算眼睛开合大小[0, 20] [闭, 睁]
    left_eye = face_landmarks[37][1] + face_landmarks[38][1] - face_landmarks[40][1] - face_landmarks[41][1]
    right_eye = face_landmarks[42][1] + face_landmarks[43][1] - face_landmarks[47][1] - face_landmarks[46][1]
    eye_open = int((left_eye + right_eye - 0.09)*200)  # [ 0, 20]

    ################################
    # 嘴部特征
    ################################
    # 1、计算嘴巴开合大小[0, 20] [闭, 开]
    mouse_up = face_landmarks[61][1] + face_landmarks[62][1] + face_landmarks[63][1]
    mouse_down = face_landmarks[67][1] + face_landmarks[66][1] + face_landmarks[65][1]
    mouse_open = int((mouse_up - mouse_down) * 50)
    # 2、计算抿嘴程度[0, 20] [自然, 抿嘴]
    mouse_height = face_landmarks[50][1] + face_landmarks[51][1] + face_landmarks[52][1] - (
                face_landmarks[56][1] + face_landmarks[57][1] + face_landmarks[58][1])
    mouse_width = face_landmarks[64][0] + face_landmarks[54][0] - (face_landmarks[60][0] + face_landmarks[48][0])
    mouse_close = int((mouse_width / mouse_height - 1.9) * 10)
    # 3、计算嘴角上扬角度，并以上下嘴唇作为参照值
    mouse_angle = face_landmarks[[48, 54], 1].mean() - face_landmarks[[61, 62, 63, 65, 66, 67], 1].mean()
    mouse_angle = int(mouse_angle * 300)

    ##############################
    # 眉毛特征
    ##############################
    # 1、计算皱眉程度[0, 20]
    # 计算眉心距离
    eyebrow2nose_norm = np.linalg.norm(face_landmarks[22] - face_landmarks[27]) + np.linalg.norm(
        face_landmarks[21] - face_landmarks[27])
    # 计算眼角和鼻梁的距离，作为眉心距离的参照值，
    eye2nose_norm = np.linalg.norm(face_landmarks[39] - face_landmarks[27]) + np.linalg.norm(
        face_landmarks[42] - face_landmarks[27])
    # 皱眉根据头部姿态进行动态调整
    brow_frown = int((1 - eyebrow2nose_norm / eye2nose_norm) * 100)
    # 2、计算眉毛高度[0, 20]
    left_brow_high = face_landmarks[17:22, 1].mean() - face_landmarks[[36, 39], 1].mean()
    right_brow_high = face_landmarks[22:27, 1].mean() - face_landmarks[[42, 45], 1].mean()
    brow_high = int((left_brow_high + right_brow_high - 0.34) * 300)

    static_au = {'eye_open': eye_open,
                 'mouse_open': mouse_open, 'mouse_close': mouse_close, 'mouse_angle': mouse_angle,
                 'head': head_angle,
                 'brow_high': brow_high, 'brow_frown': brow_frown
                 }
    print(f'静态特征 \n 眼睛开合:{eye_open} '
          f'嘴巴开合:{mouse_open} 抿嘴:{mouse_close} 嘴角弧度:{mouse_angle}  '
          f'头部：{head_angle}')
    return static_au


def cal_dynamics_face_au(face_landmarks, face_boxes, head_angle_old, brow_high_avg=0,  brow_flow_avg=0):
    face_landmarks = list(map(lambda p: (p.x, p.y), face_landmarks.parts()))
    face_landmarks = np.array(face_landmarks, dtype=np.float)

    ##############################
    # 头部特征
    ##############################
    head_angle = get_pose_estimation((480, 640, 3), face_landmarks)
    ##############################
    # 关键点归一化
    ##############################
    x, y, w, h = face_boxes
    face_landmarks[:, 0] = (face_landmarks[:, 0] - x - w / 2) / w
    face_landmarks[:, 1] = -(face_landmarks[:, 1] - y - h / 2) / h
    ##############################
    # 眉毛特征
    ##############################
    # 1、计算皱眉程度[0, 20]
    # 计算眉心距离
    eyebrow2nose_norm = np.linalg.norm(face_landmarks[22] - face_landmarks[27]) + np.linalg.norm(
        face_landmarks[21] - face_landmarks[27])
    # 计算眼角和鼻梁的距离，作为眉心距离的参照值，
    eye2nose_norm = np.linalg.norm(face_landmarks[39] - face_landmarks[27]) + np.linalg.norm(
        face_landmarks[42] - face_landmarks[27])
    # 皱眉根据头部姿态进行动态调整
    brow_frown = int((1 - eyebrow2nose_norm / eye2nose_norm)*100)
    dynamic_brow_frown = brow_flow_avg-brow_frown
    brow_flow_avg = (0.5 * brow_frown + brow_flow_avg)/1.5
    # 2、计算眉毛高度[0, 20]
    left_brow_high = face_landmarks[17:22, 1].mean() - face_landmarks[[36, 39], 1].mean()
    right_brow_high = face_landmarks[22:27, 1].mean() - face_landmarks[[42, 45], 1].mean()
    brow_high = int((left_brow_high + right_brow_high - 0.34) * 300)
    dynamic_brow_high = brow_high-brow_high_avg
    brow_high_avg = (0.5 * brow_high + brow_high_avg)/1.5
    dynamics_au = {'dy_brow_frown': dynamic_brow_frown, 'dy_brow_high': dynamic_brow_high}
    head_angle_speed = [abs(head_angle[0]-head_angle_old[0]), abs(head_angle[1]-head_angle_old[1]), abs(head_angle[2]-head_angle_old[2])]
    print(f"眉毛高度:{int(dynamic_brow_high)}(残差) {brow_high}(当前)  {int(brow_high_avg)}(全局) |  "
          f"皱眉:{int(dynamic_brow_frown)}(残差) {brow_frown}(当前)  {int(brow_flow_avg)}(全局)")
    return dynamics_au, brow_high_avg, brow_flow_avg, head_angle_speed, head_angle


def au2emotion(feature):
    # 提取au特征
    eye_open = feature['eye_open']
    mouse_open, mouse_close, mouse_angle = feature['mouse_open'], feature['mouse_close'], feature['mouse_angle']
    st_brow_high, st_brow_frown = feature['brow_high'], feature['brow_frown']
    dy_brow, dy_brow_high = feature['dy_brow_frown'], feature['dy_brow_high']

    head = feature['head']

    ##############################
    # 情感定量
    ##############################
    emo_id, emotion, quantify = 0, ['normal', 'happy', 'anxious', 'surprised'], ''
    # 判断惊讶
    # 眼睛开合度大，嘴巴开合度大，眉毛高度高, 嘴角度小
    if eye_open > 7 and mouse_open > 8 and mouse_open + dy_brow_high + eye_open > 22 and mouse_angle<5:
        emo_id = 3
        quantify = "+" + str(mouse_open + dy_brow_high + eye_open)

    # 判断焦躁
    # 眼睛开合度小 and 眉毛高度低，(皱眉 or 抿嘴)
    print(int(dy_brow+dy_brow_high-mouse_close*1.8), 8-head[0]*0.5)
    if eye_open < 7 and mouse_close < 20 and dy_brow+dy_brow_high-mouse_close*1.8<-10 and head[0]<23 and mouse_angle<(12-head[0]*0.5):

    # if eye_open < 7 and dy_brow+dy_brow_high<-11 and mouse_close<5 and mouse_angle<3:
        emo_id = 2
        quantify = "+" + str(dy_brow)
    # 判断微笑
    # 微笑：眼睛开合度较小， 嘴角幅度大，
    # 抬头大笑：眼睛开合度很小， 嘴巴开合度很大
    # 礼貌性微笑：眼睛开合度大，嘴角上扬
    if (eye_open < 13 and mouse_angle > 3 and mouse_open>2) or (eye_open < 2 and mouse_open > 10) or \
       (eye_open > 10 and mouse_angle > 4 and mouse_open>=6):
        emo_id = 1

        quantify = "+" + str(mouse_angle)

    # print(mouse_close + brow*1.2 - (brow_high*0.3))
    # if -20<head[0]<20:
    #     emo_id=0
    return emotion[emo_id]# + quantify
