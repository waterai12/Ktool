#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 11:31
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : face_tool.py
import cv2
import dlib
import openface
import numpy as np

class FaceDetector:
    def __init__(self, detection_type='box'):
        '''
        一个基于OpenCV的人脸检测类, 根据需要选择返回人脸框、人脸图像或绘制人脸框的原始图像
        :param detection_type: ['crop', 'box', 'draw']
        '''
        self.output_video_size = (224, 224)

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.face_detector_dilb = dlib.get_frontal_face_detector()
        self.get_landmarks = dlib.shape_predictor(r"C:\Users\ykjiang2\PycharmProjects\FER_iflytek\corecode\ktool\shape_predictor_68_face_landmarks.dat")

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
    def get_landmarks_openface(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.face_detector_dilb(gray, 0)
        # 遍历检测到的人脸
        if len(rects) == 1:
            # 使用关键点检测器定位人脸关键点
            face_landmarks = self.get_landmarks(gray, rects[0])
            return face_landmarks, [rects[0].left(), rects[0].top(), rects[0].width(), rects[0].height()]
        elif len(rects)>1:
            # 使用关键点检测器定位人脸关键点
            print("检测到多个人脸")
            # 检测到多个人脸
            max_size, max_face_coords, max_idx = 0, [0, 0, 0, 0], 0
            for idx, rect in enumerate(rects):
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                if w * h > max_size:
                    max_size = w * h
                    max_face_coords = [x, y, w, h]
                    max_idx = idx
            face_landmarks = self.get_landmarks(gray, rects[max_idx])
            return face_landmarks, max_face_coords
        else:
            face_landmarks = None
            print("没检测到人脸")
            return face_landmarks, [0, 0, 0, 0]

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

    def draw_img(self, frame, face_box=None, face_landmarks=None, face_emotion=None):
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
            x, y, w, h = face_box
            cv2.putText(frame, face_emotion, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, [0, 0, 255], 2)
        return frame

    def convert_landmarks_np(self, face_landmarks):
        return np.array([(p.x, p.y) for p in face_landmarks.parts()])
