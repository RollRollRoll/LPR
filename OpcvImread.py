# -*- coding: utf-8 -*-
import cv2
import numpy as np



def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


if __name__ == '__main__':
    path = 'C:/Users/KG/Desktop/images/津RB7992.jpg'
    img = cv_imread(path)
    cv2.namedWindow('lena', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('lena', img)
    k = cv2.waitKey(0)