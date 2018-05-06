# coding=utf-8
from . import detect
from . import finemapping  as  fm
from . import cache
from . import finemapping_vertical as fv
from . import typeDistinguish as td
from . import segmentation
from . import e2e_new as e2e

import cv2

import time
import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import sys
import imp

imp.reload(sys)
fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0);


# 寻找车牌左右边界

def find_edge(image):
    sum_i = image.sum(axis=0)
    sum_i = sum_i.astype(np.float)
    sum_i /= image.shape[0] * 255
    # print sum_i

    start = 0;
    end = image.shape[1] - 1

    for i, one in enumerate(sum_i):
        if one > 0.4:
            start = i;
            if start - 3 < 0:
                start = 0
            else:
                start -= 3

            break;
    for i, one in enumerate(sum_i[::-1]):

        if one > 0.4:
            end = end - i;
            if end + 4 > image.shape[1] - 1:
                end = image.shape[1] - 1
            else:
                end += 4
            break
    return start, end


# 垂直边缘检测
def verticalEdgeDetection(image):
    image_sobel = cv2.Sobel(image.copy(), cv2.CV_8U, 1, 0)
    # image = auto_canny(image_sobel)

    # img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT
    # canny_image  = auto_canny(image)
    flag, thres = cv2.threshold(image_sobel, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    print(flag)
    flag, thres = cv2.threshold(image_sobel, int(flag * 0.7), 255, cv2.THRESH_BINARY)
    # thres = simpleThres(image_sobel)
    kernal = np.ones(shape=(3, 15))
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernal)
    return thres


# 确定粗略的左右边界
def horizontalSegmentation(image):
    thres = verticalEdgeDetection(image)
    # thres = thres*image
    head, tail = find_edge(thres)
    # print head,tail
    # cv2.imshow("edge",thres)
    tail = tail + 5
    if tail > 135:
        tail = 135
    image = image[0:35, head:tail]
    image = cv2.resize(image, (int(136), int(36)))
    return image


# 打上boundingbox和标签
def drawRectBox(image, rect, addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2,
                  cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    # draw.text((int(rect[0]+1), int(rect[1]-16)), addText.decode("utf-8"), (255, 255, 255), font=fontC)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)

    return imagex


def SimpleRecognizePlateByE2E(image):
    t0 = time.time()
    images = detect.detectPlateRough(image, image.shape[0], top_bottom_padding_rate=0.1)
    res_set = []
    for j, plate in enumerate(images):
        plate, rect, origin_plate = plate
        # 测试代码s
        cv2.imshow("plate"+str(j), plate)
        cv2.imshow("origin_plate"+str(j), origin_plate)
        # plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        # 测试代码e

        plate = cv2.resize(plate, (136, 36 * 2))
        # 测试代码s
        cv2.imshow("plate_resize"+str(j), plate)
        print("plate_resize.shape:", plate.shape)
        # 测试代码e

        res, confidence = e2e.recognizeOne(origin_plate)
        # 测试代码s
        print("old res:", res, "old confidence:", confidence)
        t1 = time.time()
        # 测试代码e

        ptype = td.SimplePredict(plate)
        # 测试代码s
        print("ptype:", ptype)
        # 测试代码e

        if ptype > 0 and ptype < 5:
            # pass
            plate = cv2.bitwise_not(plate)
            # 测试代码s
            cv2.imshow("plate_bitwise_not", plate)
            # 测试代码e

        image_rgb = fm.findContoursAndDrawBoundingBox(plate)
        # 测试代码s
        cv2.imshow("image_rgb_findContoursAndDrawBoundingBox"+str(j), image_rgb)
        # 测试代码e

        image_rgb = fv.finemappingVertical(image_rgb)
        # 测试代码s
        cv2.imshow("image_rgb_finemappingVertical"+str(j), image_rgb)
        # 测试代码e

        # cache.verticalMappingToFolder(image_rgb)
        # cv2.imwrite("./" + str(j) + ".jpg", image_rgb)
        res, confidence = e2e.recognizeOne(image_rgb)
        # 测试代码s
        print("new res:", res, "new confidence:", confidence)
        # 测试代码e

        res_set.append([[], res, confidence])
        if confidence > 0.7:
            image = drawRectBox(image, rect, res + " " + str(round(confidence, 3)))
    return image, res_set