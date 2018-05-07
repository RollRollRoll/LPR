import os
import cv2 as cv
import numpy as np
import LPRLite as lpr


# 读取图像，解决imread不能读取中文路径的问题


def cv_imread(filePath):
    cv_img = cv.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


model = lpr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")

grr = cv_imread("image/京A88731.jpg")
for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(grr):
    if confidence > 0.8:
        print("plate_str", pstr)
        print("plate_confidence", confidence)


# files = open("data.txt", 'w')
# n = 0
# for i, file in enumerate(os.listdir("image/")):
#     grr = cv_imread("image/" + file)
#     print(i)
#     for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(grr):
#         if confidence > 0.8:
#             print("plate_name", file[:-4])
#             print("plate_str", pstr)
#             print("plate_confidence", confidence)
#             l = "Number:"+str(i)+" "+file[:-4] + ":  " + pstr + " Confidence:"+str(confidence)
#             files.write(l)
#             if (pstr == file[:-4]):
#                 n = n + 1
#                 files.write('\n')
#             else:
#                 print("error")
#                 files.write("  ERROR" + '\n')
#     print('\n')
# print(n)
# print(i)
