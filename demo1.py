import os
import cv2 as cv
import numpy as np
import LPRLite as lpr


# 读取图像，解决imread不能读取中文路径的问题


def cv_imread(filePath):
    cv_img = cv.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


model = lpr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")


# grr = cv_imread("image/津E22602.jpg")
# grr = cv_imread("image/辽B57368.jpg")
# grr = cv_imread("image/辽B99999.jpg")
# grr = cv_imread("image/辽H05B08.jpg")
# grr = cv_imread("image/辽BNR378.jpg")
# grr = cv_imread("image/鄂A85044.jpg")
# grr = cv_imread("image/鄂AE3456.jpg")
# grr = cv_imread("image/鄂AV6985.jpg")
# grr = cv_imread("image/闽A339U9.jpg")
# grr = cv_imread("image/闽A661J3.jpg")
# grr = cv_imread("image/闽A763H5.jpg")
# grr = cv_imread("image/闽AM902X.jpg")
# grr = cv_imread("image/闽AWF197.jpg")
# grr = cv_imread("image/闽D00169.jpg")
# grr = cv_imread("image/闽DA9077.jpg")
# grr = cv_imread("image/闽DQ332Y.jpg")
# grr = cv_imread("image/闽DT0572.jpg")
# grr = cv_imread("image/闽DTC775.jpg")
# grr = cv_imread("image/闽DTK089.jpg")
# grr = cv_imread("image/闽FHK691.jpg")
# grr = cv_imread("image/闽HB1508.jpg")
# grr = cv_imread("image/闽J12345.jpg")
# grr = cv_imread("image/闽K71537.jpg")
# grr = cv_imread("image/陕AGL110.jpg")
# grr = cv_imread("image/鲁BQG527.jpg")
# grr = cv_imread("image/鲁C08888.jpg")
# grr = cv_imread("image/鲁E00000.jpg")
# grr = cv_imread("image/鲁ENB911.jpg")
# grr = cv_imread("image/鲁JD9309.jpg")
# grr = cv_imread("image/鲁LC1336.jpg")
# grr = cv_imread("image/黑A16341.jpg")
# grr = cv_imread("image/黑AB4444.jpg")
grr = cv_imread("image/京PC5U22.jpg")
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
