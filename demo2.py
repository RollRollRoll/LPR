from lpr_py3 import pipline
from lpr_py3 import pipline_new as piplineNew
import os
import numpy as np
import cv2


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


# image = cv_imread("image/津E22602.jpg")
# image = cv_imread("image/辽H05B08.jpg")
# image = cv_imread("image/辽BNR378.jpg")
# image = cv_imread("image/鄂A85044.jpg")
# image = cv_imread("image/鄂AE3456.jpg")
# image = cv_imread("image/鄂AV6985.jpg")
# image = cv_imread("image/闽A339U9.jpg")
# image = cv_imread("image/闽A661J3.jpg")
# image = cv_imread("image/闽A763H5.jpg")
# image = cv_imread("image/闽AM902X.jpg")
# image = cv_imread("image/闽AWF197.jpg")
# image = cv_imread("image/闽D00169.jpg")
# image = cv_imread("image/闽DA9077.jpg")
# image = cv_imread("image/闽DQ332Y.jpg")
# image = cv_imread("image/闽DT0572.jpg")
# image = cv_imread("image/闽DTC775.jpg")
# image = cv_imread("image/闽DTK089.jpg")
# image = cv_imread("image/闽FHK691.jpg")
# image = cv_imread("image/闽HB1508.jpg")
# image = cv_imread("image/闽J12345.jpg")
# image = cv_imread("image/闽K71537.jpg")
# image = cv_imread("image/陕AGL110.jpg")
# image = cv_imread("image/鲁BQG527.jpg")
# image = cv_imread("image/鲁C08888.jpg")
# image = cv_imread("image/鲁E00000.jpg")
# image = cv_imread("image/鲁ENB911.jpg")
# image = cv_imread("image/鲁JD9309.jpg")
# image = cv_imread("image/鲁LC1336.jpg")
# image = cv_imread("image/黑A16341.jpg")
# image = cv_imread("image/黑AB4444.jpg")
# image, res = pipline.SimpleRecognizePlateByE2E(image)
# print(res)
# cv2.imshow("image", image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# -------------------------------------------------------------------
# image = cv_imread("image/黑AB4444.jpg")
# # image = cv_imread("dataset/0.jpg")
# image, res = pipline.SimpleRecognizePlateByE2E(image)
# print(res)
# cv2.imshow("image", image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# -------------------------------------------------------------------
# files = open("data.txt", 'w')
# n = 0
# for i, file in enumerate(os.listdir("image/")):
#     grr = cv_imread("image/" + file)
#     print(i)
#     img, res_set = pipline.SimpleRecognizePlateByE2E(grr)
#     for res in res_set:
#         pstr = res[1]
#         confidence = res[2]
#         if confidence > 0.8:
#                 print("plate_name", file[:-4])
#                 print("plate_str", pstr)
#                 print("plate_confidence", confidence)
#                 l = "Number:"+str(i)+" "+file[:-4] + ":  " + pstr + " Confidence:"+str(confidence)
#                 files.write(l)
#                 if (pstr == file[:-4]):
#                     n = n + 1
#                     files.write('\n')
#                 else:
#                     print("error")
#                     files.write("  ERROR" + '\n')
#         print('\n')
#
# print(n)
# print(i)


# -------------------------------------------------------------------
image = cv_imread("image/京V02633.jpg")
# image = cv_imread("dataset/0.jpg")
image, res = piplineNew.SimpleRecognizePlateByE2E(image)
print(res)
cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyAllWindows()


# -------------------------------------------------------------------
# files = open("data.txt", 'w')
# n = 0
# for i, file in enumerate(os.listdir("image/")):
#     grr = cv_imread("image/" + file)
#     print(i)
#     img, res_set = piplineNew.SimpleRecognizePlateByE2E(grr)
#     for res in res_set:
#         pstr = res[1]
#         confidence = res[2]
#         if confidence > 0.8:
#             print("plate_name", file[:-4])
#             print("plate_str", pstr)
#             print("plate_confidence", confidence)
#             l = "Number:" + str(i) + " " + file[:-4] + ":  " + pstr + " Confidence:" + str(confidence)
#             files.write(l)
#             if (pstr == file[:-4]):
#                 n = n + 1
#                 files.write('\n')
#             else:
#                 print("error")
#                 files.write("  ERROR" + '\n')
#         print('\n')
#
# print(n)
# print(i)
