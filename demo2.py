from lpr_py3 import pipline
from lpr_py3 import pipline_new as piplineNew
from xml.dom.minidom import Document
from xml.dom.minidom import parse
import os
import codecs
import numpy as np
import cv2


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


# image = cv_imread("image/京PC5U22.jpg")
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
image = cv_imread("image/湘A97971.jpg")
# image = cv_imread("dataset/0.jpg")
image, res = piplineNew.SimpleRecognizePlateByE2E(image)
print(res)
cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyAllWindows()


# -------------------------------------------------------------------
# dom = parse("data.xml")
# collection = dom.documentElement
# plates = collection.getElementsByTagName("plate")
# doc = Document()
# root = doc.createElement("root")
# enumber = 0
# for i, file in enumerate(os.listdir("image/")):
#     grr = cv_imread("image/" + file)
#     print(i)
#     print("File Name:", file[:-4])
#     for plate in plates:
#         name = plate.getElementsByTagName("Name")[0]
#         if file[:-4] == name.childNodes[0].data:
#             pstr = plate.getElementsByTagName("pstr")
#             break
#     else:
#         print("文件不存在")
#         exit(1)
#
#     image, res_set = piplineNew.SimpleRecognizePlateByE2E(grr)
#     cnumber = 0
#     for placeholder, res, confidence in res_set:
#         print("res:", res)
#         print("confidence:", confidence)
#         if confidence > 0.8:
#             for s in pstr:
#                 if res == s.childNodes[0].data:
#                     cnumber = cnumber + 1
#                     break
#             else:
#                 break
#     if cnumber != len(pstr):
#         enumber = enumber + 1
#         plate = doc.createElement("plate")
#         Name = doc.createElement("Name")
#         name_str = doc.createTextNode(file[:-4])
#         Name.appendChild(name_str)
#         plate.appendChild(Name)
#         for placeholder, res, confidence in res_set:
#             ps = doc.createElement("pstr")
#             pstr_str = doc.createTextNode(res)
#             ps.appendChild(pstr_str)
#             con = doc.createElement("confidence")
#             con_str = doc.createTextNode(str(confidence))
#             con.appendChild(con_str)
#             plate.appendChild(ps)
#             plate.appendChild(con)
#         root.appendChild(plate)
# doc.appendChild(root)
# print('\n')
# print(i)
# print(enumber)
#
# f = codecs.open("Error.xml", 'w', 'utf-8')
# f.write(doc.toprettyxml(indent='  '))
# f.close()
