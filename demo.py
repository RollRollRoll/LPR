import os
import time
import codecs
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from xml.dom.minidom import Document
import LPRLite as pr
import cv2
import numpy as np

fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


def SpeedTest(image_path):
    grr = cv2.imread(image_path)
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for x in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0) / 20.0
    print("Image size :" + str(grr.shape[1]) + "x" + str(grr.shape[0]) + " need " + str(round(t * 1000, 2)) + "ms")


def drawRectBox(image, rect, addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2,
                  cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText.decode("utf-8"), (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


# grr = cv2.imread("images_rec/2_.jpg")
# model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
# for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(grr):
#     if confidence > 0.7:
#         image = drawRectBox(grr, rect, pstr + " " + str(round(confidence, 3)))
#         print
#         "plate_str:"
#         print
#         pstr
#         print
#         "plate_confidence"
#         print
#         confidence
#
# cv2.imshow("image", image)
# cv2.waitKey(0)
#
# SpeedTest("images_rec/2_.jpg")


# ----------------------------------------------------------------------------------------------------------------------
doc = Document()
root = doc.createElement("root")
model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
for i, file in enumerate(os.listdir("image/")):
    print(i)
    grr = cv_imread("image/" + file)
    plate = doc.createElement("plate")
    index = doc.createElement("index")
    index_str = doc.createTextNode(str(i))
    index.appendChild(index_str)
    fileName = doc.createElement("Name")
    fileName_str = doc.createTextNode(file[:-4])
    fileName.appendChild(fileName_str)
    plate.appendChild(index)
    plate.appendChild(fileName)
    for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(grr):
        platestr = doc.createElement("pstr")
        if confidence > 0.8:
            print("file name:", file[:-4])
            print("plate_str:", pstr)
            print("plate_confidence:", confidence)
            if (file[:-4] == pstr):
                pstr_str = doc.createTextNode(pstr)
                platestr.appendChild(pstr_str)
            else:
                pstr_str = doc.createTextNode("Error " + pstr)
                platestr.appendChild(pstr_str)
        else:
            pstr_str = doc.createTextNode("Error " + pstr + "confidence: " + str(confidence))
            platestr.appendChild(pstr_str)
        plate.appendChild(platestr)
    print('\n')
    root.appendChild(plate)
doc.appendChild(root)

f = codecs.open("data.xml", 'w', 'utf-8')
f.write(doc.toprettyxml(indent='  '))
f.close()
