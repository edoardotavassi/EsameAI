import cv2
import numpy as np
from numpy.lib.type_check import imag

# from tensorflow.keras.models import load_model


"""def initializePredictionModel():
    model = load_model("models/model.h5")
    return model

"""
# image processing
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray scale
    imgGray = cv2.fastNlMeansDenoising(imgGray)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # gaussian blur
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, 1, 1, 11, 4
    )  # apply adaptive threshold
    return imgThreshold


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:  # shape rectangle
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def splitBoxes(img):
    rows = np.vsplit(img, 7)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 7)
        for box in cols:
            boxes.append(box)
    return boxes


"""def getPredictions(boxes, model):
    result = []
    for image in boxes:
        # prepare image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.asarray(image)
        image = cv2.resize(image, (28, 28))
        image = image / 255
        image = image.reshape(1, 784)
        cv2.imshow("ciao", image)
        # prediction
        predictions = model.predict(image)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        print(classIndex, probabilityValue)
        # save value
        result.append(classIndex[0])
    return result
"""

# stack imgaes to one window
def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale
                    )
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y],
                        (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x],
                    (imgArray[0].shape[1], imgArray[0].shape[0]),
                    None,
                    scale,
                    scale,
                )
            if len(imgArray[0].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
