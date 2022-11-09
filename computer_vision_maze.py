from numpy.matrixlib.defmatrix import matrix
from utils import *
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from tensorflow import expand_dims
from tensorflow.keras.models import load_model


pathImage = "./img/maze_4.jpg"

class ImageAnalysis :
    
    def __init__(self, pathImage):
        self.path = pathImage
        self.model = load_model("models/model.h5")
        self.mapp = pd.read_csv("./datasets/new_mapping.txt", delimiter=" ", index_col=0, header=None, squeeze=True)
    
    #utils
    def inside(self, r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        return (x1 > x2) and (y1 > y2) and (x1 + w1 < x2 + w2) and (y1 + h1 < y2 + h2)


    def wrap_digit(self, rect, img_w, img_h):
        x, y, w, h = rect
        x_center = x + w // 2
        y_center = y + h // 2
        if h > w:
            w = h
            x = x_center - (w // 2)
        else:
            h = w
            y = y_center - (h // 2)

        padding = 5
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        if x < 0:
            x = 0
        elif x > img_w:
            x = img_w

        if y < 0:
            y = 0
        elif y > img_h:
            y = img_h

        if x + w > img_w:
            w = img_w - x

        if y + h > img_h:
            h = img_h - y

        return x, y, w, h
    
    def flatten(l):
        return [item for sublist in l for item in sublist]

    def sortBoundingBox(self, rectangles): 
        _,temp,_,_ = rectangles[0]
        tempArr=[]
        outArr=[]
        for r in rectangles:
            x,y,_,_ = r
            if y < temp-15 or y > temp + 15:
                tempArr.sort(key=lambda y: y[0])
                outArr.insert(0,tempArr.copy())
                tempArr.clear()
            tempArr.append(r)
            temp = y
            
        tempArr.sort(key=lambda y: y[0])
        outArr.insert(0,tempArr.copy())
        tempArr.clear()
        """for r in outArr:
            tempArr.append(r)"""
        return flatten(outArr)
    
    
    def digital_maze_detection(self):
        img = cv2.imread(self.path,0)
        img = imutils.resize(img, height=500)
        self.imgOriginal = img.copy()
        plt.imshow(self.imgOriginal,cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imshow(img,cmap='gray', vmin=0, vmax=255)
        plt.show()
        
        _,threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        plt.imshow(threshold,cmap='gray', vmin=0, vmax=255)
        plt.show()
        
        erode_kernel = np.ones((2, 2), np.uint8)
        threshold = cv2.erode(threshold, erode_kernel, threshold, iterations=2)
        plt.imshow(threshold,cmap='gray', vmin=0, vmax=255)
        plt.show()
        return self.detection(threshold, True)
        
    def detection(self, img, digital):
        outImg = img.copy()
        outImg = cv2.cvtColor(outImg, cv2.COLOR_GRAY2BGR)
        rectangles = []
        contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        outArr=[]
        img_h, img_w = img.shape[:2]
        img_area = img_w * img_h
        for c in contours:
            if not digital:
                a = cv2.contourArea(c)
                if a >= 0.98 * img_area or a <= 0.001 * img_area:
                    continue

            r = cv2.boundingRect(c)
            is_inside = False
            for q in rectangles:
                if self.inside(r, q):
                    is_inside = True
                    break
            if not is_inside:
                rectangles.append(r)
        count = 0
        rectangles = self.sortBoundingBox(rectangles)    
        for r in rectangles:
            count += 1
            x, y, w, h = self.wrap_digit(r, img_w, img_h)
            roi = img[y : y + h, x : x + w].copy()
            
            #adapt
            roi=cv2.bitwise_not(roi)
            roi=cv2.resize(roi, (28,28), interpolation = cv2.INTER_AREA)
            #plt.imshow(roi, cmap='gray', vmin=0, vmax=255)
            #plt.show()
            cv2.imwrite("./test/%d.png" % count, roi)
            print(roi.shape)
            
            roi = cv2.imread("./test/%d.png" % count, 0)
            print(roi.shape)
            roi = cv2.bitwise_not(roi)
            roi = roi.astype("float32")/255

            roi = expand_dims(roi, axis=0)
            predictions = self.model.predict(roi)
            #########################################
            """Correct the prediction for the generated image
            due to a font inconsistency between "1" and "T" """
            if not digital and predictions[0][0] > 0.1:
                classIndex=[0]
            else:
                classIndex = np.argmax(predictions, axis=-1)
            ##################################
            print(chr(self.mapp[classIndex[0]]))
            outArr.append(chr(self.mapp[classIndex[0]]))
            if os.path.exists("./test/%d.png" % count):
                os.remove("./test/%d.png" % count)
            #predictions = model.predict(roi)
            #classIndex = np.argmax(predictions, axis=-1)
            
            cv2.rectangle(outImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                outImg, "%d:" % count + chr(self.mapp[classIndex[0]]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

        plt.imshow(outImg)
        plt.show()    
        return outArr
                
"""analyzer = ImageAnalysis(pathImage)

analyzer.digital_maze_detection()"""