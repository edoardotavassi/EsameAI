from numpy.matrixlib.defmatrix import matrix
from utils import *
import imutils
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import queue
from tensorflow import expand_dims
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("TkAgg")
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
    
    def flatten(self,l):
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
        return flatten(outArr)
    
    def customFloodFill(self, image,seed,color,areaType):
# Fills connected components of white cells starting at the given point (seed) and returns the area
# of the filled portion
        q = queue.Queue()
        q.put(seed)
        rows,cols = np.shape(image)
        minX,minY,maxX,maxY = 20000,20000,0,0
        
        count = 0
        l = 1000
        while not q.empty():
            point = q.get()
            if image[point[0],point[1]] == 255:
                if point[0] < (rows/5):
                    l = min(l,point[1])
                count = count + 1
                image[point[0],point[1]] = color
                for i in range(3):
                    for j in range(3):
                        x = point[0] + i - 1
                        y = point[1] + j - 1
                        if x >= 0 and y >= 0 and x < rows and y < cols:
                            if image[x][y] == 255:
                                q.put([x,y])
                                minX = min(minX,x)
                                minY = min(minY,y)
                                maxX = max(maxX,x)
                                maxY = max(maxY,y)
                                
        boundingArea = (maxX-minX) * (maxY-minY)
        if areaType == 0:
            return image,boundingArea,l
        else: return image,count,l
    
    
    def extractOuterGrid(self, img):
        rows,cols = np.shape(img)
        maxArea = 0
        point = [0,0]

        imgOriginal = img.copy()
        for i in range(rows):
            for j in range(cols):
                if img[i][j] == 255:
                    img,area,_ = self.customFloodFill(img,[i,j],100,0)
                    if area > maxArea:
                        maxArea = area
                        point = [i,j]
                        
        img = imgOriginal
        img,area,_ = self.customFloodFill(img,[point[0],point[1]],100,0)	
        for i in range(rows):
            for j in range(cols):
                if img[i][j] == 100:
                    img[i][j] = 255
                else: img[i][j] = 0
        return img,point
    
    def getDistance(self, p1,p2):
        return pow(pow((p1[0]-p2[0]),2) + pow((p1[1]-p2[1]),2),0.5)
    
    #####################################################
    def digital_maze_detection(self):
        img = cv2.imread(self.path,0)
        img = imutils.resize(img, height=500)
        self.imgOriginal = img.copy()
        plt.imshow(self.imgOriginal,cmap='gray', vmin=0, vmax=255)
        plt.title("Original Image")
        plt.show()
        plt.imshow(img,cmap='gray', vmin=0, vmax=255)
        plt.title("Resized Image")
        plt.show()
        
        _,threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        plt.imshow(threshold,cmap='gray', vmin=0, vmax=255)
        plt.title("Threshold Image")
        plt.show()
        
        erode_kernel = np.ones((2, 2), np.uint8)
        threshold = cv2.erode(threshold, erode_kernel, threshold, iterations=2)
        plt.imshow(threshold,cmap='gray', vmin=0, vmax=255)
        plt.title("Eroded Image")
        plt.show()
        return self.detection(threshold, True)
    
    def hand_maze_detection(self):
        img = cv2.imread(self.path,0)
        img = imutils.resize(img, height=500)
        imgX, imgY = np.shape(img)
        self.imgOriginal = img.copy()
        img = cv2.GaussianBlur(img,(11,11),0)
        
        plt.imshow(self.imgOriginal,cmap='gray', vmin=0, vmax=255)
        plt.title('Original Image')
        plt.show()
        plt.imshow(img,cmap='gray', vmin=0, vmax=255)
        plt.title('Blurred and Gray Image')
        plt.show()
        
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)
        erode_kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, erode_kernel, img, iterations=3)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title('Adaptive Thresholding')
        plt.show()
        
        # Dilate using cross-type kernel
        kernel = np.ones((3,3),np.uint8)
        kernel[0,0] = 0
        kernel[0,2] = 0
        kernel[2,0] = 0
        kernel[2,2] = 0
        img = cv2.dilate(img, kernel, iterations=3)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("Dilated Image")
        plt.show()
        
        self.imgOriginal = img.copy()
        img,point = self.extractOuterGrid(img)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("Extracted Outer Grid")
        plt.show()
        
        # Get corner points by basic algo
        l,r,t,b = 1000,0,10000,0
        for i in range(imgX):
            for j in range(imgY):
                if(img[i][j] == 255):
                    l = min(l,j)
                    r = max(r,j)
                    t = min(t,i)
                    b = max(t,i)
        topLeft = (l,t)
        topRight = (r,t)
        bottomLeft = (l,b)
        bottomRight = (r,b)

                    
        # Draw corner points on image
        cv2.circle(img,topLeft,5,(255,255,255),-1)		
        cv2.circle(img,topRight,5,(255,255,255),-1)		
        cv2.circle(img,bottomLeft,5,(255,255,255),-1)		
        cv2.circle(img,bottomRight,5,(255,255,255),-1)		
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("Corner Points")
        plt.show()
        
        # Change perspective
        maxLength = int(max(self.getDistance(topLeft,topRight),self.getDistance(topLeft,bottomLeft),self.getDistance(topRight,bottomRight),self.getDistance(bottomLeft,bottomRight)))
        src = np.array([topLeft,topRight,bottomLeft,bottomRight])
        dst = np.array([[0,0],[maxLength-1,0],[0,maxLength-1],[maxLength-1,maxLength-1]])
        h,_ = cv2.findHomography(src,dst)

        img,_,_ = self.customFloodFill(self.imgOriginal,[point[0],point[1]],0,0)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("No Grid Image")
        plt.show()
        img = cv2.warpPerspective(img,h,(maxLength,maxLength))


        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("Warped Image")
        plt.show()
        
        return self.detection(img, False)
        
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

            cv2.imwrite("./test/%d.png" % count, roi)
            print(roi.shape)
            
            roi = cv2.imread("./test/%d.png" % count, 0)
            roi = cv2.bitwise_not(roi)
            roi = roi.astype("float32")/255

            roi = expand_dims(roi, axis=0)
            predictions = self.model.predict(roi)
            #########################################
            """Correct the prediction for the generated image
            due to a font inconsistency between "1" and "T" """
            if digital and predictions[0][0] > 0.1:
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
        plt.title("Detected (index: label)")
        plt.show()    
        return outArr
                