import matplotlib as plt
import numpy as np
import math
import cv2


def get_gauss_kernel(size, sigma, teta, gamma, lmbda):
    center = (int)(size/2)
    kernel = np.zeros((size, size))
    for i in range(-center, center + 1, 1):
        for j in range(-center, center + 1, 1):
            x1 = i * math.cos(teta) + j * math.sin(teta)
            y1 = -1 * i * math.sin(teta) + j * math.cos(teta)
            kernel[i+2,j+2] = np.exp(-(x1**2 + gamma**2 * y1**2) / (2 * sigma ** 2)) * math.cos(2 * math.pi * x1 / lmbda)
    return kernel
y1 = 266
video = cv2.VideoCapture("tesjalanfix.mp4")
kernel2 = np.ones((5,5), np.uint8)
kernel3 = np.ones((3,3), np.uint8)
jmlLan = 0
#akses video
while(True):
    #detecting horizontal line
    ret, orig_frame = video.read()   
    gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    canny1 = cv2.Canny(blur, threshold1=75, threshold2=100)
    sobely = cv2.Sobel(blur, ddepth=-1, dx=0, dy=1, ksize=1)
    
    kernelku = get_gauss_kernel(25, 5, 0, 1, 15)

    hasil = cv2.filter2D(sobely, ddepth=-1, kernel=kernelku)
    ret,thresh1=cv2.threshold(hasil, 220, 255,cv2.THRESH_TOZERO)
    erosion = cv2.erode(thresh1, kernel2, iterations=1)
    dilate = cv2.dilate(erosion, kernel3, iterations=1)
    ret,thresh2=cv2.threshold(dilate, 220, 255,cv2.THRESH_BINARY)
    con, hir = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for(i, c) in enumerate(con):
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(orig_frame, (x-10, y-10), (x+w+10, y+h+10), (0,0,255),2)
    lines = cv2.HoughLinesP(thresh2,1,np.pi/180,40,minLineLength=30,maxLineGap=30)
    
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(orig_frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            jmlLan += 1 
    #creating line
    cv2.line(orig_frame, (0,y1), (1200,y1), (0,255,0), 3)
    
    
    cv2.imshow('Original', orig_frame)
    cv2.imshow('Hasil', thresh2)

    if cv2.waitKey(1)>0:
        break
    
video.release()
cv2.destroyAllWindows()

