from re import I
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
import timeit

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def gaussian_blur(img,ksize):
    return cv2.GaussianBlur(img,(ksize,ksize),0)
def conv2canny(img,low_thresh, high_thresh):
    return cv2.Canny(img,low_thresh, high_thresh)


#transfer learning
img = cv2.imread('test_images\\straight_lines1.jpg')
grayimg = grayscale(img)
blurimg = gaussian_blur(grayimg,5)
#kernel size = 5

#https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
#Canny edge does basically what sobelx and sobely were doing in ImageCleaning2

low_thresh = 100
high_thresh = 200
edgeimg = conv2canny(blurimg, low_thresh, high_thresh)
start = timeit.default_timer()
#plt.plot(edgeimg.sum(axis = 1)[0:650])
#plt.figure()

#plt.show()
#print(max(edgeimg.sum(axis = 1))) shows in the lower line 
#print(np.where(arr == max(edgeimg.sum(axis = 1))))

#print(max(edgeimg.sum(axis = 1)[0:650]))

k = np.where(edgeimg.sum(axis = 1) == max(edgeimg.sum(axis = 1)[0:650]))

plt.imshow(edgeimg, cmap = 'gray')
plt.figure()
#plt.show()

lowerlimitY = k[0][0]
upperlimitY = 650        #cropping out the lower line
#print(type(k))
#print(max(edgeimg.sum(axis = 1)))
#stop = timeit.default_timer()
#print('Time: ', stop - start)
'''
def roi(img, poi):
    mask = np.zeros_like(img)
    ignore_mask_colour = 255
    cv2.fillPoly(mask,poi,ignore_mask_colour)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

lowerLeftPoint = [130, 540]
upperLeftPoint = [410, 350]
upperRightPoint = [570, 350]
lowerRightPoint = [915, 540]
poi = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, lowerRightPoint]], dtype=np.int32)
maskimg = roi(edgeimg,poi)
'''
'''
cv2.imshow('grayimg',grayimg)
cv2.imshow('blurimg',blurimg)
cv2.imshow('edgeimg',edgeimg)
#cv2.imshow('maskimg',maskimg)
cv2.waitKey(0)
'''
'''
print(edgeimg)

l = list()
edgeimglist = list(edgeimg)
k = 0
print(edgeimg.shape)
print(type(edgeimg))
a = edgeimg.shape
for i in range(0,a[0]):
    for j in range(0,a[1]):
        l[k] = l[k] + i[j]
    k = k+1

'''

#CALCULATING IMAGE LIMITS

#print(edgeimg.sum(axis = 0))
#print(edgeimg.sum(axis = 1))
#print(len(edgeimg.sum(axis = 0)))
#print(len(edgeimg.sum(axis = 1)))

poi = np.array([[0,lowerlimitY], [0,upperlimitY], [1200,upperlimitY], [1200,upperlimitY]])

def ymask(img, poi):
    mask = np.zeros_like(img)
    ignore_mask_colour = 255
    cv2.fillPoly(mask,poi,ignore_mask_colour)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

lowerLeftPoint = [300, 650]
upperLeftPoint = [300, 475]
upperRightPoint = [900, 475]
lowerRightPoint = [900, 650]
poi = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, lowerRightPoint]], dtype=np.int32)
ymaskimg = ymask(edgeimg,poi)

plt.imshow(ymaskimg)
plt.figure()

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    
#hough lines
rho = 1
theta = np.pi/180
threshold = 30
min_line_len = 20 
max_line_gap = 300

houged = hough_lines(ymaskimg, rho, theta, threshold, min_line_len, max_line_gap)
plt.imshow(houged)
plt.figure()


houged2 = cv2.cvtColor(houged,cv2.COLOR_BGR2GRAY)   #Converted to grey to reduce dimensions



##   EXTRACTION OF POINTS FROM THE GRAPH   ##

#print(houged)
#print(houged.shape)
plt.imshow(houged2)
plt.show()
#print(houged2.shape)
#print(houged2.sum(axis = 1))
plt.plot(houged2.sum(axis = 1))
plt.figure()
#print(len(houged2.sum(axis = 1)))
arr = houged2.sum(axis = 1)
#print(type(arr))
'''k0 = 0
for i in range(len(arr)-1):
    if(arr[i+1]>arr[i]):
        k0 = i 
        break
print(k0)
k1 = 720
for i in range(k0,len(arr)-1,-1):
    if(arr[i-1]>arr[i]):
        k1 = i
        break
print(k1)
'''
plt.plot(np.diff(arr))      #calculate a[i+1] - a[i] to see where the y-coordinates lie
plt.show()
#print(np.diff(arr))
k0 = 0
k1 = 0
for i in range(len(np.diff(arr))):
    if(np.diff(arr)[i]>0):
        k0 = i
        break
print(k0)
rev_arr = arr[::-1]
#print(rev_arr)
#print(np.diff(rev_arr))
for i in range(len(np.diff(rev_arr))):
    if(np.diff(rev_arr)[i]>0):
        k1 = i
        break
print(720-k1)

arr1 = houged2.sum(axis = 0)
#plt.plot(arr1)
#plt.show()
k2 = 0
k3 = 0
#plt.plot(np.diff(arr1))
#plt.show()

for i in range(len(np.diff(arr1))):
    if(np.diff(arr1)[i]>0):
        k2 = i
        break
print(k2)
rev_arr1 = arr1[::-1]
for i in range(len(np.diff(rev_arr1))):
    if(np.diff(rev_arr1)[i]>0):
        k3 = i
        break
print(1280-k3)

#stop = timeit.default_timer()
#print('Time: ', stop - start)