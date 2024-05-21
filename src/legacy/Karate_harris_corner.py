import cv2
import numpy as np
import matplotlib.pyplot as plt
from CameraParamParser import * 
from GeometryUtility import *
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import json
import time
from HelperFunctions import *
import glob
import re
import os

import pickle

annotations = {}
imgs = {}
cameras = {}

seed = 1234
random.seed(seed)

folder = "./karate"
folder_anno = "pose"
folder_calib = "sync"
folder_calib_new = "calib"
imgs_clean = {}
keypoints = {}
bboxes = {}
bad_keypoints = {}
matches = {}
colors = {}
valid_keys = cameras.keys() 
 
filename = 'K4A_Gianni.mp4_000000.png'
filename_out = "Harris_"+filename
filename_out_sub = "Sub_Harris_"+filename
filename_out_center = "Centroid_Harris_"+filename
filename = os.path.join(folder,folder_calib_new,filename)
filename_out = os.path.join(folder,folder_calib_new,filename_out)
filename_out_sub = os.path.join(folder,folder_calib_new,filename_out_sub)
filename_out_center = os.path.join(folder,folder_calib_new,filename_out_center)

img = cv2.imread(filename)
img1 = cv2.imread(filename)
img2 = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
 
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
 
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
 
cv2.imwrite(filename_out,img)
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
 
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
 
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
 
# filter corners in a specific rect 
filtered  = []
x_min = 790
x_max = 915
y_min = 848 
y_max = 950    


for i in range(len(corners)):
    if corners[i][0] >= x_min and corners[i][0] <= x_max and corners[i][1] >= y_min and corners[i][1] <= y_max:
        filtered.append(corners[i])

print(filtered)
corners_filtered = np.array(filtered)
print(corners_filtered)
print(corners.shape)
print(corners_filtered.shape)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)


img2[res[:,1],res[:,0]]=[0,0,255]
img1[res[:,3],res[:,2]] = [0,255,0]
 
cv2.imwrite(filename_out_sub,img1)
cv2.imwrite(filename_out_center,img2)