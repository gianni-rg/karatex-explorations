#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# 2023-07-14    Checkerboard 3: A3, 7x4, 50mm
CHECKERBOARD = (7,4)
CHECKERBOARD_SQUARE_SIZE_IN_M = 0.050  # 50mm size of square

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
# https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_SIZE_IN_M
prev_img_shape = None

with open('xyz_points.json', 'w') as f:
    json.dump(objp, f, indent=4, cls=NumpyEncoder)
