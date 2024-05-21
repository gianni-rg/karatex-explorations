#!/usr/bin/env python

# pip install opencv-python

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

def process_frame(frameIdx, img):
    #img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display
    them on the images of checker board
    """
    if ret == True:
        print(f'{frameIdx} --> OK')
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    else:
        print(f'{frameIdx} --> KO')
    return img, ret
    #cv2.imshow('img',cv2.resize(img,(1280,800)))
    #cv2.waitKey(0)

# Defining the dimensions of checkerboard
# CHECKERBOARD = (4,4)
# CHECKERBOARD_SQUARE_SIZE_IN_M = 0.04  # 40mm size of square

# Defining the dimensions of checkerboard (new tests, with bigger checkerboard)
# CHECKERBOARD = (3,3)
# CHECKERBOARD_SQUARE_SIZE_IN_M = 0.074  # 80mm size of square

# 2023-07-14 Checkerboard 1, 2: A4, 6x4, 38mm
# CHECKERBOARD = (6,4)
# CHECKERBOARD_SQUARE_SIZE_IN_M = 0.038  # 38mm size of square

# 2023-07-14    Checkerboard 3: A3, 7x4, 50mm
CHECKERBOARD = (7,4)
CHECKERBOARD_SQUARE_SIZE_IN_M = 0.050  # 50mm size of square

# See: https://stackoverflow.com/questions/37310210/camera-calibration-with-opencv-how-to-adjust-chessboard-square-size
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Extracting path of individual image stored in a given directory
#imagePath = './images/*.jpg'
#imagePath = 'E:\\Users\\Gianni\\Videos\\Karate\\Frames_DJI_0259\\*.png'
#imagePath = 'E:\\Users\\Gianni\\Videos\\Karate\\Frames_S20\\*.png'
#imagePath = 'E:\\Users\\Gianni\\Videos\\Karate\\DJI_0266\\*.jpg'
#imagePath = 'E:\\Users\\Gianni\\Videos\\Karate\\20230202_204617\\*.jpg'

#images = glob.glob(imagePath)[0:12]

# define a video capture object
video_files = [Path(p) for p in glob.glob("D:\\Datasets\\karate\\Synchronized\\**\\*.mp4", recursive=True)]
#video_files = [Path("D:\\Datasets\\karate\\Synchronized\\20230714_193412\\S20.mp4")]

#video_file_path = Path("E:\\Users\\Gianni\\Videos\\Karate\\20230202_204308.mp4")
#video_file_path = Path("\\\\diskstation02\\photo\\2023\\Karate (esperimenti tracking)\\Karate_casa\\DJI_0267.MP4")
#video_file_path = Path("D:\\Personal\\BodyTracking-Karate\\2023-07-14\\KinectG_Checker1_A4_6x4_38mm.mp4")
#video_file_path = Path("D:\\Personal\\BodyTracking-Karate\\2023-07-14\\KinectG_Checker2_A4_6x4_38mm.mp4")
#video_file_path = Path("D:\\Personal\\BodyTracking-Karate\\2023-07-14\\KinectG_Checker3_A3_7x4_49mm.mp4")

#video_file_path = Path("D:\\Datasets\\karate\\Synchronized\\20230714_193412\\K4A_Master.mp4")
#video_file_path = Path("D:\\Datasets\\karate\\Synchronized\\20230714_193412\\K4A_Gianni.mp4")
#video_file_path = Path("D:\\Datasets\\karate\\Synchronized\\20230714_193412\\S20.mp4")

for video_file_path in video_files:

    print(f"Processing {video_file_path}")

    output_file = Path.joinpath(video_file_path.parent, f"calibration_{video_file_path.stem}.json")
    if output_file.exists():
        print(f"Calibration already exists, skipping...")
        continue

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    # https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_SIZE_IN_M
    prev_img_shape = None

    vid = cv2.VideoCapture(str(video_file_path))

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    FRAME_STEP = 15 #(for Kinect)
    #FRAME_STEP = 60

    frame_start = 0
    frame_end = 1000 #total_frames #4499 #(for DJI)
    frame_step = FRAME_STEP
    frame_count = 0

    valid_frames = 0
    # https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
    while(vid.isOpened()):

        # Capture the video frame by frame
        ret, frame = vid.read()

        if not ret or frame_count >= frame_end:
            break

        frame_count += 1
        if frame_count < frame_start:
            continue

        frame_step -= 1
        if frame_step > 0:
            continue

        frame_step = FRAME_STEP
        frame, is_valid = process_frame(frame_count, frame)
        frame_size = frame.shape[:2]
        if is_valid:
            valid_frames += 1

        # Display the resulting frame
        cv2.imshow('frame', cv2.resize(frame, (1280, 720)))

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

    #h,w = img.shape[:2]

    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
    """
    if valid_frames == 0:
        print("No valid frames found. Skipping...")
        continue
        #exit()

    print(f"Calibrating camera (using {valid_frames} frames), please wait...")

    # To find the camera's internal parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    # Estimate Pose (camera's position and orientation in the world)
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(objpoints, imgpoints, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    # See: https://betterprogramming.pub/how-to-calibrate-a-camera-using-python-and-opencv-23bab86ca194
    # http://amroamroamro.github.io/mexopencv/matlab/cv.calibrateCamera.html
    camera = {}

    for variable in ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']:
        camera[variable] = eval(variable)

    is_dji = "\\DJI_" in str(video_file_path)
    cam_name = "_dji_" if is_dji else "_s20_"

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error/len(objpoints)))

    camera["reproj_error"] = mean_error/len(objpoints)

    #with open(f"camera{cam_name}{video_file_path.stem}.json", 'w') as f:
    with open(output_file, 'w') as f:
        json.dump(camera, f, indent=4, cls=NumpyEncoder)

    # See: https://ksimek.github.io/2012/08/22/extrinsic/

    # print("Camera matrix : \n")
    # print(mtx)
    # print("dist : \n")
    # print(dist)
    # print("rvecs : \n")
    # print(rvecs)
    # print("tvecs : \n")
    # print(tvecs)
