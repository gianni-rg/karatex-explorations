from click import File
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
cameras_params = parse_ini_file(ini_file)
imgs_clean = {}
keypoints = {}
bboxes = {}
bad_keypoints = {}
matches = {}
colors = {}
valid_keys = cameras.keys() 

def get_all_files(src_folder,extention=".json",sub_folder= "20230714_193412") -> list:
    files = glob.glob(src_folder + f"/{sub_folder}/*{extention}")
    
    return files

def load_json(filepath):
    json_file = {}
    with open(filepath,'r') as f:
        json_file = json.load(f)
    return json_file

def camera_factory(calib_info: dict, do_average_results = False) -> Camera:
    
    
    camera_matrix  = np.array(calib_info['mtx'])    
    dist_coeffs = np.array(calib_info['dist'])
    
    if do_average_results == False:
        rvec = np.array(calib_info['rvecs'][0])
    
        tvec = np.array(calib_info['tvecs'][0])
    
    else:
        rvec = np.mean(np.array(calib_info['rvecs']),axis=0)
        tvec = np.mean(np.array(calib_info['tvecs']),axis=0)
   
    
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rvec_bak, _ = cv2.Rodrigues(rotation_matrix)
    
    pos = -np.dot(rotation_matrix.T,tvec)
    

    camera = Camera(camera_matrix,rotation_matrix,tvec,rvec,pos,dist_coeffs)
    
    
    return camera

def process_annotation(anno: dict) -> (dict,dict):
    keypoints = {}
    bboxes = {}
    keypoints = anno['instance_info']
    num_frames  = len(anno['instance_info'])
    frames = {}
    for i in range(num_frames):
        frame_id = keypoints[i]['frame_id']
        frames[frame_id] = []
        num_persons_framse = len(keypoints[i]['instances'])
        frame_data = []
        for j in range(num_persons_framse):
            person = {"bbox":keypoints[i]['instances'][j]['bbox']}
            person['keypoints'] = keypoints[i]['instances'][j]['keypoints'][0:23]
            frame_data.append(person)    
        
        frames[frame_id].append(frame_data)
    return frames

if __name__ == "__main__":
    pose_full = os.path.join(folder,folder_calib)
    
    sub_folders = [f.path for f in os.scandir(pose_full) if f.is_dir()]
    max_num_frames = 10
    nconverter = lambda x: f"{x:06d}"
    for subfolder in sub_folders:
        video_files = glob.glob( subfolder+"/*.mp4" )
        for video_file in video_files:
            video_name = os.path.basename(video_file)
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened(): 
                print("Error opening video file")
            else:
                frame_count = 0
                while cap.isOpened() and frame_count < max_num_frames:
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret:
                        # Save each frame to a file
                        filename = os.path.join(subfolder, f"{video_name}_{nconverter(frame_count)}.png")    
                        cv2.imwrite( filename, frame)
                        frame_count += 1
                    else: 
                        break

            # When everything done, release the video capture object
            cap.release()

        # Close all the frames
        cv2.destroyAllWindows()
       
    print("main")