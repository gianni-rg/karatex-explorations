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
from PIL import Image
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

def draw_point_on_image(image,x,y,color=(0,255,0)):
    cv2.circle(image, (int(x), int(y)), 3, color, -1)


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


if __name__ == "__main__":
    pose_full = os.path.join(folder,folder_calib)
    
    sub_folders = [f.path for f in os.scandir(pose_full) if f.is_dir()]
    max_num_frames = 10
    nconverter = lambda x: f"{x:06d}"
    image_files = {}
    for subfolder in sub_folders:
        clip_name = os.path.basename(subfolder)
        image_files[clip_name] = {}
        png_files = glob.glob( subfolder+"/*.png" )
        for video_file in png_files:
            video_name = os.path.basename(video_file)
            idx = video_name.rfind("_")
            
            camera_name = video_name[0:idx]
            if(camera_name not in image_files[clip_name]):
                image_files[clip_name][camera_name] = []
            image_files[clip_name][camera_name].append(video_file)
          
    sub_folder= "20230714_200355"
    sub_folder= "20230714_193412"
    pickles = glob.glob(folder + f'/{sub_folder}_*.pickle')
    
    with open(pickles[0],'rb') as f:
        cameras_old = pickle.load(f)
    
    cameras ={}
    cam_id = 1
    for key in cameras_old:
        cameras[str(cam_id)] = cameras_old[key]
        cameras[str(cam_id)].pos = cameras[str(cam_id)].pos.reshape((3,))
        cameras[str(cam_id)].tvec = cameras[str(cam_id)].tvec.reshape((3,1))
        cameras[str(cam_id)].rvec = cameras[str(cam_id)].rvec.reshape((3,1))
        cam_id += 1
        
    with open(pickles[1],'rb') as f:
        frames = pickle.load(f)


    xyz = np.array([[0,0,0],[.1,0,0],[-.1,0,0],
                    [0,0.1,0],[0,-0.1,0],
                    [0,0,0.1],[0,0,-0.1]
                   ],dtype=np.float32)
    
    camera_id = "3"
    
    c1 = cameras[camera_id] 
    camera_mapping = {"1": "K4A_Gianni.mp4","2": "K4A_Master.mp4","3": "S20.mp4"}
    camera_name = camera_mapping[camera_id]

    xyz_np = np.array(xyz)
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    
    #  cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix)
    
    point_2d,_ = cv2.projectPoints(xyz, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
    
    print(point_2d)
    path_img = image_files[sub_folder]["S20.mp4"][0]
    path_img = image_files[sub_folder]["K4A_Master.mp4"][0]
    path_img = image_files[sub_folder]["K4A_Gianni.mp4"][0]
    path_img = image_files[sub_folder][camera_name][0]
    print(f"Loading img {path_img}")
    img = Image.open(path_img)

    # Display the image
    img.show()
    
        
    img = cv2.imread(path_img,cv2.IMREAD_UNCHANGED)

    # Display the image in a window
    cv2.imshow('image', img)
    cv2.imwrite("test.png",img)
    colors=[(255,255,255),(0,127,0),(0,255,0),(127,0,0),(255,0,0),(0,0,127),(0,0,255)] #  BGR
    xyz = np.array([[0,0,0],[.1,0,0],[-.1,0,0],
                    [0,0.1,0],[0,-0.1,0],
                    [0,0,0.1],[0,0,-0.1]
                   ],dtype=np.float32)
    
    
    for i in range(0,len(point_2d)):
        draw_point_on_image(img,point_2d[i,0,0],point_2d[i,0,1],color=colors[i])
    # Wait for any key to close the window
    cv2.waitKey(0)
    filename = f"test_pixel_{sub_folder}_c{camera_id}.png"
    print(f"Writing img to {filename}.")
    cv2.imwrite(filename,img)
     
# Destroy all windows   
    cv2.destroyAllWindows()    


    print("main")