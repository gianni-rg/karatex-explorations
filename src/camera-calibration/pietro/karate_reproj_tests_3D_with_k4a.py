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

from karate_calib_data_test_display import xyz_coords

def triangulate(proj_mat1, proj_mat2, point1, point2):

    # Convert points to homogeneous coordinates
    #point1_hom = np.array([point1[0], point1[1], 1])
    #point2_hom = np.array([point2[0], point2[1], 1])

    point1_hom = np.array([point1[0], point1[1]])
    point2_hom = np.array([point2[0], point2[1]])

    # Triangulate points
    point_4d_hom = cv2.triangulatePoints(proj_mat1, proj_mat2, point1_hom, point2_hom)

    # Convert to non-homogeneous coordinates
    point_3d = point_4d_hom[:3] / point_4d_hom[3]

    return point_3d


def draw_point_on_image(image,x,y,color=(0,255,0)):
    cv2.circle(image, (int(x), int(y)), 1, color, -1)


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

def calculate_3D_points_from_2D_points(cameras,matches,colors):
    pass


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
    

    camera_pickle_path = "C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate\\new_sync\\20230714_193412\\cameras_new.pkl"
    with open(camera_pickle_path,'rb') as f:
        cameras_old = pickle.load(f)
    
    cameras = {"1":cameras_old["calib_gianni_c1"], "2":cameras_old["calib_gianni_c2"]}
    cam_id = 1
    
    with open(pickles[1],'rb') as f:
        frames = pickle.load(f)
    
    del frames['c3']
    del frames['c4']


    xyz = np.array([[0,0,0],[.1,0,0],[-.1,0,0],
                    [0,0.1,0],[0,-0.1,0],
                    [0,0,0.1],[0,0,-0.1]
                   ],dtype=np.float32)
    

    xyz = np.array(xyz_coords,dtype=np.float32)
  
    camera_mapping = {"1": "K4A_Gianni.mp4","2": "K4A_Master.mp4"}


    
    #  cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix)

    
    

    # Display the image in a window
    colors=[(255,255,255),(0,127,0),(0,255,0),(127,0,0),(255,0,0),(0,0,127),(0,0,255)] #  BGR

    
    camera_id1 = "1"
    camera_id2 = "2"
    c1 = cameras[camera_id1]
    c2 = cameras[camera_id2]
    camera_name_1 = camera_mapping[camera_id1] 
    camera_name_2 = camera_mapping[camera_id2] 
    image_c1 = image_files[sub_folder][camera_name_1][0]
    image_c2 = image_files[sub_folder][camera_name_2][0]
    
    point_2d1,_ = cv2.projectPoints(xyz, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
    point_2d2,_ = cv2.projectPoints(xyz, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)
    #point_2d1 = cv2.undistortPoints(point_2d1, cameraMatrix=c1.K,distCoeffs=c1.dist_coeffs)
    #point_2d2 = cv2.undistortPoints(point_2d2, cameraMatrix=c2.K,distCoeffs=c2.dist_coeffs)
    point_2d1 = point_2d1.reshape((point_2d1.shape[0],2))
    point_2d2 = point_2d2.reshape((point_2d2.shape[0],2))
    
    point2d_out_c1 = np.zeros((point_2d1.shape[0],2),dtype=np.float32)
    point2d_out_c2 = np.zeros((point_2d2.shape[0],2),dtype=np.float32)
    
    img1 = cv2.imread(image_c1,cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image_c2,cv2.IMREAD_UNCHANGED)

    # calculate the 3D points from the 2D points
    for i in range(xyz.shape[0]):
        dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(point_2d1[i,:],c1.K_inv,c1.R_inv,c1.tvec,c1.dist_coeffs,c1)
        dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(point_2d2[i,:],c2.K_inv,c2.R_inv,c2.tvec,c2.dist_coeffs,c2)
        
        # find 3D point from 2D points
        a,b,_  =  GeometryUtilities.RayRayIntersectionExDualUpdated(c1.pos,dir_1,c2.pos,dir_2)

        # vieww coord system
        intersection = (a+b) * -0.5
        
        p1_c1_a = cv2.projectPoints(a, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
        p1_c2_a = cv2.projectPoints(b, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)

        p1_c1_b = cv2.projectPoints(a, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
        p1_c2_b = cv2.projectPoints(b, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)

        p1_c1,_ = cv2.projectPoints(intersection, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
        p1_c2,_ = cv2.projectPoints(intersection, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)
        point2d_out_c1[i,:] = p1_c1
        point2d_out_c2[i,:] = p1_c2
        
        
       

    
        print(f"3D point {xyz[i,:]} = {intersection} error = {xyz[i,:] - intersection}")

  
    #for i in range(0,len(point_2d2)):
    #    draw_point_on_image(img1,point_2d1[i,0,0],point2d_out_c1[i,0,1],color=colors[i])
    #    draw_point_on_image(img2,point2d_out_c2[i,0,0],point2d_out_c2[i,0,1],color=colors[i])
    
    cv2.imshow('image_c1', img1)
    cv2.waitKey(0)

    cv2.imshow('image_c2', img2)
    cv2.waitKey(0)


    filename = f"test_3D_{sub_folder}_c{camera_id1}_out.png"
    cv2.imwrite(filename,img1)
  
    filename = f"test_3D_{sub_folder}_c{camera_id2}_out.png"
    cv2.imwrite(filename,img2)
# Destroy all windows   
    cv2.destroyAllWindows()    


    print("main")