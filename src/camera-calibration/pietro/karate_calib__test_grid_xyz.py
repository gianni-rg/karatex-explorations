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

    xyz_gen = []
    num_points = 8
    step = 0.049
    points = []
    
    for i in range(num_points):
        for j in range(num_points):
            points.append([i*step,j*step,0])
            points.append([i*step,-j*step,0])
            points.append([-i*step,-j*step,0])
            points.append([-i*step,j*step,0])
           
            points.append([i*step,j*step,2*step])
            points.append([i*step,-j*step,2*step])
            points.append([-i*step,-j*step,2*step])
            points.append([-i*step,j*step,2*step])
           
    points = np.array(points,dtype=np.float32)

    tot_point = len(points)

    xyz = np.array([[0,0,0],[.1,0,0],[-.1,0,0],
                    [0,0.1,0],[0,-0.1,0],
                    [0,0,0.1],[0,0,-0.1]
                   ],dtype=np.float32)
    
    camera_id = "3"
    
    c1 = cameras[camera_id] 
    camera_mapping = {"1": "K4A_Gianni.mp4","2": "K4A_Master.mp4","3": "S20.mp4","4":"K4A_Gianni_bis.mp4"}
    camera_name = camera_mapping[camera_id]

    xyz_np = np.array(xyz)
    
    #  cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix)
    
    point_2d,_ = cv2.projectPoints(xyz, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
    
    print(point_2d)
    
    path_img = image_files[sub_folder][camera_name][0]
    print(f"Loading img {path_img}")
    img = Image.open(path_img)

    # Display the image
    img.show()
    
        
    img = cv2.imread(path_img,cv2.IMREAD_UNCHANGED)

    # Display the image in a window
    colors=[(255,255,255),(0,127,0),(0,255,0),(127,0,0),(255,0,0),(0,0,127),(0,0,255)] #  BGR
    xyz = np.array([[0,0,0],[.1,0,0],[-.1,0,0],
                    [0,0.1,0],[0,-0.1,0],
                    [0,0,0.1],[0,0,-0.1]
                   ],dtype=np.float32)
    
    camera_id1 = "1"
    camera_id2 = "2"
    camera_id4 = "4"
    c1 = cameras[camera_id1]
    c2 = cameras[camera_id2]
    c4 = cameras[camera_id4]
    camera_name_1 = camera_mapping[camera_id1] 
    camera_name_2 = camera_mapping[camera_id2] 
    camera_name_4 = camera_mapping[camera_id4] 
    image_c1 = image_files[sub_folder][camera_name_1][0]
    image_c2 = image_files[sub_folder][camera_name_2][0]
    
    point_2d1,_ = cv2.projectPoints(xyz, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
    point_2d2,_ = cv2.projectPoints(xyz, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)
    #point_2d1 = cv2.undistortPoints(point_2d1, cameraMatrix=c1.K,distCoeffs=c1.dist_coeffs)
    #point_2d2 = cv2.undistortPoints(point_2d2, cameraMatrix=c2.K,distCoeffs=c2.dist_coeffs)

    point2_gen_c1,_ = cv2.projectPoints(points, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
    point2_gen_c2,_ = cv2.projectPoints(points, c2.rvec, c2.tvec, c4.K,c4.dist_coeffs)
   

    point_2d2 = point_2d2.reshape((7,2))
    point_2d1 = point_2d1.reshape((7,2))
    
    point2_gen_c1 = point2_gen_c1.reshape((tot_point,1,2))
    point2_gen_c2 = point2_gen_c2.reshape((tot_point,1,2))
    
    point2d_out_c1 = np.zeros((7,2),dtype=np.float32)
    point2d_out_c2 = np.zeros((7,2),dtype=np.float32)
    
    img1 = cv2.imread(image_c1,cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image_c2,cv2.IMREAD_UNCHANGED)

    for i in range(0,len(points)):
        draw_point_on_image(img1,point2_gen_c1[i,0,0],point2_gen_c1[i,0,1],color=colors[0])
        draw_point_on_image(img2,point2_gen_c2[i,0,0],point2_gen_c2[i,0,1],color=colors[0])
    
    cv2.imshow('image_c1', img1)
    cv2.waitKey(0)

    cv2.imshow('image_c2', img2)
    cv2.waitKey(0)

    filename = f"calib_3D_{sub_folder}_c{camera_id1}_out.png"
    cv2.imwrite(filename,img1)
  
    filename = f"calib_3D_{sub_folder}_c{camera_id2}_out.png"
    cv2.imwrite(filename,img2)



    p1 = c1.pos.reshape(3,1)
    p2 = c2.pos.reshape(3,1)
    proj_mat1 = np.dot(c1.K, np.hstack((np.eye(3), p1)))
    proj_mat2 = np.dot(c2.K, np.hstack((np.eye(3), p2)))
                       
    # using cv2.triangulatePoints to find the 3D points
    for i in range(xyz.shape[0]):
        point1 = point_2d1[i]
        point2 = point_2d2[i]
        point_3d = triangulate(proj_mat1, proj_mat2, point1, point2)
        print(f"3D point {xyz[i,:]} = {point_3d} error = {xyz[i,:] - point_3d}")

    # calculate the 3D points from the 2D points
    for i in range(xyz.shape[0]):
        dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(point_2d1[i,:],c1.K_inv,c1.R_inv,c1.tvec,c1.dist_coeffs,c1)
        dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(point_2d2[i,:],c2.K_inv,c2.R_inv,c2.tvec,c2.dist_coeffs,c2)
        
        # find 3D point from 2D points
        a,b,_  =  GeometryUtilities.RayRayIntersectionExDualUpdated(c1.pos,dir_1,c2.pos,dir_2)

        intersection = (a+b) * 0.5
        p1_c1_a = cv2.projectPoints(a, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
        p1_c2_a = cv2.projectPoints(b, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)

        p1_c1_b = cv2.projectPoints(a, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
        p1_c2_b = cv2.projectPoints(b, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)

        p1_c1,_ = cv2.projectPoints(intersection, c1.rvec, c1.tvec, c1.K,c1.dist_coeffs)
        p1_c2,_ = cv2.projectPoints(intersection, c2.rvec, c2.tvec, c2.K,c2.dist_coeffs)
        point2d_out_c1[i,:] = p1_c1
        point2d_out_c2[i,:] = p1_c2
        
        
       

    
        print(f"3D point {xyz[i,:]} = {intersection} error = {xyz[i,:] - intersection}")

    point2d_out_c1 = point2d_out_c1.reshape((7,1,2))
    point2d_out_c2= point2d_out_c2.reshape((7,1,2))
    
    err_1 = np.sum(np.abs(point_2d1 - point2d_out_c1))
    err_2 = np.sum(np.abs(point_2d2 - point2d_out_c2))
    print(f"Error 1 = {err_1} Error 2 = {err_2}")
    
    for i in range(0,len(point_2d)):
        draw_point_on_image(img1,point2d_out_c1[i,0,0],point2d_out_c1[i,0,1],color=colors[i])
        draw_point_on_image(img2,point2d_out_c2[i,0,0],point2d_out_c2[i,0,1],color=colors[i])
    
    cv2.imshow('image_c1', img1)
    cv2.waitKey(0)

    cv2.imshow('image_c2', img2)
    cv2.waitKey(0)


    filename = f"test_3D_{sub_folder}_{camera_name_1}_out.png"
    cv2.imwrite(filename,img1)
  
    filename = f"test_3D_{sub_folder}_{camera_name_2}_out.png"
    cv2.imwrite(filename,img2)
# Destroy all windows   
    cv2.destroyAllWindows()    


    print("main")