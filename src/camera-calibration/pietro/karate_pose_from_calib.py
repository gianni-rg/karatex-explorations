from calendar import c
import os
import pickle
import cv2
import numpy as np
import json

from HelperFunctions import Camera
from karate_calib_data_test_display import xyz_coords,uvs_K4A_Gianni,uvs_K4A_Master 

# data taken from for old kinect https://github.com/knexfan0011/Kinect-V2-Camera-Calibration-Data
# https://littlewing.hatenablog.com/entry/2019/08/25/183629
Camera_matrix=[ 1071.062, 0, 973.732,
                0, 1074.326, 491.147,
                0, 0, 1]

Distortion = [0.068412, -0.057890, -0.010089, 0.005318] #+/- [0.004484, 0.012715, 0.000938, 0.001083]

#Additional Error ranges:
#f_x +/- 2.433
#f_y +/- 2.889
#c_x +/- 3.342
#c_y +/- 3.105

# ===== Device 0: {DeviceID:XXXXXX} =====
principal_point_c2 = (968.939209,558.608459) # (x,y)
focal_length_c2 = (899.693420,899.449646)
# radial_distortion_coefficients 
k1 = 0.678679
k2 = -2.779486
k3 = 1.569404
k4 = 0.554819
k5 = -2.610379
k6 = 1.500811
center_of_distortion = ( 0.000000,0.000000)
p1 = tangential_distortion_coefficient_x = 0.000476
p2 = tangential_distortion_coefficient_y = 0.000104
metric_radius = 0.000000

# https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/examples/opencv_compatibility/main.cpp
dist_coeff_c2 = [k1,k2,p1,p2,k3,k4,k5,k6]

# ===== Device 1: {DeviceID:XXXXXX} =====
resolution = (1920,1080)
principal_point = (961.302551,554.008667) #cx,cy
focal_length = (909.712341,909.587952) #f_x,f_y
#radial distortion coefficients:
k1 = 0.692706
k2 = -2.979956
k3 = 1.730745
k4 = 0.569762
k5 = -2.802186
k6 = 1.655197
p1 = 0.000765
p2 = 0.000095
center_of_distortion = (0.000000,0.000000)
metric_radius = 0.000000


dist_coeff = [k1,k2,p1,p2,k3,k4,k5,k6]
def find_pose_from_img_points(img_points):
    pass

def factory_camera_from_k4a(resolution,principal_point,focal_length,dist_params ) -> Camera :
    K = np.array([ [focal_length[0],0,principal_point[0]],
                   [0,focal_length[1],principal_point[1]],
                   [0,0,1]],dtype = np.float64)
    R = np.eye(3,dtype=np.float64)
    r_vec = np.array([0,0,0],dtype=np.float64) 
    t_vec = np.array([0,0,0],dtype=np.float64)
    pos = np.array([0,0,0],dtype=np.float64)
    dist_coeff = np.array(dist_params,dtype=np.float64)
    camera = Camera(K=K,Rot=R,tvec=t_vec,rvec=r_vec,pos=pos,dist_coeffs=dist_coeff)
    return camera


xyz_gen = []
num_points = 32
step = 0.049 
points = []
cameras_out = {}

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
        
        points.append([i*step,j*step,-2*step])
        points.append([i*step,-j*step,-2*step])
        points.append([-i*step,-j*step,-2*step])
        points.append([-i*step,j*step,-2*step])
        
points_1 = np.array(points,dtype=np.float64)


points  = []
for i in range(num_points):
    for j in range(num_points):
        points.append([i*step,0,j*step])
        points.append([i*step,0,-j*step])
        points.append([-i*step,0,-j*step])
        points.append([-i*step,0,j*step])
           
        for k in range(1,5):
            val = k*2
            points.append([i*step,val*step,j*step])
            points.append([i*step,val*step,-j*step])
            points.append([-i*step,val*step,-j*step])
            points.append([-i*step,val*step,j*step])
        
            points.append([i*step,-val*step,j*step])
            points.append([i*step,-val*step,-j*step])
            points.append([-i*step,-val*step,-j*step])
            points.append([-i*step,-val*step,j*step])
 
points = []
for i in range(num_points):
    for j in range(num_points):
        points.append([i*step,.2415,j*step])
        #points.append([i*step,0,-j*step])
        #points.append([-i*step,0,-j*step])
        #points.append([-i*step,0,j*step])
           
        for k in range(1,5):
            val = k*2
            #points.append([i*step,val*step,j*step])
            #points.append([i*step,val*step,-j*step])
            #points.append([-i*step,val*step,-j*step])
            #points.append([-i*step,val*step,j*step])
        
            #points.append([i*step,-val*step,j*step])
            #points.append([i*step,-val*step,-j*step])
            #points.append([-i*step,-val*step,-j*step])
            #points.append([-i*step,-val*step,j*step])
points = np.array(points,dtype=np.float64)

#points = points_1

cameras_out = {}

if __name__ == "__main__":
    c1 = factory_camera_from_k4a(resolution,principal_point,focal_length,dist_coeff)
    c2 = factory_camera_from_k4a(resolution,principal_point,focal_length_c2,dist_coeff_c2)
    
    folder= 'C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate'
    folder_calib = "new_sync"
    sub_folder = '20230714_193412'
    filenames = ["K4A_Gianni.mp4_000000.png","K4A_Master.mp4_000000.png"]
    # find pose from image points
    uvs = [np.array(uvs_K4A_Gianni,dtype=np.float64),np.array(uvs_K4A_Master,dtype=np.float64)]
    xyz = np.array(xyz_coords,dtype=np.float64)
    
    circle_size = 3
    cameras = [c1,c2]
    names  = ["calib_gianni","calib_internet"]
    for camera,name in zip (cameras,names) :
        ctr = 0
        for filename in filenames:
            
            img = cv2.imread(os.path.join(folder,folder_calib,sub_folder,filename),cv2.IMREAD_COLOR)
            img1 = cv2.imread(os.path.join(folder,folder_calib,sub_folder,filename),cv2.IMREAD_COLOR)
            img2 = cv2.imread(os.path.join(folder,folder_calib,sub_folder,filename),cv2.IMREAD_COLOR)
            
            key = f"{name}_c{ctr+1}"

            cv2.imshow('frame', cv2.resize(img, (1280, 720)))
            cv2.waitKey(0)
            filename_out = f"{name}_pose_" + filename
            filename_out_orig_uv = f"{name}_orig_" + filename
            uv = uvs[ctr]
           
            # display results
            for uvp in uv:
                x = int(uvp[0])
                y = int(uvp[1])
                cv2.circle(img1,(x,y),circle_size,(0,0,255),-1)
            cv2.imshow('frame', cv2.resize(img1, (1280, 720)))
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(folder,folder_calib,sub_folder,filename_out_orig_uv),img1)


            (success, rotation_vector, translation_vector) = cv2.solvePnP(xyz, uv, camera.K, camera.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            rotation_matrix,_ = cv2.Rodrigues(rotation_vector) #3x3 matrix
            camera_pos = np.linalg.inv( rotation_matrix) @ translation_vector

            
            # update camera values
            camera.Rot = rotation_matrix
            camera.R_inv = np.linalg.inv( rotation_matrix)
            camera.tvec = translation_vector.reshape(3,)
            camera.rvec = rotation_vector.reshape(3,)
            camera.pos = camera_pos.reshape(3,) 
            
            cam = Camera(camera.K,camera.Rot,camera.tvec,camera.rvec,camera.pos,camera.dist_coeffs)

            cameras_out[key] = cam
            
            # reproject points on img1
            uv_proj,_ = cv2.projectPoints(points,cam.rvec, cam.tvec, cam.K, cam.dist_coeffs);
            # display results
            for uvp in uv_proj:
                x = int( uvp[0,0])
                y = int(uvp[0,1])
                cv2.circle(img2,(x,y),circle_size,(0,0,255),-1)
            cv2.imshow('frame', cv2.resize(img2, (1280, 720)))
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(folder,folder_calib,sub_folder,filename_out),img2)
            ctr += 1 
    
    # save cameras
    camera_fileout = os.path.join(folder,folder_calib,sub_folder,"cameras_new.pkl")  
    with open(camera_fileout,"wb") as f:        
        pickle.dump(cameras_out,f)
    
        
    print("Done.")