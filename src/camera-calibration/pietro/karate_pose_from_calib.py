from calendar import c
import os
import pickle
import cv2
import numpy as np
import json

from HelperFunctions import Camera
from karate_calib_data_test_display_new import xyz_coords, uvs

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

# ===== Device 0: {DeviceID:K4A_Master} =====
principal_point_c2 = (958.849487,552.383789) # (x,y)
focal_length_c2 = (907.834412, 907.864136)
# radial_distortion_coefficients
k1 = 0.737361
k2 = -3.039608
k3 = 1.771284
k4 = 0.613389
k5 = -2.861687
k6 = 1.695149
center_of_distortion = (0.000000,0.000000)
p1 = 0.000604 #tangential_distortion_coefficient_x
p2 = -0.000080 #tangential_distortion_coefficient_y
metric_radius = 0.000000

# https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/examples/opencv_compatibility/main.cpp
dist_coeff_c2 = [k1,k2,p1,p2,k3,k4,k5,k6]

# ===== Device 1: {DeviceID:K4A_Gianni} =====
resolution = (1920,1080)
principal_point_c1 = (957.322998,548.698486) #cx,cy
focal_length_c1 = (903.628906, 903.349182) #f_x,f_y
#radial distortion coefficients:
k1 = 0.514305
k2 = -2.843920
k3 = 1.724399
k4 = 0.391681
k5 = -2.657261
k6 = 1.641451
p1 = 0.000335 #tangential_distortion_coefficient_x
p2 = -0.000006 #tangential_distortion_coefficient_y
center_of_distortion = (0.000000,0.000000)
metric_radius = 0.000000

dist_coeff_c1 = [k1,k2,p1,p2,k3,k4,k5,k6]

# ===== Device 3: {DeviceID:K4A_Tino} =====
principal_point_c3 = (961.302551,554.008667) # (x,y)
focal_length_c3 = (909.712341, 909.587952)
# radial_distortion_coefficients
k1 = 0.692706
k2 = -2.979956
k3 = 1.730745
k4 = 0.569762
k5 = -2.802186
k6 = 1.655197
center_of_distortion = (0.000000,0.000000)
p1 = 0.000765 #tangential_distortion_coefficient_x
p2 = 0.000095 #tangential_distortion_coefficient_y
metric_radius = 0.000000

dist_coeff_c3 = [k1,k2,p1,p2,k3,k4,k5,k6]

def find_pose_from_img_points(img_points):
    pass

def factory_camera_from_k4a(resolution,principal_point,focal_length,dist_params) -> Camera :
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

num_points = 64
step = 0.050

points = []
for i in range(num_points):
    for j in range(num_points):
        points.append([ i*step, 0.223,  j*step])
        points.append([ i*step, 0.223, -j*step])
        points.append([-i*step, 0.223, -j*step])
        points.append([-i*step, 0.223,  j*step])

points = np.array(points,dtype=np.float64)

cameras_out = {}

if __name__ == "__main__":

    c_K4A_Gianni = factory_camera_from_k4a(resolution,principal_point_c1,focal_length_c1,dist_coeff_c1)
    c_K4A_Master = factory_camera_from_k4a(resolution,principal_point_c2,focal_length_c2,dist_coeff_c2)
    c_K4A_Tino = factory_camera_from_k4a(resolution,principal_point_c3,focal_length_c3,dist_coeff_c3)

    folder= 'D:\\Datasets\\karate\\Synchronized\\Calibration'

    circle_size = 1

    cameras = [c_K4A_Gianni, c_K4A_Master,c_K4A_Tino]
    camera_ids  = ["K4A_Gianni", "K4A_Master", "K4A_Tino"]
    refImgs  = ["1", "2", "3"]
    for refImg in refImgs:
        for camera, cameraId in zip(cameras, camera_ids):

            # Find pose from image points
            uv = np.array(uvs[cameraId][f"ref{refImg}"], dtype=np.float64)
            xyz = np.array(xyz_coords[cameraId],dtype=np.float64)

            filename = f"{cameraId}-ReferenceFrame{refImg}.png"
            img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_COLOR)
            img1 = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_COLOR)
            img2 = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_COLOR)

            # cv2.imshow(f"{filename} ORIG", cv2.resize(img, (1280, 720)))
            # cv2.waitKey(0)
            filename_out = f"{cameraId}_pose.png"
            filename_out_orig_uv = f"{cameraId}_orig.png"

            # display results
            for uvp in uv:
                x = int(uvp[0])
                y = int(uvp[1])
                cv2.circle(img1,(x,y),circle_size,(0,0,255),-1)
            # cv2.imshow(f"{filename} REF", cv2.resize(img1, (1280, 720)))
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(folder,filename_out_orig_uv),img1)

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

            #cameras_out[key] = cam

            # reproject points on img1
            uv_proj,_ = cv2.projectPoints(points, cam.rvec, cam.tvec, cam.K, cam.dist_coeffs);
            xyz_proj,_ = cv2.projectPoints(xyz, cam.rvec, cam.tvec, cam.K, cam.dist_coeffs);
            # display results
            for idxx, uvp in enumerate(uv_proj):
                x = int(uvp[0,0])
                y = int(uvp[0,1])
                cv2.circle(img2,(x,y),circle_size,(0,0,255),-1)
            for idxx, uvp in enumerate(xyz_proj):
                x = int(uvp[0,0])
                y = int(uvp[0,1])
                cv2.circle(img2,(x,y),circle_size,(0,255,0),-1)

            cv2.imshow(f"{filename} PROJ", cv2.resize(img2, (1280, 720)))
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(folder,filename_out),img2)

    # save cameras
    # camera_fileout = os.path.join(folder,"cameras_new.pkl")
    # with open(camera_fileout,"wb") as f:
    #     pickle.dump(cameras_out,f)

    print("Done.")