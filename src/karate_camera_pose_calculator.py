import os
import cv2
import numpy as np
import json
import click
from karate_utilities import Camera

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

# ===== Device 1: {DeviceID:XXXXXX} ===== Gianni
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
step = 0.05
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
        points.append([i*step,.223,j*step])
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


#points = points_1

cameras_out = {}

@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help='annotations root folder')
@click.option('--calibration_file', type=click.STRING, required=True, default="camera_data/camera_calib_xyz_uv.json", help='JSON calibration file')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_200355", help='name of the clip to export in 3D')
@click.option('--output_folder', type=click.STRING, required=True, default="camera_data", help='relative path folder for the output')
@click.option('--output_file_name', type=click.STRING, required=True, default="camera.json", help='numerical format of the annotation (i.e: 00001.json)')
@click.option('--display_results',type=click.BOOL,required=True,default=True,help='display the reprojection results')
@click.option('--extra_points',type=click.FLOAT,default=None)
def main(input_path,calibration_file,clip_name,output_folder,output_file_name,display_results,extra_points):
    camera_calib_xyz_uvs_file = os.path.join(input_path,clip_name,calibration_file)

   # points_new = np.array(points,dtype=np.float64)
   # if extra_points != None:
   #     points = np.array(extra_points)

    with open(camera_calib_xyz_uvs_file,"r") as f:
        camera_calib_xyz_uvs = json.load(f)

    camera_dictionary = {}
    for key,camera_params in camera_calib_xyz_uvs['cameras'].items():
        resolution = camera_params['resolution']
        principal_point = camera_params['principal_point']
        focal_length = camera_params['focal_length']
        dist_coeff = camera_params['distortion_coefficients']
        xyz = np.array(camera_params['xyz_coords'],dtype=np.float64)
        uvs = np.array(camera_params['uvs'],dtype=np.float64)
        camera = factory_camera_from_k4a(resolution,principal_point,focal_length,dist_coeff)

        # find the pose based on the xyz - uvs pair
        (success, rotation_vector, translation_vector) = cv2.solvePnP(xyz, uvs, camera.K,camera.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotation_matrix,_ = cv2.Rodrigues(rotation_vector) #3x3 matrix
        # negate the pos to get the camera position in world coordinates (in OpenCV format)
        # do not negate the pos to get the camera position in world coordinates (in Vieww format)
        camera_pos = -np.linalg.inv( rotation_matrix) @ translation_vector

        # update camera values
        camera.Rot = rotation_matrix
        camera.R_inv = np.linalg.inv( rotation_matrix)
        camera.tvec = translation_vector.reshape(3,)
        camera.rvec = rotation_vector.reshape(3,)
        camera.pos = camera_pos.reshape(3,)
        camera_dictionary[key] = camera.to_json_serializeable()

        circle_size = 2


        # reproject points on img1
        if display_results:
            filename = camera_params['reference_img']
            img = cv2.imread(os.path.join(input_path,clip_name,output_folder,filename),cv2.IMREAD_COLOR)
            cv2.imshow('frame', cv2.resize(img, (1280, 720)))
            cv2.waitKey(0)
            points_tot = np.vstack((points,xyz))
            uvs_proj,_ = cv2.projectPoints(points_tot,rotation_vector, translation_vector, camera.K, camera.dist_coeffs);
            uvs_proj = uvs_proj.reshape(-1,2)
            for uvp in uvs:
                x = int(uvp[0])
                y = int(uvp[1])
                cv2.circle(img,(x,y),circle_size,(255,0,0),-1)

            for uvp in uvs_proj:
                x = int(uvp[0])
                y = int(uvp[1])
                cv2.circle(img,(x,y),circle_size,(0,0,255),-1)
            #cv2.imshow('frame', img)
            cv2.imshow('frame', cv2.resize(img, (1280, 720)))
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(input_path,clip_name,output_folder,"reproj_"+filename),img)
            
    # seroialize results
    output_file = os.path.join(input_path,clip_name,output_folder,output_file_name) 
    
    with open(output_file, 'w') as f:
        json.dump(camera_dictionary, f, indent=4)        
    

if __name__ == "__main__":
    main()
