import os
import cv2
import numpy as np
import json
import click
from karate_utilities import Camera

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

points = []
floor = 0.223
for i in range(num_points):
    for j in range(num_points):
        points.append([i*step,floor,j*step])

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

    # serialize results
    output_file = os.path.join(input_path,clip_name,output_folder,output_file_name)

    with open(output_file, 'w') as f:
        json.dump(camera_dictionary, f, indent=4)

if __name__ == "__main__":
    main()
