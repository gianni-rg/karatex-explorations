
import cv2
import numpy as np

import random
import json
from Karate_utilities import Camera,get_random_color
import glob
import os


from karate_export_frames import load_json


def display_2d_pose(img,poses,size=3,color=(0,0,255)):
    person_data_array = poses['person_data']
    for j in range(len(person_data_array)):
        keypoints = person_data_array[j]['keypoints']
        for uv in keypoints:
            p =np.array(uv,dtype=np.int32)
            cv2.circle(img,p,3,(0,0,255),-1)
    
def display_recon_pose(img,camera,poses,size=3,color=(255,0,0)):
    for person in poses['reconstructedObjects']:
        keypoints = person['points']
        keypoints = np.array(keypoints,dtype=np.float64).reshape(-1,3)
        uv_proj,_ = cv2.projectPoints(keypoints, camera.rvec, camera.tvec, camera.K, camera.dist_coeffs)
        uv_proj = uv_proj.reshape(-1,2)
        for uv in uv_proj:
            x = int(uv[0])
            y = int(uv[1])
            cv2.circle(img,(x,y),size,color,-1)
    
def display_debug_poses(img,camera,poses,threshold=150,size=3,colors=[(255,0,0),(255,255,0)]):
    i = 0
    num_colors = len(colors)
    for cameras_key in poses:
        for key_pair,person in poses[cameras_key].items():
            error = person[0]
            if(error > threshold):
                continue
            idx_color =  i % num_colors 
            col = colors[idx_color]
            i +=1
            keypoints = person[1 :]
            keypoints = np.array(keypoints,dtype=np.float64).reshape(-1,3)
            uv_proj,_ = cv2.projectPoints(keypoints, camera.rvec, camera.tvec, camera.K, camera.dist_coeffs)
            uv_proj = uv_proj.reshape(-1,2)
            for uv in uv_proj:
                x = int(uv[0])
                y = int(uv[1])
                cv2.circle(img,(x,y),size,col,-1)
    
if __name__ == "__main__":
    input_path = "C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate"
    clip_name = "20230714_193412"
    calibration_file = "camera.json"
    camera_data_path = "camera_data"
    output_path = "my_output"
    output_path = "output_3d_500_3cams"
    
    camera_calib = os.path.join(input_path,clip_name,camera_data_path,calibration_file)
    
    # create camera objects
    with open(camera_calib,'r') as f:
        cameras_json = json.load(f)
    
    cameras = {}
    i = 1
    cameras_xi_names = {}
    cameras_names_xi = {}
    # this is used to force a certain order in camera comparison 
    #cameras_names_xi = {"K4A_Gianni":"1","K4A_Master":"2","K4A_Tino":"3"}
    for key,camera_json in cameras_json.items():
       
       camera = Camera()
       camera.from_json(camera_json)
       camera_new_id = str(i)
       camera.id = camera_new_id
       cameras_xi_names[camera_new_id] = key
       cameras_names_xi[key] = camera_new_id
       cameras[camera_new_id] = camera
       i+=1
       
    multiview_frames = {}
    for camera_id,_ in cameras.items():
        camera_frames = glob.glob(input_path + f'/{clip_name}/{cameras_xi_names[camera_id]}/*.json')
        if camera_id not in multiview_frames:
            multiview_frames[camera_id] = []
        multiview_frames[camera_id] += camera_frames
    
    
    debug_poses_frames = glob.glob(input_path + f'/{clip_name}/{output_path}/debug/poses*.json')
    poses_frames = glob.glob(input_path + f'/{clip_name}/{output_path}/*.json')
    colors = [get_random_color() for i in range(0,10)]
    num_frames = len(multiview_frames['1'])
    for key,camera in cameras.items():
        for i in range(0,num_frames,5):
            video_file = os.path.join(input_path,clip_name,f"{cameras_xi_names[key]}.mp4")
            cap = cv2.VideoCapture(video_file) 
            if not cap.isOpened(): 
                print("Error opening video file")
            else:
                # Capture frame-by-frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, img = cap.read()
                frame_num = f"{i:06d}"
                if ret:
                    poses_2d  = load_json(multiview_frames[key][i])
                    # display 2d poses 
                    #display_2d_pose(img,poses_2d,size=3,color=(0,0,255))
                    
                    # display 3d poses 
                    poses_3d = load_json(poses_frames[i])
                    #display_recon_pose(img,camera,poses_3d,size=3,color=(255,0,0))
                    
                    debug_poses = load_json(debug_poses_frames[i])
                    display_debug_poses(img,camera,debug_poses,threshold=350,size=3,colors=colors)
                    cv2.imshow(f'Preview',  cv2.resize(img, (1280, 720)))
                    cv2.waitKey(1)
                    cap.release()  # Close the video capture object
                else: 
                    break

    



    print("main")