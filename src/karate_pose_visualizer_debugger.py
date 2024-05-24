
import click
import cv2
import numpy as np

import random
import json
from karate_utilities import Camera,get_random_color,load_json
import glob
import os
import copy
from pathlib import Path


def display_2d_pose(img,poses,size=3,colors=[(0,0,255),(255,0,0),(255,0,0)]):
    person_data_array = poses['person_data']
    len_colors = len(colors)
    for j in range(len(person_data_array)):
        keypoints = person_data_array[j]['keypoints']
        color = colors[person_data_array[j]['track_id'] % len_colors]
        for uv in keypoints:
            p =np.array(uv,dtype=np.int32)
            cv2.circle(img,p,3,color,-1)
    
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

def display_bounding_boxes_and_labels(img, poses, colors=[(255,0,0),(0,255,0),(0,0,255)], thickness=2,font_size=4):
    """
    Display bounding boxes and labels for each pose.

    Parameters:
    img (np.ndarray): The image on which to draw the bounding boxes and labels.
    poses (dict): The poses data, each pose should have 'bbox' and 'id' fields.
    color (tuple): The color of the bounding boxes and labels (default is green).
    thickness (int): The thickness of the bounding boxes.

    Returns:
    np.ndarray: The image with bounding boxes and labels drawn.
    """
    len_colors = len(colors)
    for pose in poses['person_data']:
        bbox = pose['bbox']
        trackId = pose['track_id']
        color = colors[trackId % len_colors]
        # Draw the bounding box
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
        # Draw the label
        cv2.putText(img, str(trackId), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, font_size)

    return img

@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help='annotations root folder')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_193559", help='name of the clips folder')
@click.option('--camera_data_path', type=click.STRING, required=True, default="camera_data", help='relative path folder for the camera calibration file')
@click.option('--calibration_file', type=click.STRING, required=True, default="camera.json", help='JSON calibration file')
@click.option('--input_2d_poses', type=click.STRING, required=True, default="cleaned", help='relative path folder for the output')
@click.option('--input_3d_poses', type=click.STRING, required=True, default="output_3d_150", help='relative path folder for the output')
@click.option('--display_tracking_info', type=click.BOOL, required=True, default=True, help='Display tracking info')
@click.option('--skip_frame_step', type=click.INT, required=True, default=5, help='Skip Frames Step')
@click.option('--playback_fps', type=click.INT, required=True, default=30, help='Playback FPS')
def main(input_path,clip_name,calibration_file,camera_data_path,input_2d_poses,input_3d_poses,display_tracking_info, skip_frame_step, playback_fps):
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
        poses_2d_path = Path.joinpath(Path(input_path),clip_name,input_2d_poses,cameras_xi_names[camera_id])
        camera_frames = glob.glob(str(poses_2d_path) + '/*.json')
        if camera_id not in multiview_frames:
            multiview_frames[camera_id] = []
        multiview_frames[camera_id] += camera_frames

    debug_poses_frames = glob.glob(input_path + f'/{clip_name}/{input_3d_poses}/debug/poses*.json')
    poses_frames = glob.glob(input_path + f'/{clip_name}/{input_3d_poses}/*.json')
    colors = [get_random_color() for i in range(0,10)]
    num_frames = len(multiview_frames['1'])
    for key,camera in cameras.items():
        video_file = os.path.join(input_path,clip_name,f"{cameras_xi_names[key]}.mp4")
        cap = cv2.VideoCapture(video_file)
        #playback_fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened():
            print("Error opening video file")
            continue
        for i in range(0,num_frames,skip_frame_step):
            # TODO: read the fps from the video file

            # Capture frame-by-frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = cap.read()
            if not ret:
                break

            frame_num = f"{i:06d}"
            if ret:
                poses_2d  = load_json(multiview_frames[key][i])
                # display 2d poses
                img_poses2d = copy.deepcopy(img)
                if(display_tracking_info):
                    # Display the bounding boxes and labels
                    poses = load_json(multiview_frames[key][i])
                    img_poses2d = display_bounding_boxes_and_labels(img_poses2d, poses,colors=colors)
                display_2d_pose(img_poses2d,poses_2d,size=3,colors=colors)

                # display 3d poses
                img_poses3d = copy.deepcopy(img)
                if len(poses_frames) > 0:
                    poses_3d = load_json(poses_frames[i])
                    if len(poses_3d) > 0:
                        display_recon_pose(img_poses3d,camera,poses_3d,size=3,color=(255,0,0))

                img_debug_poses = copy.deepcopy(img)
                if len(debug_poses_frames) > 0:
                    debug_poses = load_json(debug_poses_frames[i])
                    display_debug_poses(img_debug_poses,camera,debug_poses,threshold=500,size=3,colors=colors)

                img_horizontal = np.hstack((img, img_poses2d))
                img_horizontal1 = np.hstack((img_poses3d, img_debug_poses))

                img_tot = np.vstack((img_horizontal,img_horizontal1))
                cv2.imshow(f'Preview [{cameras_xi_names[key]}]', cv2.resize(img_tot, (1280, 720)))
                cv2.waitKey(1)
            else:
                break
        cv2.destroyAllWindows()
        cap.release()  # Close the video capture object

if __name__ == "__main__":
   main()
    
