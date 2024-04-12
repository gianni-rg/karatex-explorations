import cv2
import numpy as np
import matplotlib.pyplot as plt
from CameraParamParser import * 
from GeometryUtility import *
import random
import json
import time
from HelperFunctions import *
import glob
import re
import os 

import multiprocessing
import pprint
import pickle

# for parsing arguments
import click


from concurrent.futures import ThreadPoolExecutor, wait



def serialize_to_json(fullpath, serializable_obj):
    with open(fullpath,'w') as f:
       json.dump(serializable_obj, f, indent=0)

def deserialize_from_json(fullpath):
    obj = None
    with open(fullpath,'r') as f:
       obj = json.load(f) 
    
    return obj

def poses_to_serializable(poses: dict):
   out_list = []
   for key in poses:
       out_list.extend(poses[key].reshape(-1).tolist())
   
   return out_list

def results_to_serializable_format(result_dic):
    exportable_dict = {}
    for camerakey in result_dic:
        key = str(camerakey[0]) + '_' +str(camerakey[1])
        exportable_dict[key] = {}
        for i in range(len(result_dic[camerakey]['pairs'])):
            pair_key = result_dic[camerakey]['pairs'][i]
            if(pair_key[1] < 0 ): # invalid match 
                continue

            pair = str(pair_key[0]) + '_' +str(pair_key[1])
            mse = result_dic[camerakey]['mse'][i]
            # make all the keypoints into a list of float/double 
            keys_3d = []
            keys_3d.append(mse)
            for j in range(len(result_dic[camerakey]['keys_3d'][i])):
                keys_3d.extend(result_dic[camerakey]['keys_3d'][i][j].tolist())
               # keys_3d.append(result_dic[camerakey]['keys_3d'][i][j].tolist())

            #pdata= PlayerData(keys_3d,pair,mse)
            pdata= {"points":keys_3d,"id_pair":pair,"mse":mse}
            exportable_dict[key][pair]  = keys_3d
            
    return exportable_dict

def export_intermediate_data(fullpath,dic):
    serializable_dic = results_to_serializable_format(dic)
    serialize_to_json(fullpath,serializable_dic)
    
    
def import_calibration_info(path_to_json): 
    dic = deserialize_from_json(path_to_json)
    out_dic = {}
    xyz = []
    uv = []
    for i in range(len(dic['xyz'])):
        n = int(len(dic['xyz'][i])/3)
        tmp_xyz = np.array(dic['xyz'][i])   
        tmp_xyz = tmp_xyz.reshape(n,3)
        tmp_uv = np.array(dic['uv'][i])
        tmp_uv = tmp_uv.reshape(n,2)
        xyz.append(tmp_xyz)
        uv.append(tmp_uv)
              
    out_dic['xyz'] = xyz 
    out_dic['uv'] = uv 
    out_dic['camera_ini'] = dic['camera_ini'] 
    out_dic['img_size'] = dic['img_size'] 
    out_dic['skip_cameras'] = dic['skip_cameras'] 

    return out_dic
    


# returns a dictionary of camera objects
def get_camera_params(root_folder: str, dic_xyz: dict,use_Vieww_only=False,verbose=False):
    skip_cameras = dic_xyz['skip_cameras']
    ini_file_fullpath = root_folder + '/' + dic_xyz['camera_ini']
    camera_json = root_folder + '/' + "calibration_cameras_gt.json"
    cameras_params = parse_ini_file(ini_file_fullpath,camera_json)
      # using vieww ini file only.
    toMatOpenCV = np.zeros((3,3))
    toVecOpenCV = np.zeros((3,3))

    toOpenCV = np.zeros((3,3),dtype ='double')
    toMatOpenCV[0,0] = 1
    toMatOpenCV[1,1] = -1
    toMatOpenCV[2,2] = -1


    toVecOpenCV = np.zeros((3,3),dtype ='double')
    toVecOpenCV[0,0] = -1
    toVecOpenCV[1,1] = -1
    toVecOpenCV[2,2] = -1 
    size = dic_xyz['img_size']
    cameras = {}
    n = 0
    for key in cameras_params:
        num = re.findall('[0-9]+',key)
        if int(num[0]) not in skip_cameras:
            xyz = np.float32(dic_xyz['xyz'][n])

            uv = np.float32(dic_xyz['uv'][n])
            if(verbose):
                print(f'{num[0]} {int(num[0])}')
           
            key_int = int(num[0])
            if(verbose):
                print(f'Creating camera Object')
          
            dist_coeffs = np.zeros((4,1))
            camera_matrix  = GeometryUtilities.CalculateIntrinsicMatrix(cameras_params[key],size,4.8e-3)
            dist_coeffs[0:3,0] = cameras_params[key]['dist']
            (rot_mat, rotation_vector, translation_vector,cam_pos) = GeometryUtilities.GetCameraInformationFromPoints(xyz,uv,camera_matrix,dist_coeffs)
            
          

            rot_me,tvec_me  = GeometryUtilities.CalculateRotationMatrixRHS(cameras_params[key],False)
            rot_me = toMatOpenCV.dot(rot_me)
            tvec_me = toMatOpenCV.dot(tvec_me)
            pos_me  = toVecOpenCV.dot(cameras_params[key]['pos'])
            rot_vec_me,_ = cv2.Rodrigues(rot_me)
            tvec_me = tvec_me.reshape((3,1))
            rot_vec_me = rot_vec_me.reshape((3,1))
            pos_me = pos_me.reshape((3,))

            if use_Vieww_only == True:
                cameras[key_int] = Camera(camera_matrix,rot_me,tvec_me,rot_vec_me,pos_me,dist_coeffs)
            else:
                cameras[key_int] = Camera(camera_matrix,rot_mat,translation_vector,rotation_vector,cam_pos[:,0],dist_coeffs)
        
            n += 1
    return cameras
    
nconverter = lambda x: f"{x:06d}"
# parse all annotation for each view of the frame
def parse_frame_annotations(cameras:dict,frame_path:str,frame_nbr:int,skip_cameras = [4,8,10],nformat="06d",verbose=False): #last one is not used 
    if(verbose):
        print(frame_path)
    # load all the annotations 
    annotations = {}

    keypoints = {}
    bboxes = {}
    bad_keypoints = {}
    
    for key_int in cameras:
        frame_to_format = format(frame_nbr,nformat)
        json_file = frame_path + f"\\c{key_int}\\{frame_to_format}.json"
        
        if(verbose):
            print(json_file)

        annotations[key_int] = ReadJsonAnnotation(json_file)
        keypoints_c,bboxes_c = GetKeypointsAndBbox(annotations[key_int])
        keypoints[key_int] = keypoints_c
        bboxes[key_int] = bboxes_c
        bad_keypoints[key_int] = []
        for key in keypoints_c:
            if(ValidKeypoints(keypoints_c[key]) == False):
                bad_keypoints[key_int].append(key)

     
    return keypoints,bboxes,bad_keypoints     

def get_3D_poses(cameras, keypoints,bboxes,bad_keypoints,verbose=False):
    results_camera = {}
    valid_keys = list(cameras.keys())
    if(verbose):
        print(valid_keys)

    for i in range(len(valid_keys)):
       for j in range(i+1,len(valid_keys)):      
           if i == 3 and j == 4:
               print("Breakpoint")
           if(verbose): 
               print(f'Computing {valid_keys[i]} {valid_keys[j]}')
           key = (valid_keys[i],valid_keys[j])
           res = find_matches_dual_camera_medium(cameras,keypoints,valid_keys[i],valid_keys[j],bboxes,bad_keypoints)
           results_camera[key] = res[key]
   
    return results_camera 

def find_unique_id(poses_dic: dict, output_folder:str,frame_to_format:str,mse_threshold=50.0,verbose=False,ifiles=False):
    matches = {}
    char = 'A'

    #clean up matches: remove those for mse is greater than the threshold 

    for key in poses_dic:
       if(verbose):
            print(f'Starting with cameras {key} \n')
            pairs = poses_dic[key]['pairs']
            mse_res = poses_dic[key]['mse']
            print(f'{pairs}')
            print(f'{mse_res}')
       # find matches across the selected cameras 
       if(key[0] not in  matches):
            matches[key[0]] = {}
       if(key[1] not in  matches):
            matches[key[1]] = {}
       for i in range(len(poses_dic[key]['pairs'])):
         id_1 = poses_dic[key]['pairs'][i][1]
         id_2 = poses_dic[key]['pairs'][i][0]

         if(poses_dic[key]['mse'][i] >0 and poses_dic[key]['mse'][i] <=  mse_threshold):
            if( id_1 not in matches[key[0]] and  id_2 not in matches[key[1]]):
                 matches[key[0]][id_1] = char
                 matches[key[1]][id_2] = char
                 char = chr(ord(char) +1)
            elif( id_1 not in matches[key[0]] and id_2 in matches[key[1]]):
                 matches[key[0]][id_1] = matches[key[1]][id_2]
            elif(id_1 in matches[key[0]] and id_2 not in matches[key[1]]):
                 matches[key[1]][id_2] = matches[key[0]][id_1]
            if(verbose):
                 print(f"{poses_dic[key]['pairs'][i]} {poses_dic[key]['mse'][i]}")


       if verbose:
           print(matches)
    if verbose:
        print(matches)
    matches_full = output_folder + f"/matches_{frame_to_format}.json"
    
    # save to disk 
    if(ifiles == True):
        serialize_to_json(matches_full,matches)

    return matches

def parse_multiframe_dic(dic: dict):
    
    keypoints = {}
    boxes = {}
    for key in dic:
        keypoints[key] = {}
        boxes[key] = {}
        #print(len(dic[key]))
        #print(dic[key][0])
        for i in range(len(dic[key][0])):
            keys = np.array(dic[key][0][i]['keypoints'])
            bbox = np.array(dic[key][0][i]['bbox'])
            boxes[key][i]= bbox
            keypoints[key][i] = keys

    return keypoints, boxes
# aggregate data
def average_poses(poses_dic, matches_dic,threshold=50.0,verbose=True):
    unique_poses = {}
    for key_cam in poses_dic:
        for i in range(len(poses_dic[key_cam]['pairs'])):
            if poses_dic[key_cam]['mse'][i] > threshold:
                continue

            key_pair  = poses_dic[key_cam]['pairs'][i]
            #if poses_dic[key_cam][key_pair]: # check if threshold 
            id_camera = key_cam[0]
            # NOTE le bounding box sono invertite 
            id_box    = poses_dic[key_cam]['pairs'][i][1]    
            id = "NOT_VALID"
            if(id_camera in matches_dic):
                id =  matches_dic[id_camera][id_box]

            if id  not in unique_poses:
                unique_poses[id] = []
            
            # Note : the average could be computed directly here 
            unique_poses[id].append(poses_dic[key_cam]['keys_3d'][i])
  
    # evaluate average
    average_poses = {}
    
    for key in unique_poses:
        tmp = np.array(unique_poses[key])
        average_poses[key] =  np.average(tmp, axis=0)
    
        
    return unique_poses,average_poses
    

def export_frame(cameras :dict,frame_path: str,frame_nbr:int,skip_cameras: set,output_folder: str,threshold: float,nformat="06d",ifiles=False):
    print(f"Exporting frame {frame_nbr}.")
    keypts, bboxes, bad_keypts = parse_frame_annotations(cameras,frame_path,frame_nbr,skip_cameras,nformat)
    poses = get_3D_poses(cameras,keypts,bboxes,bad_keypts)
    output_fullpath =  frame_path + '/'+ output_folder
    frame_to_format = format(frame_nbr,nformat)
    output_pose_intermediate = output_fullpath + f"/poses_{frame_to_format}.json"
    
    # these two generate intermediate values to be used in teh current version of the viewer, 
    # the average 3D points are then calulated in unity to display the results.
    # this is for debug purpose: remove in the future.
    if(ifiles==True):
        export_intermediate_data(output_pose_intermediate,poses)
    unique_dic = find_unique_id(poses,output_fullpath,frame_to_format, mse_threshold=threshold,ifiles=ifiles,verbose=True)

    unique_poses,avg_poses = average_poses(poses, unique_dic,threshold=threshold)
    poses = poses_to_serializable(avg_poses)
    dic_poses = {"data":poses}
    poses_path  = output_fullpath + f"/frame_{frame_to_format}.json"
    serialize_to_json(poses_path,dic_poses)
    print(f"Finished exporting frame {frame_nbr}.")

def export_frame_from_dic(cameras :dict,frames: dict,frame_nbr:int,skip_cameras: set,output_folder: str,threshold: float,nformat="06d",ifiles=False):
    print(f"Exporting frame {frame_nbr}.")
    keypts, bboxes = parse_multiframe_dic(frames)
    bad_keypts= {"1":[],"2":[]}
    poses = get_3D_poses(cameras,keypts,bboxes,bad_keypts)
    output_fullpath =  output_folder
    frame_to_format = format(frame_nbr,nformat)
    output_pose_intermediate = output_fullpath + f"/poses_{frame_to_format}.json"
    
    # these two generate intermediate values to be used in teh current version of the viewer, 
    # the average 3D points are then calulated in unity to display the results.
    # this is for debug purpose: remove in the future.
    if(ifiles==True):
        export_intermediate_data(output_pose_intermediate,poses)
    unique_dic = find_unique_id(poses,output_fullpath,frame_to_format, mse_threshold=threshold,ifiles=ifiles,verbose=True)

    unique_poses,avg_poses = average_poses(poses, unique_dic,threshold=threshold)
    poses = poses_to_serializable(avg_poses)
    dic_poses = {"data":poses}
    poses_path  = output_fullpath + f"/frame_{frame_to_format}.json"
    serialize_to_json(poses_path,dic_poses)
    print(f"Finished exporting frame {frame_nbr}.")


@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate", help='annotations root folder')
@click.option('--calibration_file', type=click.STRING, required=True, default="calibration_info.json", help='JSON calibration file')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_193412", help='name of the clip to export in 3D')
@click.option('--output_folder', type=click.STRING, required=True, default="output_3d", help='relative path folder for the output')
@click.option('--threshold', type=click.FLOAT, required=True, default=100, help='maximum error threshold')
@click.option('--start_frame_nbr', type=click.INT, required=True, default=0, help='frame to start from')
@click.option('--end_frame_nbr', type=click.INT, required=True, default=10, help='last frame to process')
@click.option('--ifiles', type=click.BOOL,required=True, default=True, help='Output intermediate files for debug purpose')
@click.option('--nformat', type=click.STRING, required=True, default="06d", help='numerical format of the annotation (i.e: 00001.json)')
def main(input_path,calibration_file,clip_name,start_frame_nbr,end_frame_nbr,ifiles,nformat,threshold,output_folder):
    pickles = glob.glob(input_path + f'/{clip_name}_*.pickle')
    
    camera_pickle_path = "C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate\\new_sync\\20230714_193412\\cameras_new.pkl"
    with open(camera_pickle_path,'rb') as f:
        cameras_old = pickle.load(f)
    
    cameras = {"1":cameras_old["calib_gianni_c1"], "2":cameras_old["calib_gianni_c2"]}
    #cameras = {"1":cameras_old["calib_internet_c1"], "2":cameras_old["calib_internet_c2"]}
    cam_id = 1
    
    with open(pickles[1],'rb') as f:
        frames = pickle.load(f)
    
    del frames['c3']
    del frames['c4']
   
    output_fullpath = input_path+'/'+output_folder+'/'+clip_name
    if not os.path.exists(output_fullpath):
        os.makedirs(output_fullpath)

    frame_path = input_path
    frame_nbr = 0
    # get all subfolders in the 
    # this is the part to iterate through
    futures = []
    num_threads = multiprocessing.cpu_count()
    skip_cameras = []
    pool = ThreadPoolExecutor(num_threads)
    ifiles = True
    for i in range(start_frame_nbr,end_frame_nbr):
        multiview_frame = {"1":frames['c1'][i+1],"2":frames['c2'][i+1]}
        #futures.append(pool.submit(export_frame,cameras,frame_path,i,skip_cameras,output_folder,threshold,nformat,ifiles))
        export_frame_from_dic(cameras,multiview_frame,i+1,skip_cameras,output_fullpath,threshold,nformat,ifiles)
    
    #wait(futures)
    
    

if __name__ == "__main__":
    main()
    