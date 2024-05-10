import numpy as np
import json
import glob
import os 
import click
import multiprocessing

from concurrent.futures import ThreadPoolExecutor, wait

from Karate_utilities import Camera,ProjectPointsCV,GetPointCameraRayFromPixel,RayRayIntersectionExDualUpdated
from karate_load_data_exporter import load_json






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



    
nconverter = lambda x: f"{x:06d}"

def get_3d_poses_debug(cameras, keypoints,bboxes,invalid_poses_dic,verbose=False):
    results_camera = {}
    valid_keys = list(cameras.keys())
    
    # iterate through camera pairs
    for i in range(len(valid_keys)):
       for j in range(i+1,len(valid_keys)):      
           cam1_id = valid_keys[i]
           cam2_id = valid_keys[j]
           print(f'Computing reconstruction for camera {valid_keys[i]} and camera {valid_keys[j]}')
           key_camera_pair = (valid_keys[i],valid_keys[j])
           
           # iterate through all the poses in the first camera and poses in the second camera
           #res = find_matches_dual_camera_medium(cameras,keypoints,valid_keys[i],valid_keys[j],bboxes,bad_keypoints)
           
           # just for testing purpose
           #tmp = cam1_id
           #cam1_id = cam2_id
           #cam2_id = tmp
           
           cam1 = cameras[cam1_id]
           cam2 = cameras[cam2_id]
           poses_cam1 = keypoints[cam1_id]
           poses_cam2 = keypoints[cam2_id]
           invalid_poses_c1 = set(invalid_poses_dic[cam1_id])
           invalid_poses_c2 = set(invalid_poses_dic[cam2_id])
           
           #remove invalid poses form the two lists from each camera
           results_camera[key_camera_pair] = {"pairs":[],"mse":[],"keys_3d":[]}
           # iterate through all the poses in the first camera and poses in the second camera
           # and find the match 
           for id_pose_c1,pose_c1 in poses_cam1.items():
               if id_pose_c1 in invalid_poses_c1:
                   continue
               if(verbose):
                    print(f"Finding best match for pose {id_pose_c1} for camera {cam1_id}" 
                         +f" considering all poses from camera {cam2_id}")
               best_id = -1
               best_3dpoints = []
               min_mse = 10e10
               for id_pose_c2,pose_c2 in poses_cam2.items():
                   if id_pose_c2 in invalid_poses_c2:
                       continue
                   # find the match reprojection error for the two poses
                   points3d = [] 
                   mse = 0        
                   num_keypoints  = len(pose_c1)
                   bSkip = False
                   
                   # reproject the 3D points in the 3D space and compute the reprojection error
                   for k in range(num_keypoints):
                       orig_pixel_c1 = pose_c1[k,:]
                       orig_pixel_c2 = pose_c2[k,:]
                       dir_1 = GetPointCameraRayFromPixel(orig_pixel_c1,cam1.K_inv,cam1.R_inv,cam1.tvec,cam1.dist_coeffs,cam1)
                       dir_2 = GetPointCameraRayFromPixel(orig_pixel_c2,cam2.K_inv,cam2.R_inv,cam2.tvec,cam2.dist_coeffs,cam2)
                       a,b,sin = RayRayIntersectionExDualUpdated(cam1.pos,dir_1,cam2.pos,dir_2,eps=1e-4)
                       # a,b,sin =  GeometryUtilities.RayRayIntersectionExDualVieww(cam1.pos,dir_1,cam2.pos,dir_2,eps=1e-4)
                       
                       # check if no intersection or contraints violated (TODO: add contrain check)
                       if(a is None or b is None):
                            print("BAD POINT No Intersection")
                            count += 1
                            bSkip = True
                            break
                       
                       intersection = (a+b) * 0.5
                       intersection = -1 * intersection # still in Vieww space
                       points3d.append(intersection)
                       pixel_c1 = ProjectPointsCV(intersection,cam1.K,cam1.rvec,cam1.tvec,cam1.dist_coeffs)
                       pixel_c2 = ProjectPointsCV(intersection,cam2.K,cam2.rvec,cam2.tvec,cam2.dist_coeffs)

                       error_abs_c1 = np.abs(pixel_c1 - orig_pixel_c1)
                       error_abs_c2 = np.abs(pixel_c2 - orig_pixel_c2)
                       error_sqr_c1 = np.linalg.norm((pixel_c1 - orig_pixel_c1))
                       error_sqr_c2 = np.linalg.norm((pixel_c2 - orig_pixel_c2))
                       
                       # this should be the average 1st error 
                       err_sqr = (error_sqr_c1 + error_sqr_c2)/2.0
                       err_abs = (error_abs_c1 + error_abs_c2)/2.0
               
                       # total error for the two poses
                       mse += err_sqr #error_sqr
                       verbose = False
                       if(verbose):
                            print(f"Joint_{k+1} -> 3D: {intersection}: original pixels c_{cam1_id} {orig_pixel_c1}"+ 
                                  f"c{cam2_id}: {orig_pixel_c2} \n"+
                                  f"reproj: {pixel_c1} {pixel_c2} error: {err_sqr}, err_abs {err_abs}, tot error: {mse}")
                   
                   if(True):    
                       print(f"Comparing pose with id: {id_pose_c1} from camera {cam1_id} with pose with id: {id_pose_c2} from camera {cam2_id}: error: {mse}") # id_cam1 is the left camera 4-5 is camera 5
                   
                   if( mse < min_mse and bSkip == False):
                        best_id = id_pose_c2
                        min_mse = mse
                        best_3dpoints = points3d
                            
                       #dx,key3d,mse = find_match_avg(poses_cam1,poses_cam2,id_pose_c1,cam1,cam2,invalid_poses_c2,verbose)
                    
                # store the best match for the pose nth from camera 1 found among all poses in camera 2  
               #results_camera[key_camera_pair]= [(id_pose_c1,best_id)] = (best_id,best_3dpoints,mse)
               results_camera[key_camera_pair]['pairs'].append((id_pose_c1,best_id))
               results_camera[key_camera_pair]['mse'].append(min_mse)
               results_camera[key_camera_pair]['keys_3d'].append(best_3dpoints)
           
           #results_camera[key_camera_pair] = res[key_camera_pair]
   
    return results_camera


def find_unique_id(poses_dic: dict, output_folder:str,frame_to_format:str,mse_threshold=50.0,verbose=False,ifiles=False):
    matches = {}
    char = 'A'

    # iterate thoruh all the poses and delete those below a certain threshold
    #for camera_pair_key in poses_dic:
     #   for pose_pair_key in poses_dic[camera_pair_key]:
      #      print("TODO")

    #clean up matches: remove those for mse is greater than the threshold 
    

    for camera_pair_key in poses_dic:
       if(verbose):
            print(f'Starting with cameras {camera_pair_key} \n')
            pairs = poses_dic[camera_pair_key]['pairs']
            mse_res = poses_dic[camera_pair_key]['mse']
            print(f'{pairs}')
            print(f'{mse_res}')
       # find matches across the selected cameras
       camera1_id = camera_pair_key[0] 
       camera2_id = camera_pair_key[1] 
       if(camera1_id not in  matches):
            matches[camera1_id] = {}
       if(camera2_id not in  matches):
            matches[camera2_id] = {}
       # iterate through all the poses in the first camera and poses in the second camera
       for i in range(len(poses_dic[camera_pair_key]['pairs'])):
         pose1_id = poses_dic[camera_pair_key]['pairs'][i][0]
         pose2_id = poses_dic[camera_pair_key]['pairs'][i][1]

         if(poses_dic[camera_pair_key]['mse'][i] >0 and poses_dic[camera_pair_key]['mse'][i] <=  mse_threshold):
            if( pose1_id not in matches[camera1_id] and  pose2_id not in matches[camera2_id]):
                 matches[camera1_id][pose1_id] = char
                 matches[camera2_id][pose2_id] = char
                 char = chr(ord(char) +1)
            elif( pose1_id not in matches[camera1_id] and pose2_id in matches[camera2_id]):
                 matches[camera1_id][pose1_id] = matches[camera2_id][pose2_id]
            elif(pose1_id in matches[camera1_id] and pose2_id not in matches[camera2_id]):
                 matches[camera2_id][pose2_id] = matches[camera1_id][pose1_id]
            if(verbose):
                 print(f"{poses_dic[camera_pair_key]['pairs'][i]} {poses_dic[camera_pair_key]['mse'][i]}")
         
       if verbose:
           print(matches)
    if verbose:
        print(matches)
    matches_full = output_folder + f"/debug/matches_{frame_to_format}.json"
    
    # save to disk 
    if(ifiles == True):
        serialize_to_json(matches_full,matches)

    return matches


def find_unique_id_mse(poses_dic: dict, output_folder:str,frame_to_format:str,mse_threshold=50.0,verbose=False,ifiles=False):
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
    matches_full = output_folder + f"/debug/matches_{frame_to_format}.json"
    
    # save to disk 
    if(ifiles == True):
        serialize_to_json(matches_full,matches)

    return matches

def parse_multiframe_dic(dic: dict):
    
    keypoints = {}
    boxes = {}
    #for key in dic: # old version for frame loaded from pickle
    for key in dic:
        keypoints[key] = {}
        boxes[key] = {}
        #print(len(dic[key]))
        #print(dic[key][0])
        for i in range(len(dic[key]['person_data'])):
            keys = np.array(dic[key]['person_data'][i]['keypoints'])
            bbox = np.array(dic[key]['person_data'][i]['bbox'])
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
            id_box    = poses_dic[key_cam]['pairs'][i][0]    
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
 

def export_frame_from_dic(cameras :dict,frames: dict,frame_nbr:int,skip_cameras: set,output_folder: str,threshold: float,nformat="06d",ifiles=False):
    print(f"Exporting frame {frame_nbr}.")
    keypts, bboxes = parse_multiframe_dic(frames)
    invalid_poses = {"1":[],"2":[],"3":[]}
    # new version of the function
    poses = get_3d_poses_debug(cameras,keypts,bboxes,invalid_poses)
    #poses = get_3D_poses(cameras,keypts,bboxes,bad_keypts)
    output_fullpath =  output_folder
    frame_to_format = format(frame_nbr,nformat)
    debug_folder = output_fullpath + f"/debug"
    os.makedirs(debug_folder,exist_ok=True)
    output_pose_intermediate = debug_folder + f"/poses_{frame_to_format}.json"
    
    # these two generate intermediate values to be used in teh current version of the viewer, 
    # the average 3D points are then calulated in unity to display the results.
    # this is for debug purpose: remove in the future.
    if(ifiles==True):
        export_intermediate_data(output_pose_intermediate,poses)
    unique_dic = find_unique_id(poses,output_fullpath,frame_to_format, mse_threshold=threshold,ifiles=ifiles,verbose=False)

    unique_poses,avg_poses = average_poses(poses, unique_dic,threshold=threshold)
    poses = poses_to_serializable(avg_poses)
    ##dic_poses = {"data":poses}
    frame_dic = {"reconstructedObjects":[],"frameIndex":frame_nbr}
    ctr = 0
    for key in avg_poses:
        person = {"trackId":ctr,"objectTypeId":1,"points":[]}
        
        person['points'] =avg_poses[key].flatten().tolist()
        frame_dic['reconstructedObjects'].append(person)
        ctr += 1
    
    poses_path  = output_fullpath + f"/frame_{frame_to_format}.json"
    serialize_to_json(poses_path,frame_dic)
    print(f"Finished exporting frame {frame_nbr}.")


@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate", help='annotations root folder')
@click.option('--calibration_file', type=click.STRING, required=True, default="camera_data/camera.json", help='JSON calibration file')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_193412", help='name of the clip to export in 3D')
@click.option('--output_folder', type=click.STRING, required=True, default="output_3d_150", help='relative path folder for the output')
@click.option('--threshold', type=click.FLOAT, required=True, default=150, help='maximum error threshold')
@click.option('--start_frame_nbr', type=click.INT, required=True, default=0, help='frame to start from')
@click.option('--end_frame_nbr', type=click.INT, required=True, default=-1, help='last frame to process')
@click.option('--ifiles', type=click.BOOL,required=False, default=True, help='Output intermediate files for debug purpose')
@click.option('--nformat', type=click.STRING, required=True, default="06d", help='numerical format of the annotation (i.e: 00001.json)')
def main(input_path,calibration_file,clip_name,start_frame_nbr,end_frame_nbr,ifiles,nformat,threshold,output_folder):
    
    camera_calib = os.path.join(input_path,clip_name,calibration_file)
    
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

   
    output_fullpath = input_path+'/'+clip_name+'/'+output_folder
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
    camera_ids = list(cameras.keys())
    num_frames = len(multiview_frames[camera_ids[0]])
    start_frame = start_frame_nbr
    end_frame = end_frame_nbr
    if(start_frame < 0):
        start_frame = 0
  
   
    
    if(end_frame < 0):
        end_frame = num_frames
    for i in range(start_frame,end_frame):
        # create the frame by loading the files 
        
        multiview_frame = {f"{j+1}": load_json(multiview_frames[camera_ids[j]][i]) for j in range(len(camera_ids))}
        
        
        #futures.append(pool.submit(export_frame_from_dic,cameras,multiview_frame,i+1,skip_cameras,output_fullpath,threshold,nformat,ifiles))
        export_frame_from_dic(cameras,multiview_frame,i+1,skip_cameras,output_fullpath,threshold,nformat,ifiles)
    
    #wait(futures)
    
    

if __name__ == "__main__":
    main()
    