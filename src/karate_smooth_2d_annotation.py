import numpy as np
import click
import glob
import os
import json 
import copy

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from karate_utilities import load_json
from karate_tracking_utility import get_track_id

from karate_tracking_utility import  id_BBoxes



def extract_frames_from_json_files(json_files):
    poses_dic = {}
    for json_file in json_files:
        frame = load_json(json_file)
        frame_index = frame['frameIndex']
        poses_dic[frame_index] = []
        for pose in frame['reconstructedObjects']:
            points_array = np.array(pose['points'],dtype=np.float32)   
            poses_dic[frame_index].append(points_array)
            
    return poses_dic

def replace_none_with_nans(frames,shape=(5,3)):
    """
    Replace None in frames with an array of NaNs with the same shape.

    Parameters:
    frames (list): List of (5, 3) numpy arrays for each frame, with None for missing frames.

    Returns:
    np.ndarray: Frames with None replaced by NaNs.
    """
    return [np.full(shape, np.nan) if frame is None else frame for frame in frames]

def interpolate_using_window(frames, window_size=2, method='linear'):
    """
    Interpolate missing frames using a window of frames.

    Parameters:
    frames (list): List of (5, 3) numpy arrays for each frame, with NaNs for missing frames.
    window_size (int): Number of frames before and after to use for interpolation.
    method (str): Interpolation method ('linear', 'spline', etc.).

    Returns:
    np.ndarray: Interpolated frames.
    """
    num_frames = len(frames)
    interpolated_frames = np.array(frames, dtype=np.float32)
    num_points = frames[0].shape[0]
    num_dims = frames[0].shape[1]
    
    for i in range(num_frames):
        if np.isnan(frames[i]).all():
            # Determine the range for the interpolation window
            start = max(0, i - window_size)
            end = min(num_frames, i + window_size + 1)
            valid_indices = [j for j in range(start, end) if not np.isnan(frames[j]).all()]
            
            if len(valid_indices) >= 2:  # At least two points are needed for interpolation
                valid_frames = np.array([frames[j] for j in valid_indices])
                
                for point in range(num_points):  # For each of the 5 keypoints
                    for dim in range(num_dims):  # For each dimension (x, y, z)
                        f = interp1d(valid_indices, valid_frames[:, point, dim], kind=method, fill_value="extrapolate")
                        interpolated_frames[i][point][dim] = f(i)
            else:
                # If not enough points to interpolate, use the nearest valid frame
                nearest_index = min(valid_indices, key=lambda j: abs(j - i))
                interpolated_frames[i] = frames[nearest_index]
    
    return interpolated_frames

def detect_and_correct_errors(frames, threshold=1.0):
    """
    Detect and correct errors in the frames.

    Parameters:
    frames (np.ndarray): Array of shape (num_frames, 5, 3).
    threshold (float): Threshold for detecting outliers.

    Returns:
    np.ndarray: Corrected frames.
    """
    corrected_frames = frames.copy()
    num_frames = frames.shape[0]
    num_points = frames.shape[1]
    for i in range(1, num_frames - 1):
        prev_frame = frames[i - 1]
        current_frame = frames[i]
        next_frame = frames[i + 1]
        for point in range(num_points):
            if (np.linalg.norm(current_frame[point] - prev_frame[point]) > threshold and 
                np.linalg.norm(current_frame[point] - next_frame[point]) > threshold):
                corrected_frames[i][point] = (prev_frame[point] + next_frame[point]) / 2  # Correction by averaging neighboring frames
    return corrected_frames

def smooth_frames(frames, window_length=5, polyorder=2):
    """
    Apply smoothing to the frames.

    Parameters:
    frames (np.ndarray): Array of shape (num_frames, 5, 3).
    window_length (int): The length of the filter window (must be a positive odd integer).
    polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
    np.ndarray: Smoothed frames.
    """
    smoothed_frames = frames.copy()
    num_frames = frames.shape[0]
    num_points = frames.shape[1]
    num_dims = frames.shape[2]
    for point in range(num_points):
        for dim in range(num_dims):
            smoothed_frames[:, point, dim] = savgol_filter(frames[:, point, dim], window_length, polyorder)
    return smoothed_frames


def process_annotation(anno: dict) -> (dict,dict):
    keypoints = {}
    bboxes = {}
    keypoints = anno['instance_info']
    num_frames  = len(anno['instance_info'])
    frames = {}
    for i in range(num_frames):
        frame_id = keypoints[i]['frame_id']
        frames[frame_id] = []
        num_persons_framse = len(keypoints[i]['instances'])
        frame_data = []
        for j in range(num_persons_framse):
            person = {"bbox":keypoints[i]['instances'][j]['bbox']}
            person['keypoints'] = keypoints[i]['instances'][j]['keypoints'][0:23] # modify to adpapt to the number of keypoints
            frame_data.append(person)    
        
        #frames[frame_id].append(frame_data)
        frames[frame_id] = frame_data
        
    return frames

@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate", help='annotations root folder')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_193412", help='name of the clip to export in 3D')
@click.option('--annotation_folder', type=click.STRING, required=True, default="stripped", help='relative path folder for the output')
@click.option('--output_folder', type=click.STRING, required=True, default="stripped_smmothed", help='relative path folder for the output')
@click.option('--calibration_file', type=click.STRING, required=True, default="camera_data/camera.json", help='JSON calibration file')
@click.option('--threshold', type=click.FLOAT, required=True, default=100, help='maximum error threshold')
@click.option('--window_size', type=click.INT, required=True, default=10, help='frame to start from')
@click.option('--window_length', type=click.INT, required=True, default=5, help='frame to start from')
def main(input_path,clip_name,output_folder,calibration_file,annotation_folder,threshold,window_size,window_length):
   pose_folder  = os.path.join(input_path,clip_name,"stripped")
   pose_files = glob.glob(pose_folder + "/*.json")
   camera_names = []
   multiview_data = {}
   poses = []
   

   camera_calib = os.path.join(input_path,clip_name,calibration_file)
    
   with open(camera_calib,'r') as f:
        cameras_json = json.load(f)
    
   frames_per_camera = {}
   for camera_name,_ in cameras_json.items():
       folder_frames = os.path.join(input_path,clip_name,annotation_folder,camera_name)
       files = glob.glob(folder_frames+"/*.json")
       poses_dic_frames = {}
       original_frames = []
       for i, file in enumerate(files):
           frame = load_json(file)
           frame_index = frame['frame_index']
           original_frames.append(frame)
           for pose in frame['person_data']:
               points_array = np.array(pose['keypoints'],dtype=np.float32)   
               track_id = pose['track_id']
               if track_id not in poses_dic_frames:
                   poses_dic_frames[track_id] = []
               
                   # TODO: check if correct insert missing frames as None 
                   if (len(poses_dic_frames[track_id])) < i:
                      diff = i - len(poses_dic_frames[track_id])
                      poses_dic_frames[track_id] += [None]*diff
                    
               poses_dic_frames[track_id].append(points_array)
       
       frames_per_camera[camera_name] = poses_dic_frames
       

       # apply smoothing to the frames for each track_id
       for track_id,frames in poses_dic_frames.items():
              frames = replace_none_with_nans(frames)
              frames = interpolate_using_window(frames, window_size=window_size, method='linear')
              frames = detect_and_correct_errors(frames, threshold=threshold)
              frames = smooth_frames(frames, window_length=window_length, polyorder=2)
              poses_dic_frames[track_id] = frames
       
       # save the smoothed frames
       for frame in original_frames:
           frame_index = frame['frame_index']
           for pose in frame['person_data']:
               track_id = pose['track_id']
               del pose['keypoints']
               pose['keypoints'] = poses_dic_frames[track_id][frame_index-1].tolist()
                       
        
           #save the smoothed frames to the output folder
           output_folder_name = os.path.join(input_path,clip_name,output_folder,camera_name)
           os.makedirs(output_folder_name, exist_ok=True)
           output_file_name = f"{frame_index:06d}.json"
           fullpath_outfile = os.path.join(output_folder_name,output_file_name)
           with open(fullpath_outfile, 'w') as f:
                json.dump(frame, f)
   
   


if __name__ == "__main__":
    main()
    
    
    



