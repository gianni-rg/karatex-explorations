import numpy as np
import click
import glob
import os
import json 
import copy

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from karate_utilities import load_json

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

# Example usage
frames = [
    np.array([[1.0, 2.0, 3.0]] * 5),
    None,  # Missing frame
    np.array([[1.2, 2.1, 3.1]] * 5),
    np.array([[1.4, 2.2, 3.3]] * 5),
    None,  # Missing frame
    np.array([[1.5, 2.3, 3.4]] * 5),
]

# Replace None with NaNs
frames_with_nans = replace_none_with_nans(frames)

# Interpolate missing frames using a window
interpolated_frames = interpolate_using_window(frames_with_nans)
print("Interpolated Frames:\n", interpolated_frames)

# Detect and correct errors
corrected_frames = detect_and_correct_errors(interpolated_frames)
print("Corrected Frames:\n", corrected_frames)

# Smooth the frames
smoothed_frames = smooth_frames(corrected_frames)
print("Smoothed Frames:\n", smoothed_frames)

@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="C:\\Projects\\Extra\\python\\FastAI\\Recon3D\\karate", help='annotations root folder')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_193412", help='name of the clip to export in 3D')
@click.option('--input_folder', type=click.STRING, required=True, default="output_3d_100", help='relative path folder for the output')
@click.option('--output_folder', type=click.STRING, required=True, default="output_3d_Smooth", help='relative path folder for the output')
@click.option('--threshold', type=click.FLOAT, required=True, default=100, help='maximum error threshold')
@click.option('--window_size', type=click.INT, required=True, default=50, help='frame to start from')
@click.option('--window_length', type=click.INT, required=True, default=5, help='frame to start from')
def main(input_path,clip_name,input_folder,window_size,threshold,output_folder,window_length):
    input_files_path = os.path.join(input_path,clip_name,input_folder)
    input_files  = glob.glob(input_files_path + "/*.json")
    poses_dic = extract_frames_from_json_files(input_files)
    poses_list = []
    # execute per pose
    shape = (23,3)
    #poses_list = [][ for frame_index, pose_data in poses_dic.items() if len(pose_data) > 0 else None]
    for frame_index, pose_data in poses_dic.items():
        if len(pose_data) > 0:
            
            poses_list.append(pose_data[0].reshape(-1,3))
        else:
            poses_list.append(None)
    # Replace None with NaNs
    frames_with_nans = replace_none_with_nans(poses_list, shape=shape)
    # Interpolate missing frames using a window
  
    # Interpolate missing frames using a window
    interpolated_frames = interpolate_using_window(frames_with_nans,window_size=window_size)

    # Detect and correct errors
    corrected_frames = detect_and_correct_errors(interpolated_frames)

    # Smooth the frames
    smoothed_frames = smooth_frames(corrected_frames,window_length=window_length)
    json_out_template = {"reconstructedObjects":[],"frameIndex":0}
    output_folder_full = os.path.join(input_path,clip_name,output_folder)
    os.makedirs(output_folder_full, exist_ok=True)
    
    for i in range(smoothed_frames.shape[0]):
        json_out = copy.deepcopy(json_out_template)
        json_out["reconstructedObjects"].append({"points":smoothed_frames[i,:,:].reshape(-1).tolist(),"objectTypeId":1,"trackId":0})
        json_out["frameIndex"] = i
        filename =f'{output_folder_full}/{i:06d}.json'   
        with open(filename, 'w') as f:
            json.dump(json_out, f)
    






if __name__ == "__main__":
    main()

