import glob
import json
import os

import click
import numpy as np

from dtaidistance import dtw
from karate_utilities import load_json

class Score(object):

    def __init__(self, skeleton_keypoints=23, coordinates=2):
        # TODO: change default to the number of keypoints in the skeleton
        # TODO: change default to the coordinates 2D or 3D
        self.skeleton_keypoints = skeleton_keypoints
        self.coordinates = coordinates

    def percentage_score(self, score):
        # TODO: to be replaced with a better scoring algorithm, if found in the future
        percentage = 100 - (score * 100)
        return int(percentage)

    def dtwdis(self, input_points, reference_points, num_input_points, num_reference_points):
        input_points = input_points.reshape(self.coordinates * num_input_points)
        input_points = input_points / np.linalg.norm(input_points)
        reference_points = reference_points.reshape(self.coordinates * num_reference_points)
        reference_points = reference_points / np.linalg.norm(reference_points)
        return self.percentage_score(dtw.distance_fast(reference_points, input_points))

    def normalize(self, input_test):
        for keypoint in range(0, self.skeleton_keypoints):
            input_test[:, keypoint] = input_test[:, keypoint] / np.linalg.norm(input_test[:, keypoint])
        return input_test

    def compare(self, input_points, reference_points, input_frames_count, reference_frames_count):
        input_points = self.normalize(input_points)
        scores = []
        for keypoint in range(0, self.skeleton_keypoints):
            scores.append(self.dtwdis(input_points[:, keypoint], reference_points[:, keypoint], input_frames_count, reference_frames_count))
        return np.mean(scores), scores

def load_poses_for_clip(input_path, clip_name, annotation_folder, calibration_file):

    frames_per_camera = {}

    # Load the camera/calibration  file
    camera_calib = os.path.join(input_path, clip_name, calibration_file)
    with open(camera_calib,'r') as f:
         cameras_json = json.load(f)

    for camera_name, _ in cameras_json.items():
        print(f"[{clip_name}] Loading frames for camera {camera_name}")
        folder_frames = os.path.join(input_path, clip_name, annotation_folder, camera_name)
        files = glob.glob(folder_frames + "/*.json")
        poses_dic_frames = {}

        for i, file in enumerate(files):
            frame = load_json(file)
            frame_index = frame["frame_index"]

            if len(frame["person_data"]) == 0:
                #print(f"[{clip_name}] Empty frame {frame_index} in {camera_name}")
                continue

            for pose in frame["person_data"]:
                points_array = np.array(pose["keypoints"], dtype=np.double) # required for dtw (fast version), otherwise use np.float32
                track_id = pose["track_id"]
                if track_id not in poses_dic_frames:
                    poses_dic_frames[track_id] = []
                poses_dic_frames[track_id].append(points_array)
        frames_per_camera[camera_name] = poses_dic_frames

    return frames_per_camera

@click.command()
@click.option("--input_path", type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help="annotations root folder")
@click.option("--input_clip_name", type=click.STRING, required=True, default="20230714_193412", help="name of the clip to export in 3D")
@click.option("--reference_clip_name", type=click.STRING, required=True, default="20230714_193412", help="name of the clip to export in 3D")
@click.option("--calibration_file", type=click.STRING, required=True, default="camera_data/camera.json", help="JSON calibration file")
@click.option("--input_annotation_folder", type=click.STRING, required=True, default="cleaned_smoothed_4", help="relative path folder for the output")
@click.option("--reference_annotation_folder", type=click.STRING, required=True, default="cleaned_smoothed_4", help="relative path folder for the output")
@click.option('--output_folder', type=click.STRING, required=True, default="comparisons", help='relative path folder for the output')
def main(input_path, input_clip_name, reference_clip_name, calibration_file, input_annotation_folder, reference_annotation_folder, output_folder):

    # Load reference poses
    reference_poses = load_poses_for_clip(input_path, reference_clip_name, reference_annotation_folder, calibration_file)

    # Load poses to compare against the reference
    input_poses = load_poses_for_clip(input_path, input_clip_name, input_annotation_folder, calibration_file)

    # Sanity checks
    ###############

    # Check that the number of cameras is the same
    #assert len(reference_poses) == len(input_poses), "Reference-Input cameras count mismatch"

    # Check that the number of people is the same
    # IT DOES NOT ALWAYS WORK: the camera names may not be the same in the two clips
    for camera_name in reference_poses.keys():
        assert len(reference_poses[camera_name]) == len(input_poses[camera_name]), f"Reference-Input tracked people count in camera {camera_name} mismatch"

    # Check that the number of frames is the same
    # IT DOES NOT ALWAYS WORK: the track ids may not be the same in the two clips
    # for camera_name in reference_poses.keys():
    #     for track_id in reference_poses[camera_name].keys():
    #         assert len(reference_poses[camera_name][track_id]) == len(target_poses[camera_name][track_id]), f"Reference-Input frames count for track {track_id} in camera {camera_name} mismatch"

    # Compare poses
    ###############

    clip_scorer = Score(skeleton_keypoints=23, coordinates=2)
    scores = {}
    for camera_name in reference_poses.keys():
        # Gets the first available track id in each camera
        reference_track_id = list(reference_poses[camera_name].keys())[0]
        input_track_id = list(input_poses[camera_name].keys())[0]
        reference_frames_count = len(reference_poses[camera_name][reference_track_id])
        input_frames_count = len(input_poses[camera_name][input_track_id])

        # FOR TESTING PURPOSES
        #reference_frames_count = 300 # about 10s
        #input_frames_count = 300

        print(f"Comparing {camera_name} for {input_frames_count} frames...")
        final_score, score_list = clip_scorer.compare(
            np.asarray(input_poses[camera_name][input_track_id][:input_frames_count]),
            np.asarray(reference_poses[camera_name][reference_track_id][:reference_frames_count]),
            input_frames_count,
            reference_frames_count)

        scores[camera_name] = {
            'overall_score': final_score,
            'scores_list': score_list
        }

        print(f"Similarity *{reference_clip_name}* <-> {input_clip_name} [{camera_name}]: {final_score:.2f}%")

    # Save the result to the output folder
    # output_folder_name = os.path.join(input_path, input_clip_name, output_folder)
    # os.makedirs(output_folder_name, exist_ok=True)
    # output_file_name = f"comparison_2d.json"
    # fullpath_outfile = os.path.join(output_folder_name, output_file_name)
    # with open(fullpath_outfile, "w") as f:
    #     json.dump(scores, f)

if __name__ == "__main__":
    main()
