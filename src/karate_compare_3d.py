import glob
import json
import os

import click
import numpy as np

from karate_utilities import load_json, PoseSimilarityScorer

def load_3d_poses_for_clip(input_path, clip_name, annotation_folder):

    input_files_path = os.path.join(input_path, clip_name, annotation_folder)
    input_files  = glob.glob(input_files_path + "/*.json")

    poses_dic = []
    for json_file in input_files:
        frame = load_json(json_file)
        for pose in frame['reconstructedObjects']:
            points_array = np.asarray(pose['points'], dtype=np.double).reshape(-1,3) # required for dtw (fast version), otherwise use np.float32
            poses_dic.append(points_array)

    return poses_dic


@click.command()
@click.option("--input_path", type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help="annotations root folder")
@click.option("--input_clip_name", type=click.STRING, required=True, default="20230714_193412", help="name of the clip to load as input")
@click.option("--reference_clip_name", type=click.STRING, required=True, default="20230714_193412", help="name of the clip to load as reference")
@click.option("--calibration_file", type=click.STRING, required=True, default="camera_data/camera.json", help="JSON calibration file")
@click.option("--input_annotation_folder", type=click.STRING, required=True, default="output_3d_150_smooth3d", help="relative path folder to load the input annotations")
@click.option("--reference_annotation_folder", type=click.STRING, required=True, default="output_3d_150_smooth3d", help="relative path folder to load the reference annotations")
@click.option("--output_folder", type=click.STRING, required=True, default="comparisons", help="relative path folder for the output")
def main(input_path, input_clip_name, reference_clip_name, calibration_file, input_annotation_folder, reference_annotation_folder, output_folder):

    # Load reference poses
    reference_poses = load_3d_poses_for_clip(input_path, reference_clip_name, reference_annotation_folder)

    # Load poses to compare against the reference
    input_poses = load_3d_poses_for_clip(input_path, input_clip_name, input_annotation_folder)

    # Sanity checks
    ###############

    # TODO: evaluate if any have to be done

    # Compare poses
    ###############

    clip_scorer = PoseSimilarityScorer(skeleton_keypoints=23, coordinates=3)

    reference_frames_count = len(reference_poses)
    input_frames_count = len(input_poses)

    # FOR TESTING PURPOSES (if not using the DWT Fast version)
    # reference_frames_count = 300 # about 10s
    # input_frames_count = 300

    print(f"Comparing poses for {input_frames_count} frames...")
    final_score, score_list = clip_scorer.compare(
        np.asarray(input_poses[:input_frames_count]),
        np.asarray(reference_poses[:reference_frames_count]),
        input_frames_count,
        reference_frames_count,
    )

    scores = {"overall_score": final_score, "scores_list": score_list}

    print(f"Similarity *{reference_clip_name}* <-> {input_clip_name}: {final_score:.2f}%")

    # TODO: Save the result to the output folder
    # output_folder_name = os.path.join(input_path, input_clip_name, output_folder)
    # os.makedirs(output_folder_name, exist_ok=True)
    # output_file_name = f"comparison_2d.json"
    # fullpath_outfile = os.path.join(output_folder_name, output_file_name)
    # with open(fullpath_outfile, "w") as f:
    #     json.dump(scores, f)


if __name__ == "__main__":
    main()
