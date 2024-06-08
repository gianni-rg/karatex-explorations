import glob
import json
import os

import click
import numpy as np

import matplotlib.pyplot as plt

from karate_utilities import load_json, PoseSimilarityScorer

def load_3d_poses_for_clip(input_path, clip_name, annotation_folder, center=False, normalize=False):

    input_files_path = os.path.join(input_path, clip_name, annotation_folder)
    input_files  = glob.glob(input_files_path + "/*.json")

    poses_dic = []
    for json_file in input_files:
        frame = load_json(json_file)
        for pose in frame['reconstructedObjects']:
            points_array = np.asarray(pose['points'], dtype=np.double).reshape(-1,3) # required for dtw (fast version), otherwise use np.float32

            # Center the data (substract the center of the shoulders from all the points)
            # https://github.com/jflazaro/Kinect-SDK-Dynamic-Time-Warping-DTW-Gesture-Recognition-SDK1.8/blob/master/Skeleton2DDataExtract.cs
            # "5": "left_shoulder",
            # "6": "right_shoulder",
            shoulderLeft = points_array[5, :3]
            shoulderRight = points_array[6, :3]
            shoulderCenter = (shoulderLeft + shoulderRight) / 2
            shoulderCenter = shoulderCenter * 0.9 + 0.1 * points_array[0]
            points_array = np.append(points_array, shoulderCenter.reshape(-1,3), axis=0)

            hipLeft = points_array[11, :3]
            hipRight = points_array[12, :3]
            hipCenter = (hipLeft + hipRight) / 2
            hipCenter = hipCenter * 0.9 + shoulderCenter * 0.1
            points_array = np.append(points_array, hipCenter.reshape(-1,3), axis=0)

            spine = 0.30 * shoulderCenter + 0.7 * hipCenter
            points_array = np.append(points_array, spine.reshape(-1,3), axis=0)

            chest = 0.6 * spine + 0.4 * hipCenter
            points_array = np.append(points_array, chest.reshape(-1,3), axis=0)

            upperchest = shoulderCenter * 0.65 + hipCenter * 0.35
            points_array = np.append(points_array, upperchest.reshape(-1,3), axis=0)

            if center:
                points_array[:, :3] -= shoulderCenter

            # Normalization of the coordinates (divide by the distance between the shoulders)
            if normalize:
                shoulderDist = np.sqrt(np.sum((shoulderLeft - shoulderRight) ** 2))
                points_array[:, :3] /= shoulderDist

            # Normalize the vectors (unit vectors)
            points_array[:, :3] = points_array[:, :3] / np.linalg.norm(points_array[:, :3])

            poses_dic.append(points_array)

    return poses_dic

def get_frames_from_window(window_idx, offset, total_frames, compare_window_size, compare_window_stride = None):

    if compare_window_stride is None:
        compare_window_stride = compare_window_size

    frame_start = offset + window_idx * compare_window_stride
    frame_start = max(0, frame_start)
    frame_end = frame_start + compare_window_size
    frame_end = min(frame_end, total_frames)
    frames_count = frame_end - frame_start

    return frame_start, frame_end, frames_count

def compare_2_clips(input_path, input_clip_name, reference_clip_name, input_annotation_folder, reference_annotation_folder, output_folder, compare_window_size = 60, compare_window_stride = 15, input_offset=0, reference_offset=0):

     # Load reference poses
    reference_poses = load_3d_poses_for_clip(input_path, reference_clip_name, reference_annotation_folder, center=True, normalize=True)

    # Load poses to compare against the reference
    input_poses = load_3d_poses_for_clip(input_path, input_clip_name, input_annotation_folder, center=True, normalize=True)

    # Sanity checks
    ###############

    # TODO: evaluate if any have to be done

    # Compare poses
    ###############

    clip_scorer = PoseSimilarityScorer(skeleton_keypoints=23, coordinates=3)

    reference_frames_count = len(reference_poses)
    input_frames_count = len(input_poses)
    input_frames_count = min(input_frames_count, reference_frames_count)

    # Windowed comparison
    scores = []
    for window_idx in range(0, input_frames_count // compare_window_stride):
        input_frame_start, input_frame_end, input_frames_count = get_frames_from_window(window_idx, input_offset, len(input_poses), compare_window_size, compare_window_stride)
        reference_frame_start, reference_frame_end, reference_frames_count = get_frames_from_window(window_idx, reference_offset, len(reference_poses), compare_window_size, compare_window_stride)
        print(f"Comparing poses for window {window_idx} ({input_frame_start}:{input_frame_end})")
        if reference_frames_count < 0 or input_frames_count < 0:
            break
        final_score, score_list = clip_scorer.compare(
            np.asarray(input_poses[input_frame_start:input_frame_end]),
            np.asarray(reference_poses[reference_frame_start:reference_frame_end]),
            input_frames_count,
            reference_frames_count,
        )
        print(f"Similarity *{reference_clip_name}* <-> {input_clip_name}: {final_score:.2f}%")
        scores.append(max(0, final_score))

    # Not windowed
    reference_frames_count = len(reference_poses)
    input_frames_count = len(input_poses)
    input_frames_count = min(input_frames_count, reference_frames_count)
    print(f"Comparing poses for {input_frames_count} frames (not windowed)")
    full_clip_final_score, full_clip_score_list = clip_scorer.compare(
        np.asarray(input_poses[0:input_frames_count]),
        np.asarray(reference_poses[0:input_frames_count]),
        input_frames_count,
        reference_frames_count,
    )

    comparison_results = {
        "reference_clip_name": reference_clip_name,
        "input_clip_name": input_clip_name,
        "input_offset": input_offset,
        "reference_offset": reference_offset,
        "compare_window_size": compare_window_size,
        "compare_window_stride": compare_window_stride,
        "windowed_overall_similarity_score": np.mean(scores),
        "window_scores_list": scores,
        "overall_score": full_clip_final_score,
        "overall_score_list": full_clip_score_list,
    }

    print(f"Similarity *{reference_clip_name}* <-> {input_clip_name}: {final_score:.2f}%")

    # Save the result to the output folder
    output_folder_name = os.path.join(input_path, output_folder)
    os.makedirs(output_folder_name, exist_ok=True)
    fullpath_outfile = os.path.join(output_folder_name, f"{reference_clip_name}--{input_clip_name}_comparison.json")
    with open(fullpath_outfile, "w") as f:
        json.dump(comparison_results, f)

    # Create comparison plot (windowed scores)
    fig, (ax_windowed, ax_full) = plt.subplots(1, 2, figsize=(20, 10))

    # Set labels and title
    ax_full.set_title('Comparison (full clip)')
    ax_full.set_ylim(0, 100)
    # ax_full.autoscale(enable=True, axis='both', tight=True)
    ax_full.set_xlabel('Bins (frames)')
    ax_full.set_ylabel('Similarity (%)')

    ax_windowed.set_title('Comparison (windowed)')
    ax_windowed.set_ylim(0, 100)
    # ax_windowed.autoscale(enable=True, axis='both', tight=True)
    ax_windowed.set_xlabel(f'Windows ({compare_window_size} frames)')
    ax_windowed.set_ylabel('Similarity (%)')

    # Plot the scores
    ax_windowed.plot(scores, label='Scores', linestyle='-', color='b')
    ax_windowed.axhline(np.max(scores), color='k', linestyle='dashed', linewidth=1)
    ax_windowed.axhline(np.mean(scores), color='k', linestyle='dashed', linewidth=1)
    ax_full.plot(full_clip_score_list, label='Scores', linestyle='-', color='b')
    ax_full.axhline(np.max(full_clip_score_list), color='k', linestyle='dashed', linewidth=1)
    ax_full.axhline(np.mean(full_clip_score_list), color='k', linestyle='dashed', linewidth=1)

    # Save the plot
    fig.savefig(os.path.join(output_folder_name, f"{reference_clip_name}--{input_clip_name}_comparison.png"))

    return scores

@click.command()
@click.option("--input_path", type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help="annotations root folder")
@click.option("--input_clip_name", type=click.STRING, required=True, default="20230714_195835", help="name of the clip to load as input")
@click.option("--reference_clip_name", type=click.STRING, required=True, default="20230714_195553", help="name of the clip to load as reference")
@click.option("--input_annotation_folder", type=click.STRING, required=True, default="output_3d_150_smooth3d", help="relative path folder to load the input annotations")
@click.option("--reference_annotation_folder", type=click.STRING, required=True, default="output_3d_250_smooth3d", help="relative path folder to load the reference annotations")
@click.option("--output_folder", type=click.STRING, required=True, default="comparisons", help="relative path folder for the output")
@click.option("--comparison_matrix", type=click.BOOL, required=False, default=False, help="if set, the comparison will be done for all the clips in the input folder")
def main(input_path, input_clip_name, reference_clip_name, input_annotation_folder, reference_annotation_folder, output_folder, comparison_matrix):

    if not comparison_matrix:
        compare_2_clips(input_path, input_clip_name, reference_clip_name, input_annotation_folder, reference_annotation_folder, output_folder, compare_window_size=60, compare_window_stride=15, input_offset=0, reference_offset=90)
    else:
        all_clips = os.listdir(input_path)
        for clip_index_1 in range(len(all_clips)):
            for clip_index_2 in range(clip_index_1, len(all_clips)):
                compare_2_clips(input_path, all_clips[clip_index_1], all_clips[clip_index_2], input_annotation_folder, reference_annotation_folder, output_folder)


if __name__ == "__main__":
    main()
