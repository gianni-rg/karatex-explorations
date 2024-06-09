import glob
import os

import click

from karate_compare_3d import compare_2_clips

@click.command()
@click.option("--input_path", type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help="annotations root folder")
@click.option("--input_annotation_folder_pattern", type=click.STRING, required=True, default="output_3d_*_smooth3d", help="relative path pattern folder to load the input annotations")
@click.option("--reference_annotation_folder_pattern", type=click.STRING, required=True, default="output_3d_*_smooth3d", help="relative path pattern folder to load the reference annotations")
@click.option("--output_folder", type=click.STRING, required=True, default="comparisons", help="relative path folder for the output")
def main(input_path, input_annotation_folder_pattern, reference_annotation_folder_pattern, output_folder):

    all_clips = os.listdir(input_path)
    for clip_index_1 in range(len(all_clips)):
        if all_clips[clip_index_1] == output_folder:
            continue
        reference_annotation_folders = glob.glob(os.path.join(input_path, all_clips[clip_index_1], reference_annotation_folder_pattern))
        if len(reference_annotation_folders) > 0 :
            reference_annotation_folder = reference_annotation_folders[0]
        else:
            print(f"No annotations found: {os.path.join(os.path.join(input_path, all_clips[clip_index_1], reference_annotation_folder_pattern))})")
            continue
        for clip_index_2 in range(clip_index_1+1, len(all_clips)):
            if all_clips[clip_index_2] == output_folder:
                continue
            input_annotation_folders = glob.glob(os.path.join(input_path, all_clips[clip_index_2], input_annotation_folder_pattern))
            if len(input_annotation_folders) > 0 :
                input_annotation_folder = input_annotation_folders[0]
            else:
                print(f"No annotations found: {os.path.join(input_path, all_clips[clip_index_2], input_annotation_folder_pattern)})")
                continue
            compare_2_clips(input_path, all_clips[clip_index_2], all_clips[clip_index_1], input_annotation_folder, reference_annotation_folder, output_folder, compare_window_size=120, compare_window_stride=120, input_offset=0, reference_offset=0)


if __name__ == "__main__":
    main()
