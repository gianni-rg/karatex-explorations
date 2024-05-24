import click
import os   
import json
import numpy as np
import glob
from tqdm import tqdm

from karate_utilities import load_json


# TODO: add support for both file or dictionary for the pose to be kept for each camera -> --valid_poses_dic

@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help='annotations root folder')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_193921", help='name of the clip to export in 3D')
@click.option('--output_folder', type=click.STRING, required=True, default="cleaned", help='relative path folder for the output')
@click.option('--valid_poses_dic', type=click.STRING, required=True, default='{"K4A_Gianni":[1],"K4A_Master":[1],"K4A_Tino":[1]}', help='maximum error threshold')
@click.option('--merge_poses_dic', type=click.STRING, required=True, default='{"K4A_Gianni":[],"K4A_Master":[],"K4A_Tino":[]}', help='maximum error threshold')
def main(input_path,clip_name,output_folder,valid_poses_dic,merge_poses_dic):
   pose_folder  = os.path.join(input_path,clip_name)
   pose_files = glob.glob(pose_folder + "/*.json")
   camera_names = []

   # TODO: add support for both file or dictionary for the pose to be kept for each camera
   poses_to_keep_dic = json.loads(valid_poses_dic)
   poses_dic= json.loads(merge_poses_dic)
   for i in range(len(pose_files)):
       basename = os.path.basename(pose_files[i])
       camera_name = basename.split(".")[0].split("_")
       camera_name = camera_name[1] + "_" + camera_name[2]
       camera_names.append(camera_name)

   for camera_name in camera_names:
       camera_path = os.path.join(pose_folder,camera_name)
       files = glob.glob(camera_path+"/*.json")

       valid_pose_ids =  poses_to_keep_dic[camera_name]
       print(f"Correcting frames for {camera_name}")

       for file in tqdm(files):
            new_person_data = []
            frame_annotations = load_json(file)
            frame_index = frame_annotations['frame_index']
            person_data = frame_annotations['person_data']

            for pose in person_data:
                # merge poses
                if len(poses_dic[camera_name]) > 0:
                    for matchings in poses_dic[camera_name]:
                        if pose['track_id'] in matchings:
                            pose['track_id']  = matchings[0]

                # keep only valid poses
                if pose['track_id'] in valid_pose_ids:
                     new_person_data.append(pose)

            frame_annotations['person_data'] = new_person_data
            output_camera_folder = os.path.join(input_path,clip_name,output_folder,camera_name)
            os.makedirs(output_camera_folder, exist_ok=True)
            output_file_name = f"{frame_index:06d}.json"
            filename = os.path.join(output_camera_folder,output_file_name)
            # write json file
            with open(filename, 'w') as f:
                json.dump(frame_annotations, f)

if __name__ == "__main__":
    main()
