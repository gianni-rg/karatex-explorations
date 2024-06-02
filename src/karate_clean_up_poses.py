import click
import os
import json
import glob
from tqdm import tqdm

from karate_utilities import load_json

@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help='annotations root folder')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_194103", help='name of the clip to export in 3D')
@click.option('--output_folder', type=click.STRING, required=True, default="cleaned", help='relative path folder for the output')
@click.option('--clean_poses_file', type=click.STRING, required=False, default='valid_merge_poses.txt', help='File containing the poses to keep and merge for each camera. An alternative to the valid_poses_dic and merge_poses_dic option.')
@click.option('--valid_poses_dic', type=click.STRING, required=False, default='{"K4A_Gianni":[1],"K4A_Master":[1],"K4A_Tino":[1]}', help='Specify the poses to keep for each camera')
@click.option('--merge_poses_dic', type=click.STRING, required=False, default='{"K4A_Gianni":[],"K4A_Master":[[1,6]],"K4A_Tino":[[1,7]]}', help='Merge poses for each camera')
def main(input_path,clip_name,output_folder,clean_poses_file,valid_poses_dic,merge_poses_dic):
   pose_folder  = os.path.join(input_path, clip_name)
   pose_files = glob.glob(pose_folder + "/*.json")
   camera_names = []

   clean_poses_path = os.path.join(input_path,clip_name,clean_poses_file)
   if clean_poses_path is not None and os.path.exists(clean_poses_path):
       print(f"Loading clean & merge settings from file: {clean_poses_path}")
       with open(clean_poses_path, 'r') as file:
           file_dic = json.load(file)
           poses_to_keep_dic = file_dic['valid_poses']
           poses_dic = file_dic['merge_poses']
   else:
       print(f"Loading clean & merge settings from params")
       poses_to_keep_dic = json.loads(valid_poses_dic)
       poses_dic= json.loads(merge_poses_dic)

   for i in range(len(pose_files)):
       basename = os.path.basename(pose_files[i])
       camera_name = basename.split(".")[0].split("_")
       camera_name = camera_name[1] + "_" + camera_name[2]
       camera_names.append(camera_name)

   for camera_name in camera_names:
       camera_path = os.path.join(pose_folder, camera_name)
       files = glob.glob(camera_path + "/*.json")

       valid_pose_ids = poses_to_keep_dic[camera_name]

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
