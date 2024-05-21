import json
import glob
import os
import click
from karate_utilities import load_json

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
            person['keypoints'] = keypoints[i]['instances'][j]['keypoints'][0:23] # TODO: modify to adapt to the number of keypoints
            frame_data.append(person)

        frames[frame_id] = frame_data

    return frames

@click.command()
@click.option('--input_path', type=click.STRING, required=True, default="D:\\Datasets\\karate\\Test", help='annotations root folder')
@click.option('--clip_name', type=click.STRING, required=True, default="20230714_193559", help='name of the clip to export in 3D')
@click.option('--output_folder', type=click.STRING, required=True, default="camera_data", help='relative path folder for the output')
@click.option('--output_file_name', type=click.STRING, required=True, default="multiview_frames.json", help='numerical format of the annotation (i.e: 00001.json)')
def main(input_path,clip_name,output_folder,output_file_name):
   pose_folder  = os.path.join(input_path,clip_name)
   pose_files = glob.glob(pose_folder + "/*.json")
   camera_names = []
   multiview_data = {}
   for i in range(len(pose_files)):
       basename = os.path.basename(pose_files[i])
       camera_name = basename.split(".")[0].split("_")
       camera_name = camera_name[1] + "_" + camera_name[2]
       camera_names.append(camera_name)
       pose = load_json(pose_files[i])
       frames = process_annotation(pose)
       multiview_data[camera_name] = frames
       output_folder = os.path.join(input_path,clip_name,camera_name)
       os.makedirs(output_folder, exist_ok=True)
       for frame_index,frame_data in frames.items():
           print(f"Exporting frame {frame_index} from camera {camera_name}")
           output_file_name = f"{frame_index:06d}.json"
           frame_dic = {"frame_index":frame_index,"person_data":frame_data}
           fullpath_outfile = os.path.join(output_folder,output_file_name)
           with open(fullpath_outfile, 'w') as f:
               json.dump(frame_dic, f)

if __name__ == "__main__":
    main()
