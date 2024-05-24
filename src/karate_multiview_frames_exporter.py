import numpy as np
import click
import glob
import os
import json 
import copy


from karate_utilities import load_json
from karate_tracking_utility import  id_BBoxes


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
@click.option('--output_folder', type=click.STRING, required=True, default="camera_data", help='relative path folder for the output')
@click.option('--threshold', type=click.FLOAT, required=True, default=100, help='maximum error threshold')
@click.option('--window_size', type=click.INT, required=True, default=10, help='frame to start from')
@click.option('--window_length', type=click.INT, required=True, default=5, help='frame to start from')
@click.option('--enable_tracking', type=click.BOOL, required=True, default=False, help='perform tracking between poses and frames')
def main(input_path,clip_name,output_folder,threshold,window_size,window_length,enable_tracking):
   pose_folder  = os.path.join(input_path,clip_name)
   pose_files = glob.glob(pose_folder + "/*.json")
   camera_names = []
   multiview_data = {}
   poses = []
   
 
   for i in range(len(pose_files)):
       basename = os.path.basename(pose_files[i])
       camera_name = basename.split(".")[0].split("_")
       camera_name = camera_name[1] + "_" + camera_name[2]
       camera_names.append(camera_name)
       pose = load_json(pose_files[i])

       frames = process_annotation(pose)
       
       if enable_tracking:    
           next_id = 0
           system_bboxes=id_BBoxes()
           # add track_id to each pose
           for j,frame in frames.items():
                pose_results = frame 
                frame_bboxes=[]
                for ii in range(0,len(pose_results)):
                    person = pose_results[ii]
                    bbox_list = []
                    keypoints_list = []
                    for bbox_i in range(len(person['bbox'])):
                        bbox_list += person['bbox'][bbox_i]
                
                    person['bbox'] = bbox_list
                    bbox=person['bbox']
                    idx=system_bboxes.get_id(bbox)

                    keypoints_list = np.array(person['keypoints'],dtype=np.float32)
                    keypoints_list = keypoints_list.reshape(-1,2)
            
                    person['keypoints'] = keypoints_list.tolist()
               
                    if idx not in frame_bboxes:
                        system_bboxes.add_track(idx, bbox)
                        frame_bboxes.append(idx)
                        player_id=idx
                    else:
                        new_idx=max(system_bboxes.bboxes.keys())+1
                        system_bboxes.add_track(new_idx, bbox)
                        frame_bboxes.append(new_idx)
                        player_id=new_idx
            
                    # update track_id
                    person['track_id'] = player_id
                    # update track_id
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
            
                    person['area'] = w*h
                    person['frameIndex'] = j
          

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