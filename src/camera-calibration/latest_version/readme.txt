# start from 
karate_camera_pose_calculator.py to export from our json to camera pose + intrinsic 

karate_mutliview_frames_exporter.py -> from single file poses to list of pose per camera and per frame 

# karate_3d_reconstruction_final.py to perform 3d reconstruction 

# karate_pose_visualizer_debugger.py to see 2d pose, intermiediate results and final 3d results.

// TODO: 
fix orientation in the karate_camera_pose_calculator -ve sign to cam.pos && change the sign in the karate_3d_reconstruction_final after rayray intersection no more -1 result (it was fo Vieww space)
