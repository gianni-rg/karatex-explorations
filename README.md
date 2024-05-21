# karatex-explorations

KarateX - Explorations to revolutionize martial arts with AI

## Introduction

**It is an on-going work in progress, built in my spare time for fun & learning.**

## Getting Started

### Project Organization

    ├── LICENSE
    ├── README.md                      <- The top-level README for developers using this project
    ├── docs                           <- Project documentation
    ├── src                            <- Source code
    │   ├── ...
    |
    └── ...                            <- other files

### Setup a local copy

1. Clone the repository (_dev_ branch)
2. Create Python 3.10.x virtual environment

    ```shell
    python -m venv .venv
    .\.venv\Scripts\activate
    python -m pip install -U pip wheel setuptools
    pip install -r .\requirements.txt
    ```

## How to run the pipeline

1. Export from custom calibration JSON file to camera pose + intrinsic configuration file
    `karate_camera_pose_calculator.py`

2. Convert single file poses to list of poses per camera and per frame
   `karate_multiview_frames_exporter.py`

3. Perform 3D reconstruction
   `karate_3d_reconstruction_final.py` 

4. Debug tool to see 2D poses, intermediate results and final 3D results.
   `karate_pose_visualizer_debugger.py` 

## Contribution

The project is constantly evolving and contributions are warmly welcomed.

I'm more than happy to receive any kind of contribution to this experimental project: from helpful feedbacks to bug reports, documentation, usage examples, feature requests, or directly code contribution for bug fixes and new and/or improved features.

Feel free to file issues and pull requests on the repository and I'll address them as much as I can, *with a best effort approach during my spare time*.

> Development is mainly done on Windows, so other platforms are not directly developed, tested or supported.  
> An help is kindly appreciated in make it work on other platforms as well.

## License

You may find specific license information for third party software in the [third-party-programs.txt](./third-party-programs.txt) file.  
Where not otherwise specified, everything is licensed under the [APACHE 2.0 License](./LICENSE).

Copyright (C) 2024 Gianni Rosa Gallina, Pietro Termine.
