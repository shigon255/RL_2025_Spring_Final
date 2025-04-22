## Install steps
+ Run on system: Ubuntu 22.04
  + Running RLBench on Ubuntu 24.04 will fail.
+ Create conda env with python 3.9
+ Install torch 2.1.0 with cu118
+ Install 3d diffuser actor
  + Install dgl: pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
  + Install flash attention and diffuser as in the 3d diffuser actor instruction
  + Install 3d diffuser actor: 'pip install -e .'.
+ Install CALVIN
  + Please install CALVIN follow the instruction in 3d diffuser actor.
  + Then, replace the code 'calvin/calvin_env/calvin_env/camera/{gripper|static}_camera.py' with the version in this repository. This version add the focal length and intrinsic matrix get function in camera class.
    + Please don't use the instruction in get_started_calvin.md
    + Please omit the warning when installing the CALVIN about the torch version compability. CALVIN can work on torch 2.1.0+cu118.
+ Install RLBench
  + Install Open3D and PyRep as in the 3d diffuser actor instruction
  + Install RLBench follow the instruction in 3d diffuser actor.
  + Revise the close_jar condition. ([Ref](https://github.com/MohitShridhar/RLBench/pull/1/commits/587a6a0e6dc8cd36612a208724eb275fe8cb4470))
  + Replace the following file with the same file in thie repository.
    + RLBench/rlbench/action_modes.py
    + RLBench/rlbench/backend/scene.py
    + RLBench/rlbench/task_environment.py
    + PyRep/pyrep/objects/vision_sensor.py
+ Install xvfb
  + If running on headless cluster, xvfb is needed to run RLBench. 
  + Install xvfb first. And run something like 'xvfb-run python test.py'
+ Install Depth Pro
  + Please follow the install instruction on the official [repo](https://github.com/apple/ml-depth-pro)
+ Install UniDepth and Unik3D
  + Please cd into their directory, and 'pip install -e .'
  + Note that the code in the original git repository cannot work in python 3.9 due to typing issue. I've fixed it in the directory in this repository, so please use this version.
+ Testing
  + Test on RLBench
    + Please prepare the testing dataset as in the 3d diffuser actor instruction. And also prepare the pretrained weight.
    + PerAct setup stands for multi-view setup, and GNFactor stands for single-view setup.
    + Run testing: 'xvfb-run bash online_evaluation_rlbench/eval_peract.sh'
  + Test on CALVIN
    + Prepare dataset for CALVIN. (Warning: 500G data download needed)
      + If you are using CPLab workstation, the data is downloaded in /project2/yi-ray/dataset/calvin/
    + Run testing: 'bash scripts/test_trajectory_calvin.sh'
+ Gotcha
  + Sometimes install these things will install new version of numpy, setuptools, or pytorch. Please stick to the version. If you encounter any issue, please check the version of these package first
    + numpy==1.23.5
    + setuptools==57.5.0
    + torch==2.1.0+cu118
    + xformer==0.0.22.post4 ([Ref](https://github.com/facebookresearch/xformers/issues/897))
  + opencv-python sometimes cause 'xcb' related erros. Please try opencv-python_headless instead.
  + Please install PyRep on Ubuntu22.04, or GLIBC 2.83 will occur. [Ref](https://www.reddit.com/r/linux4noobs/comments/1bycyya/running_into_a_version_glibc_238_not_found_error/)
