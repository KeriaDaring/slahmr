# Decoupling Human and Camera Motion from Videos in the Wild

Official PyTorch implementation of the paper Decoupling Human and Camera Motion from Videos in the Wild

[Project page](https://vye16.github.io/slahmr/)

[ArXiv](https://vye16.github.io/slahmr/)

<img src="./teaser.png">

## Getting started
This code was tested on Ubuntu 20.04 LTS and requires a CUDA-capable GPU

### Please read this first

#### The problem I've met
- The package PHALP_plus may appears wrong. But don't worry I fixed it by install the correct torch.
- The launch of slam may happend no val wrong or something like that, and we change the start and end parameter in `launch_slam.py` in `get_slam_command`
- The last progress of the project which should generate the final video and the detection2 join in at the same time, but it doesn't work. And we change our torch into the 1.11.0 then complie it again.After that, it works fine.
- Do not use env.yml which is used to create a conda env.

Please follow my step, you'll find the right way yaaa!

1. Clone repository and submodules
```
git clone --recursive https://github.com/rerun-io/slahmr.git
```
or initialize submodules if already cloned
```
git submodule update --init --recursive
```

2. You must use the `torch 1.10.0` before you complie the pytorch3d, apex. And this will be changed into 'torch 1.11.0'at the end of the whole progress.

> conda create -n env python==3.8

> conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
or 
> conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

However, if you deploy this project on linux, please make sure you have the `gcc-7` and `g++-7` which could help you to complie the `wheel` and the `cuda-11.3`. [You may need this blog](https://blog.csdn.net/weixin_43279138/article/details/126728005)

3. After you install the pytorch, you should do this stuff. Depending on the install.md from the repository of pytorch3d, you should use these command below to install it normally.
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

conda install pytorch3d -c pytorch3d
```
4. Fix up the package by using requirement.txt and three package contains in project when you use git clone
```
pip install -r requirement.txt
pip install -e . # For the function use by tracking the path
```

5. Install ViTPose
This is significant useful, and it will be use in the first run to process your video.
```
pip install -v -e third_party/PHALP_plus/ViTPose 
```

6. Install  DROID-SLAM (will take a while)
This will be use in the second command.
```
cd third_party/DROID-SLAM
python setup.py install
```
7. Install the mmcv-full=1.3.9
```
pip intall openmim -U

mim install mmcv-full==1.3.9 (optional add `-f  https://mirrors.tuna.tsinghua.edu.cn/simple) 
```
8. Download models from [here](https://drive.google.com/file/d/1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW/view?usp=sharing).
```
./download_models.sh

or
gdown https://drive.google.com/uc?id=1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW
unzip -q slahmr_dependencies.zip
rm slahmr_dependencies.zip
```

## Data
We provide configurations for dataset formats in `slahmr/confs/data`:
1. Posetrack in `slahmr/confs/data/posetrack.yaml`
2. Egobody in `slahmr/confs/data/egobody.yaml`
3. 3DPW in `slahmr/confs/data/3dpw.yaml`
4. DAVIS in `slahmr/confs/data/davis.yaml`
5. Custom video in `slahmr/confs/data/video.yaml`

**Please make sure to update all paths to data in the config files.**

We include tools to both process existing datasets we evaluated on in the paper, and to process custom data and videos.
We include experiments from the paper on the Egobody, Posetrack, and 3DPW datasets.

If you want to run on a large number of videos, or if you want to select specific people tracks for optimization,
we recommend preprocesing in advance. 
For a single downloaded video, there is no need to run preprocessing in advance.

From the `slahmr/preproc` directory, run PHALP on all your sequences
```
python launch_phalp.py --type <DATASET_TYPE> --root <DATASET_ROOT> --split <DATASET_SPLIT> --gpus <GPUS>
```
and run DROID-SLAM on all your sequences
```
python launch_slam.py --type <DATASET_TYPE> --root <DATASET_ROOT> --split <DATASET_SPLIT> --gpus <GPUS>
```
You can also update the paths to datasets in `slahmr/preproc/datasets.py` for repeated use.

## Run the code
Make sure all checkpoints have been unpacked `_DATA`.
We use hydra to launch experiments, and all parameters can be found in `slahmr/confs/config.yaml`.
If you would like to update any aspect of logging or optimization tuning, update the relevant config files.

From the `slahmr` directory (replace `<DATA_CFG>` with the dataset config name, e.g., `davis`),

When you run this, please change the torch into 1.11.0 and complie the detction2 again.

```
python run_opt.py data=<DATA_CFG> run_opt=True run_vis=True
```

We've provided a helper script `launch.py` for launching many optimization jobs in parallel.
You can specify job-specific arguments with a job spec file, such as the example files in `job_specs`,
and batch-specific arguments shared across all jobs as
```
python launch.py --gpus 1 2 -f job_specs/pt_val_shots.txt -s data=posetrack exp_name=posetrack_val
```

We've also provided a separate `run_vis.py` script for running visualization in bulk.

In addition you can get an interactive visualization of the optimization procedure and the final output using [Rerun](https://github.com/rerun-io/rerun) with `python run_rerun_vis.py --log_root <LOG_DIR>`.


###The example command we use
```
python launch_phalp.py --type 3dpw --root /root/autodl-tmp/3DPW   --seqs  courtyard_basketball_00  --gpus 0 -y 

python launch_slam.py --type 3dpw --root /root/autodl-tmp/3DPW --seqs courtyard_basketball_00 --gpus 0

python run_opt.py data=3dpw run_opt=True run_vis=True
```


## BibTeX

If you use our code in your research, please cite the following paper:
```
@inproceedings{ye2023slahmr,
    title={Decoupling Human and Camera Motion from Videos in the Wild},
    author={Ye, Vickie and Pavlakos, Georgios and Malik, Jitendra and Kanazawa, Angjoo},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2023}
}
```
