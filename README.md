# Online Distillation

This repository implements a benchmarking version of the online distillation framework.
It is intended to be used internally in the lab and not shared with third parties.

The framework works in two steps:
1. Compute the teacher pseudo-groundtruth segmentation mask with one teacher once for all frames of the video, or a subset.
2. Train and evaluate the student network in an iterative way based on these pseudo-groundtruth teacher masks.


## Getting Started
The following instructions will help you install the required libraries and the dataset to run the code. The code runs in python 3.9 and was tested in a conda environment. Pytorch is used as deep learning library. 

### Create environment
To create and setup the conda environment, simply follow the following steps:

```
conda create -n online-distillation python=3.9
conda activate online-distillation
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
apt-get update && apt-get install libgl1
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full
pip install mmsegmentation
```

Note: make sure that the pytorch-cuda version matches your cuda version installed on your computer.
Otherwise, visit: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/) for oler pytorch versions.

### Setting up the dataset

Create one folder per dataset containing a *videos* subfolder with all videos that need to be run as if they formed one big video.
The video order will be sorted by file name, so pay attention to your file path convention.
The code now expects 25fps videos at 1920x1080 resolution, but this can be changed easily if needed.

## Running the code

The following instructions will help you run the code and evaluate the results.

### Getting the pseudo-groundtruth segmentation masks

You will first need to download the baseline checkpoints and place them in mmsegmentation/checkpoints:
[pspnet](https://drive.google.com/file/d/1C3CLTcyPEgBEXyrHv-oWKEoQFcjkhtsp/view?usp=share_link), [segformer](https://drive.google.com/file/d/12tZ_JIKUkXOtlyjoXbMlGpr18-9CmByB/view?usp=share_link), [pointrend_seg](https://drive.google.com/file/d/1KufQbKrBLLL3ow-49aFJ1BQi-AnKk_dL/view?usp=share_link), [setr](https://drive.google.com/file/d/12tZ_JIKUkXOtlyjoXbMlGpr18-9CmByB/view?usp=share_link).

To compute and save the pseudo-groundtruth segmentation masks, simply run:
```
cd Preprocessing
python pseudogroundtruth-segmentation --dataset path/to/your/dataset/ --teacher teacher_name --teachersubsample 75
```

The masks will be saved in a new folder next to the videos folder, in compressed .npz format.

Additional arguments can be provided 
`--start` and `--stop` for index of first and last video to consider (python list indexing, default=None).
`--batchsize` size of the batch to use for inference, greater means faster inference but also larger memory needs (default=1).
`--teachersubsample` will take one out of x frames from the video. For 25 fps video, a value of 75 means that the teacher takes 3 seconds to compute the pseudo-groundtruth.

The teacher name can be pspnet, segformer, pointrend_seg, or setr. They are all trained on CityScapes.
If you wish to use another teacher that is not trained on Cityscape, simply download the checkpoint from the mmsegmentation github and add the config file and downloaded checkpoint path to the `init_model` function. Don't forget to change the `--num_classes` argument below if necessary.

### Training and evaluating the student

To train and evaluate the student with default parameters, simply run:

```
cd Benchmark
python main.py \
    --dataset path/to/dataset/ \
    --numclasses 19 \
    --save ../Results/  \
    --teacher pspnet \
    --groundtruth pspnet \
    --batchsize 1 \
    --subfolder pspnet \
    --teachersubsample 75
```

At each iteration, the student will be trained on the last 67 frames (`--datasetsize`, so around 5 minutes with teachersubsample = 75 and 25 fps videos) compared to its current position in the video and evaluated on the next 20 frames (`--testingsetsize`, so exactly one minute ). Next the position inside the video is increased by 20 frames (`--studentsubsample`, so exactly one minute).

The results are saved in a the `--save/--subfolder/experiment_x/` folder as a csv file with one line per iteration and two values per line (accuracy and mIoU). Note that the mIoU only discards classes with no groundtruth and no predictions.

