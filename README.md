# Online Distillation

This repository implements a benchmarking version of the continual online distillation framework. This work will be presented at the CVPR 2023 workshop.

Link to the paper: [Paper](https://openaccess.thecvf.com/content/CVPR2023W/CLVision/html/Houyon_Online_Distillation_With_Continual_Learning_for_Cyclic_Domain_Shifts_CVPRW_2023_paper.html)

To cite this paper or repository, please use the following bibtex entry:


The framework works in two steps:
1. Compute the teacher pseudo-groundtruth segmentation mask with one teacher once for all frames of the video, or a subset.
2. Train and evaluate the student network in an iterative way based on these pseudo-groundtruth teacher masks, according to the method you want to use (repla-based and/or regularization-based).

```bibtex
@InProceedings{Houyon_2023_CVPR,
    author    = {Houyon, Joachim and Cioppa, Anthony and Ghunaim, Yasir and Alfarra, Motasem and Halin, Ana{\"\i}s and Henry, Maxim and Ghanem, Bernard and Van Droogenbroeck, Marc},
    title     = {Online Distillation With Continual Learning for Cyclic Domain Shifts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2436-2445}
}
```

## Short description of the online knowledge distillation framework

This framework is an extension of the online distillation framework proposed by Cioppa et al. (Link to [Paper](https://dial.uclouvain.be/pr/boreal/object/boreal%3A219162/datastream/PDF_01/view)) that mitigates the phenomenom of catastrophic forgetting when a domain shift occurs. To do so, we adress this issue by leveraging the power of continual learning methods to reduce the impact of domain shifts. Precisely, we extend the online distillation framework by incorporating replay-based methods and/or regularization-based methods.

<p align="center">
    <image src="img/pipeline.png" width="100%"/>
<p>

## Replay-based methods

**FIFO**: stores the most recent samples in the replay buffer while removing oldest ones. This is equivalent to the original framework's update strategy.

**Uniform**: stores incoming data at randomly selected replay buffer indices. This strategy leads to an expected remaining lifespan of data decay exponentially. As for memory selection, it performs a random selection from memory for constructing a training batch.

**Prioritized**: It assigns an importance score to each frame in the replay buffer that is used as a probability of determining which samples to remove from the replay buffer. To perform the memory selection, it performs the same way.

**MIR**: It selects a subset of the replay buffer samples that are maximally interfered by the incoming data in a stream. In other words, it constructs a set of training samples from memory that are negatively affected the most by the next parameter update. It stores incoming data the same way as in **Uniform**.

## Regularization-based methods

**ER-ACE**: It aims at reducing the changes in the learned representation when training on samples from a new class. It does so by applying an asymmetric parameter update on the incoming data and the previously seen data that are sampled from a replay buffer. 

**LWF**: It uses knowledge distillation to encourage the current network's output to resemble that of a network trained on data from previous time steps. In our setup, LwF keeps a previous version of our student network $S_c$ to guide the future parameter updates of this network.

**MAS**: It assigns an importance weight for each network parameter by approximating the sensitivity of the network output to a parameter change. 

**RWalk**: it is a generalized formulation that combines a modifier version of the two popular importance-based methods: EWC and PI. RWalk computed importance scores for network parameters, similar to MAS, and regularized over the network parameters.

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
The code now expects 30fps videos at 1920x1080 resolution, but this can be changed easily if needed.

## Running the code

The following instructions will help you run the code and evaluate the results.

### Getting the pseudo-groundtruth segmentation masks

You will first need to download the baseline checkpoints and place them in mmsegmentation/checkpoints:
[pspnet](https://drive.google.com/file/d/1C3CLTcyPEgBEXyrHv-oWKEoQFcjkhtsp/view?usp=share_link), [segformer](https://drive.google.com/file/d/12tZ_JIKUkXOtlyjoXbMlGpr18-9CmByB/view?usp=share_link), [pointrend_seg](https://drive.google.com/file/d/1KufQbKrBLLL3ow-49aFJ1BQi-AnKk_dL/view?usp=share_link), [setr](https://drive.google.com/file/d/12tZ_JIKUkXOtlyjoXbMlGpr18-9CmByB/view?usp=share_link).

To compute and save the pseudo-groundtruth segmentation masks, simply run:
```
cd Preprocessing
python pseudogroundtruth-segmentation --dataset path/to/your/dataset/ --teacher teacher_name --teachersubsample 90
```

The masks will be saved in a new folder next to the videos folder, in compressed .npz format.

Additional arguments can be provided 
`--start` and `--stop` for index of first and last video to consider (python list indexing, default=None).
`--batchsize` size of the batch to use for inference, greater means faster inference but also larger memory needs (default=1).
`--teachersubsample` will take one out of x frames from the video. For 30 fps video, a value of 90 means that the teacher takes 3 seconds to compute the pseudo-groundtruth.

The teacher name can be pspnet, segformer, pointrend_seg, or setr. They are all trained on CityScapes.
If you wish to use another teacher that is not trained on Cityscape, simply download the checkpoint from the mmsegmentation github and add the config file and downloaded checkpoint path to the `init_model` function. Don't forget to change the `--num_classes` argument below if necessary.

### Training and evaluating the student

To train and evaluate the student, simply run:

```
cd Benchmark
python main.py \
    --dataset path/to/dataset/ \
    --numclasses 19 \
    --save ../Results/  \
    --teacher segformer \
    --groundtruth segformer \
    --batchsize 1 \
    --subfolder segformer \
    --teachersubsample 30 \
    --onlinedataset <replay-based-method> \
    --trainer <regularization-based-method> \
    --sequence-length 20
```

At each iteration, the student will be trained on 100 frames (`--datasetsize`, representing 5 minutes of video that has already been read, with teachersubsample = 90 and 30 fps videos) according to the chosen replay strategy (`--onlinedataset`) and the regularization-based strategy (`--trainer`). and evaluated on the next 20 frames (`--testingsetsize`, so exactly one minute). Next, the position inside the video is increased by 20 frames (`--studentsubsample`, so exactly one minute). Frames from the previous position and the current video are used to update the replay-buffer, according to the chosen replay strategy (`--onlinedataset`). The video is expected to contain sequences of 20 minutes long (`--sequence-length`)

The results are saved in a the `--save/--subfolder/experiment_x/` folder as two csv files: The first file, denoted as `performance.log` has one line per iteration and 6 values per line (accuracy, accuracy BWT, accuracy FWT, mIoU, mIoU BWT and mIoU FWT). The second file, denoted as `final.log`, has one line per iteraiton and 2 values per line (accuracy FBWT, mIoU FBWT). Note that the mIoU only discards classes with no groundtruth and no predictions. The BWT and FWT for each metric is computed according to the given time shift (`--sequence-length`).

## Method-specific paremeters

### Replay-based methods

To adjust the capacity of the buffer, use the argument `--er-capacity`. 

Here is how you can use these replay-based methods, and which arguments you must provide::

- **Uniform**: `--onlinedataset uniform`

- **Prioritized**: `--onlinedataset prioritized`
    - `--alpha`: Adjust the conservation of frames

- **MIR**: `--onlinedataset mir`
    - `-C`: subsample of the replay buffer
    - `-k`: Top-k samples to retrieve from the C samples

### Regularization-based methods

MAS, LWF and RWalk were meant to work in an offline setup. We propose to adapt them on online streams without task boundaries by using a warmup (`--warmup`) which defines a period for the network to be initialized during the warmup phase, setting $\mathcal{R} = 0$. The update frequency (`--update-freq`) simulates an artificial task boundary. 

Here is how you can use these regularization-based methods, and which arguments you must provide:

- **ACE**: `--trainer ACE`
    - `--onlinedataset`: Type of replay-based method to use

- **MAS**: `--trainer MAS`
    - `--reg`: Factor determining the importance of the weight loss
    - `--decay`: Exponential averaging factor for the importance weights

- **LWF**: `--trainer LWF`
    - `--temperature`: Adjust the temperature of the softmax layer
    - `--alpha`: Scaling factor on the regularizer

- **RWalk**:
    - `--delta-t`: Rate (in iterations) at which a score update must be triggered
    - `--decay`: Exponential averaging factor for the importance weights 

## Plots

To get the plot of the mIoU, BWT, FWT and FBWT through time, use the jupyter notebook located at `Results/_compare.ipynb`. Modify the two variables

```python
experiments = [
]
sequence_length = 
plot_methods(experiments, sequence_length)
```

- `experiments` expects a list of tuples of type (`folder_name`, `experiment_number`, `label`, `color`) where
    - `folder_name` is the folder such that `performances/folder_name` contains your experiments
    - `experiment_number` is the experiment number such that `performances/folder_name/experiment_number` contains the files `performance.log` and `final.log`
    - `label` is the method name that will be shown on the legend of the plot
    - `color` is the color of the plotted values of the metrics

For instance, 

```python
experiment = [
    ('uniform_rwalk', 2, 'Uniform-RWalk', 'orange'),
    ('mir_ace', 2, 'MIR-ACE', 'red'),
]
sequence_length = 20
plot_methods(experiments, sequence_length)
```

will plot the two methods on the same plot.

