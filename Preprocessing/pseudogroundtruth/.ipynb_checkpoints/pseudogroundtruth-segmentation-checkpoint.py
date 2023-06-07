import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from glob import glob
from natsort import natsorted

from mmseg.apis import inference_segmentor, init_segmentor

def segment(dataset_path, model_name, start, stop, batch_size, teachersubsample):

    # Getting the videos from the dataset
    file_path = natsorted(glob(os.path.join(dataset_path,'videos','*.mp4')))
    if start is not None and stop is not None:
        file_path = file_path[start:stop]
    elif start is not None and stop is None:
        file_path = file_path[start:]
    elif start is None and stop is not None:
        file_path = file_path[:stop]

    # Initialization of the segmentation model
    model = init_model(model_name)
    
	# Loop over all selected videos
    for file in tqdm(file_path):
        # Create the folder for saving the segmentation masks
        output_video_path = os.path.join(dataset_path, 'pseudo_groundtruth_seg', model_name, os.path.basename(file)[:-4])
        os.makedirs(output_video_path, exist_ok=True)
		
        video = cv2.VideoCapture(file)
		
        frame_number = 0
        ret,frame = video.read()
        batch = list()
        batch_index = list()
        pbar = tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT))
        while(ret):
            if frame_number % teachersubsample != 0:
                ret, frame = video.read()
                frame_number += 1
                continue
                
            batch.append(frame)
            batch_index.append(frame_number)

            ret, frame = video.read()
            frame_number += 1

            if len(batch) == batch_size or not ret:
                results = inference_segmentor(model, batch)
                for i, seg in enumerate(results):
                    np.savez_compressed(os.path.join(output_video_path, str(batch_index[i]).zfill(6)), seg=seg.astype('uint8'))
                batch = list()
                batch_index = list()
                pbar.update(batch_size * teachersubsample)

        video.release()

def init_model(model_name):
	if model_name == "pspnet":
		config = os.path.join('..','..','mmsegmentation','configs','pspnet','pspnet_r101b-d8_512x1024_80k_cityscapes.py')
		checkpoint = os.path.join('..','..','mmsegmentation','checkpoints','pspnet_r101b-d8_512x1024_80k_cityscapes_20201226_170012-3a4d38ab.pth')
	elif model_name == "pointrend_seg":
		config = os.path.join('..','mmsegmentation','configs','point_rend','pointrend_r101_512x1024_80k_cityscapes.py')
		checkpoint = os.path.join('..','..','mmsegmentation','checkpoints','pointrend_r101_512x1024_80k_cityscapes_20200711_170850-d0ca84be.pth')
	elif model_name == "segformer":
		config = os.path.join('..','..','mmsegmentation','configs','segformer','segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py')
		checkpoint = os.path.join('..','..','mmsegmentation','checkpoints','segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth')
	elif model_name == "setr":
		config = os.path.join('..','..','mmsegmentation','configs','setr','setr_vit-large_pup_8x1_768x768_80k_cityscapes.py')
		checkpoint = os.path.join('..','..','mmsegmentation','checkpoints','setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth')
	return init_segmentor(config, checkpoint, device='cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":

    from arguments import args

    segment(args.dataset, args.teacher, args.start, args.stop, args.batchsize, args.teachersubsample)





