"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - see AUTHORS file

This file is part of the Online Distillation software.

This program is free software: you can redistribute it and/or modify it under the terms 
of the GNU Affero General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this 
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from natsort import natsorted
from glob import glob
from abc import abstractmethod
import numpy as np
import torch
import cv2
import os

class OnlineDataset:

    def __init__(self, args): 

        # Save the arguments
        self.args = args
        self.num_classes = args.numclasses

        # Choose between train or test mode
        self.train_mode = True

        # Create a list of entries containing video path
        # For the whole dataset
        self.current_position = 0
        video_list = natsorted(glob(os.path.join(args.dataset,'videos','*.mp4')))
        if args.start is not None and args.stop is not None:
            video_list = video_list[args.start:args.stop]
        elif args.start is not None and args.stop is None:
            video_list = video_list[args.start:]
        elif args.start is None and args.stop is not None:
            video_list = video_list[:args.stop]

        self.Frame_list = list()

        # Loop over all videos in the dataset
        for video_path in video_list:

            tmp_frame_list = list()
            video = cv2.VideoCapture(video_path)

            # Loop over all teacher subsampled frames
            for frame_id in np.arange(np.floor(video.get(cv2.CAP_PROP_FRAME_COUNT)/args.teachersubsample))*args.teachersubsample:

                tmp_frame_list.append(Frame(video_path,frame_id))

            self.Frame_list += tmp_frame_list

            video.release()
        
    def __len__(self):
        return len(self.Frame_list)
            
        
    def create_testset(self):

        frame_list = self.Frame_list[self.current_position: self.current_position + self.args.testingsetsize]
        dataset = TorchDataset(frame_list, self.args, test=True)
        
        return(DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False,num_workers=self.args.numworkers,pin_memory=False))

    
    def create_custom_set(self, start, end):
        if start < 0 or end > len(self.Frame_list):
            return None 
        frame_list = self.Frame_list[start:end]
        dataset = TorchDataset(frame_list, self.args, test=True)
        return(DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False,num_workers=self.args.numworkers,pin_memory=False))
    
    
    @abstractmethod
    def step(self):
        pass
    
    
    @abstractmethod
    def create_trainset(self):
        pass


class TorchDataset(Dataset):

	def __init__(self, frames, args, test=False):

		self.framelist = frames
		self.args = args
		self.transform = transforms.ToTensor()
		self.gt = self.args.groundtruth if test else self.args.teacher

        
	def __getitem__(self, index):

		frame = self.transform(self.read_frame(self.framelist[index].video_path, self.framelist[index].timestamp))
		target = np.load(os.path.join(self.framelist[index].video_path[::-1].replace("videos"[::-1],(os.path.join("pseudo_groundtruth_seg", self.gt))[::-1],1)[::-1][:-4], str(int(self.framelist[index].timestamp)).zfill(6) + ".npz"))['seg']
		target = torch.from_numpy(target)
		return frame, target.type(torch.LongTensor)

    
	def __len__(self):
		return len(self.framelist)
	
    
	def read_frame(self, video_path, position):
		video = cv2.VideoCapture(video_path)
		video.set(cv2.CAP_PROP_POS_FRAMES,position)
		_, frame = video.read()
		video.release()
		return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    
# Definition of the Data type
class Frame():
	def __init__(self, video_path, timestamp):
		self.video_path=video_path
		self.timestamp=timestamp