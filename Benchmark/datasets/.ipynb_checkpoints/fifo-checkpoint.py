import torch
import cv2
import os

from torch.utils.data import Dataset, DataLoader

from datasets.dataset import *

class FifoDataset(OnlineDataset):

    def __init__(self, args): 
        super().__init__(args)
		
        
    def create_trainset(self):
        frame_list = self.Frame_list[max(0,-self.args.datasetsize + self.current_position): self.current_position]
        dataset = TorchDataset(frame_list, self.args)
        return(DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False,num_workers=self.args.numworkers,pin_memory=False))
    
    
    def step(self):
        self.current_position += self.args.studentsubsample