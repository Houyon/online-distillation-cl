import torch
import cv2
import os

from random import randrange
from torch.utils.data import Dataset, DataLoader
from datasets.dataset import *

class UniformDataset(OnlineDataset):

    def __init__(self, args): 
        super().__init__(args)
        self.train_list = list()


    def create_trainset(self):
        dataset = TorchDataset(self.train_list, self.args)
        return(DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False,num_workers=self.args.numworkers,pin_memory=False))

    
    def step(self):
        frames_to_add = self.Frame_list[self.current_position: self.current_position + self.args.studentsubsample]

        for frame in frames_to_add:
            #remove an element from the list if the data set is full
            if len(self.train_list) == self.args.datasetsize:
                self.train_list.pop(randrange(len(self.train_list)))
            self.train_list.append(frame)

        self.current_position += self.args.studentsubsample