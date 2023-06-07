import random
import numpy as np
import torch.optim as optim
import copy

from torch.utils.data import Dataset, DataLoader

from datasets.dataset import *
from training.classic import ClassicNetworkTrainer

class MIR(OnlineDataset):

    def __init__(self, args, network, criterion, optimizer): 
        super().__init__(args)
        
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.C = args.C
        self.k = args.k
        
        self.replay_buffer = np.array([])
        self.video_stream_list = []
        
        
    def simulate_training_epoch(self, dataloader: DataLoader):
        network_copy = copy.deepcopy(self.network)
        optimizer = optim.Adam(network_copy.parameters(), lr=self.args.learningrate)
        
        trainer = ClassicNetworkTrainer(network_copy, self.optimizer, self.criterion, self.args.device, self.args)
        trainer.train_epoch(dataloader)
        
        return trainer.network
        
    
    def retrieve_top_k_samples(self, k, future_model, subset: list):
        interference = []
        dataset = TorchDataset(subset, self.args)
        
        for i in range(len(dataset)):
            torch.cuda.empty_cache()

            # compute interference associated to that sample
            image, target = dataset[i]

            image = image.unsqueeze(0).to(self.args.device)
            target = target.unsqueeze(0).to(self.args.device)
            
            pre_loss = self.criterion(self.network.forward(image), target).item()
            post_loss = self.criterion(future_model.forward(image), target).item()
            
            interference.append([post_loss-pre_loss, i])
        
        # Sort in the descending order
        interference = np.array(interference)
        interference = interference[interference[:, 0].argsort()][::-1]
        
        return interference[:, 1][:k]
        
    def create_trainset(self):
        
        if self.current_position-self.args.studentsubsample < 0:
            dataset = TorchDataset([], self.args)
            return DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers, pin_memory=False) 
        
        # Select the latest batch of frames from the video stream
        self.video_stream_list = np.array(self.Frame_list[max(self.current_position-self.args.studentsubsample, 0):self.current_position]) 
        
        # Perform one training epoch with the ongoing video stream frames
        subset = TorchDataset(self.video_stream_list, self.args)
        future_model = self.simulate_training_epoch(DataLoader(subset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers, pin_memory=False))
        
        # Retrieve C samples from the replay buffer, randomly:
        k = None
        if len(self.replay_buffer) < self.C:
            k = len(self.replay_buffer)
            replay_buffer_list = np.array(self.replay_buffer.copy())
        else:
            k = self.k
            replay_buffer_list = np.array(random.sample(self.replay_buffer.tolist(), self.C))
        
        # Build the final dataset
        train_list = np.array(self.video_stream_list.copy())
        
        if k > 0:
            # Retrieve top-k samples according to the interference produced by training with the stream data
            ids = self.retrieve_top_k_samples(k, future_model, replay_buffer_list.tolist())
            train_list = np.append(train_list, replay_buffer_list[ids.astype('int').tolist()])
        
        dataset = TorchDataset(train_list, self.args)
        return DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers, pin_memory=False)
    
    
    def step(self):
        for frame in self.video_stream_list:
            #remove an element from the list if the replay buffer is full
            if len(self.replay_buffer) == self.args.er_capacity:
                self.replay_buffer = np.delete(self.replay_buffer, random.randrange(len(self.replay_buffer)))
            self.replay_buffer = np.append(self.replay_buffer, frame)

        self.current_position += self.args.studentsubsample