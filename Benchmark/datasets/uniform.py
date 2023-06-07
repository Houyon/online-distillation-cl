import random
from torch.utils.data import Dataset, DataLoader
from datasets.dataset import *

class UniformDataset(OnlineDataset):

    def __init__(self, args): 
        super().__init__(args)
        
        self.replay_buffer = list()
        self.video_stream_list = None 
        
    def create_trainset(self):
        
        # Select the latest batch of frames from the video stream
        self.video_stream_list = self.Frame_list[max(self.current_position-self.args.studentsubsample, 0):self.current_position] 
        
        # Select randomly datasetsize-studentsubsample frames in the replay buffer
        if len(self.replay_buffer) < self.args.datasetsize-self.args.studentsubsample:
            replay_buffer_list = self.replay_buffer.copy()
        else:
            replay_buffer_list = random.sample(self.replay_buffer, self.args.datasetsize-self.args.studentsubsample)
        
        # Build the train list
        if self.video_stream_list is None:
            train_list = []
        else:
            train_list = self.video_stream_list.copy()
        train_list.extend(replay_buffer_list)
        
        dataset = TorchDataset(train_list, self.args)
        return(DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False,num_workers=self.args.numworkers,pin_memory=False))
    
    def step(self):
        for frame in self.video_stream_list:
            #remove an element from the list if the replay buffer is full
            if len(self.replay_buffer) == self.args.er_capacity:
                self.replay_buffer.pop(random.randrange(len(self.replay_buffer)))
            self.replay_buffer.append(frame)

        self.current_position += self.args.studentsubsample