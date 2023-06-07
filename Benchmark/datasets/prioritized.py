import random
from datasets.dataset import *
from datasets.utils import *

class PrioritizedDataset(OnlineDataset):
    
    def __init__(self, args, network, criterion): 
        super().__init__(args)
        
        self.network = network
        self.criterion = criterion
        
        self.replay_buffer = list()
        self.video_stream_list = None 
        
        self.inverse_loss_list = np.array([])
        self.probabilities = list()
        self.alpha = args.alpha
        
    def create_trainset(self):
        
        # Select the latest batch of frames from the video stream
        self.video_stream_list = self.Frame_list[max(self.current_position-self.args.studentsubsample, 0):self.current_position] 
        
        # Select randomly datasetsize-studentsubsample frames in the replay buffer
        if len(self.replay_buffer) < self.args.datasetsize-self.args.studentsubsample:
            replay_buffer_list = self.replay_buffer.copy()
        else:
            replay_buffer_list = np.random.choice(self.replay_buffer, self.args.datasetsize-self.args.studentsubsample, p=self.probabilities, replace=False)
        
        # Build the train list
        if self.video_stream_list is None:
            train_list = []
        else:
            train_list = self.video_stream_list.copy()
        train_list.extend(replay_buffer_list)
        
        dataset = TorchDataset(train_list, self.args)
        return(DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers, pin_memory=False)) 
    
    
    def step(self):
        
        #Compute loss of each new frame to add in the replay buffer
        video_stream_inverse_losses = compute_inverse_loss_list(TorchDataset(self.video_stream_list, self.args), self.network, self.criterion, self.args.device)
        
        for frame, inverse_loss in zip(self.video_stream_list, video_stream_inverse_losses):
            #remove an element from the list if the replay buffer is full
            if len(self.replay_buffer) == self.args.er_capacity:
                # select an index randomly, based on the losses
                index = np.random.choice(len(self.inverse_loss_list), p=self.probabilities)
                self.replay_buffer.pop(index)
                self.inverse_loss_list = np.delete(self.inverse_loss_list, index)
            
            # Add new frame to replay buffer
            self.replay_buffer.append(frame)
            self.inverse_loss_list = np.append(self.inverse_loss_list, inverse_loss)
            
            # Update probabilities
            self.probabilities = compute_probabilities(self.inverse_loss_list, self.alpha)
        
        # Jump in the video stream
        self.current_position += self.args.studentsubsample