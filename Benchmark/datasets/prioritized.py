from datasets.dataset import *
from datasets.utils import *

class PrioritizedDataset(OnlineDataset):
    
    def __init__(self, args, network, criterion): 
        super().__init__(args)
        
        self.train_list = list()
        
        self.inverse_loss_list = np.array([])
        self.probabilities = list()
        self.alpha = args.alpha
        
        self.network = network
        self.criterion = criterion

        
    def create_trainset(self):
        dataset = TorchDataset(self.train_list, self.args)
        return(DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers, pin_memory=False))   
    
    
    def step(self):
        # Update probabilities after the epoch
        self.inverse_loss_list = compute_inverse_loss_list(TorchDataset(self.train_list, self.args), self.network, self.criterion, self.args.device)
        self.probabilities = compute_probabilities(self.inverse_loss_list, self.alpha)
        
        # Inverse losses of the new frames
        frames_to_add = self.Frame_list[self.current_position:self.current_position + self.args.studentsubsample]
        frames_to_add_inverse_losses = compute_inverse_loss_list(TorchDataset(frames_to_add, self.args), self.network, self.criterion, self.args.device)
        
        for frame, inverse_loss in zip(frames_to_add, frames_to_add_inverse_losses):
            if len(self.train_list) == self.args.datasetsize:
                # select an index randomly, based on the losses
                index = np.random.choice(len(self.inverse_loss_list), p=self.probabilities)
                
                # pop the sample attached to that index
                self.train_list.pop(index)
                self.inverse_loss_list = np.delete(self.inverse_loss_list, index)
            
            # add frame, inverse of the loss into the lists, update probabilities
            self.train_list.append(frame)
            self.inverse_loss_list = np.append(self.inverse_loss_list, inverse_loss)
            self.probabilities = compute_probabilities(self.inverse_loss_list, self.alpha)
            
            
        self.current_position += self.args.studentsubsample
        