from training.trainer import *
import numpy as np
        

class MASNetworkTrainer(NetworkTrainer):
    
    def __init__(self, network, optimizer, criterion, device, args):
        super().__init__(network, optimizer, criterion, device, args)
        
        # Importance weights
        self.omega = [0 for _ in self.network.parameters()]
        self.decay = args.decay
        
        # Parameter values of the last importance weight update
        self.prev_parameters = [0 for _ in self.network.parameters()]
        
        # Regularizer; how much should we care of the change of parameters
        self.regularizer = args.reg
        
        self.warmup = args.warmup
        self.update_freq = args.update_freq
        self.epoch_counter = 0
        self.first_update = True
    
    def total_loss(self, network_outputs, targets):
        
        # Loss of the current task
        total_loss = self.criterion(network_outputs, targets)
        
        # Loss on the change of parameters
        for i, p in enumerate(self.network.parameters()):
            continual_loss = self.regularizer * torch.sum(self.omega[i]*(p-self.prev_parameters[i])**2)
            total_loss += continual_loss
        
        return total_loss
    
    
    def update_omega(self, dataloader: DataLoader):
        
        gradients = [0 for _ in self.network.parameters()]
        
        # Compute Network output for each sample in the data loader
        for i, (images, targets) in enumerate(dataloader):
            torch.cuda.empty_cache()
            
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.network.zero_grad()
            
            torch.mean(torch.linalg.vector_norm(self.network.forward(images), dim=(1, 2, 3), ord=2)).backward()
            
            images = images.to("cpu")
            targets = targets.to("cpu")
            
            for i, p in enumerate(self.network.parameters()):
                gradients[i] += torch.abs(p.grad.data.clone().detach()) / len(dataloader)
        
        if self.first_update:
            for i in range(len(gradients)):
                self.omega[i] = gradients[i]
            self.first_update = False
        else:
            # Exponential averaging
            for i in range(len(gradients)):
                self.omega[i] = self.decay*self.omega[i] + (1-self.decay)*gradients[i]
        
        self.prev_parameters = []
        for i, p in enumerate(self.network.parameters()):
            self.prev_parameters.append(p.data.clone().detach())
    
    def train_epoch(self, dataloader: DataLoader):
        
        if len(dataloader) == 0:
            return
        
        self.network.train()
            
        for i, (images, targets) in enumerate(dataloader):
            torch.cuda.empty_cache()

            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.network.forward(images)

            self.optimizer.zero_grad()
            loss = self.total_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            images = images.to("cpu")
            targets = targets.to("cpu")
            
        if self.first_update and self.epoch_counter == self.warmup:
            self.first_update = False
            self.update_omega(dataloader)
            self.epoch_counter = 0
            
        if not self.first_update and self.epoch_counter == self.update_freq:
            self.update_omega(dataloader)
            self.epoch_counter = 0
        
        self.epoch_counter += 1