from training.trainer import *
from training.classic import *

import sys
sys.path.append('../')

from datasets.dataset import *

def split_dataloader(dataloader: DataLoader, args):
    incoming_frames = dataloader.dataset.framelist[:args.studentsubsample]
    replay_frames = dataloader.dataset.framelist[args.studentsubsample:2*args.studentsubsample]
    remaining_frames = dataloader.dataset.framelist[2*args.studentsubsample:]
    
    incoming_dataset = TorchDataset(incoming_frames, args, test=False)
    replay_dataset = TorchDataset(replay_frames, args, test=False)
    remaining_dataset = TorchDataset(remaining_frames, args, test=False)
    
    incoming_dataloader = DataLoader(incoming_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=False)
    replay_dataloader = DataLoader(replay_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=False)
    remaining_dataloader = DataLoader(remaining_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=False)
    
    return incoming_dataloader, replay_dataloader, remaining_dataloader

class ACENetworkTrainer(NetworkTrainer):
    
    def __init__(self, network, optimizer, criterion, device, args):
        super().__init__(network, optimizer, criterion, device, args)
        
        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.device)
    
    def compute_ace_loss(self, image, target):
        
        image = image.to(self.device)
        target = target.to(self.device)
        
        present = target.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        
        # process data
        logits = self.network(image)
        mask = torch.zeros_like(logits)
        
        # unmask current classes
        mask[:, present] = 1
        
        #unmask unseen classs
        unseen_classes = torch.range(0, logits.size(1)-1, dtype=torch.int64)
        m = torch.ones(unseen_classes.numel(), dtype=torch.bool)
        m[self.seen_so_far] = False
        unseen_classes = unseen_classes[m]
        mask[:, unseen_classes] = 1
        
        logits = logits.masked_fill(mask == 0, -1e9)
        
        # compute loss
        loss = self.criterion(logits, target)
        
        # put back to CPU
        image = image.to('cpu')
        target = image.to('cpu')
        
        return loss
    
    
    def train_epoch(self, dataloader: DataLoader):
        
        # split
        incoming_dataloader, replay_dataloader, remaining_dataloader = split_dataloader(dataloader, self.args)
        
        self.network.train()
        
        # ACE training
        for (incoming_image, incoming_target), (replay_image, replay_target) in zip(incoming_dataloader, replay_dataloader):
            torch.cuda.empty_cache()
            
            self.optimizer.zero_grad()
            loss = self.compute_ace_loss(incoming_image, incoming_target)
            print("")
            print("LOSS")
            print(loss)
            print("")
            replay_image = replay_image.to(self.device)
            replay_target = replay_target.to(self.device)

            output = self.network.forward(replay_image)

            loss += self.criterion(output, replay_target)
            loss.backward()
            self.optimizer.step()

            replay_image = replay_image.to("cpu")
            replay_target = replay_target.to("cpu")
            
        
        # classic train on the remaining frames
        classic_trainer = ClassicNetworkTrainer(self.network, self.optimizer, self.criterion, self.device, self.args)
        classic_trainer.train_epoch(remaining_dataloader)
        self.network = classic_trainer.network