import torch
import numpy as np
import os 

from abc import abstractmethod
from torch.utils.data import DataLoader

from evaluation import SegmentationEvaluator

class NetworkTrainer():
    
    def __init__(self, network, optimizer, criterion, device, args):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.image_counter = 0
    
    @abstractmethod
    def train_epoch(self, dataloader: DataLoader):
        pass
    
    def _save_mask(self, outputs, targets):
        
        masks = np.argmax(outputs.detach().numpy(), axis=1)
        
        for mask, target in zip(masks, targets):
            output_file = 'student_' + str(self.image_counter).zfill(6) + '.npz'
            teacher_output_file = 'teacher_' + str(self.image_counter).zfill(6) + '.npz'
            output_dir = os.path.join(self.args.dataset, 'pseudo_groundtruth_seg', self.args.teacher, 'mask', self.args.onlinedataset + '-' + self.args.trainer)
            
            np.savez_compressed(os.path.join(output_dir, output_file), seg=mask)
            np.savez_compressed(os.path.join(output_dir, teacher_output_file), seg=target)
            
            self.image_counter += 1
    
    def evaluate(self, dataloader: DataLoader, evaluator: SegmentationEvaluator):
        if dataloader is None:
            return
        
        self.network.eval()
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(dataloader):
                torch.cuda.empty_cache()

                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.network.forward(images)

                images = images.to("cpu")
                targets = targets.to("cpu")
                outputs = outputs.to("cpu")
                evaluator.update(targets.cpu().numpy(),outputs.cpu().detach().numpy())
                
                self._save_mask(outputs, targets)