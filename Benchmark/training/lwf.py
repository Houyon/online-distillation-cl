from training.trainer import *

import copy

class LWFNetworkTrainer(NetworkTrainer):
    
    def __init__(self, network, optimizer, criterion, device, args):
        super().__init__(network, optimizer, criterion, device, args)
        
        self.prev_network = None
        self.prev_classes = set()
        self.seen_classes = set()
        
        self.temperature = args.temperature
        self.alpha = args.alpha
        self.update_freq = args.update_freq
        self.warmup = args.warmup
        
        self.first_update = True
        
        self.epoch_counter = 0
    
    
    def _distillation_loss(self, out, prev_out, active_units):
        au = list(active_units)
        log_p = torch.log_softmax(out[:, au] / self.temperature, dim=1)
        q = torch.softmax(prev_out[:, au] / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

    
    def penalty(self, out, x):
        
        if self.prev_network is None:
            return 0
        else:
            with torch.no_grad():
                y_prev = self.prev_network(x)
                y_curr = out
        dist_loss = 0
        yp = y_prev
        yc = y_curr
        au = self.prev_classes
        
        dist_loss += self._distillation_loss(yc, yp, au)
        return self.alpha*dist_loss  
    
    
    def update_model(self):
        self.prev_network = copy.deepcopy(self.network)
        self.prev_classes = copy.deepcopy(self.seen_classes)
    
    
    def train_epoch(self, dataloader: DataLoader):
        self.network.train()

        for i, (images, targets) in enumerate(dataloader):
            torch.cuda.empty_cache()

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.network.forward(images)

            self.optimizer.zero_grad()
            penal = self.penalty(outputs, images)
            loss = self.criterion(outputs, targets) + penal
            loss.backward()
            self.optimizer.step()

            images = images.to("cpu")
            targets = targets.to("cpu")
            
            current_classes = set(targets[:dataloader.batch_size].unique().tolist())
            self.seen_classes = self.seen_classes.union(current_classes)
        
        if self.first_update and self.epoch_counter == self.warmup:
            self.first_update = False
            self.update_model()
            self.epoch_counter = 0
            
        if not self.first_update and self.epoch_counter == self.update_freq:
            self.update_model()
            self.epoch_counter = 0
        
        self.epoch_counter += 1