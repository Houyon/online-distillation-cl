from training.trainer import *

class ClassicNetworkTrainer(NetworkTrainer):
    
    def __init__(self, network, optimizer, criterion, device, args):
        super().__init__(network, optimizer, criterion, device, args)

    def train_epoch(self, dataloader: DataLoader):
        self.network.train()

        for i, (images, targets) in enumerate(dataloader):
            torch.cuda.empty_cache()

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.network.forward(images)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            images = images.to("cpu")
            targets = targets.to("cpu")
        
        