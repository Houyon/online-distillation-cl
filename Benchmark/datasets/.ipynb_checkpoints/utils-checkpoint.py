import copy
import torch.optim as optim

from datasets.dataset import *

def compute_probabilities(inverse_loss_list: list, alpha: float):
    probabilities = inverse_loss_list / np.sum(inverse_loss_list)
    return probabilities**alpha / np.sum(probabilities**alpha)

    
def compute_inverse_loss_list(dataset: TorchDataset, network, criterion, device):
    inverse_losses = np.array([])

    for i in range(len(dataset)):
        torch.cuda.empty_cache()

        # compute loss on that new sample
        image, target = dataset[i]

        image = image.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        inverse_loss = criterion(network.forward(image), target).item()**-1

        image = image.to('cpu')
        target = target.to('cpu')

        inverse_losses = np.append(inverse_losses, inverse_loss)

    return inverse_losses
    
    
