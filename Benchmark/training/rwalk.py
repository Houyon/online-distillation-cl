from training.trainer import *
import copy

def copy_params_dict(model, copy_grad=False):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.clone()) for k, p in model.named_parameters()]

    
def zerolike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.

    :param model: a pytorch model
    """

    return [
        (k, torch.zeros_like(p).to(p.device))
        for k, p in model.named_parameters()
    ]
    
    
class RWalkNetworkTrainer(NetworkTrainer):
    
    def __init__(self, network, optimizer, criterion, device, args):
        super().__init__(network, optimizer, criterion, device, args)
        
        self.update_freq = args.update_freq
        self.warmup = args.warmup
        
        self.ewc_alpha = args.decay
        self.ewc_lambda = args.reg
        self.delta_t = args.delta_t
        self.init_exp_info = False
        
        self.checkpoint_loss = zerolike_params_dict(network)
        self.checkpoint_scores = zerolike_params_dict(network)
        self.checkpoint_params = copy_params_dict(network)
        
        self.iter_grad = None
        self.iter_importance = None
        self.iter_params = None
        
        self.exp_scores = None
        self.exp_params = None
        self.exp_importance = None
        self.exp_penalties = None
        
        self.internal_counter = 0
        
        self.first_update = True
        self.epoch_counter = 0

        
    def _is_checkpoint_iter(self):
        return (self.internal_counter + 1) % self.delta_t == 0
    
    
    def _is_update_iter(self):
        return (self.internal_counter + 1) % self.update_freq == 0
    
    
    def _update_grad(self, image, target, network):
        network.eval()
        self.optimizer.zero_grad()
        output = network(image)
        loss = self.criterion(output, target)
        loss.backward()
        
        self.iter_grad = copy_params_dict(self.network, copy_grad=True)
    
    
    @torch.no_grad()
    def _update_loss(self):
        for (k1, old_p), (k2, new_p), (k3, p_grad), (k4, p_loss) in zip(
            self.iter_params,
            self.network.named_parameters(),
            self.iter_grad,
            self.checkpoint_loss,
        ):
            assert k1 == k2 == k3 == k4, "Error in delta-loss approximation."
            p_loss -= p_grad * (new_p - old_p)
        
    # Update parameter importance (EWC++, Eq. 6 of the RWalk paper)
    def _update_importance(self, network):
        importance = [
            (k, p.grad.data.clone().pow(2))
            for k, p in network.named_parameters()
        ]

        if self.iter_importance is None:
            self.iter_importance = importance
        else:
            old_importance = self.iter_importance
            self.iter_importance = []

            for (k1, old_imp), (k2, new_imp) in zip(old_importance, importance):
                assert k1 == k2, "Error in importance computation."
                self.iter_importance.append(
                    (
                        k1,
                        (
                            self.ewc_alpha * new_imp
                            + (1 - self.ewc_alpha) * old_imp
                        ),
                    )
                )

                
    # Add scores for a single delta_t (referred to as s_t1^t2 in the paper)
    @torch.no_grad()
    def _update_score(self):
        for (k1, score), (k2, loss), (k3, imp), (k4, old_p), (k5, new_p) in zip(
            self.checkpoint_scores,
            self.checkpoint_loss,
            self.iter_importance,
            self.checkpoint_params,
            self.network.named_parameters(),
        ):
            assert (
                k1 == k2 == k3 == k4 == k5
            ), "Error in RWalk score computation."
            
            eps = torch.finfo(loss.dtype).eps
            score += loss / (0.5 * imp * (new_p - old_p).pow(2) + eps)  
    
    def before_forward(self, image, target):
        if (self.first_update and self.epoch_counter == self.warmup) or (not self.first_update and self.epoch_counter == self.update_freq):
            
            copied_network = copy.deepcopy(self.network)
            self._update_grad(image, target, copied_network)
            self._update_importance(copied_network)
            self.iter_params = copy_params_dict(copied_network)
            self.optimizer.zero_grad()
            
            self.epoch_counter = 0
  
    
    def before_backward(self):
        if self.init_exp_info:
            exc_loss = 0
            
            for (k1, penalty), (k2, param_exp), (k3, param) in zip(
                self.exp_penalties,
                self.exp_params,
                self.network.named_parameters()
            ):
                ewc_loss += (penalty * (param-param_exp).pow(2)).sum()
                
            return self.ewc_lambda*ewc_loss
        
        return 0
            
    def after_training_iteration(self):      
        if (self.first_update and self.epoch_counter == self.warmup) or (not self.first_update and self.epoch_counter == self.update_freq):
            self.first_update = False
            
            self._update_loss()
            
            if self.internal_counter == 0 or self._is_checkpoint_iter():
                self._update_score()
                self.checkpoint_loss = zerolike_params_dict(self.network)
                self.checkpoint_params = copy_params_dict(self.network)
            
            if self.internal_counter == 0 or self._is_update_iter():
                self.update_experience_information()
                self.init_exp_info = True
                self.internal_counter = self.epoch_counter
                
            self.internal_counter += 1
    
    
    def update_experience_information(self):
        self.exp_importance = self.iter_importance
        self.exp_params = copy_params_dict(self.network)
        
        if self.exp_scores is None:
            self.exp_scores = self.checkpoint_scores
            
        else:
            exp_scores = []
            
            for (k1, p_score), (k2, p_cp_score) in zip(
                self.exp_scores, self.checkpoint_scores
            ):
                assert k1 == k2, "Error in RWalk score computation."
                exp_scores.append((k1, 0.5 * (p_score + p_cp_score)))

            self.exp_scores = exp_scores
            
        # Compute weight penalties once for all successive iterations
        # (t_k+1 variables remain constant in Eq. 8 in the paper)
        self.exp_penalties = []

        # Normalize terms in [0,1] interval, as suggested in the paper
        # (the importance is already > 0, while negative scores are relu-ed
        # out, hence we scale only the max-values of both terms)
        max_score = max(map(lambda x: x[1].max(), self.exp_scores))
        max_imp = max(map(lambda x: x[1].max(), self.exp_importance))

        for (k1, imp), (k2, score) in zip(self.exp_importance, self.exp_scores):
            assert k1 == k2, "Error in RWalk penalties computation."

            self.exp_penalties.append(
                (k1, imp / max_imp + F.relu(score) / max_score)
            )

        self.checkpoint_scores = zerolike_params_dict(self.network)
        
    
    def train_epoch(self, dataloader: DataLoader):
        self.network.train()

        for i, (images, targets) in enumerate(dataloader):
            torch.cuda.empty_cache()

            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.before_forward(images, targets)
            
            outputs = self.network.forward(images)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets) + self.before_backward()
            loss.backward()
            self.optimizer.step()

            images = images.to("cpu")
            targets = targets.to("cpu")
            
            self.after_training_iteration()
            
        self.epoch_counter += 1

        