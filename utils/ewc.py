# utils/ewc.py

import torch
import torch.nn as nn

class EWC(object):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
        """
        Initializes EWC by computing the Fisher Information matrix.
        
        Parameters:
            model (nn.Module): The neural network model.
            dataloader (DataLoader): DataLoader for computing Fisher Information.
            device (torch.device): Device to perform computations on.
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        # Store the parameters after training on the current task
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        # Compute the Fisher Information matrix
        self.fisher = self._compute_fisher()
    
    def _compute_fisher(self):
        """
        Computes the Fisher Information matrix for each parameter.
        
        Returns:
            dict: Fisher Information matrix.
        """
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(self.dataloader.dataset)
        return fisher
    
    def penalty(self, model: nn.Module):
        """
        Computes the EWC penalty.
        
        Parameters:
            model (nn.Module): The neural network model.
        
        Returns:
            torch.Tensor: EWC penalty.
        """
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss
