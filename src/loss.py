import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LDAMLoss(nn.Module):
    """
    LDAM (Label-Distribution-Aware Margin) Loss.
    Reference: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
    (https://arxiv.org/abs/1906.07413)
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        # Calculate margins: margin is proportional to 1 / sqrt(N_j)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.cuda.FloatTensor(m_list) if torch.cuda.is_available() else torch.FloatTensor(m_list)
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor)
        # Apply margin only to the logit of the correct class
        m_list = self.m_list.to(x.device)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1)).view((-1, 1))
        x_m = x - batch_m
    
        # Combine margin-adjusted correct logit with other original logits
        output = torch.where(index, x_m, x)
        
        # Scale by s (softmax temperature)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss using effective number of samples.
    Reference: Class-Balanced Loss Based on Effective Number of Samples 
    (https://arxiv.org/abs/1901.05555)
    """
    def __init__(self, cls_num_list, beta=0.9999, loss_type="focal", gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, cls_num_list)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(cls_num_list)
        
        self.weights = torch.FloatTensor(weights)
        self.loss_type = loss_type
        self.gamma = gamma

    def forward(self, x, target):
        # Move weights to correct device dynamically
        weights = self.weights.to(x.device)
        
        if self.loss_type == "ce":
            return F.cross_entropy(x, target, weight=weights)
        elif self.loss_type == "focal":
            # Class-balanced focal loss
            ce_loss = F.cross_entropy(x, target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
            # Apply class weights
            weight_per_sample = weights[target]
            return (focal_loss * weight_per_sample).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
