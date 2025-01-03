"""
Author: Zhe LI
Date: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupMarginCon(nn.Module):
   
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, margin=0.1):
        super(SupMarginCon, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin
        

    def forward(self, features, labels=None, mask=None):
        """Compute the contrastive loss for the model.
        
        Args:
            features: Hidden vectors of shape [bsz, n_views, ...].
            labels: Ground truth labels of shape [bsz].
            mask: Contrastive mask of shape [bsz, bsz], where mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A scalar loss value.
        """
        device = torch.device('cuda' if features.is_cuda else 'cpu')
       
        if len(features.shape) < 3:
            raise ValueError('`features` must have at least 3 dimensions: [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown contrast_mode: {self.contrast_mode}')

        # Compute cosine similarity between anchors and contrast features
        similarity = torch.matmul(anchor_feature, contrast_feature.T)  # [anchor_count, bsz * n_views]

        # Repeat mask to match cosine similarity matrix dimensions
        mask = mask.repeat(anchor_count, contrast_count)  # [anchor_count, bsz * n_views]
               
        sim_with_margin = similarity - self.margin * mask
        sim_with_margin = torch.clamp(sim_with_margin, -1 + 1e-7, 1 - 1e-7)
        
        anchor_dot_contrast = torch.div(sim_with_margin, self.temperature)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        
        mask = mask * logits_mask
                
        # Numerical stability adjustment
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Compute log-probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss computation
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

# Instantiate SupMarginConLoss
SupMarginConLoss = SupMarginCon()

# Set parameters
bsz = 128
dim = 512

# Randomly generate features and labels
features = torch.rand(2 * bsz, dim)
f1, f2 = torch.split(features, [bsz, bsz], dim=0)
features = torch.stack([f1, f2], dim=1)

# Dummy labels: assume the first 64 samples belong to class 0, the rest to class 1
labels = torch.cat([torch.zeros(bsz // 2), torch.ones(bsz // 2)]).long()

# Compute loss
loss = SupMarginConLoss(features, labels)
print("Loss:", loss.item())