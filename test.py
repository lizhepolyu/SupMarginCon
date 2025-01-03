"""
Author: Zhe LI
Date:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupMarginContrastiveLoss(nn.Module):
    """
    Supervised Margin Contrastive Loss.

    This loss function extends the supervised contrastive loss by incorporating a margin
    to the similarity scores, which can enhance the discriminative power of the learned features.
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin=0.2):
        super(SupMarginContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin

    def forward(self, features, labels=None, mask=None):
        """
        Compute the loss given the features and labels or mask.

        Args:
            features (torch.Tensor): Feature matrix with shape [batch_size, n_views, ...].
            labels (torch.Tensor, optional): Ground truth labels with shape [batch_size].
            mask (torch.Tensor, optional): Contrastive mask with shape [batch_size, batch_size].

        Returns:
            torch.Tensor: The computed loss.
        """
        device = features.device

        if features.dim() < 3:
            raise ValueError("`features` must have at least 3 dimensions [batch_size, n_views, ...]")

        # Flatten features if necessary
        batch_size = features.size(0)
        n_views = features.size(1)
        features = features.view(batch_size, n_views, -1)

        # Normalize features
        features = F.normalize(features, p=2, dim=2)

        if labels is not None and mask is not None:
            raise ValueError("Cannot specify both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.size(0) != batch_size:
                raise ValueError("Number of labels does not match batch_size")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Contrast features
        contrast_features = features.view(batch_size * n_views, -1)

        if self.contrast_mode == 'one':
            anchor_features = features[:, 0]
        elif self.contrast_mode == 'all':
            anchor_features = contrast_features
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute similarity matrix
        similarity_matrix = torch.matmul(anchor_features, contrast_features.T)

        # Apply margin
        mask_repeat = mask.repeat(n_views, n_views)
        similarity_with_margin = similarity_matrix - self.margin * mask_repeat

        # For numerical stability
        logits = similarity_with_margin / self.temperature

        # Mask out self-contrast cases
        logits_mask = torch.ones_like(mask_repeat, device=device).scatter_(
            1,
            torch.arange(batch_size * n_views, device=device).view(-1, 1),
            0
        )
        mask_repeat = mask_repeat * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (mask_repeat * log_prob).sum(dim=1) / (mask_repeat.sum(dim=1) + 1e-12)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


if __name__ == "__main__":
    # Instantiate the loss function
    loss_fn = SupMarginContrastiveLoss()

    # Set parameters
    batch_size = 128
    feature_dim = 512

    # Generate random features and labels
    features = torch.randn(batch_size, 2, feature_dim)
    labels = torch.cat([torch.zeros(batch_size // 2), torch.ones(batch_size // 2)]).long()

    # Compute loss
    loss = loss_fn(features, labels)
    print("Loss:", loss.item())