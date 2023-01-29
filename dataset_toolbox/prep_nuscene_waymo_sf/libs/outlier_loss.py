import torch

class OutlierLoss():
    """
    Outlier loss used regularize the training of the ego-motion. Aims to prevent Sinkhorn algorithm to 
    assign to much mass to the slack row and column.

    """
    def __init__(self):

        self.reduction = 'mean'

    def __call__(self, perm_matrix):

        ref_outliers_strength = []
        src_outliers_strength = []

        for batch_idx in range(len(perm_matrix)):
            ref_outliers_strength.append(1.0 - torch.sum(perm_matrix[batch_idx], dim=1))
            src_outliers_strength.append(1.0 - torch.sum(perm_matrix[batch_idx], dim=2))

        ref_outliers_strength = torch.cat(ref_outliers_strength,1)
        src_outliers_strength = torch.cat(src_outliers_strength,0)

        if self.reduction.lower() == 'mean':
            return torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
        
        elif self.reduction.lower() == 'none':
            return  torch.mean(ref_outliers_strength, dim=1) + \
                                             torch.mean(src_outliers_strength, dim=1)