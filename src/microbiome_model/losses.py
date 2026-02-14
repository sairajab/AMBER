import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_count_loss(relative_counts, count_pred, loss_type='mse'):
        
        #print(torch.abs(relative_counts - count_pred).max(), torch.abs(relative_counts - count_pred).min(), torch.abs(relative_counts - count_pred).median())

        if loss_type == 'mse':
            loss = torch.square(relative_counts - count_pred)
        else:
            loss = F.smooth_l1_loss(count_pred, relative_counts, beta=0.2, reduction='none')  # Huber loss
        #print("Count Loss ", loss.shape)
        return torch.squeeze(loss, axis=-1)
    

def compute_nuc_loss(tokens, nuc_pred, mask=None):
    """
    tokens:      [B, L, S]        (ground truth: int labels from 0–4)
    nuc_pred:    [B, L, S, 5]     (logits for 5 nucleotide classes)
    mask:        [B, L]           (optional; 1 for valid sequences, 0 for padding)
    """
    B, L, S, C = nuc_pred.shape
    tokens = tokens.to(dtype=torch.long)  # Ensure both are on the same device
    mask = mask.to(dtype=torch.long)  # Ensure mask is also long

    # Flatten everything for CrossEntropyLoss
    pred_flat = nuc_pred.view(-1, C)         # [B*L*S, 5]
    tokens_flat = tokens.view(-1)            # [B*L*S]


    # If there's a mask, apply it
    if mask is not None:
        # Expand mask to [B, L, S] so it aligns with every base
        mask_expanded = mask.unsqueeze(-1).expand(B, L, S)   # [B, L, S]
        mask_flat = mask_expanded.reshape(-1).float()        # [B*L*S]
        
        # Use per-token loss (no reduction)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fn(pred_flat, tokens_flat)     # [B*L*S]
        
        masked_loss = per_token_loss * mask_flat
        return masked_loss.sum() / (mask_flat.sum() + 1e-8)  # Avoid divide by zero
    else:
        # Regular loss without masking
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(pred_flat, tokens_flat)


def pairwise_distances(embeddings, squared=False):
    distances = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
    distances = distances ** 2
    distances = distances.sum(dim=-1)
    
    if not squared:
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-7  # ✅ avoid in-place +=
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)  # ✅ avoid in-place *=
    
    return distances


class PairwiseLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(PairwiseLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, y_true, y_pred):
        y_pred_dist = pairwise_distances(y_pred, squared=False)
        y_true = y_true.to(y_pred.device)  # ensure both on same device
        differences = (y_pred_dist - y_true) ** 2
        differences = torch.triu(differences)  # Upper triangular matrix
        return differences
