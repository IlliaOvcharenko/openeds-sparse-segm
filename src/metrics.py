import torch
import torchvision

import pandas as pd
import numpy as np

# TODO rename jaccard_with_logits
def jaccard(y_true, y_pred):
    """ Jaccard a.k.a IoU score for batch of images
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    
    if y_true_flat.sum() == 0 and y_pred_flat.sum() == 0:
        return torch.tensor(1.0).to(y_true.device)
    
    intersection = (y_true_flat * y_pred_flat).sum(1)
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)
    
    score = (intersection) / (union + eps)
    score = score.sum() / num
    return score

# def get_jaccard_with_logits(class_ids, threshold=0.5):    
#     if isinstance(class_ids, int):
#         class_ids = [class_ids]
    
#     def jaccard_with_logits(y_true, logits):
#         scores = []
#         y_pred = (logits.sigmoid() > threshold)
        
#         for idx in class_ids:
#             y_tr = y_true[:, idx, :, :]
#             y_pr = y_pred[:, idx, :, :]
#             if idx in class_ids:
#                 scores.append(jaccard(y_tr.long(), y_pr.long()))

#         return torch.stack(scores).mean()
    
#     return jaccard_with_logits

def get_jaccard_with_logits(class_ids):    
    if isinstance(class_ids, int):
        class_ids = [class_ids]
    
    def jaccard_with_logits(y_true, logits):
        scores = []
        y_pred = torch.argmax(logits, dim=1)
        
        for class_id in class_ids:
            
            scores.append(jaccard(
                (y_true == class_id).long(),
                (y_pred == class_id).long()
            ))

        return torch.stack(scores).mean()
    
    return jaccard_with_logits


# def dice_single_channel(probability, truth, eps = 1e-9):
#     p = probability.view(-1).float()
#     t = truth.view(-1).float()
#     dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
#     return dice

# def dice(probability, truth):
#     batch_size = truth.shape[0]
#     channel_num = truth.shape[1]
#     mean_dice_channel = 0.
#     with torch.no_grad():
#         for i in range(batch_size):
#             for j in range(channel_num):
#                 channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :])
#                 mean_dice_channel += channel_dice/(batch_size * channel_num)
#     return mean_dice_channel

# def dice_with_logits(masks, outs):
#     """ add threshold param
#     """
#     return dice((outs.sigmoid() > 0.5).float(), masks)
