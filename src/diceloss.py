import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, class_num, weight=None, backprop=True):
        super(SoftDiceLoss, self).__init__()
        self.class_num = class_num
        self.weight = weight
        self.backprop = backprop

    def forward(self, preds, targets):
        smooth = 1
        if self.backprop:
            loss = torch.tensor(0.0, requires_grad=True)
            target_ori = targets.squeeze(1)
            for c in range(self.class_num):
                pred = preds[:, c, :, :]
                target = (target_ori == c).int()
                if not target.any():
                    continue
                intersection = (pred * target).sum()
                loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
                loss = loss + loss_
            loss = loss / self.class_num
            return loss
        else:
            score = torch.tensor(0.0, requires_grad=False)
            pred = preds.argmax(dim=1)
            target = targets.squeeze(1)
            for c in range(self.class_num):
                pred_c = (pred == c)
                label_c = (target == c)
                if not label_c.any():
                    continue
                intersection = (pred_c & label_c).sum().float()
                union = (pred_c | label_c).sum().float()
                score_ = 2. * (intersection + smooth) / (union + smooth)
                score_ = 1 - score_.sum()
                assert not torch.any(torch.isnan(score_))
                score = score_ + score
            return score / self.class_num


