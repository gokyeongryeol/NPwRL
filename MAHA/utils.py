import numpy as np
import torch
import torch.nn as nn


def calc_log_prob(dist, squashed, 
                  dim=-1, form=None, mask=None):
    if form == '-11': #tanh activation
        squashed_ = torch.clamp(squashed, -1.0 + 1e-6, 1.0 - 1e-6)
        log_unsquashed = dist.log_prob(torch.atanh(squashed_))
        log_unsquashed = torch.clamp(log_unsquashed, -100, 100)
        log_det = torch.log(1 - torch.pow(squashed, 2) + 1e-6)
        log_prob = log_unsquashed - log_det
    elif form == '01': #sigmoid activation
        squashed_ = torch.clamp(squashed, 0.0 + 1e-6, 1.0 - 1e-6)
        log_unsquashed = dist.log_prob(torch.log(squashed_/(1-squashed_)))
        log_unsquashed = torch.clamp(log_unsquashed, -100, 100)
        log_det = torch.log(squashed * (1-squashed) + 1e-6)
        log_prob = log_unsquashed - log_det
    else: #no activation
        log_prob = dist.log_prob(squashed)
        log_prob = torch.clamp(log_prob, -100, 100)
        
    if mask is None: #done signal
        log_prob = log_prob.sum(dim=dim)
    else:
        log_prob = (log_prob * mask).sum(dim=dim)
    return log_prob

def initialize_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)