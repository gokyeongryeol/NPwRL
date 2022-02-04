import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as normal
import torch.distributions.kl as kl

import math
import numpy as np

from prev_module import MLP, Prior, Encoder, Decoder


class NP(nn.Module):
    def __init__(self, x_dim, hid_dim, r_dim, z_dim, y_dim, model,
                 lr=5e-4, clipping=False):
        super(NP, self).__init__()
        
        self.x_dim, self.y_dim = x_dim, y_dim
        self.model = model

        self.z_encoder = Encoder(x_dim, hid_dim, z_dim, y_dim, 
                                 is_stochastic=True, 
                                 is_attention='ANP' in self.model,
                                 is_multiply='MAHA' in self.model)
        
        self.decoder = Decoder(x_dim, hid_dim, r_dim, z_dim, y_dim,
                               is_attention='ANP' in self.model,
                               is_multiply='MAHA' in self.model)
        
        self.NP_optim = optim.Adam(self.parameters(), lr=lr)
        self.loss = {'KL':[], 'NLL':[]}
        self.clipping = clipping
        
    def forward(self, C, T, update_, return_, KL=None, NLL=None):
        if KL is None and NLL is None:
            Tx, Ty = T[:,:,:self.x_dim], T[:,:,-self.y_dim:]
            num_target = Tx.size(1)

            z_prior, z_p, _ = self.z_encoder.calc_latent(C, num_target, 
                                                         Tx=Tx if 'ANP' in self.model else None)
            z_posterior, z_q, _ = self.z_encoder.calc_latent(T, num_target, 
                                                             Tx=Tx if 'ANP' in self.model else None)

            KL = kl.kl_divergence(z_posterior, z_prior).sum(dim=-1).mean()

            z = z_q if self.training else z_p
            
            Ty_dist, Ty_hat = self.decoder.calc_y(C, Tx, z)

            NLL = -Ty_dist.log_prob(Ty).sum(dim=-1).mean()

        NP_loss = KL + NLL
            
        if update_:
            self.NP_optim.zero_grad()
            NP_loss.backward()
            if self.clipping:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.NP_optim.step()       
            
            self.loss['KL'].append(KL.item())
            self.loss['NLL'].append(NLL.item())
        
        if return_:
            T_MSE = F.mse_loss(Ty, Ty_hat).item()
            return T_MSE, Ty_dist
