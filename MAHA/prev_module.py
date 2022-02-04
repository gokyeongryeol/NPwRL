import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from utils import calc_log_prob, initialize_weight


class MLP(nn.Module):
    def __init__(self, inp, hid, out, activation, num_layer):
        super(MLP, self).__init__()
        
        unit = inp
        layer_list = []
        for _ in range(num_layer-1):
            layer_list.append(nn.Linear(unit, hid))
            layer_list.append(activation)
            unit = hid
        layer_list.append(nn.Linear(hid, out))
        self.net = nn.Sequential(*layer_list).apply(initialize_weight)
    
    def forward(self, x):
        x = self.net(x)
        return x
    
        
class Prior(nn.Module):
    def __init__(self, z_dim, std=1.0):
        super(Prior, self).__init__()
        self.z_dim = z_dim
        self.std = std
        
    def forward(self, batch_size):
        mean = torch.zeros(batch_size, 1, self.z_dim).cuda()
        std = self.std * torch.ones(batch_size, 1, self.z_dim).cuda()
        dist = Normal(mean, std)
        return dist

       
class Encoder(nn.Module):
    def __init__(self, x_dim, hid_dim, lat_dim, y_dim, 
                 pre_layer=4, post_layer=2, 
                 is_stochastic=False, is_attention=False, is_multiply=False):
        super(Encoder, self).__init__()
        
        self.x_dim, self.hid_dim, self.y_dim = x_dim, hid_dim, y_dim
        
        self.pre_encoder = MLP(x_dim+y_dim, hid_dim, hid_dim, nn.ReLU(), pre_layer)
        
        if is_attention:
            self.fc_qk = MLP(x_dim, hid_dim, hid_dim, nn.ReLU(), 2)
            self.fc_v = MLP(hid_dim, hid_dim, 
                            lat_dim*2 if is_stochastic else lat_dim,
                            nn.ReLU(), 2)
        else:
            self.post_encoder = MLP(hid_dim, hid_dim, 
                                    lat_dim*2 if is_stochastic else lat_dim,
                                    nn.ReLU(), post_layer)
        
        if is_multiply and not is_stochastic:
            self.fc_z = MLP(lat_dim, hid_dim, lat_dim, nn.ReLU(), 2)
            self.LN = nn.LayerNorm(lat_dim)
                
        self.is_stochastic = is_stochastic
        self.is_attention = is_attention
        self.is_multiply = is_multiply
        
    def calc_latent(self, set_, num_target, Tx, 
                    z=None, just_dist=False):
        si = self.pre_encoder(set_)
        
        if self.is_attention:
            set_x = set_[:,:,:self.x_dim]
            Q, K = self.fc_qk(Tx), self.fc_qk(set_x)        
            s = self.fc_v(si)
            A = torch.softmax(Q.bmm(K.transpose(1,2))/math.sqrt(self.hid_dim), dim=2)
            lat = A.bmm(s)
        else:
            s = si.mean(dim=1, keepdim=True)
            lat = self.post_encoder(s)
            lat = lat.repeat(1, num_target, 1)
        
        if self.is_stochastic:
            mu, omega = torch.chunk(lat, chunks=2, dim=-1)
            sigma = 0.1+0.9*torch.sigmoid(omega)

            dist = Normal(mu, sigma)

            if just_dist:
                return dist
            else:
                if self.training:
                    z = dist.rsample()
                else:
                    z = dist.mean
                
                entropy = - calc_log_prob(dist, z)
                return dist, z, entropy
        else:
            r = self.LN(lat + self.fc_z(z)) if self.is_multiply else lat
            return r
    
class Decoder(nn.Module):
    def __init__(self, x_dim, hid_dim, r_dim, z_dim, y_dim, 
                 num_layer=3, is_attention=False, is_multiply=False):
        super(Decoder, self).__init__()
        
        self.r_encoder = Encoder(x_dim, hid_dim, r_dim, y_dim,
                                 pre_layer=2 if is_attention else 4,
                                 is_stochastic=False,
                                 is_attention=is_attention,
                                 is_multiply=is_multiply)
        
        if is_multiply:
            dec_inp_dim = x_dim+r_dim
        else:
            dec_inp_dim = x_dim+r_dim+z_dim
        
        self.decoder = MLP(dec_inp_dim, hid_dim, y_dim*2, nn.ReLU(), num_layer)
        
        self.is_multiply = is_multiply
        
    def calc_y(self, C, Tx, z):
        num_target = Tx.size(1)
        r = self.r_encoder.calc_latent(C, num_target, Tx, 
                                       z=z if self.is_multiply else None)
        if self.is_multiply:
            dec_inp = torch.cat([Tx, r], dim=-1)
        else:
            dec_inp = torch.cat([Tx, r, z], dim=-1)
        
        y_param = self.decoder(dec_inp)
        
        mu, omega = torch.chunk(y_param, chunks=2, dim=-1)
        sigma = 0.1+0.9*F.softplus(omega)
        dist = Normal(mu, sigma)
        if self.training:
            pred = dist.rsample()
        else:
            pred = dist.mean    
        return dist, pred