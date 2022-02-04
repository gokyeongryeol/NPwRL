import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal

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
        dist = normal.Normal(mean, std)
        return dist

       
class Encoder(nn.Module):
    def __init__(self, x_dim, hid_dim, lat_dim, y_dim, 
                 pre_layer=4, post_layer=2, 
                 is_stochastic=False, is_attention=False, is_multiply=False):
        super(Encoder, self).__init__()
        
        self.x_dim, self.hid_dim, self.y_dim = x_dim, hid_dim, y_dim
        
        if is_attention:
            self.fc_qk = MLP(x_dim, hid_dim, hid_dim, nn.ReLU(), 2)
            self.fc_v = MLP(hid_dim, hid_dim, hid_dim, nn.ReLU(), 2)
        else:
            self.post_encoder = MLP(hid_dim, hid_dim, lat_dim*2 if is_stochastic else lat_dim,
                                    nn.ReLU(), post_layer)

        
        if is_multiply:
            self.S = nn.Parameter(torch.Tensor(1, lat_dim, hid_dim))
            nn.init.xavier_uniform_(self.S)
        
        self.pre_encoder = MLP(x_dim+y_dim, hid_dim, hid_dim, nn.ReLU(), pre_layer)
        
        self.is_stochastic = is_stochastic
        self.is_attention = is_attention
        self.is_multiply = is_multiply
        
    def calc_latent(self, set_, num_target=None, Tx=None, z=None, just_dist=False):
        si = self.pre_encoder(set_)
            
        if not self.is_stochastic:
            if self.is_attention: #ANP r
                set_x = set_[:,:,:self.x_dim]
                Q, K = self.fc_qk(Tx), self.fc_qk(set_x)        
                s = self.fc_v(si)
                s = s.bmm(z.transpose(1,2)) if self.is_multiply else s
                A = torch.softmax(Q.bmm(K.transpose(1,2))/math.sqrt(self.hid_dim), dim=2)
                r = A.bmm(s)
            else: # CNP/NP r
                s = si.mean(dim=1, keepdim=True)
                r = self.post_encoder(s)
                r = r.repeat(1, num_target, 1)
            return r
        else:
            if self.is_multiply: #LD
                Q = self.S.repeat(si.size(0), 1, 1)
                A = torch.softmax(Q.bmm(si.transpose(1,2))/math.sqrt(self.hid_dim), 2)
                z_param = self.post_encoder(A.bmm(si))
            else: # NP/ANP z
                s = si.mean(dim=1, keepdim=True)
                z_param = self.post_encoder(s)
                
            mu, omega = torch.chunk(z_param, chunks=2, dim=-1)
            sigma = 0.1+0.9*torch.sigmoid(omega)
            
            dist = normal.Normal(mu, sigma)

            if just_dist:
                return dist
            else:
                if self.training:
                    z = dist.rsample()
                else:
                    z = dist.mean

                entropy = - calc_log_prob(dist, z)
                
                if not self.is_multiply:
                    z = z.repeat(1, num_target, 1)
                return dist, z, entropy

    
class Decoder(nn.Module):
    def __init__(self, x_dim, hid_dim, r_dim, z_dim, y_dim, 
                 num_layer=3, is_attention=False, is_multiply=False):
        super(Decoder, self).__init__()
        
        if is_multiply:
            dec_inp_dim = x_dim+r_dim
        else:
            dec_inp_dim = x_dim+r_dim+z_dim
            
        self.r_encoder = Encoder(x_dim, hid_dim, r_dim, y_dim,
                                 pre_layer=2 if is_attention else 4,
                                 is_attention=is_attention)
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
        dist = normal.Normal(mu, sigma)
        if self.training:
            pred = dist.rsample()
        else:
            pred = dist.mean    
        return dist, pred
    

class Actor(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim, act_limit, 
                 num_layer=2, min_log=-20, max_log=2, is_stochastic=True):
        super(Actor, self).__init__()
        
        self.act_limit = act_limit
        self.is_stochastic = is_stochastic
        self.actor = MLP(obs_dim, hid_dim, act_dim*2 if is_stochastic else act_dim,
                         nn.ReLU(), num_layer)
        
        self.min_log, self.max_log = min_log, max_log
        
    def calc_action(self, obs, *args):
        if self.is_stochastic: #SAC
            param = self.actor(obs)
            mu, log_sig = torch.chunk(param, chunks=2, dim=-1)
            log_sig = torch.clamp(log_sig, self.min_log, self.max_log)     
            sigma = torch.exp(log_sig) + 1e-6
            
            dist = normal.Normal(mu, sigma)
            if self.training:
                unsquashed = dist.rsample()
            else:
                unsquashed = dist.mean
            squashed = torch.tanh(unsquashed)
            action_hat = squashed * self.act_limit
            entropy = - calc_log_prob(dist, squashed, form='-11')
            return dist, action_hat, entropy
            
        else: #TD3
            dist = None
            unsquashed = self.actor(obs)
            if self.training:
                if 'explore' in args:
                    noise = 0.1 * torch.randn(unsquashed.size()).cuda()
                elif 'smoothing' in args:
                    noise = 0.2 * torch.randn(unsquashed.size()).cuda()
                    noise = torch.clamp(noise, -0.5, 0.5)
                else:
                    noise = 0.0
            else:
                noise = 0.0
            unsquashed = unsquashed + noise
            action_hat = torch.clamp(unsquashed, -self.act_limit, self.act_limit)
            entropy = 0.0
            return dist, action_hat, entropy
        

class Critic(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim, 
                 num_layer=2):
        super(Critic, self).__init__()
        
        self.critic = MLP(obs_dim+act_dim, hid_dim, 1, nn.ReLU(), num_layer)
        
    def calc_q_value(self, obs, action):
        concatted = torch.cat([obs, action], dim=-1)
        q_val = self.critic(concatted)
        return q_val
