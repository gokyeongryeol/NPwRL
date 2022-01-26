import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as normal
import torch.distributions.kl as kl

import math
import numpy as np

from utils import hard_update, soft_update, convert
from module import MLP, Prior, Encoder, Decoder, Actor, Critic
from memory import Memory


class NP(nn.Module):
    def __init__(self, x_dim, hid_dim, r_dim, z_dim, y_dim, model,
                 lr=5e-4, annealing_period=1000, clipping=False):
        super(NP, self).__init__()
        
        self.x_dim, self.y_dim = x_dim, y_dim
        self.model = model

        self.z_encoder = Encoder(x_dim, hid_dim, z_dim, y_dim, 
                                 is_stochastic=True, is_multiply='mul' in self.model)
        
        self.decoder = Decoder(x_dim, hid_dim, r_dim, z_dim, y_dim,
                               is_multiply='mul' in self.model)
        
        self.NP_optim = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.NP_optim, T_max=annealing_period)
        self.loss = {'KL':[], 'NLL':[]}
        self.clipping = clipping
        
    def forward(self, C, T, update_, return_, KL=None, NLL=None):
        if KL is None and NLL is None:
            Tx, Ty = T[:,:,:self.x_dim], T[:,:,-self.y_dim:]
            num_target = Tx.size(1)

            z_prior, z_p, _ = self.z_encoder.calc_latent(C, num_target)
            z_posterior, z_q, _ = self.z_encoder.calc_latent(T, num_target)

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
            self.scheduler.step()
            
            self.loss['KL'].append(KL.item())
            self.loss['NLL'].append(NLL.item())
        
        if return_:
            T_MSE = F.mse_loss(Ty, Ty_hat).item()
            return T_MSE, Ty_dist

        
class SAC(nn.Module):
    upd_cnt = 0
    def __init__(self, x_dim, y_dim, hid_dim, act_dim, act_limit,
                 lr, clipping, delay_cnt, 
                 batch_size, memory_size, 
                 tunable, alpha, gamma, rho):
        super(SAC, self).__init__()

        self.batch_size = batch_size
        self.memory = Memory(memory_size)
        
        self.critic_1 = Critic(act_dim*2, hid_dim, act_dim)
        self.critic_1_prime = Critic(act_dim*2, hid_dim, act_dim)
        self.critic_2 = Critic(act_dim*2, hid_dim, act_dim)
        self.critic_2_prime = Critic(act_dim*2, hid_dim, act_dim)
        critic_param_list = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.critic_optim = optim.Adam(critic_param_list, lr=lr)
        
        self.critic_list = [self.critic_1, self.critic_2]
        self.critic_prime_list = [self.critic_1_prime, self.critic_2_prime]
        hard_update(self.critic_prime_list, self.critic_list)
        
        self.actor = Actor(act_dim*2, hid_dim, act_dim, act_limit)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr)
        
        self.clipping = clipping
        self.delay_cnt = delay_cnt
        
        self.loss = {'critic':[], 'actor':[]}
        
        if tunable:
            self.log_alpha = nn.Parameter(torch.tensor([math.log(alpha + 1e-6)]))    
            self.alpha_optim = optim.Adam([self.log_alpha], lr)
            self.target_entropy = (-1) * act_dim
            self.loss['alpha'] = []
            
        self.tunable = tunable
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
            
    def _min_double_q_trick(self, obs, action_hat, critic_list):
        q_value_list = []
        for critic in critic_list:
            q_value = critic.calc_q_value(obs, action_hat)
            q_value_list.append(q_value)
        min_q_value = torch.min(torch.cat(q_value_list, dim=-1), dim=-1)[0]
        return min_q_value

    def calc_target(self, next_obs, rew, done):
        _, next_action_hat, next_entropy = self.actor.calc_action(next_obs, 'smoothing')
        min_q_prime = self._min_double_q_trick(next_obs, next_action_hat, self.critic_prime_list)
        target = rew + self.gamma * (1-done) * (min_q_prime + self.alpha * next_entropy)
        return target
    
    def _critic_update(self, obs, action, next_obs, rew, done):
        with torch.no_grad():
            target = self.calc_target(next_obs, rew, done)
        
        critic_loss_list = []
        for critic in self.critic_list:
            q_value = critic.calc_q_value(obs, action)
            loss_c = torch.pow(q_value.squeeze(dim=-1) - target, 2)
            critic_loss_list.append(loss_c)
        
        critic_loss = torch.stack(critic_loss_list, dim=-1).mean()
        self.loss['critic'].append(critic_loss.item())
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.clipping:
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        self.critic_optim.step()
        
    def _actor_update(self, obs, action):
        dist, action_hat, entropy = self.actor.calc_action(obs)
        
        min_q_value = self._min_double_q_trick(obs, action_hat, self.critic_list)
        actor_loss = - min_q_value - self.alpha * entropy
        actor_loss = actor_loss.mean()
        self.loss['actor'].append(actor_loss.item())
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.clipping:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()
        
        if self.tunable:
            with torch.no_grad():
                _, _, entropy = self.actor.calc_action(obs)
                    
            alpha_loss = self.log_alpha * (entropy - self.target_entropy)
            alpha_loss = alpha_loss.mean()
            self.loss['alpha'].append(alpha_loss.item())
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            if self.clipping:
                nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optim.step()

            self.alpha = torch.exp(self.log_alpha).item()
            
    def update_param(self): 
        batch = self.memory.sample(self.batch_size)
        obs, action, next_obs, rew, done = convert(batch)
        
        self._critic_update(obs, action, next_obs, rew, done)

        if self.upd_cnt % self.delay_cnt == 0: 
            self._actor_update(obs, action)
            soft_update(self.critic_prime_list, self.critic_list, self.rho)

        self.upd_cnt += 1