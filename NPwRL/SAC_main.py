import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from tqdm import tqdm
from itertools import count

import random
import numpy as np
import argparse
import pickle
import json
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal
import torch.distributions.multivariate_normal as multinorm
import torch.distributions.kl as kl

from algo import NP, SAC

def compute_kernel(x1, x2):
    x1_, x2_ = x1.unsqueeze(dim=2), x2.unsqueeze(dim=1)
    sq_norm = torch.pow(x1_-x2_, exponent=2).sum(dim=-1) 
    kernel = torch.exp(-0.5 * sq_norm)
    if x1.size(1) == x2.size(1):
        kernel += 0.02**2 * torch.eye(x1.size(1)).cuda()
    kernel = kernel.double()
    return kernel

def step(obs, action, model, C, Tx, t):
    mean, stddev = torch.chunk(obs, chunks=2, dim=-1)
    z = mean + action * stddev
    next_obs = torch.cat([z, stddev], dim=-1)
    
    Cx = C[:,:,:args.x_dim]
    z_c = z if 'mul' in args.model else z.repeat(1, Cx.size(1), 1)
    _, Cy_hat = model.decoder.calc_y(C, Cx, z_c)
    
    z_t = z if 'mul' in args.model else z.repeat(1, Tx.size(1), 1)
    Ty_dist, Ty_hat = model.decoder.calc_y(C, Tx, z_t)
    
    ktc, kcc = compute_kernel(Tx, Cx), compute_kernel(Cx, Cx)
    inv_kcc = kcc.inverse()
    Ty_pred = torch.bmm(ktc, torch.bmm(inv_kcc, Cy_hat.double()))
    
    ktt = compute_kernel(Tx, Tx)
    ktct = torch.bmm(ktc, torch.bmm(inv_kcc, ktc.transpose(1,2)))
    cov = ktt - ktct
    var = cov.diagonal(dim1=1, dim2=2).unsqueeze(dim=2)
    var = torch.clamp(var, min=1e-6, max=10)
    dist = normal.Normal(Ty_pred, torch.sqrt(var))
    rew = dist.log_prob(Ty_hat).sum(dim=-1).mean().item() 
    
    #rew = - F.mse_loss(Ty_hat, Ty_pred).item()
    done = (t+1 == args.max_steps)
    info = Ty_dist
    return next_obs, rew, done, info
        
def evaluation(agent, model, data_que): 
    index = np.random.randint(1000)
    C, T = data_que[index]
    C, T = C.float().cuda(), T.float().cuda()
    
    z_dist = model.z_encoder.calc_latent(C, just_dist=True)
    obs = torch.cat([z_dist.mean, z_dist.stddev], dim=-1)
    
    cum_rew = 0.0    
    for t in count():
        _, action, _ = agent.actor.calc_action(obs)
        Tx = T[:,:,:args.x_dim]
        next_obs, rew, done, info = step(obs, action, model, C, Tx, t)
        
        cum_rew += rew
        
        if done:
            Ty_dist = info
            Ty = T[:,:,-args.y_dim:]
            MSE = F.mse_loss(Ty_dist.mean, Ty).item()
            break
        else:
            obs = next_obs
    return cum_rew, MSE

def train_agent():  
    #set data
    train_data_que = torch.load(f'data/{args.data}_True_False.pt')
    valid_data_que = torch.load(f'data/{args.data}_True_True.pt')
    
    #set model
    model = NP(args.x_dim, args.hid_dim, args.r_dim, args.z_dim, args.y_dim, args.model)
    #model.load_state_dict(torch.load(f'model/{args.model}_True_model.pt'))
    model = model.cuda()
    #model.eval()
    
    #set agent
    agent = SAC(args.x_dim, args.y_dim, args.hid_dim, args.act_dim, args.act_limit,
                args.lr, args.clipping, args.delay_cnt, 
                1, args.memory_size,
                args.tunable, args.alpha, args.gamma, args.rho)
    agent = agent.cuda()
    
    #train agent
    steps_done = 0
    cum_rew_list, MSE_list = [], []
    for episode in range(args.n_episodes):
        for index in range(len(train_data_que)):
            agent.train()
            C, T = train_data_que[index]
            C, T = C.float().cuda(), T.float().cuda()
                
            z_dist = model.z_encoder.calc_latent(C, just_dist=True)
            obs = torch.cat([z_dist.mean, z_dist.stddev], dim=-1)
            
            for t in count():
                steps_done += 1

                if steps_done < args.explore_before:
                    action = torch.randn(z_dist.stddev.size()).cuda()
                    action = torch.clip(action, min=-args.act_limit, max=args.act_limit)
                else:
                    _, action, _ = agent.actor.calc_action(obs)
                
                Tx = T[:,:,:args.x_dim]
                next_obs, rew, done, info = step(obs, action, model, C, Tx, t)
                    
                agent.memory.push(obs.detach().to('cpu'), action.detach().to('cpu'), 
                                  next_obs.detach().to('cpu'), rew, done)
                
                if done:
                    if steps_done >= args.update_after:
                        z_prior = z_dist
                        z_q, _ = torch.chunk(next_obs, chunks=2, dim=-1)
                        z_posterior = normal.Normal(z_q, z_dist.stddev)
                        KL = kl.kl_divergence(z_posterior, z_prior).sum(dim=-1).mean()
                        
                        Ty_dist = info
                        Ty = T[:,:,-args.y_dim:]
                        NLL = -Ty_dist.log_prob(Ty).sum(dim=-1).mean()

                        model(C, T, update_=True, return_=False, KL=KL, NLL=NLL)
                        agent.update_param()
                    break
                else:
                    obs = next_obs

            if (index+1) % 1000 == 0:            
                agent.eval()
                cum_rew_sum, MSE_sum = 0.0, 0.0
                for _ in range(5):
                    cum_rew, MSE = evaluation(agent, model, valid_data_que)
                    cum_rew_sum += cum_rew
                    MSE_sum += MSE
                cum_rew_avg = cum_rew_sum / 5
                MSE_avg = MSE_sum / 5
                
                cum_rew_list.append(cum_rew_avg)
                MSE_list.append(MSE_avg)

                print('Episode: {} \t Steps : {} \t Cum_rew: {} \t MSE: {}'.format(episode, steps_done, cum_rew_avg, MSE_avg))
                torch.save(agent.state_dict(), f'agent/{args.data}_{args.agent}_{args.model}_agent.pt')
                torch.save(agent.loss, f'loss/{args.data}_{args.agent}_{args.model}_loss.pt')
                torch.save(cum_rew_list, f'cum_rew/{args.data}_{args.agent}_{args.model}_cum_rew.pt')
                torch.save(MSE_list, f'MSE/{args.data}_{args.agent}_{args.model}_MSE.pt')

    
def test_agent():
    #set data
    test_data_que = torch.load(f'data/{args.data}_False_True.pt')
    C, T = test_data_que[0]
    C, T = C.float().cuda(), T.float().cuda()

    Cx, Cy = C[:,:,:args.x_dim], C[:,:,-args.y_dim:]
    Tx, Ty = T[:,:,:args.x_dim], T[:,:,-args.y_dim:]
    
    #set model
    model = NP(args.x_dim, args.hid_dim, args.r_dim, args.z_dim, args.y_dim, args.model)
    model.load_state_dict(torch.load(f'model/{args.data}_{args.model}_True_model.pt'))
    model = model.cuda()
    model.eval()

    #set agent
    agent = SAC(args.x_dim, args.y_dim, args.hid_dim, args.act_dim, args.act_limit,
                args.lr, args.clipping, args.delay_cnt, 
                1, args.memory_size,
                args.tunable, args.alpha, args.gamma, args.rho)
    agent.load_state_dict(torch.load(f'agent/{args.data}_{args.agent}_{args.model}_agent.pt'))
    agent = agent.cuda()
    agent.eval()

    z_dist = model.z_encoder.calc_latent(C, just_dist=True)
    obs = torch.cat([z_dist.mean, z_dist.stddev], dim=-1)
    
    for t in count():
        _, action, _ = agent.actor.calc_action(obs)
        next_obs, rew, done, info = step(obs, action, model, C, Tx, t)

        if done:
            Ty_dist = info
            MSE = F.mse_loss(Ty_dist.mean, Ty).item()
            break
        else:
            obs = next_obs

    print(MSE)

    Cx, Cy = Cx.squeeze().cpu(), Cy.squeeze().cpu()
    Tx, Ty = Tx.squeeze().cpu(), Ty.squeeze().cpu()    
    pred = Ty_dist.mean.detach().squeeze().cpu()
    std = Ty_dist.stddev.detach().squeeze().cpu()

    plt.figure()
    plt.title(f'{args.model} regression')
    plt.plot(Tx, Ty, 'k:', label='True')
    plt.plot(Cx, Cy, 'k^', markersize=10, label='Contexts')
    plt.plot(Tx, pred, 'b', label='Predictions')
    plt.fill(torch.cat([Tx, torch.flip(Tx, dims=[0])], dim=0),
             torch.cat([pred - 1.96 * std, torch.flip(pred + 1.96 * std, dims=[0])], dim=0),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.savefig(f'plot/{args.data}_{args.agent}_{args.model}_regression.png')
    plt.legend(loc='lower right')
    plt.close()

        
if __name__ == '__main__':
        
    os.makedirs('config', exist_ok=True)
    os.makedirs('agent', exist_ok=True)
    os.makedirs('loss', exist_ok=True)
    os.makedirs('cum_rew', exist_ok=True)
    os.makedirs('MSE', exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Implementing Tacking Meta-learning via reInforcement learning")
    
    parser.add_argument("--data", default='GP', type=str, help="task generating function")
    parser.add_argument("--agent", default='SAC', type=str, help="off-policy actor critic algorithm")
    parser.add_argument("--model", default='NP', type=str, help="Name of Neural Processes variant")
    
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--memory_size", default=1000, type=int, help="memory size a.k.a capcity")
    
    parser.add_argument("--n_episodes", default=5, type=int, help="number of episodes")
    parser.add_argument("--explore_before", default=10000, type=int, help="number of random actions in the beginning")
    parser.add_argument("--update_after", default=10000, type=int, help="number of steps to start parameter update")
    parser.add_argument("--max_steps", default=5, type=int, help="number of maximum steps within episode")
    
    parser.add_argument("--x_dim", default=1, type=int, help='number of input units')
    parser.add_argument("--hid_dim", default=128, type=int, help='number of hidden units')
    parser.add_argument("--r_dim", default=128, type=int, help='number of r units')
    parser.add_argument("--z_dim", default=128, type=int, help='number of z units')
    parser.add_argument("--y_dim", default=1, type=int, help='number of hidden units')
    parser.add_argument("--act_dim", default=128, type=int, help='number of action units')
    parser.add_argument("--act_limit", default=2.0, type=float, help='limit of action')
    
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--clipping", action='store_true', default=False, help="whether to clip to unit norm for gradient") 
    parser.add_argument("--delay_cnt", default=1, type=int, help="number of critic updates between actor(and etc.) updates")
    
    parser.add_argument("--tunable", action='store_false', default=True, help='whether to automatically tune alpha')
    parser.add_argument("--alpha", default=1.0, type=float, help='entropy coefficient')
    parser.add_argument("--gamma", default=0.99, type=float, help='discount factor')
    parser.add_argument("--rho", default=0.995, type=float, help='polyak averaging constant for target network')
    
    args = parser.parse_args()
    
    with open(f'config/{args.agent}_{args.model}.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    train_agent()
    with torch.no_grad():
        test_agent()
    
    