import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from tqdm import tqdm

import random
import numpy as np
import argparse
import pickle
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from algo import NP

def evaluation(model, data_que): 
    index = np.random.randint(1000)
    C, T = data_que[index]
    C, T = C.float().cuda(), T.float().cuda()
    T_MSE, _ = model(C, T, update_=False, return_=True)
    return T_MSE

def train_model():
    #set data
    train_data_que = torch.load(f'data/{args.data}_True_False.pt')
    valid_data_que = torch.load(f'data/{args.data}_True_True.pt')
    
    #set model
    model = NP(args.x_dim, args.hid_dim, args.r_dim, args.z_dim, args.y_dim, args.model,
               args.lr, args.annealing_period, args.clipping)
    if args.is_pretrain:
        pass
    else:
        model.load_state_dict(torch.load(f'model/{args.data}_{args.model}_True_model.pt'))
    model = model.cuda()

    MSE_list = []
    for epoch in range(args.n_epochs):
        for index in range(len(train_data_que)):
            model.train()
            C, T = train_data_que[index]
            C, T = C.float().cuda(), T.float().cuda()
            model(C, T, update_=True, return_=False)

            if (index+1) % 1000 == 0:            
                with torch.no_grad():
                    model.eval()
                    MSE = evaluation(model, valid_data_que)
                    MSE_list.append(MSE)

                print('Epoch: {} \t Index: {} \t MSE : {}'.format(epoch, index, MSE))
                torch.save(model.state_dict(), f'model/{args.data}_{args.model}_{args.is_pretrain}_model.pt')
                torch.save(model.loss, f'loss/{args.data}_{args.model}_{args.is_pretrain}_loss.pt')
                torch.save(MSE_list, f'MSE/{args.data}_{args.model}_{args.is_pretrain}_MSE.pt')
            
def test_model():
    #set data
    test_data_que = torch.load(f'data/{args.data}_False_True.pt')
    C, T = test_data_que[0]
    C, T = C.float().cuda(), T.float().cuda()

    Cx, Cy = C[:,:,:args.x_dim], C[:,:,-args.y_dim:]
    Tx, Ty = T[:,:,:args.x_dim], T[:,:,-args.y_dim:]
    Cx, Cy = Cx.squeeze().cpu(), Cy.squeeze().cpu()
    Tx, Ty = Tx.squeeze().cpu(), Ty.squeeze().cpu()
        
    #set model
    model = NP(args.x_dim, args.hid_dim, args.r_dim, args.z_dim, args.y_dim, args.model)
    model.load_state_dict(torch.load(f'model/{args.data}_{args.model}_{args.is_pretrain}_model.pt'))
    model = model.cuda()
    model.eval()

    T_MSE, Ty_dist = model(C, T, update_=False, return_=True)
    print(T_MSE)

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
    plt.savefig(f'plot/{args.data}_{args.model}_{args.is_pretrain}_regression.png')
    plt.legend(loc='lower right')
    plt.close()

if __name__ == '__main__':
        
    os.makedirs('config', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('MSE', exist_ok=True)
    os.makedirs('loss', exist_ok=True)
    
    parser = argparse.ArgumentParser(description="1D regression with Neural Processes as context-based meta-learning")
    
    parser.add_argument("--data", default='mixture', type=str, help="task generating function")
    parser.add_argument("--model", default='NP', type=str, help="Name of Neural Processes variant")
    parser.add_argument("--is_pretrain", action='store_false', default=True, help="whether to pretrain") 
    
    parser.add_argument("--n_epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--annealing_period", default=1000, type=int, help="cosine annealing period")
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    
    parser.add_argument("--x_dim", default=1, type=int, help='number of input units')
    parser.add_argument("--hid_dim", default=128, type=int, help='number of hidden units')
    parser.add_argument("--r_dim", default=128, type=int, help='number of r units')
    parser.add_argument("--z_dim", default=128, type=int, help='number of z units')
    parser.add_argument("--y_dim", default=1, type=int, help='number of hidden units')
    
    parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
    parser.add_argument("--clipping", action='store_true', default=False, help="whether to clip to unit norm for gradient") 
    
    args = parser.parse_args()
    
    with open(f'config/{args.data}_{args.model}_{args.is_pretrain}.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    train_model()
    test_model()
    