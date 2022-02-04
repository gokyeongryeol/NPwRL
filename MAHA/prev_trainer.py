import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from prev_algo import NP


def evaluation(model, data_que): 
    index = np.random.randint(1000)
    C, T = data_que[index]
    C, T = C.float().cuda(), T.float().cuda()
    T_MSE, _ = model(C, T, update_=False, return_=True)
    return T_MSE

def train_model(args):
    #set data
    train_data_que = torch.load(f'data/{args.data}_True_False.pt')
    valid_data_que = torch.load(f'data/{args.data}_True_True.pt')
    
    #set model
    model = NP(args.x_dim, args.hid_dim, args.r_dim, args.z_dim, args.y_dim, args.model,
               args.lr, args.clipping)
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
                torch.save(model.state_dict(), f'model/{args.data}_{args.model}_model.pt')
                torch.save(model.loss, f'loss/{args.data}_{args.model}_loss.pt')
                torch.save(MSE_list, f'MSE/{args.data}_{args.model}_MSE.pt')
            
def test_model(args):
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
    model.load_state_dict(torch.load(f'model/{args.data}_{args.model}_model.pt'))
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
    plt.savefig(f'plot/{args.data}_{args.model}_regression.png')
    plt.legend(loc='lower right')
    plt.close()
