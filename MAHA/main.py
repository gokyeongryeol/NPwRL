import os
import argparse
import json


parser = argparse.ArgumentParser(description="1D regression with Neural Processes as context-based meta-learning")
    
parser.add_argument("--gpu_ids", default=0, type=int, help="assigned gpu ids")
parser.add_argument("--data", default='mixture', type=str, help="task generating function")
parser.add_argument("--model", default='NP', type=str, help="name of Neural Processes variant: NP, ANP, mulNP, MAHA")
parser.add_argument("--phase", default='pretrain', type=str, help="whether to pretrain or finetune in case of MAHA")

parser.add_argument("--n_epochs", default=100, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=100, type=int, help="batch size")

parser.add_argument("--x_dim", default=1, type=int, help='number of input units')
parser.add_argument("--hid_dim", default=128, type=int, help='number of hidden units')
parser.add_argument("--r_dim", default=128, type=int, help='number of r units')
parser.add_argument("--z_dim", default=128, type=int, help='number of z units')
parser.add_argument("--y_dim", default=1, type=int, help='number of hidden units')

parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
parser.add_argument("--clipping", action='store_true', default=False, help="whether to clip to unit norm for gradient") 

args = parser.parse_args()

with open(f'config/{args.data}_{args.model}.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    
    
os.makedirs('config', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('MSE', exist_ok=True)
os.makedirs('loss', exist_ok=True)
os.makedirs('plot', exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_ids)

if __name__ == '__main__':
    from trainer import train_model, test_model
    
    train_model(args)
    test_model(args)
    