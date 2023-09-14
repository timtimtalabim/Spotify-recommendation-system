"""
Command line arguments for training
@author Bryce Forrest
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import torch

import mlp
import gmf
import ncf
import train

def main():
    # all the arguments your heart desires
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument("-e", '--epochs', type=int)
    parser.add_argument("-lr", '--learning_rate', type=float)
    parser.add_argument("-bs", '--batch_size', type=int)
    parser.add_argument("-pt", '--pretrain', nargs='*', type=str)
    parser.add_argument("-eo", '--epoch_offset', type=int, default=0)
    parser.add_argument("-a", '--alpha', type=float)
    
    args = parser.parse_args()
    c, sparse_matrix = pickle.load(open('user_item_matrix.p', 'rb'))
    M=sparse_matrix.tocsr()
    model_dict = {'mlp':mlp.MLP, 'gmf':gmf.GMF, 'ncf':ncf.NCF}
    
    if args.pretrain is not None:
        if args.model == 'ncf' and len(args.pretrain) == 2:
            _gmf=gmf.GMF(M.shape[1], M.shape[0])
            _gmf.load_state_dict(torch.load(args.pretrain[0]))
            _mlp=mlp.MLP(M.shape[1], M.shape[0])
            _mlp.load_state_dict(torch.load(args.pretrain[1]))

            model = model_dict[args.model](M.shape[1], M.shape[0],_gmf,_mlp)


        elif len(args.pretrain) == 1:
            model = model_dict[args.model](M.shape[1], M.shape[0])
            model.load_state_dict(torch.load(args.pretrain[0]))

    else:
        if args.model == 'ncf' and args.alpha is not None:
            model = model_dict[args.model](M.shape[1], M.shape[0], alpha=args.alpha)
        else:
            model = model_dict[args.model](M.shape[1], M.shape[0])
        
    train.train(M, model, args.batch_size, args.epochs, args.learning_rate, epoch_offset=args.epoch_offset)

if __name__ == '__main__':main()