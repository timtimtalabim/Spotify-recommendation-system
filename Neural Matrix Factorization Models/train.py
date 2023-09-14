"""
Training function and dataloader for NMF, GMF, MLP
@author Bryce Forrest
"""

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sklearn.utils
from tqdm import tqdm
from matplotlib import pyplot as plt
from itertools import groupby
import sklearn.utils
import numpy as np
import pandas as pd
import math

class Loader:
    """
    Dataloader for training.
    Takes in (row,col) coordinates, and optional label, y. Iterates through and returns batches.
    Can shuffle, and force stop after certain number of iterations.
    """
    def __init__(self, row, col, y=None, batch_size=None,
                 shuffle=False, max_iteration=None):
        self.row=row
        self.col=col
        self.y=y
        self.bs=len(row) if batch_size is None else batch_size
        self.shuf=shuffle
        
        if max_iteration is None or max_iteration > len(self.row):
            self.end = len(self.row)
        else:
            self.end = max_iteration
                             
        if self.shuf:
           self.__shuffle__()   
        
        self.i=0
        self.j=0

    def __len__(self):
        return len(self.row)
            
    def __shuffle__(self):
        if self.y is None:
            self.row, self.col = sklearn.utils.shuffle(self.row, self.col)
        else:
            self.row, self.col, self.y = sklearn.utils.shuffle(
                self.row, self.col, self.y
            )
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.j=self.i
        self.i += self.bs
        if self.i <= self.end:
            if self.y is None:
                return self.row[self.j:self.i],self.col[self.j:self.i]
            else:
                return self.row[self.j:self.i], \
                       self.col[self.j:self.i], \
                       self.y[self.j:self.i]
        
        self.i = self.j = 0
        self.__shuffle__()
        raise StopIteration
    
    

def train(M, model, batch_size=256, epochs=100, learning_rate=0.001, epoch_offset=0):
    torch.manual_seed(0) # make sure we are training the same model

    # get non-zero values, and take a random sample of one positive interaction per user
    holdout=set(next(g) for _, g in groupby(zip(*M.nonzero()), key=lambda x:x[0]))
    coords=set(zip(*M.nonzero()))
    # remove sample from set of all positive interactions
    unique = coords-holdout
    holdout_row, holdout_col = np.array(list(map(np.array, holdout))).T
    
    # make dataset of positive interactions
    row_nz, col_nz = np.array(list(map(np.array, unique))).T
    
    # randomly generate coordinates, which will MOST LIKELY be zero values, since
    # matrix is 99% sparse. 4 negative per 1 positive interaction
    zero_coords = set(zip(
                    np.random.randint(M.shape[0], size=len(row_nz)*4),
                    np.random.randint(M.shape[1], size=len(col_nz)*4))
                )
    
    # make sure we only have zero values by removing positives
    unique = zero_coords-coords
    # make dataset of negative interactions
    row_z, col_z = np.array(list(map(np.array, unique))).T
    
    # generate labels
    y = [1]*len(row_nz) + [0]*len(row_z)
    
    # concatenate and shuffle everything
    row, col, y = sklearn.utils.shuffle(np.concatenate((row_nz, row_z)), \
                                     np.concatenate((col_nz, col_z)), y)
    
    # test train split
    train_row, test_row, train_col, test_col, train_y, test_y = train_test_split(
        row,col,y, test_size=0.2, random_state=0
    )

    trainloader = Loader(
        train_row, train_col, train_y, 
        batch_size=batch_size, shuffle=True, max_iteration=None
    )
    
    testloader = Loader(
        test_row, test_col, test_y,
        batch_size=None, shuffle=False, max_iteration=None
    )
    
    validationloader = Loader(
        holdout_row, holdout_col, batch_size=1, shuffle=False, max_iteration=None
    )
    
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    ave_loss_lst_train = []
    ave_loss_lst_test = []
    hr_lst = []
    ndcg_lst = []
    
    M=M.todense()
    
    # remove holdouts from matrix
    for i,j in holdout:
        M[i,j]=0
    
    # begin training
    with tqdm(range(epoch_offset, epoch_offset+epochs),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as pbar:
        for epoch in pbar:
            pbar.set_description("Epoch " + str(epoch))
            
            # save model every 5 epoch
            if epoch%5==0:
                torch.save(
                    model.state_dict(),
                    '{}_{}_{:.0e}.pt'.format(model.name, epoch, learning_rate)
                )
            
            train_ave_loss = []
            test_ave_loss = []
            
            acc_ave=0
            acc=0
            ndcg = []
            hits=[]

            # get some (probably) negative values to rate holdout against
            negative=M.T[np.random.choice(M.shape[1], 100, replace=False)]
            model.eval()
            for u,v in validationloader:
                pred = model(
                        torch.tensor(M[u], dtype=torch.float32),
                        torch.tensor(np.concatenate(
                            (negative, M[:,v].T)),dtype=torch.float32
                        )
                )
                
                # evaluate results
                topK = ((-pred.flatten())).argsort()[:10]
                if ((len(pred)-1) in topK):
                    hits.append(1)
                    _i = topK.tolist().index((len(pred)-1))
                    ndcg.append(math.log(2)/math.log(_i+2))
                else:
                    hits.append(0)
                    ndcg.append(0)

            ndcg_ave=np.array(ndcg).mean()
            hr=np.array(hits).mean()
            hr_lst.append(hr)
            ndcg_lst.append(ndcg_ave)

            # test performance
            for u, v, y in testloader:
                pred = model(
                    torch.tensor(M[u, :], dtype=torch.float32),
                    torch.tensor(M[:, v].T, dtype=torch.float32),
                )

                loss = loss_func(
                    pred.squeeze(), torch.tensor(y, dtype=torch.float32).squeeze()
                )

                test_ave_loss.append(loss.detach().item())
            
            
            # back to training
            model.train()
            for u, v, y in trainloader:
                pred = model(
                    torch.tensor(M[u, :], dtype=torch.float32),
                    torch.tensor(M[:, v].T, dtype=torch.float32),
                )

                loss = loss_func(
                    pred.squeeze(), torch.tensor(y, dtype=torch.float32).squeeze()
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_ave_loss.append(loss.detach().item())

            train_loss_ave = np.array(train_ave_loss).mean()
            ave_loss_lst_train.append(train_loss_ave)
            test_loss_ave = np.array(test_ave_loss).mean()
            ave_loss_lst_test.append(test_loss_ave)

            pbar.set_postfix(
                train_loss=train_loss_ave, test_loss=test_loss_ave, 
                hr=hr, ndcg=ndcg_ave
            )

    # print results to csv for later analysis
    pd.DataFrame([ave_loss_lst_train, ave_loss_lst_test, hr_lst, ndcg_lst]).to_csv(
        '{}_{}_{:.0e}.csv'.format(model.name, batch_size, learning_rate)
    )      
    
    # CODE BELOW USED TO GENERATE PLOTS

    # print('Maximum accuracy of {:.2f}, achieved at epoch {}'.format(max_acc[0],max_acc[1]))
    
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.set_title("Average Training Loss")
    # ax1.plot(range(len(ave_loss_lst_train)), ave_loss_lst_train, color='green')
    # # ax1.plot(range(len(ave_loss_lst_test)), ave_loss_lst_test, color='lime')
    # # ax1.legend()
    # ax2.set_title("Accuracy")
    # ax2.plot(range(len(ave_acc_lst)), ave_acc_lst, color='green')
    # fig.tight_layout(pad=1.0)
    # plt.show()  
