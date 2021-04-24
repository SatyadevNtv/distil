from .strategy import Strategy
import numpy as np

import torch
from torch import nn
import random
import math

from scipy import stats


def init_centers(X, K, device):
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    #gram = np.matmul(X[indsAll], X[indsAll].T)
    #val, _ = np.linalg.eig(gram)
    #val = np.abs(val)
    #vgt = val[val > 1e-2]
    return indsAll

class BADGE(Strategy):
    """
    This method is based on the paper Deep Batch Active Learning by Diverse, Uncertain Gradient 
    Lower Bounds :footcite:`DBLP-Badge`. According to the paper, this strategy, Batch Active 
    learning by Diverse Gradient Embeddings (BADGE), samples groups of points that are disparate 
    and high magnitude when represented in a hallucinated gradient space, a strategy designed to 
    incorporate both predictive uncertainty and sample diversity into every selected batch. 
    Crucially, BADGE trades off between uncertainty and diversity without requiring any hand-tuned 
    hyperparameters. Here at each round of selection, loss gradients are computed using the 
    hypothesised labels. Then to select the points to be labeled are selected by applying 
    k-means++ on these loss gradients. 
    
    Parameters.
    ----------
    X: Numpy array 
        Features of the labled set of points 
    Y: Numpy array
        Lables of the labled set of points 
    unlabeled_x: Numpy array
        Features of the unlabled set of points 
    net: class object
        Model architecture used for training. Could be instance of models defined in `distil.utils.models` or something similar.
    handler: class object
        It should be a subclasses of torch.utils.data.Dataset i.e, have __getitem__ and __len__ methods implemented, so that is could be passed to pytorch DataLoader.Could be instance of handlers defined in `distil.utils.DataHandler` or something similar.
    nclasses: int 
        No. of classes in tha dataset
    args: dictionary
        This dictionary should have 'batch_size' as a key. 
    """

    def __init__(self, X, Y, unlabeled_x, net, handler,nclasses, args):

        super(BADGE, self).__init__(X, Y, unlabeled_x, net, handler,nclasses, args)

    def select(self, budget, grad_batch_size=None):
        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
            
        grad_batch_size: int
            If specified, uses a per-batch approximation that instead uses the average of the specified number of gradient embeddings. Useful in reducing the size of the full gradient embedding tensor.
        
        Returns
        ----------
        chosen: list
            List of selected data point indexes with respect to unlabeled_x
        """ 

        if grad_batch_size is not None:
            gradEmbedding, batch_idxs = self.get_grad_embedding(self.unlabeled_x,bias_grad=False,batch_size=grad_batch_size)
            chosen_batch = init_centers(gradEmbedding.cpu().numpy(), budget, self.device)
            
            # Get actual list of indices by taking apart each batch
            chosen = list()
            
            for expand_batch_idx in chosen_batch:
                indices_to_add = batch_idxs[expand_batch_idx]
                chosen.extend(indices_to_add)
                
            return chosen
        else:
            gradEmbedding = self.get_grad_embedding(self.unlabeled_x,bias_grad=False)
            chosen = init_centers(gradEmbedding.cpu().numpy(), budget, self.device)
        return chosen