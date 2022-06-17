#!/usr/bin/python3

# python ./PCA+KNN/pca+knn-exp.py ../train-images-idx3-ubyte.npz ../train-labels-idx1-ubyte.npz "50 60.." 90 10

from os import XATTR_REPLACE
import sys
import math
import numpy as np
from pca import pca
from knn import knn
from knn_batch import knnB

if len(sys.argv)!=6:
  print('Usage: %s <trdata> <trlabels> <ks> <%%trper> <%%dvper>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
ks=np.fromstring(sys.argv[3],dtype=int,sep=' ');
trper=int(sys.argv[4]);
dvper=int(sys.argv[5]);

N=X.shape[0];
np.random.seed(23); perm=np.random.permutation(N);
X=X[perm]; xl=xl[perm];

# Selecting a subset for train and dev sets
Ntr=round(trper/100*N);
Xtr=X[:Ntr]; xltr=xl[:Ntr];
Ndv=round(dvper/100*N);
Xdv=X[N-Ndv:]; xldv=xl[N-Ndv:];

#
# HERE YOUR CODE
#



m, W = pca(Xtr)
for k in ks:
  XtrP = (Xtr - m) @ W[:,:k]  
  XdvP = (Xdv - m) @ W[:,:k]
  #err = knn(XtrP, xltr, XdvP, xldv, 1)
  err = knn(XtrP, xltr, XdvP, xldv, 1)
  print(k,err)