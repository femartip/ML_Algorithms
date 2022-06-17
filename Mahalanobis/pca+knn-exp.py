#!/usr/bin/python3

from os import XATTR_REPLACE
import sys
import math
import numpy as np
from pca import pca
from knn import knnMD

if len(sys.argv)!=7:
  print('Usage: %s <trdata> <trlabels> <PCA> <alphas> <%%trper> <%%dvper>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
PCA=np.fromstring(sys.argv[3],dtype=int,sep=' ');
alphas=np.fromstring(sys.argv[4],dtype=float,sep=' ');
trper=int(sys.argv[5]);
dvper=int(sys.argv[6]);

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
for a in alphas:
  for dim in PCA:
    XtrP = (Xtr - m) @ W[:,:dim]  
    XdvP = (Xdv - m) @ W[:,:dim]
    err = knnMD(XtrP, xltr, XdvP, xldv, 1, a)
    print(a,dim,err)
    