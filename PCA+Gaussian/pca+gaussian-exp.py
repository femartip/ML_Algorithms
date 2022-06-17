#!/usr/bin/python3

#python ./Gaussian_Naive_Bayes/gaussian-exp.py ../train-images-idx3-ubyte.npz ../train-labels-idx1-ubyte.npz "0.9 0.1 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001" 90 10 

from os import XATTR_REPLACE
import sys
import numpy as np
from gaussian import gaussian
from pca import pca 

if len(sys.argv)!=7:
  print('Usage: %s <trdata> <trlabels> <alphas> <ks> <%%trper> <%%dvper>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
alphas=np.fromstring(sys.argv[3],dtype=np.double,sep=' ');
ks= np.fromstring(sys.argv[4],dtype=int,sep=' ')
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

for k in ks:
  print("k = "+str(k))
  XtrP = (Xtr - m) @ W[:,:k]  
  XdvP = (Xdv - m) @ W[:,:k]
  err = gaussian(XtrP, xltr, XdvP, xldv, alphas)
