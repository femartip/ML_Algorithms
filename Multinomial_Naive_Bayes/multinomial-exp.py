#!/usr/bin/python3

#python ./Multinomial_Naive_Bayes/multinomial-exp.py ../train-images-idx3-ubyte.npz ../train-labels-idx1-ubyte.npz "0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001 0.0000000001" 90 10

from os import XATTR_REPLACE
import sys
import numpy as np
from multinomial import multinomial

if len(sys.argv)!=6:
  print('Usage: %s <trdata> <trlabels> <es> <%%trper> <%%dvper>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
es=np.fromstring(sys.argv[3],dtype=np.double,sep=' ');
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

for e in es:
  err = multinomial(Xtr, xltr, Xdv, xldv, e)
  print(e,err)