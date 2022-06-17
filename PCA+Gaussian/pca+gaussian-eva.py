#!/usr/bin/python3

#python ./PCA+Gaussian/pca+gaussian-eva.py ../train-images-idx3-ubyte.npz ../train-labels-idx1-ubyte.npz ../t10k-images-idx3-ubyte.npz ../t10k-labels-idx1-ubyte.npz 100 0.0001  
#Error con eva = 100 0.0001 -> 4.14

import sys
import math
import numpy as np
from pca import pca
from gaussian import gaussian

if len(sys.argv)!=7:
  print('Usage: %s <trdata> <trlabels> <tedata> <telabels> <k> <alpha>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
Y= np.load(sys.argv[3])['Y'];
yl=np.load(sys.argv[4])['yl'];
k=int(sys.argv[5]);
alpha=np.fromstring(sys.argv[6],dtype=np.double,sep=' ');

m, W = pca(X)
XtrP = (X - m) @ W[:,:k]
YtrP = (Y - m) @ W[:,:k]
gaussian(XtrP, xl, YtrP, yl, alpha)

