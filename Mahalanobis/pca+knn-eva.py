#!/usr/bin/python3

#Mejor resultado para eva 1e-06 50 -> 

import sys
import math
import numpy as np
from pca import pca
from knn import knnMD

if len(sys.argv)!=7:
  print('Usage: %s <trdata> <trlabels> <tedata> <telabels> <dim> <alpha>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
Y= np.load(sys.argv[3])['Y'];
yl=np.load(sys.argv[4])['yl'];
dim=int(sys.argv[5]);
alpha=float(sys.argv[6]);

m, W = pca(X)
XtrP = (X - m) @ W[:,:dim]
YtrP = (Y - m) @ W[:,:dim]
err = knnMD(XtrP, xl, YtrP, yl, 1, alpha)
print(err)

