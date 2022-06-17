#!/usr/bin/python3

#python ./PCA+KNN/pca+knn-eva.py ../train-images-idx3-ubyte.npz ../train-labels-idx1-ubyte.npz ../t10k-i ../t10k-l "50 60 .."

import sys
import math
import numpy as np
from knn_batch import knnB
from pca import pca
from knn import knn

if len(sys.argv)!=6:
  print('Usage: %s <trdata> <trlabels> <tedata> <telabels> <k>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
Y= np.load(sys.argv[3])['Y'];
yl=np.load(sys.argv[4])['yl'];
k=int(sys.argv[5]);

m, W = pca(X)
XtrP = (X - m) @ W[:,:k]
YtrP = (Y - m) @ W[:,:k]
err = knn(XtrP, xl, YtrP, yl, 1)
print(err)

