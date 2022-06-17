#!/usr/bin/python3

#python ./Gaussian_Naive_Bayes/gaussian-eva.py ../train-images-idx3-ubyte.npz ../train-labels-idx1-ubyte.npz ../t10k-images-idx3-ubyte.npz ../t10k-labels-idx1-ubyte.npz 0.0001 
# Error con EVA = 0.0001 4.279999999999999

import sys
import math
import numpy as np
from gaussian import gaussian


if len(sys.argv)!=6:
  print('Usage: %s <trdata> <trlabels> <tedata> <telabels> <alpha>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
Y= np.load(sys.argv[3])['Y'];
yl=np.load(sys.argv[4])['yl'];
alpha=np.fromstring(sys.argv[5],dtype=np.double,sep=' ');

err = gaussian(X, xl, Y, yl, alpha)

