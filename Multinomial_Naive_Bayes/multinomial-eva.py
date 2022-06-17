#!/usr/bin/python3

#Error -> 16.32

import sys
import numpy as np
from multinomial import multinomial

if len(sys.argv)!=6:
  print('Usage: %s <trdata> <trlabels> <tedata> <telabels> <e>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];
Y= np.load(sys.argv[3])['Y'];
yl=np.load(sys.argv[4])['yl'];
e=np.double(sys.argv[5]);

err = multinomial(X, xl, Y, yl, e)
print(err)