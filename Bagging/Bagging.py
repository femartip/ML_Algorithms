#!/usr/bin/python3

from random import randrange
import numpy as np

#Submuestra de Xtr, donde m es la cantidad de submuestras creadas y n la cantidad de muestras que contiene la submuestra
def subsample(Xtr,xltr,n,m):
    N,D = Xtr.shape
    Nn = int(N * (n/100))
    sample = np.empty((m, Nn, D))      
    sample_labels = np.empty((m, Nn, 1))   
    for sm in range(m):
        for sn in range(n):
            index = randrange(N)
            sample[sm][sn] = Xtr[index]
            sample_labels[sm][sn] = xltr[index]
    
    return sample, sample_labels