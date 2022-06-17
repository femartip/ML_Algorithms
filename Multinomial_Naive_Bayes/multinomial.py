#!/usr/bin/python3

import numpy as np

def multinomial(Xtr,xltr,Xdv,xldv,epsilon):
    # HERE YOUR CODE
     
    #Multinomial, dada la matriz pc, clasifica segun max. ver. de bernouilli
    #Calcular pc
    labs = np.unique(xltr).astype(int)      #D = Lx1 Pasas etiquetas a lista de numero unicos (1,2...9)
    N,D = Xtr.shape                      
    mpc = np.zeros((np.shape(labs)[0],D))      #Dim = LxD Creas matriz vacia de tamaño (nº etiquetas, nº muestras)
    wc0 = np.empty((np.shape(labs)[0],1))        #Dim = Lx1
    for c in labs:  
        idc = np.where(xltr==c)             #Obtener todas las muestras para la clase c
        wc0[c] = np.shape(idc)[1]/ N    #Dim = 1x1   P(X)=Nc/N      wc0 = log P(X)
        sc = np.sum(Xtr[idc], axis=0)
        mpc[c] = sc/np.sum(sc)
        #for i in range(len(idc)):
        #    mpc[c][idc[i]] = sc[i]/np.sum(sc)
    
    #Suavizado de Laplace, normaliza el valor pc
    mpc = mpc + epsilon
    spc = np.zeros((np.shape(labs)[0],D))   #Dim = LxD
    for l in range(len(labs)):
        spc[l] = mpc[l]/np.sum(mpc + epsilon, axis=1)[l]
    """spc = np.zeros((np.shape(labs)[0],D))   #Dim = LxD
    for prot in mpc:
        total = np.sum(prot + epsilon)     
        for l in labs:    
            #print(np.divide(prot + epsilon, total))
            spc[l] = prot + epsilon/ total
            #for param in range(len(prot)):
                #spc[c][param] = (prot[param] + epsilon)/total   
    """
    wc = np.log2(spc)          #wc = log pc
    g = Xdv@np.transpose(wc) + np.transpose(np.log2(wc0))               #Dim NxL = NxD @ LxD' + Lx1'      g(x) = log pc + log p(c) 
    max = np.argmax(g, axis= 1)           #Maximo de cada muestra (por filas)
    err = np.mean(xldv!=max)*100 #Porcentaje de error
    return err