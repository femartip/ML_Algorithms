import numpy as np
import math
import sys

def gaussian(Xtr,xltr,Xdv,xldv,alphas):
    #Estimacion param por clase
    labs = np.unique(xltr).astype(int)      #Pasas etiquetas a lista de numero unicos (1,2...9)
    p = np.empty((1,np.shape(labs)[0]))
    #mu = np.empty((np.shape(labs)[0], np.shape(Xtr)[1], np.shape(Xtr)[1]))
    mu = np.empty((np.shape(labs)[0], np.shape(Xtr)[1]))
    cov = np.empty((np.shape(labs)[0], np.shape(Xtr)[1], np.shape(Xtr)[1]))
    for l in labs:
        idc = np.where(xltr == l)           #Muestras que son de la clase
        p[0][l] = np.shape(idc)[0]/np.shape(xltr)[0]              #Probabilidad de la clase
        mu[l] = np.mean(Xtr[idc], axis=0)     #Media de los datos de entrenamiento de la clase
        cov[l] = np.divide(np.dot(np.transpose(Xtr[idc]),Xtr[idc]), np.shape(Xtr[idc])[0])         #Calculamos la matriz de covarianza de la clase
    
    #Para cada valor de alpha, suavizar y calcular funcion disc. por cada clase para Xdv con funcion auxilia pxc = gc(). Con esto hacemos la clasi. argmax
    err = np.empty((1, np.shape(labs)[0]))
    g = np.empty((np.shape(labs)[0], np.shape(Xdv)[0])) #Dim LxN
    for a in alphas:
        for l in labs:    
            flatsm = a * cov[l] + (1-a)* np.identity(np.shape(cov[l])[0])       #Smoothing, mismas dimension que cov. 
            g[l] = pxc(p[0][l], mu[l], flatsm, Xdv)     #Funcion discriminante
        pc = np.argmax(g, axis=0)                     #En que clase se clasifica
        err = np.mean(xldv!=pc)*100         #Porcentaje de error
        print(a, err)

def pxc(pc, mu, sigma, X):
    res = np.empty((np.shape(X)[1], np.shape(X)[0]))        #DxN 
    casicero = sys.float_info.min
    X = np.where(X == 0, casicero, X)   #Sustituir elementos de cero por casi cero
    inv = np.linalg.pinv(sigma)         #matriz pseudoinvera de la cov
    Wc = -1/2 * inv    #DxD
    wc = np.sum(inv * mu, axis = 1)      #Dim = Dx1
    wc0 = np.log2(pc) - 1/2 * logdet(sigma) - 1/2 * (np.transpose(mu) @ inv @ mu)      #dim = 1  
    res = np.sum((X@Wc)*X, axis=1) + X @ wc + wc0   #Dim Nx1 = Nx1 + Nx1 + 1
    #res = np.transpose(X) * Wc * X + np.transpose(wc) * np.transpose(X) + wc0    #Dim DxN = DxD * DxN + 1
    return np.transpose(res[:,None].real)        #Dim 1xN

def logdet(sigma):
    eigval, eigvec = np.linalg.eig(sigma)
    #Compara si log es 0
    if np.any(eigval <= 0): return np.log2(eigval)  #Dx1  Vector de valores simples
    else: return np.sum(np.log2(eigval))