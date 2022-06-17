#!/usr/bin/python3

import numpy as np

def pca(X):
    m = np.mean(X, axis=0)     #Media de los datos de entrenamiento 
    Xm = np.subtract(X, m)      #Resta de media a los datos de entrenamiento 
    Xmt = np.transpose(Xm)      #Transpuesta de la matriz Xm
    dot = np.dot(Xmt,Xm)              #Producto de Xm' por Xm, equivalente a @ 
    f,c = np.shape(X)               #Obtenemos nº filas y columnas de los datos de entrenamiento
    cov = np.divide(dot, f)         #Calculamos la matriz de covarianza
    eigval, eigvec = np.linalg.eig(cov)     #Calculamos los eigenvalues y eigenvectors de la matriz de covarianza 
    sorted_index = np.argsort(eigval)[::-1]           #Ordenamos los valores propios de mayor a menor
    sort_eigvec = eigvec[:,sorted_index]                   #Ordenamos los vectores propios segun la ordenacion de los valores propios.               
    return m,sort_eigvec.real
