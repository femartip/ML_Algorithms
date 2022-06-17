import numpy as np
#Calculate matrix W corresponding to weighted Mahalanobis distance per class
def Md4cweights(X,xl,alphas):
    W = np.empty(np.shape(X))    #W (np.shape(xl), 1)
    for l in range(len(X)):
        #Find all xl that are the same as l
        xl_l = np.where(xl==xl[l])[0]
        var = np.var(X[xl_l], axis=0)
        var_suavizada = alphas*var + (1-alphas)*1
        W[l] = var_suavizada        #Dim = NxD
    return W

#Calculate euclidean ponderated distance
# Efficient implementation of L2 weighed distance without square root
#with ĉ(x)=c
# d(x,y) = \sum_d w_cd*(x_d - y_d)^2 =
#          \sum_d w_cd*x_d^2 + \sum_d w_cd*y_d^2 - 2*\sum_d w_cd*x_d*y_d
def wL2dist(X,Y,W):
    #Implementacion no eficiente
    #D = np.empty((X.shape[0],Y.shape[0]))
    #for l in range(X.shape[0]):
    #    D[l,:] = np.sum(W[l] * (X[l,:]-Y)**2,axis=1) 
    XX = np.transpose(np.sum(np.multiply(W,np.square(X)),axis=1))[:,None]       #Dim = 1xN, Mult. X*X^2 elemento a elemento
    YY = W@np.transpose(np.square(Y))  #Dim = Nx1
    MM  = np.multiply(2*W, X)@np.transpose(Y)   #Dim = NxM
    D = XX + YY - MM        #Dim = 1xN + Nx1 -NxM
    return D

'''
Debes tener en cuenta que necesitas calcular la matriz de distancias D
de dimensiones N x M, es decir, la distancia que hay de cada muestra y
a todas las muestras x donde cada x tiene su vector de pesos asociado.
Así que te hace falta un producto matricial entre Y y W donde cada
muestra y se ha multiplicado por cada uno de los vectores de pesos en
W, ya que cada vector de pesos va asociado a una muestra x.
'''