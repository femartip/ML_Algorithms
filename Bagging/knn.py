import math
import numpy as np
from scipy import stats
from L2dist import L2dist

# Computes the error rate of k nearest neighbors 
# of the test set Y with respect to training set X
# X  is a n x d training data matrix 
# xl is a n x 1 training label vector 
# Y is a m x d test data matrix
# yl is a m x 1 test label vector 
# k is the number of nearest neigbors
def knn(X,xl,Y,yl,k):

  D = np.empty((np.shape(X)[0], np.shape(X)[1], np.shape(Y)[0]))
  #idx = np.empty((np.shape(X)[0], np.shape(X)[1], np.shape(Y)[0]))
  cz = False
  # D is a distance matrix where training samples are by rows 
  # and test sample by columns
  for n in range(np.shape(X)[0]):
    D[n] = L2dist(X[n],Y)
    #mean = np.mean([D[j] for j in range(1,np.shape(X)[0])], axis = 0)
    # Sorting descend per column from closest to farthest
    idx = np.argsort(D[n],axis=0);     #Nos devuelve la posicion de los indices ordenados
    # indexes of k nearest neighbors of each test sample
    idx = idx[:k,:];       #Nos quedamos con los k vecinos mas cercanos a cada muestra
  # Classification of the test samples in the majority class 
  # among the k nearest neighbors using the mode
    c,_ = stats.mode(xl[n][idx]);      #Calcula para cada muestra la moda de la clase mas frecuente 
    if cz == False: classif = c[0]; cz = True
    else:
      classif = np.hstack((classif,c[0]))
  
  classif = np.round(np.mean(classif, axis = 1))
  #classif,_ = stats.mode(np.round(classif), axis = 1)

  #classification = classif[0];        #Clase mas frecuente 
 
  # percentage of error
  err = np.mean(yl!=classif)*100;      #Comparo etiqueta estimada con etiqueta real para obtener prob. de error (media de si falla(1) o acierta (0))

  return err;

