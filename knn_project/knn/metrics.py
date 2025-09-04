import numpy as np
import pandas as pd

def matriz_confusao(pred, y_test): #gera a matriz de confusão
    classes = np.unique(np.concatenate([y_test, pred])) #separa as classes
    matriz = np.zeros((len(classes), len(classes)), dtype=int)#cria uma matriz de zeros

    #preenche a matriz
    for real, predito in zip(y_test, pred):
        i = np.where(classes == real)[0][0]   #índice da classe real
        j = np.where(classes == predito)[0][0] #índice da classe prevista
        matriz[i, j] += 1 

    return pd.DataFrame(matriz, index=classes, columns=classes)