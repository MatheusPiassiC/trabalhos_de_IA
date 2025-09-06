import numpy as np
from collections import Counter



def calcula_distancias(X_train: np.ndarray, x:np.ndarray) -> np.ndarray:
    dif = X_train - x; #subtrai os valores de x em X_train
    dif = dif**2       #eleva a diferença ao quadrado
    dif = dif.sum(axis=1) #faz o somatório
    dist = np.sqrt(dif)   #tira a raiz
    return dist  

def pegar_k_vizinhos(X_train: np.ndarray, y_train: np.ndarray, x: np.ndarray, k: int):
    dist = calcula_distancias(X_train, x) #calcula as distâncias de todos até x
    indices_ordenados = np.argsort(dist)  #ordena os indices
    return y_train[indices_ordenados[:k]]

def votar(vizinhos): #retorna a classe mais comum entre os vizinhos mais próximos
    #print("DEBUG vizinhos:", vizinhos)
    contagem = Counter(vizinhos)
    classe_mais_comum = None
    maior_quantidade = 0

    for classe,quantidade in contagem.items():
        if quantidade > maior_quantidade:
            maior_quantidade = quantidade
            classe_mais_comum = classe

    return classe_mais_comum

def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    X_train_np = X_train.values.astype(float)
    y_train_np = y_train.values  # se já for string, mantém
    for i in range(len(X_test)):
        x = X_test.iloc[i].values.astype(float)
        vizinhos = pegar_k_vizinhos(X_train_np, y_train_np, x, k)
        classe = votar(vizinhos)
        y_pred.append(classe)
    return np.array(y_pred)




