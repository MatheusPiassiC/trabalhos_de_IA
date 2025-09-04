import numpy as np
import pandas as pd

def matriz_confusao(pred, y_test, classes): #gera a matriz de confusão
    matriz = np.zeros((len(classes), len(classes)), dtype=int)#cria uma matriz de zeros

    #preenche a matriz
    for real, predito in zip(y_test, pred):
        i = np.where(classes == real)[0][0]   #índice da classe real
        j = np.where(classes == predito)[0][0] #índice da classe prevista
        matriz[i, j] += 1 

    return pd.DataFrame(matriz, index=classes, columns=classes)

def calcular_acuracia(cm):
    acertos = cm.values.diagonal().sum() #soma os valores da diagonal
    total = cm.values.sum() #soma todos os valores
    return acertos / total

def calcular_precisao(cm, classes):
    precisao = {}
    for classe in classes:
        TP = cm.loc[classe, classe]
        FP = cm[classe].sum() - TP
        precisao[classe] = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precisao

def calcular_revocacao(cm, labels):
    revocacao = {}
    for classe in labels:
        TP = cm.loc[classe, classe]
        FN = cm.loc[classe].sum() - TP
        revocacao[classe] = TP / (TP + FN) if (TP + FN) > 0 else 0
    return revocacao

def avaliar_resultados(y_test, pred, k=None):
    classes = np.unique(np.concatenate([y_test, pred])) #separa as classes
    cm = matriz_confusao(y_test, pred, classes)
    acuracia = calcular_acuracia(cm)
    precisao = calcular_precisao(cm, classes)
    recall = calcular_revocacao(cm, classes)

    # Impressão organizada
    print("\n" + "="*50)
    if k is not None:
        print(f"Resultados para k = {k}")
    print("="*50)
    print("Matriz de confusão:")
    print(cm)

    print(f"\nAcurácia: {acuracia:.4f}")

    print("\nPrecisão por classe:")
    for c, v in precisao.items():
        print(f"  {c}: {v:.4f}")

    print("\nRevocação por classe:")
    for c, v in recall.items():
        print(f"  {c}: {v:.4f}")


    return {
        "k": k,
        "acuracia": acuracia,
        "precisao": precisao,
        "recall": recall,
    }

