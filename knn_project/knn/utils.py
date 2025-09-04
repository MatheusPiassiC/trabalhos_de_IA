import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple

def train_test_split_simple(df: pd.DataFrame,
                            label_col: str = "Species",
                            train_frac: float = 0.5 ,
                            random_state: int = 13) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    if label_col not in df.columns: #verifica a existência da coluna de espécies
        raise ValueError(f"Coluna de rótulo '{label_col}' não encontrada no DataFrame.")

    if not (0 < train_frac < 1):
        raise ValueError("train_frac deve estar entre 0 e 1 (ex.: 0.7).")

    #Embaralha e reseta índice
    df = df.drop(columns=["Id"])
    # df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    n = len(df_shuffled)
    train_size = int(train_frac * n)
    if train_size == 0 or train_size == n:
        raise ValueError("train_frac resulta em conjunto de treino ou teste vazio para este dataset.")

    train_df = df_shuffled.iloc[:train_size]
    test_df  = df_shuffled.iloc[train_size:]

    X_train = train_df.drop(columns=[label_col]).reset_index(drop=True)
    y_train = train_df[label_col].reset_index(drop=True)
    X_test  = test_df.drop(columns=[label_col]).reset_index(drop=True)
    y_test  = test_df[label_col].reset_index(drop=True)

    return X_train, y_train, X_test, y_test

def carregar_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: '{filepath}'. Verifique o caminho e tente novamente.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"O arquivo '{filepath}' está vazio.")
    except pd.errors.ParserError:
        raise ValueError(f"O arquivo '{filepath}' não pôde ser interpretado como CSV.")
    
#Calcula min e max por coluna no treino.
#Vale observar que, calculamos apenas o min e max do treino, pois no mundo real
#não sabemos os dados que iremos analisar. Por exemplo, se o min de um atributo treino for 7.0,
#e no teste tivermos um atributo de valor 7.5, o KNN teóricamente funciona do mesmo modo,
#adaptando-se a este valor inesperado
def calcular_min_max_params(X_train: pd.DataFrame): #descobre o min-max de um conjunto
    params = {}
    for col in X_train.columns:
        min_val = X_train[col].min()
        max_val = X_train[col].max()
        params[col] = (min_val, max_val)
    return params

def aplicar_min_max_scaling(X: pd.DataFrame, params: dict) -> pd.DataFrame: #normalização usando mim-max
    X_scaled = X.copy()
    for col, (min_val, max_val) in params.items():
        X_scaled[col] = (X[col] - min_val) / (max_val - min_val)
    return X_scaled

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

