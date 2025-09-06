import pandas as pd # type: ignore
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
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # df_shuffled = df.sample(frac=1).reset_index(drop=True)

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


