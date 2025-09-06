import pandas as pd # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score # type: ignore

def knn_sklearn(filepath: str, k:int, train_frac: int = 0.5):
    df = pd.read_csv(filepath)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"]) #retira a coluna de id's, pois não influencia no knn

    df_shuffled = df.sample(frac=1, random_state=13).reset_index(drop=True)
    # df_shuffled = df.sample(frac=1).reset_index(drop=True)

    n = len(df_shuffled)
    train_size = int(train_frac * n)
    if train_size == 0 or train_size == n:
        raise ValueError("train_frac resulta em conjunto de treino ou teste vazio para este dataset.")

    train_df = df_shuffled.iloc[:train_size]
    test_df  = df_shuffled.iloc[train_size:]

    X_train = train_df.drop(columns=["Species"]).reset_index(drop=True)
    y_train = train_df["Species"].reset_index(drop=True)
    X_test  = test_df.drop(columns=["Species"]).reset_index(drop=True)
    y_test  = test_df["Species"].reset_index(drop=True)

    scaler = MinMaxScaler() #cria um objeto MinMaxScaler
    X_train_scaled = scaler.fit_transform(X_train) #calcula o min e o max e aplica a formula de normalização em todos os itens
    X_test_scaled = scaler.transform(X_test) #scaler já sabe o min e o max do treino, então só aplica a formula

    knn = KNeighborsClassifier(n_neighbors=k) #cria um onjeto knn
    knn.fit(X_train_scaled, y_train) #por padrão, usa distância euclidiana e vptação uniforme
                                     #como eu fiz na implementação manual, e treina o modelo com os dados normalizados

    y_pred = knn.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average=None, labels=knn.classes_)
    recall = recall_score(y_test, y_pred, average=None, labels=knn.classes_)

    print("\n" + "="*50)
    print(f"Resultados (Sklearn) para k = {k}")
    print("="*50)

    print("Matriz de confusão:")
    print(pd.DataFrame(cm, index=knn.classes_, columns=knn.classes_))

    print(f"\nAcurácia: {acuracia:.4f}")

    print("\nPrecisão por classe:")
    for c, v in zip(knn.classes_, precisao):
        print(f"  {c}: {v:.4f}")

    print("\nRevocação por classe:")
    for c, v in zip(knn.classes_, recall):
        print(f"  {c}: {v:.4f}")
    return y_test, y_pred, knn.classes_

    
