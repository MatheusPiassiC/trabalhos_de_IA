# Projeto KNN – Implementação Manual vs. Scikit-Learn

## 🎯 Enunciado do Trabalho
O trabalho consiste em implementar um classificador **K-Nearest Neighbors (KNN)** de duas formas:

1. **Implementação manual (hardcore):**
   - Desenvolver do zero o algoritmo KNN, incluindo:
     - Funções de leitura e pré-processamento do dataset.
     - Divisão treino/teste manual.
     - Normalização dos dados via **Min-Max Scaling**.
     - Implementação da lógica de classificação pelo método da **distância euclidiana**.
     - Cálculo das métricas de avaliação.

2. **Implementação usando Scikit-Learn:**
   - Utilizar a biblioteca `scikit-learn` para:
     - Dividir os dados (`train_test_split`).
     - Normalizar os dados (`MinMaxScaler`).
     - Treinar o modelo (`KNeighborsClassifier`).
     - Avaliar o modelo (`confusion_matrix`, `classification_report`, `accuracy_score`).

3. **Comparação de desempenho entre as abordagens:**
   - Comparar as métricas de avaliação:
     - **Acurácia**
     - **Precisão**
     - **Revocação**
   - Analisar semelhanças e diferenças entre a implementação manual e a implementação com scikit-learn.

---

## 🛠️ Implementações

### 🔹 KNN Manual (hardcore)
- Escrita do zero, sem bibliotecas externas.
- Fluxo:
  1. Carregar dataset Iris (`Iris.csv`).
  2. Remover coluna de IDs.
  3. Dividir manualmente em treino/teste.
  4. Calcular parâmetros `min` e `max` apenas com o conjunto de treino.
  5. Normalizar os dados.
  6. Implementar a função `knn_predict` usando distância euclidiana.
  7. Calcular métricas de desempenho.

### 🔹 KNN com Scikit-Learn
- Reaproveita funções já implementadas na biblioteca.
- Fluxo:
  1. Carregar dataset Iris (`Iris.csv`).
  2. Dividir em treino/teste com `train_test_split`.
  3. Normalizar dados com `MinMaxScaler`.
  4. Criar e treinar o classificador `KNeighborsClassifier`.
  5. Avaliar com `confusion_matrix`, `classification_report` e `accuracy_score`.

---

## 📊 Métricas de Avaliação

As métricas usadas foram:

- **Acurácia:** Proporção de previsões corretas.
- **Precisão (Precision):** Entre os exemplos classificados como uma classe, quantos realmente pertencem a ela.
- **Revocação (Recall):** Entre os exemplos que pertencem a uma classe, quantos foram corretamente recuperados.

Todas as métricas foram calculadas **por classe** e também no agregado.

---

## 📈 Resultados

- Os resultados foram **muito semelhantes** entre a implementação manual e a do Scikit-Learn.  
- Com o valor de random_state utilizado, a implementação manual teve acurácia idêntica à implementação com a biblioteca.  
- Mais detalhes dos rtesultados podem ser vistos no pdf de relatório.

---

## 🚀 Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone <url-do-repo>
   cd knn_project
