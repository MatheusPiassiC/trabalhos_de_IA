import utils 
import metrics



df = utils.carregar_csv("../data/Iris.csv")
X_train, y_train, X_test, y_test = utils.train_test_split_simple(df)

# print(X_train[:5])
# print(y_train[:5])
# print(X_test[:5])
# print(y_test[:5])

min_max = utils.calcular_min_max_params(X_train) #calcula apenas para o df de treino
X_train_scaled = utils.aplicar_min_max_scaling(X_train, min_max)
X_test_scaled = utils.aplicar_min_max_scaling(X_test, min_max)

#print(X_train_scaled[:5])
#print(X_test_scaled[:5])

# print(X_train_scaled.dtypes)
# print(X_test_scaled.dtypes)

k = 7
pred = utils.knn_predict(X_train_scaled, y_train, X_test_scaled, k)
acertos = 0
# tam = len(pred)
# for i in range(tam):
#     acertou = False
#     if pred[i] == y_test[i]:
#          acertos += 1 
#          acertou = True
#     print(pred[i] + " " + y_test[i] + " " + str(acertou))

matriz = metrics.matriz_confusao(pred, y_test)
print(matriz)


    
# print(str(acertos) + "/" + str(tam))
# print(pred[:5])
# print(y_test[:5])


