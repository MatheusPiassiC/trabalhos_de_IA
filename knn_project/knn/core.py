import utils 
import metrics



df = utils.carregar_csv("../data/Iris.csv")
X_train, y_train, X_test, y_test = utils.train_test_split_simple(df)

min_max = utils.calcular_min_max_params(X_train) #calcula apenas para o df de treino
X_train_scaled = utils.aplicar_min_max_scaling(X_train, min_max)
X_test_scaled = utils.aplicar_min_max_scaling(X_test, min_max)


k1 = 1
pred = utils.knn_predict(X_train_scaled, y_train, X_test_scaled, k1)
metrics.avaliar_resultados(y_test, pred, k1)

k3 = 3
pred = utils.knn_predict(X_train_scaled, y_train, X_test_scaled, k3)
metrics.avaliar_resultados(y_test, pred, k3)

k5 = 5
pred = utils.knn_predict(X_train_scaled, y_train, X_test_scaled, k5)
metrics.avaliar_resultados(y_test, pred, k5)

k7 = 7
pred = utils.knn_predict(X_train_scaled, y_train, X_test_scaled, k7)
metrics.avaliar_resultados(y_test, pred, k7)




