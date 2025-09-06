from knn_manual import utils, metrics, core  
from knn_sklearn import core as sklearn_core

def main():
    df = utils.carregar_csv("data/Iris.csv")
    
    X_train, y_train, X_test, y_test = utils.train_test_split_simple(df)

    min_max = utils.calcular_min_max_params(X_train)
    X_train_scaled = utils.aplicar_min_max_scaling(X_train, min_max)
    X_test_scaled = utils.aplicar_min_max_scaling(X_test, min_max)
    
    ks = [1, 3, 5, 7]

    print("===== KNN Manual =====")
    for k in ks:
        pred = core.knn_predict(X_train_scaled, y_train, X_test_scaled, k)
        metrics.avaliar_resultados(y_test, pred, k)

    print("\n===== KNN Sklearn =====")
    for k in ks:
        y_test_skl, y_pred_skl, classes_skl = sklearn_core.knn_sklearn("data/Iris.csv", k)

if __name__ == "__main__":
    main()