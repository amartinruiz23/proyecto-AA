import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from collections import Counter
from aux import *
import sklearn.pipeline as pl
import sklearn.decomposition as skld
import sklearn.feature_selection as fs
import sklearn.decomposition as skld
import sklearn.preprocessing as sklpre
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import SelectKBest, chi2
np.random.seed(0)
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')

# --- Lectura de los datos ---

description = "data/adult.names"

attr = []
with open(description, "r") as f:
    for line in f:
        if line.startswith("@attribute"):
            line = line.split()
            attr.append(line[1])

X, y = class_division("data/adult.data", attr)
X_tst, y_tst = class_division("data/adult.test", attr)

encode_categorical_variables(X, X_tst)

print_class_balance(y, y_tst)

# -- preprocesado    le.fit(union[feature])


tam = info_size(X, 'Tamaño de los datos después de encode:')

preprocesado = pl.make_pipeline(fs.VarianceThreshold(threshold=0.01),
                                sklpre.StandardScaler(),
                                skld.PCA(n_components=tam-1))

preprocesado.fit(X)
X = preprocesado.transform(X)
X_tst = preprocesado.transform(X_tst)
info_size(X, 'Tamaño de los datos después del preprocesado:')


########### Validación cruzada:
def resultados(
    clf,
    X,
    y,
    X_tst,
    y_tst,
    msg=None,
):
    if msg is not None:
        print(msg)


    pred_tra = clf.predict(X)
    pred_tst = clf.predict(X_tst)
    print("Accuracy tra: ", accuracy_score(y, pred_tra))
    print("f1-score tst: ", f1_score(y, pred_tra, average='macro'))

    print("Accuracy tst: ", accuracy_score(y_tst, pred_tst))
    print("f1-score tst: ", f1_score(y_tst, pred_tst, average='macro'))

    cv_results = cross_validate(clf,
        X,
        y,
        cv=5,
    )
    #print("E_cv: ",
    #      sum(cv_results['test_score']) / len(cv_results['test_score']))


########### Modelo lineal

penaltys = ['l1', 'l2']
solvers = ['lbfgs', 'liblinear']
max_iters = [100, 200, 1000]
C = [0.1,1,10]
print('\n\n -- Resultados Modelo lineal')
for penalty in penaltys:
    for solver in solvers:
        for max_iter in max_iters:
            for c in C:
                if (penalty != 'l1' or solver != 'lbfgs') and (
                    penalty != 'none' or solver != 'liblinear'
                ):  # incluimos pq lbfgs no soporta regularizaciónl1

                    clasificador = LR(max_iter=max_iter,
                                  penalty=penalty,
                                  random_state=0,
                                  solver=solver, C=c
                    )
                    
                    clasificador.fit(X, y)

                    resultados(
                        clasificador, X, y, X_tst, y_tst,
                        '\nResultados de ' + solver + ' con regularización' +
                        penalty + ' C = ' +str(c) +' y ' +
                        str(max_iter) + ' iteraciones.')
#input("\n--- Pulsar tecla para continuar ---\n")

########### Modelo lineal variables cuadráticas
poly = PolynomialFeatures(2)
poly.fit(X)
poly.transform(X)
poly.transform(X_tst)
penaltys = ['l1', 'l2']
solvers = ['lbfgs', 'liblinear']
max_iters = [100, 200, 1000]
C = [0.1,1,10]
print('\n\n -- Resultados Modelo lineal variables cuadráticas')
for penalty in penaltys:
    for solver in solvers:
        for max_iter in max_iters:
            for c in C:
                if (penalty != 'l1' or solver != 'lbfgs') and (
                    penalty != 'none' or solver != 'liblinear'
                ):  # incluimos pq lbfgs no soporta regularizaciónl1

                    clasificador = LR(max_iter=max_iter,
                                  penalty=penalty,
                                  random_state=0,
                                  solver=solver, C=c
                    )
                    
                    clasificador.fit(X, y)

                    resultados(
                        clasificador, X, y, X_tst, y_tst,
                        '\nResultados de ' + solver + ' con regularización' +
                        penalty + ' C = ' +str(c) +' y ' +
                        str(max_iter) + ' iteraciones.')

#input("\n--- Pulsar tecla para continuar ---\n")
                                
# --- Random forest ---

clf = RandomForestClassifier(n_estimators = 600, criterion = 'entropy', max_depth = 50)
clf.fit(X,y)
resultados(clf, X, y, X_tst, y_tst)