import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing, svm
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
from sklearn.metrics import precision_score
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

X, X_tst = encode_categorical_variables(X, X_tst)

print_class_balance(y, y_tst)

###### Analisis exploratorio de los datos

plt.rcParams['figure.figsize'] = [12, 8]
sns.set(style = 'whitegrid')

sns.distplot(X['age'], bins = 90, color = 'mediumslateblue')
plt.ylabel("Distribution", fontsize = 15)
plt.xlabel("Age", fontsize = 15)
plt.margins(x = 0)
plt.show()





# -- preprocesado    le.fit(union[feature])

tam = info_size(X, 'Tamaño de los datos después de encode:')

preprocesado = pl.make_pipeline(fs.VarianceThreshold(threshold=0.01),
                                sklpre.StandardScaler(),
                                skld.PCA(n_components=tam - 1))

preprocesado.fit(X)
X = preprocesado.transform(X)
X_tst = preprocesado.transform(X_tst)
info_size(X, 'Tamaño de los datos después del preprocesado:')

#### PCA

print( 'Analisis de componentes principales')
pca = PCA(tol=0.01, n_components = X.shape[1])
X = pca.fit(X)

print(pca.explained_variance_)

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
    print("precision  tra: ", precision_score(y, pred_tra))
    
    print("Accuracy tst: ", accuracy_score(y_tst, pred_tst))
    print("f1-score tst: ", f1_score(y_tst, pred_tst, average='macro'))
    print("precision tst: ", precision_score(y_tst, pred_tst))

    #cv_results = cross_validate(clf,
    #    X,
    #    y,
    #    cv=5,
    #)
    #print("E_cv: ",
    #      sum(cv_results['test_score']) / len(cv_results['test_score']))


########### Modelo lineal
# print('\n-- Modelo lineal --\n')
# # mejor modelo: liblinear con regularizaciónl1 C = 0.1 y 100 iteraciones

# clf = LR(max_iter=100, penalty='l1', random_state=0, solver='liblinear', C=0.1)

# clf.fit(X, y)
# resultados(clf, X, y, X_tst, y_tst)

#input("\n--- Pulsar tecla para continuar ---\n")

# --- Random forest ---

# print('\n-- Random Forest --\n')

# clf = RandomForestClassifier(n_estimators=600,
#                              criterion='entropy',
#                              max_depth=50,
#                              oob_score=True)
# clf.fit(X, y)
# resultados(clf, X, y, X_tst, y_tst)

# print('\n-- Support vector machine --\n')

# clf = svm.LinearSVC(max_iter=30000)
# clf.fit(X, y)
# resultados(clf, X, y, X_tst, y_tst)
