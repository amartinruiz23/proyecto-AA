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
from matplotlib.pyplot import plot
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

# print_class_balance(y, y_tst)
# TODO: importante, esta función hay que citarla en internet como adaptada
# print_outliers(X)
###### Analisis exploratorio de los datos

# plt.rcParams['figure.figsize'] = [12, 8]
# sns.set(style = 'whitegrid')

# sns.distplot(X['age'], bins = 90, color = 'mediumslateblue')
# plt.ylabel("Distribution", fontsize = 15)
# plt.xlabel("Age", fontsize = 15)
# plt.margins(x = 0)
# plt.show()

##### preprocesado 

tam = info_size(X, 'Tamaño de los datos después de encode:')

preprocesado = pl.make_pipeline(fs.VarianceThreshold(threshold=0.01),
                                sklpre.StandardScaler(),
                                skld.PCA(tol=0.3))

preprocesado.fit(X)
X = preprocesado.transform(X)
X_tst = preprocesado.transform(X_tst)
info_size(X, 'Tamaño de los datos después del preprocesado varianceThreshold:')

#input("\n--- Pulsar tecla para continuar ---\n")

#### PCA

print( 'Analisis de componentes principales')
pca = PCA(tol=0.1)
pca.fit(X)
print('\n Variabilidad explicada por PCA:\n' ,pca.explained_variance_,'\n\n')
pca = PCA(n_components = 55)
pca.fit(X)
X_tst = pca.transform(X_tst)
X = pca.transform(X)
info_size(X, 'Tamaño de los datos después del preprocesado PCA:')

#input("\n--- Pulsar tecla para continuar ---\n")
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
    print("Training: \n - Accuracy: ", accuracy_score(y, pred_tra))
    print("- F1-score: ", f1_score(y, pred_tra, average='macro'))
    #    print("precision  tra: ", precision_score(y, pred_tra))
    
    print("Test: \n - Accuracy: ", accuracy_score(y_tst, pred_tst))
    print(" - F1-score: ", f1_score(y_tst, pred_tst, average='macro'))
    # print("precision tst: ", precision_score(y_tst, pred_tst))

    cv_results = cross_validate(clf,
                                X,
                                y,
                                cv=5,
    )
    print("E_cv:\n - Accuracy: ",
         sum(cv_results['test_score']) / len(cv_results['test_score']))
    cv_results = cross_validate(clf,
                                X,
                                y,
                                cv=5,
                                scoring='f1_macro',
    )
    print(" - F1-score ",
         sum(cv_results['test_score']) / len(cv_results['test_score']))


########### Modelo lineal


# print('\n\nEstudio de la variabilidad polinómica de los datos')
# X_copy = X.copy()
# for i in range(1,3):
#     print('Estudio con dimensión: ',i )
#     poly = PolynomialFeatures(i)
#     np.random.seed(0)
#     poly.fit(X)
#     poly.transform(X)
#     poly.transform(X_tst)
#     clf = LR(random_state=0)
#     clf.fit(X, y)
#     resultados(clf, X, y, X_tst, y_tst)
#     X = X_copy.copy()
# input("\n--- Pulsar tecla para continuar ---\n")


# print('\n\nEstudio de la Fuerza de Regularización Lineal (tarda un poco).')

# acu = []
# fsc = []
# x_axis = [i for i in range(-5,10)]
# for i in x_axis:
#     clf = LR( penalty='l2', random_state=0, solver='liblinear', C=10**i)
#     clf.fit(X, y)

#     cv_results = cross_validate(clf,
#                                 X,
#                                 y,
#                                 cv=5,
#     )
    
#     acu.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
#     cv_results = cross_validate(clf,
#                                 X,
#                                 y,
#                                 cv=5,
#                                 scoring='f1_macro',
#     )
#     fsc.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
# plt.figure()
# plt.title('Fuerza de regularización')
# plt.xlabel('Valor en escala logaritmica base 10')
# plot(x_axis, acu, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='accuracy')
# plot(x_axis, fsc, 'go',color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='f1-score')
# plt.legend()
# plt.show()

# print('Mejor resultado: ', acu.index(max(acu)))

# input("\n--- Pulsar tecla para continuar ---\n")



# print('\n-- Modelo lineal --\n')



# #mejor modelo: liblinear con regularizaciónl2 C = 1.9 y 100 iteraciones
# clf = LR( penalty='l2', random_state=0, solver='liblinear', C=1.9)

# clf.fit(X, y)
# resultados(clf, X, y, X_tst, y_tst)

#input("\n--- Pulsar tecla para continuar ---\n")

# print('\n-- Random Forest --\n')

# clf = RandomForestClassifier(n_estimators=400, max_depth=50)
# clf.fit(X, y)
# resultados(clf, X, y, X_tst, y_tst)

"""
print('\nComparación de criterios\n')
for crit in  ['gini', 'entropy']:
    clf = RandomForestClassifier(n_estimators=400, max_depth=50, criterion = crit)
    clf.fit(X, y)
    resultados(clf, X, y, X_tst, y_tst)

print('\nComparación de profundidad\n')

e_cv = []
e_in = []
x_axis = [80, 60, 40, 30, 20, 10, 5, 3, 1]
for i in x_axis:
    clf = RandomForestClassifier(n_estimators=400, max_depth=i)
    clf.fit(X, y)

    cv_results = cross_validate(clf,X,y,cv=5)    
    e_cv.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
    #print(e_cv)
    
    pred_tra = clf.predict(X)
    e_in.append(accuracy_score(y, pred_tra))
    #print(e_in)
    
plt.figure()
plt.title('Profundidad, máxima cross validation')
plot(x_axis, e_cv, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e cv')
plt.show()

plt.figure()
plt.title('Profundidad, error en training')
plot(x_axis, e_in,color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e in')
plt.show()


print('\nComparación de n estimadores\n')

e_cv = []
e_in = []
x_axis = [10, 50, 100, 200, 400, 500, 600]
for i in x_axis:
    clf = RandomForestClassifier(n_estimators=i, max_depth=50)
    clf.fit(X, y)

    cv_results = cross_validate(clf,X,y,cv=5)    
    e_cv.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
    #print(e_cv)
    
    pred_tra = clf.predict(X)
    e_in.append(accuracy_score(y, pred_tra))
    #print(e_in)
    
plt.figure()
plt.title('Estimadores, cross validation')
plot(x_axis, e_cv, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e cv')
plt.show()

plt.figure()
plt.title('Estimadores, error en training')
plot(x_axis, e_in, color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e in')
plt.show()
"""

print('\n-- Support vector machine --\n')

clf = svm.SVC(max_iter=10000, C=2, kernel='rbf')
clf.fit(X, y)
resultados(clf, X, y, X_tst, y_tst)


print('\n\nEstudio de kernel (tarda un poco).')

for kernel in ['rbf','linear']:
    clf = svm.SVC(max_iter=15000, C=2, kernel=kernel)
    clf.fit(X, y)
    resultados(clf, X, y, X_tst, y_tst)



print('\n\nEstudio de la Fuerza de Regularización Lineal (tarda un poco).')

acuGaus = []
fscGaus = []
x_axis = [i for i in range(-5,6,2)]
for i in x_axis:
    clf = svm.SVC(max_iter=3000, C=10**(i))
    clf.fit(X, y)

    cv_results = cross_validate(clf,
                                X,
                                y,
                                cv=5,
    )
    
    acuGaus.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
    print(acuGaus)
    cv_results = cross_validate(clf,
                                X,
                                y,
                                cv=5,
                                scoring='f1_macro',
    )
    fscGaus.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
    print(fscGaus)


plt.figure()
plt.title('Fuerza de regularización')
plt.xlabel('Valor en escala logaritmica base 10')
plot(x_axis, acuGaus, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='accuracy')
plot(x_axis, fscGaus, 'go',color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='f1-score')
plt.legend()
plt.show()


# input("\n--- Pulsar tecla para continuar ---\n")
