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
from sklearn.feature_selection import SelectKBest, chi2
np.random.seed(0)

#warnings.filterwarnings('ignore')

# --- Lectura de los datos ---

description = "data/adult.names"


attr = []
with open(description, "r") as f:
    for line in f:
        if line.startswith("@attribute"):
            line = line.split()
            attr.append(line[1])

X, X_tst, y, y_tst = class_division("data/adult.data", attr)
balanceo_clases(y, y_tst)

# -- preprocesado

info_size(X, 'Tamaño de los datos después de las dummy variables:')

preprocesado = pl.make_pipeline(fs.VarianceThreshold(threshold=0.01),
                                sklpre.StandardScaler()
                                )

preprocesado.fit(X)
X = preprocesado.transform(X)
X_tst = preprocesado.transform(X_tst)


# keys = X.keys()
# selector = SelectKBest(k=5)
# selector.fit(X,y)
# mask = selector.get_support()
# print(keys[mask])

info_size(X, 'Tamaño de los datos después del preprocesado:')


# --- Random forest ---

clf = RandomForestClassifier(n_estimators = 600, criterion = 'entropy', max_depth = 50)

clf.fit(X, y)
print("E_tra: ", clf.score(X, y))
print("E_tst: ", clf.score(X_tst, y_tst))

# pred = clf.predict(X_tst)
# print("f1_score tst: ", f1_score(y_tst, pred, average='macro'))

# cv_results = cross_validate(clf, X, y, cv=5,)
# print("E_cv: ", sum(cv_results['test_score'])/len(cv_results['test_score']) )
