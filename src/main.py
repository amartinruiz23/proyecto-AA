import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from collections import Counter
from aux import class_division, scale
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

X,  X_tst,y, y_tst = class_division("data/adult.data", attr)
X = scale(X)

# --- Random forest ---

clf = RandomForestClassifier(n_estimators = 600, criterion = 'entropy', max_depth = 50)

clf.fit(X, y)
print("E_tra: ", clf.score(X, y))
print("E_tst: ", clf.score(X_tst, y_tst))

cv_results = cross_validate(clf, X, y, cv=5,)
print("E_cv: ", sum(cv_results['test_score'])/len(cv_results['test_score']) )
