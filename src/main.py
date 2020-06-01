import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate

np.random.seed(0)

############ Lectura de los datos

description = "data/adult.names"
attr = []
with open(description, "r") as f:
    for line in f:
        if line.startswith("@attribute"):
            line = line.split()
            attr.append(line[1])

def class_division(filename, attr):
    df = pd.read_csv(
        filename,
    names = attr        
    )
    df = df.replace('?', np.nan)
    df_mode=df.mode()
    for x in df.columns.values:
        df[x]=df[x].fillna(value=df_mode[x].iloc[0])

    y = df.pop("Class")
    df = pd.get_dummies(df)
    return df, y

X, y = class_division("data/adult.data", attr)


print(X)

clf = RandomForestClassifier()
# clf.fit(X, y)
# pred_tra = clf.predict(X)

#print("E_tra: ", clf.score(X, y))
cv_results = cross_validate(clf, X, y, cv=5,)
print("E_cv: ", sum(a['test_score'])/len(a['test_score']) )
