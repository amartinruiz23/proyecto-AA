import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.model_selection import train_test_split

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

df = class_division("data/adult.data", attr)
X_tra, X_tes, y_tra, y_tes = train_test_split(
     X, y, test_size=0.3, shuffle=False)

print(X)

clf = svm.LinearSVC(max_iter=30000)
clf.fit(X_tra, y_tra)


print("E_tra: ", clf.score(X_tra, y_tra))
print("E_tes: ", clf.score(X_tes, y_tes))
# cv_results = cross_validate(clf, X, y, cv=5,)
# print("E_cv: ", sum(cv_results['test_score'])/len(cv_results['test_score']) )
