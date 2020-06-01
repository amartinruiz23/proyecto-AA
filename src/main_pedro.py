import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing

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
    df_tes = pd.read_csv(
        filename,
    names = attr        
    )
    df = df.replace('?', np.nan)
    df_mode=df.mode()
    for x in df_tra.columns.values:
        df[x]=df[x].fillna(value=df_mode[x].iloc[0])

    y = df.pop("Class")

    return df, y

X_tra, y_tra = class_division("data/adult.data", attr)
X_tes, y_tes = class_division("data/adult.test", attr)



clf = RandomForestClassifier()
clf.fit(X_tra, y_tra)
