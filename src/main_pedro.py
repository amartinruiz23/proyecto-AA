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

df_tra = pd.read_csv(
    "data/adult.data",
    names=attr,
)

df_tes = pd.read_csv(
    "data/adult.test",
    names = attr        
)


############ Exposición valores perdidos


print('Value : Number of diferent values : Number of missing values')
for x in df_tra.keys():
    print(x, ':', len(set(df_tra[x])), ':', len(df_tra[df_tra[x] == '?']))


############ Division de Clase
def class_división(df):
    df = df.replace('?', np.nan)
    df_mode=df.mode()
    for x in df_tra.columns.values:
        df[x]=df[x].fillna(value=df_mode[x].iloc[0])
    y = df.pop("Class")
    return df, y

df_tra, y_tra = prepocess(df_tra)
df_tes, y_tes = prepocess(df_tes)


# Random forest

#clf = RandomForestClassifier()
#clf.fit(X_tra, Y_tra)
