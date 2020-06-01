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


print('Value : Number of diferent values : Number of missing values')
for x in df_tra.keys():
    print(x, ':', len(set(df_tra[x])), ':', len(df_tra[df_tra[x] == '?']))



print(df_tra.shape)
df_tra = df_tra.replace('?', np.nan) # Reemplazamos los valores ? por valores perdidos
df_tes = df_tes.replace('?', np.nan)

elem, cols = df_tra.shape
df_mode=df_tra.mode()
for x in df_tra.columns.values:
    df_tra[x]=df_tra[x].fillna(value=df_mode[x].iloc[0])
    
df_mode=df_tes.mode()
for x in df_tes.columns.values:
    df_tes[x]=df_tes[x].fillna(value=df_mode[x].iloc[0])

# Separación de X e Y
    
Y_tra = df_tra.pop("Class")
X_tra = df_tra
Y_tes = df_tes.pop("Class")
X_tes = df_tes

# Random forest

#clf = RandomForestClassifier()
#clf.fit(X_tra, Y_tra)
