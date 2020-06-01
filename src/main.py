import numpy as np
import sklearn as skl
import pandas as pd

np.random.seed(0)

############ Lectura de los datos
description = "data/adult.names"
attr = []
with open(description, "r") as f:
    for line in f:
        if line.startswith("@attribute"):
            line = line.split()
            attr.append(line[1])

df = pd.read_csv(
    "data/adult.data",
    names=attr,
)


print('Value : Number of diferent values : Number of missing values')
for x in df.keys():
    print(x, ':', len(set(df[x])), ':', len(df[df[x] == '?']))



print(df.shape)
df = df.replace('?', np.nan) # Reemplazamos los valores ? por valores perdidos

elem, cols = df.shape
df_mode=df.mode()
for x in df.columns.values:
    df[x]=df[x].fillna(value=df_mode[x].iloc[0])
