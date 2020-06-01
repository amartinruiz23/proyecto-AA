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

df = df.replace('?', np.nan) # Reemplazamos los valores ? por valores perdidos

elem, cols = df.shape
print(df)

for x in df.keys():
    print(x, ':', len(set(df[x])), ':', len(df[df[x] == '?']))

