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

elem, cols = df.shape
print(df)
