import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from collections import Counter
from sklearn.preprocessing import StandardScaler


# --- Funciones auxiliares ---


def replace_lost_categorical_values(df):
    
    # Obtenemos las columnas con variables categóricas
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    ind = list(set(cols) - set(num_cols))
    
    # Iteramos sobre estas columnas para calcular su distribución de probabilidad
    # y asignar un valor a los valores perdidos.
    
    for i in ind:
        count = Counter(df[i]) # Cuenta las apariciones de cada clase
        acumulated = {} # Diccionario en el que almacenaremos los valores acumulados
        total = 0
        for x in count: # Rellenamos el diccionario con los valores acumulados
            if x != np.nan:
                total += count[x]
                acumulated[x] = total
        acumulated = {k : v / total for k, v in acumulated.items()} # Normalizamos
        
        # Asignamos ahora un valor a los valores perdidos
        for j in range(len(df[i])):
           if df[i][j] == np.nan: # Para cada valor perdido
               prob = np.random.uniform() # Generamos probabilidad
               less = {}
               for y in acumulated:
                   if prob <= acumulated[y]:
                       less[y] = acumulated[y]
               df[i][j] = min(less, key=less.get) # Le asignamos la clave que tenga el menor valor que sea mayor a la probabilidad 

    return df        

def class_division(filename, attr):
    df = pd.read_csv(
        filename,
    names = attr
    )
    df_mode=df.mode()
    for x in df.columns.values:
        df[x]=df[x].fillna(value=df_mode[x].iloc[0])

    elem, cols = df.shape
    print('Lectura de los datos realizada.')
    print(' - Numero de datos recopilados:', elem)
    print(' - Dimension de estos datos (con la variable de clase):', cols)
    print(' Información general del dataset' )
    df = df.replace('?', np.nan)
    print(df.info())

    input("\n--- Pulsar tecla para continuar ---\n")
    
    df = replace_lost_categorical_values(df)
    #df.to_csv('prueba.csv')

    y = df.pop("Class")
    df = pd.get_dummies(df)
    return df, y

def normalize(df):
    scaler = StandardScaler()
    x = df.values
    scaled = scaler.fit_transform(x)
    df2=pd.DataFrame(scaled, columns=df.columns)
    # print(df2)
    return df2