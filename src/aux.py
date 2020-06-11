import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
from sklearn.model_selection import train_test_split
import collections as c

#warnings.filterwarnings('ignore')

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

def info_size(df, msg):
    elem, cols = df.shape
    print(msg)
    print(' - Numero de datos recopilados:', elem)
    print(' - Dimension:', cols)
    input("\n--- Pulsar tecla para continuar ---\n")

def info(df):
    print(' Información general de valores perdidos' )
    clases = ['workclass','occupation','native-country']
    for x in clases:
        print(x, ':', len(set(df[x])), ':', len(df[df[x] == '?']))
    input("\n--- Pulsar tecla para continuar ---\n")
    

def gráficas(df):
    print_outliers(df)
    data_relevance(df)
    continous_variables_graphs(df)
    correlationMatrix(df)
    input("\n--- Pulsar tecla para continuar ---\n")
    
def class_division(filename, attr):
    df = pd.read_csv(
        filename,
    names = attr
    )
    df_mode=df.mode()
    for x in df.columns.values:
        df[x]=df[x].fillna(value=df_mode[x].iloc[0])

    # info(df)
    df = df.replace('?', np.nan)

    
    df = replace_lost_categorical_values(df)
    #df.to_csv('prueba.csv')
    y = df.pop('Class')
    print(y.shape, df.shape)
    df = pd.get_dummies(df)
    return train_test_split(df, y, test_size=0.2, random_state=42)



def print_outliers(df):
    plt.figure()
    variables = df.select_dtypes(include=['int64']).columns

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.boxplot(df[variables[i]])
        plt.title(variables[i])
    plt.show()

def data_relevance(df):
    fig, ((a,b),(c,d)) = plt.subplots(2,2,figsize=(15,20))
    plt.xticks(rotation=45)
    sns.countplot(df['workclass'],hue=df['Class'],ax=a)
    sns.countplot(df['relationship'],hue=df['Class'],ax=b)
    sns.countplot(df['race'],hue=df['Class'],ax=d)
    sns.countplot(df['sex'],hue=df['Class'],ax=c)
    plt.show()

def continous_variables_graphs(df):
    con_var=['age', 'fnlwgt', 'education-num','hours-per-week']

    plt.figure(figsize=(15,10))
    plt.subplot(221)
    
    i=0
    for x in con_var:
        plt.subplot(2, 2, i+1)
        i += 1
        ax1=sns.kdeplot(df[df['Class'] == '0'][x], shade=True,label="income <=50K")
        sns.kdeplot(df[df['Class'] == 1][x], shade=False,label="income >50K", ax=ax1)
        plt.title(x,fontsize=15)

    plt.show()
    
def correlationMatrix(df):
    plt.figure()
    sns.heatmap(df[df.keys()].corr(),annot=True, fmt = ".2f", cmap = "YlGnBu")
    plt.title("Correlation Matrix")
    plt.show()

def balanceo_clases(y,y_tst):
    d = c.defaultdict(int)
    d_tst = c.defaultdict(int)
    for x in y:
        d[int(x)]+=1
    for x in y_tst:
        d_tst[(x)]+=1
    
    print('\nBalanceo de clases:\nClase | n veces Train | n veces Test')
    for x in set(y):
        print(x,'|', d[x], '|' ,d_tst[x])
    input("\n--- Pulsar tecla para continuar ---\n")
