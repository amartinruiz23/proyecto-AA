import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from collections import Counter
import sklearn.pipeline as pl
import sklearn.decomposition as skld
import sklearn.feature_selection as fs
import sklearn.decomposition as skld
import sklearn.preprocessing as sklpre
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import SelectKBest, chi2
np.random.seed(0)
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.pyplot import plot
import warnings
import collections as c
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

ejecutar_cross_validation = 'N' # Variable global para decidir la ejecución de cv en los modelos

# --- Funciones auxiliares ---

def resultados(clf,X,y,X_tst,y_tst,msg=None,):
    # Función para imprimir los resultados de un modelo
    # Parámetros:
    # - clf: el modelo del que queremos obtener los resultados
    # - X: conjunto de training
    # - y: etiquetas del conjunto de training
    # - X_tst: conjunto de test
    # - y_tst: etiquetas del conjunto de test
    # - msg: mensaje que se escribe junto al resultado. Por defecto ninguno
    
    if msg is not None: # Escribe mensaje si lo hubiera
        print(msg)

    pred_tra = clf.predict(X) # Predicción del conjunto tra
    pred_tst = clf.predict(X_tst) # Predicción del conjunto test
    print("Training: \n - Accuracy: ", accuracy_score(y, pred_tra)) # Imprime accuracy en tra
    print("- F1-score: ", f1_score(y, pred_tra, average='macro')) # Imprime f1 en tra
    #    print("precision  tra: ", precision_score(y, pred_tra))
    
    print("Test: \n - Accuracy: ", accuracy_score(y_tst, pred_tst)) # Imprime accuracy en test
    print(" - F1-score: ", f1_score(y_tst, pred_tst, average='macro')) # Imcrime f1 en test
    # print("precision tst: ", precision_score(y_tst, pred_tst))

    # Si se ha pedido obtener los resultados de cv para los modelos, se ejecuta
    if (ejecutar_cross_validation == 'S'):
        cv_results = cross_validate(clf,X,y,cv=5,) # Cross validation con accuracy
        print("E_cv:\n - Accuracy: ", sum(cv_results['test_score']) / len(cv_results['test_score'])) # Imprime resultado
        cv_results = cross_validate(clf,X,y,cv=5,scoring='f1_macro') # cross validation con f1
        print(" - F1-score ",sum(cv_results['test_score']) / len(cv_results['test_score'])) # Imprime resutado

def replace_lost_categorical_values(df):
    # Función para reemplazar valores categóricos perdidos en un dataframe
    # Parámetros
    # - df: el dataframe donde queremos reemplazar
    
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
            if x != '?':
                total += count[x]
                acumulated[x] = total
        acumulated = {k : v / total for k, v in acumulated.items()} # Normalizamos
        
        # Asignamos ahora un valor a los valores perdidos
        for j in range(len(df[i])):
           if df[i][j] == '?': # Para cada valor perdido
               prob = np.random.uniform() # Generamos probabilidad
               less = {}
               for y in acumulated:
                   if prob <= acumulated[y]:
                       less[y] = acumulated[y]
               df[i][j] = min(less, key=less.get) # Le asignamos la clave que tenga el menor valor que sea mayor a la probabilidad 

    return df        

def info_size(df, msg):
    # Imprime información sobre el tamaño del df
    # Parámetros:
    # - df: dataframe del que se quiere obtener información
    # - msg: mensaje para escribir junto a los resultados
    elem, cols = df.shape # Obtención de filas y columnas del df
    print(msg) # Imprime mensaje
    print(' - Numero de datos recopilados:', elem) # Imprime filas
    print(' - Dimension:', cols) # Imprime columnas
    #input("\n--- Pulsar intro para continuar ---\n")

def info(df):
    # Imprime información sobre valores perdidos del df
    # Parámetros:
    # - df: dataframe del que se quiere obtener información
    
    print(' Información general de valores perdidos' )
    clases = ['workclass','occupation','native-country']
    for x in clases:
        print(x, ':', len(set(df[x])), ':', len(df[df[x] == '?']))
    #input("\n--- Pulsar intro para continuar ---\n")
    
def graficas(df):
    # Imprime gráficas sobre el df
    # Parámetros:
    # - df: dataframe del que se quiere obtener información
    
    print_outliers(df) # Imprime gráfica de outliers
    data_relevance(df) # Imprime gráfica de relevancia de datos
    continous_variables_graphs(df) # Imprime gráfica de variables continuas
    correlationMatrix(df) # Imprime matriz de correlación
    
def class_division(filename, attr):
    # Lee un archivo y crea un dataframe con las condiciones necesarias para su posterior tratado
    # Divide entre características y etiquetas
    # Parámetros
    # - filename: nombre del archivo a leer
    # - attr: nombre de lasc olumnas
    
    df = pd.read_csv(
        filename,
    names = attr
    ) # lectura del archivo
    df_mode=df.mode()
    for x in df.columns.values:
        df[x]=df[x].fillna(value=df_mode[x].iloc[0]) # introducción de columnas

    df = replace_lost_categorical_values(df) # sustituye valores perdidos en variables categóricas
    df = df.replace('?', np.nan) # sustituye valores perdidos restantes por nan de numpy
    y = df.pop('Class') # separa clase
    
    return df, y

def print_outliers(X):
    # Imprime gráfica de outliers
    # Parámetros
    # - X: df del que se quieren obtener la información
    plt.figure()
    plt.title(' Selección de outliers' )
    var = X.select_dtypes(include=['int64']).columns
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.boxplot(X[var[i]])
        plt.title(var[i])
    plt.show()

def data_relevance(df):
    # Imprime gráfica de relevancia de datos
    # Parámetros
    # - df: dataframe del que se quieren obtener la información
    
    fig, ((a,b),(c,d)) = plt.subplots(2,2,figsize=(15,20))
    plt.xticks(rotation=45)
    sns.countplot(df['workclass'],hue=df['Class'],ax=a)
    sns.countplot(df['relationship'],hue=df['Class'],ax=b)
    sns.countplot(df['race'],hue=df['Class'],ax=d)
    sns.countplot(df['sex'],hue=df['Class'],ax=c)
    plt.show()

def continous_variables_graphs(df):
    # Imprime gráfica de variables continuas
    # Parámetros
    # - df: dataframe del que se quieren obtener la información
    
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
    # Imprime matriz de correlación
    # Parámetros
    # - df: dataframe del que se quieren obtener la información
    
    plt.figure()
    sns.heatmap(df[df.keys()].corr(),annot=True, fmt = ".2f", cmap = "YlGnBu")
    plt.title("Correlation Matrix")
    plt.show()

def print_class_balance(y,y_tst):
    # Imprime información sobre balanceo de clases
    # Parámetros
    # - y: vector de clases de training
    # - y_tst: vector de clases de test
    
    d = c.defaultdict(int)
    d_tst = c.defaultdict(int)
    for x in y:
        d[int(x)]+=1
    for x in y_tst:
        d_tst[(x)]+=1
    
    print('\nBalanceo de clases:\nClase | n veces Train | n veces Test')
    for x in set(y):
        print(x,'|', d[x], '|' ,d_tst[x])
    #input("\n--- Pulsar intro para continuar ---\n")

def encode_categorical_variables(X, X_tst):  
    # Codifica variables categóricas mediante dummy variables para su correcta utilziación en los modelos
    # Parámetros:
    # - X: datos de training
    # - X_tst: datos de test
    
    X = pd.get_dummies(X)
    X_tst = pd.get_dummies(X_tst)
    
    # Completar training con los valores que no contenga, pero sí test
    missing = set(X_tst.columns) - set(X.columns)
    for i in missing:
        X[i] = 0

    # Completar test con los valores que no contenga, pero sí training
    missing = set(X.columns) - set(X_tst.columns)
    #    print("PERDIDOS2: ", missing)
    for i in missing:
        X_tst[i] = 0
        
    return X, X_tst

# --- Lectura de los datos ---

print("Leyendo datos...")
description = "data/adult.names"

attr = []
with open(description, "r") as f: # Lectura de nombres de columnas
    for line in f:
        if line.startswith("@attribute"):
            line = line.split()
            attr.append(line[1])

X, y = class_division("data/adult.data", attr) # Lectura de datos de training
X_tst, y_tst = class_division("data/adult.test", attr) # Lectura de datos de test
print("Lectura completada...")

print("Codificando variables categóricas...")
X, X_tst = encode_categorical_variables(X, X_tst)
print("Variables categóricas codificadas")
info_size(X, 'Tamaño de los datos después de encode:')

# --- Información sobre balanceo de clases ---
print_class_balance(y, y_tst)

# --- Gráfica de outliers ---

print("Outliers")
print_outliers(X)

# --- Preprocesado --- 

print("Preprocesado de datos...")
preprocesado = pl.make_pipeline(fs.VarianceThreshold(threshold=0.01),
                                sklpre.StandardScaler(),
                                skld.PCA(tol=0.3))

preprocesado.fit(X)
X = preprocesado.transform(X)
X_tst = preprocesado.transform(X_tst)
print()
info_size(X, 'Tamaño de los datos después del preprocesado varianceThreshold:')

input("\n--- Pulsar intro para continuar ---\n")

# --- PCA ---

print( 'Analisis de componentes principales')
pca = PCA(tol=0.1)
pca.fit(X)
print('\n Variabilidad explicada por PCA:\n' ,pca.explained_variance_,'\n\n')
pca = PCA(n_components = 55)
pca.fit(X)
X_tst = pca.transform(X_tst)
X = pca.transform(X)
info_size(X, 'Tamaño de los datos después del preprocesado PCA:')

input("\n--- Pulsar intro para continuar ---\n")

ejecutar_cross_validation = input("¿Desea ejecutar cross validation para los modelos? Puede suponer tiempos de ejecución largos para los más complejos (S/N)" )
# --- Modelo lineal ---

print("\n--- Modelo lineal ---\n")
resp = input( '¿Quiere ejecutar la búsqueda de hiperparámetros y generación de gráficos? Tiempo aproximado 30 min - 1 hora. (S/N)' )

if resp == 'S':
    print('\n\nEstudio de la variabilidad polinómica de los datos')
    X_copy = X.copy()
    for i in range(1,3):
        print('Estudio con dimensión: ',i )
        poly = PolynomialFeatures(i)
        np.random.seed(0)
        poly.fit(X)
        poly.transform(X)
        poly.transform(X_tst)
        clf = LR(random_state=0)
        clf.fit(X, y)
        resultados(clf, X, y, X_tst, y_tst)
        X = X_copy.copy()
    input("\n--- Pulsar intro para continuar ---\n")


    print('\n\nEstudio de la Fuerza de Regularización Lineal (tarda un poco).')

    acu = []
    fsc = []
    x_axis = [i for i in range(-5,10)]
    for i in x_axis:
        clf = LR( penalty='l2', random_state=0, solver='liblinear', C=10**i)
        clf.fit(X, y)
        cv_results = cross_validate(clf,X,y,cv=5)
        acu.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
        cv_results = cross_validate(clf,X,y,cv=5,scoring='f1_macro')
        fsc.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
    
    plt.figure()
    plt.title('Fuerza de regularización')
    plt.xlabel('Valor en escala logaritmica base 10')
    plot(x_axis, acu, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='accuracy')
    plot(x_axis, fsc, 'go',color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='f1-score')
    plt.legend()
    plt.show()

    print('Mejor resultado: ', acu.index(max(acu)))

print('Mejor modelo: ')
clf = LR( penalty='l2', random_state=0, solver='liblinear', C=1.9)
clf.fit(X, y)
resultados(clf, X, y, X_tst, y_tst)

input("\n--- Pulsar intro para continuar ---\n")

# --- Random forest ---

print('\n-- Random Forest --\n')

resp = input( '¿Quiere ejecutar la búsqueda de hiperparámetros y generación de gráficos? Tiempo aproximado 30 min - 1 hora. (S/N)' )

if resp == 'S':
    print('\nComparación de criterios\n')
    for crit in  ['gini', 'entropy']: # Imprime resultados para cada uno de los criterios
        clf = RandomForestClassifier(n_estimators=400, max_depth=50, criterion = crit)
        clf.fit(X, y)
        resultados(clf, X, y, X_tst, y_tst)

    print('\nComparación de profundidad\n')

    e_cv = []
    e_in = []
    x_axis = [80, 60, 40, 30, 20, 10, 5, 3, 1]
    for i in x_axis: # Almacena resultados para cada uno de los posibles valores de profundidad
        clf = RandomForestClassifier(n_estimators=400, max_depth=i)
        clf.fit(X, y)

        cv_results = cross_validate(clf,X,y,cv=5)    
        e_cv.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
    
        pred_tra = clf.predict(X)
        e_in.append(accuracy_score(y, pred_tra))
    
    plt.figure()
    plt.title('Profundidad, máxima cross validation')
    plot(x_axis, e_cv, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e cv')
    plt.show() # Genera gráfica de cv con los valores de profundidad obtenidos

    plt.figure()
    plt.title('Profundidad, resultado en training')
    plot(x_axis, e_in,color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e in')
    plt.show() # Genera gráfica de resultado en training con los valores de profundidad obtenidos

    print('\nComparación de n estimadores\n')

    e_cv = []
    e_in = []
    x_axis = [10, 50, 100, 200, 400, 500, 600]
    for i in x_axis: # Almacena resultados para cada uno de los posibles valores de estimadores
        clf = RandomForestClassifier(n_estimators=i, max_depth=50)
        clf.fit(X, y)
        
        cv_results = cross_validate(clf,X,y,cv=5)    
        e_cv.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
        #print(e_cv)
        
        pred_tra = clf.predict(X)
        e_in.append(accuracy_score(y, pred_tra))
        #print(e_in)
        
    plt.figure()
    plt.title('Estimadores, cross validation')
    plot(x_axis, e_cv, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e cv')
    plt.show() # Genera gráfica de cv con los valores obtenidos

    plt.figure()
    plt.title('Estimadores, resultado en training')
    plot(x_axis, e_in, color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='e in')
    plt.show() # Genera gráfica de resultado en training con los valores obtenidos

print("Mejor modelo: ")
clf = RandomForestClassifier(n_estimators=400, max_depth=50)
clf.fit(X, y)
resultados(clf, X, y, X_tst, y_tst)
input("\n--- Pulsar intro para continuar ---\n")


# --- Support vector machine

print('\n-- Support vector machine --\n')

resp = input( '¿Quiere ejecutar la búsqueda de hiperparámetros y generación de gráficos? Tiempo aproximado 30 min - 1 hora. (S/N)' )

if resp == 'S':

    print('\n\nEstudio de kernel.')

    for kernel in ['rbf','linear']:
        clf = svm.SVC(max_iter=15000, C=2, kernel=kernel)
        clf.fit(X, y)
        resultados(clf, X, y, X_tst, y_tst)

    print('\n\nEstudio de la Fuerza de Regularización Lineal (tarda un poco).')

    acuGaus = []
    fscGaus = []
    x_axis = [i for i in range(-5,6,2)]

    for i in x_axis:
        clf = svm.SVC(max_iter=3000, C=10**(i))
        clf.fit(X, y)

        cv_results = cross_validate(clf,X,y,cv=5)
    
        acuGaus.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
        print(acuGaus)
        cv_results = cross_validate(clf,X,y,cv=5,scoring='f1_macro')
        fscGaus.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
        print(fscGaus)

    plt.figure()
    plt.title('Fuerza de regularización')
    plt.xlabel('Valor en escala logaritmica base 10')
    plot(x_axis, acuGaus, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='accuracy')
    plot(x_axis, fscGaus, 'go',color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='f1-score')
    plt.legend()
    plt.show()

print("Mejor modelo: ")
clf = svm.SVC(max_iter=10000, C=2, kernel='rbf')
clf.fit(X, y)
resultados(clf, X, y, X_tst, y_tst)
input("\n--- Pulsar intro para continuar ---\n")


# --- Perceptrón multi capa ---
print('\n-- Perceptrón multi capa --\n')

resp = input( '¿Quiere ejecutar la búsqueda de hiperparámetros y generación de gráficos? Tiempo aproximado 30 min - 1 hora. (S/N)' )

if resp == 'S':
    e_cv = []
    e_in = []
    x_axis = [50, 60, 80, 100]


    for i in x_axis: # Recorre los valores para la primera capa
        for j in x_axis: # Recorre los valores para la segunda capa
        
            clf = MLPClassifier(hidden_layer_sizes=[i, j]) # Genera el modelo 
            clf.fit(X, y)
    
            cv_results = cross_validate(clf,X,y,cv=5)    
            e_cv.append(sum(cv_results['test_score']) / len(cv_results['test_score']))
            print("cv ", i," ", j, " ", e_cv[-1]) # Imprime resultado de cv
    
            pred_tra = clf.predict(X)
            e_in.append(accuracy_score(y, pred_tra))
            #print(e_in)
            print("e_in ", i," ", j, " ", e_in[-1]) # Imprime resultado en trainingn
        

print("Mejores modelos: ") 

print("Tres capas")
clf = MLPClassifier(hidden_layer_sizes=[50, 60])
clf.fit(X, y)
resultados(clf, X, y, X_tst, y_tst)

print("Dos capas")
clf = MLPClassifier(hidden_layer_sizes=60)
clf.fit(X, y)
resultados(clf, X, y, X_tst, y_tst)
