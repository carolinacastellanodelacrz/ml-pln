#!/usr/bin/env python
# coding: utf-8

# ![image info](https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/images/banner_1.png)

# # Taller: Construcción e implementación de árboles de decisión y métodos de ensamblaje
# 
# En este taller podrá poner en práctica los sus conocimientos sobre construcción e implementación de árboles de decisión y métodos de ensamblajes. El taller está constituido por 9 puntos, 5 relacionados con árboles de decisión (parte A) y 4 con métodos de ensamblaje (parte B).

# ## Parte A - Árboles de decisión
# 
# En esta parte del taller se usará el conjunto de datos de Capital Bikeshare de Kaggle, donde cada observación representa el alquiler de bicicletas durante una hora y día determinado. Para más detalles puede visitar los siguientes enlaces: [datos](https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip), [dicccionario de datos](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset#).

# ### Datos prestamo de bicicletas

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importación de librerías
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz


# In[3]:


bikes = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/bikeshare.csv', index_col='datetime', parse_dates=True)

# "count" to "total"
bikes.rename(columns={'count':'total'}, inplace=True)

# Crear la hora como una variable 
bikes['hour'] = bikes.index.hour

# Visualización de los datos
bikes.head()


# ### Punto 1 - Análisis descriptivo
# 
# Ejecute las celdas 1.1 y 1.2. A partir de los resultados realice un análisis descriptivo sobre las variables "season" y "hour", escriba sus inferencias sobre los datos. Para complementar su análisis puede usar métricas como máximo, mínimo, percentiles entre otros.

# In[4]:


# Celda 1.1
bikes.groupby('season').total.mean()


# In[5]:


# Definiendo un diccionario para nombrar las temporadas
season_names = {1: 'Invierno', 2: 'Primavera', 3: 'Verano', 4: 'Otoño'}

# Estadísticas descriptivas
season_stats = bikes.groupby(bikes['season'].map(season_names))['total'].describe()
print("Estadísticas descriptivas por temporada:")
print(season_stats)


# La variable "season" es categórica, indica la estación del año y va de 1 a 4, donde:
# - 1: invierno
# - 2: primavera
# - 3: verano
# - 4: otoño.
# 
# De este set de datos el que mayor promedio tiene en alquiler de bicicletas es el verano, con una media de 234.41, seguido de la primavera, con una media de 215.25.
# Adicionalmente, podemos observar las estadísticas por temporada: los percentiles dan a conocer la distribución de los datos. En este contexto, en que estamos analizando los alquileres de bicicletas por temporada, los percentiles permiten comprender cómo varía la demanda de alquileres.
# La desviación estándar es una medida de dispersión que indica cuánto varían los alquileres de bicicletas alrededor de la media. En este caso, la desviación estándar durante el invierno es de aproximadamente 125.27, mostrando una cantidad significativa de variabilidad en los alquileres diarios en esta temporada. Sin embargo, esta temporada tiene la menor desviación estándar, contrario a verano, que tiene una desviación de 197.15, seguida de la primavera con una desviación de 192 aproximadamente. Por su parte, el mínimo indica el valor mínimo de alquileres de bicicletas en un día y podemos observar que para todas las estaciones el valor es 1.
# La columna max muestra el valor máximo de alquileres de bicicletas. El valor máximo es de 801 en invierno, el cual es significativa y obviamente menor al resto de las estaciones. En contraste y como es de esperarse, en verano y otoño los máximos son mayores: 977 y 948 respectivamente.

# In[6]:


# Celda 1.2
bikes.groupby('hour').total.mean()


# Entendiendo la hora 0 como las 12am y las 23 como las 11pm, podemos observar cómo varía el número de bicicletas alquiladas a lo largo del día. Se pueden ver horas en que la demanda es mayor, como las 8am y entre las 4 y 7pm. Por su parte, se puede observar también que la demanda es menor en la madrugada luego de media noche, siendo las 4am el horario con menor promedio, con un aproximado de 6 bicicletas alquiladas.

# ### Punto 2 - Análisis de gráficos
# 
# Primero ejecute la celda 2.1 y asegúrese de comprender el código y el resultado. Luego, en cada una de celdas 2.2 y 2.3 escriba un código que genere una gráfica del número de bicicletas rentadas promedio para cada valor de la variable "hour" (hora) cuando la variable "season" es igual a 1 (invierno) e igual a 3 (verano), respectivamente. Analice y escriba sus hallazgos.

# In[7]:


# Celda 2.1 - rentas promedio para cada valor de la variable "hour"
bikes.groupby('hour').total.mean().plot()


# In[8]:


# Celda 2.2 - "season"=1 escriba su código y hallazgos 
bikes[bikes['season'] == 1].groupby('hour').total.mean().plot()


# Podemos observar en la gráfica que para el invierno el horario con menor demanda de bicicletas es entre las 12 de la madrugada y las 5am. A partir de las 5 se ve un crecimiento en la misma, y esto puede estar relacionado a la alta temperatura a la que puede llegar en esta época. Vemos picos entre aproximadamente las 7-8am y luego nuevamente a las 5-6pm. Los mínimos oscilan entre 0 y 50, mientras que los picos que indican mayor demanda parecen estar alrededor de los 250.

# In[9]:


# Celda 2.3 - "season"=3 escriba su código y hallazgos 
bikes[bikes['season'] == 3].groupby('hour').total.mean().plot()


# Similarmente, para el horario del verano, también encontramos mínimos en las demandas de bicicletas en las horas de la madrugada. Sin embargo, los picos tienen ahora valores más altos debido a que la demanda es mayor en esta época del año, y van desde 400 y superan los 500 para las 8-9am aproximadamente, y 5-6pm respectivamente.

# ### Punto 3 - Regresión lineal
# En la celda 3 ajuste un modelo de regresión lineal a todo el conjunto de datos, utilizando "total" como variable de respuesta y "season" y "hour" como las únicas variables predictoras, teniendo en cuenta que la variable "season" es categórica. Luego, imprima los coeficientes e interprételos. ¿Cuáles son las limitaciones de la regresión lineal en este caso?

# In[10]:


bikes.head()


# In[11]:


# Celda 3
# Convirtiendo a dummy la variable categórica 'season'
season_dummies = pd.get_dummies(bikes['season'], prefix='season')

# Luego c/u en 0s y 1s
season_dummies = season_dummies.astype(bool).astype(int)

# Reemplazamos las dummies en el df bikes_2 para no modificar el original
bikes_2 = bikes.copy() 
bikes_2 = pd.concat([bikes_2.drop(columns='season'), season_dummies], axis=1)

# Separamos x's y y
X_2 = bikes_2[['hour'] + season_dummies.columns.tolist()]  # Variables predictoras (Xs)
y = bikes_2['total']  # Variable de respuesta (Y)

# Ajuste del modelo
model = LinearRegression()
model.fit(X_2, y)

# Coef
print("Coeficientes del modelo:")
print("Intercepto:", model.intercept_)
print("Coeficientes de 'hour':", model.coef_[0])
print("Coeficientes de 'season':", model.coef_[1:])


# In[12]:


# En esta celda intenté modelar X sin hacer season una dummy, sino dejando sus valores originales. Sin embargo,
# la interpretación no está correcta puesto que se toman los valores de "season" como numéricos.
X = bikes[['hour', 'season']]
y = bikes['total']

model = LinearRegression()
model.fit(X, y)

# Coeficientes
print("Coeficientes del modelo:")
print("Intercepto:", model.intercept_)
print("Coeficientes de 'hour':", model.coef_[0])
print("Coeficientes de 'season':", model.coef_[1:])


# Interpretración:
# 
# El intercepto representa el número esperado de alquileres de bicicleta cuando todas las variables predictoras son cero. En este caso, tiene un signo negativo, es decir, se podría decir que sin las variables "hour" y "season", el número "base" de alquileres esperados es de -8225.
# El coeficiente de "hour" es de 10.54 e indica el cambio esperado en el número de alquileres por un incremento unitario en la variable hora.
# 
# Posibles limitaciones de la regresión lineal en este caso:
# - Los valores obtenidos en los coeficientes parecen estar incorrectos, por lo que la interpretación se puede ver afectada. El coeficiente del intercepto y el de season parecen no estar normalizados, tener multicolinealidad o estar sobreajustados.
# - Si la base de datos contiene outliers podría también verse afectada debido a su sensibilidad ante los mismos.

# In[13]:


bikes.head()


# ### Punto 4 - Árbol de decisión manual
# En la celda 4 cree un árbol de decisiones para pronosticar la variable "total" iterando **manualmente** sobre las variables "hour" y  "season". El árbol debe tener al menos 6 nodos finales.

# In[14]:


# Celda 4
print(bikes.columns)


# In[15]:


print(X.columns)


# Construcción de árbol. Para la primera variable "season" se calculan los posibles puntos de corte y el gini index de un punto especifico

# In[16]:


j=0
print(bikes.columns[j])


# In[17]:


# Definición de parámetros y criterios de parada
max_depth = None
num_pct = 10
min_gain = 0.001


# In[18]:


# División de la variable season en num_ctp puntos (parámetro definido anteriormente) para obtener posibles puntos de corte
splits = np.percentile(X.iloc[:, j], np.arange(0, 100, 100.0 / num_pct).tolist())
splits = np.unique(splits)
splits


# In[19]:


# División de las observaciones usando el punto de corte en la posición 5 de la lista de splits
k=5
filter_l = X.iloc[:, j] < splits[k]

# División de la variable de respuesta de acuerdo a si la observación cumple o no con la regla binaria
# y_l: la observación tiene un valor menor al punto de corte seleccionado
# y_r: la observación tiene un valor mayor o igual al punto de corte seleccionado
y_l = y.loc[filter_l]
y_r = y.loc[~filter_l]


# In[20]:


# Definición de la función que calcula el gini index
def gini(y):
    if y.shape[0] == 0:
        return 0
    else:
        return 1 - (y.mean()**2 + (1 - y.mean())**2)


# In[21]:


# Gini index de las observaciones que tienen un valor menor al punto de corte seleccionado
gini_l = gini(y_l)
gini_l


# In[22]:


# Gini index de las observaciones que tienen un valor mayor o igual al punto de corte seleccionado
gini_r = gini(y_r)
gini_r


# In[23]:


# Definición de la función gini_imputiry para calular la ganancia de una variable predictora j dado el punto de corte k
def gini_impurity(X_col, y, split):
    
    filter_l = X_col < split
    y_l = y.loc[filter_l]
    y_r = y.loc[~filter_l]
    
    n_l = y_l.shape[0]
    n_r = y_r.shape[0]
    
    gini_y = gini(y)
    gini_l = gini(y_l)
    gini_r = gini(y_r)
    
    gini_impurity_ = gini_y - (n_l / (n_l + n_r) * gini_l + n_r / (n_l + n_r) * gini_r)
    
    return gini_impurity_


# In[24]:


# Ganancia de la variable 'Hits' en el punto de corte selecionado
gini_impurity(X.iloc[:, j], y, splits[k])


# In[25]:


# Definición de la función best_split para calcular cuál es la mejor variable y punto de cortepara hacer la bifurcación del árbol
def best_split(X, y, num_pct=10):
    
    features = range(X.shape[1])
    
    best_split = [0, 0, 0]  # j, split, gain
    
    # Para todas las varibles 
    for j in features:
        
        splits = np.percentile(X.iloc[:, j], np.arange(0, 100, 100.0 / (num_pct+1)).tolist())
        splits = np.unique(splits)[1:]
        
        # Para cada partición
        for split in splits:
            gain = gini_impurity(X.iloc[:, j], y, split)
                        
            if gain > best_split[2]:
                best_split = [j, split, gain]
    
    return best_split


# In[26]:


# Obtención de la variable 'j', su punto de corte 'split' y su ganancia 'gain'
j, split, gain = best_split(X, y, 5)
j, split, gain


# In[27]:


# División de las observaciones usando la mejor variable 'j' y su punto de corte 'split'
filter_l = X.iloc[:, j] < split

y_l = y.loc[filter_l]
y_r = y.loc[~filter_l]


# In[28]:


y.shape[0], y_l.shape[0], y_r.shape[0]


# In[29]:


y.mean(), y_l.mean(), y_r.mean()


# In[30]:


# Definición de la función tree_grow para hacer un crecimiento recursivo del árbol
def tree_grow(X, y, level=0, min_gain=0.001, max_depth=None, num_pct=10):
    
    # Si solo es una observación
    if X.shape[0] == 1:
        tree = dict(y_pred=y.iloc[:1].values[0], y_prob=0.5, level=level, split=-1, n_samples=1, gain=0)
        return tree
    
    # Calcular la mejor división
    j, split, gain = best_split(X, y, num_pct)
    
    # Guardar el árbol y estimar la predicción
    y_pred = int(y.mean() >= 0.5) 
    y_prob = (y.sum() + 1.0) / (y.shape[0] + 2.0)  # Corrección Laplace 
    
    tree = dict(y_pred=y_pred, y_prob=y_prob, level=level, split=-1, n_samples=X.shape[0], gain=gain)
    # Revisar el criterio de parada 
    if gain < min_gain:
        return tree
    if max_depth is not None:
        if level >= max_depth:
            return tree   
    
    # Continuar creando la partición
    filter_l = X.iloc[:, j] < split
    X_l, y_l = X.loc[filter_l], y.loc[filter_l]
    X_r, y_r = X.loc[~filter_l], y.loc[~filter_l]
    tree['split'] = [j, split]

    # Siguiente iteración para cada partición
    
    tree['sl'] = tree_grow(X_l, y_l, level + 1, min_gain=min_gain, max_depth=max_depth, num_pct=num_pct)
    tree['sr'] = tree_grow(X_r, y_r, level + 1, min_gain=min_gain, max_depth=max_depth, num_pct=num_pct)
    
    return tree


# In[31]:


# Aplicación de la función tree_grow
tree_grow(X, y, level=0, min_gain=0.001, max_depth=1, num_pct=10)


# In[32]:


tree = tree_grow(X, y, level=0, min_gain=0.001, max_depth=3, num_pct=10)
tree


# In[33]:


# Definición de la función tree_predict para hacer predicciones según las variables 'X' y el árbol 'tree'

def tree_predict(X, tree, proba=False):
    
    predicted = np.ones(X.shape[0])

    # Revisar si es el nodo final
    if tree['split'] == -1:
        if not proba:
            predicted = predicted * tree['y_pred']
        else:
            predicted = predicted * tree['y_prob']
            
    else:
        
        j, split = tree['split']
        filter_l = (X.iloc[:, j] < split)
        X_l = X.loc[filter_l]
        X_r = X.loc[~filter_l]

        if X_l.shape[0] == 0:  # Si el nodo izquierdo está vacio solo continua con el derecho 
            predicted[~filter_l] = tree_predict(X_r, tree['sr'], proba)
        elif X_r.shape[0] == 0:  #  Si el nodo derecho está vacio solo continua con el izquierdo
            predicted[filter_l] = tree_predict(X_l, tree['sl'], proba)
        else:
            predicted[filter_l] = tree_predict(X_l, tree['sl'], proba)
            predicted[~filter_l] = tree_predict(X_r, tree['sr'], proba)

    return predicted

tree_predict(X, tree)


# ## Utilizando el paquete de sklearn

# In[34]:


max_depth_range = range(1, 21)
# Lista para guardar los valores del RMSE para cada valor de máxima profundidad (max_depth)
accuracy_scores = []

# Importación de modelos de sklearn 
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Loop para obtener el desempeño del modelo de acuerdo con la máxima profundidad
for depth in max_depth_range:
    # Definición del árbol de decisión usando DecisionTreeClassifier de la libreria sklearn
    clf = DecisionTreeClassifier(max_depth=depth, random_state=1)
    accuracy_scores.append(cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean())


# In[35]:


# Mejor accuracy (desempeño del modelo) y su correspondiente max_depth
sorted(zip(accuracy_scores, max_depth_range))[::-1][0]


# In[36]:


# Función para calcular la impureza Gini
def gini(labels):
    if len(labels) == 0:
        return 0
    else:
        p = (labels == 1).mean()
        return 1 - p ** 2 - (1 - p) ** 2

# Función para calcular la ganancia de información
def information_gain(y, y_l, y_r):
    p_l = len(y_l) / len(y)
    p_r = len(y_r) / len(y)
    return gini(y) - (p_l * gini(y_l) + p_r * gini(y_r))

# Función para encontrar la mejor división
def find_best_split(X, y):
    best_gain = 0
    best_feature = None
    best_value = None
    
    for j in range(X.shape[1]):
        splits = np.percentile(X[:, j], np.arange(0, 100, 100.0 / (num_pct+1)).tolist())
        splits = np.unique(splits)[1:]
        
        for split in splits:
            filter_l = X[:, j] < split
            y_l = y[filter_l]
            y_r = y[~filter_l]
            
            gain = information_gain(y, y_l, y_r)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_value = split
                
    return best_feature, best_value


# ### Punto 5 - Árbol de decisión con librería
# En la celda 5 entrene un árbol de decisiones con la **librería sklearn**, usando las variables predictoras "season" y "hour" y calibre los parámetros que considere conveniente para obtener un mejor desempeño. Recuerde dividir los datos en conjuntos de entrenamiento y validación para esto. Comente el desempeño del modelo con alguna métrica de desempeño de modelos de regresión y compare desempeño con el modelo del punto 3.

# In[37]:


# Celda 5
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Divisón de train-test
X = bikes[['season', 'hour']]  # Variables predictoras
y = bikes['total']              # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árboles de decisión
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del desempeño del modelo:
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[38]:


from sklearn.metrics import accuracy_score
# Convertimos las predicciones a valores enteros 
y_pred_int = y_pred.astype(int)
# Evaluar el desempeño del modelo mediante accuracy
accuracy = accuracy_score(y_test, y_pred_int)
print("Accuracy Score:", accuracy)


# Parece que el modelo de árbol de decisiones tiene un bajo rendimiento en términos de precisión y un alto error cuadrático medio (MSE). Esto sugiere que el modelo no está capturando bien la relación entre las variables predictoras y la variable objetivo. Es posible que se necesiten ajustes adicionales en el modelo o que se deban considerar otros enfoques de modelado para mejorar su desempeño.

# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# Gráfica max_depth versus RMSE (error del modelo)
plt.plot(max_depth_range, accuracy_scores)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')


# ## Parte B - Métodos de ensamblajes
# En esta parte del taller se usará el conjunto de datos de Popularidad de Noticias Online. El objetivo es predecir si la notica es popular o no, la popularidad está dada por la cantidad de reacciones en redes sociales. Para más detalles puede visitar el siguiente enlace: [datos](https://archive.ics.uci.edu/ml/datasets/online+news+popularity).

# ### Datos popularidad de noticias

# In[40]:


# Lectura de la información de archivo .csv
df = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/mashable.csv', index_col=0)
df.head()


# In[41]:


df.info()


# In[42]:


# Definición variable de interés y variables predictoras
X = df.drop(['url', 'Popular'], axis=1)
y = df['Popular']
y.mean()


# In[43]:


X


# In[44]:


y


# In[45]:


# División de la muestra en set de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# ### Punto 6 - Árbol de decisión y regresión logística
# En la celda 6 construya un árbol de decisión y una regresión logística. Para el árbol calibre al menos un parámetro y evalúe el desempeño de cada modelo usando las métricas de Accuracy y F1-Score.

# In[58]:


# Celda 6
models_for_use = {'lr': LinearRegression(),
          'dt': DecisionTreeRegressor(),
                 'logr': LogisticRegression()}


# In[59]:


# Entrenamiento de cada modelo
for model in models_for_use.keys():
    models_for_use[model].fit(X_train, y_train)


# In[60]:


# Predicción de las observaciones del set de test para cada modelo
y_pred = pd.DataFrame(index=X_test.index, columns=models_for_use.keys())
for model in models_for_use.keys():
    y_pred[model] = models_for_use[model].predict(X_test)


# In[64]:


from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

for model in models_for_use.keys():
    if isinstance(models_for_use[model], LinearRegression):
        mse = mean_squared_error(y_pred[model], y_test)
        print(f"Modelo {model} - MSE: {mse}")
    elif isinstance(models_for_use[model], DecisionTreeRegressor):
        y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred[model]]
        accuracy = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        print(f"Modelo {model} - Accuracy: {accuracy}, F1-score: {f1}")
    elif isinstance(models_for_use[model], LogisticRegression):
        accuracy = accuracy_score(y_test, y_pred[model])
        f1 = f1_score(y_test, y_pred[model])
        print(f"Modelo {model} - Accuracy: {accuracy}, F1-score: {f1}")


# Se creó un diccionario que contuviese los modelos a correr. Posteriormente, se iteró sobre cada modelo en este diccionario y mediante el método "fit" se entrenó el modelo con los datos de entrenamiento X_train y y_train.
# Luego se creó un df para las después guardar las predicciones de cada modelo: y_pred.
# Finalmente, para evaluar el desempeño se corrieron el acurracy (predicción) y el F1-Score. Decidí además compararlos con el modelo de RL, por lo que también se encuentra en el loop for. Por favor omitir.
# 
# 
# Podemos entonces inferir entonces que:
# - Modelo de árbol de decisión: el accuracy mide la proporción que está correcta de las predicciones realizadas. Debido a que esta es de 0.54, no se puede argumentar que el modelo esté aprendiendo, puesto que el valor es bajo. De igual manera el F1-score es de 0.53, lo cual puede indicar que este modelo no está funcionando de manera óptima para el conjunto de datos.
# - Modelo de regresión logística: el accuracy es mayor al del anterior modelo, 0.61 y el F1-score también, 0.60. Debido a que estás métricas evaluan el desempeño del modelo, se podría decir que el modelo de regresión logística tiene mejores predicciones (más precisas) para los datos, que el modelo de árbol de decisión.

# ### Punto 7 - Votación Mayoritaria
# En la celda 7 elabore un esamble con la metodología de **Votación mayoritaria** compuesto por 300 muestras bagged donde:
# 
# -las primeras 100 muestras vienen de árboles de decisión donde max_depth tome un valor de su elección\
# -las segundas 100 muestras vienen de árboles de decisión donde min_samples_leaf tome un valor de su elección\
# -las últimas 100 muestras vienen de regresiones logísticas
# 
# Evalúe cada uno de los tres modelos de manera independiente utilizando las métricas de Accuracy y F1-Score, luego evalúe el ensamble de modelos y compare los resultados. 
# 
# Nota: 
# 
# Para este ensamble de 300 modelos, deben hacer votación mayoritaria. Esto lo pueden hacer de distintas maneras. La más "fácil" es haciendo la votación "manualmente", como se hace a partir del minuto 5:45 del video de Ejemplo práctico de emsablajes en Coursera. Digo que es la más fácil porque si hacen la votación mayoritaria sobre las 300 predicciones van a obtener lo que se espera.
# 
# Otra opción es: para cada uno de los 3 tipos de modelos, entrenar un ensamble de 100 modelos cada uno. Predecir para cada uno de esos tres ensambles y luego predecir como un ensamble de los 3 ensambles. La cuestión es que la votación mayoritaria al usar los 3 ensambles no necesariamente va a generar el mismo resultado que si hacen la votación mayoritaria directamente sobre los 300 modelos. Entonces, para los que quieran hacer esto, deben hacer ese último cálculo con cuidado.
# 
# Para los que quieran hacerlo como ensamble de ensambles, digo que se debe hacer el ensamble final con cuidado por lo siguiente. Supongamos que:
# 
# * para los 100 árboles del primer tipo, la votación mayoritaria es: 55% de los modelos predicen que la clase de una observación es "1"
# * para los 100 árboles del segundo tipo, la votación mayoritaria es: 55% de los modelos predicen que la clase de una observación es "1"
# * para las 100 regresiones logísticas, la votación mayoritaria es: 10% de los modelos predicen que la clase de una observación es "1"
# 
# Si se hace la votación mayoritaria de los 300 modelos, la predicción de esa observación debería ser: (100*55%+100*55%+100*10%)/300 = 40% de los modelos votan porque la predicción debería ser "1". Es decir, la predicción del ensamble es "0" (dado que menos del 50% de modelos predijo un 1).
# 
# Sin embargo, si miramos cada ensamble por separado, el primer ensamble predice "1", el segundo ensamble predice "1" y el último ensamble predice "0". Si hago votación mayoritaria sobre esto, la predicción va a ser "1", lo cual es distinto a si se hace la votación mayoritaria sobre los 300 modelos.

# In[65]:


# Celda 7
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

# Entrenamiento de 100 modelos de árboles de decisión con max_depth=3
tree_max_depth = DecisionTreeClassifier(max_depth=3)
bagged_tree_max_depth = BaggingClassifier(base_estimator=tree_max_depth, n_estimators=100)
bagged_tree_max_depth.fit(X_train, y_train)

# Entrenamiento de 100 modelos de árboles de decisión con min_samples_leaf=10
tree_min_samples_leaf = DecisionTreeClassifier(min_samples_leaf=10)
bagged_tree_min_samples_leaf = BaggingClassifier(base_estimator=tree_min_samples_leaf, n_estimators=100)
bagged_tree_min_samples_leaf.fit(X_train, y_train)

# Entrenamiento de 100 modelos de regresión logística
logistic_regression = LogisticRegression()
bagged_logistic_regression = BaggingClassifier(base_estimator=logistic_regression, n_estimators=100)
bagged_logistic_regression.fit(X_train, y_train)

# Predicción por modelo
predictions_tree_max_depth = bagged_tree_max_depth.predict(X_test)
predictions_tree_min_samples_leaf = bagged_tree_min_samples_leaf.predict(X_test)
predictions_logistic_regression = bagged_logistic_regression.predict(X_test)

# Votación mayoritaria
final_predictions = []

for i in range(len(X_test)):
    votos = [predictions_tree_max_depth[i], predictions_tree_min_samples_leaf[i], predictions_logistic_regression[i]]
    final_prediction = max(set(votos), key=votos.count)
    final_predictions.append(final_prediction)

# Métricas: accuracy y f1
accuracy = accuracy_score(y_test, final_predictions)
f1 = f1_score(y_test, final_predictions)

print("Métrica accuracy del ensamble con votación mayoritaria:", accuracy)
print("Métrica F1-Score del ensamble con votación mayoritaria:", f1)


# ### Punto 8 - Votación Ponderada
# En la celda 8 elabore un ensamble con la metodología de **Votación ponderada** compuesto por 300 muestras bagged para los mismos tres escenarios del punto 7. Evalúe los modelos utilizando las métricas de Accuracy y F1-Score

# In[66]:


# Celda 8

# Entrenamiento de 100 modelos de árboles de decisión con max_depth=5
tree_max_depth = DecisionTreeClassifier(max_depth=10)
bagged_tree_max_depth = BaggingClassifier(base_estimator=tree_max_depth, n_estimators=100)
bagged_tree_max_depth.fit(X_train, y_train)

# Entrenamiento de 100 modelos de árboles de decisión con min_samples_leaf elegido por nosotros
tree_min_samples_leaf = DecisionTreeClassifier(min_samples_leaf=10)
bagged_tree_min_samples_leaf = BaggingClassifier(base_estimator=tree_min_samples_leaf, n_estimators=100)
bagged_tree_min_samples_leaf.fit(X_train, y_train)

# Entrenamiento de 100 modelos de regresión logística
logistic_regression = LogisticRegression()
bagged_logistic_regression = BaggingClassifier(base_estimator=logistic_regression, n_estimators=100)
bagged_logistic_regression.fit(X_train, y_train)

# Predicciones de cada modelo
predictions_tree_max_depth = bagged_tree_max_depth.predict(X_test)
predictions_tree_min_samples_leaf = bagged_tree_min_samples_leaf.predict(X_test)
predictions_logistic_regression = bagged_logistic_regression.predict(X_test)

# Calcular los pesos de cada modelo
weights = []

accuracy_tree_max_depth = accuracy_score(y_test, predictions_tree_max_depth)
accuracy_tree_min_samples_leaf = accuracy_score(y_test, predictions_tree_min_samples_leaf)
accuracy_logistic_regression = accuracy_score(y_test, predictions_logistic_regression)

total_accuracy = accuracy_tree_max_depth + accuracy_tree_min_samples_leaf + accuracy_logistic_regression
weight_tree_max_depth = accuracy_tree_max_depth / total_accuracy
weight_tree_min_samples_leaf = accuracy_tree_min_samples_leaf / total_accuracy
weight_logistic_regression = accuracy_logistic_regression / total_accuracy

# Votación ponderada
final_predictions_weighted = []

for i in range(len(X_test)):
    weighted_votes = [
        predictions_tree_max_depth[i] * weight_tree_max_depth,
        predictions_tree_min_samples_leaf[i] * weight_tree_min_samples_leaf,
        predictions_logistic_regression[i] * weight_logistic_regression
    ]
    final_prediction_weighted = round(sum(weighted_votes))
    final_predictions_weighted.append(final_prediction_weighted)

# Métricas de evaluación
accuracy_weighted = accuracy_score(y_test, final_predictions_weighted)
f1_weighted = f1_score(y_test, final_predictions_weighted)

print("Métrica accuracy del ensamble con votación ponderada:", accuracy_weighted)
print("Métrica F1-Score del ensamble con votación ponderada:", f1_weighted)


# ### Punto 9 - Comparación y análisis de resultados
# En la celda 9 comente sobre los resultados obtenidos con las metodologías usadas en los puntos 7 y 8, compare los resultados y enuncie posibles ventajas o desventajas de cada una de ellas.

# #### Celda 9
# En el punto 7, se utilizó la votación mayoritaria para combinar las predicciones de 2 tipos de modelos: árboles de decisión con diferentes parámetros (max_depth y min_samples_leaf) y regresión logística. La precisión del ensamble con votación mayoritaria fue de aproximadamente 0.649 y el F1-Score fue de aproximadamente 0.643.
# 
# En el punto 8, se implementó la votación ponderada para combinar las predicciones de los dos tipos de modelos: árboles de decisión con diferentes parámetros (max_depth y min_samples_leaf) y regresión logística. La precisión del ensamble con votación ponderada fue de aproximadamente 0.657 y el F1-Score fue de aproximadamente 0.662.
# 
# Comparando los resultados, observamos que el ensamble con votación ponderada obtuvo una precisión superior y un F1-Score mayor en comparación con el ensamble con votación mayoritaria. Esto sugiere que asignar pesos a las predicciones de cada modelo en función de su desempeño puede mejorar (aunque ligeramente) el rendimiento del ensamble.
# 
# En cuanto a ventajas y desventajas:
# 
# Votación mayoritaria:
# - Es facil de implementar, completer e interpretrar.
# - No requiere cálculo de pesos. Sin embargo, el hecho de que asigne el mismo peso a todas las predicciones, puede causar que se sobre-estime el desempeño de algún modelo sobre otro equívocamente.
# 
# Votación ponderada:
# - Permite asignar pesos a cada modelo en función de su desempeño, lo que puede mejorar la precisión del ensamble. Asimismo, tiene en cuenta la calidad de las predicciones de cada modelo al combinarlas. Sin embargo este requiere un proceso de cálculo que puede suponer mayor complejidad al momento del cálculo de cada modelo. 
# - Cualquier error en la ponderación podría traducirse en un declive en la efectividad del ensamble.

# In[ ]:




