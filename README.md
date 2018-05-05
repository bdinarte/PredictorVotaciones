# Predictor de Votaciones CR

El objetivo de este software es entrenar y evaluar distintos modelos de clasificación. Para lograrlo se hace uso de un conjunto de `votantes` generados con base en los indicadores cantonales del estado de la nación del 2011 y de los escrutinios de votos para presidencia del 2018. 

## Instalación

### Requisitos Previos

**Nota**: Si se trabaja con Windows 7 o posterior, se debe contar con la versión `3.5.x` o `3.6.x` de Python de 64-bit.

1. Abrir la consola de comandos con permisos de administrador.

3. Instalar la versión adecuada de Tensorflow, `cpu` o `gpu` mediante alguno de los siguientes comandos según corresponda:

    > pip install --upgrade tensorflow

    > pip install --upgrade tensorflow-gpu


## Manual de Uso

#### Ejecución del árbol de decisión
- Primero que todo, al ejecutar el modelo se debe tener claro que es necesario  proporcionar ciertos parámetros para ajustar el comportamiento del modelo a generar. En este caso son necesarios los que acontecen en el ejemplo:
>--arbol
--umbral-poda
0.1
--prefijo
arbolres
--poblacion
1000
--porcentaje-pruebas
10
--k-segmentos
10

Explicación de los parámetros:
- **--arbol** es el flag para indicar que se debe usar el modelo de clasificación con árboles de decisión.
- **--umbral-poda** ajusta el umbral de poda a utilizar con el árbol de decisión.
- **--prefijo** es el nombre del archivo resultante que se plasmará en el documento final con las predicciones.
- **--poblacion** es el tamaño de la muestra que se va generar para realizar el entrenamiento, la validación y las pruebas.
- **k-segmentos** indica los k segmentos en que se dividirá el set de entrenamiento, para realizar el proceso de cross-validation.

## Reportes de Métodos Implementados
### Clasificación basada en modelos lineales

### Clasificación basada en redes neuronales

### Clasificación basada árboles de decisión

Antes de abordar el tema del modelo basado en árboles de decisión, es fundamental tener una perspectiva clara de lo que implica dicho modelo.
Un árbol de decisión toma como entrada un objeto o una situación descrita a través de un conjunto de atributos y devuelve una **decisión**: el valor previsto de la salida dada la entrada. 
- Los atributos de entrada pueden ser discretos o continuos.
- El valor de la salida puede ser a su vez discreto o continuo

Además, se debe tener claro que aprender una función de valores discretos se denomina clasificación y aprender una función continua se denomina regresión. En este caso las clasificaciones no son booleanas, ya que dicha clasificación puede tomar una de las 13 etiquetas que representan a los partidos políticos.
Algunas de las características mencionadas en el libro de IA, A Modern Approach son las que se mencionan a continuación:
- Un árbol de decisión desarrolla una secuencia de test para poder alcanzar una decisión.
- Cada nodo interno del árbol corresponde con un test sobre el valor de una de las propiedades, y las ramas que salen del nodo están etiquetadas con los posibles valores de dicha propiedad.
- Cada nodo hoja del árbol representa el valor que ha de ser devuelto
si dicho nodo hoja es alcanzado.

---
#### Algoritmo de aprendizaje
Ahora dejando de lado la teoría, se procede a enfocar la atención en la sección del algoritmo que se debe emplear, este se muestra en la siguiente imagen, tomada del libro IA, A Modern Approach.
#### Algoritmo de aprendizaje del árbol de decisión
![Algoritmo DTs](/imgs/algoritmo_dts.PNG "Algoritmo de Aprendizaje")

---
#### Fórmulas empleadas
Otro aspecto importante que se debe conocer, es la generación de cálculos matemáticos que nos ayudarán a calcular la ganancia de información que tiene un determinado atributo, y con ello poder decidir sobre cuál de todos hacer una determinada **bifurcación** en el proceso de entrenamiento del modelo de árbol de decisión.

#### Entropía
El primer cálculo a tener en cuenta es la entropía, que indica el grado de incertidumbre que posee un cierto atributo. En este caso se implementa para determinar tanto la incertidumbre del set de datos o muestras, y también para el atributo específico sobre el que se quiere bifurcar. A continuación se muestra dicha fórmula:

<a href="https://www.codecogs.com/eqnedit.php?latex=Entropia(s)&space;=&space;\sum_{i=1}^{n}-p_{i}log_{2}p_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Entropia(s)&space;=&space;\sum_{i=1}^{n}-p_{i}log_{2}p_{i}" title="Entropia(s) = \sum_{i=1}^{n}-p_{i}log_{2}p_{i}" /></a>

Donde:
- **S**: es una colección de objetos
- **Pi** : es la probabilidad de los posibles valores
- **i**: las posibles respuestas de los objetos

#### Ganancia de Información
Por otra parte, se encuentra la formula de ganancia, la cual hace uso del **Resto** y la **Entropía**. Esta consiste básicamente en restar la entropía del set de datos con el resto obtenido del atributo A, obteniendo así el valor de ganancia de dicho atributo. A continuación la fórmula:

<a href="https://www.codecogs.com/eqnedit.php?latex=Ganancia(A)&space;=&space;I(\frac{p}{p&plus;n},&space;\frac{n}{p&plus;n})&space;-&space;Resto(A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Ganancia(A)&space;=&space;I(\frac{p}{p&plus;n},&space;\frac{n}{p&plus;n})&space;-&space;Resto(A)" title="Ganancia(A) = I(\frac{p}{p+n}, \frac{n}{p+n}) - Resto(A)" /></a>

#### Resto
Por último, se encuentra la fórmula Resto, traducida del inglés **Remainder**.

<a href="https://www.codecogs.com/eqnedit.php?latex=Resto(A)&space;=&space;\sum_{i=1}^{v}\frac{p_{i}&plus;n_{i}}{p&plus;n}&space;I(\frac{p_{i}}{p_{i}&plus;n_{i}},&space;\frac{n_{i}}{p_{i}&plus;n_{i}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Resto(A)&space;=&space;\sum_{i=1}^{v}\frac{p_{i}&plus;n_{i}}{p&plus;n}&space;I(\frac{p_{i}}{p_{i}&plus;n_{i}},&space;\frac{n_{i}}{p_{i}&plus;n_{i}})" title="Resto(A) = \sum_{i=1}^{v}\frac{p_{i}+n_{i}}{p+n} I(\frac{p_{i}}{p_{i}+n_{i}}, \frac{n_{i}}{p_{i}+n_{i}})" /></a>

---
#### Pasos llevados a cabo
Respecto a la clasificación basada en DTs (Decision Trees), la misma se resume en una serie de pasos que se mencionan a continuación:
- Generar el conjunto de muestras, que en este caso se obtienen del simulador de votantes que se empleó en el proyecto corto 1 (PC1). Esto por medio de los import:
> from tec.ic.ia.pc1.g03 import generar_muestra_pais
> from tec.ic.ia.pc1.g03 import generar_muestra_provincia

- Luego de ello, por medio de los parámetros recibos, se deben ajustar las muestras de entrenamiento, validación y pruebas. Es decir, crear los subconjuntos a partir de las muestras generadas con el simulador.
- Por último, solamente quedará por ejecutar todo el proceso de entrenamiento, validación y prueba, para cada predicción a generar. 

**Nota:** Dichas predicciones son las solicitadas en el proyecto, y que deben ser generadas por cada modelo, plasmando el resultado de cada predicción en un archivo final con exensión **.csv**
Estas predicciones se mencionan a continuación:
- Predicción de Ronda #1
- Predicción de Ronda #2
- Predicción de Ronda #2 con Ronda#1
- Y por último, una etiqueta la cual no representa una predicción, pero es una clasificación para indicar si la muestra fue tomada para entrenamiento o en el proceso de pruebas.

---
#### Resultados obtenidos
##### Ejecución de ejemplo #1
Conjunto de parámetros utilizados en la ejecución:
> --arbol --umbral-poda 0.1 --prefijo arbol_ex1 --poblacion 1000 --porcentaje-pruebas 10 --k-segmentos 10

**Nota:** Se debe tener claro que se utilizará el árbol con mejor precisión según el resultado arrojado por medio del proceso de cross-validation que se lleva a cabo en el entrenamiento.

**Precisiones obtenidas en la Ronda#1**
- **Mejor Precisión:** 21.11%
- **Peor Precisión:** 12.22%
- Gráfico de comportamiento:
Lo que indica es el valor de precisión que se obtuvo por cada grupo K, en el cross validation (K-Fold)

![pr1e1](/imgs/ex1_precisiones1.png "Preciones Ronda 1 - Ejemplo1")



**Precisiones obtenidas en la Ronda#2**
- **Mejor Precisión:** 54.44%
- **Peor Precisión:** 32.22%
- Gráfico de comportamiento:
Lo que indica es el valor de precisión que se obtuvo por cada grupo K, en el cross validation (K-Fold)

![pr2e1](/imgs/ex1_precisiones2.png "Preciones Ronda 2 - Ejemplo1")



**Precisiones obtenidas en la Ronda#2 con Ronda#1**
- **Mejor Precisión:** 60.33%
- **Peor Precisión:** 39.54%
- Gráfico de comportamiento:
Lo que indica es el valor de precisión que se obtuvo por cada grupo K, en el cross validation (K-Fold)

![pr21e1](/imgs/ex1_precisiones21.png "Preciones Ronda2 con Ronda1 - Ejemplo1")



---
##### Ejecución de ejemplo #2
Conjunto de parámetros utilizados en la ejecución:
> --arbol --umbral-poda 0.2 --prefijo arbol_ex2 --poblacion 2000 --porcentaje-pruebas 20 --k-segmentos 10

**Nota:** Se debe tener claro que se utilizará el árbol con mejor precisión según el resultado arrojado por medio del proceso de cross-validation que se lleva a cabo en el entrenamiento.

**Precisiones obtenidas en la Ronda#1**
- **Mejor Precisión:** 13,12%
- **Peor Precisión:** 9,37%
- Gráfico de comportamiento:
Lo que indica es el valor de precisión que se obtuvo por cada grupo K, en el cross validation (K-Fold)

![pr1e2](/imgs/ex2_precisiones1.png "Preciones Ronda 1 - Ejemplo2")



**Precisiones obtenidas en la Ronda#2**
- **Mejor Precisión:** 44,37%
- **Peor Precisión:** 20%
- Gráfico de comportamiento:
Lo que indica es el valor de precisión que se obtuvo por cada grupo K, en el cross validation (K-Fold)

![pr2e2](/imgs/ex2_precisiones2.png "Preciones Ronda 2 - Ejemplo2")



**Precisiones obtenidas en la Ronda#2 con Ronda#1**
- **Mejor Precisión:** 42,5%
- **Peor Precisión:** 19,37%
- Gráfico de comportamiento:
Lo que indica es el valor de precisión que se obtuvo por cada grupo K, en el cross validation (K-Fold)

![pr21e2](/imgs/ex2_precisiones21.png "Preciones Ronda2 con Ronda1 - Ejemplo2")



### Clasificación basada en KNN con Kd-trees

#### Teoría 

La definición de la función $$NN(k, x_i)$$ es sencilla; dado un conjunto de $$N$$ muestras y una consulta $$x_q$$,  se debe retornar la muestra $n_i$ que resulte en la menor distancia  con $x_q$. Esto tiene un costo computacional de $O(N)$ por lo que no resulta eficiente ante grandes cantidades de datos . 

Una solución a esto es disminuir el tiempo de consulta por medio de la estructura `k-d tree` . Con esta estructura se puede reducir el espacio de búsqueda a la mitad cada vez que se realiza una iteración. En cada una de estas iteraciones se selecciona un atributo mediante algún criterio, como la varianza, o por medio de alguna secuencia definida. 

En el siguiente ejemplo se puede apreciar el procedimiento utilizado para contruir un `k-d tree` con solamente dos dimensiones. 

![Kd-Tree](/imgs/kd_tree_estructura.png "Kd-Tree")

Primeramente, se dividen las muestras utilizando el atributo $x$ ![#f03c15](https://placehold.it/15/f03c15/000000?text=+), posteriormente se utiliza el atributo $y$ para los dos subconjuntos resultantes (![#FFBF00](https://placehold.it/15/FFBF00/000000?text=+), ![#2E64FE](https://placehold.it/15/#2E64FE/000000?text=+)) recursivamente se aplica el procedimiento hasta que solamente se conserve un par $(x, y)$ en los nodos resultantes ![#f1f1f1](https://placehold.it/15/f1f1f1/000000?text=+). 

#### Implementación

El `k-d tree` ha sido implementado mediante diccionarios de `Python`.  Esto debido a que era conveniente para depurar, puesto que era fácilmente visualizable mediante la función `pprint`. 

Debido a la necesidad calcular la distancia entre dos vectores, era indispensable que los valores de todas las columnas fueran númericos, por lo que se realiza una etapa de pre-procesamiento que consisten en dos parte. 

La primera se encarga de convertir los atributos categóricos en un representacion númerica. Para ello se usa el algoritmo `One Hot Encoding`, el cual crea una columna adicional con valores binarios por cada categoría posible que pueda tener un atributo en particular. Este fue implementado mediante la función `get_dummies` de la librería `pandas`. Un ejemplo de como funciona el algoritmo se ve reflejado en las siguientes tablas: 

| Canton    | Edad | Sexo |
| --------- | ---- | ---- |
| Paraíso   | 20   | M    |
| Cervantes | 30   | F    |
| Cervantes | 40   | F    |

De la tabla anterior, la primera columna debe se convertida a una representación númerica, mientras que la 3 debe ser binaria, por lo que la tabla, luego de este procedimiento, quedaría de la siguiente manera: 

| CantonParaiso | CantonCervantes | Edad | Sexo |
| ------------- | --------------- | ---- | ---- |
| 1             | 0               | 20   | 1    |
| 0             | 1               | 30   | 0    |
| 0             | 1               | 40   | 0    |

La segunda etapa...

El modo de seleccionar el atributo encargado de bifurcar el ábol es elegido calculando la varianza de cada columna de la matriz de muestras.  

#### Resultados obtenidos  

Se ha ejecutado el modelo utilizando distintos valores de $$k$$,  y con una cantidad de muestras $$n$$. Cada entrenamiento del modelo ha sido realizado utilizando un $10\%$ como porcentaje reservado para pruebas. 

A continuación, un resumen de los promedios de precisión obtenidos mediante el proceso de `k-fold-cross-validation`. 

|   n/k   |     1      |     5      |     11     |     21     |     41     |
| :-----: | :--------: | :--------: | :--------: | :--------: | :--------: |
|   100   | 24, 54, 54 | 22, 49, 56 | 19, 50, 45 | 22, 62, 63 | 18, 63, 58 |
|   500   | 18, 56, 50 | 23, 56, 53 | 22, 61, 56 | 22, 57, 53 | 20, 53, 52 |
|  1.000  | 20, 48, 53 | 22, 54, 53 | 20, 53, 55 | 23, 53, 54 | 21, 57, 58 |
|  2.000  | 16, 55, 55 | 20, 56, 57 | 22, 56, 53 | 21, 57, 55 | 20, 55, 51 |
|  4.000  | 19, 52, 52 | 19, 53, 52 | 19, 53, 56 | 23, 52, 53 | 20, 58, 55 |
| 10.000  | 19, 51, 52 | 19, 53, 54 | 22, 52, 54 | 20, 53, 56 | 22, 57, 58 |
| 100.000 | 18, 50, 51 | 18, 55, 55 | 21, 58, 57 | 21, 59, 58 | 22, 57, 58 |

La siguiente tabla resume las mejores precisiones obtenidas mediante el mismo proceso de `k-fold-cross-validation`. 

|   n/k   |     1      |     5      |     11     |     21     |     41     |
| :-----: | :--------: | :--------: | :--------: | :--------: | :--------: |
|   100   | 50, 67, 61 | 28, 67, 61 | 33, 61, 67 | 39, 67, 67 | 22, 78, 78 |
|   500   | 24, 63, 58 | 24, 60, 64 | 26, 66, 66 | 34, 63, 54 | 28, 60, 59 |
|  1.000  | 26, 55, 56 | 23, 60, 59 | 25, 55, 59 | 32, 62, 63 | 23, 63, 62 |
|  2.000  | 19, 61, 61 | 21, 61, 61 | 24, 60, 59 | 24, 61, 63 | 22, 59, 60 |
|  4.000  | 20, 57, 56 | 20, 57, 54 | 21, 59, 61 | 25, 60, 59 | 22, 61, 59 |
| 10.000  | 20, 53, 53 | 22, 56, 56 | 24, 57, 56 | 23, 61, 62 | 24, 61, 61 |
| 100.000 | 18, 52, 55 | 19, 57, 57 | 23, 58, 59 | 22, 60, 60 | 28, 59, 59 |

Finalmente, se presentan por medio de la siguiente tabla, las precisiones obtenidas para el conjunto reservado para pruebas. 

|   n/k   |     1      |     5      |     11     |     21     |     41     |
| :-----: | :--------: | :--------: | :--------: | :--------: | :--------: |
|   100   | 30, 60, 40 | 20, 60, 60 | 20, 80, 80 | 10, 80, 80 | 30, 70, 70 |
|   500   | 26, 56, 64 | 22, 50, 52 | 28, 58, 58 | 14, 64, 07 | 22, 60, 60 |
|  1.000  | 12, 52, 52 | 17, 58, 52 | 19, 54, 60 | 23, 64, 67 | 18, 68, 61 |
|  2.000  | 20, 64, 57 | 24, 55, 62 | 26, 61, 60 | 22, 59, 57 | 20, 59, 60 |
|  4.000  | 19, 55, 53 | 20, 59, 55 | 23, 59, 59 | 23, 58, 64 | 24, 64, 63 |
| 10.000  | 19, 54, 54 | 20, 54, 53 | 24, 57, 59 | 24, 62, 63 | 23, 59, 57 |
| 100.000 | 19, 53, 55 | 18, 56, 57 | 23, 58, 59 | 22, 60, 60 | 23, 60, 60 |

## Acerca de
Integrantes del proyecto:

| Nombre                    | Carné      |
| ------------------------- | ---------- |
| Brandon Dinarte Chavarría | 2015088894 |
| Armando López Cordero     | 2015125414 |
| Julian Salinas Rojas      | 2015114132 |

Estudianteas de Ingeniería en Computación, en el Instituto Tecnológico de Costa Rica.