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

## Reportes de Métodos Implementados
###Clasificación basada en modelos lineales

### Clasificación basada en redes neuronales

### Clasificación basada árboles de decisión

#### Ejemplo de un árbol de decisión
![Arbol de Decision](/imgs/arbol_decision.jpg "Árbol de Decisión")
<img src="/imgs/arbol_decision.jpg" alt="Árbol de Decisión" style="width: 100px; height: 100px;"/>

#### Algoritmo de aprendizaje del árbol de decisión
![Algoritmo DTs](/imgs/algoritmo_dts.PNG "Algoritmo de Aprendizaje")

#### Entropía
<a href="https://www.codecogs.com/eqnedit.php?latex=Entropia(s)&space;=&space;\sum_{i=1}^{n}-p_{i}log_{2}p_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Entropia(s)&space;=&space;\sum_{i=1}^{n}-p_{i}log_{2}p_{i}" title="Entropia(s) = \sum_{i=1}^{n}-p_{i}log_{2}p_{i}" /></a>

Donde:
**S**: es una colección de objetos
**Pi** : es la probabilidad de los posibles valores
**i**: las posibles respuestas de los objetos

#### Ganancia de Información
<a href="https://www.codecogs.com/eqnedit.php?latex=Ganancia(A)&space;=&space;I(\frac{p}{p&plus;n},&space;\frac{n}{p&plus;n})&space;-&space;Resto(A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Ganancia(A)&space;=&space;I(\frac{p}{p&plus;n},&space;\frac{n}{p&plus;n})&space;-&space;Resto(A)" title="Ganancia(A) = I(\frac{p}{p+n}, \frac{n}{p+n}) - Resto(A)" /></a>




#### Resto
<a href="https://www.codecogs.com/eqnedit.php?latex=Resto(A)&space;=&space;\sum_{i=1}^{v}\frac{p_{i}&plus;n_{i}}{p&plus;n}&space;I(\frac{p_{i}}{p_{i}&plus;n_{i}},&space;\frac{n_{i}}{p_{i}&plus;n_{i}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Resto(A)&space;=&space;\sum_{i=1}^{v}\frac{p_{i}&plus;n_{i}}{p&plus;n}&space;I(\frac{p_{i}}{p_{i}&plus;n_{i}},&space;\frac{n_{i}}{p_{i}&plus;n_{i}})" title="Resto(A) = \sum_{i=1}^{v}\frac{p_{i}+n_{i}}{p+n} I(\frac{p_{i}}{p_{i}+n_{i}}, \frac{n_{i}}{p_{i}+n_{i}})" /></a>



### Clasificación basada en KNN con Kd-trees

La definición de la función $NN(k, x_i)$ es sencilla :blush:; dado un conjunto de N muestras y una consulta $x_q$,  se debe retornar la muestra $n_i$ que resulte en la menor distancia  con $x_q$. Esto tiene un costo computacional de $O(N)$ por lo que no resulta eficiente ante grandes cantidades de datos :disappointed_relieved:. 

Una solución a esto es disminuir el tiempo de consulta por medio de la estructura `k-d tree` . Con esta estructura se puede reducir el espacio de búsqueda a la mitad cada vez que se realiza una iteración. En cada una de estas iteraciones se selecciona un atributo mediante algún criterio, como la varianza, o por medio de alguna secuencia definida. 

En el siguiente ejemplo se puede apreciar el procedimiento utilizado para contruir un `k-d tree` con solamente dos dimensiones. 

![Kd-Tree](/imgs/kd_tree_estructura.png "Kd-Tree")

Primeramente, se dividen las muestras utilizando el atributo $x$ ![#f03c15](https://placehold.it/15/f03c15/000000?text=+), posteriormente se utiliza el atributo $y$ para los dos subconjuntos resultantes (![#FFBF00](https://placehold.it/15/FFBF00/000000?text=+), ![#2E64FE](https://placehold.it/15/#2E64FE/000000?text=+)) recursivamente se aplica el procedimiento hasta que solamente se conserve un par $(x, y)$ en los nodos resultantes ![#f1f1f1](https://placehold.it/15/f1f1f1/000000?text=+). 



## Acerca de
Respecto a los integrantes del proyecto, se encuentran los estudiantes:
- Brandon Dinarte Chavarría 2015088894
- Armando López Cordero     2015125414
- Julian Salinas Rojas      2015114132

Mismos que cursan la carrera de Ingeniería en Computación, en el Instituto Tecnológico de Costa Rica.