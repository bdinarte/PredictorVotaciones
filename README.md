# Predictor de Votaciones CR

Módulo que puede entrenar distintos modelos de clasificación de votantes y genera una serie de archivos de salida analizando el rendimiento de cada modelo.

## Instalación

### Requisitos Previos

**Nota**: Si se trabaja con Windows 7 o posterior, se debe contar con la versión `3.5.x` o `3.6.x` de Python de 64-bit.

1. Abrir la consola de comandos con permisos de administrador.

3. Instalar la versión adecuada de Tensorflow, `cpu` o `gpu` mediante alguno de los siguientes comandos según corresponda:

    > pip install --upgrade tensorflow

    > pip install --upgrade tensorflow-gpu


## Manual de Uso

## Reportes de Métodos Implementados
### Clasificación basada en modelos lineales

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

## Acerca de
Respecto a los integrantes del proyecto, se encuentran los estudiantes:
- Brandon Dinarte Chavarría 2015088894
- Armando López Cordero     2015125414
- Julian Salinas Rojas      2015114132

Mismos que cursan la carrera de Ingeniería en Computación, en el Instituto Tecnológico de Costa Rica.