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
![Con titulo](/imgs/arbol_decision.jpg "Árbol de Decisión")

#### Algoritmo de aprendizaje del árbol de decisión
![Con titulo](/imgs/algoritmo_dts.png "Algoritmo de Aprendizaje")

#### Entropía
$$Entropia(s) =  \sum_{i=1}^{n}-p_{i}log_{2}p_{i}$$
Donde:
**S**: es una colección de objetos
**Pi** : es la probabilidad de los posibles valores
**i**: las posibles respuestas de los objetos

#### Ganancia de Información
$$Ganancia(A) = I(\frac{p}{p+n}, \frac{n}{p+n}) - Resto(A)$$

#### Resto
$$Resto(A) = \sum_{i=1}^{v}\frac{p_{i}+n_{i}}{p+n} I(\frac{p_{i}}{p_{i}+n_{i}}, \frac{n_{i}}{p_{i}+n_{i}}) $$

### Clasificación basada en KNN con Kd-trees

## Acerca de
Respecto a los integrantes del proyecto, se encuentran los estudiantes:
- Brandon Dinarte Chavarría 2015088894
- Armando López Cordero     2015125414
- Julian Salinas Rojas      2015114132

Mismos que cursan la carrera de Ingeniería en Computación, en el Instituto Tecnológico de Costa Rica.