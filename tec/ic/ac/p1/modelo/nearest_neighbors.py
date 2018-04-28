# -----------------------------------------------------------------------------

"""
Clasificador K-Nearest-Neightbor utilizando kd-tree
- "asado en la sección 18.8.2 del Libro AI-A Modern Approach, 737.
- Se sigue el ejemplo de la siguiente página https://goo.gl/TGQaEe
- La información debe ser preprocesada
    * One Hot Encoding para atributos categóricos
    * Normalización para que todos los atributos tengan la misma escala
"""

import numpy as np
from pprint import pprint

# -----------------------------------------------------------------------------


def k_nearest_neighbors(ejemplos, k_vecinos=5):

    # Resulta mejor cuando hay más ejemplos que dimensiones
    # k también representa el número de dimensiones del kd-tree

    # TODO: Encontrar el k más óptimo con cross-validation
    # K debe ser impar para evitar empates (pág 738)
    # Esto se verifica al parsear los argmentos

    return None

# -----------------------------------------------------------------------------


def normalizar(datos, menor=0.0, mayor=1.0):

    """
    Normaliza los valores de la matriz para que todas la variables puedan
    competir utilizando la misma escala.

    NOTA: Es equivalente a:
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(datos)

    :param datos: datos: 2D np.array
    :param menor: Número más bajo que tendrá la matriz luego de normalizar
    :param mayor: Número más alto que tendrá la matriz luego de normalizar
    :return: Los mismos datos de entrada pero normalizados

    Ejemplo:
    >>> datos = np.array([[250, 0.15, 12],
    ...                   [120, 0.65, 20],
    ...                   [500, 0.35, 19]])
    >>> np.round(normalizar(datos), decimals=2)
    array([[0.34, 0.  , 0.  ],
           [0.  , 1.  , 1.  ],
           [1.  , 0.4 , 0.88]])
    """

    maxs = np.max(datos, axis=0)
    mins = np.min(datos, axis=0)
    difs = maxs - mins
    prom = mayor - menor
    return mayor - ((prom * (maxs - datos)) / difs)

# -----------------------------------------------------------------------------


def ordenar(datos, columna):

    """
    Ordena de menor a mayor los datos (tabla) según los valores
    de una sola columna.

    :param datos: 2D np.array
    :param columna: Indíce de la columna
    :return: Los mismos datos de entrada pero ordenados

    Ejemplo:
    >>> datos = np.array([[  1,  10,   3],
    ...                   [ 34,  56,  43],
    ...                   [123,   9, 120]])
    >>> ordenar(datos, columna=1)
    array([[123,   9, 120],
           [  1,  10,   3],
           [ 34,  56,  43]])
    """

    columna = [columna]
    return datos[np.lexsort(datos.T[columna])]

# -----------------------------------------------------------------------------


def distancia(vector_x, vector_y):

    """
    Calcula la distancia euclidiana entre dos vectores

    :param vector_x: np.array tamaño N
    :param vector_y: np.array tamaño N
    :return: Número que representa la distancia entre los dos vectores

    Ejemplo:
    >>> vector_x = np.array([3, 6, 8])
    >>> vector_y = np.array([1, 4, 5])
    >>> distancia(vector_x, vector_y)
    4.123105625617661
    """

    # Sumatoria de las diferencias al cuadrado, es lo mismo que:
    # math.sqrt(np.sum(np.square(vector_x - vector_y)))
    return np.linalg.norm(vector_x - vector_y)

# -----------------------------------------------------------------------------
