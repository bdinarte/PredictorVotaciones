# -----------------------------------------------------------------------------

import math
from itertools import chain

# -----------------------------------------------------------------------------


def separar(indices, porcentaje=20):

    """
    Se separan los indices que apuntan a los datos en un conjunto
    para entrenamiento y otro para pruebas.
    :param indices: Indices apuntando a los datos
    :param porcentaje: Porcentaje de los `indices` para pruebas
    :return: ([indices_datos_pruebas], [indices_datos_entrenamiento])

    Ejemplo:
    >>> indices = list(range(8))
    >>> separar(indices, porcentaje=25)
    ([0, 1], [2, 3, 4, 5, 6, 7])
    """

    cant_datos = len(indices)
    cant_pruebas = math.floor(cant_datos * (porcentaje / 100.0))

    indices_pruebas = indices[0:cant_pruebas]
    indices_entrenamiento = indices[cant_pruebas: cant_datos]

    return indices_pruebas, indices_entrenamiento

# -----------------------------------------------------------------------------


def segmentar(indices, cant_segmentos=10):

    """
    Divide los indices de entrenamiento en `k_segmentos`

    :param indices: Conjunto de iudices que apuntan a los datos
    :param cant_segmentos: Cantidad de segmentos
    :return: Lista de listas, donde cada una representa un segmento por medio
    de indices apuntando a los indices

    Ejemplos:

    >>> indices = list(range(6))
    >>> segmentar(indices, cant_segmentos=4)
    [[0, 1], [2, 3], [4, 5]]

    >>> datos = list(range(7))
    >>> segmentar(datos, cant_segmentos=4)
    [[0, 1], [2, 3], [4, 5], [6]]

    >>> datos = list(range(8))
    >>> segmentar(datos, cant_segmentos=4)
    [[0, 1], [2, 3], [4, 5], [6, 7]]
    """

    cant_datos = len(indices)

    # Tamaño de cada uno de los segmentos
    t_segmento = round(cant_datos / cant_segmentos)

    # Se necesita en caso de que la cantidad de indices no sea
    # múltiplo de `cant_segmentos`
    t_ultimo_segmento = cant_datos - ((cant_segmentos - 1) * t_segmento)

    segmentos = []
    indice_actual = 0
    for _ in range(cant_segmentos - 1):
        segmentos.append(indices[indice_actual:indice_actual + t_segmento])
        indice_actual += t_segmento

    # El último segmento es lo que sobra de los indices
    if t_ultimo_segmento is not 0:
        segmentos.append(indices[indice_actual:cant_datos])

    return segmentos

# -----------------------------------------------------------------------------


def agrupar(indices, segmentos):

    """
    Agrupa los segmentos de entrenamiento en un solo conjunto apartando
    del segmento destinado a validación
    :param indices: Apunta al segmento que se usará para validación
    :param segmentos: Lista de listas
    :return: (indices_validacion, conjunto_entrenamiento)

    Ejemplo:

    >>> segmentos = [[0, 1], [2, 3], [4, 5]]
    >>> agrupar(1, segmentos)
    ([2, 3], [0, 1, 4, 5])
    """

    cant_segmentos = len(segmentos)
    indices_validacion = segmentos[indices]

    # Se ignora `segmento_prueba`
    inicio = segmentos[0:indices]
    final = segmentos[indices + 1: cant_segmentos]

    # `chain.from_iterable` fue la función más rápida para
    # aplanar la lista. Fuente: https://goo.gl/28KsUx
    indices_entrenamiento = list(chain.from_iterable(inicio + final))

    return indices_validacion, indices_entrenamiento

# -----------------------------------------------------------------------------
