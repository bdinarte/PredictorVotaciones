# -----------------------------------------------------------------------------

"""
Módulo con las funciones necesarias para realizar cross-validation
"""

import math
import numpy as np
from random import shuffle

# -----------------------------------------------------------------------------


def k_fold_cross_validation(datos, porcentaje_pruebas=20, k_segmentos=4):

    """
    1. Dividide los datos en un conjunto de pruebas y otro de entrenamiento
    2. El conjunto de entrenamiento lo dividide en `k_segmentos`
    3. Continuará
    :param datos: Lista
    :param porcentaje_pruebas: Porcentaje de los `datos` para pruebas
    :param k_segmentos: Cantidad de segmentos
    :return:
    """

    shuffle(datos)

    datos_pruebas, datos_entrenamiento = \
        dividir_datos(datos, porcentaje_pruebas)

    segmentos = obtener_segmentos(datos_entrenamiento, k_segmentos)

    # TODO: Aquí comenzar la evaluación de los modelos

    return None

# -----------------------------------------------------------------------------


def dividir_datos(datos, porcentaje_pruebas=20):

    """
    Divide los datos en dos:
        Datos para pruebas
        Datos para entrenamiento
    :param datos: Lista
    :param porcentaje_pruebas: Porcentaje de los `datos` para pruebas
    :return: ([datos_pruebas], [datos_entrenamiento])

    Ejemplo:
    >>> datos = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    >>> dividir_datos(datos, porcentaje_pruebas=25)
    (['A', 'B'], ['C', 'D', 'E', 'F', 'G', 'H'])
    """

    cant_datos = len(datos)
    cant_pruebas = math.floor(cant_datos * (porcentaje_pruebas / 100.0))
    return datos[0:cant_pruebas], datos[cant_pruebas: cant_datos]

# -----------------------------------------------------------------------------


def obtener_segmentos(datos, k_segmentos):

    """
    Divide los datos de entrenamiento en `k_segmentos`

    :param datos: Lista
    :param k_segmentos: Cantidad de segmentos
    :return: Lista de listas, donde cada una es un segmento de los datos

    Ejemplos:

    >>> datos_entrenamiento = ['A', 'B', 'C', 'D', 'E', 'F']
    >>> obtener_segmentos(datos_entrenamiento, k_segmentos=4)
    [['A', 'B'], ['C', 'D'], ['E', 'F']]

    >>> datos_entrenamiento = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    >>> obtener_segmentos(datos_entrenamiento, k_segmentos=4)
    [['A', 'B'], ['C', 'D'], ['E', 'F'], ['G']]

    >>> datos_entrenamiento = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    >>> obtener_segmentos(datos_entrenamiento, k_segmentos=4)
    [['A', 'B'], ['C', 'D'], ['E', 'F'], ['G', 'H', 'I']]

    """

    cant_datos = len(datos)

    # Tamaño de cada uno de los segmentos 
    t_segmento = round(cant_datos/k_segmentos)

    # Se necesita en caso de que la cantidada de datos no sea
    # múltiplo de `k_segmentos`
    t_ultimo_segmento = cant_datos - ((k_segmentos - 1) * t_segmento)

    segmentos = []
    indice_actual = 0
    for _ in range(k_segmentos - 1):
        segmentos.append(datos[indice_actual:indice_actual + t_segmento])
        indice_actual += t_segmento

    # El último segmento es lo que sobra de los datos
    if t_ultimo_segmento is not 0:
        segmentos.append(datos[indice_actual:cant_datos])

    return segmentos

# -----------------------------------------------------------------------------
