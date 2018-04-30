# -----------------------------------------------------------------------------

"""
Módulo con las funciones necesarias para realizar cross-validation
"""

import math
from random import shuffle
from itertools import chain

# -----------------------------------------------------------------------------


def k_fold_cross_validation(datos, clasificador,
                            porcentaje_pruebas=20, k_segmentos=4):

    """
    1. Se dividen los datos en un conjunto de pruebas y otro de entrenamiento
    2. El conjunto de entrenamiento lo dividide en `k_segmentos`
    3. Repetir k veces:
        A. Obtener el k segmento para pruebas y el resto para entrenamiento
        B. Entrenar el clasificador
        C. Ejecutar el clasificador para obtener el error
    4. El error se convierte en el promedio de los errores obtenidos
    NOTA: El clasificador también es conocido como modelo
    :param datos: Lista
    :param clasificador: TODO: Por definir
    :param porcentaje_pruebas: Porcentaje de los `datos` para pruebas
    :param k_segmentos: Cantidad de segmentos
    :return: El promedio de error obtenidos de los clasificadores
    """

    # Se aparta un conjunto
    shuffle(datos)
    datos_divididos = dividir_datos(datos, porcentaje_pruebas)
    datos_pruebas = datos_divididos[0]
    datos_entrenamiento = datos_divididos[1]

    errores = list()
    segmentos = obtener_segmentos(datos_entrenamiento, k_segmentos)

    for k in range(k_segmentos):
        conjuntos = agrupar_segmentos(k, segmentos)
        conjunto_validacion = conjuntos[0]
        conjunto_entrenamiento = conjuntos[1]
        # TODO: Se entrena el clasificador
        # clasificador = entrenar(clasificador, conjunto_entrenamiento)
        # TODO: Se ejecuta el clasificador
        # errores.append(clasificar(clasificador, conjunto_validacion))

    return sum(errores)

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
    t_segmento = round(cant_datos / k_segmentos)

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


def agrupar_segmentos(indice_pruebas, segmentos):

    """
    Agrupa los segmentos de entrenamiento en un solo conjunto apartando
    del segmento destinado a pruebas
    :param indice_pruebas: Lugar del segmento de pruebas dentro del
    conjunto de segmentos
    :param segmentos: Lista de listas
    :return: (conjunto_prueba, conjunto_entrenamiento)

    Ejemplos:

    >>> segmentos = [['A', 'B'], ['C', 'D'], ['E', 'F']]
    >>> agrupar_segmentos(0, segmentos)
    (['A', 'B'], ['C', 'D', 'E', 'F'])

    >>> segmentos = [['A', 'B'], ['C', 'D'], ['E', 'F']]
    >>> agrupar_segmentos(1, segmentos)
    (['C', 'D'], ['A', 'B', 'E', 'F'])

    >>> segmentos = [['A', 'B'], ['C', 'D'], ['E', 'F']]
    >>> agrupar_segmentos(2, segmentos)
    (['E', 'F'], ['A', 'B', 'C', 'D'])
    """

    cant_segmentos = len(segmentos)
    conjunto_pruebas = segmentos[indice_pruebas]

    # Se ignora `segmento_prueba`
    inicio = segmentos[0:indice_pruebas]
    final = segmentos[indice_pruebas + 1: cant_segmentos]

    # `chain.from_iterable` fue la función más rápida para
    # aplanar la lista. Fuente: https://goo.gl/28KsUx
    conjunto_entrenamiento = list(chain.from_iterable(inicio + final))

    return conjunto_pruebas, conjunto_entrenamiento

# -----------------------------------------------------------------------------
