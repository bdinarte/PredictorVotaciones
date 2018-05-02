# -----------------------------------------------------------------------------

"""
Módulo con las funciones necesarias para realizar cross-validation
"""

import sys
sys.path.append('../..')

import math
from random import shuffle
from itertools import chain
from p1.modelo.nearest_neighbors import *
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------


def cross_validation(args, datos, columnas, columnas_c):

    errores = list()

    indices = list(range(len(datos)))
    shuffle(indices)

    indices = separar(indices, args.porcentaje_pruebas[0])

    indices_pruebas = indices[0]
    indices_entrenamiento = indices[1]

    # Se dividen los datos de entrenamiento en varios k bloques o segmentos
    if args.k_segmentos is None:
        segmentos = segmentar(indices_entrenamiento)
    else:
        segmentos = segmentar(indices_entrenamiento, args.k_segmentos[0])

    if args.knn:

        # Preprocesar es necesario porque la matriz de entrenamiento
        # necesita ser númerica y estar normalizada.
        # columnas_c son los nombres de las columnas categoricas
        datos, etiquetas = preprocesar(datos, columnas, columnas_c)

    # Se hacen k iteraciones. En cada iteración se usa un segmento
    # diferente para validación, el resto se usa para entrenamiento
    for k in range(len(segmentos)):

        print('\nPasada #' + str(k))
        indices_validacion, indices_entrenamiento = agrupar(k, segmentos)

        # Modelos lineales: Regresión logistica
        # Argumentos específicos necesarios:
        #   args.l1
        #   args.l2
        if args.regresion_logistica:
            print('Regresión logística')
            print('Argumentos: ')
            print('\t l1 ' + str(args.l1))
            print('\t l2 ' + str(args.l2))

        # Red neuronal
        # Argumentos específicos necesarios:
        #   args.numero_capas
        #   args.unidades_por_capa
        #   args.funcion_activacion
        elif args.red_neuronal:
            print('Red neuronal')
            print('Argumentos: ')
            print('\t numero_capas ' + str(args.numero_capas))
            print('\t unidades_por_capa ' + str(args.unidades_por_capa))
            print('\t funcion_activacion ' + str(args.funcion_activacion))

        # Árbol de decisión
        # Argumentos específicos necesarios:
        #   args.umbral_poda
        elif args.arbol:
            print('Árbol de decisión')
            print("Argumentos: ")
            print('\t umbral_poda' + str(args.umbral_poda))

        # K Nearest Neighbors
        # Argumentos específicos necesarios:
        #   args.k
        elif args.knn:

            print('K Nearest Neighbors')

            # Si no se especifica k, se usa 5 por defecto
            k_vecinos = 5 if args.k is None else args.k[0]
            print("Argumentos: k -> " + str(k_vecinos))

            # Después de normazalizar se dividen los datos
            datos_validacion = datos[indices_validacion]
            datos_entrenamiento = datos[indices_entrenamiento]

            # Usadas para la validación
            etiqs_correctas = etiquetas[indices_validacion]

            # Correspondiendientes a los datos de entrenamiento
            etiqs_entrenamiento = etiquetas[indices_entrenamiento]

            # Se obtiene el árbol, es decir el clasificador entrenado
            clasificador = kdtree((datos_entrenamiento, etiqs_entrenamiento))

            # Para cada votante en los datos de validación predice
            # el partido por el que votó.
            etiqs_predicciones = np.array([
                predecir(clasificador, votante, k_vecinos)
                for votante in datos_validacion
            ])

            precision = accuracy_score(etiqs_correctas, etiqs_predicciones)
            errores.append(precision)
            print("Precisión: " + str(precision))

    return np.mean(errores)

# -----------------------------------------------------------------------------


def separar(indices_datos, porcentaje_pruebas=20):

    """
    Se separan los indices que apuntan a los datos en un conjunto
    para entrenamiento y otro para pruebas.
    :param indices_datos: Indices apuntando a los datos
    :param porcentaje_pruebas: Porcentaje de los `indices_datos` para pruebas
    :return: ([indices_datos_pruebas], [indices_datos_entrenamiento])

    Ejemplo:
    >>> indices_datos = list(range(8))
    >>> separar(indices_datos, porcentaje_pruebas=25)
    ([0, 1], [2, 3, 4, 5, 6, 7])
    """

    cant_datos = len(indices_datos)
    cant_pruebas = math.floor(cant_datos * (porcentaje_pruebas / 100.0))

    indices_pruebas = indices_datos[0:cant_pruebas]
    indices_entrenamiento = indices_datos[cant_pruebas: cant_datos]

    return indices_pruebas, indices_entrenamiento

# -----------------------------------------------------------------------------


def segmentar(indices_entrenamiento, k_segmentos=10):

    """
    Divide los indices_entrenamiento de entrenamiento en `k_segmentos`

    :param indices_entrenamiento: Conjunto de iudices que apuntan a los datos
    :param k_segmentos: Cantidad de segmentos
    :return: Lista de listas, donde cada una representa un segmento por medio
    de indices apuntando a los indices_entrenamiento

    Ejemplos:

    >>> indices_entrenamiento = list(range(6))
    >>> segmentar(indices_entrenamiento, k_segmentos=4)
    [[0, 1], [2, 3], [4, 5]]

    >>> datos_entrenamiento = list(range(7))
    >>> segmentar(datos_entrenamiento, k_segmentos=4)
    [[0, 1], [2, 3], [4, 5], [6]]

    >>> datos_entrenamiento = list(range(8))
    >>> segmentar(datos_entrenamiento, k_segmentos=4)
    [[0, 1], [2, 3], [4, 5], [6, 7]]
    """

    cant_datos = len(indices_entrenamiento)

    # Tamaño de cada uno de los segmentos 
    t_segmento = round(cant_datos / k_segmentos)

    # Se necesita en caso de que la cantidad de indices_entrenamiento no sea
    # múltiplo de `k_segmentos`
    t_ultimo_segmento = cant_datos - ((k_segmentos - 1) * t_segmento)

    segmentos = []
    indice_actual = 0
    for _ in range(k_segmentos - 1):
        segmentos.append(indices_entrenamiento[indice_actual:indice_actual + t_segmento])
        indice_actual += t_segmento

    # El último segmento es lo que sobra de los indices_entrenamiento
    if t_ultimo_segmento is not 0:
        segmentos.append(indices_entrenamiento[indice_actual:cant_datos])

    return segmentos

# -----------------------------------------------------------------------------


def agrupar(indice_pruebas, segmentos):

    """
    Agrupa los segmentos de entrenamiento en un solo conjunto apartando
    del segmento destinado a validación
    :param indice_pruebas: Apunta al segmento que se usará para validación
    :param segmentos: Lista de listas
    :return: (indices_validacion, conjunto_entrenamiento)

    Ejemplo:

    >>> segmentos = [[0, 1], [2, 3], [4, 5]]
    >>> agrupar(1, segmentos)
    ([2, 3], [0, 1, 4, 5])
    """

    cant_segmentos = len(segmentos)
    indices_validacion = segmentos[indice_pruebas]

    # Se ignora `segmento_prueba`
    inicio = segmentos[0:indice_pruebas]
    final = segmentos[indice_pruebas + 1: cant_segmentos]

    # `chain.from_iterable` fue la función más rápida para
    # aplanar la lista. Fuente: https://goo.gl/28KsUx
    indices_entrenamiento = list(chain.from_iterable(inicio + final))

    return indices_validacion, indices_entrenamiento

# -----------------------------------------------------------------------------
