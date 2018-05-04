# -----------------------------------------------------------------------------

from __future__ import division

import os
import sys

sys.path.append('../..')

import pandas as pd
from p1.util.util import *
from p1.modelo.columnas import *
from collections import Counter


# -----------------------------------------------------------------------------

def preprocesar_ronda(datos, ronda):
    """
    Función especifica para el modelo de los votantes.
        ♣ Se eliminan las columnas apropiadas según la ronda
        ♣ Se aplica One Hot Encoding a las columnas que lo requierenr
        ♣ Se normalizan los datos

    :param datos: Generados por el simulador de votantes
    :param ronda: Número de ronda. Puede ser 1, 2 o 3

    :return: Tupla con la siguiente información:
        ♣ [0] = Matriz N x M con valores númericos y normalizados
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz
    """

    # Ronda #1: No se necesita la columna r2
    if ronda is 1:
        datos_r1 = np.delete(datos, 22, axis=1)
        columnas_r1 = np.delete(columnas_csv, 22)
        return datos_r1, columnas_r1

    # Ronda #2 sin ronda #1: No se necesita la columan r1
    elif ronda is 2:
        datos_r2 = np.delete(datos, 21, axis=1)
        columnas_r2 = np.delete(columnas_csv, 21)
        return datos_r2, columnas_r2

    # Ronda #2 usando ronda #1
    # La columna de la ronda #1 pasa a ser categorica
    else:
        return datos, columnas_csv


# -----------------------------------------------------------------------------

class Datos:
    def __init__(self, clasificador):
        """
        Clase que contendra el conjunto de datos(muestras, atributos, la meta, etc)
        que seran utlizadas para generar el arbol
        :param clasificador: es el atributo meta(Goal) sobre el que se basa el modelo de clasificacion
        """
        # Es el conjunto de muestras generadas por el simulador
        self.muestras = []

        # Se puede ver como el conjunto de atributos que identifican cada columna de las muestras
        self.atributos = []

        # Indica por cada atributo si el mismo es numerico o no
        self.tipos_atributos = []

        # Es el atributo meta sobre el que se basa el modelo de calsificacion
        self.clasificador = clasificador

        # indica la posicion en la que se encuentra el atributo meta
        self.indice_clasificador = None


# -------------------------------------------------------------------------------

def preprocess2(set_datos):
    print("Preprocesando los datos...")

    valores_clasificador = [muestra[set_datos.indice_clasificador] for muestra in set_datos.muestras]

    # clasificador_plural, es el atributo que mas apariciones tiene en el set de valores meta
    clasificador_plural = Counter(valores_clasificador)
    clasificador_plural = clasificador_plural.most_common(1)[0][0]

    for indice_atributo in range(len(set_datos.atributos)):

        ex_pacclass = filter(lambda x: x[set_datos.indice_clasificador] == 'PAC', set_datos.muestras)
        values_pacclass = [muestra[indice_atributo] for muestra in ex_pacclass]

        ex_plnclass = filter(lambda x: x[set_datos.indice_clasificador] == 'PLN', set_datos.muestras)
        values_plnclass = [muestra[indice_atributo] for muestra in ex_plnclass]

        ex_puscclass = filter(lambda x: x[set_datos.indice_clasificador] == 'PUSC', set_datos.muestras)
        values_puscclass = [muestra[indice_atributo] for muestra in ex_puscclass]

        values = Counter(values_pacclass)  # value_counts = values.most_common()
        modepln = values.most_common(1)[0][0]
        if modepln == '?':
            modepln = values.most_common(2)[1][0]

        values = Counter(values_plnclass)
        modepac = values.most_common(1)[0][0]
        if modepac == '?':
            modepac = values.most_common(2)[1][0]

        values = Counter(values_puscclass)
        modepusc = values.most_common(1)[0][0]
        if modepusc == '?':
            modepusc = values.most_common(2)[1][0]

        mode_012 = [modepln, modepac, modepusc]

        attr_modes = [0] * len(set_datos.atributos)
        attr_modes[indice_atributo] = mode_012

        for muestra in set_datos.muestras:
            if muestra[indice_atributo] == '?':
                if muestra[set_datos.indice_clasificador] == 'PAC':
                    muestra[indice_atributo] = attr_modes[indice_atributo][0]
                elif muestra[set_datos.indice_clasificador] == 'PLN':
                    muestra[indice_atributo] = attr_modes[indice_atributo][1]
                elif muestra[set_datos.indice_clasificador] == 'PUSC':
                    muestra[indice_atributo] = attr_modes[indice_atributo][2]
                else:
                    muestra[indice_atributo] = clasificador_plural

        # convert atributos that are numeric to floats
        for muestra in set_datos.muestras:
            for x in range(len(set_datos.muestras[0])):
                if set_datos.atributos[x] == 'True':
                    muestra[x] = float(muestra[x])


# -------------------------------------------------------------------------------

class NodoArbol:
    def __init__(self, es_hoja, clasificacion, atributo_bif_index, atributo_bif_value,
                 nodo_padre, hijo_der, hijo_izq, peso):
        """
        Clase que representa los nodos del arbol que sera utilizado para implementar el modelo DT
        :param es_hoja: atributo que indica si el nodo se encuentra en el ultimo nivel(es hoja)
        :param clasificacion: es el valor meta con el que se clasifica el nodo
        :param atributo_bif_index: indice del atributo sobre el que se esta haciendo la bifurcacion
        :param atributo_bif_value: valor del atributo sobre el que se esta haciendo la bifurcacion
        :param nodo_padre: indicador del nodo predecesor
        :param hijo_der: nodo que se encuentra al lado derecho (lado mayor) en la bifurcacion
        :param hijo_izq: nodo que se encuentra al lado izquierdo (lado menor) en la bifurcacion
        :param peso: es el nivel de profundidad en el que se encuentra el nodo segun el nodo raiz
        """
        self.es_hoja = True
        self.clasificacion = None
        self.atributo_bif = None
        self.atributo_bif_index = None
        self.atributo_bif_value = None
        self.nodo_padre = nodo_padre
        self.hijo_der = None
        self.hijo_izq = None
        self.peso = None


# -------------------------------------------------------------------------------

def crear_arbol_decision(set_datos, nodo_padre, clasificador):
    """
    Funcion encargada de la construccion del arbol de decision, que se basa en un conjunto de muestras
    observadas, un nodo padre que al principio de la ejecucion sera nulo, y que con forme se va ejecutando
    la recursion, el arbol se ira construyendose al anidar nodos.
    Se tiene que tener claro que el clasificador hace refenrencia al atributo sobre el cual se basa el
    modelo que se esta construyendo, es decir es el valor a predecir en el proceso de testing.

    :param set_datos: es el conjunto de muestras, atributos y demas elementos que se utilizaran para
    generar el modelo de arbol de decision
    :param nodo_padre: es el nodo raiz sobre el que se creara una nueva bifurcacion o se determinara que
    es un nodo hoja
    :param clasificador: es el atributo sobre el que se va clasificar en el modelo
    :return: el nodo raiz del arbol de decision una vez que se encuentra totalmente entrenado
    """

    nodo = NodoArbol(True, None, None, None, nodo_padre, None, None, 0)
    if nodo_padre is None:
        nodo.peso = 0
    else:
        nodo.peso = nodo.nodo_padre.peso + 1

    cant_valores_distintos = contar_valores_distintos(set_datos.muestras, set_datos.atributos, clasificador)

    if cant_valores_distintos == 0:
        nodo.clasificacion = 'NULO'
        nodo.es_hoja = True
        return nodo
    elif len(set_datos.muestras) == cant_valores_distintos:
        nodo.clasificacion = set_datos.muestras[0][-1]
        nodo.es_hoja = True
        return nodo
    else:
        nodo.es_hoja = False

    atributo_bif = None  # atributo sobre el que se realizara la bifurcacion
    ganancia_max = 0  # ganancia maxima de informacion que podra obtener el atributo
    atributo_bif_value = None  # valor del atributo sobre el que se bifurcara
    ganancia_min = 0.0001  # ganancia minima de informacion que podra obtener el atributo

    # se procede a calcular la entropia del set de datos
    entropia_set_datos = calcular_entropia(set_datos, clasificador)
    for indice_atributo in range(len(set_datos.muestras[0])):

        if set_datos.atributos[indice_atributo] != clasificador:
            ganancia_max_temp = 0
            atributo_bif_value_temp = None

            # conjunto de valores sobre los que se podran realizar bifurcaciones
            lista_valores_atributos = [muestra[indice_atributo] for muestra in set_datos.muestras]

            # remover valores de atributos redundates
            lista_valores_atributos = list(set(lista_valores_atributos))

            # if len(lista_valores_atributos) > 100:
            #     lista_valores_atributos = sorted(lista_valores_atributos)
            #     total = len(lista_valores_atributos)
            #     porcentaje = int(total / 10)
            #     nuevos_valores = []
            #     for x in range(0, 9):
            #         nuevos_valores.append(lista_valores_atributos[x * porcentaje])
            #         lista_valores_atributos = nuevos_valores

            for val in lista_valores_atributos:
                # calcular la ganancia de informacion si se bifurca en este valor
                ganancia_local = calcular_ganancia(set_datos, entropia_set_datos, val, indice_atributo)

                # si la ganancia es mejor que la ganancia_max_temp, se guarda la ganancia y el valor
                if ganancia_local >= ganancia_max_temp:
                    ganancia_max_temp = ganancia_local
                    atributo_bif_value_temp = val

            if ganancia_max_temp >= ganancia_max:
                ganancia_max = ganancia_max_temp
                atributo_bif_value = atributo_bif_value_temp
                atributo_bif = indice_atributo

    # atributo_bif es el mejor atributo de acuerdo al calculo de la ganancia de informacion

    if ganancia_max <= ganancia_min or nodo.peso > 20\
            or atributo_bif_value is None or atributo_bif is None:
        nodo.es_hoja = True
        nodo.clasificacion = clasificar_hoja(set_datos, clasificador)
        return nodo

    nodo.atributo_bif_index = atributo_bif
    nodo.atributo_bif = set_datos.atributos[atributo_bif]
    nodo.atributo_bif_value = atributo_bif_value

    # se crean dos conjuntos de datos, que seran las bifurcaciones del nodo
    set_datos_der = Datos(clasificador)
    set_datos_izq = Datos(clasificador)

    # se le anhaden el conjunto de atributos que tienen las muestras
    set_datos_der.atributos = set_datos.atributos
    set_datos_izq.atributos = set_datos.atributos

    # se le agregan los tipos de atributos que tienen las muestras
    set_datos_der.tipos_atributos = set_datos.tipos_atributos
    set_datos_izq.tipos_atributos = set_datos.tipos_atributos

    for muestra in set_datos.muestras:
        if atributo_bif is not None and atributo_bif_value is not None:
            if atributo_bif is not None and muestra[atributo_bif] >= atributo_bif_value:
                set_datos_der.muestras.append(muestra)
            elif atributo_bif is not None:
                set_datos_izq.muestras.append(muestra)

    # llamada recursiva para crear los siguientes nodos
    nodo.hijo_der = crear_arbol_decision(set_datos_der, nodo, clasificador)
    nodo.hijo_izq = crear_arbol_decision(set_datos_izq, nodo, clasificador)

    # se devuelve el nodo raiz del arbol de decision que ha sido creado
    return nodo


# -------------------------------------------------------------------------------

def clasificar_hoja(set_datos, clasificador):
    """
    Funcion utilizada para determinar el valor que toma una dterminada hoja
    :param set_datos: conjunto de muestras, atributos y demas datos sobre los que se determinara
    el valor de clasificacion de la hoja
    :param clasificador: es el atributo meta sobre el que se basa el modelo
    :return: valor de clasificacion que se le asignara al nodo hoja
    """

    cant_valores_distintos = contar_valores_distintos(set_datos.muestras, set_datos.atributos, clasificador)
    total = len(set_datos.muestras)
    diferencia = total - cant_valores_distintos
    if cant_valores_distintos >= diferencia:
        return 1
    else:
        return 0


# -------------------------------------------------------------------------------

def calcular_entropia(set_datos, clasificador):
    """
    Calcula el valor de entropia del set de datos que son pasados por parametros, esto segun la formula
    de entropia que aparece en el libro IA A Modern Approach
    :param set_datos: es el conjunto de datos sobre los que se determinara la entropia
    :param clasificador: es el atributo meta sobre el que se basa el modelo
    :return: valor de entropia que se obtiene a partir de los datos de entrada
    """

    cant_valores_distintos = contar_valores_distintos(set_datos.muestras, set_datos.atributos, clasificador)
    total_muestras = len(set_datos.muestras)

    entropia = 0
    p = cant_valores_distintos / total_muestras
    if p != 0:
        entropia += p * math.log(p, 2)
    p = (total_muestras - cant_valores_distintos) / total_muestras
    if p != 0:
        entropia += p * math.log(p, 2)

    entropia = -entropia
    return entropia


# -------------------------------------------------------------------------------

def calcular_ganancia(set_datos, entropia, val, indice_atributo):
    """
    Funcion utilizada para calcular la ganancia de informacion de un determinado
    valor dentro del conjunto de datos
    :param set_datos: conjunto de datos sobre los que se determina la ganancia de informacion del atributo
    :param entropia: valor de entropia que se obtiene del set de datos con que se esta constuyendo el arbol
    :param val: valor al que se le determinara la ganancia de informacion
    :param indice_atributo: posicion que tiene en el conjunto de atributos, la columna sobre la que se esta
    determinando la ganancia de informacion
    :return: valor de ganancia de informacion
    """

    clasificador = set_datos.atributos[indice_atributo]
    entropia_atributo = 0.01
    total_muestras = len(set_datos.muestras)

    ganancia_set_der = Datos(clasificador)
    ganancia_set_izq = Datos(clasificador)

    ganancia_set_der.atributos = set_datos.atributos
    ganancia_set_izq.atributos = set_datos.atributos

    ganancia_set_der.tipos_atributos = set_datos.tipos_atributos
    ganancia_set_izq.tipos_atributos = set_datos.tipos_atributos

    for muestra in set_datos.muestras:
        if muestra[indice_atributo] >= val:
            ganancia_set_der.muestras.append(muestra)
        elif muestra[indice_atributo] < val:
            ganancia_set_izq.muestras.append(muestra)

    # Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
    if len(ganancia_set_der.muestras) == 0 or len(ganancia_set_izq.muestras) == 0:
        return -1

    entropia_atributo += calcular_entropia(ganancia_set_der, clasificador) * len(
        ganancia_set_der.muestras) / total_muestras

    entropia_atributo += calcular_entropia(ganancia_set_izq, clasificador) * len(
        ganancia_set_izq.muestras) / total_muestras

    return abs(entropia - entropia_atributo)


# -------------------------------------------------------------------------------

def get_nombre_atributos(indice_atributo, muestras):
    """
    Funcion utilizada para obtener el conjunto de valores posibles
    que puede obtener un determinado atributo según el conjunto de
    datos observados con que se esta entrenando el modelo
    :param indice_atributo: Es el indice del atributo sobre el cual se tomara la columna respectiva
    y se determinaran los posibles valores que pueden obtener las muestras
    :param muestras: es el conjunto de muestras observadas
    :return: lista de valores que se le pueden asignar al atributo
    """

    # lista_atributos = []

    lista_atributos = [muestra[indice_atributo] for muestra in muestras]

    # remover valores de atributos redundates
    lista_atributos = list(set(lista_atributos))

    #for i in range(len(muestras)):
    #    if lista_atributos.count(muestras[i][indice_atributo]) == 0:
    #        lista_atributos.append(muestras[i][indice_atributo])

    return lista_atributos


# -------------------------------------------------------------------------------

def contar_valores_distintos(muestras, atributos, clasificador):
    """
    Funcion utilizada para contar la cantidad de valores distintos
    que se pueden asignar a un determinado atributo
    :param muestras: conjunto de muestras observadas con las que se esta entrenando el sistema
    :param atributos: conjunto de etiquetas que representan los atributos del set de datos
    :param clasificador: es el atributo meta sobre el que se basa el modelo
    :return: cantidad de valores posibles que se le pueden asignar
    """

    indice_clasificador = atributos.tolist().index(clasificador)

    lista_atributos = get_nombre_atributos(indice_clasificador, muestras)

    cantidad = len(lista_atributos)

    return cantidad


# -------------------------------------------------------------------------------

def podar_arbol(nodo_raiz, nodo, set_datos, mejor_puntaje):
    """
    Funcion de poda del arbol, que se encarga de eliminar nodos
    que no aportan demasiada informacion al arbol de decision
    :param nodo_raiz: nodo raiz del arbol
    :param nodo: es el mismo nodo raiz del arbol
    :param set_datos: es el conjunto de datos con que se entreno el modelo
    :param mejor_puntaje: es el valor actual de precision que tiene el modelo,
    y sobre el cual se intentara mejorar
    :return: nuevo puntaje de precision que obtiene el arbol
    """

    # si el nodo es una hoja
    if nodo.es_hoja:

        # ejecutar la funcion validar_arbol sobre el arbol
        # con nodo_padre como la hoja con la clasificacion actual
        nodo.nodo_padre.es_hoja = True
        nodo.nodo_padre.clasificacion = nodo.clasificacion

        if nodo.peso < 20:
            nuevo_puntaje, predic = validar_arbol(nodo_raiz, set_datos)
        else:
            nuevo_puntaje = 0

        # si el nuevo puntaje es mejor, cambiarlos
        if nuevo_puntaje >= mejor_puntaje:
            return nuevo_puntaje
        else:
            nodo.nodo_padre.es_hoja = False
            nodo.nodo_padre.clasificacion = None
            return mejor_puntaje

    # si no es una hoja
    else:
        # podar el arbol con el hijo derecho
        nuevo_puntaje = podar_arbol(nodo_raiz, nodo.hijo_der, set_datos, mejor_puntaje)

        # si es una hoja, devolverlo
        if nodo.es_hoja:
            return nuevo_puntaje

        # podar el arbol con el hijo izquierdo
        nuevo_puntaje = podar_arbol(nodo_raiz, nodo.hijo_izq, set_datos, nuevo_puntaje)

        # si es una hoja, devolverlo
        if nodo.es_hoja:
            return nuevo_puntaje

        return nuevo_puntaje


# -------------------------------------------------------------------------------

def validar_arbol(nodo, set_datos):
    """
    Funcion utilizada para validar el arbol de decision
    que se construyo con el conjunto de datos de entrenamiento
    :param nodo: nodo raiz del arbol de decision construido
    :param set_datos: conjunto de datos utilizados en el modelo creado
    :return: porcentaje de precision (correctos/ total) del proceso de validacion
    """

    total = len(set_datos.muestras)
    correctos = 0
    predicciones = []

    for muestra in set_datos.muestras:
        resultado = validar_muestra(nodo, muestra)
        correctos += resultado[0]  # validar la muestra
        predicciones.append(resultado[1])

    return correctos / total, predicciones


# -------------------------------------------------------------------------------

def validar_muestra(nodo, muestra):
    """
    valida si el valor arrojado por el arbol de
    decision es el mismo que representa a la muestra
    :param nodo: nodo raiz del arbol de decision que ya se encuentra entrenado
    :param muestra: datos del votante y el voto efectuado por el mismo,
    con el que se evaluara la precision del arbol de decision
    :return: valor de 1 o 0 para indicar si el resultado obtenido era el esperado o no
    """

    if nodo.es_hoja:
        calif_proyectada = nodo.clasificacion
        calif_actual = muestra[-1]
        if calif_proyectada == calif_actual:
            return 1, calif_proyectada
        else:
            return 0, calif_proyectada

    value = muestra[nodo.atributo_bif_index]
    if value >= nodo.atributo_bif_value:
        return validar_muestra(nodo.hijo_der, muestra)
    else:
        return validar_muestra(nodo.hijo_izq, muestra)


# -------------------------------------------------------------------------------

def probar_muestra(muestra, nodo, indice_clasificador):
    """
    Funcion de prueba, que predice para una muestra determinada,
    el valor del atributo meta que se le asignara
    :param muestra: es la muestra observada a la que se le esta prediciendo el voto
    :param nodo: nodo raiz del arbol de decision
    :param indice_clasificador: es el indice de la posicion que ocupa el atributo meta
    :return: clasificacion que se le asignara a la muestra probada
    """

    if nodo.es_hoja:
        return nodo.clasificacion
    else:
        if muestra[nodo.atributo_bif_index] >= nodo.atributo_bif_value:
            return probar_muestra(muestra, nodo.hijo_der, indice_clasificador)
        else:
            return probar_muestra(muestra, nodo.hijo_izq, indice_clasificador)


# -------------------------------------------------------------------------------

def imprimir_arbol(nodo):
    """
    Funcion utlizada para imprimir el arbol de decision
    :param nodo: nodo raiz del arbol de decision
    :return: No aplica
    """
    if nodo.es_hoja:
        for x in range(nodo.peso):
            print("\t", )
        print("clasificacion: " + str(nodo.clasificacion))
        return
    for x in range(nodo.peso):
        print("\t", )
    print("Split index: " + str(nodo.atributo_bif))
    for x in range(nodo.peso):
        print("\t", )
    print("Split value: " + str(nodo.atributo_bif_value))
    imprimir_arbol(nodo.hijo_der)
    imprimir_arbol(nodo.hijo_izq)


# -------------------------------------------------------------------------------

muestra_entrenamiento = [
    ['SAN JOSE', '1', '25', 'SI', 'PAC'],
    ['HEREDIA', '1', '25', 'NO', 'PUSC'],
    ['CARTAGO', '0', '25', 'SI', 'PAC'],
    ['HEREDIA', '0', '22', 'NO', 'PLN'],
    ['CARTAGO', '0', '22', 'NO', 'PLN'],
    ['SAN JOSE', '1', '25', 'NO', 'PUSC'],
    ['CARTAGO', '1', '22', 'SI', 'RES'],
    ['HEREDIA', '1', '50', 'SI', 'RES'],
    ['HEREDIA', '0', '50', 'NO', 'RES'],
    ['HEREDIA', '1', '25', 'SI', 'RES'],
    ['CARTAGO', '0', '25', 'NO', 'RES'],
    ['HEREDIA', '0', '22', 'SI', 'RES'],
    ['CARTAGO', '0', '22', 'SI', 'RES']
]

muestra_validacion = [
    ['SAN JOSE', '1', '25', 'SI', 'PAC'],
    ['CARTAGO', '0', '22', 'NO', 'PAC'],
    ['HEREDIA', '1', '50', 'NO', 'RES'],
    ['SAN JOSE', '1', '50', 'SI', 'RES'],
    ['HEREDIA', '1', '50', 'NO', 'PUSC'],
    ['PUNTARENAS', '1', '22', 'SI', 'PLN'],
    ['HEREDIA', '0', '25', 'NO', 'PAC'],
    ['HEREDIA', '1', '25', 'NO', 'PAC']
]

header = ['PROVINCIA', 'TRABAJADOR', 'EDAD', 'SOLTERO', 'VOTO']


# -----------------------------------------------------------------------------

def cross_validation(muestras_entrenamiento, atributos, k_segmentos=10):
    precisiones = list()
    predicciones = list()
    mejor_arbol = None
    mejor_precision = 0

    indices = list(range(len(muestras_entrenamiento)))
    segmentos = segmentar(indices, k_segmentos)

    # Se hacen k iteraciones. En cada iteración se usa un segmento
    # diferente para validación, el resto se usa para entrenamiento
    for k in range(len(segmentos)):

        print('\nUtilizando el segmento ', k + 1, ' para realizar la validacion...')

        ind_validacion, ind_entrenamiento = agrupar(k, segmentos)

        datos_validacion = muestras_entrenamiento[ind_validacion]

        datos_entrenamiento = muestras_entrenamiento[ind_entrenamiento]

        set_datos = Datos("")
        set_datos.muestras = datos_entrenamiento
        set_datos.atributos = atributos
        clasificador = set_datos.atributos[-1]  # GOAL
        set_datos.clasificador = clasificador
        set_datos.tipos_atributos = np.zeros(datos_entrenamiento.shape[1], dtype=bool).tolist()

        for a in range(len(set_datos.atributos)):
            if set_datos.atributos[a] == set_datos.clasificador:
                set_datos.indice_clasificador = a
            else:
                set_datos.indice_clasificador = \
                    range(len(set_datos.atributos))[-1]

        nodo_raiz = crear_arbol_decision(set_datos, None, clasificador)

        validateset = Datos(clasificador)
        validateset.muestras = datos_validacion
        validateset.atributos = atributos
        validateset.tipos_atributos = np.zeros(datos_validacion.shape[1], dtype=bool).tolist()

        for a in range(len(validateset.atributos)):
            if validateset.atributos[a] == validateset.clasificador:
                validateset.indice_clasificador = a
            else:
                validateset.indice_clasificador = range(len(validateset.atributos))[-1]

        precision, etiqs_predics = validar_arbol(nodo_raiz, validateset)

        print('\t -> Precision obtenida ', precision)
        print('\t -> Mejor precision actual ', mejor_precision)

        if mejor_precision < precision:
            print('\t****** Se encontro una mejor precision ******\n\tSe actualiza el arbol y la precision')
            mejor_precision = precision
            mejor_arbol = nodo_raiz

        predicciones += etiqs_predics

        precisiones.append(precision)

    return mejor_arbol, np.mean(precisiones), predicciones


# -----------------------------------------------------------------------------

def analisis_arbol_decision(args, muestras):
    print('\nGenerando Analisis del Arbol de Decision')
    umbral_poda = args.umbral_poda
    porcentaje_pruebas = args.porcentaje_pruebas[0]
    prefijo_archivos = "arbol" if args.prefijo is None else args.prefijo[0]

    # Para agregar las 4 columnas solicitadas
    salida = np.concatenate((muestras, np.zeros((muestras.shape[0], 4))), axis=1)

    indices = list(range(muestras.shape[0]))
    ind_pruebas, ind_entrenamiento = separar(indices, porcentaje_pruebas)

    # Se llena la columna 'es_entrenamiento'
    salida[ind_pruebas, 23] = 0
    salida[ind_entrenamiento, 23] = 1

    for n_ronda in range(1, 4):

        predicciones = list()

        if n_ronda == 3:
            print('\nComenzando prediccion ronda 2 con primeras votaciones')
        else:
            print('\nComenzando prediccion ronda ', n_ronda, ' sin la otra ronda')

        print('Dividiendo los datos pruebas-entrenamiento')
        muestras_ronda, atributos = preprocesar_ronda(muestras, ronda=n_ronda)

        muestras_pruebas = muestras_ronda[ind_pruebas]
        muestras_entrenamiento = muestras_ronda[ind_entrenamiento]

        print('Entrenando el modelo')
        arbol, precision, predic_generadas = cross_validation(muestras_entrenamiento, atributos, k_segmentos=10)

        predicciones += predic_generadas

        print('\nProbando el modelo')
        predicciones_prueba = generar_test(muestras_pruebas, atributos, arbol)

        predicciones = predicciones_prueba + predicciones

        salida[:, 23 + n_ronda] = predicciones

    salida = pd.DataFrame(salida, columns=columnas_salida)

    print('\nGenerando archivo de salida')
    # Se guarda el archivo con las 4 columnas de la especificación
    nombre_salida = os.path.join("archivos", prefijo_archivos + ".csv")
    salida.to_csv(nombre_salida)

    print('\tVer resultados en:\n\t', nombre_salida)
    return None


# -----------------------------------------------------------------------------

def generar_test(muestras, atributos, arbol):
    print('\t Con una cantidad de ', len(muestras), 'muestra(s)')

    testset = Datos("")
    testset.muestras = muestras
    testset.atributos = atributos
    clasificador = testset.atributos[-1]  # GOAL
    testset.clasificador = clasificador
    testset.tipos_atributos = [False] * len(atributos)

    for a in range(len(testset.atributos)):
        if testset.atributos[a] == testset.clasificador:
            testset.indice_clasificador = a
        else:
            testset.indice_clasificador = range(len(testset.atributos))[-1]

    for muestra in testset.muestras:
        muestra[testset.indice_clasificador] = probar_muestra(muestra, arbol, testset.indice_clasificador)

    print('\tPredicciones finalizadas')
    return testset.muestras[:, -1].tolist()
