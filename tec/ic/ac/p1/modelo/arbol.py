import math


# ---------------------------------------------------------------------

def entropia(q):
    """
    Funcion que se encarga de determinar la geomancia de informacion de una cierta variable
    Se utiliza para la seleccion de atributos, permitiendo seleccionar los que redezcan la incertidumbre
    de clasificacion.
    :param q: variable a la cual se le calculara el valor de entropia, para determinar el grado de incertidumbre
    con que cuenta, y asi poder utilizar el calculo en la ganancia de información final
    :return: valor de entropia de la variable evaluada
    """

    valor_entropia = -(q * math.log(q, 2) + (1-q) * math.log(1-q, 2))

    return valor_entropia


# ---------------------------------------------------------------------

def ganancia(muestra):
    """
    Funcion para determinar la ganancia de informacion que brinda una determinada variable o atributo
    :param muestra: es el conjunto de datos de una determinada variable de la cual se obtendra la ganancia
    de informacion
    :return: valor con la ganancia de informacion
    """

    proporcion_pos = porporcion_positivos(muestra)
    valor_entropia = entropia(proporcion_pos)

    return valor_entropia - resto(muestra)


# ---------------------------------------------------------------------

def resto(conjunto_muestras):
    largo_muestra = len(conjunto_muestras)

    resultado = 0
    for i in range(0, largo_muestra-1):
        muestra = conjunto_muestras[i]

    return resultado


# ---------------------------------------------------------------------

def porporcion_positivos(muestra):
    """
    Funcion encargada de determinar la porporcion de positivos que el conjunto de muestras
    :param muestra: Es el conjunto de observaciones que se van a utilizar para entrenar el sistema
    de las cuales se determinara la proporcion de positivos
    :return: porcentaje de positivos (del atributo meta) que contiene el conjunto de muestras
    """

    total_muestras = len(muestra)
    positivos = 0
    for i in range(total_muestras):
        muestra_aux = muestra[i]
        dato_meta = muestra_aux[-1]
        if dato_meta:
            positivos += 1

    return positivos / total_muestras


# ---------------------------------------------------------------------
# Ejemplos y pruebas
# ---------------------------------------------------------------------

array = [
    [1,2,3,4,5,True],
    [1,1,3,4,5,False],
    [1,9,1,4,8,True],
    [1,2,3,3,4,False],
    [1,7,2,4,5,False],
    [1,7,2,4,5,True]
]

ie_rest = [
    [True],
    [False],
    [True],
    [True],
    [False],
    [True],
    [False],
    [True],
    [False],
    [False],
    [True],
    [True]
]


q = porporcion_positivos(ie_rest)
print("Proporcion posiivos ", q)

print("Entropia ", entropia(q))