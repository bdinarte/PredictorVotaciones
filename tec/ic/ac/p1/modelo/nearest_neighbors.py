# -----------------------------------------------------------------------------

"""

Clasificador K-Nearest-Neightbor utilizando kd-Tree

    ♣ Basado en la sección 18.8.2 del Libro AI-A Modern Approach, 737
    ♣ Se sigue el ejemplo de la siguiente página https://goo.gl/TGQaEe

    ♦ One Hot Encoding para atributos categóricos
    ♦ Normalización para que todos los atributos tengan la misma escala

NOTAS:

    ♣ Resulta mejor cuando hay más ejemplos que dimensiones
    ♣ k también representa el número de dimensiones del kd-Tree

    ♣ TODO: Encontrar el k más óptimo con cross-validation
    ♣ k debe ser impar para evitar empates (pág 738)

"""

import numpy as np
from pprint import pprint

# -----------------------------------------------------------------------------

# Se ha elegido un diccionario para crear el kd-Tree.

# Tokens para obtener el valor correspondiente en un nodo.
# Se antepone una letra para que las propiedades de ese nodo
# sean mostradas en ese orden. Esto con fines de depuración.

ATRIBUTO = "A. ATRIBUTO_USADO"
INDICE_MEDIO = "B. INDICE_MEDIO"
VECTOR_MEDIO = "C. VECTOR_MEDIO"
DATOS_IZQUIERDA = "D. NODO_IZQUIERDO"
DATOS_DERECHA = "E. NODO_DERECHO"

# -----------------------------------------------------------------------------


def kdtree(datos):

    """
    Crea el kd-Tree como un diccionario anidado.
    En cada nodo se parten los datos con base en el atributo que más varia.

    Se puede ver una demostración de la estructura del árbol (con datos
    triviales) ejecutando este archivo como el principal:
        >> python nearest_neighbors.py

    :param datos: Lista de listas o np.array donde todos los elementos
    son númericos.
    :return: kdtree_aux(datos, atributos_no_usados)
    """

    # Para que todos los atributos tengan la misma escala
    datos = normalizar(datos)

    tamanho_datos = len(datos)
    cant_atributos = len(datos[0])

    # Máscara para saber cuales atributos ya ha sido usados y no repetir
    atributos_no_usados = np.ones(cant_atributos, dtype=bool)

    # Si no cumple retorna None implícitamente
    if tamanho_datos > 1:
        return kdtree_aux(datos, atributos_no_usados)

# -----------------------------------------------------------------------------


def kdtree_aux(datos, atributos_no_usados):

    """
    Función de apoyo para kdtree(datos). Esto para que dicha funcíón no
    tenga que recibir el argumento `atributos_no_usados`.

    :param datos: Lista de listas con todos los elementos númericos
    :param atributos_no_usados: np.array con elementos booleanos.
    Un True representa que el atributo en esa misma posición en los datos
    no ha sido utilizado para dividir un nodo.

    :return: Diccionario con la siguiente información:
        ATRIBUTO: índice de la columna utilizada para dividir
        INDICE_MEDIO: índice de la fila utilizada para dividir
        VECTOR_MEDIO: fila en la posición `INDICE_MEDIO`
        DATOS_IZQUIERDA: datos donde atributo es < al del vector medio
        DATOS_DERECHA: datos donde el atributo es > al del vector medio
    """

    # Si no cumple entonces retorna None de manera implícita
    if len(datos) > 1:

        # Se copia para no alterar el del padre
        atributos_no_usados = np.copy(atributos_no_usados)

        # Se selecciona el atributo con mayor varianza
        # y que no ha sido usado por el nodo padre
        atributo = seleccionar(datos, atributos_no_usados)

        # Se marca el atributo seleccionado como usado
        atributos_no_usados[atributo] = False

        # Se ordenan con base al atributo selecionado
        datos = ordenar(datos, atributo)

        # Se parte el árbol según al atributo seleccionado
        nodo = dividir(datos, atributo)
        izquierda = nodo[DATOS_IZQUIERDA]
        derecha = nodo[DATOS_DERECHA]

        # Se aplica recursivamente en los hijos
        nodo[DATOS_IZQUIERDA] = kdtree_aux(izquierda, atributos_no_usados)
        nodo[DATOS_DERECHA] = kdtree_aux(derecha, atributos_no_usados)

        return nodo

# -----------------------------------------------------------------------------


def normalizar(datos, menor=0.0, mayor=1.0):
    """
    Normaliza los valores de la matriz para que todas la variables puedan
    competir utilizando la misma escala.

    NOTA: Es equivalente a:
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(datos)

    :param datos: 2D np.array
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


def seleccionar(datos, atributos_no_usados):
    """
    Selecciona el atributo que se usará para dividir el nodo.
    Se usa como criterio la columna que tiene mayor varianza.

    :param datos: 2D np.array
    Para este caso los datos son los que tiene un nodo en específico
    :param atributos_no_usados: Máscara binaria que representa cuales
    atributos no han sido usados (si no ha sido usado es un True)
    :return: índice de la columna con mayor varianza en los `datos`

    Ejemplos:
    >>> datos = np.array([[1,  5, 8],
    ...                   [3, 10, 7],
    ...                   [5, 15, 6],
    ...                   [7, 20, 5]])
    >>> no_usados = [True, True, True]
    >>> seleccionar(datos, no_usados)
    1
    >>> no_usados = [True, False, True]
    >>> seleccionar(datos, no_usados)
    0
    >>> no_usados = [False, False, True]
    >>> seleccionar(datos, no_usados)
    2
    """

    # Se aplica la máscara para solo obtener la varianza
    # de los atributos que no han sido utilizados
    varianzas = np.zeros(shape=len(datos[0]))
    datos_no_usados = datos[:, atributos_no_usados]
    varianzas[atributos_no_usados] = np.var(datos_no_usados, axis=0)

    return np.argmax(varianzas)

# -----------------------------------------------------------------------------


def ordenar(datos, columna):
    """
    Ordena de menor a mayor los datos (matriz) según los valores
    de una sola columna.

    :param datos: 2D np.array
    :param columna: Indíce de la columna
    :return: Los mismos datos de entrada pero ordenados

    Ejemplo:
    >>> datos = np.array([[  1, 10,   3],
    ...                   [ 34, 56,  43],
    ...                   [123,  9, 120]])
    >>> ordenar(datos, columna=1)
    array([[123,   9, 120],
           [  1,  10,   3],
           [ 34,  56,  43]])
    """

    columna = [columna]
    return datos[np.lexsort(datos.T[columna])]

# -----------------------------------------------------------------------------


def dividir(datos, atributo):
    """
    Divide los datos en dos usando un atributo como criterio.

    :param datos: 2D np.array
    Para este caso los datos son los que tiene un nodo en específico
    :param atributo: Indice de una columna de los `datos`
    Se usa como criterio para ordenar los datos y luego para hacer la división

    :return: Diccionario con la siguiente información:
        ATRIBUTO: índice de la columna utilizado para dividir,
        INDICE_MEDIO: índice de la fila utilizada para dividir,
        VECTOR_MEDIO: fila en la posición `INDICE_MEDIO`,
        DATOS_IZQUIERDA: datos donde atributo es < al del vector medio
        DATOS_DERECHA: datos donde el atributo es > al del vector medio

    :raises IndexError: En caso que `datos` este vacío

    Ejemplos:
    >>> atributo = 2
    >>> datos = np.array([[22, 38, 21],
    ...                   [20, 50, 24],
    ...                   [16, 44, 27], # <-- Se divide aquí
    ...                   [14, 62, 30],
    ...                   [18, 56, 33]])
    >>> resultado = dividir(datos, atributo)
    >>> resultado["VECTOR_MEDIO"]
    array([16, 44, 27])
    >>> resultado["DATOS_IZQUIERDA"]
    array([[22, 38, 21],
           [20, 50, 24]])
    >>> resultado["DATOS_DERECHA"]
    array([[14, 62, 30],
           [18, 56, 33]])
    """

    tamanho_datos = len(datos)
    mitad = int(tamanho_datos // 2)

    return {
        ATRIBUTO: atributo,
        INDICE_MEDIO: mitad,
        VECTOR_MEDIO: datos[mitad],
        DATOS_IZQUIERDA: datos[0: mitad],
        DATOS_DERECHA: datos[mitad + 1: tamanho_datos]
    }

# -----------------------------------------------------------------------------


def distancia(vector_x, vector_y):
    """
    Calcula la distancia euclidiana entre dos vectores.

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


def cercano(datos, vector_x):
    """
    Obtiene la fila de los datos que se parece más al vector de entrada.

    :param datos: 2D np.array
    Para este caso los datos son los que tiene un nodo en específico

    :param vector_x: 1D np.array
    :return: 1D np.array más cercano dentro de los `datos`

    Ejemplos:
    >>> datos = np.array([[  1,  10,   3],
    ...                   [ 34,  56,  43],
    ...                   [123,   9, 120]])
    >>> cercano(datos, np.array([50, 40, 30]))
    array([34, 56, 43])
    >>> cercano(datos, np.array([80, 5, 100]))
    array([123,   9, 120])
    """

    # El más cercano por el momento es la primera fila
    vector_mejor = datos[0]
    distancia_mejor = distancia(vector_x, datos[0])

    # Se empieza desde datos[1:] porque el [0] ya fue medido
    for vector_actual in datos[1:]:
        distancia_actual = distancia(vector_x, vector_actual)

        # Si se obtuvo menos distancia entonces es mejor
        if distancia_actual < distancia_mejor:
            distancia_mejor = distancia_actual
            vector_mejor = vector_actual

    return vector_mejor

# -----------------------------------------------------------------------------


def ejemplo_kdtree():

    """
    Crear un árbol con información trivial e imprimir su estructura de
    una manera humano-compatible

    Ls solución se puede verificar con la siguiente página:
        ♣ https://goo.gl/TGQaEe

    """

    dt = np.array([[22, 38, 21],
                   [4,   8,  6],
                   [2,  14,  3],
                   [8,  20, 12],
                   [10, 26, 18],
                   [12, 32, 15],
                   [18, 56, 33],
                   [16, 44, 27],
                   [20, 50, 24],
                   [14, 62, 30],
                   [6,   2,  9]])

    tree = kdtree(dt)
    pprint(tree)

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    ejemplo_kdtree()

# -----------------------------------------------------------------------------
