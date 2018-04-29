# -----------------------------------------------------------------------------

"""

Clasificador K-Nearest-Neightbor utilizando kd-Tree

    ♣ Basado en la sección 18.8.2 del Libro AI-A Modern Approach, 737
    ♣ Se sigue el ejemplo de la siguiente página https://goo.gl/TGQaEe

    ♦ One Hot Encoding para atributos categóricos
    ♦ Normalización para que todos los atributos tengan la misma escala

NOTAS:

    ♣ Resulta mejor cuando hay más ejemplos que dimensiones
    ♣ Para clasificar se usa el voto por mayoría de los k vecinos más cercanos

    ♣ TODO: Encontrar el k más óptimo con cross-validation
    ♣ k debe ser impar para evitar empates (pág 738)

"""

import sys
sys.path.append('..')

import numpy as np
from pprint import pprint
from util.timeit import timeit

# -----------------------------------------------------------------------------

# Se ha elegido un diccionario para crear el kd-Tree.

# Tokens para obtener el valor correspondiente en un nodo.
# Se antepone una letra para que las propiedades de ese nodo
# sean mostradas en ese orden. Esto con fines de depuración.

ATRIBUTO = 'A. ATRIBUTO_USADO'
VALOR_ATRIBUTO = 'B. VALOR_USADO'
VECTOR_MEDIO = 'C. VECTOR_MEDIO'
DATOS_IZQUIERDA = 'D. NODO_IZQUIERDO'
DATOS_DERECHA = 'E. NODO_DERECHO'

# -----------------------------------------------------------------------------


def kdtree(datos, max_profundidad=None):

    """
    Crea el kd-Tree como un diccionario anidado.
    En cada nodo se parten los datos con base en el atributo que más varia.

    Se puede ver una demostración de la estructura del árbol (con datos
    triviales) ejecutando este archivo como el principal:
        >> python nearest_neighbors.py

    :param datos: Lista de listas o np.array donde todos los elementos
    son númericos.
    :param max_profundidad: Representa la cantidad de niveles máx del árbol
    :return: kdtree_aux(datos, atributos_no_usados)
    """

    if max_profundidad is 0:
        return None

    if type(datos) is list:
        datos = np.array(datos)

    # Para que todos los atributos tengan la misma escala
    # datos = normalizar(datos)

    tamanho_datos = len(datos)
    cant_atributos = len(datos[0])

    if max_profundidad is None:
        max_profundidad = cant_atributos

    # Máscara para saber cuales atributos ya ha sido usados y no repetir
    atributos_no_usados = np.ones(cant_atributos, dtype=bool)

    # Si no cumple retorna None implícitamente
    if tamanho_datos > 1:
        return kdtree_aux(datos, atributos_no_usados, max_profundidad)

# -----------------------------------------------------------------------------


def kdtree_aux(datos, atributos_no_usados, max_profundidad):

    """
    Función de apoyo para kdtree(datos). Esto para que dicha funcíón no
    tenga que recibir el argumento `atributos_no_usados`.

    :param datos: np.array donde todos los elementos son númericos
    :param atributos_no_usados: np.array con elementos booleanos.
    Un True representa que el atributo en esa misma posición en los datos
    no ha sido utilizado para dividir un nodo.
    :param max_profundidad: Representa la cantidad de niveles máx del árbol

    :return: Diccionario con la siguiente información:
        ATRIBUTO: índice de la columna utilizada para dividir
        INDICE_MEDIO: índice de la fila utilizada para dividir
        VECTOR_MEDIO: fila en la posición `INDICE_MEDIO`
        DATOS_IZQUIERDA: datos donde atributo es < al del vector medio
        DATOS_DERECHA: datos donde el atributo es > al del vector medio
    """

    tamanho_datos = len(datos)

    if tamanho_datos < 1:
        return None

    if tamanho_datos == 1 or max_profundidad <= 1:
        return datos

    # Si no cumple entonces retorna un None
    if tamanho_datos > 1:

        # Se copia para no alterar el del padre
        atributos_no_usados = np.copy(atributos_no_usados)

        # Se selecciona el atributo con mayor varianza
        # y que no ha sido usado por el nodo padre
        atributo = seleccionar(datos, atributos_no_usados)

        # Se marca el atributo seleccionado como usado
        atributos_no_usados[atributo] = False

        # Se ordenan con base al atributo seleccionado
        datos = ordenar(datos, atributo)

        # Se parte el árbol según al atributo seleccionado
        nodo = dividir(datos, atributo)
        izquierda = nodo[DATOS_IZQUIERDA]
        derecha = nodo[DATOS_DERECHA]

        # Se aplica recursivamente en los hijos
        nodo[DATOS_IZQUIERDA] = \
            kdtree_aux(izquierda, atributos_no_usados, max_profundidad - 1)

        nodo[DATOS_DERECHA] = \
            kdtree_aux(derecha, atributos_no_usados, max_profundidad - 1)

        return nodo

# -----------------------------------------------------------------------------


def knn(arbol, consulta, k_vecinos, recorrido=False):
    """
    Busca los `k_vecinos` más cercanos y sus distancias.
    :param arbol: Diccionario kd-Tree previamente creado
    :param consulta: np.array con la misma estructura que tienen
    las muestras que se utilizaron para crear el kd-tree
    :param k_vecinos: Cantidad máxima de vecinos más cercanos
    :param recorrido: True si se desea obtener la lista de nodos
    que se tuvieron que recorrer antes de llegar a la hoja (sin incluirla)
    :return: (vecinos_mas_cercanos, distancias_de_cada_vecino)
    """

    vecinos = list()
    nodo_actual = arbol

    # Se recorren los nodos hasta llegar a una hoja
    while type(nodo_actual) == dict:

        vecino = nodo_actual[VECTOR_MEDIO]
        vecinos.append(vecino)

        # Indíce del atributo que se usó para dividir
        atributo = nodo_actual[ATRIBUTO]

        v_atrib_consulta = consulta[atributo]
        v_atrib_actual = nodo_actual[VALOR_ATRIBUTO]

        # Si hay empate se testea la distancia entre la consulta
        # y los dos hijos. Se elige el camino con menor distancia
        if v_atrib_consulta == v_atrib_actual:

            test_izq = nodo_actual[DATOS_IZQUIERDA][VECTOR_MEDIO]
            dist_izq = distancia(consulta[atributo], test_izq)

            test_der = nodo_actual[DATOS_DERECHA][VECTOR_MEDIO]
            dist_der = distancia(consulta[atributo], test_der)

            # Si la distancia entre la consulta y el hijo de la izquierda
            # es menor que el de la derecha, se resta un 1 para que funcione
            # en el siguiente `if`
            v_atrib_consulta += -1 if dist_izq <= dist_der else 1

        if v_atrib_consulta < v_atrib_actual:
            nodo_actual = nodo_actual[DATOS_IZQUIERDA]

        else:
            nodo_actual = nodo_actual[DATOS_DERECHA]

    # Si ya no es un diccionario, se trata de una hoja
    # que en este caso es una lista, de lo contrario puede ser
    # None por lo que no hay que aplicar ninguna operación
    if type(nodo_actual) != list:

        vecinos = np.array(vecinos)
        nodos_recorridos = np.copy(vecinos)
        vecinos = np.append(nodo_actual, vecinos, axis=0)
        mas_cercanos, distancias = cercanos(vecinos, consulta, k_vecinos)

        if not recorrido:
            return mas_cercanos, distancias
        else:
            return mas_cercanos, distancias, nodos_recorridos


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
    >>> resultado[VECTOR_MEDIO]
    array([16, 44, 27])
    >>> resultado[DATOS_IZQUIERDA]
    array([[22, 38, 21],
           [20, 50, 24]])
    >>> resultado[DATOS_DERECHA]
    array([[14, 62, 30],
           [18, 56, 33]])
    """

    tamanho_datos = len(datos)
    mitad = int(tamanho_datos // 2)

    return {
        ATRIBUTO: atributo,
        VALOR_ATRIBUTO: datos[mitad][atributo],
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


def cercanos(datos, consulta, k_vecinos=1):
    """
    Obtiene lax filas que se parece más al vector de entrada.

    :param datos: 2D np.array
    Para este caso los datos son los que tiene un nodo en específico

    :param consulta: 1D np.array
    :param k_vecinos: Cantidad máxima de vecinos a retornar
    :return:
        ♣ 2D np.array, más cercanos dentro de los `datos`
        ♣ Lista de distancias en el mismo orden

    Ejemplos:
    >>> datos = np.array([[  1,  10,   3],
    ...                   [ 34,  56,  43],
    ...                   [123,   9, 120]])
    >>> vecinos, distancias = cercanos(datos, np.array([50, 40, 30]))
    >>> vecinos
    array([[34, 56, 43]])
    >>> distancias
    array([26.0959767])
    >>> vecinos, distancias = cercanos(datos, np.array([80, 5, 100]), 2)
    >>> vecinos
    array([[123,   9, 120],
           [ 34,  56,  43]])
    >>> distancias
    array([47.59201614, 89.25245095])
    """

    # Se calcula todas las distancias
    distancias = [distancia(consulta, vector) for vector in datos]
    distancias = np.array(distancias)

    # Se ordenan las distancias de menor a mayor y se toman las
    # primeras igual a `k_vecinos`
    orden = np.argsort(distancias)

    return datos[orden][0:k_vecinos], distancias[orden][0:k_vecinos]

# -----------------------------------------------------------------------------


@timeit
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

    tree = kdtree(dt, max_profundidad=4)
    print("Kd-Tree")
    pprint(tree)

    consulta = np.array([14, 62, 30])
    print("Consulta con k = 2: " + str(consulta))

    # Vecinos, distancias, nodos_recorridos
    v, d, r = knn(tree, consulta, k_vecinos=2, recorrido=True)
    print("K vecinos más cercanos: \n" + str(v))
    print("Distancias de los k vecinos: " + str(d))
    print("Nodos recorridos: " + str(r))


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    ejemplo_kdtree()

# -----------------------------------------------------------------------------
