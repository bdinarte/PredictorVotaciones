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
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from pprint import pprint
from pc1.util.timeit import timeit

# -----------------------------------------------------------------------------

# Se ha elegido un diccionario para crear el kd-Tree.

# Tokens para obtener el valor correspondiente en un nodo.
# Se antepone una letra para que las propiedades de ese nodo
# sean mostradas en ese orden. Esto con fines de depuración.

ATRIBUTO = 'A. ATRIBUTO'
V_ATRIBUTO = 'B. V_ATRIBUTO'

MEDIANA = 'C. MEDIANA'
ETIQUETA = 'D. ETIQUETA'

IZQUIERDA = 'E. IZQUIERDO'
DERECHA = 'F. DERECHA'

# -----------------------------------------------------------------------------


def kdtree(muestras, max_profundidad=100):
    """
    Crea el kd-Tree como un diccionario anidado.
    En cada nodo se parten los matriz con base en el atributo que más varia.

    Se puede ver una demostración de la estructura del árbol (con matriz
    triviales) ejecutando este archivo como el principal:
        >> python nearest_neighbors.py

    :param muestras: Tupla donde:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    NOTA: Ambos deben ser del tipo np.array

    :param max_profundidad: Representa la cantidad de niveles máx del árbol
    Es para hacer menos exhaustiva la búsqueda a través de los nodos

    :return: kdtree_aux(matriz, atributos_no_usados)
    """

    if max_profundidad is 0:
        return None

    matriz = muestras[0]
    tamanho_datos = len(matriz)
    cant_atributos = len(matriz[0])

    if max_profundidad is None:
        max_profundidad = cant_atributos

    # Máscara para saber cuales atributos ya ha sido usados y no repetir
    atribs_no_usados = np.ones(cant_atributos, dtype=bool)

    # Si no cumple retorna None implícitamente
    if tamanho_datos > 1:
        return kdtree_aux(muestras, atribs_no_usados, max_profundidad)

# -----------------------------------------------------------------------------


def kdtree_aux(muestras, atribs_no_usados, max_profundidad):
    """
    Función de apoyo para kdtree(matriz). Esto para que dicha funcíón no
    tenga que recibir el argumento `atributos_no_usados`.

    :param muestras: Tupla donde:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    :param atribs_no_usados: np.array con elementos booleanos.
    Un True representa que el atributo en esa misma posición en los matriz
    no ha sido utilizado para bifurcar un nodo.

    :param max_profundidad: Representa la cantidad de niveles máx del árbol

    :return: Diccionario con la siguiente información:
        ATRIBUTO: índice de la columna usada para bifurcar
        V_ATRIBUTO: valor del atributo usado para bifurcar
        ETIQUETA: clase del atributo usado para bifurcar
        MEDIANA: fila en la posición usada para bifurcar
        IZQUIERDA: nodo donde el atributo es < al de la mediana
        DERECHA: nodo donde el atributo es > al de la mediana
    """

    matriz = muestras[0]
    tamanho_datos = len(matriz)

    if tamanho_datos < 1:
        return None

    if tamanho_datos == 1 or max_profundidad <= 1:
        return muestras

    # Si no cumple entonces retorna un None
    if tamanho_datos > 1:

        # Se copia para no alterar el del padre
        atribs_no_usados = np.copy(atribs_no_usados)

        # Se selecciona el atributo con mayor varianza
        # y que no ha sido usado por el nodo padre
        atributo = seleccionar(muestras[0], atribs_no_usados)

        # Se marca el atributo seleccionado como usado
        atribs_no_usados[atributo] = False

        # Se ordenan con base al atributo seleccionado
        muestras = ordenar(muestras, atributo)

        # Se dividen las muestras con base al atributo seleccionado
        nodo = bifurcar(muestras, atributo)

        nodo[IZQUIERDA] = \
            kdtree_aux(nodo[IZQUIERDA], atribs_no_usados, max_profundidad - 1)

        nodo[DERECHA] = \
            kdtree_aux(nodo[DERECHA], atribs_no_usados, max_profundidad - 1)

        return nodo

# -----------------------------------------------------------------------------


def knn(arbol, consulta, k_vecinos, recorrido=False):
    """
    Busca los `k_vecinos` más cercanos y sus distancias.

    :param arbol: Diccionario kd-Tree previamente creado
    :param consulta: np.array con la misma estructura que tienen
    las filas de la matriz que se utilizaron para crear el kd-tree
    :param k_vecinos: Cantidad máxima de vecinos más cercanos
    :param recorrido: True si se desea obtener la lista de nodos
    que se tuvieron que recorrer antes de llegar a la hoja (sin incluirla)

    :return: Tupla con la siguiente información:
        ♣ Matriz tamaño k x M, donde k_i representa un vecino
        ♣ Vector tamaño k con la etiqueta o clase de cada vecino
        ♣ Vector tamaño k con las distancias entre la consulta y los vecinos
        ♣ Matriz tamaño R x M, donde R es la cantidad de nodos recorridos
        para llegar hasta la hoja (ésta no se incluye)
    """

    vecinos = list()
    etiquetas = list()
    nodo_actual = arbol

    # Se recorren los nodos hasta llegar a una hoja
    while type(nodo_actual) == dict:

        vecinos.append(nodo_actual[MEDIANA])
        etiquetas.append(nodo_actual[ETIQUETA])

        # Indíce del atributo que se usó para bifurcar
        atributo = nodo_actual[ATRIBUTO]

        v_atrib_consulta = consulta[atributo]
        v_atrib_actual = nodo_actual[V_ATRIBUTO]

        # Si hay empate se testea la distancia entre la consulta
        # y los dos hijos. Se elige el camino con menor distancia
        if v_atrib_consulta == v_atrib_actual:

            test_izq = nodo_actual[IZQUIERDA][MEDIANA]
            dist_izq = distancia(consulta[atributo], test_izq)

            test_der = nodo_actual[DERECHA][MEDIANA]
            dist_der = distancia(consulta[atributo], test_der)

            # Si la distancia entre la consulta y el hijo de la izquierda
            # es menor que el de la derecha, se resta un 1 para que funcione
            # en el siguiente `if`
            v_atrib_consulta += -1 if dist_izq <= dist_der else 1

        if v_atrib_consulta < v_atrib_actual:
            nodo_actual = nodo_actual[IZQUIERDA]

        else:
            nodo_actual = nodo_actual[DERECHA]

    # Se ha llegado a una hoja, si no es una tupla es una hoja
    # vacía por lo que retorna None
    if type(nodo_actual) == tuple:

        vecinos = np.array(vecinos)
        etiquetas = np.array(etiquetas)
        nodos_recorridos = np.copy(vecinos)

        vecinos = np.append(nodo_actual[0], vecinos, axis=0)
        etiquetas = np.append(nodo_actual[1], etiquetas)

        vecinos, etiquetas, distancias = cercanos(
            vecinos, etiquetas, consulta, k_vecinos)

        if not recorrido:
            return vecinos, etiquetas, distancias
        else:
            return vecinos, etiquetas, distancias, nodos_recorridos

# -----------------------------------------------------------------------------


def predecir(arbol, consulta, k_vecinos=5):
    """
    Predice una etiqueta con base en una consulta.
    :param arbol: Diccionario kd-Tree previamente creado
    :param consulta: np.array con la misma estructura que tienen
    las filas matriz que se utilizaron para crear el kd-tree
    :param k_vecinos: Cantidad máxima de vecinos más cercanos
    :return: etiqueta
    """

    etiquetas = knn(arbol, consulta, k_vecinos)[1]

    # Diccionario (etiqueta, freacuencia)
    # Esto para tomar el voto por mayoría de los vecinos más cercanos
    votos = dict()
    for i in range(len(etiquetas)):

        if votos.__contains__(etiquetas[i]):
            votos[etiquetas[i]] += 1
        else:
            votos[etiquetas[i]] = 1

    # Retorna la etiqque de quién tenga más votos
    return max(votos, key=votos.get)

# -----------------------------------------------------------------------------


def seleccionar(matriz, atribs_no_usados):
    """
    Selecciona el atributo que se usará para bifurcar el nodo.
    Se usa como criterio la columna que tiene mayor varianza.

    :param matriz: Matriz N x M con valores númericos

    :param atribs_no_usados: np.array con elementos booleanos.
    Un True representa que el atributo en esa misma posición en los muestras
    no ha sido utilizado para bifurcar un nodo.

    :return: índice de la columna con mayor varianza en la `matriz`

    Ejemplos:
    >>> matriz = np.array([[1,  5, 8],
    ...                    [3, 10, 7],
    ...                    [5, 15, 6],
    ...                    [7, 20, 5]])
    >>> no_usados = [True, True, True]
    >>> seleccionar(matriz, no_usados)
    1
    >>> no_usados = [True, False, True]
    >>> seleccionar(matriz, no_usados)
    0
    >>> no_usados = [False, False, True]
    >>> seleccionar(matriz, no_usados)
    2
    """

    # Se aplica la máscara para solo obtener la varianza
    # de los atributos que no han sido utilizados
    varianzas = np.zeros(shape=len(matriz[0]))
    datos_no_usados = matriz[:, atribs_no_usados]
    varianzas[atribs_no_usados] = np.var(datos_no_usados, axis=0)

    return np.argmax(varianzas)

# -----------------------------------------------------------------------------


def ordenar(muestras, columna):
    """
    Ordena de menor a mayor los matriz según los valores
    de una sola columna de la matriz.

    :param muestras: Tupla donde:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    :param columna: Indíce de la columna
    :return: Los mismos matriz y etiquetas de entrada pero ordenados

    Ejemplo:
    >>> matriz = np.array([[  1, 10,   3],
    ...                    [ 34, 56,  43],
    ...                    [123,  9, 120]])
    >>> etiquetas = np.array(['ETIQ_1',
    ...                       'ETIQ_2',
    ...                       'ETIQ_3'])
    >>> muestras = (matriz, etiquetas)
    >>> muestras = ordenar(muestras, columna=1)
    >>> muestras[0] # Matriz
    array([[123,   9, 120],
           [  1,  10,   3],
           [ 34,  56,  43]])
    >>> muestras[1] # Etiquetas
    array(['ETIQ_3', 'ETIQ_1', 'ETIQ_2'], dtype='<U6')
    """

    matriz = muestras[0]
    etiquetas = muestras[1]

    columna = [columna]
    orden = np.lexsort(matriz.T[columna])

    return matriz[orden], etiquetas[orden]

# -----------------------------------------------------------------------------


def bifurcar(muestras, atributo):
    """
    Divide los matriz en dos usando un atributo como criterio.

    :param muestras: Tupla donde:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    :param atributo: Indice de una columna

    :return: Diccionario con la siguiente información:
        ATRIBUTO: índice de la columna usada para bifurcar
        V_ATRIBUTO: valor del atributo usado para bifurcar
        ETIQUETA: clase del atributo usado para bifurcar
        MEDIANA: fila en la posición usada para bifurcar
        IZQUIERDA: nodo donde el atributo es < al de la mediana
        DERECHA: nodo donde el atributo es > al de la mediana

    Ejemplos:
    >>> atributo = 2
    >>> matriz = np.array([[22, 38, 21],
    ...                    [20, 50, 24],
    ...                    [16, 44, 27], # <-- Se divide aquí
    ...                    [14, 62, 30],
    ...                    [18, 56, 33]])
    >>> etiquetas = np.array(['ETIQ_1',
    ...                       'ETIQ_2',
    ...                       'ETIQ_3',
    ...                       'ETIQ_4',
    ...                       'ETIQ_5'])
    >>> muestras = (matriz, etiquetas)
    >>> resultado = bifurcar(muestras, atributo)
    >>> resultado[MEDIANA]
    array([16, 44, 27])
    >>> nodo = resultado[IZQUIERDA]
    >>> nodo[0]
    array([[22, 38, 21],
           [20, 50, 24]])
    >>> nodo[1]
    array(['ETIQ_1', 'ETIQ_2'], dtype='<U6')
    >>> nodo = resultado[DERECHA]
    >>> nodo[0]
    array([[14, 62, 30],
           [18, 56, 33]])
    >>> nodo[1]
    array(['ETIQ_4', 'ETIQ_5'], dtype='<U6')
    """

    matriz = muestras[0]
    etiquetas = muestras[1]
    tamanho_datos = len(matriz)

    # Si la cantidad de filas de las matriz es par, la
    # la mediana es la posición inferior más cercana al centro.
    # La mediana se queda en el nodo en vez de ir hacia uno de
    # sus hijos.
    mitad = int(tamanho_datos // 2)
    mediana = matriz[mitad]

    # La mitad superior va hacia la izquierda
    mitad_matriz = matriz[0: mitad]
    mitad_etiquetas = etiquetas[0: mitad]
    izquierda = mitad_matriz, mitad_etiquetas

    # La mitad inferior va hacia la derecha
    mitad_matriz = matriz[mitad + 1: tamanho_datos]
    mitad_etiquetas = etiquetas[mitad + 1: tamanho_datos]
    derecha = mitad_matriz, mitad_etiquetas

    return {
        ATRIBUTO: atributo, V_ATRIBUTO: mediana[atributo],
        MEDIANA: matriz[mitad], ETIQUETA: etiquetas[mitad],
        IZQUIERDA: izquierda, DERECHA: derecha
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


def cercanos(matriz, etiquetas, consulta, k_vecinos=1):
    """
    Obtiene lax filas que se parece más al vector de entrada.

    :param matriz: Matriz N x M con valores númericos
    :param etiquetas: Vector tamaño N con las etiquetas de cada N_i
    :param consulta: Vector tamaño N
    :param k_vecinos: Cantidad máxima de vecinos a retornar

    :return: Tupla con los siguiente valores:
        ♣ Matriz k x M, donde k_i es un vecino
        ♣ Vector tamaño k con las etiquetas de cada k_i de la matriz anterior
        ♣ Lista de distancias en el mismo orden

    Ejemplos:
    >>> matriz = np.array([[  1,  10,   3],
    ...                    [ 34,  56,  43],
    ...                    [123,   9, 120]])
    >>> etiqs = np.array(['ETIQ_1',
    ...                   'ETIQ_2',
    ...                   'ETIQ_3'])
    >>> consulta = np.array([50, 40, 30])
    >>> vecs, etiqs, dists = cercanos(matriz, etiqs, consulta, 2)
    >>> vecs
    array([[34, 56, 43],
           [ 1, 10,  3]])
    >>> etiqs
    array(['ETIQ_2', 'ETIQ_1'], dtype='<U6')
    >>> dists
    array([26.0959767 , 63.48228099])
    """

    # Se calcula todas las distancias
    distancias = [distancia(consulta, vector) for vector in matriz]
    distancias = np.array(distancias)

    # Se ordenan las distancias de menor a mayor y se toman las
    # primeras igual a `k_vecinos`
    orden = np.argsort(distancias)
    matriz = matriz[orden][0:k_vecinos]
    etiquetas = etiquetas[orden][0:k_vecinos]
    distancias = distancias[orden][0:k_vecinos]

    return matriz, etiquetas, distancias

# -----------------------------------------------------------------------------


def normalizar(matriz, menor=0.0, mayor=1.0):
    """
    Normaliza los valores de la matriz para que todas la variables puedan
    competir utilizando la misma escala.

    NOTA: Es equivalente a:
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(matriz)

    :param matriz: Matriz N x M con valores númericos
    :param menor: Número más bajo que tendrá la matriz luego de normalizar
    :param mayor: Número más alto que tendrá la matriz luego de normalizar
    :return: Los mismos matriz de entrada pero normalizados

    Ejemplo:
    >>> matriz = np.array([[250, 0.15, 12],
    ...                    [120, 0.65, 20],
    ...                    [500, 0.35, 19]])
    >>> np.round(normalizar(matriz), decimals=2)
    array([[0.34, 0.  , 0.  ],
           [0.  , 1.  , 1.  ],
           [1.  , 0.4 , 0.88]])
    """

    maxs = np.max(matriz, axis=0)
    mins = np.min(matriz, axis=0)
    difs = maxs - mins
    prom = mayor - menor
    return mayor - ((prom * (maxs - matriz)) / difs)

# -----------------------------------------------------------------------------


def preprocesar(matriz, columnas, columnas_c, reordenar=True):

    """
    A partir de la matriz de muestrss, se convierten los atributos
    categóricos a númericos; seguidamente se normaliza la matriz.

    NOTA: Se asume que las columnas que no deben ser consideredas ya han
    sido borradas. En caso el caso particular de las elecciones, se asume
    que la columna 'VOTO_R1' ya ha sido borrada cuando se solicita realizar
    predicciones de 'VOTO_R2' sin utilizar 'VOTO_R1.

    :param matriz: Matriz donde cada fila es una muestra
    :param columnas:
    :param columnas_c: Lista de los nombres de las columnas categoricas a las
    que se les debe aplicar el algortimo ONE HOT ENCODING para convertirlas
    :param reordenar: True si se debe aplicar 'shuffle' a las filas

    :return: Tupla con la siguiente información:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    Ejemplos:

    >>> mat = np.array([[2, 'CAT_1', 5, 'ETIQ_1'],
    ...                 [3, 'CAT_2', 8, 'ETIQ_2'],
    ...                 [7, 'CAT_3', 9, 'ETIQ_3']])
    >>> cols = np.array(['COL_1', 'COL_2', 'COL_3', 'COL_4'])
    >>> cols_c = ['COL_2'] # Coumna categórica
    >>> mat, etiqs = preprocesar(mat, cols, cols_c,reordenar=False)
    >>> mat
    array([[0.  , 0.  , 1.  , 0.  , 0.  ],
           [0.2 , 0.75, 0.  , 1.  , 0.  ],
           [1.  , 1.  , 0.  , 0.  , 1.  ]])
    >>> etiqs
    array(['ETIQ_1', 'ETIQ_2', 'ETIQ_3'], dtype='<U11')
    """

    if type(matriz) == list:
        matriz = np.array(matriz)

    if reordenar:
        np.random.shuffle(matriz)

    # De la matriz de muestras, la última columna son las etiquetas
    etiquetas = matriz[:, matriz.shape[1] - 1]
    matriz = matriz[:, 0: matriz.shape[1] - 1]

    # No interesa el nombre columna de las etiquetas
    columnas = columnas[:len(columnas)-1]

    # Se aplica One Hot Encoding a las columnas categóricas
    # Por facilidad se convierte a un Dataframe y usar
    df = pd.DataFrame(matriz, columns=columnas)
    df = pd.get_dummies(df, columns=columnas_c)

    # Por alguna razón df.as_matriz retorna una matriz de tipo
    # str por lo que es necesario cambiar el tipo, luego se normaliza
    matriz = df.as_matrix().astype(float)
    matriz = normalizar(matriz)

    return matriz, etiquetas

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

    et = np.array(['ETIQ_1',
                   'ETIQ_2',
                   'ETIQ_3',
                   'ETIQ_4',
                   'ETIQ_5',
                   'ETIQ_6',
                   'ETIQ_1',
                   'ETIQ_8',
                   'ETIQ_9',
                   'ETIQ_1',
                   'ETIQ_11'])

    tree = kdtree((dt, et), max_profundidad=2)
    print("Kd-Tree")
    pprint(tree)

    consulta = np.array([14, 62, 30])
    print("Consulta con k = 2: " + str(consulta))

    # Vecinos, etiquetas, distancias, nodos_recorridos
    v, e, d, r = knn(tree, consulta, k_vecinos=5, recorrido=True)

    print("K vecinos más cercanos: \n" + str(v))
    print("Etiquetas de los K vecinos: \n" + str(e))
    print("Distancias de los k vecinos: \n" + str(d))
    print("Nodos recorridos: \n" + str(r))

    print("Realizando predicción")
    x = predecir(tree, consulta, k_vecinos=5)
    pprint(x)


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    ejemplo_kdtree()

# -----------------------------------------------------------------------------
