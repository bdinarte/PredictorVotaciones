# -----------------------------------------------------------------------------

"""
Clasificador K-Nearest-Neightbor utilizando kd-Tree

    ♣ Basado en la sección 18.8.2 del Libro AI-A Modern Approach, 737
    ♣ Se sigue el ejemplo de la siguiente página https://goo.gl/TGQaEe

    ♦ One Hot Encoding para atributos categóricos
    ♦ Normalización para que todos los atributos tengan la misma escala

    ♣ Se puede ver una demostración de la estructura del árbol (con
    datos triviales) ejecutando este archivo como el principal:

NOTAS:

    ♣ Resulta mejor cuando hay más ejemplos que dimensiones
    ♣ Para clasificar se usa el voto por mayoría de los k vecinos más cercanos
"""

import os
import sys
sys.path.append('../..')

from p1.util.util import *
from p1.util.timeit import *
from p1.modelo.columnas import *
from p1.modelo.manejo_archivos import *

from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

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


def kdtree(muestras, max_profundidad=50):
    """
    Crea el kd-Tree como un diccionario anidado.
    En cada nodo se parten los datos con base en el atributo que más varia.

    :param muestras: Tupla donde:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz
        ♣ NOTA: Ambos deben ser del tipo np.array

    :param max_profundidad: Representa la cantidad de niveles máx del árbol.
    Esto es porque, entre más nodos existen, más posibilidad hay de que
    el vecino más cercano no esté del lado del árbol donde se está buscando.

    :return: kdtree_aux(matriz, atributos_no_usados)
    """

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
        ♣ NOTA: Ambos deben ser del tipo np.array

    :param atribs_no_usados: np.array con elementos booleanos.
    Un True representa que el atributo en esa misma posición de la matriz
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

        # Se ordenan con base en el atributo seleccionado
        muestras = ordenar(muestras, atributo)

        # Se dividen las muestras con base en el atributo seleccionado
        nodo = bifurcar(muestras, atributo)

        nodo[IZQUIERDA] = kdtree_aux(nodo[IZQUIERDA],
                                     atribs_no_usados, max_profundidad - 1)

        nodo[DERECHA] = kdtree_aux(nodo[DERECHA],
                                   atribs_no_usados, max_profundidad - 1)

        return nodo

# -----------------------------------------------------------------------------


def knn(arbol, muestra, k_vecinos):
    """
    Busca los `k_vecinos` más cercanos y sus distancias

    :param arbol: Diccionario kd-Tree previamente creado

    :param muestra: np.array con la misma estructura que tienen
    las filas de la matriz que se utilizó para crear el kd-tree

    :param k_vecinos: Cantidad máxima de vecinos más cercanos
    que se tuvieron que recorrer antes de llegar a la hoja (sin incluirla)

    :return: Tupla con la siguiente información:
        ♣ Matriz tamaño k x M, donde k_i representa un vecino
        ♣ Vector tamaño k con la etiqueta o clase de cada vecino
        ♣ Vector tamaño k con las distancias entre la consulta y los vecinos
    """

    vecinos = []
    etiquetas = []
    nodo_actual = arbol

    # Se recorren los nodos hasta llegar a una hoja
    while type(nodo_actual) == dict:

        vecinos.append(nodo_actual[MEDIANA])
        etiquetas.append(nodo_actual[ETIQUETA])

        # Indíce del atributo que se usó para bifurcar
        atributo = nodo_actual[ATRIBUTO]

        v_atrib_consulta = muestra[atributo]
        v_atrib_actual = nodo_actual[V_ATRIBUTO]

        # Siempre que quedan dos vectores, uno se queda en el nodo
        # y el otro en la izquierda, por lo que, siempre se cumple que
        # si el hijo de la derecha es None el hijo izquierdo no lo es
        if nodo_actual[DERECHA] is None:
            nodo_actual = nodo_actual[IZQUIERDA]

        else:

            # Si hay empate se testea la distancia entre la muestra
            # y los dos hijos. Se elige el camino con menor distancia
            if v_atrib_consulta == v_atrib_actual:

                test_izq = nodo_actual[IZQUIERDA]
                test_der = nodo_actual[DERECHA]

                # Si es una hoja se extrae el elemento 0 de la tupla
                # que es el vector, si es un nodo selecciona la mediana
                test_izq = test_izq[MEDIANA if type(test_izq) == dict else 0]
                test_der = test_der[MEDIANA if type(test_der) == dict else 0]

                # Se testea la distancia entre la consulta y los hijos
                # con base en el atributo que había sido usada para bifurcar
                dist_izq = distancia(v_atrib_consulta, test_izq)
                dist_der = distancia(v_atrib_consulta, test_der)

                # Si la distancia entre la muestra y el hijo de la
                # izquierda es menor que el de la derecha, se resta un 1
                # para que el `nodo_actual` se actualice en el siguiente `if`
                v_atrib_consulta += -1 if dist_izq <= dist_der else 1

            if v_atrib_consulta < v_atrib_actual:
                nodo_actual = nodo_actual[IZQUIERDA]
            else:
                nodo_actual = nodo_actual[DERECHA]

    # Se ha llegado a una hoja, si no es una tupla es una hoja
    # vacía por lo que retorna None
    if type(nodo_actual) == tuple and vecinos:
        vecinos = np.array(vecinos)
        etiquetas = np.array(etiquetas)
        vecinos = np.append(nodo_actual[0], vecinos, axis=0)
        etiquetas = np.append(nodo_actual[1], etiquetas)

    return cercanos(vecinos, etiquetas, muestra, k_vecinos)

# -----------------------------------------------------------------------------


def predecir(arbol, muestra, k_vecinos=5):
    """
    Predice una etiqueta con base en una consulta.

    :param arbol: Diccionario kd-Tree previamente creado
    :param muestra: np.array con la misma estructura que tienen
    las filas de la matriz que se utilizó para crear el kd-tree
    :param k_vecinos: Cantidad máxima de vecinos más cercanos
    :return: etiqueta predecida para la muestra
    """

    # Etiquetas de los vecinos más cercanos
    etiquetas = knn(arbol, muestra, k_vecinos)[1]

    # Diccionario (etiqueta, freacuencia)
    # Esto para tomar el voto por mayoría de los vecinos más cercanos
    votos = dict()
    for i in range(len(etiquetas)):

        if votos.__contains__(etiquetas[i]):
            votos[etiquetas[i]] += 1
        else:
            votos[etiquetas[i]] = 1

    # Retorna la etiqueta de quién tenga más votos
    return max(votos, key=votos.get)


# -----------------------------------------------------------------------------

def predecir_conjunto(arbol, muestras, k_vecinos=5):
    """
    Predice las etiquetas para un conjunto de muestras
    :param arbol: Diccionario kd-Tree previamente creado
    :param muestras: matriz donde cada fila es una muesrta a predecir
    :param k_vecinos: Cantidad máxima de vecinos más cercanos
    :return: etiquetas predecidas para un conjunto de muestras
    """

    return [predecir(arbol, muestra, k_vecinos) for muestra in muestras]


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
    varianzas = np.zeros(len(matriz[0]))
    datos_no_usados = matriz[:, atribs_no_usados]
    varianzas[atribs_no_usados] = np.var(datos_no_usados, axis=0)

    # Se retorna la posición del atributo con mayor varianza
    return np.argmax(varianzas)

# -----------------------------------------------------------------------------


def ordenar(muestras, columna):
    """
    Ordena de menor a mayor los elementos de la matriz según los
    valores de una sola columna .

    :param muestras: Tupla donde:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    :param columna: Indíce de la columna
    :return: La misma matriz y etiquetas de entrada pero ordenados

    Ejemplo:
    >>> matriz = np.array([[  1, 10,   3],
    ...                    [ 34, 56,  43],
    ...                    [123,  9, 120]])
    >>> etiquetas = np.array(['ETIQ_1',
    ...                       'ETIQ_2',
    ...                       'ETIQ_3'])
    >>> matriz, etiquetas = ordenar((matriz, etiquetas), columna=1)
    >>> matriz
    array([[123,   9, 120],
           [  1,  10,   3],
           [ 34,  56,  43]])
    >>> etiquetas
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
    Divide la matriz en dos usando un atributo como criterio.

    :param muestras: Tupla donde:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    :param atributo: Indice de una columna que se usará para bifurcar

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
    >>> resultado = bifurcar((matriz, etiquetas), atributo)
    >>> resultado[MEDIANA]
    array([16, 44, 27])
    >>> resultado[ETIQUETA]
    'ETIQ_3'
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

    # Si la cantidad de filas de las matriz es par, la mediana es
    # la posición inferior más cercana al centro. La mediana se queda
    # el nodo en vez de ir hacia uno de sus hijos.
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
    :return: La misma matriz de entrada pero normalizados

    Ejemplo:
    >>> matriz = np.array([[250, 0.15, 12],
    ...                    [120, 0.65, 20],
    ...                    [500, 0.35, 19]])
    >>> np.round(normalizar(matriz), decimals=2)
    array([[0.34, 0.  , 0.  ],
           [0.  , 1.  , 1.  ],
           [1.  , 0.4 , 0.88]])
    """

    # maxs = np.max(matriz, axis=0)
    # mins = np.min(matriz, axis=0)
    # difs = maxs - mins
    # prom = mayor - menor
    # return mayor - ((prom * (maxs - matriz)) / difs)
    return MinMaxScaler().fit_transform(matriz)

# -----------------------------------------------------------------------------


def preprocesar(matriz, columnas, columnas_c):
    """
    A partir de la matriz de muestrss, se convierten los atributos
    categóricos a númericos; seguidamente se normaliza la matriz.

    NOTA: Se asume que las columnas que no deben ser consideredas ya han
    sido borradas. En caso el caso particular de las elecciones, se asume
    que la columna 'VOTO_R1' ya ha sido borrada cuando se solicita realizar
    predicciones de 'VOTO_R2' sin utilizar 'VOTO_R1.

    :param matriz: Matriz donde cada fila es una muestra
    :param columnas: Nombres de las columnas_csv de la matriz
    :param columnas_c: Lista de los nombres de las columnas_csv
    categoricas a las que se les debe aplicar el algortimo ONE HOT ENCODING

    :return: Tupla con la siguiente información:
        ♣ [0] = Matriz N x M con valores númericos
        ♣ [1] = Vector tamaño N con las etiquetas de cada N_i de la matriz

    Ejemplos:

    >>> mat = np.array([[2, 'CAT_1', 5, 'ETIQ_1'],
    ...                 [3, 'CAT_2', 8, 'ETIQ_2'],
    ...                 [7, 'CAT_3', 9, 'ETIQ_3']])
    >>> cols = np.array(['COL_1', 'COL_2', 'COL_3', 'COL_4'])
    >>> cols_c = ['COL_2'] # Coumna categórica
    >>> mat, etiqs = preprocesar(mat, cols, cols_c)
    >>> mat
    array([[0.  , 0.  , 1.  , 0.  , 0.  ],
           [0.2 , 0.75, 0.  , 1.  , 0.  ],
           [1.  , 1.  , 0.  , 0.  , 1.  ]])
    >>> etiqs
    array(['ETIQ_1', 'ETIQ_2', 'ETIQ_3'], dtype='<U11')
    """

    if type(matriz) == list:
        matriz = np.array(matriz)

    # De la matriz de muestras, la última columna son las etiquetas
    etiquetas = matriz[:, matriz.shape[1] - 1]
    matriz = matriz[:, 0: matriz.shape[1] - 1]

    # No interesa el nombre columna de las etiquetas
    columnas = columnas[:len(columnas)-1]

    # Se aplica One Hot Encoding a las columnas_csv categóricas
    # Por facilidad se convierte a un Dataframe
    df = pd.DataFrame(matriz, columns=columnas)
    df = pd.get_dummies(df, columns=columnas_c)

    # Por alguna razón df.as_matriz retorna una matriz de tipo
    # str por lo que es necesario cambiar el tipo, luego se normaliza
    matriz = df.as_matrix().astype(float)
    matriz = normalizar(matriz)

    return matriz, etiquetas

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

    # Columnas a las que se les debe aplicar One Hot Encoding
    columnas_c = [columnas_csv[0]]

    # Ronda #1: No se necesita la columna r2
    if ronda is 1:
        datos_r1 = np.delete(datos,  22, axis=1)
        columnas_r1 = np.delete(columnas_csv, 22)
        return preprocesar(datos_r1, columnas_r1, columnas_c)

    # Ronda #2 sin ronda #1: No se necesita la columan r1
    elif ronda is 2:
        datos_r2 = np.delete(datos, 21, axis=1)
        columnas_r2 = np.delete(columnas_csv, 21)
        return preprocesar(datos_r2, columnas_r2, columnas_c)

    # Ronda #2 usando ronda #1
    # La columna de la ronda #1 pasa a ser categorica
    else:
        columnas_c.append(columnas_csv[21])
        return preprocesar(datos, columnas_csv, columnas_c)

# -----------------------------------------------------------------------------


def cross_validation(datos, etiquetas, k_vecinos, k_segmentos):
    """
    Los datos se dividen en `k_segmentos`. Se itera k veces el entrenamiento.
    Se elige 1/K para validación en cada iteración. En cada iteración se
    obtiene la precision. Se obtiene el promedio de precisiones junto con
    cada una de las predicciones relizadas.

    :param datos: Matriz de entrenamien N x M
    :param etiquetas: Vector tamaño N
    :param k_vecinos: Cantidad máxima de vecinos más cercanos
    :param k_segmentos: Cantidad de segmentos en los que se deben dividir
    los datos. Representa también la cantidad de iteraciones.
    :return: (Precision, Predicciones)
    """

    precisiones = list()
    predicciones = list()

    indices = list(range(len(datos)))
    segmentos = segmentar(indices, k_segmentos)

    # Se hacen k iteraciones. En cada iteración se usa un segmento
    # diferente para validación, el resto se usa para entrenamiento
    for k in range(len(segmentos)):

        ind_validacion, ind_entrenamiento = agrupar(k, segmentos)

        # Validadción (para el final)
        datos_validacion = datos[ind_validacion]
        etiqs_validacion = etiquetas[ind_validacion]

        # Entrenamiento
        datos_entrenamiento = datos[ind_entrenamiento]
        etiqs_entrenamiento = etiquetas[ind_entrenamiento]

        # Se entrena el sistema con loa datos y etiquetas de entrenamiento
        arbol = kdtree((datos_entrenamiento, etiqs_entrenamiento))

        # Etiquetas obtenidas del conjunto de validación
        etiqs_predics = predecir_conjunto(arbol, datos_validacion, k_vecinos)
        predicciones += etiqs_predics

        # Cantidad de etiquetas predecidas correctamente vs predicciones
        precisiones.append(accuracy_score(etiqs_validacion, etiqs_predics))

    return np.mean(precisiones), predicciones

# -----------------------------------------------------------------------------


def analisis_knn(args, datos):

    print("K Nearest Neighbors")
    print("Usando k -> " + str(args.k[0]))

    # Para agregar las 4 columnas solicitadas
    salida = np.concatenate((datos, np.zeros((datos.shape[0], 4))), axis=1)

    k_vecinos = args.k[0]
    porcentaje_pruebas = args.porcentaje_pruebas[0]
    prefijo_archivos = "knn" if args.prefijo is None else args.prefijo[0]

    indices = list(range(datos.shape[0]))
    ind_pruebas, ind_entrenamiento = separar(indices, porcentaje_pruebas)

    # Se llena la columna 'es_entrenamiento'
    salida[ind_pruebas, 23] = 0
    salida[ind_entrenamiento, 23] = 1

    for n_ronda in range(1, 4):

        predicciones = list()
        datos_ronda, etiqs_ronda = preprocesar_ronda(datos, ronda=n_ronda)

        datos_pruebas = datos_ronda[ind_pruebas]
        datos_entrenamiento = datos_ronda[ind_entrenamiento]
        etiqs_entrenamiento = etiqs_ronda[ind_entrenamiento]

        prec_ronda, predics_ronda = \
            cross_validation(datos_entrenamiento,
                             etiqs_entrenamiento,
                             k_vecinos, args.k_segmentos[0])

        predicciones += predics_ronda

        # Predicciones sobre los datos de pruebas
        clasificador = kdtree((datos_entrenamiento, etiqs_entrenamiento))
        predics = predecir_conjunto(clasificador, datos_pruebas, k_vecinos)
        predicciones = predics + predicciones

        salida[:, 23 + n_ronda] = predicciones

        print("Precisión ronda {} -> {}".format(n_ronda, prec_ronda))

    salida = pd.DataFrame(salida, columns=columnas_salida)

    # Se guarda el archivo con las 4 columnas de la especificación
    nombre_salida = os.path.join("archivos", prefijo_archivos + "_1.csv")
    salida.to_csv(nombre_salida)


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

    tree = kdtree((dt, et), max_profundidad=5)
    pprint(tree)

    consulta = np.array([18, 55, 30])
    print("Consulta con k -> 2 ")
    print(consulta)

    # Vecinos, etiquetas, distancias
    v, e, d = knn(tree, consulta, k_vecinos=5)
    print("K vecinos más cercanos: \n" + str(v))
    print("Etiquetas de los K vecinos: \n" + str(e))
    print("Distancias de los k vecinos: \n" + str(d))

    print("Realizando predicción de prueba")
    print(predecir(tree, consulta, k_vecinos=5))


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    ejemplo_kdtree()

# -----------------------------------------------------------------------------
