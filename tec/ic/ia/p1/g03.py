
# -----------------------------------------------------------------------------

import sys
sys.path.append('..')

import argparse
from tec.ic.ia.pc1.g03 import *
from p1.modelo.columnas import *
from p1.modelo.cross_validation import cross_validation

# -----------------------------------------------------------------------------


def obtener_argumentos():
    """
    Se leen los comandos que se ingresaron en el cmd

    NOTAS:
        ♣ Al parsear los argumentos, recordar que si no se colocaron,
          por defecto se quedan en 'false' o en 'None' en caso de
          que sean 'int', 'float' o 'str'.

        ♣ Todos los argumentos 'str' o 'int' se retornan como una lista
          de un solo elemento.

        ♣ El 'action=store_true' significa que se almacena un
          'true' si el comando se encuentra. No es necesario añadir un
           valoral lado del argumento

   :return: Namespace con los valores
   """

    parser = argparse.ArgumentParser()

    # Globales
    parser.add_argument('--prefijo', nargs=1, type=str)
    parser.add_argument('--poblacion', nargs=1, type=int)
    parser.add_argument('--provincia', nargs=1, type=str)
    parser.add_argument('--porcentaje-pruebas', nargs=1, type=int)

    # Tipos de predicción
    parser.add_argument('--prediccion-r1', action='store_true')
    parser.add_argument('--prediccion-r2', action='store_true')
    parser.add_argument('--prediccion-r2-con-r1', action='store_true')

    # Opcional. Cantidad de segmentos en k-fold-cross-validation
    parser.add_argument('--k-segmentos', nargs=1, type=int)

    # Modelos lineales
    parser.add_argument('--regresion-logistica', action='store_true')
    parser.add_argument('--l1', action='store_true')
    parser.add_argument('--l2', action='store_true')

    # Redes neuronales
    # En sus argumentos se debe colocar un valor después de las banderas
    # TODO: Se deben colocar cuáles son los valores válidos
    parser.add_argument('--red-neuronal', action='store_true')
    parser.add_argument('--numero-capas', nargs=1, type=int)
    parser.add_argument('--unidades-por-capa', nargs=1, type=int)
    parser.add_argument('--funcion-activacion', nargs='*')

    # Árboles de decisión
    # Ganancia de información mínima para realizar la partición
    # en la poda del árbol
    parser.add_argument('--arbol', action='store_true')
    parser.add_argument('--umbral-poda', nargs=1, type=float)

    # KNN - K Nearest Neighbors
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--k', nargs=1, type=int)

    return parser.parse_args()

# -----------------------------------------------------------------------------


def obtener_datos(poblacion, provincia=None):
    """
    Se usa el simulador para obtener la lista de listas
    :param poblacion: Tamaño de la muestra a generar
    :param provincia: Si no se especifica se realiza a nivel país
    :return: Lista de listas donde cada una es un votante
    """

    provincias = ['CARTAGO', 'SAN JOSE', 'LIMON',
                  'PUNTARENAS', 'GUANACASTE', 'ALAJUELA', 'HEREDIA']

    if provincia is None:
        return np.array(generar_muestra_pais(poblacion))

    provincia = provincia.upper()

    if provincias.__contains__(provincia):
        return np.array(generar_muestra_provincia(poblacion, provincia))

    print('La provincia especificada no existe')
    exit(-1)

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    args = obtener_argumentos()
    datos = obtener_datos(args.poblacion[0], args.provincia[0])

    # Índice de la columna que no se debe usar en la predicción
    no_usable = None

    # Columnas que no son númericas
    columnas_c = [columnas[0]]

    # Si se quiere predecir r1 no es necesario la columna r2 (la última)
    if args.prediccion_r1:
        no_usable = 22

    # En caso de predecir r2 sin r1, se debe borrar r1 (la penúltima)
    elif args.prediccion_r2:
        no_usable = 21

    # Si se predice r2 usando r1, voto_r1 pasa a ser una columna categorica
    else:
        columnas_c.append(columnas[21])

    if no_usable is not None:
        datos = np.delete(datos, no_usable, axis=1)
        columnas = np.delete(columnas, no_usable)

    # Si no se específica, por defecto se ejecuta la prediccion_r2_con_r1
    metrica = cross_validation(args, datos, columnas, columnas_c)
    print("\nPromedio: " + str(metrica))

# -----------------------------------------------------------------------------
