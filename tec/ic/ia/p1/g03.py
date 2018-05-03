
# -----------------------------------------------------------------------------

import sys
sys.path.append('..')

import argparse
from tec.ic.ia.pc1.g03 import *
from p1.modelo.nearest_neighbors import *

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
    parser.add_argument('--porcentaje-pruebas', nargs=1, type=int, default=10)

    # Opcional. Cantidad de segmentos en k-fold-cross-validation
    parser.add_argument('--k-segmentos', nargs=1, type=int, default=10)

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
    parser.add_argument('--k', nargs=1, type=int, default=5)

    return parser.parse_args()

# -----------------------------------------------------------------------------


def obtener_datos(args):
    """
    Se usa el simulador para obtener la lista de listas
    :param args: Argumentos de la línea de comandos
    :return: Lista de listas donde cada una es un votante
    """

    provincias = ['CARTAGO', 'SAN JOSE', 'LIMON',
                  'PUNTARENAS', 'GUANACASTE', 'ALAJUELA', 'HEREDIA']

    if args.provincia is None:
        return np.array(generar_muestra_pais(args.poblacion[0]))

    provincia = args.provincia[0].upper()

    if provincias.__contains__(provincia):
        return np.array(generar_muestra_provincia(args.poblacion[0], provincia))

    print('La provincia especificada no existe')
    exit(-1)

# -----------------------------------------------------------------------------


def main():
    args = obtener_argumentos()
    datos = obtener_datos(args)
    np.random.shuffle(datos)

    if args.knn:
        analisis_knn(args, datos)

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------
