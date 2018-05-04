
# -----------------------------------------------------------------------------

import sys
sys.path.append('..')

import argparse
import numpy as np

from tec.ic.ia.pc1.g03 import generar_muestra_pais
from tec.ic.ia.pc1.g03 import generar_muestra_provincia

from p1.modelo.arbol import analisis_arbol_decision
from p1.modelo.nearest_neighbors import analisis_knn

# -----------------------------------------------------------------------------


def obtener_argumentos():
    """
    Se leen los comandos que se ingresaron en el cmd

    NOTAS:

        ♣ Al parsear los argumentos, recordar que si no se colocaron,
        por defecto se quedan en 'False' si se trata de un valor bool.
        En caso de que sean 'int', 'float' o 'str', el default es 'None'

        ♣ Todos los argumentos 'str' o 'int' se retornan como una lista
        de un solo elemento.

        ♣ El 'action=store_true' significa que se almacena un 'true'
        si el comando se encuentra. No es necesario añadir un
        valoral lado del argumento

   :return: Namespace con los argumentos
   """

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefijo', nargs=1, type=str)
    parser.add_argument('--provincia', nargs=1, type=str)
    parser.add_argument('--poblacion', nargs=1, type=int, default=1000)
    parser.add_argument('--porcentaje-pruebas', nargs=1, type=int, default=10)

    # Opcional. Cantidad de segmentos en k-fold-cross-validation
    parser.add_argument('--k-segmentos', nargs=1, type=int, default=10)

    # Modelos lineales
    parser.add_argument('--regresion-logistica', action='store_true')
    parser.add_argument('--l1', action='store_true')
    parser.add_argument('--l2', action='store_true')

    # Redes neuronales
    parser.add_argument('--red-neuronal', action='store_true')
    parser.add_argument('--numero-capas', nargs=1, type=int)
    parser.add_argument('--unidades-por-capa', nargs=1, type=int)
    parser.add_argument('--funcion-activacion', nargs='*')

    # Árboles de decisión
    parser.add_argument('--arbol', action='store_true')
    parser.add_argument('--umbral-poda', nargs=1, type=float, default=20)

    # KNN - K Nearest Neighbors
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--k', nargs=1, type=int, default=5)
    parser.add_argument('--max-profundidad', nargs=1, type=int, default=50)

    return parser.parse_args()

# -----------------------------------------------------------------------------


def obtener_datos(args):
    """
    Se usa el simulador para obtener la lista de votantes
    :param args: Argumentos de la línea de comandos
    :return: Lista de listas donde cada una es un votante
    """

    # Si no se específica por defecto es 1000
    poblacion = args.poblacion[0]

    # Provincias válidas, pueden ingresarsen en minúscula
    provincias = ['CARTAGO', 'SAN JOSE', 'LIMON',
                  'PUNTARENAS', 'GUANACASTE', 'ALAJUELA', 'HEREDIA']

    # Si no se específica, se generan muestras del país
    if args.provincia is None:
        return np.array(generar_muestra_pais(poblacion))

    provincia = args.provincia[0].upper()

    if provincias.__contains__(provincia):
        return np.array(generar_muestra_provincia(poblacion, provincia))

    print('La provincia especificada no existe')
    exit(-1)

# -----------------------------------------------------------------------------


def main():
    """
    Obtiene los argumentos, genera los datos con el simulador y
    ejecuta el modelo seleccionado según dichos argumentos.
    :return:
    """

    args = obtener_argumentos()
    datos = obtener_datos(args)
    np.random.shuffle(datos)

    if args.knn:
        analisis_knn(args, datos)

    elif args.arbol:
        analisis_arbol_decision(args, datos)

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------
