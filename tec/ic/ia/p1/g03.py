
# -----------------------------------------------------------------------------

import argparse

# -----------------------------------------------------------------------------


def read_args():

    """
    Se leen los comandos que se ingresaron en el cmd
    Ejemplo:

    > python g03.py --red-neuronal
                    --prefijo 'prueba_redes_1'
                    --porcentaje-pruebas 15
                    --numero-capas 2
                    --unidades-por-capa 3
                    --funcion_activacion ?
    Al parsear los argumentos, recordar que si no se colocaron,
    por defecto se quedan en 'false' o en 'None' en caso de
    que sean 'int', 'float' o 'str'.

    NOTA: El 'action=store_true' significa que se almacena un
    'true' si el comando se encuentra. No es necesario añadir un
    valoral lado del argumento

    :return: Namespace con los valores
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefijo', nargs=1, type=str)
    parser.add_argument('--porcentaje-pruebas', nargs=1, type=int)

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
    # + Ganancia de información mínima para realizar la partición
    # en la poda del árbol
    parser.add_argument('--arbol', action='store_true')
    parser.add_argument('--umbral-poda', nargs=1, type=float)

    # KNN - K Nearest Neighbors
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--k', nargs=1, type=int)

    return parser.parse_args()


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    """
    Se leen cada uno de los argumentos de la línea de comandos y se 
    ejecuta la función correspondiente 
    """

    # Solo para dar un margen de una línea
    print()

    args = read_args()

    # Modelos lineales: Regresión logistica
    # Argumentos específicos necesarios:
    #   args.l1
    #   args.l2
    if args.regresion_logistica:

        # TODO: Invocar aquí la función de regresión logística
        # error, pruebas = regresion_logitica(
        # args.prefijo, args.porcentaje_pruebas, args.l1, args.l2)
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

        # TODO: Invocar aquí la función de red neuronal
        # error, pruebas = red_neuronal(
        # args.prefijo, args.porcentaje_pruebas,
        # args.numero_capas, args.unidades_por_capa, args.funcion_activacion)
        print('Red neuronal')
        print('Argumentos: ')
        print('\t numero_capas ' + str(args.numero_capas))
        print('\t unidades_por_capa ' + str(args.unidades_por_capa))
        print('\t funcion_activacion ' + str(args.funcion_activacion))

    # Árbol de decisión
    # Argumentos específicos necesarios:
    #   args.umbral_poda
    elif args.arbol:

        # TODO: Invocar aquí la función de árbol decisión
        # error, pruebas = arbol(
        # args.prefijo, args.porcentaje_pruebas, args.umbral_poda)
        print('Árbol de decisión')
        print("Argumentos: ")
        print('\t umbral_poda' + str(args.umbral_poda))

    # K Nearest Neighbors
    # Argumentos específicos necesarios:
    #   args.k
    elif args.knn:

        # TODO: Invocar aquí la función KNN
        # error, pruebas = knn(
        # args.prefijo, args.porcentaje_pruebas, args.k)
        print('K Nearest Neighbors')
        print("Argumentos: ")
        print('\t k' + str(args.k))

# -----------------------------------------------------------------------------
