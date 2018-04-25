# -----------------------------------------------------------------------------

import pandas as pd
from string import ascii_uppercase as asc

# -----------------------------------------------------------------------------


def obtener_encabezado(n_columnas):
    """
    Función para generar los encabezados de un dataframe al igual que
    los de excel, es decir, ['A', 'B', 'C', 'D' ... 'AA', 'AB', 'AC' ... }
    :param n_columnas: cantidad de columnas que contiene el dataframe
    :return: lista de nombres de columnas
    """

    if n_columnas <= 26:
        return list(asc[0:n_columnas])
    else:
        combinaciones = [l1 + l2 for l1 in asc for l2 in asc]
        return list(asc) + combinaciones[0:n_columnas - 26]

# -----------------------------------------------------------------------------


def leer_csv(ruta_csv):
    """
    Lee un dataframe desde un csv.
    :param ruta_csv: nombre completo del archivo csv
    :param encabezado: Dice si dejar o no el encabezado leído en el csv
    :return: Dataframe obtenido del csv
    """

    try:
        return pd.read_csv(ruta_csv)

    except FileNotFoundError or TypeError:
        print("Archivo no encontrado: " + ruta_csv)
        exit(-1)

# -----------------------------------------------------------------------------


def guardar_como_csv(df, nombre_archivo):
    """
    Escribe una lista de lista en un csv como si fuese una tabla
    :param df: Dataframe a guarda
    :param nombre_archivo: ruta + nombre del archivo
    """

    try:
        df.to_csv(nombre_archivo)

    except Exception:
        print("Error al guardar el archivo: " + nombre_archivo)
        print("Puede que este siendo utilizado por otro proceso")


# -----------------------------------------------------------------------------


def obtener_dataframe(ruta_csv):
    """
    Lee un dataframe desde un csv.
    :param ruta_csv: nombre completo del archivo csv
    :param encabezado: Boolean que dice si dejar o no el encabezado leído
    :return: Dataframe donde la columna 0 son los nombres de los cantones, la
    columna corresponde a la provincia de ese canton.
    """

    dataframe = leer_csv(ruta_csv)

    # Se coloca cual columna se utiliza para busquedas
    return dataframe.set_index(dataframe.columns[0])

# -----------------------------------------------------------------------------
