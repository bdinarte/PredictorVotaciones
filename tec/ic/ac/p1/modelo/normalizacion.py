
import pandas as pd


def reg_log_normalize(data, attributes_to_norm, scale_function='fs'):
    """
    feature scaling(default), 'ss' standard dev, 'os' overmax scaling
    :param data: dataframe
    :param attributes_to_norm: lista de columnas a normalizar
    :param scale_function: default = feature_scaling, 'ss' standard scaling,
    'os' overmax scaling
    :return: dataframe normalizado
    """

    f = __feature_scaling
    if scale_function == 'ss':
        f = __standard_scaling
    if scale_function == 'os':
        f = __overmax_scaling

    for col in attributes_to_norm:
        data[col] = f(data[col])

    return data


def reg_log_str_to_bool(data):
    """
    Convierte las columnas categoricas a numéricas
    :param data: dataframe con los datos
    :return: dataframe modificado
    """
    pass

    return data


def col_map_str_to_bool(col, str_false, str_true):

    return col.map({str_false: 0, str_true: 1})


def __feature_scaling(df):

    # x_normalized = (x - xmin) / (xmax - xmin)
    return (df - df.min()) / (df.max() - df.min())


def __standard_scaling(df):

    # x_normalized = (x - xmean) / xstddev
    return (df - df.mean()) / df.std()


def __overmax_scaling(df):

    # x_normalized = x / xmax
    return df / df.max()
