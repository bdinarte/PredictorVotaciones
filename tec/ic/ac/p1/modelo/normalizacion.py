
import pandas as pd


def reg_log_normalize(data, attribute_to_norm, scale_function='fs'):
    """
    feature scaling(default), 'ss' standard dev, 'os' overmax scaling
    :param data: dataframe
    :param attribute_to_norm: lista de columnas a normalizar
    :param scale_function: default = feature_scaling, 'ss' standard scaling,
    'os' overmax scaling
    :return: dataframe normalizado
    """

    f = __feature_scaling
    if scale_function == 'ss':
        f = __standard_scaling
    if scale_function == 'os':
        f = __overmax_scaling

    for col in attribute_to_norm:
        data[col] = f(data[col])

    return data


def __feature_scaling(df):

    # x_normalized = (x - xmin) / (xmax - xmin)
    return (df - df.min()) / (df.max() - df.min())


def __standard_scaling(df):

    # x_normalized = (x - xmean) / xstddev
    return (df - df.mean()) / df.std()


def __overmax_scaling(df):

    # x_normalized = x / xmax
    return df / df.max()
