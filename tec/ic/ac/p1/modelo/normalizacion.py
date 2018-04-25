
import pandas as pd


def normalize(data, attribute_to_norm, scale_function):

    f = feature_scaling
    if scale_function == 'ss':
        f = standard_scaling
    if scale_function == 'os':
        f = overmax_scaling

    for col in attribute_to_norm:
        data[col] = f(data[col])


def feature_scaling(df):

    # x_normalized = (x - xmin) / (xmax - xmin)
    return (df - df.min()) / (df.max - df.min)


def standard_scaling(df):

    # x_normalized = (x - xmean) / xstddev
    return (df - df.mean()) / df.std


def overmax_scaling(df):

    # x_normalized = x / xmax
    return df - df.max()
