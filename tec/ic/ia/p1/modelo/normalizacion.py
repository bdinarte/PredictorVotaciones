
def normalize(data, attributes_to_norm, scale_function='fs'):
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


def __feature_scaling(df):

    # x_normalized = (x - xmin) / (xmax - xmin)
    return (df - df.min()) / (df.max() - df.min())


def __standard_scaling(df):

    # x_normalized = (x - xmean) / xstddev
    return (df - df.mean()) / df.std()


def __overmax_scaling(df):

    # x_normalized = x / xmax
    return df / df.max()


def partidos_r1_to_id():
    return {'ACCESIBILIDAD SIN EXCLUSION': 0,
            'ACCION CIUDADANA': 1,
            'ALIANZA DEMOCRATA CRISTIANA': 2,
            'DE LOS TRABAJADORES': 3,
            'FRENTE AMPLIO': 4,
            'INTEGRACION NACIONAL': 5,
            'LIBERACION NACIONAL': 6,
            'MOVIMIENTO LIBERTARIO': 7,
            'NUEVA GENERACION': 8,
            'RENOVACION COSTARRICENSE': 9,
            'REPUBLICANO SOCIAL CRISTIANO': 10,
            'RESTAURACION NACIONAL': 11,
            'UNIDAD SOCIAL CRISTIANA': 12,
            'NULO': 13, 'BLANCO': 14}


def partidos_r2_to_id():
    return {'RESTAURACION NACIONAL': 0,
            'ACCION CIUDADANA': 1,
            'NULO': 2, 'BLANCO': 3}


def categoric_to_numeric(data):
    data['VOTO_R1'] = data['VOTO_R2'].map(partidos_r1_to_id())
    data['VOTO_R2'] = data['VOTO_R2'].map(partidos_r2_to_id())
    return data


def id_to_partidos_r1():
    return {0: 'ACCESIBILIDAD SIN EXCLUSION',
            1: 'ACCION CIUDADANA',
            2: 'ALIANZA DEMOCRATA CRISTIANA',
            3: 'DE LOS TRABAJADORES',
            4: 'FRENTE AMPLIO',
            5: 'INTEGRACION NACIONAL',
            6: 'LIBERACION NACIONAL',
            7: 'MOVIMIENTO LIBERTARIO',
            8: 'NUEVA GENERACION',
            9: 'RENOVACION COSTARRICENSE',
            10: 'REPUBLICANO SOCIAL CRISTIANO',
            11: 'RESTAURACION NACIONAL',
            12: 'UNIDAD SOCIAL CRISTIANA',
            13: 'NULO', 14: 'BLANCO'}


def id_to_partidos_r2():
    return {0: 'RESTAURACION NACIONAL',
            1: 'ACCION CIUDADANA',
            2: 'NULO', 3: 'BLANCO'}
