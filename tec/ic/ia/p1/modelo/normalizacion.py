
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


def __feature_scaling(df):

    # x_normalized = (x - xmin) / (xmax - xmin)
    return (df - df.min()) / (df.max() - df.min())


def __standard_scaling(df):

    # x_normalized = (x - xmean) / xstddev
    return (df - df.mean()) / df.std()


def __overmax_scaling(df):

    # x_normalized = x / xmax
    return df / df.max()


def categoric_to_numeric(data):
    data['ES_URBANO'] = data['ES_URBANO'].map({'NO URBANO': 0, 'URBANO': 1})
    data['SEXO'] = data['SEXO'].map({'M': 0, 'F': 1})
    data['ES_DEPENDIENTE'] = data['ES_DEPENDIENTE'].map({
        'NO DEPENDIENTE': 0,'DEPENDIENTE': 1})
    data['ESTADO_VIVIENDA'] = data['ESTADO_VIVIENDA'].map({
        'V. MAL ESTADO': 0,'V. BUEN ESTADO': 1})
    data['E.HACINAMIENTO'] = data['E.HACINAMIENTO'].map({
        'V. NO HACINADA': 0,'V. HACINADA': 1})
    data['ALFABETIZACION'] = data['ALFABETIZACION'].map({
        'NO ALFABETIZADO': 0, 'ALFABETIZADO': 1})
    data['ASISTENCIA_EDUCACION'] = data['ASISTENCIA_EDUCACION'].map({
        'EDUCACION REGULAR INACTIVA': 0, 'EN EDUCACION REGULAR': 1})
    data['FUERZA_DE_TRABAJO'] = data['FUERZA_DE_TRABAJO'].map({
        'DESEMPLEADO': 0, 'EMPLEADO': 1})
    data['SEGURO'] = data['SEGURO'].map({'NO ASEGURADO': 0, 'ASEGURADO': 1})
    data['N.EXTRANJERO'] = data['N.EXTRANJERO'].map({
        'NO EXTRANJERO': 0, 'EXTRANJERO': 1})
    data['C.DISCAPACIDAD'] = data['C.DISCAPACIDAD'].map({
        'NO DISCAPACITADO': 0, 'DISCAPACITADO': 1})
    '''data['VOTO_R1'] = data['VOTO_R2'].map({
        'ACCESIBILIDAD SIN EXCLUSION': 0,
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
        'NULO': 13, 'BLANCO': 14})
    data['VOTO_R2'] = data['VOTO_R2'].map({
        'RESTAURACION NACIONAL': 0,
        'ACCION CIUDADANA': 1,
        'NULO': 2, 'BLANCO': 3})'''
    return data
