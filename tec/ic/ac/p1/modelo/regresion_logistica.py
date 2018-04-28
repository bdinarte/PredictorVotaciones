
import tensorflow as tf
from modelo.manejo_archivos import obtener_dataframe, guardar_como_csv
from modelo.normalizacion import normalize

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [''], [''], [''], [''], [''],
                        [0], [''], [''], [''], [''], [''], [0], [0], [0],
                        [0], [0], [0], [0], [''], ['']]

_CSV_COLUMNS = ['VOTANTE', 'CANTON', 'EDAD', 'ES URBANO', 'SEXO',
                'ES DEPENDIENTE', 'ESTADO VIVIENDA', 'E.HACINAMIENTO',
                'ALFABETIZACION', 'ESCOLARIDAD PROMEDIO',
                'ASISTENCIA EDUCACION', 'FUERZA DE TRABAJO', 'SEGURO',
                'N.EXTRANJERO', 'C.DISCAPACIDAD', 'POBLACION TOTAL',
                'SUPERFICIE', 'DENSIDAD POBLACION',
                'VIVIENDAS INDIVIDUALES OCUPADAS', 'PROMEDIO DE OCUPANTES',
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA', 'VOTO_R1', 'VOTO_R2']

_SHUFFLE_BUFFER = 1000

data = obtener_dataframe('../archivos/pruebas.csv')

numeric_cols = ['EDAD', 'ESCOLARIDAD PROMEDIO', 'POBLACION TOTAL',
                'SUPERFICIE', 'DENSIDAD POBLACION',
                'VIVIENDAS INDIVIDUALES OCUPADAS', 'PROMEDIO DE OCUPANTES',
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA']

data = normalize(data, numeric_cols, 'os')

print(data)


def __build_model_columns():

    tf_numeric = tf.feature_column.numeric_column
    # Columnas de valores numericos
    edad = tf_numeric('EDAD')
    escolaridad = tf_numeric('ESCOLARIDAD PROMEDIO')
    poblacion = tf_numeric('POBLACION TOTAL')
    superficie = tf_numeric('SUPERFICIE')
    densidad = tf_numeric('DENSIDAD POBLACION')
    viviendas_ocupadas = tf_numeric('VIVIENDAS INDIVIDUALES OCUPADAS')
    ocupantes = tf_numeric('PROMEDIO DE OCUPANTES')
    p_jefat_fem = tf_numeric('P.JEFAT.FEMENINA')
    p_jefat_com = tf_numeric('P.JEFAT.COMPARTIDA')

    tf_categoric = tf.feature_column.categorical_column_with_vocabulary_list
    # Columnas de valores categorizables
    urbano = tf_categoric(_CSV_COLUMNS[3], ['NO URBANO', 'URBANO'])
    genero = tf_categoric(_CSV_COLUMNS[4], ['M', 'F'])
    est_vivienda = tf_categoric(_CSV_COLUMNS[6], ['V. MAL ESTADO',
                                                  'V. BUEN ESTADO'])
    hacinamiento = tf_categoric(_CSV_COLUMNS[7], ['V. NO HACINADA',
                                                  'V. HACINADA'])
    alfabetismo = tf_categoric(_CSV_COLUMNS[8], ['NO ALFABETIZADO',
                                                 'ALFABETIZADO'])
    educacion = tf_categoric(_CSV_COLUMNS[10], ['EDUCACION REGULAR INACTIVA',
                                                'EN EDUCACION REGULAR'])
    trabajo = tf_categoric(_CSV_COLUMNS[11], ['DESEMPLEADO', 'EMPLEADO'])
    seguro = tf_categoric(_CSV_COLUMNS[12], ['NO ASEGURADO', 'ASEGURADO'])
    extranjero = tf_categoric(_CSV_COLUMNS[13], ['NO EXTRANJERO',
                                                 'EXTRANJERO'])
    discapacidad = tf_categoric(_CSV_COLUMNS[14], ['NO DISCAPACITADO',
                                                   'DISCAPACITADO'])

    # Columnas de valores categorizados desconocidos
    canton = tf.feature_column.categorical_column_with_hash_bucket(
        'CANTON', hash_bucket_size=1000)

    # Agrupando columnas segun su tipo de dato
    numeric_tf_cols = [edad, escolaridad, poblacion, superficie, densidad,
                       viviendas_ocupadas, ocupantes, p_jefat_fem, p_jefat_com]
    categoric_tf_cols = [urbano, genero, est_vivienda, hacinamiento,
                         alfabetismo, educacion, trabajo, seguro, extranjero,
                         discapacidad, canton]

    # TODO: quizas definir crossed_column entre canton y sus indicadores

    return numeric_tf_cols + categoric_tf_cols


def entrenar(datos, regularizacion):
    pass


def __input_fn(data_file, num_epochs, shuffle, batch_size):

    def parse_csv(line):
        parsed_line = tf.decode_csv(line, _CSV_COLUMN_DEFAULTS)
        _features = tf.reshape(parsed_line[2:-2], shape=(20,))
        label = tf.reshape(parsed_line[-2], shape=())
        return _features, label

    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

