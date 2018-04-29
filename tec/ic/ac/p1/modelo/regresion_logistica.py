
import tensorflow as tf

from pandas import DataFrame
from datetime import datetime

from modelo.manejo_archivos import guardar_como_csv
from modelo.normalizacion import reg_log_normalize


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
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA', 'VOTO_R1',
                'VOTO_R2']
numeric_cols = ['EDAD', 'ESCOLARIDAD PROMEDIO', 'POBLACION TOTAL',
                'SUPERFICIE', 'DENSIDAD POBLACION',
                'VIVIENDAS INDIVIDUALES OCUPADAS', 'PROMEDIO DE OCUPANTES',
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA']
_SHUFFLE_BUFFER = 1000


# ---------------------- Funciones 'públicas' ---------------------------------


def clasificador_regresion_logistica(learning_rate=0.1, regularization='None'):
    dir_name = 'regresion_log_' + str(datetime.now().time())
    dir_name = dir_name.replace(':', '_').replace('.', '_')
    return __build_estimator(dir_name, learning_rate, regularization)


def entrenar_regr_log(classifier, train_data_list):
    data = DataFrame(train_data_list)
    data = reg_log_normalize(data, numeric_cols, 'os')
    train_file = __save_data_file(data, 'train_data_')
    classifier.train(input_fn=lambda: __train_input_fn(train_file))
    return classifier


def validar_regr_log(classifier, test_data_list):
    data = DataFrame(test_data_list)
    data = reg_log_normalize(data, numeric_cols, 'os')
    validation_file = __save_data_file(data, 'validat_data_')
    results = classifier.evaluate(input_fn=lambda: __eval_input_fn(
        validation_file))
    return results


# ---------------------- Funciones de la regresion ----------------------------


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


def __build_estimator(model_dir, learning_rate=0.1, regularization='None'):

    columns = __build_model_columns()
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
    _L1, _L2 = (1, 0) if regularization == 'l1' else (0, 1)

    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                       l1_regularization_strength=_L1,
                                       l2_regularization_strength=_L2)
    if regularization == 'None':
        return tf.estimator.LinearClassifier(model_dir=model_dir,
                                             feature_columns=columns,
                                             config=run_config)

    return tf.estimator.LinearClassifier(model_dir=model_dir,
                                         feature_columns=columns,
                                         config=run_config,
                                         optimizer=optimizer)


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


def __train_input_fn(train_file, epochs_between_evals=2, batch_size=40):
    return __input_fn(train_file, epochs_between_evals, True, batch_size)


def __eval_input_fn(validat_file, epochs=1, batch=40):
    return __input_fn(validat_file, epochs, False, batch)


# ---------------------- Funciones auxiliares ---------------------------------


def __save_data_file(data_list, prefix):
    data_df = DataFrame(data_list)
    filename = prefix + str(datetime.now().time()) + '.csv'
    filename = filename.replace(':', '_').replace('.', '_')

    guardar_como_csv(data_df, filename)
    return filename


def main():
    pass


if __name__ == '__main__':
    main()


