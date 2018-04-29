
import tensorflow as tf

from pandas import DataFrame
from datetime import datetime

from modelo.manejo_archivos import guardar_como_csv
from modelo.normalizacion import reg_log_normalize
from tec.ic.ia.pc1.g03 import *

_CSV_COLUMN_DEFAULTS = [[''], [0], [''], [''], [''], [''], [''], [''],
                        [0], [''], [''], [''], [''], [''], [0], [0], [0],
                        [0], [0], [0], [0], [''], ['']]
_CSV_COLUMNS = ['CANTON', 'EDAD', 'ES_URBANO', 'SEXO',
                'ES_DEPENDIENTE', 'ESTADO_VIVIENDA', 'E.HACINAMIENTO',
                'ALFABETIZACION', 'ESCOLARIDAD_PROMEDIO',
                'ASISTENCIA_EDUCACION', 'FUERZA_DE_TRABAJO', 'SEGURO',
                'N.EXTRANJERO', 'C.DISCAPACIDAD', 'POBLACION_TOTAL',
                'SUPERFICIE', 'DENSIDAD_POBLACION',
                'VIVIENDAS_INDIVIDUALES_OCUPADAS', 'PROMEDIO_DE_OCUPANTES',
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA', 'VOTO_R1', 'VOTO_R2']
numeric_cols = ['EDAD', 'ESCOLARIDAD_PROMEDIO', 'POBLACION_TOTAL',
                'SUPERFICIE', 'DENSIDAD_POBLACION',
                'VIVIENDAS_INDIVIDUALES_OCUPADAS', 'PROMEDIO_DE_OCUPANTES',
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA']
_SHUFFLE_BUFFER = 1000

# TODO: IMPORTANTE! Agregar _CSV_COLUMNS como primer elemento del resultado
# del simulador


# ---------------------- Funciones 'públicas' ---------------------------------


def clasificador_regresion_logistica(learning_rate=0.1, regularization='None'):
    dir_name = 'regresion_log_' + str(datetime.now().time())
    dir_name = dir_name.replace(':', '_').replace('.', '_')
    return __build_estimator(dir_name, learning_rate, regularization)


def entrenar_regr_log(classifier, train_data_list):
    data = DataFrame(train_data_list, columns=_CSV_COLUMNS)
    data = reg_log_normalize(data, numeric_cols, 'os')
    train_file = __save_data_file(data, 'train_data_')
    classifier.train(input_fn=lambda: __train_input_fn(train_file))
    return classifier


def validar_regr_log(classifier, test_data_list):
    data = DataFrame(test_data_list, columns=_CSV_COLUMNS)
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
    escolaridad = tf_numeric('ESCOLARIDAD_PROMEDIO')
    poblacion = tf_numeric('POBLACION_TOTAL')
    superficie = tf_numeric('SUPERFICIE')
    densidad = tf_numeric('DENSIDAD_POBLACION')
    viviendas_ocupadas = tf_numeric('VIVIENDAS_INDIVIDUALES_OCUPADAS')
    ocupantes = tf_numeric('PROMEDIO_DE_OCUPANTES')
    p_jefat_fem = tf_numeric('P.JEFAT.FEMENINA')
    p_jefat_com = tf_numeric('P.JEFAT.COMPARTIDA')

    tf_categoric = tf.feature_column.categorical_column_with_vocabulary_list
    # Columnas de valores categorizables
    urbano = tf_categoric(_CSV_COLUMNS[2], ['NO URBANO', 'URBANO'])
    genero = tf_categoric(_CSV_COLUMNS[3], ['M', 'F'])
    est_vivienda = tf_categoric(_CSV_COLUMNS[5], ['V. MAL ESTADO',
                                                  'V. BUEN ESTADO'])
    hacinamiento = tf_categoric(_CSV_COLUMNS[6], ['V. NO HACINADA',
                                                  'V. HACINADA'])
    alfabetismo = tf_categoric(_CSV_COLUMNS[7], ['NO ALFABETIZADO',
                                                 'ALFABETIZADO'])
    educacion = tf_categoric(_CSV_COLUMNS[9], ['EDUCACION REGULAR INACTIVA',
                                               'EN EDUCACION REGULAR'])
    trabajo = tf_categoric(_CSV_COLUMNS[10], ['DESEMPLEADO', 'EMPLEADO'])
    seguro = tf_categoric(_CSV_COLUMNS[11], ['NO ASEGURADO', 'ASEGURADO'])
    extranjero = tf_categoric(_CSV_COLUMNS[12], ['NO EXTRANJERO',
                                                 'EXTRANJERO'])
    discapacidad = tf_categoric(_CSV_COLUMNS[13], ['NO DISCAPACITADO',
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

    if regularization == 'None':
        return tf.estimator.LinearClassifier(model_dir=model_dir,
                                             feature_columns=columns,
                                             config=run_config)

    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                       l1_regularization_strength=_L1,
                                       l2_regularization_strength=_L2)

    return tf.estimator.LinearClassifier(model_dir=model_dir,
                                         feature_columns=columns,
                                         config=run_config,
                                         optimizer=optimizer)


def __input_fn(data_file, num_epochs, shuffle, batch_size):

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        _features = dict(zip(_CSV_COLUMNS, columns))
        label = _features.pop('VOTO_R2')
        label = _features.pop('VOTO_R1')
        return _features, label
        #_features = tf.reshape(parsed_line[2:-2], shape=(20,))
        #label = tf.reshape(parsed_line[-2], shape=())
        #return _features, label

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
    data_df = DataFrame(data_list, columns=_CSV_COLUMNS)
    filename = prefix + str(datetime.now().time())
    filename = filename.replace(':', '_').replace('.', '_')
    filename += '.csv'

    guardar_como_csv(data_df, filename)
    return filename


def main():
    train_data = generar_muestra_pais(800)
    test_data = generar_muestra_pais(200)

    clasificador = clasificador_regresion_logistica()
    clasificador = entrenar_regr_log(clasificador, train_data)
    #resultados = validar_regr_log(clasificador, test_data)



if __name__ == '__main__':
    main()


