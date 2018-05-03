
import os
import sys
sys.path.append('../..')

from pandas import DataFrame
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt

from tec.ic.ia.pc1.g03 import generar_muestra_pais
from p1.modelo.normalizacion import normalize, categoric_to_numeric
from p1.modelo.normalizacion import partidos_r1_to_id, partidos_r2_to_id
from p1.modelo.normalizacion import id_to_partidos_r1, id_to_partidos_r2
from p1.modelo.manejo_archivos import guardar_como_csv


# --------------------------- Constantes --------------------------------------

data_columns = ['CANTON', 'EDAD', 'ES_URBANO', 'SEXO', 'ES_DEPENDIENTE',
                'ESTADO_VIVIENDA', 'E.HACINAMIENTO', 'ALFABETIZACION',
                'ESCOLARIDAD_PROMEDIO', 'ASISTENCIA_EDUCACION',
                'FUERZA_DE_TRABAJO', 'SEGURO', 'N.EXTRANJERO',
                'C.DISCAPACIDAD', 'POBLACION_TOTAL', 'SUPERFICIE',
                'DENSIDAD_POBLACION', 'VIVIENDAS_INDIVIDUALES_OCUPADAS',
                'PROMEDIO_DE_OCUPANTES', 'P.JEFAT.FEMENINA',
                'P.JEFAT.COMPARTIDA', 'VOTO_R1', 'VOTO_R2']
cols_to_norm = ['EDAD', 'ESCOLARIDAD_PROMEDIO', 'POBLACION_TOTAL',
                'SUPERFICIE', 'DENSIDAD_POBLACION',
                'VIVIENDAS_INDIVIDUALES_OCUPADAS', 'PROMEDIO_DE_OCUPANTES',
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA']
#
# entre mas pequeño es el batch, mas lento el entrenamiento pero converge con
# menos pasadas
batch_size = 1500
__predicting = ''
__learning_rate = 0.03

# ------------------------ Funciones públicas ---------------------------------


def regresion_logistica(prefix, regularization='None', predicting='r1'):
    """
    Define un modelo de regresion, regularization -> 'l1', 'l2', 'None'
    :param prefix: prefijo para archivos generados
    :param regularization: 'l1', 'l2', 'None' -> tipo de regularización
    :param predicting: ronda a predecir 'r1', 'r2', 'r2_with_r1'
    :return: modelo de regresion logistica
    """
    global __predicting
    __predicting = predicting
    columns = __build_model_columns()
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
    _L1, _L2 = (1, 0) if regularization == 'l1' else (0, 1)

    class_amount = 15 if predicting == 'r1' else 4

    if regularization == 'None':
        return tf.estimator.LinearClassifier(model_dir=prefix,
                                             feature_columns=columns,
                                             config=run_config,
                                             n_classes=class_amount)

    optimizer = tf.train.FtrlOptimizer(learning_rate=__learning_rate,
                                       l1_regularization_strength=_L1,
                                       l2_regularization_strength=_L2)

    return tf.estimator.LinearClassifier(model_dir=prefix,
                                         feature_columns=columns,
                                         config=run_config,
                                         optimizer=optimizer,
                                         n_classes=class_amount)


def rl_entrenar(model, train_data, prefix):
    global __predicting
    if __predicting == 'r2_with_r1':
        for _ in range(5):
            model.train(input_fn=lambda: __input_fn(train_data,
                                                    prefix))
    return model


def rl_validar(model, validation_data, prefix):
    results = model.evaluate(input_fn=lambda: __input_fn(validation_data,
                                                         prefix))
    return results['accuracy'], results['average_loss']


def rl_predict(model, df_data, prefix):

    if __predicting == 'r2_with_r1':
        df_data.drop(labels=['VOTO_R2'], axis=1)
    else:
        df_data.drop(labels=['VOTO_R1', 'VOTO_R2'], axis=1)

    if __predicting == 'r1':
        classes = id_to_partidos_r1()
    else:
        classes = id_to_partidos_r2()

    predictions = model.predict(input_fn=lambda: __input_fn(df_data, prefix))

    template = 'Prediction is "{}" ({:.1f}%)'

    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(classes[class_id], 100 * probability))


def rl_normalize(data_list, normalization):
    #
    # Setup de los datos generados por el simulador
    data = DataFrame(data_list, columns=data_columns)
    #
    # Normalización de las columnas densas
    data = normalize(data, cols_to_norm, normalization)
    #
    # Conversión de partidos a números
    data = categoric_to_numeric(data)
    # data = data.drop('CANTON', axis=1)

    return data


# ---------------------- Funciones del modelo ---------------------------------


def __input_fn(data, prefix):

    def parse_csv(value):
        global __predicting
        if __predicting == 'r1':
            example_defaults = [[''], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                                [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                                [0.], [0.], [0.], [0.], [0.], [0], [0]]
        else:
            example_defaults = [[''], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                                [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                                [0.], [0.], [0.], [0.], [0.], [0.], [0]]

        columns = tf.decode_csv(value, example_defaults)
        _features = dict(zip(data_columns, columns))
        if __predicting == 'r1':
            _features.pop('VOTO_R2', None)
            _label = _features.pop('VOTO_R1')
        elif __predicting == 'r2':
            _features.pop('VOTO_R1', None)
            _label = _features.pop('VOTO_R2')
        else:
            _label = _features.pop('VOTO_R2')
        return _features, _label

    dataset = __rl_build_dataset(data, prefix, parse_csv)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def __rl_build_dataset(data, prefix, parse_function):
    #
    # Generar archivos de datos
    filename = __save_data_file(data, prefix)
    #
    # Parsear el archivo con data
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size)

    return dataset


def __build_model_columns():

    tf_numeric = tf.feature_column.numeric_column
    edad = tf_numeric('EDAD')
    escolaridad = tf_numeric('ESCOLARIDAD_PROMEDIO')
    poblacion = tf_numeric('POBLACION_TOTAL')
    superficie = tf_numeric('SUPERFICIE')
    densidad = tf_numeric('DENSIDAD_POBLACION')
    viviendas_ocupadas = tf_numeric('VIVIENDAS_INDIVIDUALES_OCUPADAS')
    ocupantes = tf_numeric('PROMEDIO_DE_OCUPANTES')
    p_jefat_fem = tf_numeric('P.JEFAT.FEMENINA')
    p_jefat_com = tf_numeric('P.JEFAT.COMPARTIDA')
    voto_r1 = tf_numeric('VOTO_R1')

    # columnas categoricas convertidas a binario
    urbano = tf_numeric(data_columns[2])
    genero = tf_numeric(data_columns[3])
    est_vivienda = tf_numeric(data_columns[5])
    hacinamiento = tf_numeric(data_columns[6])
    alfabetismo = tf_numeric(data_columns[7])
    educacion = tf_numeric(data_columns[9])
    trabajo = tf_numeric(data_columns[10])
    seguro = tf_numeric(data_columns[11])
    extranjero = tf_numeric(data_columns[12])
    discapacidad = tf_numeric(data_columns[13])

    # Columnas de valores categorizados desconocidos
    canton = tf.feature_column.categorical_column_with_hash_bucket(
        'CANTON', hash_bucket_size=1000)

    cols = [canton, edad, escolaridad, poblacion, superficie, densidad,
            viviendas_ocupadas, ocupantes, p_jefat_fem,
            p_jefat_com, urbano, genero, est_vivienda, hacinamiento,
            alfabetismo, educacion, trabajo, seguro, extranjero,
            discapacidad]

    if __predicting == 'r2_with_r1':
        cols.append(voto_r1)

    return cols


# ---------------------- Funciones auxiliares ---------------------------------


def __save_data_file(df, prefix):

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    filename = prefix + '/' + prefix + str(datetime.now().time())
    filename = filename.replace(':', '_').replace('.', '_')
    filename += '.data'

    guardar_como_csv(df, filename)

    return filename


def main():
    #
    # generar y normalizar las muestras
    data = generar_muestra_pais(20000)
    df_data = rl_normalize(data, 'os')
    #
    # separar 80% para entrenar y validar
    t_data = df_data.sample(frac=0.8)
    # el otro 20% del inicial para set holdout
    data_to_predict = df_data.drop(t_data.index)
    #
    # separar 25% del conjunto de entrenamiento para validar
    v_data = t_data.sample(frac=0.25)
    t_data = t_data.drop(v_data.index)
    #
    # instanciar el modelo
    modelo = regresion_logistica('reg_log',
                                 regularization='l2',
                                 predicting='r2_with_r1')
    #
    # entrenar el modelo
    modelo = rl_entrenar(modelo, t_data, 'reg_log')

    acc, avg_loss = rl_validar(modelo, v_data, 'reg_log')
    print('Precisión: ' + str(acc))
    print('Pérdida: ' + str(avg_loss))

    rl_predict(modelo, data_to_predict, 'reg_log')


if __name__ == '__main__':
    main()
