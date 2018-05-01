
from pandas import DataFrame

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tec.ic.ia.pc1.g03 import generar_muestra_pais
from modelo.normalizacion import normalize, categoric_to_numeric
from modelo.manejo_archivos import guardar_como_csv

tf.enable_eager_execution()

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


def red(data_list, v_data, normalization):
    data = DataFrame(data_list, columns=data_columns)
    data = normalize(data, cols_to_norm, normalization)
    data = categoric_to_numeric(data)
    data = data.drop('CANTON', axis=1)
    guardar_como_csv(data, 'training.data')

    train_dataset = tf.data.TextLineDataset('training.data')
    train_dataset = train_dataset.map(parse_csv)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(20,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(15)
    ])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 500

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        for x, y in tfe.Iterator(train_dataset):

            grads = grad(model, x, y)
            optimizer.apply_gradients(
                zip(grads, model.variables),
                global_step=tf.train.get_or_create_global_step())

            epoch_loss_avg(loss(model, x, y))
            epoch_accuracy(tf.argmax(model(x), axis=1,
                                     output_type=tf.int32), y)

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, "
                  "Accuracy: {:.3%}".format(epoch,
                                            epoch_loss_avg.result(),
                                            epoch_accuracy.result()))


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[:-2], shape=(20,))
    label = tf.reshape(parsed_line[-2], shape=())
    return features, label


def setup_data(data_list, normalization):

    data = DataFrame(data_list, columns=data_columns)
    data = normalize(data, cols_to_norm, normalization)
    data = categoric_to_numeric(data)
    data = data.drop('CANTON', axis=1)

    features = data.loc[:, ['EDAD', 'ES_URBANO', 'SEXO', 'ES_DEPENDIENTE',
                            'ESTADO_VIVIENDA', 'E.HACINAMIENTO',
                            'ALFABETIZACION', 'ESCOLARIDAD_PROMEDIO',
                            'ASISTENCIA_EDUCACION', 'FUERZA_DE_TRABAJO',
                            'SEGURO', 'N.EXTRANJERO', 'C.DISCAPACIDAD',
                            'POBLACION_TOTAL', 'SUPERFICIE',
                            'DENSIDAD_POBLACION',
                            'VIVIENDAS_INDIVIDUALES_OCUPADAS',
                            'PROMEDIO_DE_OCUPANTES', 'P.JEFAT.FEMENINA',
                            'P.JEFAT.COMPARTIDA']]

    labels = data.loc[:, ['VOTO_R1']]
    features, labels = codificar_one_hot(features, labels)
    return data, features, labels


def codificar_one_hot(features, labels):

    one_hot = OneHotEncoder()
    one_hot.fit(features)
    features = one_hot.transform(features).toarray()
    one_hot.fit(labels)
    labels = one_hot.transform(labels).toarray()

    return features, labels


def main():
    t_data = generar_muestra_pais(500)
    v_data = generar_muestra_pais(20)
    red(t_data, v_data, 'os')


if __name__ == '__main__':
    main()
