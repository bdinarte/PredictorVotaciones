
from pandas import DataFrame

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tec.ic.ia.pc1.g03 import generar_muestra_pais
from modelo.normalizacion import normalize, categoric_to_numeric

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


def red(data, normalization):

    data, features, labels = setup_data(data, normalization)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=0)

    #
    # Ajustando el modelo
    learning_rate = 0.001
    num_epochs = 1500
    display_step = 1

    features = tf.placeholder(tf.float32, [None, train_features.shape[1]])
    labels = tf.placeholder(tf.float32, [None, train_labels.shape[1]])

    weights = tf.Variable(tf.zeros([train_features.shape[1],
                                    train_labels.shape[1]]))
    bias = tf.Variable(tf.zeros([train_labels.shape[1]]))

    labels_ = tf.nn.softmax(tf.add(tf.matmul(features, weights), bias))

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                   logits=labels_)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(cost)

    tf.Session().run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        cost_per_epoch = 0
        _, curr_cost = tf.Session().run([optimizer, cost],
                                        feed_dict={features: train_features,
                                                   labels: train_labels})
        cost_per_epoch += curr_cost

    correct_prediction = tf.equal(tf.argmax(labels_, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({features: test_features,
                                      labels: test_labels}))


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

    data = generar_muestra_pais(100)
    red(data, 'os')


if __name__ == '__main__':
    main()
