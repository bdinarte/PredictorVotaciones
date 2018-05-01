
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

import tensorflow as tf
import tensorflow.contrib.eager as tfe

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


def red(data_list, normalization, prefijo):

    #
    # Setup de los datos generados por el simulador
    data = DataFrame(data_list, columns=data_columns)
    #
    # Normalización de las columnas densas
    data = normalize(data, cols_to_norm, normalization)
    #
    # Conversión de partidos a números
    data = categoric_to_numeric(data)
    data = data.drop('CANTON', axis=1)
    #
    # Generar archivos de datos
    filename = __save_data_file(data, prefijo)

    #
    # Parsear el archivo con data
    train_dataset = tf.data.TextLineDataset(filename)
    train_dataset = train_dataset.map(parse_csv)
    train_dataset = train_dataset.batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(20,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(15)
    ])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)

    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 101

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


# ---------------------- Funciones auxiliares ---------------------------------


def __save_data_file(df, prefix):

    filename = prefix + str(datetime.now().time())
    filename = filename.replace(':', '_').replace('.', '_')
    filename += '.data'

    guardar_como_csv(df, filename)

    return filename


def main():
    t_data = generar_muestra_pais(100)
    red(t_data, 'os', 'training')


if __name__ == '__main__':
    main()
