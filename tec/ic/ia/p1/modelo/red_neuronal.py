
from pandas import DataFrame
from datetime import datetime

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tec.ic.ia.pc1.g03 import generar_muestra_pais
from modelo.normalizacion import normalize, categoric_to_numeric
from modelo.manejo_archivos import guardar_como_csv

tf.enable_eager_execution()

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
walk_data_times = 100
available_activations = ['relu', 'softmax']
shuffle_buffer_size = 10000
batch_size = 100
__predicting = ''

# ------------------------ Funciones públicas ---------------------------------


def red(data_list, normalization, prefix, layer_amount=3,
        units_per_layer=10, activation_f='relu', predicting='r1'):
    #
    # tipo de prediccion para ser leido por parse_csv
    global __predicting
    __predicting = predicting
    #
    # 'relu' por defecto en caso de no existir la ingresada
    if activation_f not in available_activations:
        activation_f = 'relu'
    #
    # por defecto se requiere una capa como mínimo
    layer_amount -= 2
    if layer_amount < 1:
        layer_amount = 1
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
    filename = __save_data_file(data, prefix)
    #
    # Parsear el archivo con data
    train_dataset = tf.data.TextLineDataset(filename)
    train_dataset = train_dataset.map(__parse_csv)
    train_dataset = train_dataset.shuffle(shuffle_buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    #
    # calcular la cantidad de atributos
    input_shape = (21,) if predicting == 'r2_with_r1' else (20,)
    #
    # definir las capas para el modelo
    nn_layers = list()
    nn_layers.append(tf.keras.layers.Dense(units_per_layer,
                                           activation=activation_f,
                                           input_shape=input_shape))
    for layer in range(layer_amount):
        nn_layers.append(tf.keras.layers.Dense(units_per_layer,
                                               activation=activation_f))
    #
    # modificar la cantidad de unidades de salida segun las etiquetas
    output_amount = 4 if predicting == 'r2_with_r1' else 15
    nn_layers.append(tf.keras.layers.Dense(output_amount))
    #
    # Crear el modelo según las capas definidas
    model = tf.keras.Sequential(nn_layers)

    return model, train_dataset


def entrenar(model, train_dataset):
    #
    # definir el tipo de optimizacion y learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
    #
    # recopilar la precision y perdida por pasada a los datos
    loss_for_training = []
    accuracy_for_training = []
    #
    # comenzar las pasadas a los datos para entrenar
    for current_walk in range(walk_data_times):
        current_walk_loss_avg = tfe.metrics.Mean()
        current_walk_accuracy = tfe.metrics.Accuracy()

        for x, y in tfe.Iterator(train_dataset):
            grads = __grad(model, x, y)
            optimizer.apply_gradients(
                zip(grads, model.variables),
                global_step=tf.train.get_or_create_global_step())

            current_walk_loss_avg(__loss(model, x, y))
            current_walk_accuracy(tf.argmax(model(x), axis=1,
                                            output_type=tf.int32), y)
        #
        # salvar las metricas
        loss_for_training.append(current_walk_loss_avg.result())
        accuracy_for_training.append(current_walk_accuracy.result())

        if current_walk % 33 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, "
                  "Accuracy: {:.3%}".format(current_walk,
                                            current_walk_loss_avg.result(),
                                            current_walk_accuracy.result()))


# ---------------------- Funciones del modelo ---------------------------------


def __loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def __grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = __loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def __parse_csv(line):
    global __predicting
    example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                        [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    if __predicting == 'r1':
        features = tf.reshape(parsed_line[:-2], shape=(20,))
        label = tf.reshape(parsed_line[-2], shape=())
    elif __predicting == 'r2':
        features = tf.reshape(parsed_line[:-2], shape=(20,))
        label = tf.reshape(parsed_line[-1], shape=())
    else:
        features = tf.reshape(parsed_line[:-1], shape=(21,))
        label = tf.reshape(parsed_line[-1], shape=())
    return features, label


# ---------------------- Funciones auxiliares ---------------------------------


def __save_data_file(df, prefix):

    filename = prefix + str(datetime.now().time())
    filename = filename.replace(':', '_').replace('.', '_')
    filename += '.data'

    guardar_como_csv(df, filename)

    return filename


def main():
    t_data = generar_muestra_pais(500)
    modelo, training_dataset = red(t_data, 'os', 'training', 4, 5,
                                   predicting='r2_with_r1')
    entrenar(modelo, training_dataset)


if __name__ == '__main__':
    main()
