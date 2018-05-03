
import os
import sys
sys.path.append('../..')

from pandas import DataFrame
from datetime import datetime

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

from tec.ic.ia.pc1.g03 import generar_muestra_pais, generar_muestra_provincia
from p1.modelo.normalizacion import normalize, categoric_to_numeric
from p1.modelo.normalizacion import id_to_partidos_r1, id_to_partidos_r2
from p1.modelo.manejo_archivos import guardar_como_csv
from p1.util.util import agrupar

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
walk_data_times = 51
available_activations = ['relu', 'softmax', 'softplus']
#
# entre mas pequeño es el batch, mas lento el entrenamiento pero converge con
# menos pasadas
batch_size = 1500
__predicting = ''
__learning_rate = 0.03
__k_fold_amount = 4
__display_console_info = False

# ------------------------ Funciones públicas ---------------------------------


def neural_network(layer_amount=3, units_per_layer=10, activation_f='relu',
                   activation_on_output='', predicting='r1'):
    """
    funcion para crear un nuevo modelo de red neuronal
    :param layer_amount: cantidad de capas de la red
    :param units_per_layer: unidades por cada intermedia
    :param activation_f: 'relu', 'softmax', 'softplus'
    :param activation_on_output: activation_f o '' para dejar outputs sin
    funcion
    :param predicting: 'r1', 'r2', 'r2_with_r1' -> ronda a predecir
    :return: modelo de red neuronal
    """

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
    output_amount = 15 if predicting is 'r1' else 4
    #
    # para la última capa es posible no definir una función de activación
    if activation_on_output:
        nn_layers.append(tf.keras.layers.Dense(
            output_amount,
            activation=activation_on_output))
    else:
        nn_layers.append(tf.keras.layers.Dense(output_amount))
    #
    # Crear el modelo según las capas definidas
    model = tf.keras.Sequential(nn_layers)

    return model


def nn_entrenar(model, train_data, prefix):
    """
    Entrena un modelo de red neuronal
    :param model: modelo de red neuronal
    :param train_data: dataframe con datos de entrenamiento normalizados
    :param prefix: prefijo para los nombres de archivos
    :return:
    """
    global __learning_rate
    train_dataset = __nn_build_dataset(train_data, prefix)
    #
    # definir el tipo de optimizacion y learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=__learning_rate)
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

        if __display_console_info:
            if current_walk % 10 == 0:
                print("Pasada {:02d}: Pérdida: {:.3f}, Precisión: "
                      "{:.2%}".format(
                        current_walk,
                        current_walk_loss_avg.result(),
                        current_walk_accuracy.result()))

    if __display_console_info:
        __show_graphics(loss_for_training, accuracy_for_training)

    return model


def nn_validar(model, validation_data, prefix):
    """
    Realiza la validación de la precisión de un modelo ya existente
    :param model: modelo de red neuronal
    :param validation_data: dataframe con datos para validación normalizados
    :param prefix: prefijo para los archivos generados
    :return: TODO: definir retorno
    """
    validation_dataset = __nn_build_dataset(validation_data, prefix)

    test_accuracy = tfe.metrics.Accuracy()

    for x, y in tfe.Iterator(validation_dataset):
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    #
    # retorna un flotante con precision 3
    return eval('{:.3}'.format(test_accuracy.result()))


def nn_predict(model, df_data):
    """
    Remueve las etiquetas de los datos recibidos y las predice
    :param model: modelo nn
    :param df_data: dataframe con los datos normalizados
    :return: TODO: de momento un conteo de etiquetas
    """
    data_list = df_data.values.tolist()

    if __predicting == 'r1':
        num_to_label = id_to_partidos_r1()
        data_to_predict = [row[:-2] for row in data_list]
    else:
        num_to_label = id_to_partidos_r2()
        if __predicting is 'r2_with_r1':
            data_to_predict = [row[:-1] for row in data_list]
        else:
            data_to_predict = [row[:-2] for row in data_list]

    predict_dataset = tf.convert_to_tensor(data_to_predict)

    predictions = model(predict_dataset)

    _predictions = list()
    for i, logits in enumerate(predictions):
        label_id = tf.argmax(logits).numpy()
        label = num_to_label[label_id]
        _predictions.append('{}'.format(label))

    return _predictions


def nn_validar_alt(model, validation_data, prefix):

    predictions = nn_predict(model, validation_data)

    if __predicting == 'r1':
        dicc = id_to_partidos_r1()
        true_vals = validation_data['VOTO_R1'].values.tolist()
    else:
        dicc = id_to_partidos_r2()
        true_vals = validation_data['VOTO_R2'].values.tolist()

    rights = 0
    for pred, val in zip(predictions, true_vals):
        if pred == dicc[val]:
            rights += 1

    print('')
    return rights / len(predictions)


def nn_normalize(data_list, normalization):
    """
    Normaliza atributos numéricos y convierte categóricos a binarios
    :param data_list: lista de listas con los datos
    :param normalization: 'os', 'ss', 'fs' -> overmax, standard, feature
    :return: dataframe con datos normalizados
    """
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

    return data


# ---------------------- Funciones del modelo ---------------------------------


def __nn_build_dataset(data, prefix):
    #
    # Generar archivos de datos
    filename = __save_data_file(data, prefix)
    #
    # Parsear el archivo con data
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(__parse_csv)
    dataset = dataset.batch(batch_size)

    return dataset


def __loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def __grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = __loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def __parse_csv(line):
    global __predicting
    if __predicting == 'r1':
        example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                            [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                            [0.], [0.], [0.], [0.], [0], [0]]
    else:
        example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                            [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                            [0.], [0.], [0.], [0.], [0.], [0]]

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


def __split(_list, n):
    k, m = divmod(len(_list), n)
    return [_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def __show_graphics(loss_results, accuracy_results):

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Métricas del entrenamiento')

    axes[0].set_ylabel("Pérdida", fontsize=14)
    axes[0].plot(loss_results)

    axes[1].set_ylabel("Precisión", fontsize=14)
    axes[1].set_xlabel("Pasadas", fontsize=14)
    axes[1].plot(accuracy_results)

    plt.show()


def __save_data_file(df, prefix):

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    filename = prefix + '/' + prefix + str(datetime.now().time())
    filename = filename.replace(':', '_').replace('.', '_')
    filename += '.data'

    guardar_como_csv(df, filename)

    return filename


def run_nn(sample_size=3000, normalization='os', test_percent=0.2, layers=3,
           units_per_layer=5, activation_f='relu', activation_out='',
           predicting='r1', prefix='nn_', provincia=''):
    #
    # generar y normalizar las muestras
    if provincia:
        data = generar_muestra_provincia(sample_size, provincia)
    else:
        data = generar_muestra_pais(sample_size)

    df_data = nn_normalize(data, normalization)
    #
    # separar un porcentaje para entrenar y uno de validar
    training_data = df_data.sample(frac=(1 - test_percent))
    #
    # extraer el conjunto de pruebas
    test_data = df_data.drop(training_data.index)
    #
    # la línea siguiente convierte de dataframe a lista preservando el tipo
    # de dato original
    training_list = list(list(x) for x in zip(
        *(training_data[x].values.tolist() for x in training_data.columns)))

    k_groups = __split(training_list, __k_fold_amount)
    models = list()
    accuracies = list()
    for v_index in range(__k_fold_amount):
        _prefix = prefix + '_' + str(v_index)
        #
        # instanciar el modelo
        models.append(neural_network(layer_amount=layers,
                                     units_per_layer=units_per_layer,
                                     activation_f=activation_f,
                                     activation_on_output=activation_out,
                                     predicting=predicting))
        v_data, t_subset = agrupar(v_index, k_groups)
        t_subset = DataFrame(t_subset, columns=data_columns[1:])
        v_data = DataFrame(v_data, columns=data_columns[1:])
        nn_entrenar(models[v_index], t_subset, _prefix)
        accuracies.append(nn_validar_alt(models[v_index], v_data, _prefix))
        print('Subset ' + str(v_index) + ' completo.')

    print('Precisión de cada entrenamiento:')
    for i in range(__k_fold_amount):
        print('Subset ' + str(i) + ': ' + str(accuracies[i]))


run_nn(sample_size=1000, activation_out='softmax', predicting='r2_with_r1')