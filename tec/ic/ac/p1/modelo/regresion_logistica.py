
import tensorflow as tf
from modelo.manejo_archivos import obtener_dataframe, guardar_como_csv
from modelo.normalizacion import normalize

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [''], [''], [''], [''], [''],
                        [0], [''], [''], [''], [''], [''], [0], [0], [0],
                        [0], [0], [0], [0], [''], ['']]

_CSV_COLUMNS = ['VOTANTE', 'CANTON', 'EDAD', 'ES URBANO', 'SEXO',
                'ES DEPENDIENTE', 'ESTADO VIVIENDA', 'E.HACINAMIENTO',
                'ALFABETIZACIÓN', 'ESCOLARIDAD PROMEDIO',
                'ASISTENCIA EDUCACION', 'FUERZA DE TRABAJO', 'SEGURO',
                'N.EXTRANJERO', 'C.DISCAPACIDAD', 'POBLACION TOTAL',
                'SUPERFICIE', 'DENSIDAD POBLACION',
                'VIVIENDAS INDIVIDUALES OCUPADAS', 'PROMEDIO DE OCUPANTES',
                'P.JEFAT.FEMENINA', 'P.JEFAT.COMPARTIDA', 'VOTO_R1', 'VOTO_R2']

_SHUFFLE_BUFFER = 1000

data = obtener_dataframe('../archivos/pruebas.csv')

atributos_a_normalizar = ['EDAD', 'ESCOLARIDAD PROMEDIO', 'POBLACION TOTAL',
                          'SUPERFICIE', 'DENSIDAD POBLACION',
                          'VIVIENDAS INDIVIDUALES OCUPADAS',
                          'PROMEDIO DE OCUPANTES', 'P.JEFAT.FEMENINA',
                          'P.JEFAT.COMPARTIDA']

data = normalize(data, atributos_a_normalizar, 'os')

print(data)

def __build_model_columns():
    pass


def entrenar(datos, regularizacion):
    pass


def __input_fn(data_file, num_epochs, shuffle, batch_size):

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('VOTO_R2')
        labels = features.pop('VOTO_R1')

        return features, labels

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

