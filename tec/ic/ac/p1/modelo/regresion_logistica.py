
import tensorflow as tf
from modelo.manejo_archivos import obtener_dataframe, guardar_como_csv

_CSV_COLUMN_DEFAULTS = [[0],[0],[0],[0],[0],[0]]
_CSV_COLUMNS = ['JUNTAS','ACCION CIUDADANA','RESTAURACION NACIONAL', 'NULO',
                'BLANCO','VOTOS RECIBIDOS']

def entrenar(datos, regularizacion):
    pass



def __input_fn(data_file, num_epochs, shuffle, batch_size):

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

