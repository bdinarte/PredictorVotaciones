# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------

# Columnas que retorna el simulador

columnas_csv = np.array([
    'CANTON',
    'EDAD',
    'ES_URBANO',
    'SEXO',
    'ES_DEPENDIENTE',
    'ESTADO_VIVIENDA',
    'E.HACINAMIENTO',
    'ALFABETIZACION',
    'ESCOLARIDAD_PROMEDIO',
    'ASISTENCIA_EDUCACION',
    'FUERZA_DE_TRABAJO',
    'SEGURO',
    'N.EXTRANJERO',
    'C.DISCAPACIDAD',
    'POBLACION_TOTAL',
    'SUPERFICIE',
    'DENSIDAD_POBLACION',
    'VIVIENDAS_INDIVIDUALES_OCUPADAS',
    'PROMEDIO_DE_OCUPANTES',
    'P.JEFAT.FEMENINA',
    'P.JEFAT.COMPARTIDA',
    'VOTO_R1',
    'VOTO_R2'
])

# -----------------------------------------------------------------------------

# Columnas adicionales que se debe agregar

columnas_salida = np.append(columnas_csv, [
    'ES_ENTRENAMIENTO', 'PREDICCION_R1',
    'PREDICCION_R2', 'PREDICCION_R2_CON_R1'])

# -----------------------------------------------------------------------------
