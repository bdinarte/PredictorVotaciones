# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------

# Columnas que retorna el simulador

columnas = np.array([
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

# Columnas con valores categóricos
__indices = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
columnas_c = columnas[__indices]
# print(columnas_c) # Para comprobar que son las columnas de interés

# -----------------------------------------------------------------------------

# Columnas con valores númericos (que se debe normalizar)
columnas_n = np.delete(columnas, __indices)
# print(columnas_n) # Para comprobar que son las columnas de interés

# -----------------------------------------------------------------------------
