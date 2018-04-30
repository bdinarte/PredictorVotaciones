
from math import log
from texttable import Texttable


"""
# Pseudocodigo del arbol de decision  

def decision_tree_learning(examples, attrs, parent_examples=()):
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif all_same_class(examples):
        return DecisionLeaf(examples[0][target])
    elif len(attrs) == 0:
        return plurality_value(examples)
    else:
        A = choose_attribute(attrs, examples)
        tree = DecisionFork(A, dataset.attrnames[A], plurality_value(examples))
        for (v_k, exs) in split_by(A, examples):
            subtree = decision_tree_learning(
                exs, removeall(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree
"""


# ---------------------------------------------------------------------

class Nodo:
    def __init__(self, valor):
        self.info = valor
        self.hijos = []


# ---------------------------------------------------------------------

class Arboln:
    def __init__(self):
        self.__raiz = None

    def __buscar(self, valor, hermanos=None, pos=0):

        if pos >= len(hermanos):
            return None

        if hermanos[pos].info == valor:
            return hermanos[pos]

        nodo = self.__buscar(valor, hermanos[pos].hijos)
        if nodo != None:
            return nodo

        nodo = self.__buscar(valor, hermanos, pos + 1)
        if nodo != None:
            return nodo

        return None

    def buscar(self, valor):

        if self.__raiz == valor:
            return True

        if self.__buscar(valor, self.__raiz.hijos) != None:
            return True
        return False


# ---------------------------------------------------------------------

class DataSet:
    def __init__(self, data):
        self._data = data
        self._entropy = 0
        for featureValue in Feature((data[0].__len__() - 1), data).get_values():
            self._entropy += self.minus_plog_2(featureValue.get_occurences() / data.__len__() - 1)

    def get_entropy(self):
        return self._entropy

    def get_data(self):
        return self._data

    def minus_plog_2(self, p):
        return_value = 0
        p = abs(p)
        if p != 0:
            return_value = (-1) * p * log(p, 2)
        return return_value

    def __str__(self):
        table = Texttable()
        table.header(self._data[0])
        for i in range(1, self._data.__len__()):
            table.add_row(self._data[i])
        return table.draw().__str__()


# ---------------------------------------------------------------------

class Feature:
    def __init__(self, column, data):
        self._name = data[0][column]
        self._values = set()

        for row in range(1, data.__len__()):
            self._values.add(FeatureValue(data[row][column]))
        for featureValue in self._values:
            counter = 0
            for row in range(1, data.__len__()):
                if featureValue.get_name() == data[row][column]:
                    counter += 1
                    featureValue.set_occurences(counter)

    def get_values(self):
        return self._values

    def get_name(self):
        return self._name

    def __str__(self):
        return self._name


# ---------------------------------------------------------------------

class FeatureValue:
    def __init__(self, name):
        self._name = name
        self._occurences = 0

    def __eq__(self, other):
        return (isinstance(other, FeatureValue)) and (other._name == self._name)

    def __hash__(self):
        return hash(self._name)

    def get_name(self):
        return self._name

    def get_occurences(self):
        return self._occurences

    def set_occurences(self, occurences):
        self._occurences = occurences

    def __str__(self):
        return self._name


# ---------------------------------------------------------------------

def create_data_set(featureValue, column, data):
    returnData = [[FeatureValue for i in range(data[0].__len__())]
              for j in range(featureValue.get_occurences() + 1)]

    returnData[0] = data[0]
    counter = 1

    for row in range(1, data.__len__()):
        if data[row][column] == featureValue.get_name():
            returnData[counter] = data[row]
            counter += 1
    return DataSet([row[:column] + row[column+1:] for row in returnData])


# ---------------------------------------------------------------------

def generate_info_gain_table(featuresInfoGain):
    table = Texttable()
    table.header(['Feature', 'Information Gain'])

    for key in featuresInfoGain:
        table.add_row([key, round(featuresInfoGain[key], 5)])

    return table.draw().__str__()


# ---------------------------------------------------------------------

def entropia(q):
    """
    Funcion que se encarga de determinar la geomancia de informacion de una cierta variable
    Se utiliza para la seleccion de atributos, permitiendo seleccionar los que redezcan la incertidumbre
    de clasificacion.
    :param q: variable a la cual se le calculara el valor de entropia, para determinar el grado de incertidumbre
    con que cuenta, y asi poder utilizar el calculo en la ganancia de información final
    :return: valor de entropia de la variable evaluada
    """

    valor_entropia = -(q * log(q, 2) + (1-q) * log(1-q, 2))

    return valor_entropia


# ---------------------------------------------------------------------

def ganancia(muestra):
    """
    Funcion para determinar la ganancia de informacion que brinda una determinada variable o atributo
    :param muestra: es el conjunto de datos de una determinada variable de la cual se obtendra la ganancia
    de informacion
    :return: valor con la ganancia de informacion
    """

    proporcion_pos = porporcion_positivos(muestra)
    valor_entropia = entropia(proporcion_pos)

    return valor_entropia - resto(muestra)


# ---------------------------------------------------------------------

def resto(conjunto_muestras):
    largo_muestra = len(conjunto_muestras)

    resultado = 0
    for i in range(0, largo_muestra-1):
        muestra = conjunto_muestras[i]

    return resultado


# ---------------------------------------------------------------------

def porporcion_positivos(muestra):
    """
    Funcion encargada de determinar la porporcion de positivos que el conjunto de muestras
    :param muestra: Es el conjunto de observaciones que se van a utilizar para entrenar el sistema
    de las cuales se determinara la proporcion de positivos
    :return: porcentaje de positivos (del atributo meta) que contiene el conjunto de muestras
    """

    total_muestras = len(muestra)
    positivos = 0
    for i in range(total_muestras):
        muestra_aux = muestra[i]
        dato_meta = muestra_aux[-1]
        if dato_meta:
            positivos += 1

    return positivos / total_muestras


# ---------------------------------------------------------------------
# Ejemplos y pruebas
# ---------------------------------------------------------------------

array = [
    ['a','s','d','f','g','Meta'],
    [1,1,3,4,5,False],
    [1,9,1,4,8,True],
    [1,2,3,3,4,False],
    [1,7,2,4,5,False],
    [1,7,2,4,5,True]
]

ie_rest = [
    [True],
    [False],
    [True],
    [True],
    [False],
    [True],
    [False],
    [True],
    [False],
    [False],
    [True],
    [True]
]


muestra_prueba = [
    ['PROVINCIA', 'TRABAJADOR', 'EDAD', 'SOLTERO', 'VOTO'],
    ['SAN JOSE', 'SI', '25', 'SI', 'RES'],
    ['CARTAGO', 'SI', '22', 'NO', 'PAC'],
    ['HEREDIA', 'SI', '50', 'NO', 'PAC'],
    ['HEREDIA', 'NO', '70', 'SI', 'LIB'],
    ['HEREDIA', 'SI', '30', 'NO', 'PAC'],
    ['CARTAGO', 'NO', '18', 'SI', 'RES'],
    ['HEREDIA', 'NO', '20', 'NO', 'PAC'],
]

datas = {'Random': muestra_prueba}

for key in datas:
    print(key, 'DATASET:')
    dataSet = DataSet(datas[key])
    print(dataSet)
    featuresInfoGain = {}

    for column in range(0, dataSet.get_data()[0].__len__() - 1):
        feature = Feature(column, dataSet.get_data())
        dataSets = [DataSet for i in range(feature.get_values().__len__())]

        i = 0
        for featureValue in  feature.get_values():
            dataSets[i] = create_data_set(featureValue, column, dataSet.get_data())
            i += 1

        summation = 0
        for i in range(0, dataSets.__len__()):
            summation += (((dataSets[i].get_data()).__len__() - 1) /
                          (dataSet.get_data().__len__() - 1)) * dataSets[i].get_entropy()
        featuresInfoGain[feature] = dataSet.get_entropy() - summation

    print(generate_info_gain_table(featuresInfoGain))
    print("\nBest feature to split on is: ", max(featuresInfoGain, key=featuresInfoGain.get), "\n")