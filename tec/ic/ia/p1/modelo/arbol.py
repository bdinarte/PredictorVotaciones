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

def decision_tree_learning(examples, attrs, parent_examples=()):
    """
    Construccion del arbol de decision, que se entrena a partir de un conjunto de muestras observadas
    :param examples:
    :param attrs:
    :param parent_examples:
    :return:
    """
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif all_same_class(examples):
        return Nodo(examples[0]['VOTO'], True)
    elif len(attrs) == 0:
        return plurality_value(examples)
    else:
        A = choose_attribute(examples)
        tree = Arboln()
        attrnames = get_attrnames(A, examples)
        tree = Arboln(A, dataset.attrnames[A], plurality_value(examples))
        for (v_k, exs) in split_by(A, examples):
            subtree = decision_tree_learning(
                exs, remove_all(A, attrs), examples)
            tree.insertar(v_k, subtree)
        return tree


def plurality_value(atributo_columna , examples):
    """
    Retorna el valor plural o mas comun de un conjunto de datos, es decir, el que mas
    aparicios tiene en el set ingresado

    Example from StackOverflow:
    They would have said the majority class if there were only two classes.
    Plurality is just the generalization of majority to more than 2 classes.
    It just means take the most frequent class in that leaf and return that as your prediction.
    For example, if you are classifying the colors of balls, and there are 3 blue balls, 2 red balls,
    and 2 white balls in a leaf, return blue as your prediction.
    :param atributo_columna: atributo sobre el que se determinara el valor de mayor pluralidad
    :param examples: conjunto de muestras observadas
    :return: etiqueta del atributo con mayor pluralidad
    """

    index_attr = examples[0].index(atributo_columna)
    dictionary = {}
    for i in range(1, len(examples)):
        key = examples[i][index_attr]
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1

    return max(dictionary, key=dictionary.get)


def get_attrnames(atributo_columna, examples):
    """
    Funcion utilizada para obtener el conjunto de valores posibles
    qiue puede obtener un determinado atributo según el conjunto de
    datos observados con que se esta entrenando el modelo
    :param atributo_columna: Es el atributo sobre el cual se tomara la columna respectiva
    y se determinaran los posibles valores que pueden obtener las muestras
    :param examples: es el conjunto de muestras observadas
    :return: lista de atributos que se le pueden asignar a la columna atributo_columna
    """

    list_attrnames = []
    index_attr = examples[0].index(atributo_columna)

    for i in range(1, len(examples)):
        if list_attrnames.count(examples[i][index_attr]) == 0:
            list_attrnames.append(examples[i][index_attr])

    return list_attrnames



def remove_all(atributo_columna, attrs):
    """
    Elimina de la lista de atributos, todas las apariciones de un atributo en especifico
    :param atributo_columna: atributo que se eliminara de la lista attrs
    :param attrs: lista de atributos (header) en el que se encuentran todas las etiquetas
    :return: lista de atributos (header) sin el atributo_columna que se esta ingresando
    """
    apariciones = attrs.count(atributo_columna)
    for i in range(apariciones):
        attrs.remove(atributo_columna)
    return attrs


# ---------------------------------------------------------------------

class Nodo:
    def __init__(self, valor, es_hoja=False):
        self.es_hoja = es_hoja
        self.info = valor
        self.hijos = []


# ---------------------------------------------------------------------

class Arboln:
    def __init__(self):
        self.__raiz = None

    # ******************************************************************
    # Funcion de busqueda, retorna el nodo encontrado o retorna None
    # ******************************************************************
    def __buscar(self, valor, hermanos=None, pos=0):

        if pos >= len(hermanos):
            return None

        if hermanos[pos].info == valor:
            return hermanos[pos]

        nodo = self.__buscar(valor, hermanos[pos].hijos)
        if nodo is not None:
            return nodo

        nodo = self.__buscar(valor, hermanos, pos + 1)
        if nodo is not None:
            return nodo

        return None

    # ******************************************************************
    # Funcion para buscar un valor en el arbol y decir si se encuentra o no
    # ******************************************************************

    def buscar(self, valor):

        if self.__raiz == valor:
            return True

        if self.__buscar(valor, self.__raiz.hijos) is not None:
            return True
        return False

    # ******************************************************************
    # Insertar un nuevo nodo en el arbol
    # ******************************************************************

    def insertar(self, valor, val_padre=None, pos_hijo=0):

        if self.__raiz is None:
            self.__raiz = Nodo(valor)
            return True

        if val_padre == self.__raiz.info:
            padre = self.__raiz
        else:
            padre = self.__buscar(val_padre, self.__raiz.hijos, 0)

        if padre is not None:
            padre.hijos.insert(pos_hijo, Nodo(valor))
            return True

        return False

    # ******************************************************************
    # Retorna la informacion del padre con mas hijos
    # ******************************************************************

    def padre_mas_hijos(self, nodos=None, pos=0):

        if nodos is None:
            if self.__raiz is None:
                return None
            nodos = [self.__raiz]
            self.__mayorpadre = self.__raiz

        if pos >= len(nodos):
            return 0

        if len(nodos[pos].hijos) > len(self.__mayorpadre.hijos):
            self.__mayorpadre = nodos[pos]

        self.padre_mas_hijos(nodos[pos].hijos)
        self.padre_mas_hijos(nodos, pos + 1)

        return self.__mayorpadre.info

    # ******************************************************************
    # Retorna el nro de hijos unicos (sin hermanos) en el arbol la raiz siempre es hijo unico
    # ******************************************************************

    def hijos_unicos(self, nodos=None, pos=0):
        if nodos is None:
            if self.__raiz is None:
                return 0
            nodos = [self.__raiz]

        if pos >= len(nodos):
            return 0

        h_unico = 0
        if len(nodos) == 1:
            h_unico = 1

        h_unicos_hijos = self.hijos_unicos(nodos[pos].hijos)
        h_unicos_hermanos = self.hijos_unicos(nodos, pos + 1)

        return h_unico + h_unicos_hijos + h_unicos_hermanos

    # ******************************************************************
    # Retorna True si dos valores indicados son nodos hermanos en el arbol n-ario
    # ******************************************************************

    def son_hermanos(self, fulano, sutano, nodos=None, pos=0):
        if nodos is None:
            if self.__raiz is None:
                return False
            nodos = [self.__raiz]

        if pos >= len(nodos):
            return False

        hermano = None
        if fulano == nodos[pos].info:  # Existe Fulano
            hermano = sutano
        elif sutano == nodos[pos].info:  # Existe Mengano
            hermano = fulano

        if hermano is not None:  # Buscar el hermano si exite fulano o sutano
            for nodo in nodos:
                if hermano == nodo.info:  # Encuentra al hermano
                    return True

        encontro = self.son_hermanos(fulano, sutano, nodos[pos].hijos)
        if encontro:
            return True

        return self.son_hermanos(fulano, sutano, nodos, pos + 1)

    # ******************************************************************
    # Recorrido en Preorden
    # ******************************************************************

    def preorden(self, nodos=None, pos=0):

        if nodos is None:
            if self.__raiz is None:
                return
            nodos = [self.__raiz]

        if pos >= len(nodos):
            return

        print(nodos[pos].info)
        self.preorden(nodos[pos].hijos)
        self.preorden(nodos, pos + 1)

    # ******************************************************************
    # Retorna la cantidad de nodos en el arbol que tienen mas de n hijos
    # ******************************************************************

    def nodos_mas_hijos_de(self, n, nodos=None, pos=0):
        if nodos is None:
            if self.__raiz is None:
                return 0
            nodos = [self.__raiz]

        if pos >= len(nodos):
            return 0

        cont = 0
        if len(nodos[pos].hijos) > n:
            cont = 1

        cont += self.nodos_mas_hijos_de(n, nodos[pos].hijos)
        cont += self.nodos_mas_hijos_de(n, nodos, pos + 1)

        return cont


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
    return DataSet([row[:column] + row[column + 1:] for row in returnData])


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

    valor_entropia = -(q * log(q, 2) + (1 - q) * log(1 - q, 2))

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
    for i in range(0, largo_muestra - 1):
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

def choose_attribute(examples):
    """
    Funcion utilizada para determinar el atributo que tiene mejor ganancia de informacion
    a partir de un conjunto de muestras de ejemplo que son pasadas por parametro
    :param examples: conjunto de muestras sobre las que se determina el atributo con
    mayor ganancia de informacion
    :return: etiqueta del atributo con mayor ganancia de informacion
    """

    dataSet = DataSet(examples)
    featuresInfoGain = {}

    for column in range(0, dataSet.get_data()[0].__len__() - 1):
        feature = Feature(column, dataSet.get_data())
        dataSets = [DataSet for i in range(feature.get_values().__len__())]

        i = 0
        for featureValue in feature.get_values():
            dataSets[i] = create_data_set(featureValue, column, dataSet.get_data())
            i += 1

        summation = 0
        for i in range(0, dataSets.__len__()):
            summation += (((dataSets[i].get_data()).__len__() - 1) /
                          (dataSet.get_data().__len__() - 1)) * dataSets[i].get_entropy()
        featuresInfoGain[feature] = dataSet.get_entropy() - summation

    # print(generate_info_gain_table(featuresInfoGain))
    # print("\nBest feature to split on is: ", max(featuresInfoGain, key=featuresInfoGain.get), "\n")

    return max(featuresInfoGain, key=featuresInfoGain.get)


# ---------------------------------------------------------------------
# Ejemplos y pruebas
# ---------------------------------------------------------------------

array = [
    ['a', 's', 'd', 'f', 'g', 'Meta'],
    [1, 1, 3, 4, 5, False],
    [1, 9, 1, 4, 8, True],
    [1, 2, 3, 3, 4, False],
    [1, 7, 2, 4, 5, False],
    [1, 7, 2, 4, 5, True]
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
    ['PUNTARENAS', 'NO', '20', 'NO', 'PAC']
]

print(plurality_value('PROVINCIA', muestra_prueba))