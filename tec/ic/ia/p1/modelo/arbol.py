from __future__ import division
from math import log
from texttable import Texttable
import math
from collections import Counter


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
        return plurality_value(parent_examples[0], parent_examples)
    elif len(examples[0]) == 1:
        return Nodo(examples[0][-1], True)
    elif len(attrs) == 0:
        return plurality_value(examples[0], examples)
    else:
        A = str(choose_attribute(examples))
        tree = NaryTree()
        attrnames = get_attrnames(A, examples)

        tree.insertar(A)
        for name in attrnames:
            tree.insertar(name, A)
        plural_value = plurality_value(A, examples)

        exs = delete_attr(A, examples)
        attrs = remove_attr(A, attrs)
        for v_k in attrnames:
            subtree = decision_tree_learning(
                exs, attrs, examples)
            tree.insertar(subtree, v_k)
        return tree


def delete_attr(attr, examples):
    """
    Funcion utilizada para eliminar un atributo columna(attr) de un set de datos (examples)
    :param attr: etiqueta del atributo que se va eliminar
    :param examples: conjunto de datos al que se le eliminara la etiqueta attr
    :return: set de datos sin el atributo columna, pasado por parametro
    """
    index = examples[0].index(attr)

    for i in range(len(examples)):
        del examples[i][index]

    return examples

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



def remove_attr(atributo_columna, attrs):
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

    def get_info(self):
        return self.info

    def get_hijos(self):
        return self.hijos


# ---------------------------------------------------------------------

class Arboln:
    def __init__(self):
        self.__raiz = None

    def get_raiz(self):
        return self.__raiz

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

##################################################
# data class to hold csv data
##################################################
class data():
    def __init__(self, classifier):
        self.examples = []
        self.attributes = []
        self.attr_types = []
        self.classifier = classifier
        self.class_index = None

"""

##################################################
# Preprocess dataset
##################################################
def preprocess2(dataset):
    print("Preprocessing data...")

    class_values = [example[dataset.class_index] for example in dataset.examples]
    class_mode = Counter(class_values)
    class_mode = class_mode.most_common(1)[0][0]

    for attr_index in range(len(dataset.attributes)):

        ex_0class = filter(lambda x: x[dataset.class_index] == '0', dataset.examples)
        values_0class = [example[attr_index] for example in ex_0class]

        ex_1class = filter(lambda x: x[dataset.class_index] == '1', dataset.examples)
        values_1class = [example[attr_index] for example in ex_1class]

        values = Counter(values_0class)
        value_counts = values.most_common()

        mode0 = values.most_common(1)[0][0]
        if mode0 == '?':
            mode0 = values.most_common(2)[1][0]

        values = Counter(values_1class)
        mode1 = values.most_common(1)[0][0]

        if mode1 == '?':
            mode1 = values.most_common(2)[1][0]

        mode_01 = [mode0, mode1]

        attr_modes = [0] * len(dataset.attributes)
        attr_modes[attr_index] = mode_01

        for example in dataset.examples:
            if (example[attr_index] == '?'):
                if (example[dataset.class_index] == '0'):
                    example[attr_index] = attr_modes[attr_index][0]
                elif (example[dataset.class_index] == '1'):
                    example[attr_index] = attr_modes[attr_index][1]
                else:
                    example[attr_index] = class_mode

        # convert attributes that are numeric to floats
        for example in dataset.examples:
            for x in range(len(dataset.examples[0])):
                if dataset.attributes[x] == 'True':
                    example[x] = float(example[x])
"""


##################################################
# Preprocess dataset
##################################################
def preprocess2(dataset):
    print("Preprocessing data...")

    class_values = [example[dataset.class_index] for example in dataset.examples]
    class_mode = Counter(class_values)
    class_mode = class_mode.most_common(1)[0][0]

    for attr_index in range(len(dataset.attributes)):

        ex_pacclass = filter(lambda x: x[dataset.class_index] == 'PAC', dataset.examples)
        values_pacclass = [example[attr_index] for example in ex_pacclass]

        ex_plnclass = filter(lambda x: x[dataset.class_index] == 'PLN', dataset.examples)
        values_plnclass = [example[attr_index] for example in ex_plnclass]

        ex_puscclass = filter(lambda x: x[dataset.class_index] == 'PUSC', dataset.examples)
        values_puscclass = [example[attr_index] for example in ex_puscclass]

        values = Counter(values_pacclass)  # value_counts = values.most_common()
        modepln = values.most_common(1)[0][0]
        if modepln == '?':
            modepln = values.most_common(2)[1][0]

        values = Counter(values_plnclass)
        modepac = values.most_common(1)[0][0]
        if modepac == '?':
            modepac = values.most_common(2)[1][0]

        values = Counter(values_puscclass)
        modepusc = values.most_common(1)[0][0]
        if modepusc == '?':
            modepusc = values.most_common(2)[1][0]

        mode_012 = [modepln, modepac, modepusc]

        attr_modes = [0] * len(dataset.attributes)
        attr_modes[attr_index] = mode_012

        for example in dataset.examples:
            if (example[attr_index] == '?'):
                if (example[dataset.class_index] == 'PAC'):
                    example[attr_index] = attr_modes[attr_index][0]
                elif (example[dataset.class_index] == 'PLN'):
                    example[attr_index] = attr_modes[attr_index][1]
                elif (example[dataset.class_index] == 'PUSC'):
                    example[attr_index] = attr_modes[attr_index][2]
                else:
                    example[attr_index] = class_mode

        # convert attributes that are numeric to floats
        for example in dataset.examples:
            for x in range(len(dataset.examples[0])):
                if dataset.attributes[x] == 'True':
                    example[x] = float(example[x])


##################################################
# tree node class that will make up the tree
##################################################
class treeNode():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child,
                 height):
        self.is_leaf = True
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.parent = parent
        self.upper_child = None
        self.lower_child = None
        self.height = None


##################################################
# compute tree recursively
##################################################

# initialize Tree
# if dataset is pure (all one result) or there is other stopping criteria then stop
# for all attributes a in dataset
# compute information-theoretic criteria if we split on a
# abest = best attribute according to above
# tree = create a decision node that tests abest in the root
# dv (v=1,2,3,...) = induced sub-datasets from D based on abest
# for all dv
# tree = compute_tree(dv)
# attach tree to the corresponding branch of Tree
# return tree

def compute_tree(dataset, parent_node, classifier):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1

    ones = one_count(dataset.examples, dataset.attributes, classifier)

    if ones == 0:
        node.classification = 'NULO'
        node.is_leaf = True
        return node
    elif len(dataset.examples) == ones:
        node.classification = dataset.examples[0][-1]
        node.is_leaf = True
        return node
    else:
        node.is_leaf = False
    attr_to_split = None  # The index of the attribute we will split on
    max_gain = 0  # The gain given by the best attribute
    split_val = None
    min_gain = 0.01
    dataset_entropy = calc_dataset_entropy(dataset, classifier)
    for attr_index in range(len(dataset.examples[0])):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in
                               dataset.examples]  # these are the values we can split on, now we must find the best one
            attr_value_list = list(set(attr_value_list))  # remove duplicates from list of all attribute values
            if (len(attr_value_list) > 100):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total / 10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x * ten_percentile])
                attr_value_list = new_list

            for val in attr_value_list:
                # calculate the gain if we split on this value
                # if gain is greater than local_max_gain, save this gain and this value
                local_gain = calc_gain(dataset, dataset_entropy, val,
                                       attr_index)  # calculate the gain if we split on this value

                if (local_gain > local_max_gain):
                    local_max_gain = local_gain
                    local_split_val = val

            if (local_max_gain > max_gain):
                max_gain = local_max_gain
                split_val = local_split_val
                attr_to_split = attr_index

    # attr_to_split is now the best attribute according to our gain metric
    if (split_val is None or attr_to_split is None):
        print("Something went wrong. Couldn't find an attribute to split on or a split value.")
    elif (max_gain <= min_gain or node.height > 20):

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node

    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
    node.attr_split_value = split_val
    # currently doing one split per node so only two datasets are created
    upper_dataset = data(classifier)
    lower_dataset = data(classifier)
    upper_dataset.attributes = dataset.attributes
    lower_dataset.attributes = dataset.attributes
    upper_dataset.attr_types = dataset.attr_types
    lower_dataset.attr_types = dataset.attr_types
    for example in dataset.examples:
        if (attr_to_split is not None and example[attr_to_split] >= split_val):
            upper_dataset.examples.append(example)
        elif (attr_to_split is not None):
            lower_dataset.examples.append(example)

    node.upper_child = compute_tree(upper_dataset, node, classifier)
    node.lower_child = compute_tree(lower_dataset, node, classifier)

    return node


##################################################
# Classify dataset
##################################################
def classify_leaf(dataset, classifier):
    ones = one_count(dataset.examples, dataset.attributes, classifier)
    total = len(dataset.examples)
    zeroes = total - ones
    if (ones >= zeroes):
        return 1
    else:
        return 0


##################################################
# Calculate the entropy of the current dataset
##################################################
def calc_dataset_entropy(dataset, classifier):
    ones = one_count(dataset.examples, dataset.attributes, classifier)
    total_examples = len(dataset.examples);

    entropy = 0
    p = ones / total_examples
    if (p != 0):
        entropy += p * math.log(p, 2)
    p = (total_examples - ones) / total_examples
    if (p != 0):
        entropy += p * math.log(p, 2)

    entropy = -entropy
    return entropy


##################################################
# Calculate the gain of a particular attribute split
##################################################
def calc_gain(dataset, entropy, val, attr_index):
    classifier = dataset.attributes[attr_index]
    attr_entropy = 0
    total_examples = len(dataset.examples);
    gain_upper_dataset = data(classifier)
    gain_lower_dataset = data(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attr_types = dataset.attr_types
    gain_lower_dataset.attr_types = dataset.attr_types
    for example in dataset.examples:
        if (example[attr_index] >= val):
            gain_upper_dataset.examples.append(example)
        elif (example[attr_index] < val):
            gain_lower_dataset.examples.append(example)

    if (len(gain_upper_dataset.examples) == 0 or len(
            gain_lower_dataset.examples) == 0):  # Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
        return -1

    attr_entropy += calc_dataset_entropy(gain_upper_dataset, classifier) * len(
        gain_upper_dataset.examples) / total_examples
    attr_entropy += calc_dataset_entropy(gain_lower_dataset, classifier) * len(
        gain_lower_dataset.examples) / total_examples

    return entropy - attr_entropy


##################################################
# count number of examples with classification "1"
##################################################
"""
def one_count(instances, attributes, classifier):
    count = 0
    class_index = None    

    # find index of classifier
    for a in range(len(attributes)):
        if attributes[a] == classifier:
            class_index = a
        else:
            class_index = len(attributes) - 1

    for i in instances:
        if i[class_index] == "1":
            count += 1
    return count
"""


def get_attrnames(index_attr, examples):
    """
    Funcion utilizada para obtener el conjunto de valores posibles
    qiue puede obtener un determinado atributo según el conjunto de
    datos observados con que se esta entrenando el modelo
    :param index_attr: Es el indice del atributo sobre el cual se tomara la columna respectiva
    y se determinaran los posibles valores que pueden obtener las muestras
    :param examples: es el conjunto de muestras observadas
    :return: lista de atributos que se le pueden asignar a la columna atributo_columna
    """

    list_attrnames = []

    for i in range(len(examples)):
        if list_attrnames.count(examples[i][index_attr]) == 0:
            list_attrnames.append(examples[i][index_attr])

    return list_attrnames


def one_count(instances, attributes, classifier):
    # find index of classifier
    class_index = attributes.index(classifier)

    attrs_list = get_attrnames(class_index, instances)

    count = len(attrs_list)

    return count


##################################################
# Prune tree
##################################################
def prune_tree(root, node, dataset, best_score):
    # if node is a leaf
    if (node.is_leaf == True):
        # get its classification
        classification = node.classification
        # run validate_tree on a tree with the nodes parent as a leaf with its classification
        node.parent.is_leaf = True
        node.parent.classification = node.classification
        if (node.height < 20):
            new_score = validate_tree(root, dataset)
        else:
            new_score = 0

        # if its better, change it
        if (new_score >= best_score):
            return new_score
        else:
            node.parent.is_leaf = False
            node.parent.classification = None
            return best_score
    # if its not a leaf
    else:
        # prune tree(node.upper_child)
        new_score = prune_tree(root, node.upper_child, dataset, best_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score
        # prune tree(node.lower_child)
        new_score = prune_tree(root, node.lower_child, dataset, new_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score

        return new_score


##################################################
# Validate tree
##################################################
def validate_tree(node, dataset):
    total = len(dataset.examples)
    correct = 0
    for example in dataset.examples:
        # validate example
        correct += validate_example(node, example)
    return correct / total


##################################################
# Validate example
##################################################
def validate_example(node, example):
    if (node.is_leaf == True):
        projected = node.classification
        actual = example[-1]
        if projected == actual:
            return 1
        else:
            return 0
    value = example[node.attr_split_index]
    if (value >= node.attr_split_value):
        return validate_example(node.upper_child, example)
    else:
        return validate_example(node.lower_child, example)


##################################################
# Test example
##################################################
def test_example(example, node, class_index):
    if (node.is_leaf == True):
        return node.classification
    else:
        if (example[node.attr_split_index] >= node.attr_split_value):
            return test_example(example, node.upper_child, class_index)
        else:
            return test_example(example, node.lower_child, class_index)


##################################################
# Print tree
##################################################
def print_tree(node):
    if (node.is_leaf == True):
        for x in range(node.height):
            print("\t", )
        print("Classification: " + str(node.classification))
        return
    for x in range(node.height):
        print("\t", )
    print("Split index: " + str(node.attr_split))
    for x in range(node.height):
        print("\t", )
    print("Split value: " + str(node.attr_split_value))
    print_tree(node.upper_child)
    print_tree(node.lower_child)


##################################################
# Print tree in disjunctive normal form
##################################################
def print_disjunctive(node, dataset, dnf_string):
    if (node.parent == None):
        dnf_string = "Recorrido\t"
    if (node.is_leaf == True):
        dnf_string = dnf_string[:-3]
        print(dnf_string, )
    else:
        upper = dnf_string + str(dataset.attributes[node.attr_split_index]) \
                + " >= " + str(node.attr_split_value) + " V "
        print_disjunctive(node.upper_child, dataset, upper)

        lower = dnf_string + str(dataset.attributes[node.attr_split_index]) + " < " + str(node.attr_split_value) + " V "
        print_disjunctive(node.lower_child, dataset, lower)
        return


##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

"""
def main():
    args = str(sys.argv)
    args = ast.literal_eval(args)
    if (len(args) < 2):
        print(
            "You have input less than the minimum number of arguments. Go back and read README.txt and do it right next time!")
    elif (args[1][-4:] != ".csv"):
        print("Your training file (second argument) must be a .csv!")
    else:
        datafile = args[1]
        dataset = data("")
        if ("-d" in args):
            datatypes = args[args.index("-d") + 1]
        else:
            datatypes = 'datatypes.csv'
        read_data(dataset, datafile, datatypes)
        arg3 = args[2]
        if (arg3 in dataset.attributes):
            classifier = arg3
        else:
            classifier = dataset.attributes[-1]

        dataset.classifier = classifier

        # find index of classifier
        for a in range(len(dataset.attributes)):
            if dataset.attributes[a] == dataset.classifier:
                dataset.class_index = a
            else:
                dataset.class_index = range(len(dataset.attributes))[-1]

        preprocess2(dataset)

        print("Computing tree...")
        root = compute_tree(dataset, None, classifier)
        if ("-s" in args):
            print_disjunctive(root, dataset, "")
            print("\n")
        if ("-v" in args):
            datavalidate = args[args.index("-v") + 1]
            print("Validating tree...")

            validateset = data(classifier)
            read_data(validateset, datavalidate, datatypes)
            for a in range(len(dataset.attributes)):
                if validateset.attributes[a] == validateset.classifier:
                    validateset.class_index = a
                else:
                    validateset.class_index = range(len(validateset.attributes))[-1]
            preprocess2(validateset)
            best_score = validate_tree(root, validateset)

            print("Initial (pre-pruning) validation set score: " + str(100 * best_score) + "%")
        if ("-p" in args):
            if ("-v" not in args):
                print("Error: You must validate if you want to prune")
            else:
                post_prune_accuracy = 100 * prune_tree(root, root, validateset, best_score)
                print("Post-pruning score on validation set: " + str(post_prune_accuracy) + "%")
        if ("-t" in args):
            datatest = args[args.index("-t") + 1]
            testset = data(classifier)
            read_data(testset, datatest, datatypes)
            for a in range(len(dataset.attributes)):
                if testset.attributes[a] == testset.classifier:
                    testset.class_index = a
                else:
                    testset.class_index = range(len(testset.attributes))[-1]
            print("Testing model on " + str(datatest))
            for example in testset.examples:
                example[testset.class_index] = '0'
            testset.examples[0][testset.class_index] = '1'
            testset.examples[1][testset.class_index] = '1'
            testset.examples[2][testset.class_index] = '?'
            preprocess2(testset)
            b = open('results.csv', 'w')
            a = csv.writer(b)
            for example in testset.examples:
                example[testset.class_index] = test_example(example, root, testset.class_index)
            saveset = testset
            saveset.examples = [saveset.attributes] + saveset.examples
            a.writerows(saveset.examples)
            b.close()
            print("Testing complete. Results outputted to results.csv")
"""

muestra_entrenamiento = [
    ['SAN JOSE', '1', '25', 'SI', 'PAC'],
    ['CARTAGO', '1', '22', 'NO', 'PAC'],
    ['HEREDIA', '1', '50', 'NO', 'PAC'],
    ['HEREDIA', '0', '50', 'SI', 'PAC'],
    ['HEREDIA', '1', '25', 'NO', 'PUSC'],
    ['CARTAGO', '0', '25', 'SI', 'PAC'],
    ['HEREDIA', '0', '22', 'NO', 'PLN'],
    ['PUNTARENAS', '0', '22', 'NO', 'PLN']
]

muestra_validacion = [
    ['SAN JOSE', '1', '25', 'SI', 'PAC'],
    ['CARTAGO', '0', '22', 'NO', 'PAC'],
    ['HEREDIA', '1', '50', 'NO', 'PLN'],
    ['SAN JOSE', '1', '50', 'SI', 'PAC'],
    ['HEREDIA', '1', '50', 'NO', 'PLN'],
    ['PUNTARENAS', '1', '22', 'SI', 'PLN'],
    ['HEREDIA', '0', '25', 'NO', 'PAC'],
    ['HEREDIA', '1', '25', 'NO', 'PAC']
]

header = ['PROVINCIA', 'TRABAJADOR', 'EDAD', 'SOLTERO', 'VOTO']

# create array that indicates whether each attribute is a numerical value or not
attr_types = [False, True, True, False, False]


def main():
    dataset = data("")

    dataset.examples = muestra_entrenamiento
    dataset.attributes = header
    classifier = dataset.attributes[-1]  # GOAL
    dataset.classifier = classifier
    dataset.attr_types = attr_types  # Para saber si el atributo es nummerio o no

    # find index of classifier
    for a in range(len(dataset.attributes)):
        if dataset.attributes[a] == dataset.classifier:
            dataset.class_index = a
        else:
            dataset.class_index = range(len(dataset.attributes))[-1]

    # preprocess2(dataset)

    print("Computing tree...")
    root = compute_tree(dataset, None, classifier)

    # Imprimir el arbol de desicion
    # print_tree(root)

    print("Validating tree...")

    validateset = data(classifier)

    validateset.examples = muestra_validacion

    validateset.attributes = header
    validateset.attr_types = attr_types

    for a in range(len(validateset.attributes)):
        if validateset.attributes[a] == validateset.classifier:
            validateset.class_index = a
        else:
            validateset.class_index = range(len(validateset.attributes))[-1]

    # preprocess2(validateset)
    best_score = validate_tree(root, validateset)

    print("Initial (pre-pruning) validation set score: " + str(100 * best_score) + "%")

    post_prune_accuracy = 100 * prune_tree(root, root, validateset, best_score)
    print("Post-pruning score on validation set: " + str(post_prune_accuracy) + "%")

    """
    if ("-t" in args):
        datatest = args[args.index("-t") + 1]
        testset = data(classifier)
        read_data(testset, datatest, datatypes)
        for a in range(len(dataset.attributes)):
            if testset.attributes[a] == testset.classifier:
                testset.class_index = a
            else:
                testset.class_index = range(len(testset.attributes))[-1]
        print("Testing model on " + str(datatest))
        for example in testset.examples:
            example[testset.class_index] = '0'
        testset.examples[0][testset.class_index] = '1'
        testset.examples[1][testset.class_index] = '1'
        testset.examples[2][testset.class_index] = '?'
        preprocess2(testset)
        b = open('results.csv', 'w')
        a = csv.writer(b)
        for example in testset.examples:
            example[testset.class_index] = test_example(example, root, testset.class_index)
        saveset = testset
        saveset.examples = [saveset.attributes] + saveset.examples
        a.writerows(saveset.examples)
        b.close()
        print("Testing complete. Results outputted to results.csv")

    """


if __name__ == "__main__":
    main()