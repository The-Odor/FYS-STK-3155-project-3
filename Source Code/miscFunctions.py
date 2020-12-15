"""Miscellaneous functions used in project
"""

import numpy as np

def to_categorical_numpy(integer_vector):
    """Transforms vectors into one-hot vectors, where each value is transformed
    into an array with all elements of 0 except for the element of the index of
    the value of the original element, which is 1:

    example: [1, 2, 3, 4] -> [[0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]

    Args:
        integer_vector (numpy.ndarray): Onput vector

    Returns:
        numpy.ndarray: One-hot form of input vector
    """
    # I stole this because I love the beauty of line 5
    integer_vector = integer_vector.reshape(-1)
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

def doNothing():
    """Does nothing, used for debugging tool to make breaklines
    """
    pass

def accuracyScore(y_test, y_predict):
    """Calculates what fraction of y_predict predicts y_test correctly using
    one-hot form vectors and argmax functions

    Args:
        y_test (numpy.ndarray): One-hot array form of expected values
        y_predict (numpy.ndarray): One-hot array form of predicted values

    Returns:
        float: Fraction of correct predictions
    """
    return np.sum(np.argmax(y_test,axis=1)==np.argmax(y_predict,axis=1)) / len(y_predict)

def MSE(y_data, y_model):
    """Calculates mean squared error between y_data and y_model
    Optimal value is 0, higher values is worse

    Args:
        y_data (numpy.ndarray): Expected values
        y_model (numpy.ndarray): Predicted values

    Returns:
        float: Mean Squared Error
    """
    return np.sum((y_data-y_model)**2) / np.size(y_model)

def R2(y_data, y_model):
    """Calculates R2 value between y_data and y_model
    Optimal value is 1, 0 means an average predicts equally well as y_model,
    lower value means it performs worse

    Args:
        y_data (numpy.ndarray): Expected values
        y_model (numpy.ndarray): Predicted values

    Returns:
        float: R2 value
    """
    # Optimal value is 1, with 0 implying that model performs 
    # exactly as well as predicting using the data average would.
    # Lower values imply that predicting using the average would be better
    top = np.sum((y_data - y_model)**2)
    bot = np.sum((y_data - np.mean(y_data))**2)
    return 1 - top/bot

def defaultingFunc(selfValue, 
                    inValue, 
                    defaultValue, 
                    valueType=None, 
                    variableName=None):
    """Defaults value between varying states and returns one of the input values.
    Priorities for return:
      1: inValue
      2: selfValue
      3: defaultValue
      4: will ask for input from user

    Args:
        selfValue (Any type): Secondary priority, typically set as already defined value in class
        inValue (Any type): Given value, is prioritized as return
        defaultValue (Any type): Value return defaults to if both selfValue and inValue are none
        valueType (Any type, optional): Type of value asked for if selfValue, inValue, and defaultValue are None. Defaults to None.
        variableName (Any type, optional): Name of variable asked for if selfValue, inValue, and defaultValue are None. Defaults to None.

    Returns:
        Input type: Input value as given by priority list
    """

    if selfValue is None and inValue is None:
        if defaultValue is None:
            return valueType(input("{}: ".format(variableName)))
        else:
            return defaultValue
    elif inValue is None:
        return selfValue
    else:
        return inValue

def etaDefinerDefiner(t0, t1=None):
    """Defines a function with no arguments that will return a generator.
    Returned function will be of the right format for the Neural Network class.

    Args:
        t0 (float): If t1 is not given, is given as a constant
        t1 (float, optional): If given, returns output as t0/(t+t1), where t is an integer that increments each iteration. Defaults to None.

    Returns:
        float/function: Valid input for eta in the Neural Network class
    """
    if t1 is None:
        return t0
    elif t0 is None:
        return t1
    else:
        def etaDefiner():
            t = 0
            while True:
                yield t0/(t+t1)
                t += 1
        return etaDefiner