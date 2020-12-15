"""Activation and output functions for Neural Network
"""

import numpy as np

    
def sigmoid(z, alpha=None, derivative=False):
    if not derivative:
        return 1/(1+np.exp(-z))
    elif derivative:
        return sigmoid(z, alpha)*(1 - sigmoid(z, alpha))

def ELU(z, alpha=None, derivative=False):
    if alpha is None: alpha = 0.1
    if not derivative:
        if z <  0: return alpha*(np.exp(z) - 1)
        if z >= 0: return z
    elif derivative:
        raise NotImplementedError

def ReLU(z, alpha=None, derivative=False):
    if alpha is None: alpha = 0.1
    if not derivative:
        return (z>0)*alpha*z
    elif derivative:
        return (z>=0)*alpha

def ReLU_leaky(z, alpha=None, derivative=False):
    if alpha is None: alpha = 0.1
    if not derivative:
        return (z>0)*alpha*z + (z<0)*0.01*alpha*z
    elif derivative:
        return (z>0)*alpha + (z<0)*0.01*alpha

def linearActivation(z, alpha=None, derivative=False):
    if alpha is None: alpha = 0.1
    if not derivative:
        return alpha*z
    elif derivative:
        return alpha

def softMax(z, alpha=None, derivative=False):
    if not derivative:
        e = np.exp(z)
        return e/np.sum(e,axis=1,keepdims=True)
    elif derivative:
        raise NotImplementedError
