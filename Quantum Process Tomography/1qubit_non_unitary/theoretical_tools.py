import numpy as np

def exp_value_braket(statevector,operator):
    '''
    Function that returns mathematically calculated expectation value for given statevector and operator.
    '''
    return np.dot(  np.conjugate(statevector)   , np.dot ( operator   , statevector) )