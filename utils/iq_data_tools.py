import numpy as np
import matplotlib.pyplot as plt

def convertIqArray2complex(IQ_ARRAY):
    # IQ_ARRAY_Out = np.zeros(size_IQ_Data[0,1])
    result = 1j*IQ_ARRAY[:,:,1]; result += IQ_ARRAY[:,:,0]
    return result
    
def computeCovariance(Y):
    if Y.ndim > 2:
        Y = convertIqArray2complex(Y)
    shapes = Y.shape
    num_snapshots = shapes[1]
    R = (Y @ Y) / num_snapshots
    return R
