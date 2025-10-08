import numpy as np
import cython

def KNN(x: cython.float[:], k: cython.int, training_features: cython.float[:, :], training_labels: cython.int[:]):
    n: cython.int
    n = training_features.shape[0]
    dtype = [('label', np.int32), ('distance', np.float64)]
    distances = np.zeros(n, dtype=dtype)
    i: cython.Py_ssize_t
    dx: cython.float
    dy: cython.float
    d: cython.float
    for i in range(n):
        dx = x[0] - training_features[i, 0]
        dy = x[1] - training_features[i, 1]
        d = (dx**2 + dy**2) ** 0.5
        distances = np.append(distances, np.array([(training_labels[i], d)], dtype=dtype))
    dtype = [('label', np.int32), ('distance', np.float64)]
    sorted = np.sort(distances, order='distance')            
    classes = np.zeros(2, dtype=np.int32)
    for i in range(k):
        if sorted[i]['label'] == 1:
            classes[0] += 1
        else:
            classes[0]
    if classes[0] > classes[1]:
        return 1
    else: 
        return 2   
    return 0


