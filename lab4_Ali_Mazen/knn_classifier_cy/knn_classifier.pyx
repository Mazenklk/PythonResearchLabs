import numpy as np
import cython
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memset

ctypedef packed struct distanceWithLabel:
    unsigned char label
    float distance

cdef int cmp(const void* elt1, const void* elt2) noexcept nogil:
    cdef float dist1,dist2
    cdef distanceWithLabel* p1 = <distanceWithLabel*> elt1
    cdef distanceWithLabel* p2 = <distanceWithLabel*> elt2
    if(p1.distance<p2.distance):
        return -1
    elif(p1.distance>p2.distance):
        return 1
    else:
        return 0
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef KNN(float [:] x, int k, float[:, ::1] training_features, unsigned char[::1] training_labels):
    cdef int n = training_features.shape[0]
    cdef int i
    cdef distanceWithLabel *distances
    cdef unsigned char[2] classes
    memset(classes, 0, 2)

    distances = <distanceWithLabel*> malloc(n*sizeof(distanceWithLabel))
    if not distances:
        raise MemoryError()
    
    computeEuclidianDistance(n,x,training_features,training_labels,distances)
    qsort(distances, n, sizeof(distanceWithLabel),&cmp)

<<<<<<< HEAD:lab4_Ali_Mazen/knn_classifier_cy/knn_classifier.pyx
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
=======
>>>>>>> 5724960e420be8f4eb13d5154e8083d2b535e3dc:lab4_Ali_Mazen/knn_classifier_cy/knn_classifier.py
    for i in range(k):
        if distances[i].label == 1:
            classes[0] += 1
        else:
<<<<<<< HEAD:lab4_Ali_Mazen/knn_classifier_cy/knn_classifier.pyx
            classes[0]
    if classes[0] > classes[1]:
        return 1
    else: 
        return 2   
    return 0

=======
            classes[1] +=1
    free(distances)
    return (1 if classes[0]>classes[1] else 2)
>>>>>>> 5724960e420be8f4eb13d5154e8083d2b535e3dc:lab4_Ali_Mazen/knn_classifier_cy/knn_classifier.py

cdef void computeEuclidianDistance(int size, float [:]x, float[:,::1] training_features, unsigned char[::1] training_labels, distanceWithLabel *distances):
    cdef float dist
    for i in range(size):
        dist = ((x[0] - training_features[i, 0])**2 + (x[1] - training_features[i, 1])**2)**(1/2)
        distances[i].distance=dist
        distances[i].label=training_labels[i]