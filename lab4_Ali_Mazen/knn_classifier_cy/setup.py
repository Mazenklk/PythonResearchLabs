from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension("knn_classifier", ["knn_classifier.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="knn_classifier",
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
