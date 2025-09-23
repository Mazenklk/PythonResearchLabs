# https://stackoverflow.com/questions/71983019/why-does-adding-an-init-py-change-cython-build-ext-inplace-behavior
# https://github.com/jmschrei/pomegranate/issues/382

import os

from Cython.Build import cythonize
from setuptools import Extension, setup

os.environ["CC"] = "gcc"

extensions = [
    Extension(
        "helloworld",
        ["helloworld.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
