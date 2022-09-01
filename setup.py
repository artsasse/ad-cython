from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="adwesley",
    ext_modules = cythonize("cython_functions.pyx", language_level=3),
)