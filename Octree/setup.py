from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension('octree', ['octree.pyx']),
    Extension('objects', ['objects.pyx'])
]

setup(
    name='Octree',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'})
)