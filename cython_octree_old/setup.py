from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        'octree',
        sources=['octree.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'objects',
        sources=['objects.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='Octree',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'})
)
