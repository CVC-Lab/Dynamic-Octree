from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Octree',
    ext_modules=cythonize(['octree.pyx', 'objects.pyx']),
    script_args=['build_ext', '--build-lib=build'],
)