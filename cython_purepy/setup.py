#!/usr/bin/env python

"""
setup.py  to build mandelbot code with cython
"""
from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy # to get includes

setup(
    ext_modules = cythonize("my_module.py")
)

#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("my_module", ["A.pxd"])],
#    include_dirs = [numpy.get_include(),],
#)
