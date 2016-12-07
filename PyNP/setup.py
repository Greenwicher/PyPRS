from distutils.core import setup, Extension
import numpy

setup(ext_modules=[Extension("_cutils",
      sources=["cutils.i"],
      swig_opts=['-c++','-py3'],
      include_dirs=[numpy.get_include()])])