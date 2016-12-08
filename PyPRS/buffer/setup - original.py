from distutils.core import setup, Extension

module1 = Extension('cutils',
                    sources = ['cutils.cpp'])

setup (name = 'PyNP',
       version = '1.0',
       description = 'Python Implementation of Nested Partition Algorithm',
       ext_modules = [module1])