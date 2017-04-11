from distutils.core import setup

from setuptools import find_packages

setup(
    name='chiron',
    version='',
    packages=find_packages('.'),
    package_dir={'': '.'},
    url='',
    license='',
    author='Mateusz Bednarski',
    author_email='',
    description='', install_requires=['numpy', 'gym']
)
