# setup.py
from setuptools import setup, find_packages

setup(
    name='hps',
    version='1.0',
    packages=['hps'],
    license="MIT",
    author="Anna Yesypenko",
    author_email='annayesy@utexas.edu',
    url='https://github.com/annayesy/spectralmultidomain.py',
    install_requires=[
        'numpy','matplotlib','scipy',
    ],
)
