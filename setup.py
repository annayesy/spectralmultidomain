# setup.py
from setuptools import setup, find_packages

setup(
    name='hps2d',
    version='1.0',
    packages=['hps2d'],
    license="MIT",
    author="Anna Yesypenko",
    author_email='annayesy@utexas.edu',
    url='https://github.com/annayesy/hpsmultidomaindisc.py',
    install_requires=[
        'numpy','matplotlib','scipy',
    ],
)
