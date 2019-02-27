from setuptools import setup, find_packages

setup(
    name='ctapipe_io_lst',
    packages=find_packages(),
    version='0.1',
    description='ctapipe plugin for reading LST prototype files',
    install_requires=[
        'astropy',
        'ctapipe',
        'protozfits',
    ],
    tests_require=['pytest'],
    author='Franca Cassol',
    author_email='cassol@cppm.in2p3.fr',
    license='MIT',
)
