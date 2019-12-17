from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='ctapipe_io_lst',
    packages=find_packages(),
    version='0.4.0',
    description='ctapipe plugin for reading LST prototype files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'astropy',
        'ctapipe',
        'protozfits'
    ],
    package_data={
        'ctapipe_io_lst': ['resources/*'],
    },
    tests_require=['pytest'],
    setup_requires=['pytest_runner'],
    author='Franca Cassol',
    author_email='cassol@cppm.in2p3.fr',
    license='MIT',
)
