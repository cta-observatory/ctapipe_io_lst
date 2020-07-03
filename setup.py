from setuptools import setup, find_packages
from os import path
from version import get_version, update_release_version

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


update_release_version()
version = get_version()

setup(
    name='ctapipe_io_lst',
    packages=find_packages(),
    version=version,
    description='ctapipe plugin for reading LST prototype files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'astropy',
        'ctapipe~=0.8.0',
        'protozfits @ https://github.com/cta-sst-1m/protozfitsreader/archive/v1.4.2.tar.gz',

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
