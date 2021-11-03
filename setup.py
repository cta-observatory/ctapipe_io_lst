from setuptools import setup, find_packages
import os

setup(
    packages=find_packages(),
    use_scm_version={"write_to": os.path.join("ctapipe_io_lst", "_version.py")},
    install_requires=[
        'astropy~=4.2',
        'ctapipe>=0.10.5,<0.12',
        'protozfits~=2.0',
        'setuptools_scm',
        'traitlets~=5.0.5', # remove when upgrading to ctapipe > 0.11
    ],
    package_data={
        'ctapipe_io_lst': ['resources/*'],
    },
    tests_require=['pytest'],
    setup_requires=['pytest_runner'],
)
