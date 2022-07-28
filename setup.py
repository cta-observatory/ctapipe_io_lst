from setuptools import setup, find_packages
import os

setup(
    packages=find_packages(exclude=["ctapipe_io_lst._dev_version"]),
    use_scm_version={"write_to": os.path.join("ctapipe_io_lst", "_version.py")},
    install_requires=[
        'astropy',
        'ctapipe>=0.12',
        'protozfits~=2.0',
        'setuptools_scm',
    ],
    package_data={
        'ctapipe_io_lst': ['resources/*'],
    },
    tests_require=['pytest'],
)
