from setuptools import setup, find_packages
import os

setup(
    packages=find_packages(),
    use_scm_version={"write_to": os.path.join("ctapipe_io_lst", "_version.py")},
    install_requires=[
        'astropy~=4.1',
        'ctapipe~=0.10.0',
        'protozfits @ https://github.com/cta-observatory/protozfitsreader/archive/v1.5.0.tar.gz',
        'setuptools_scm',
    ],
    package_data={
        'ctapipe_io_lst': ['resources/*'],
    },
    tests_require=['pytest'],
    setup_requires=['pytest_runner'],
)
