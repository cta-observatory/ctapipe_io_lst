from setuptools import setup, find_packages
import os

setup(
    packages=find_packages(exclude=["ctapipe_io_lst._dev_version"]),
    use_scm_version={"write_to": os.path.join("ctapipe_io_lst", "_version.py")},
    python_requires=">=3.8",
    install_requires=[
        'astropy~=5.0',
        'ctapipe~=0.17.0',
        'protozfits~=2.0',
        'setuptools_scm',
        'numpy>=1.20'
    ],
    package_data={
        'ctapipe_io_lst': ['resources/*'],
    },
    tests_require=['pytest'],
)
