from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Python package for analyzing and \
              filtering wearable device data missingness"

setup(
    name="FOMO",
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
)