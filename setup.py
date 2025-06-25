# setup.py
from setuptools import setup, find_packages

setup(
    name="arctic-atmo-gp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "xarray",
        "netCDF4",
        "h5netcdf",
        "matplotlib",
        "cartopy",    # optional but useful for some HDF5-based files
        # add other deps here...
    ],
)
