#!/usr/bin/env python
import os
import setuptools
from numpy.distutils.core import Extension, setup
from distutils.util import convert_path
from pathlib import Path
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

main_ns = {}
ver_path = convert_path(os.path.join('src', 'glowpython', 'version.py'))
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

# GLOW fortran extension
glow = Extension(
    name="glowpython.fortran",
    sources=[os.path.join('src', 'GLOW', file) for file in [
        "alt_grid.f90",
        "cglow.f90",
        "glow.f90",
        "bands.f90",
        "conduct.f90",
        "egrid.f90",
        "ephoto_init.f90",
        "ephoto.f90",
        "etrans.f90",
        "exsect.f",
        "fieldm.f",
        "gchem.f90",
        "geomag.f90",
        "maxt.f90",
        "mzgrid.f90",
        "qback.f90",
        "rcolum.f90",
        "snoem.f90",
        "snoemint.f90",
        "solzen.f90",
        "ssflux_init.f90",
        "ssflux.f90",
        "iri90.f",
        "nrlmsise00.f"
    ]] + [os.path.join('src', 'tweaks', file) for file in [
        "cglow_release.f90"
    ]],
    extra_f77_compile_args=["-std=legacy"]
)


def listfiles(path):
    allfiles = [os.path.join(path, file) for file in os.listdir(path)]
    files = [file for file in allfiles if os.path.isfile(file)]
    return files


setup(
    name='glowpython',
    version=main_ns['__version__'],
    ext_modules=[glow],
    packages=['glowpython'],
    package_dir={'glowpython': os.path.join('src', 'glowpython')},
    scripts=[os.path.join('Examples', 'Maxwellian.py'), os.path.join('Examples', 'NoPrecipitation.py')],
    data_files=[
        (os.path.join('glowpython', 'data'), listfiles(os.path.join('src', 'GLOW', 'data'))),
        (os.path.join('glowpython', 'data', 'iri90'), listfiles(os.path.join('src', 'GLOW', 'data', 'iri90'))),
    ],

    author='Sunip K. Mukherjee,George Geddes, Ph.D.',
    author_email = 'sunipkmukherjee@gmail.com',
    description='Python bindings for NCAR GLOW model',
    url='https://github.com/sunipkm/glowpython',
    license_files = ('LICENSE', ),
    long_description = (Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    test_suite='tests',
    install_requires=['numpy >= 1.10', 'xarray', 'geomagdata']
)
