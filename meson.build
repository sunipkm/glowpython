# SPDX-FileCopyrightText: 2023 Sunip K. Mukherjee
#
# SPDX-License-Identifier: Apache-2.0

project('glowpython', 'c',
  version : '1.0',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

fortran_srcs = [
  'src/GLOW/alt_grid.f90', 
  'src/GLOW/cglow.f90', 
  'src/GLOW/glow.f90', 
  'src/GLOW/bands.f90', 
  'src/GLOW/conduct.f90', 
  'src/GLOW/egrid.f90', 
  'src/GLOW/ephoto_init.f90', 
  'src/GLOW/ephoto.f90', 
  'src/GLOW/etrans.f90', 
  'src/GLOW/exsect.f', 
  'src/GLOW/fieldm.f', 
  'src/GLOW/gchem.f90', 
  'src/GLOW/geomag.f90', 
  'src/GLOW/maxt.f90', 
  'src/GLOW/mzgrid.f90', 
  'src/GLOW/qback.f90', 
  'src/GLOW/rcolum.f90', 
  'src/GLOW/snoem_init.f90', 
  'src/GLOW/snoem.f90', 
  'src/GLOW/snoemint.f90', 
  'src/GLOW/solzen.f90', 
  'src/GLOW/ssflux_init.f90', 
  'src/GLOW/ssflux.f90', 
  'src/GLOW/iri90.f', 
  'src/GLOW/nrlmsise00.f', 
  'src/tweaks/cglow_release.f90'
]

glowpython_source = custom_target('glowfortmodule.c',
  input : fortran_srcs,
  output : ['glowfortmodule.c', 'glowfort-f2pywrappers.f', 'glowfort-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'glowfort', '--lower']
)

py.install_sources(
    'src/glowpython/__init__.py',
    'src/glowpython/base.py',
    'src/glowpython/utils.py',
    'src/glowpython/version.py',
    'src/glowpython/plots.py',
    'src/glowpython/examples.py',
    subdir: 'glowpython',
    pure: false,
)

install_subdir(
    'src/GLOW/data',
    install_dir: join_paths(py.get_install_dir(), 'glowpython'),
    strip_directory: false,
)

inc_np = include_directories(incdir_numpy, incdir_f2py)

py.extension_module('glowfort',
  fortran_srcs + [glowpython_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true,
  subdir: 'glowpython'
)