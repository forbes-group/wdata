W-data Format
=============

This project contains tools for working with and manipulating the
W-data format used for analyzing superfluid data generated by the [W-SLDA
Toolkit](https://wslda.fizyka.pw.edu.pl/).

This format was originally derived from the W-SLDA project led by Gabriel Wlazlowski as
documented here:

* [W-SLDA Toolkit](https://wslda.fizyka.pw.edu.pl)
* [Original W-data format](https://gitlab.fizyka.pw.edu.pl/gabrielw/wslda/-/wikis/W-data%20format)

Here we augment this format slightly to facilitate working with Python.

Generalizations
---------------

The original format required a `.wtxt` file with lots of relevant
information.  Here we generalize the format to allow this information
to be specified in the data files, which we allow to be in the NPY
format.

Installation
------------

```bash
pip install wdata
```

Basic Usage
-----------

The W-data format stores various arrays representing physical
quantities such as the density (real), pairing field (complex),
currents (3-component real vectors) etc. on a regular lattice of shape
`Nxyz = (Nx, Ny, Nz)` at a bunch of `Nt` times.

The data is represented by two classes: 

* `Var`: These are the data variables such as density, currents,
  etc. with additional metadata (ee the `wdata.io.IVar` interface for
  details):
  
  * `Var.name`: Name of variable as it will appear in VisIt for example.
  * `Var.data`: The actual data as a NumPy array.
  * `Var.description`: Description.
  * `Var.filename`: The file where the data is stored on disk.
  * `Var.unit`: Unit (mainly for use in VisIt... does not affect the data.)

* `WData`: This represents a complete dataset.  Some relevant
  attributes are (see `wdata.io.IWData` for details):
  * `WData.infofile`: Location of the infofile (see below).  This is
    where the metadata will be stored or loaded from.
  * `WData.variables`: List of `Var` variables.
  * `WData.xyz`: Abscissa `(x, y, z)` shaped so that they can be used
    with broadcasting.  I.e. `r = np.sqrt(x**2+y**2+z**2)`.
  * `WData.t`: Array of times.
  * `WData.dim`: Dimension of dataset.  I.e. `dim==1` for 1D simulations,
    `dim==3` for 3D simulations.
  * `WData.aliases`: Dictionary of aliases.  Convenience for providing
    alternative data access in VisIt.
  * `WData.constants`: Dictionary of constants such as `kF`, `eF`.

**Minimal Example**:

Here is a minimal set of data:

```python
import numpy as np
np.random.seed(3)
from wdata.io import WData, Var

Nt = 10 
Nxyz = (4, 8, 16)
dxyz = (0.3, 0.2, 0.1)
dt = 0.1
Ntxyz = (Nt,) + Nxyz

density = np.random.random(Ntxyz)

data = WData(prefix='dataset', data_dir='_example_wdata',
             Nxyz=Nxyz, dxyz=dxyz,
             variables=[Var(density=density)],
             Nt=Nt)
data.save(force=True)
```

This will make a directory `_example_wdata` with infofile
`_example_wdata/dataset.wtxt`:

```bash
$ tree _example_wdata
_example_wdata
|-- dataset.wtxt
`-- dataset_density.wdat

0 directories, 2 files
$ cat _example_wdata/dataset.wtxt
# Generated by wdata.io: [2020-12-18 06:41:29 UTC+0000 = 2020-12-17 22:41:29 PST-0800]

NX               4    # Lattice size in X direction
NY               8    #             ... Y ...
NZ              16    #             ... Z ...
DX             0.3    # Spacing in X direction
DY             0.2    #        ... Y ...
DZ             0.1    #        ... Z ...
prefix     dataset    # datafile prefix: <prefix>_<var>.<format>
datadim          3    # Block size: 1:NX, 2:NX*NY, 3:NX*NY*NZ
cycles          10    # Number Nt of frames/cycles per dataset
t0               0    # Time value of first frame
dt               1    # Time interval between frames

# variables
# tag       name    type    unit    format    # description
var      density    real    none      wdat    # density
```

The data can be loaded by specifying the infofile:

```python
from wdata.io import WData
data = WData.load('_example_wdata/dataset.wtxt')
```

The data could be plotted using [PyVista](https://docs.pyvista.org)
for example (the random data will not look so good...):

```python
import numpy as np
import pyvista as pv
from wdata.io import WData

data = WData.load('_example_wdata/dataset.wtxt')
n = data.density[0]

grid = pv.StructuredGrid(*np.meshgrid(*data.xyz))
grid["vol"] = n.flatten(order="F")
contours = grid.contour(np.linspace(n.min(), n.max(), 5))

p = pv.Plotter()
p.add_mesh(contours, scalars=contours.points[:, 2])
p.show()
```

The recommended way to save data is to create variables for the data,
times, and abscissa, then store this:

```bash
import numpy as np
from wdata.io import WData, Var

np.random.seed(3)

Nt = 10
Nxyz = (32, 32, 32)
dxyz = (10.0/32, 10.0/32, 10.0/32)
dt = 0.1

# Abscissa.  Not strictly needed, but if you have them, then use them
# instead.
t = np.arange(Nt)*dt
xyz = np.meshgrid(*[(np.arange(_N)-_N/2)*_dx
                    for _N, _dx in zip(Nxyz, dxyz)],
                  sparse=True, indexing='ij')

# Now make the WData object and save the data.
Ntxyz = (Nt,) + Nxyz
w = np.pi/t.max()
ws = [1.0 + 0.5*np.cos(w*t), 
      1.0 + 0.5*np.sin(w*t),
      1.0 + 0*t]
density = np.exp(-sum((_x[None,...].T*_w).T**2/2 for _x, _w in zip(xyz, ws)))
delta = np.random.random(Ntxyz) + np.random.random(Ntxyz)*1j - 0.5 - 0.5j
current = np.random.random((Nt, 3,) + Nxyz) - 0.5

variables = [
    Var(density=density),
    Var(delta=delta),
    Var(current=current)
]
    
data = WData(prefix='dataset2', 
             data_dir='_example_wdata/',
             xyz=xyz, t=t,
             variables=variables)
data.save()
```

Now load and plot the data:

```bash
import numpy as np
import pyvista as pv

from wdata.io import WData
data = WData.load(infofile='_example_wdata/dataset2.wtxt')

n = data.density[0]

grid = pv.StructuredGrid(*np.meshgrid(*data.xyz))
grid["vol"] = n.flatten(order="F")
contours = grid.contour(np.linspace(n.min(), n.max(), 5))

p = pv.Plotter()
p.add_mesh(contours, scalars=contours.points[:, 2])
p.show()
```

Note: the actual data is loaded into python using [memory-mapped
arrays](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).  This
allows you to refer to very large data-sets without loading the entire data into
memory.  This will delay loading until a copy of the array is made.  For example:

```bash
import numpy as np
from wdata.io import WData
data = WData.load(infofile='_example_wdata/dataset2.wtxt')

# At this point, the data has not been fully loaded.  You can
# work with subsets efficiently.  For example, the following will
# only load the first frame of data:

n = data.density[0]

# Beware: if you make a copy of the data, explicitly *or implicitly* then it will get
# loaded.  The following will load the full array into memory so that np.cos can do its
# computations.

sum_cos_n = np.sum(np.cos(data.density))

# If this is too big, you may want to process each slice independently.  The previous
# example could be more efficiently computed using the following loop:

sum_cos_n = sum(np.cos(_n).sum() for _n in data.density)

# The Dask package may be useful for such processing in more complicated settings.
```

See Also
--------
* [NumPy memory-mapped files](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
* [Dask](https://dask.org)

Developer Notes
===============

Testing
-------

For distribution we use [poetry](https://python-poetry.org) and for testing we use
[nox](https://nox.thea.codes).  To test the code:

```bash
nox
```

Documentation
-------------

For documentation, we use [Sphinx](https://www.sphinx-doc.org).  To build this run:

```bash
poetry install  # Install all of the developer dependencies
poetry run make -C docs html
```

* `__init__()`: The default behavior of [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content)
is to merge the documentation of `__init__` methods with the class since the user never
directly calls `__init__()`.  Keep this in mind when writing the docstrings.

Changes
=======

## 0.1.3
* Address issue #4 for loading large datasets.  We now use memory mapped files.
* Started adding Sphinx documentation.  Not complete (`sphinxcontrib.zopeext` needs
  updating... something is wrong.)

## 0.1.2
* Fixed issue #2.  `datadim < 3` now works properly.
* Started working on documentation (incomplete).
