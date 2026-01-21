GauXC standalone usage
======================

The GauXC package comes with a standalone driver for testing the evaluation of the exchange-correlation energy with different functionals.
In this tutorial we will use the standalone driver to evaluate Skala based on density matrices computed with different packages.

.. tip::

   For building GauXC and running the standalone driver checkout :ref:`gauxc_install`.

Create GauXC compatible input
-----------------------------

We will use the ``skala`` package to write a GauXC compatible input for our calculation.
For this we will run a PySCF calculation and write the molecule, basis set and density matrix in the format expected by GauXC.
In this example we will use a single one atom system in a small basis set.

.. note::

   We will write the input data as HDF5 file since GauXC can read its objects directly from HDF5 datasets.
   The format in the HDF5 file does correspond to the internal structure of GauXC objects and therefore allows us to conveniently inspect the data.

.. literalinclude:: scripts/export-h5.py
   :language: python

Additionally to the inputs (molecule, basis set, and density matrix) we provide the exchange-correlation energy and potential to allow the standalone driver to compare against our reference calculation.

Running the GauXC standalone driver
-----------------------------------

The GauXC standalone driver takes a single input file, where we need to specify the path of our HDF5 file with the input data.
In the input file we specify the ``ONEDFT_MODEL`` as PBE since we used it for our input calculation as well.
Furthermore, we have parameters like ``grid``, ``pruning_scheme``, etc. which define the integration grid settings in GauXC, here we go with a fine grid, Mura-Knowles radial integration scheme and the robust pruning scheme of Psi4. 

.. code-block:: ini
   :caption: gauxc_input.inp

   [GAUXC]
   ref_file = He_def2-svp.h5
   ONEDFT_MODEL = PBE
   grid = Fine
   pruning_scheme = Robust
   RAD_QUAD = MuraKnowles
   batch_size = 512
   basis_tol = 2.22e-16
   LB_EXEC_SPACE =  Device
   INT_EXEC_SPACE = Device
   REDUCTION_KERNEL = Default
   MEMORY_SIZE = 0.1

.. tip::

   Make sure the HDF5 file ``He_def2-svp.h5`` is in the same directory as the one where we start the standalone driver.

To run the standalone driver with this input we run it from the build directory with our input file:

.. code-block:: text

   ./build/tests/standalone_driver gauxc_input.inp

For a successful run we will see the following output

.. code-block:: text

   DRIVER SETTINGS: 
     REF_FILE          = He_def2-svp.h5
     GRID              = FINE
     RAD_QUAD          = MURAKNOWLES
     PRUNING_SCHEME    = ROBUST
     BATCH_SIZE        = 512
     BASIS_TOL         = 2.22e-16
     FUNCTIONAL        = PBE0
     LB_EXEC_SPACE     = DEVICE
     INT_EXEC_SPACE    = DEVICE
     INTEGRATOR_KERNEL = DEFAULT
     LWD_KERNEL        = DEFAULT
     REDUCTION_KERNEL  = DEFAULT
     DEN (?)           = false
     VXC (?)           = true
     EXX (?)           = false
     EXC_GRAD (?)      = false
     DD_PSI (?)        = false
     DD_PSI_POTENTIAL (?)       = false
     ONEDFT_MODEL    = PBE
     FXC_CONTRACTION (?)       = false
     MEMORY_SIZE       = 0.1

   EXC: -1.054031868349e+00
   EXC = -1.054031868349e+00

   Load Balancer Timings
           LoadBalancer.CreateTasks:  1.50510e+01 ms
   MolecularWeights Timings
                   MolecularWeights:  2.98569e+01 ms
   Integrator Timings
                       XCIntegrator.Allreduce:  4.11500e-03 ms
                       XCIntegrator.LocalWork:  2.35691e+01 ms
                      XCIntegrator.LocalWork2:  9.11679e+00 ms
   XC Int Duration  = 3.35111170000000e-01 s
   EXC (ref)        = -1.05403142675144e+00
   EXC (calc)       = -1.05403186834886e+00
   EXC Diff         = -4.18960391377858e-07
   | VXC (ref)  |_F = 1.45598265614311e+00
   | VXC (calc) |_F = 1.45598296606474e+00
   RMS VXC Diff     = 7.43706533247358e-08
   | VXCz (ref)  |_F = 0.00000000000000e+00
   | VXCz (calc) |_F = 0.00000000000000e+00
   RMS VXCz Diff     = 0.00000000000000e+00

We find a reasonable difference between PySCF and GauXC computed exchange-correlation energy and potential.

.. tip::

   We can converge this value further by choosing finer grid settings both in PySCF and GauXC.

Inspecting the GauXC input data
-------------------------------

Now that we verified that GauXC can evaluate based on our PySCF produced input data, we will have a closer look of what we sent to GauXC.
For this we will inspect our HDF5 input data more closely.

.. code-block:: ipython

   In [1]: import h5py
      ...: import numpy as np

   In [2]: with h5py.File("He_def2-svp.h5") as h5:
      ...:     molecule = np.asarray(h5["MOLECULE"])
      ...:     basis = np.asarray(h5["BASIS"])
      ...:     dm_scalar = np.asarray(h5["DENSITY_SCALAR"])
      ...:     dm_z = np.asarray(h5["DENSITY_Z"])
      ...:

First, we inspect the molecule format which follows an array of structs format, combining the atomic number together with the cartesian coordinates in Bohr.
For our Helium example we expect a single entry centered at the origin:

.. code-block:: ipython

   In [3]: molecule.shape
   Out[3]: (1,)

   In [4]: molecule.dtype
   Out[4]: dtype({'names': ['Atomic Number', 'X Coordinate', 'Y Coordinate', 'Z Coordinate'], 'formats': ['<i4', '<f8', '<f8', '<f8'], 'offsets': [0, 8, 16, 24], 'itemsize': 32})

   In [5]: molecule[0]
   Out[5]: np.void((2, 0.0, 0.0, 0.0), dtype={'names': ['Atomic Number', 'X Coordinate', 'Y Coordinate', 'Z Coordinate'], 'formats': ['<i4', '<f8', '<f8', '<f8'], 'offsets': [0, 8, 16, 24], 'itemsize': 32})

The def2-SVP basis set for Helium has three functions (2s1p).
Similar to the molecule format the basis set is represented as an array of structs and combines the information of the number of primitives, the angular momentum, whether the shell is pure (spherical) or cartesian with the Gaussian exponents, contraction coefficients and cartesian coordinates of the origin in Bohr.
Note that the length of exponents in each shell is padded to 16 elements for the exponents and contraction coefficients.

.. code-block:: ipython

   In [6]: basis.shape
   Out[6]: (3,)

   In [7]: basis.dtype
   Out[7]: dtype({'names': ['NPRIM', 'L', 'PURE', 'ALPHA', 'COEFF', 'ORIGIN'], 'formats': ['<i4', '<i4', '<i4', ('<f8', (16,)), ('<f8', (16,)), ('<f8', (3,))], 'offsets': [0, 4, 8, 16, 272, 528], 'itemsize': 552})

   In [8]: basis[0]
   Out[8]: np.void((3, 0, 1, [38.354936737, 5.7689081479, 1.2399407035, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.44139562618839057, 0.6934601558999577, 0.6641335374571593, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), dtype={'names': ['NPRIM', 'L', 'PURE', 'ALPHA', 'COEFF', 'ORIGIN'], 'formats': ['<i4', '<i4', '<i4', ('<f8', (16,)), ('<f8', (16,)), ('<f8', (3,))], 'offsets': [0, 4, 8, 16, 272, 528], 'itemsize': 552})

   In [9]: basis[2]
   Out[9]: np.void((1, 1, 0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.4254109407099804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), dtype={'names': ['NPRIM', 'L', 'PURE', 'ALPHA', 'COEFF', 'ORIGIN'], 'formats': ['<i4', '<i4', '<i4', ('<f8', (16,)), ('<f8', (16,)), ('<f8', (3,))], 'offsets': [0, 4, 8, 16, 272, 528], 'itemsize': 552})

.. important::

   The orbital ordering convention for the shells in GauXC is following the common component architecture (CCA) convention for pure (spherical) shells and the row convention for cartesian ones.
   The CCA ordering for pure (spherical) shells is defined as

   - ``s`` (:math:`\ell = 0`): :math:`Y_0^0`
   - ``p`` (:math:`\ell = 1`): :math:`Y_1^{-1}`, :math:`Y_1^{0}`, :math:`Y_1^{+1}`,
   - ``d`` (:math:`\ell = 2`): :math:`Y_2^{-2}`, :math:`Y_2^{-1}`, :math:`Y_2^{0}`, :math:`Y_2^{+1}`, :math:`Y_2^{+2}`

   The row ordering for cartesian shells is defined as

   - ``s`` (:math:`\ell = 0`): `1`
   - ``p`` (:math:`\ell = 1`): ``x``, ``y``, ``z``
   - ``d`` (:math:`\ell = 2`): ``xx``, ``xy``, ``xz``, ``yy``, ``yz``, ``zz``
   - ``f`` (:math:`\ell = 3`): ``xxx``, ``xxy``, ``xxz``, ``xyy``, ``xyz``, ``xzz``, ``yyy``, ``yyz``, ``yzz``, ``zzz``

   PySCF is using CCA ordering for all pure (spherical) shells except for the p-shell where row ordering is used.
   Our export accounts for this by exporting p-shells with setting the ``pure`` entry to zero to have GauXC use row ordering.

Finally, we inspect the density matrix in our input data.
Notably, the two spin channels are stored as scalar and z component.
The scalar component contains the sum of the alpha (up) and the beta (down) spin channel, i.e. the total, while the z component contains their difference, i.e. the polarization.
Similarly, GauXC will compute the exchange-correlation potential as scalar and z component.

We can convert our from scalar and z component to alpha (up) and beta (down) channel by

.. code-block:: ipython

   In [10]: dm_a = 0.5 * (dm_scalar + dm_z)
       ...: dm_b = 0.5 * (dm_scalar - dm_z)

Since we used a restricted density matrix as input here the z component will be zero and the scalar component just the double of the individual spin channels.

Summary
-------

In summary we recommend to use the standalone driver to explore using GauXC for verifying that you formatted your inputs correctly.
Use the provided traditional functional implementations (LDA exchange, PBE and TPSS) to verify the correctness of the computed exchange-correlation energies and potentials between your own package and GauXC.

Troubleshooting
---------------

HDF5 file not found
  ensure that the HDF5 file is in the same directory where you are starting the GauXC standalone driver

GauXC standalone driver not found
  the GauXC standalone driver is a testing tool and not installed with GauXC, check the ``tests`` subdirectory in the directory where you built GauXC.
  If the standalone driver is missing make sure that you enabled testing in CMake or rebuild the GauXC project to generate the standalone driver.
