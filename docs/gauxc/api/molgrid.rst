.. _gauxc_molecular_grid_settings:

Molecular grid settings
=======================

This section provides a reference for the settings related to molecular grids in GauXC, including C++ class definitions, C bindings, and Fortran bindings for creating and managing molecular grids used in numerical integration schemes.

C++ definitions
---------------

This section provides C++ class definitions provided by the ``gauxc/molgrid.hpp`` header.

.. cpp:class:: GauXC::MolGrid

   Molecular integration grid container.

   MolGrid aggregates atom-centered quadrature grids for each atomic number and exposes access to per-element grids and batch sizing information for numerical integration.

   .. cpp:function:: inline MolGrid GauXC::MolGridFactory::create_default_molgrid(const Molecule& mol, PruningScheme scheme, BatchSize bsz, RadialQuad radial_quad, AtomicGridSizeDefault grid_size)

      Create a default MolGrid for the given molecule using the specified pruning scheme, batch size, radial quadrature, and atomic grid size.
      This constructor is available from the ``gauxc/molgrid/defaults.hpp`` header.

      :param mol: The molecule for which to create the MolGrid.
      :param scheme: The pruning scheme to use for constructing the molecular grid weights from the atomic grids.
      :param bsz: The batch size for processing grid points in parallel.
      :param radial_quad: The radial quadrature scheme to use for the atomic grids.
      :param grid_size: The default atomic grid size to use for the atomic grids.
      :returns: A MolGrid object initialized with the specified parameters for the given molecule.

   .. cpp:function:: inline MolGrid GauXC::MolGridFactory::create_default_molgrid(const Molecule& mol, PruningScheme scheme, BatchSize bsz, RadialQuad radial_quad, RadialSize rad_size, AngularSize ang_size)

      Create a default MolGrid for the given molecule using the specified pruning scheme, batch size, radial quadrature, and explicit radial and angular sizes.
      This constructor is available from the ``gauxc/molgrid/defaults.hpp`` header.

      :param mol: The molecule for which to create the MolGrid.
      :param scheme: The pruning scheme to use for constructing the molecular grid weights from the atomic grids.
      :param bsz: The batch size for processing grid points in parallel.
      :param radial_quad: The radial quadrature scheme to use for the atomic grids.
      :param rad_size: The radial size to use for the atomic grids.
      :param ang_size: The angular size to use for the atomic grids.
      :returns: A MolGrid object initialized with the specified parameters for the given molecule.


The enumerator values are available from the ``gauxc/enums.hpp`` header.

.. cpp:enum-class:: GauXC::AtomicGridSizeDefault

   Enumeration of default atomic grid sizes for molecular integration.

   The following options are available:

   .. cpp:enumerator:: FineGrid

      A default atomic grid size with 75 angular points and 302 radial points.

   .. cpp:enumerator:: UltraFineGrid

      A default atomic grid size with 99 angular points and 590 radial points.

   .. cpp:enumerator:: SuperFineGrid

      A default atomic grid size with 250 angular points and 974 radial points.

   .. cpp:enumerator:: GM3

      A default atomic grid size with 35 angular points and 110 radial points.

   .. cpp:enumerator:: GM5

      A default atomic grid size with 50 angular points and 302 radial points.


.. cpp:enum-class:: GauXC::RadialQuad

   Enumeration of radial quadrature schemes for atomic grids.

   The following options are available:

   .. cpp:enumerator:: Becke

      The Becke radial quadrature scheme.\ :footcite:`becke1988`

   .. cpp:enumerator:: MuraKnowles

      The Mura-Knowles radial quadrature scheme.\ :footcite:`mura1996`

   .. cpp:enumerator:: TreutlerAhlrichs

      The Treutler-Ahlrichs radial quadrature scheme.\ :footcite:`treutler1995`

   .. cpp:enumerator:: MurrayHandyLaming

      The Murray-Handy-Laming radial quadrature scheme.\ :footcite:`murray1993`


.. cpp:enum-class:: GauXC::PruningScheme

   Enumeration of pruning schemes for constructing molecular grid weights from atomic grids.

   The following options are available:

   .. cpp:enumerator:: Unpruned

      No pruning is applied to the atomic grids when constructing the molecular grid.

   .. cpp:enumerator:: Robust

      Robust pruning scheme from Psi4.

   .. cpp:enumerator:: Treutler

      The Treutler pruning scheme.


.. cpp:type:: GauXCRadialSize = int64_t

   Type to represent the number of radial points in an atomic grid.

.. cpp:type:: GauXCAngularSize = int64_t

   Type to represent the number of angular points in an atomic grid.

.. cpp:type:: GauXC::BatchSize = int64_t

    Defines the batch size for processing grid points in parallel.
    Default is 512 points per batch, however larger values up around 10000 are recommended for better performance.


C bindings
----------

The following C bindings are available in the ``gauxc/molgrid.h`` header for creating and managing molecular grids in the GauXC C API.

.. c:struct:: GauXCMolGrid

   Opaque struct representing a molecular grid in the GauXC C API.

   .. c:function:: GauXCMolGrid gauxc_molgrid_new_default(GauXCStatus* status, const GauXCMolecule* molecule, enum GauXC_PruningScheme pruning_scheme, int64_t batch_size, enum GauXC_RadialQuad radial_quad, enum GauXC_AtomicGridSizeDefault grid_size)

      Create a new GauXCMolGrid object with default settings for the given molecule.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param molecule: Pointer to a GauXCMolecule object representing the molecule for which to create the MolGrid.
      :param pruning_scheme: The pruning scheme to use for constructing the molecular grid weights from the atomic grids.
      :param batch_size: The batch size for processing grid points in parallel.
      :param radial_quad: The radial quadrature scheme to use for the atomic grids.
      :param grid_size: The default atomic grid size to use for the atomic grids.
      :returns: A new GauXCMolGrid object initialized with the specified parameters for the given molecule.

   .. c:function:: void gauxc_molgrid_delete(GauXCStatus* status, GauXCMolGrid* molgrid)

      Delete a GauXCMolGrid object and free its resources.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param molgrid: Pointer to the GauXCMolGrid object to delete.


.. c:enum:: GauXC_PruningScheme

   Enumeration of pruning schemes for constructing molecular grid weights from atomic grids in the GauXC C API.

   The following options are available:

   .. c:enumerator:: GauXC_PruningScheme_Unpruned

      No pruning is applied to the atomic grids when constructing the molecular grid.

   .. c:enumerator:: GauXC_PruningScheme_Robust

      Robust pruning scheme from Psi4.

   .. c:enumerator:: GauXC_PruningScheme_Treutler

      The Treutler pruning scheme.


.. c:enum:: GauXC_RadialQuad

   Enumeration of radial quadrature schemes for atomic grids in the GauXC C API.

   The following options are available:

   .. c:enumerator:: GauXC_RadialQuad_Becke

      The Becke radial quadrature scheme.\ :footcite:`becke1988`

   .. c:enumerator:: GauXC_RadialQuad_MuraKnowles

      The Mura-Knowles radial quadrature scheme.\ :footcite:`mura1996`

   .. c:enumerator:: GauXC_RadialQuad_TreutlerAhlrichs

      The Treutler-Ahlrichs radial quadrature scheme.\ :footcite:`treutler1995`

   .. c:enumerator:: GauXC_RadialQuad_MurrayHandyLaming

      The Murray-Handy-Laming radial quadrature scheme.\ :footcite:`murray1993`


.. c:enum:: GauXC_AtomicGridSizeDefault

   Enumeration of default atomic grid sizes for molecular integration in the GauXC C API.

   The following options are available:

   .. c:enumerator:: GauXC_AtomicGridSizeDefault_FineGrid

      A default atomic grid size with 75 angular points and 302 radial points.

   .. c:enumerator:: GauXC_AtomicGridSizeDefault_UltraFineGrid

      A default atomic grid size with 99 angular points and 590 radial points.

   .. c:enumerator:: GauXC_AtomicGridSizeDefault_SuperFineGrid

      A default atomic grid size with 250 angular points and 974 radial points.

   .. c:enumerator:: GauXC_AtomicGridSizeDefault_GM3

      A default atomic grid size with 35 angular points and 110 radial points.

   .. c:enumerator:: GauXC_AtomicGridSizeDefault_GM5

      A default atomic grid size with 50 angular points and 302 radial points.


Fortran bindings
----------------

.. f:module:: gauxc_molgrid
   :synopsis: Fortran bindings for GauXC molecular grid objects.

.. f:currentmodule:: gauxc_molgrid

.. f:type:: gauxc_molgrid_type

   Opaque type representing a molecular grid in the GauXC Fortran API.
   Available in the module :f:mod:`gauxc_molgrid`.

   .. f:function:: function gauxc_molgrid_new_default(status, molecule, pruning_scheme, batch_size, radial_quad, grid_size)

      :param type(gauxc_status_type) status: Output parameter to store the status of the operation.
      :param type(gauxc_molecule_type) molecule: The molecule for which to create the MolGrid.
      :param integer(c_int) pruning_scheme: The pruning scheme to use for constructing the molecular grid weights from the atomic grids.
      :param integer(c_int64_t) batch_size: The batch size for processing grid points in parallel.
      :param integer(c_int) radial_quad: The radial quadrature scheme to use for the atomic grids.
      :param integer(c_int) grid_size: The default atomic grid size to use for the atomic grids.
      :returns type(gauxc_molgrid_type): A new gauxc_molgrid_type object initialized with the specified parameters for the given molecule.

.. f:currentmodule:: gauxc_enums

.. f:type:: gauxc_radialquad

   Parameter instance of a derived type with the respective enumerator values for each member variable.

   :f integer(c_int) becke:
      The Becke radial quadrature scheme.\ :footcite:`becke1988`

   :f integer(c_int) muraknowles:
      The Mura-Knowles radial quadrature scheme.\ :footcite:`mura1996`

   :f integer(c_int) treutlerahlrichs:
      The Treutler-Ahlrichs radial quadrature scheme.\ :footcite:`treutler1995`

   :f integer(c_int) murrayhandylaming:
      The Murray-Handy-Laming radial quadrature scheme.\ :footcite:`murray1993`

.. f:type:: gauxc_pruningscheme

   Parameter instance of a derived type with the respective enumerator values for each member variable.

   :f integer(c_int) unpruned:
      No pruning is applied to the atomic grids when constructing the molecular grid.

   :f integer(c_int) robust:
      Robust pruning scheme from Psi4.

   :f integer(c_int) treutler:
      The Treutler pruning scheme.

.. f:type:: gauxc_atomicgridsizedefault

   Parameter instance of a derived type with the respective enumerator values for each member variable.

   :f integer(c_int) finegrid:
      A default atomic grid size with 75 angular points and 302 radial points.

   :f integer(c_int) ultrafinegrid:
      A default atomic grid size with 99 angular points and 590 radial points.

   :f integer(c_int) superfinegrid:
      A default atomic grid size with 250 angular points and 974 radial points.

   :f integer(c_int) gm3:
      A default atomic grid size with 35 angular points and 110 radial points.

   :f integer(c_int) gm5:
      A default atomic grid size with 50 angular points and 302 radial points.


References
----------

.. footbibliography::