GauXC Basis Set API
===================

This section provides a reference for the GauXC basis set API, including C++ class definitions, C struct definitions, and Fortran derived type definitions for representing basis sets and shells in GauXC.

C++ definitions
---------------

.. cpp:struct:: template<typename F> \
                GauXC::BasisSet : public std::vector<Shell<F>>

   A class to represent a collection of shells in a basis set.

   .. cpp:function:: inline BasisSet(std::vector<Shell<F>> shells)

      Construct a BasisSet object from a vector of Shell objects.

      :param shells: A vector of Shell objects representing the shells in the basis set.
      :returns: A BasisSet object initialized with the given shells.

   .. cpp:function:: inline int32_t nshells() const

      Get the number of shells in the basis set.

      :returns: The number of shells in the basis set.

   .. cpp:function:: inline int32_t nbf() const

      Get the total number of basis functions in the basis set, calculated as the sum of the sizes of all shells.

      :returns: The total number of basis functions in the basis set.


.. cpp:class:: template<typename F> \
               GauXC::Shell

   A class to represent a shell in a basis set, containing its angular momentum and a vector of primitives.

   .. cpp:type:: prim_array = std::array<F, 32>

      A type to represent an array of primitive exponents or coefficients in a shell.

   .. cpp:type:: cart_array = std::array<double, 3>

      A type to represent an array of Cartesian coordinates for the center of a shell.

   .. cpp:member:: PrimSize nprim_

      The number of primitives in the shell.

   .. cpp:member:: AngularMomentum l_

      The angular momentum of the shell.

   .. cpp:member:: SphericalType pure_

      Whether the shell is pure (spherical) or Cartesian.

   .. cpp:member:: prim_array alpha_

      The exponents of the primitives in the shell.

   .. cpp:member:: prim_array coeff_

      The coefficients of the primitives in the shell.

   .. cpp:member:: cart_array O_

      The Cartesian coordinates of the center of the shell.

   .. cpp:member:: double cutoff_radius_

      The cutoff radius of the shell, defined as the maximum distance from the center of the shell at which the primitives have non-negligible contributions.

   .. cpp:member:: double tolerance_

      The tolerance for pruning primitives in the shell. Primitives with coefficients below this value will be pruned. The default value is 1e-10.

   .. cpp:function:: inline void set_shell_tolerance(double tol)

      Set the tolerance for pruning primitives in the shell.
      The default value for the tolerance is 1e-10.

      :param tol: The tolerance for pruning primitives. Primitives with coefficients below this value will be pruned.

   .. cpp:function:: inline PrimSize nprim() const

      Get the number of primitives in the shell.

      :returns: The number of primitives in the shell.

   .. cpp:function:: inline AngularMomentum l() const

      Get the angular momentum of the shell.

      :returns: The angular momentum of the shell.

   .. cpp:function:: inline SphericalType pure() const

      Get whether the shell is pure (spherical) or Cartesian.

      :returns: ``1`` if the shell is pure (spherical), ``0`` if the shell is Cartesian.

   .. cpp:function:: inline const F* alpha_data() const

      Get a pointer to the exponents of the primitives in the shell.

      :returns: A pointer to the exponents of the primitives in the shell.

   .. cpp:function:: inline const F* coeff_data() const

      Get a pointer to the coefficients of the primitives in the shell.

      :returns: A pointer to the coefficients of the primitives in the shell.

   .. cpp:function:: inline const double* O_data() const

      Get a pointer to the Cartesian coordinates of the center of the shell.

      :returns: A pointer to the Cartesian coordinates of the center of the shell.

   .. cpp::function:: inline F* alpha_data()

      Get a mutable pointer to the exponents of the primitives in the shell.

      :returns: A mutable pointer to the exponents of the primitives in the shell.

   .. cpp::function:: inline F* coeff_data()

      Get a mutable pointer to the coefficients of the primitives in the shell.

      :returns: A mutable pointer to the coefficients of the primitives in the shell.

   .. cpp::function:: inline double* O_data()

      Get a mutable pointer to the Cartesian coordinates of the center of the shell.

      :returns: A mutable pointer to the Cartesian coordinates of the center of the shell.

   .. cpp:function:: inline double cutoff_radius() const

      Get the cutoff radius of the shell, defined as the maximum distance from the center of the shell at which the primitives have non-negligible contributions.

      :returns: The cutoff radius of the shell.

   .. cpp:function:: inline int32_t cart_size() const

      Get the number of Cartesian functions in the shell.

      :returns: The number of Cartesian functions in the shell.

   .. cpp:function:: inline int32_t pure_size() const

      Get the number of pure (spherical) functions in the shell.

      :returns: The number of pure (spherical) functions in the shell.

   .. cpp:function:: inline int32_t size() const

      Get the total number of functions in the shell, based on whether the shell is pure (spherical) or Cartesian.

      :returns: The total number of functions in the shell.

   .. cpp:function:: inline prim_array& alpha() const

      Get the exponents of the primitives in the shell as an array.

      :returns: An array containing the exponents of the primitives in the shell.

   .. cpp:function:: inline prim_array& coeff() const

      Get the coefficients of the primitives in the shell as an array.

      :returns: An array containing the coefficients of the primitives in the shell.

   .. cpp:function:: inline cart_array& O() const

      Get the Cartesian coordinates of the center of the shell as an array.

      :returns: An array containing the Cartesian coordinates of the center of the shell.

   .. cpp:function:: inline prim_array& alpha()

      Get the exponents of the primitives in the shell as a mutable array.

      :returns: A mutable array containing the exponents of the primitives in the shell.

   .. cpp:function:: inline prim_array& coeff()

      Get the coefficients of the primitives in the shell as a mutable array.

      :returns: A mutable array containing the coefficients of the primitives in the shell.

   .. cpp:function:: inline cart_array& O()

      Get the Cartesian coordinates of the center of the shell as a mutable array.

      :returns: A mutable array containing the Cartesian coordinates of the center of the shell.

   .. cpp:function:: inline void set_pure(bool pure)

      Set whether the shell is pure (spherical) or Cartesian.

      :param pure: ``1`` to set the shell as pure (spherical), ``0`` to set the shell as Cartesian.

   .. cpp:function:: inline bool operator==(const Shell& other) const

      Compare this shell with another shell for equality.

      :param other: Another shell to compare with.
      :returns: True if this shell is equal to the other shell, false otherwise.


.. cpp:type:: PrimSize = int32_t

   A type to represent the number of primitives in a shell.

.. cpp:type:: AngularMomentum = int32_t

   A type to represent the angular momentum of a shell.

.. cpp:type:: SphericalType = int32_t

   A type to represent whether a shell is pure (spherical) or Cartesian.
   ``1`` indicates a pure (spherical) shell, while ``0`` indicates a Cartesian shell.

C bindings
----------

.. c:struct:: GauXCBasisSet

   An opaque struct to represent a basis set in the C API.

   .. c:function:: GauXCBasisSet gauxc_basisset_new(GauXCStatus* status)

      Create a new GauXCBasisSet object.

      :param status: A pointer to a GauXCStatus variable to store the status of the operation.
      :returns: A new GauXCBasisSet object.

   .. c:function:: GauXCBasisSet gauxc_basisset_new_from_shells(GauXCStatus* status, const GauXCShell* shells, int32_t nshells, bool normalize)

      Create a new GauXCBasisSet object from an array of GauXCShell objects.

      :param status: A pointer to a GauXCStatus variable to store the status of the operation.
      :param shells: A pointer to an array of GauXCShell objects representing the shells in the basis set.
      :param nshells: The number of shells in the array.
      :param normalize: Whether to normalize the primitives in the shells when creating the basis set.
      :returns: A new GauXCBasisSet object initialized with the given shells.

   .. c:function:: void gauxc_basisset_delete(GauXCStatus* status, GauXCBasisSet* basis)

      Delete a GauXCBasisSet object and free its associated memory.

      :param status: A pointer to a GauXCStatus variable to store the status of the operation.
      :param basis: A pointer to the GauXCBasisSet object to be deleted.

.. c:struct:: GauXCShell

   Representation of a shell in a basis set for the C API, containing its angular momentum, number of primitives, and pointers to arrays of primitive exponents, coefficients, and Cartesian coordinates.

   .. c:member:: int32_t l

      The angular momentum of the shell.

   .. c:member:: bool pure

      Whether the shell is pure (spherical) or Cartesian.

   .. c:member:: int32_t nprim

      The number of primitives in the shell.

   .. c:member:: double alpha[32]

      An array of exponents of the primitives in the shell.

   .. c:member:: double coeff[32]

      An array of coefficients of the primitives in the shell.

   .. c:member:: double O[3]

      An array of Cartesian coordinates for the center of the shell.

   .. c:member:: double shell_tolerance

      The tolerance for pruning primitives in the shell.
      Primitives with coefficients below this value will be pruned.
      The default value is 1e-10.

Fortran bindings
----------------

.. f:module:: gauxc_basisset
   :synopsis: Fortran bindings for the GauXC basis set API.

.. f:currentmodule:: gauxc_basisset

.. f:type:: gauxc_basisset_type

   Opaque type representing a basis set in the GauXC Fortran API.
   Available from :f:mod:`gauxc_basisset`.

   .. f:function:: gauxc_basisset_new(status)

      Create a new gauxc_basisset_type object.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :returns type(gauxc_basisset_type): A new gauxc_basisset_type object.

   .. f:function:: gauxc_basisset_new_from_shells(status, shells, nshells, normalize)

      Create a new gauxc_basisset_type object from an array of gauxc_shell_type objects.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_shell_type) shells(*): An array of :f:type:`gauxc_shell_type` objects representing the shells in the basis set.
      :param integer(c_int32_t) nshells: The number of shells in the array.
      :param logical(c_bool) normalize: Whether to normalize the primitives in the shells when creating the basis set.
      :returns type(gauxc_basisset_type): A new gauxc_basisset_type object initialized with the given shells.

.. f:currentmodule:: gauxc_shell

.. f:type:: gauxc_shell_type

   A derived type representing a shell in the GauXC Fortran API, containing its angular momentum, number of primitives, and arrays of primitive exponents, coefficients, and Cartesian coordinates.

   :f integer(c_int32_t) l:
      The angular momentum of the shell.

   :f logical(c_bool) pure:
      Whether the shell is pure (spherical) or Cartesian.

   :f integer(c_int32_t) nprim:
      The number of primitives in the shell.

   :f real(c_double) alpha(32):
      An array of exponents of the primitives in the shell.

   :f real(c_double) coeff(32):
      An array of coefficients of the primitives in the shell.

   :f real(c_double) O(3):
      An array of Cartesian coordinates for the center of the shell.

   :f real(c_double) shell_tolerance:
      The tolerance for pruning primitives in the shell.
      Primitives with coefficients below this value will be pruned.
      The default value is 1e-10.

Serialization to HDF5
---------------------

If GauXC has been built with HDF5 support :c:macro:`GAUXC_HAS_HDF5`, the :cpp:class:`GauXC::BasisSet` class can be serialized to and deserialized from HDF5 files using the provided HDF5 interface.
This allows for easy storage and retrieval of basis set data in a standardized format.

.. cpp:function:: void GauXC::write_hdf5_record(const BasisSet& basis, const std::string& filename, const std::string& group_name)

   Write a BasisSet object to an HDF5 file.

   :param basis: The BasisSet object to be written to the file.
   :param filename: The name of the HDF5 file to write to.
   :param group_name: The name of the group in the HDF5 file where the basis set data will be stored.

.. cpp:function:: void GauXC::read_hdf5_record(BasisSet& basis, const std::string& filename, const std::string& group_name)

   Read a BasisSet object from an HDF5 file.

   :param basis: The BasisSet object to be populated with the data read from the file.
   :param filename: The name of the HDF5 file to read from.
   :param group_name: The name of the group in the HDF5 file where the basis set data is stored.

The same functions are available in the GauXC C API.

.. c:function:: void gauxc_basisset_write_hdf5_record(GauXCBasisSet basis, const char* filename, const char* group_name)

   Write a GauXCBasisSet object to an HDF5 file in C.

   :param basis: The GauXCBasisSet object to be written to the file.
   :param filename: The name of the HDF5 file to write to.
   :param group_name: The name of the group in the HDF5 file where the basis set data will be stored.

.. c:function:: void gauxc_basisset_read_hdf5_record(GauXCBasisSet basis, const char* filename, const char* group_name)

   Read a GauXCBasisSet object from an HDF5 file in C.

   :param basis: The GauXCBasisSet object to be populated with the data read from the file.
   :param filename: The name of the HDF5 file to read from.
   :param group_name: The name of the group in the HDF5 file where the basis set data is stored.

The same functions are also available in the GauXC Fortran API, available from the module :f:mod:`gauxc_external_hdf5`.

.. f:module:: gauxc_external_hdf5
   :synopsis: HDF5 serialization functions for GauXC basis set objects.

.. f:currentmodule:: gauxc_external_hdf5

.. f:subroutine:: gauxc_basisset_write_hdf5_record(basis, filename, group_name)

   Write a gauxc_basisset_type object to an HDF5 file in Fortran.

   :param type(gauxc_basisset_type) basis: The gauxc_basisset_type object to be written to the file.
   :param character(len=*) filename: The name of the HDF5 file to write to.
   :param character(len=*) group_name: The name of the group in the HDF5 file where the basis set data will be stored.

.. f:subroutine:: gauxc_basisset_read_hdf5_record(basis, filename, group_name)

   Read a gauxc_basisset_type object from an HDF5 file in Fortran.

   :param type(gauxc_basisset_type) basis: The gauxc_basisset_type object to be populated with the data read from the file.
   :param character(len=*) filename: The name of the HDF5 file to read from.
   :param character(len=*) group_name: The name of the group in the HDF5 file where the basis set data is stored.


.. _gauxc-orbital-ordering:

Orbital Ordering Convention
---------------------------

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