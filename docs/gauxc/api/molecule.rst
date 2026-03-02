GauXC Molecule API
==================

This section provides a reference for the GauXC API related to molecule objects, including C++ class definitions, C bindings, and Fortran bindings.

C++ defintions
--------------

.. cpp:class:: GauXC::Molecule : public std::Vector<GauXC::Atom>

   A class to represent a collection of atoms in a molecule.

   .. cpp:function:: inline Molecule(std::vector<Atom> atoms)

      Construct a Molecule object from a vector of Atom objects.

      :param atoms: A vector of Atom objects representing the atoms in the molecule.
      :returns: A Molecule object initialized with the given atoms.

   .. cpp:function:: inline size_t natoms() const

      Get the number of atoms in the molecule.

      :returns: The number of atoms in the molecule.

   .. cpp:function:: inline AtomicNumber maxZ() const

      Get the maximum atomic number among the atoms in the molecule.

      :returns: The maximum atomic number among the atoms in the molecule.

   .. cpp:function:: inline bool operator==(const Molecule& other) const

      Compare this molecule with another molecule for equality.

      :param other: Another molecule to compare with.
      :returns: True if this molecule is equal to the other molecule, false otherwise.

.. cpp:struct:: GauXC::Atom

   Representation of an atom in a molecule, containing its atomic number and Cartesian coordinates.

   .. cpp:member:: AtomicNumber Z

      The atomic number of the atom.

   .. cpp:member:: double x

      The x-coordinate of the atom in Cartesian coordinates.

   .. cpp:member:: double y

      The y-coordinate of the atom in Cartesian coordinates.

   .. cpp:member:: double z

      The z-coordinate of the atom in Cartesian coordinates.

   .. cpp:function:: inline Atom(AtomicNumber Z, double x, double y, double z)

      Construct an Atom object with the given atomic number and coordinates.

      :param Z: The atomic number of the atom.
      :param x: The x-coordinate of the atom in Cartesian coordinates.
      :param y: The y-coordinate of the atom in Cartesian coordinates.
      :param z: The z-coordinate of the atom in Cartesian coordinates.
      :returns: An Atom object initialized with the given atomic number and coordinates.

.. cpp:type:: GauXC::AtomicNumber = int64_t

   A type to represent the atomic number of an atom.

C bindings
----------

.. c:struct:: GauXCMolecule

   Opaque struct representing a molecule in the GauXC C API.

   .. c:function:: GauXCMolecule gauxc_molecule_new(GauXCStatus* status)

      Create a new GauXCMolecule object.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :returns: A new GauXCMolecule object.

   .. c:function:: GauXCMolecule gauxc_molecule_new_from_atoms(GauXCStatus* status, const GauXCAtom* atoms, size_t natoms)

      Create a new GauXCMolecule object from an array of GauXCAtom structs.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param atoms: Pointer to an array of GauXCAtom structs representing the atoms in the molecule.
      :param natoms: The number of atoms in the molecule.
      :returns: A new GauXCMolecule object initialized with the given atoms.

   .. c:function:: void gauxc_molecule_delete(GauXCStatus* status, GauXCMolecule* molecule)

      Delete a GauXCMolecule object.

      :param status: Pointer to a GauXCStatus variable to store the status of the operation.
      :param molecule: Pointer to the GauXCMolecule object to be deleted.

.. c:struct:: GauXCAtom

   Representation of an atom in a molecule for the GauXC C API, containing its atomic number and Cartesian coordinates.

   .. c:member:: int64_t Z

      The atomic number of the atom.

   .. c:member:: double x

      The x-coordinate of the atom in Cartesian coordinates.

   .. c:member:: double y

      The y-coordinate of the atom in Cartesian coordinates.

   .. c:member:: double z

      The z-coordinate of the atom in Cartesian coordinates.


Fortran bindings
----------------

.. f:module:: gauxc_molecule
   :synopsis: Fortran bindings for GauXC molecule objects.

.. f:currentmodule:: gauxc_molecule

.. f:type:: gauxc_molecule_type

   Opaque type representing a molecule in the GauXC Fortran API.
   Available from the module :f:mod:`gauxc_molecule`.

   .. f:function:: gauxc_molecule_new(status)

      Create a new gauxc_molecule_type object.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :returns type(gauxc_molecule_type): A new gauxc_molecule_type object.

   .. f:function:: gauxc_molecule_new_from_atoms(status, atoms, natoms)

      Create a new gauxc_molecule_type object from an array of gauxc_atom_type objects.

      :param type(gauxc_status_type) status: A variable to store the status of the operation.
      :param type(gauxc_atom_type) atoms(*): An array of :f:type:`gauxc_atom_type` objects representing the atoms in the molecule.
      :param integer(c_int64_t) natoms: The number of atoms in the molecule.
      :returns type(gauxc_molecule_type): A new gauxc_molecule_type object initialized with the given atoms.

.. f:currentmodule:: gauxc_atom

.. f:type:: gauxc_atom_type

   A derived type representing an atom in the GauXC Fortran API, containing its atomic number and Cartesian coordinates.

   :f integer(c_int64_t) Z:
      The atomic number of the atom.

   :f real(c_double) x:
      The x-coordinate of the atom in Cartesian coordinates.

   :f real(c_double) y:
      The y-coordinate of the atom in Cartesian coordinates.

   :f real(c_double) z:
      The z-coordinate of the atom in Cartesian coordinates.

Serialization to HDF5
---------------------

If GauXC has been built with HDF5 support :c:macro:`GAUXC_HAS_HDF5`, the :cpp:class:`GauXC::Molecule` class can be serialized to and deserialized from HDF5 files using the provided HDF5 interface.
This allows for easy storage and retrieval of molecular data in a standardized format.

.. cpp:function:: void GauXC::write_hdf5_record(const Molecule& molecule, const std::string& filename, const std::string& group_name)

   Write a Molecule object to an HDF5 file.

   :param molecule: The Molecule object to be written to the file.
   :param filename: The name of the HDF5 file to write to.
   :param group_name: The name of the group in the HDF5 file where the molecule data will be stored.

.. cpp:function:: void GauXC::read_hdf5_record(Molecule& molecule, const std::string& filename, const std::string& group_name)

   Read a Molecule object from an HDF5 file.

   :param molecule: The Molecule object to be populated with the data read from the file.
   :param filename: The name of the HDF5 file to read from.
   :param group_name: The name of the group in the HDF5 file where the molecule data is stored.

The same functions are available in the GauXC C API.

.. c:function:: void gauxc_molecule_write_hdf5_record(GauXCMolecule molecule, const char* filename, const char* group_name)

   Write a GauXCMolecule object to an HDF5 file in C.

   :param molecule: The GauXCMolecule object to be written to the file.
   :param filename: The name of the HDF5 file to write to.
   :param group_name: The name of the group in the HDF5 file where the molecule data will be stored.

.. c:function:: void gauxc_molecule_read_hdf5_record(GauXCMolecule molecule, const char* filename, const char* group_name)

   Read a GauXCMolecule object from an HDF5 file in C.

   :param molecule: The GauXCMolecule object to be populated with the data read from the file.
   :param filename: The name of the HDF5 file to read from.
   :param group_name: The name of the group in the HDF5 file where the molecule data is stored.

The same functions are also available in the GauXC Fortran API, available from the module :f:mod:`gauxc_external_hdf5`.

.. f:module:: gauxc_external_hdf5
   :synopsis: HDF5 serialization functions for GauXC molecule objects.

.. f:currentmodule:: gauxc_external_hdf5

.. f:subroutine:: gauxc_molecule_write_hdf5_record(molecule, filename, group_name)

   Write a gauxc_molecule_type object to an HDF5 file in Fortran.

   :param type(gauxc_molecule_type) molecule: The gauxc_molecule_type object to be written to the file.
   :param character(len=*) filename: The name of the HDF5 file to write to.
   :param character(len=*) group_name: The name of the group in the HDF5 file where the molecule data will be stored.

.. f:subroutine:: gauxc_molecule_read_hdf5_record(molecule, filename, group_name)
   
   Read a gauxc_molecule_type object from an HDF5 file in Fortran.

   :param type(gauxc_molecule_type) molecule: The gauxc_molecule_type object to be populated with the data read from the file.
   :param character(len=*) filename: The name of the HDF5 file to read from.
   :param character(len=*) group_name: The name of the group in the HDF5 file where the molecule data is stored.

.. f:currentmodule::