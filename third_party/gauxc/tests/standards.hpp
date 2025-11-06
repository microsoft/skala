/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#include <gauxc/basisset.hpp>
#include <gauxc/molecule.hpp>

namespace GauXC {

Molecule         make_water();
Molecule         make_benzene();
Molecule         make_ubiquitin();
Molecule         make_taxol();
BasisSet<double> make_631Gd( const Molecule&, SphericalType );
BasisSet<double> make_ccpvdz( const Molecule&, SphericalType );

}
