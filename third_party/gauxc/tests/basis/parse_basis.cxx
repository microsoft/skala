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
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <iterator>

namespace GauXC {

std::map<std::string,int>  atomic_number_map = {
    {"H",  1 }, 
    {"HE", 2 }, 
    {"LI", 3 }, 
    {"BE", 4 }, 
    {"B",  5 }, 
    {"C",  6 }, 
    {"N",  7 }, 
    {"O",  8 }, 
    {"F",  9 }, 
    {"NE", 10}, 
    {"NA", 11}, 
    {"MG", 12}, 
    {"AL", 13}, 
    {"SI", 14}, 
    {"P",  15}, 
    {"S",  16}, 
    {"CL", 17}, 
    {"AR", 18}, 
    {"K",  19}, 
    {"CA", 20}, 
    {"SC", 21}, 
    {"TI", 22}, 
    {"V",  23}, 
    {"CR", 24}, 
    {"MN", 25}, 
    {"FE", 26}, 
    {"CO", 27}, 
    {"NI", 28}, 
    {"CU", 29}, 
    {"ZN", 30}, 
    {"GA", 31}, 
    {"GE", 32}, 
    {"AS", 33}, 
    {"SE", 34}, 
    {"BR", 35}, 
    {"KR", 36}, 
    {"RB", 37}, 
    {"SR", 38}, 
    {"Y",  39}, 
    {"ZR", 40}, 
    {"NB", 41}, 
    {"MO", 42}, 
    {"TC", 43}, 
    {"RU", 44}, 
    {"RH", 45}, 
    {"PD", 46}, 
    {"AG", 47}, 
    {"CD", 48}, 
    {"IN", 49}, 
    {"SN", 50}, 
    {"SB", 51}, 
    {"TE", 52}, 
    {"I",  53}, 
    {"XE", 54}, 
    {"CS", 55}, 
    {"BA", 56}, 
    {"LA", 57}, 
    {"CE", 58}, 
    {"PR", 59}, 
    {"ND", 60}, 
    {"PM", 61}, 
    {"SM", 62}, 
    {"EU", 63}, 
    {"GD", 64}, 
    {"TB", 65}, 
    {"DY", 66}, 
    {"HO", 67}, 
    {"ER", 68}, 
    {"TM", 69}, 
    {"YB", 70}, 
    {"LU", 71}, 
    {"HF", 72}, 
    {"TA", 73}, 
    {"W",  74}, 
    {"RE", 75}, 
    {"OS", 76}, 
    {"IR", 77}, 
    {"PT", 78}, 
    {"AU", 79}, 
    {"HG", 80}, 
    {"TL", 81}, 
    {"PB", 82}, 
    {"BI", 83}, 
    {"PO", 84}, 
    {"AT", 85}, 
    {"RN", 86}, 
    {"FR", 87}, 
    {"RA", 88}, 
    {"AC", 89}, 
    {"TH", 90}, 
    {"PA", 91}, 
    {"U",  92}, 
    {"NP", 93}, 
    {"PU", 94}, 
    {"AM", 95}, 
    {"CM", 96}
};

std::map<std::string,int> am_map = {
  {"S",0},
  {"P",1},
  {"D",2},
  {"F",3},
  {"G",4},
  {"H",5},
  {"I",6},
  {"J",7}
};

namespace detail {
  inline static auto tokenize( std::string str,
                               std::string delim = " " ) {
    std::istringstream iss(str);
    std::vector<std::string> tokens;

    std::copy( std::istream_iterator<std::string>( iss ),
               std::istream_iterator<std::string>( ),
               std::back_inserter( tokens ) );

    
    return tokens;
  }
}

BasisSet<double> parse_basis( const Molecule& mol,
                              std::string     fname,
                              SphericalType   sph    ) {


  std::ifstream infile( fname );
  if( ! infile.good() ) throw std::runtime_error(fname + " not found!");

  std::vector<std::vector<std::string>> basis_records;
  {
    std::string line;

    while( std::getline( infile, line ) ) {

      if( line.find("!") != std::string::npos ) continue;
      if( line.size() == 0 ) continue;

      // New record
      if( line.find("****") != std::string::npos ) {
        basis_records.emplace_back();
        continue;
      }

      basis_records.back().emplace_back( line );

    }
  }

  std::map<int, BasisSet<double>> basis_shells;
  for( const auto& record : basis_records ) {
    if( record.size() == 0 ) continue;
    std::string atom_line = record.at(0);
    std::string atom_symb = atom_line.substr(0,2);
    if( atom_symb[1] == ' ' ) atom_symb = atom_symb[0];
    std::transform( atom_symb.begin(), atom_symb.end(), atom_symb.begin(),
                    [](auto a){ return std::toupper(a); } );
    
    //std::cout << atom_symb << std::endl;
    int Z = atomic_number_map.at(atom_symb);

    BasisSet<double> atom_basis;
    for( auto rec_it = record.begin()+1; rec_it != record.end(); ) {
      std::string type_line = *rec_it; rec_it++; // Read type line

      auto type_tokens = detail::tokenize( type_line ); 
      bool gencon = type_tokens.at(0) == "SP";
      int nprim = std::stoi(type_tokens.at(1));
      int l = gencon ? 0 : am_map.at(type_tokens.at(0));

      std::vector<double> alpha(nprim);
      std::vector<double> coeff_primary(nprim), coeff_secondary(nprim);
      for( int i = 0; i < nprim; ++i ) {
        std::string prim_line = *rec_it; rec_it++; // Read prim line
        for( auto& c : prim_line ) if( c == 'D' or c == 'd' ) c = 'e';
        auto prim_tokens = detail::tokenize( prim_line );

        alpha[i]         = std::stod( prim_tokens.at(0) );
        coeff_primary[i] = std::stod( prim_tokens.at(1) );
        if( gencon )
          coeff_secondary[i] = std::stod( prim_tokens.at(2) );

      }
      
      using prim_array = Shell<double>::prim_array;
      using cart_array = Shell<double>::cart_array;

      prim_array alpha_arr, coeff_primary_arr, coeff_secondary_arr;
      std::copy( alpha.begin(), alpha.end(), alpha_arr.begin() );
      std::copy( coeff_primary.begin(), coeff_primary.end(), 
                 coeff_primary_arr.begin() );
      if( gencon )
        std::copy( coeff_secondary.begin(), coeff_secondary.end(), 
                   coeff_secondary_arr.begin() );

      SphericalType sph_use = l > 1 ? sph : SphericalType(false);
      atom_basis.emplace_back( Shell<double>(
        PrimSize(nprim), AngularMomentum(l), sph_use,
        alpha_arr, coeff_primary_arr, {0., 0., 0.}
      ));

      if( gencon )
        atom_basis.emplace_back( Shell<double>(
          PrimSize(nprim), AngularMomentum(1), SphericalType(false),
          alpha_arr, coeff_secondary_arr, {0., 0., 0.}
        ));
    }

    basis_shells[Z] = atom_basis;

  }

#if 0
  std::cout << std::scientific << std::setprecision(16);
  for( const auto& [key, value] : basis_shells ) {
    std::cout << "Basis shells for Z = " << key << std::endl;
    for( const auto& sh : value ) {
      std::cout << "CEN = " << sh.O()[0] << ", " << sh.O()[1] << ", " << sh.O()[2] << std::endl;
      std::cout << "L = " << sh.l() << std::endl;
      std::cout << "CR = " << sh.cutoff_radius() << std::endl;
      std::cout << "PRIMS" << std::endl;
      for( auto p = 0; p < sh.nprim(); ++p )
        std::cout << "  " << sh.alpha()[p] << ", " << sh.coeff()[p] << std::endl;
      std::cout << std::endl;
    }
  }
#endif

  BasisSet<double> basis;
  for( auto iAt = 0; iAt < mol.size(); ++iAt ) {
    const auto& atom = mol.at(iAt);
    BasisSet<double> atom_basis = basis_shells.at(atom.Z.get());
    for( auto& sh : atom_basis ) sh.O() = {atom.x, atom.y, atom.z};
    
    basis.insert(basis.end(), atom_basis.begin(), atom_basis.end() );
  }
  return basis;
}

}
