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
#include <gauxc/molgrid/defaults.hpp>

namespace GauXC {

double default_atomic_radius(AtomicNumber Z) {

  // If the radius is in Slater-64, use it as the default
  auto slater_64 = slater_radius_64(Z);
  if( slater_64 > 0. ) return slater_64;

  // Fill in gaps with Clementi-67 data
  auto clementi_67 = clementi_radius_67(Z);
  if( clementi_67 > 0. ) return clementi_67;

  // Default to 2.01 Angstroms -> 3.79835 Bohr (???)
  return 3.79835;
  
}

long double pm_to_bohr( long double x ) {
  return x * 0.0188973000000929 / 1.00000205057;
}

/// Slater, J.C.
/// J. Chem. Phys. 41, 3199, 1964
/// https://doi.org/10.1063/1.1725697
double slater_radius_64(AtomicNumber _Z) {

  auto Z = _Z.get();
  switch(Z) {
    case 1:  /* H  */ return pm_to_bohr(25. );
  //case 2:  /* He */ return pm_to_bohr(120.);
    case 3:  /* Li */ return pm_to_bohr(145.);
    case 4:  /* Be */ return pm_to_bohr(105.);
    case 5:  /* B  */ return pm_to_bohr(85. );
    case 6:  /* C  */ return pm_to_bohr(70. );
    case 7:  /* N  */ return pm_to_bohr(65. );
    case 8:  /* O  */ return pm_to_bohr(60. );
    case 9:  /* F  */ return pm_to_bohr(50. );
  //case 10: /* Ne */ return pm_to_bohr(160.);
    case 11: /* Na */ return pm_to_bohr(180.);
    case 12: /* Mg */ return pm_to_bohr(150.);
    case 13: /* Al */ return pm_to_bohr(125.);
    case 14: /* Si */ return pm_to_bohr(110.);
    case 15: /* P  */ return pm_to_bohr(100.);
    case 16: /* S  */ return pm_to_bohr(100.);
    case 17: /* Cl */ return pm_to_bohr(100.);
  //case 18: /* Ar */ return pm_to_bohr(71. );
    case 19: /* K  */ return pm_to_bohr(220.);
    case 20: /* Ca */ return pm_to_bohr(180.);
    case 21: /* Sc */ return pm_to_bohr(160.);
    case 22: /* Ti */ return pm_to_bohr(140.);
    case 23: /* V  */ return pm_to_bohr(135.);
    case 24: /* Cr */ return pm_to_bohr(140.);
    case 25: /* Mn */ return pm_to_bohr(140.);
    case 26: /* Fe */ return pm_to_bohr(140.);
    case 27: /* Co */ return pm_to_bohr(135.);
    case 28: /* Ni */ return pm_to_bohr(135.);
    case 29: /* Cu */ return pm_to_bohr(135.);
    case 30: /* Zn */ return pm_to_bohr(135.);
    case 31: /* Ga */ return pm_to_bohr(130.);
    case 32: /* Ge */ return pm_to_bohr(125.);
    case 33: /* As */ return pm_to_bohr(115.);
    case 34: /* Se */ return pm_to_bohr(115.);
    case 35: /* Br */ return pm_to_bohr(115.);
                                              
    case 37: /* Rb */ return pm_to_bohr(235.);
    case 38: /* Sr */ return pm_to_bohr(200.);
    case 39: /* Y  */ return pm_to_bohr(180.);
    case 40: /* Zr */ return pm_to_bohr(155.);
    case 41: /* Nb */ return pm_to_bohr(145.);
    case 42: /* Mo */ return pm_to_bohr(145.);
    case 43: /* Tc */ return pm_to_bohr(135.);
    case 44: /* Ru */ return pm_to_bohr(130.);
    case 45: /* Rh */ return pm_to_bohr(135.);
    case 46: /* Pd */ return pm_to_bohr(140.);
    case 47: /* Ag */ return pm_to_bohr(160.);
    case 48: /* Cd */ return pm_to_bohr(155.);
    case 49: /* In */ return pm_to_bohr(155.);
    case 50: /* Sn */ return pm_to_bohr(145.);
    case 51: /* Sb */ return pm_to_bohr(145.);
    case 52: /* Te */ return pm_to_bohr(140.);
    case 53: /* I  */ return pm_to_bohr(140.);
                                              
    case 55: /* Cs */ return pm_to_bohr(265.);
    case 56: /* Ba */ return pm_to_bohr(215.);
    case 57: /* La */ return pm_to_bohr(195.);
    case 58: /* Ce */ return pm_to_bohr(185.);
    case 59: /* Pr */ return pm_to_bohr(185.);
    case 60: /* Nd */ return pm_to_bohr(185.);
    case 61: /* Pm */ return pm_to_bohr(185.);
    case 62: /* Sm */ return pm_to_bohr(185.);
    case 63: /* Eu */ return pm_to_bohr(185.);
    case 64: /* Gd */ return pm_to_bohr(180.);
    case 65: /* Tb */ return pm_to_bohr(175.);
    case 66: /* Dy */ return pm_to_bohr(175.);
    case 67: /* Ho */ return pm_to_bohr(175.);
    case 68: /* Er */ return pm_to_bohr(175.);
    case 69: /* Tm */ return pm_to_bohr(175.);
    case 70: /* Yb */ return pm_to_bohr(175.);
    case 71: /* Lu */ return pm_to_bohr(175.);
    case 72: /* Hf */ return pm_to_bohr(155.);
    case 73: /* Ta */ return pm_to_bohr(145.);
    case 74: /* W  */ return pm_to_bohr(135.);
    case 75: /* Re */ return pm_to_bohr(135.);
    case 76: /* Os */ return pm_to_bohr(130.);
    case 77: /* Ir */ return pm_to_bohr(135.);
    case 78: /* Pt */ return pm_to_bohr(135.);
    case 79: /* Au */ return pm_to_bohr(135.);
    case 80: /* Hg */ return pm_to_bohr(150.);
    case 81: /* Tl */ return pm_to_bohr(190.);
    case 82: /* Pb */ return pm_to_bohr(180.);
    case 83: /* Bi */ return pm_to_bohr(160.);
    case 84: /* Po */ return pm_to_bohr(190.);
                                              
    case 88: /* Ra */ return pm_to_bohr(215.);
    case 89: /* Ac */ return pm_to_bohr(195.);
    case 90: /* Th */ return pm_to_bohr(180.);
    case 91: /* Pa */ return pm_to_bohr(180.);
    case 92: /* U  */ return pm_to_bohr(175.);
    case 93: /* Np */ return pm_to_bohr(175.);
    case 94: /* Pu */ return pm_to_bohr(175.);
    case 95: /* Am */ return pm_to_bohr(175.);
  //case 96: /* Cm */ return pm_to_bohr(176.); }
 
    default: return -1.;
  }
}

/// Slater, J.C.
/// Phys. Rev. 36, 57, 1930
/// https://doi.org/10.1103/PhysRev.36.57
double slater_radii_30(AtomicNumber _Z) {

  auto Z = _Z.get();
  switch(Z) {
    case 1:   /* H  */ return pm_to_bohr(53. ); 

    case 3:   /* Li */ return pm_to_bohr(163.); 
    case 4:   /* Be */ return pm_to_bohr(109.); 
    case 5:   /* B  */ return pm_to_bohr(82. ); 
    case 6:   /* C  */ return pm_to_bohr(65. ); 
    case 7:   /* N  */ return pm_to_bohr(55. ); 
    case 8:   /* O  */ return pm_to_bohr(47. ); 
    case 9:   /* F  */ return pm_to_bohr(41. ); 

    case 11:  /* Na */ return pm_to_bohr(217.); 
    case 12:  /* Mg */ return pm_to_bohr(168.); 
    case 13:  /* Al */ return pm_to_bohr(137.); 
    case 14:  /* Si */ return pm_to_bohr(115.); 
    case 15:  /* P  */ return pm_to_bohr(100.); 
    case 16:  /* S  */ return pm_to_bohr(88. ); 
    case 17:  /* Cl */ return pm_to_bohr(78. ); 
                                                                  
    case 19:  /* K  */ return pm_to_bohr(332.); 
    case 20:  /* Ca */ return pm_to_bohr(256.); 
    case 21:  /* Sc */ return pm_to_bohr(243.); 
    case 22:  /* Ti */ return pm_to_bohr(232.); 
    case 23:  /* V  */ return pm_to_bohr(222.); 
    case 24:  /* Cr */ return pm_to_bohr(212.); 
    case 25:  /* Mn */ return pm_to_bohr(202.); 
    case 26:  /* Fe */ return pm_to_bohr(195.); 
    case 27:  /* Co */ return pm_to_bohr(187.); 
    case 28:  /* Ni */ return pm_to_bohr(180.); 
    case 29:  /* Cu */ return pm_to_bohr(173.); 
    case 30:  /* Zn */ return pm_to_bohr(167.); 
    case 31:  /* Ga */ return pm_to_bohr(146.); 
    case 32:  /* Ge */ return pm_to_bohr(129.); 
    case 33:  /* As */ return pm_to_bohr(116.); 
    case 34:  /* Se */ return pm_to_bohr(105.); 
    case 35:  /* Br */ return pm_to_bohr(96. ); 
                                                                  
    case 37:  /* Rb */ return pm_to_bohr(386.); 
    case 38:  /* Sr */ return pm_to_bohr(300.); 
    case 39:  /* Y  */ return pm_to_bohr(284.); 
    case 40:  /* Zr */ return pm_to_bohr(271.); 
    case 41:  /* Nb */ return pm_to_bohr(260.); 
    case 42:  /* Mo */ return pm_to_bohr(248.); 
    case 43:  /* Tc */ return pm_to_bohr(236.); 
    case 44:  /* Ru */ return pm_to_bohr(228.); 
    case 45:  /* Rh */ return pm_to_bohr(218.); 
    case 46:  /* Pd */ return pm_to_bohr(210.); 
    case 47:  /* Ag */ return pm_to_bohr(202.); 
    case 48:  /* Cd */ return pm_to_bohr(195.); 
    case 49:  /* In */ return pm_to_bohr(171.); 
    case 50:  /* Sn */ return pm_to_bohr(151.); 
    case 51:  /* Sb */ return pm_to_bohr(135.); 
    case 52:  /* Te */ return pm_to_bohr(122.); 
    case 53:  /* I  */ return pm_to_bohr(112.); 
                                                                  
    case 55:  /* Cs */ return pm_to_bohr(425.); 
    case 56:  /* Ba */ return pm_to_bohr(330.); 
    case 57:  /* La */ return pm_to_bohr(312.); 

    case 73:  /* Ta */ return pm_to_bohr(286.); 
    case 74:  /* W  */ return pm_to_bohr(273.); 
    case 75:  /* Re */ return pm_to_bohr(260.); 
    case 76:  /* Os */ return pm_to_bohr(251.); 
    case 77:  /* Ir */ return pm_to_bohr(240.); 
    case 78:  /* Pt */ return pm_to_bohr(231.); 
    case 79:  /* Au */ return pm_to_bohr(222.); 
    case 80:  /* Hg */ return pm_to_bohr(215.); 
    case 81:  /* Tl */ return pm_to_bohr(188.); 
    case 82:  /* Pb */ return pm_to_bohr(166.); 
    case 83:  /* Bi */ return pm_to_bohr(148.);

    default: return -1.;
  }
}

/// Clementi, E., Raimondi, D.L., Reinhardt, W.P.
/// J. Chem. Phys. 47, 1300, 1967
/// https://doi.org/10.1063/1.1712084
double clementi_radius_67(AtomicNumber _Z) {

  auto Z = _Z.get();
  switch(Z) {
    case 2:   /* He */ return pm_to_bohr(31. ); 

    case 3:   /* Li */ return pm_to_bohr(167.); 
    case 4:   /* Be */ return pm_to_bohr(112.); 
    case 5:   /* B  */ return pm_to_bohr(87. ); 
    case 6:   /* C  */ return pm_to_bohr(67. ); 
    case 7:   /* N  */ return pm_to_bohr(56. ); 
    case 8:   /* O  */ return pm_to_bohr(48. ); 
    case 9:   /* F  */ return pm_to_bohr(42. ); 
    case 10:  /* Ne */ return pm_to_bohr(38. ); 

    case 11:  /* Na */ return pm_to_bohr(190.); 
    case 12:  /* Mg */ return pm_to_bohr(145.); 
    case 13:  /* Al */ return pm_to_bohr(118.); 
    case 14:  /* Si */ return pm_to_bohr(111.); 
    case 15:  /* P  */ return pm_to_bohr(98. ); 
    case 16:  /* S  */ return pm_to_bohr(88. ); 
    case 17:  /* Cl */ return pm_to_bohr(79. ); 
    case 18:  /* Ar */ return pm_to_bohr(71. ); 
                                                                  
    case 19:  /* K  */ return pm_to_bohr(243.); 
    case 20:  /* Ca */ return pm_to_bohr(194.); 
    case 21:  /* Sc */ return pm_to_bohr(184.); 
    case 22:  /* Ti */ return pm_to_bohr(176.); 
    case 23:  /* V  */ return pm_to_bohr(171.); 
    case 24:  /* Cr */ return pm_to_bohr(166.); 
    case 25:  /* Mn */ return pm_to_bohr(161.); 
    case 26:  /* Fe */ return pm_to_bohr(156.); 
    case 27:  /* Co */ return pm_to_bohr(152.); 
    case 28:  /* Ni */ return pm_to_bohr(149.); 
    case 29:  /* Cu */ return pm_to_bohr(145.); 
    case 30:  /* Zn */ return pm_to_bohr(142.); 
    case 31:  /* Ga */ return pm_to_bohr(136.); 
    case 32:  /* Ge */ return pm_to_bohr(125.); 
    case 33:  /* As */ return pm_to_bohr(114.); 
    case 34:  /* Se */ return pm_to_bohr(103.); 
    case 35:  /* Br */ return pm_to_bohr(94. ); 
    case 36:  /* Kr */ return pm_to_bohr(88. ); 
                                                                  
    case 37:  /* Rb */ return pm_to_bohr(265.); 
    case 38:  /* Sr */ return pm_to_bohr(219.); 
    case 39:  /* Y  */ return pm_to_bohr(212.); 
    case 40:  /* Zr */ return pm_to_bohr(206.); 
    case 41:  /* Nb */ return pm_to_bohr(198.); 
    case 42:  /* Mo */ return pm_to_bohr(190.); 
    case 43:  /* Tc */ return pm_to_bohr(183.); 
    case 44:  /* Ru */ return pm_to_bohr(178.); 
    case 45:  /* Rh */ return pm_to_bohr(173.); 
    case 46:  /* Pd */ return pm_to_bohr(169.); 
    case 47:  /* Ag */ return pm_to_bohr(165.); 
    case 48:  /* Cd */ return pm_to_bohr(161.); 
    case 49:  /* In */ return pm_to_bohr(156.); 
    case 50:  /* Sn */ return pm_to_bohr(145.); 
    case 51:  /* Sb */ return pm_to_bohr(133.); 
    case 52:  /* Te */ return pm_to_bohr(123.); 
    case 53:  /* I  */ return pm_to_bohr(115.); 
    case 54:  /* Xe */ return pm_to_bohr(108.); 
                                                                  
    case 55:  /* Cs */ return pm_to_bohr(298.); 
    case 56:  /* Ba */ return pm_to_bohr(253.); 
    case 57:  /* La */ return pm_to_bohr(622.); 
    case 58:  /* Ce */ return pm_to_bohr(505.); 
    case 59:  /* Pr */ return pm_to_bohr(247.); 
    case 60:  /* Nd */ return pm_to_bohr(206.); 
    case 61:  /* Pm */ return pm_to_bohr(205.); 
    case 62:  /* Sm */ return pm_to_bohr(238.); 
    case 63:  /* Eu */ return pm_to_bohr(231.); 
    case 64:  /* Gd */ return pm_to_bohr(233.); 
    case 65:  /* Tb */ return pm_to_bohr(225.); 
    case 66:  /* Dy */ return pm_to_bohr(228.); 
    case 67:  /* Ho */ return pm_to_bohr(226.); 
    case 68:  /* Er */ return pm_to_bohr(226.); 
    case 69:  /* Tm */ return pm_to_bohr(222.); 
    case 70:  /* Yb */ return pm_to_bohr(222.); 
    case 71:  /* Lu */ return pm_to_bohr(217.); 
    case 72:  /* Hf */ return pm_to_bohr(208.); 
    case 73:  /* Ta */ return pm_to_bohr(200.); 
    case 74:  /* W  */ return pm_to_bohr(193.); 
    case 75:  /* Re */ return pm_to_bohr(188.); 
    case 76:  /* Os */ return pm_to_bohr(185.); 
    case 77:  /* Ir */ return pm_to_bohr(180.); 
    case 78:  /* Pt */ return pm_to_bohr(177.); 
    case 79:  /* Au */ return pm_to_bohr(174.); 
    case 80:  /* Hg */ return pm_to_bohr(171.); 
    case 81:  /* Tl */ return pm_to_bohr(156.); 
    case 82:  /* Pb */ return pm_to_bohr(154.); 
    case 83:  /* Bi */ return pm_to_bohr(143.);
    case 84:  /* Po */ return pm_to_bohr(135.); 
    case 85:  /* At */ return pm_to_bohr(127.); 
    case 86:  /* Rn */ return pm_to_bohr(120.);

    default: return -1.;
  }

}

// UFF atomic radii
// Atomic radii derived from the universal force field
// A. K. Rappe et. al. J. Am. Chem. Soc., 1992, 114 (25), pp 10024-10035
// https://doi.org/10.1021/ja00051a040, data given in Angström,
// will be converted to Bohr. Note that keys are normalised to lower case.
const std::vector<double> radius_uff_list = {1.443, 1.81, 1.2255, 1.3725, 2.0415, 1.9255, 1.83, 1.75, 
                            1.682, 1.6215, 1.4915, 1.5105, 
                            2.2495, 2.1475, 2.0735, 2.0175, 1.9735, 1.934, 1.906, 1.6995, 1.6475, 
                            1.5875, 1.572, 1.5115, 1.4805, 1.456, 1.436, 1.417, 1.7475, 
                            1.3815, 2.1915, 2.14, 2.115, 2.1025, 2.0945, 2.0705, 2.057, 
                            1.8205, 1.6725, 1.562, 1.5825, 1.526, 1.499, 1.4815, 1.4645, 
                            1.4495, 1.574, 1.424, 2.2315, 2.196, 2.21, 2.235, 2.25, 2.202, 
                            2.2585, 1.8515, 1.761, 1.778, 1.803, 1.7875, 1.7735, 1.76, 1.7465, 
                            1.684, 1.7255, 1.714, 1.7045, 1.6955, 1.687, 1.6775, 1.82, 1.5705, 
                            1.585, 1.5345, 1.477, 1.56, 1.42, 1.377, 1.6465, 1.3525, 2.1735, 2.1485, 
                            2.185, 2.3545, 2.375, 2.3825, 2.45, 1.8385, 1.739, 1.698, 1.712, 1.6975, 
                            1.712, 1.712, 1.6905, 1.663, 1.6695, 1.6565, 1.6495, 1.643, 1.637, 1.624, 1.618};

double uff_radius_103(AtomicNumber _Z) {
    const double RADIUS_UFF_SCALING = 1.1;
    const double DDX_BOHR_TO_ANGSTROM = 0.52917721092;
    auto Z = _Z.get();
    if (Z < 0 || Z >= radius_uff_list.size()) {
        return -1.;
    }
    return radius_uff_list[Z-1] * RADIUS_UFF_SCALING / DDX_BOHR_TO_ANGSTROM;
}
}
