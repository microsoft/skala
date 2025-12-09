import h5py
import numpy as np

from pyscf import gto

MOLECULE_DTYPE = {
    "names": ["Atomic Number", "X Coordinate", "Y Coordinate", "Z Coordinate"],
    "formats": ["<i4", "<f8", "<f8", "<f8"],
    "offsets": [0, 8, 16, 24],
    "itemsize": 32,
}
BASIS_DTYPE = {
    "names": ["NPRIM", "L", "PURE", "ALPHA", "COEFF", "ORIGIN"],
    "formats": ["<i4", "<i4", "<i4", ("<f8", (16,)), ("<f8", (16,)), ("<f8", (3,))],
    "offsets": [0, 4, 8, 16, 272, 528],
    "itemsize": 552,
}
SQRT_PI_CUBED = 5.56832799683170784528481798212
K_MINUS_1 = [
    1,
    1,
    1,
    2,
    3,
    8,
    15,
    48,
    105,
    384,
    945,
    3840,
    10395,
    46080,
    135135,
    645120,
    2027025,
    10321920,
    34459425,
    185794560,
    654729075,
    3715891200,
    13749310575,
    81749606400,
    316234143225,
    1961990553600,
    7905853580625,
    51011754393600,
    213458046676875,
    1428329123020800,
    6190283353629375,
    42849873690624000,
    191898783962510625,
    1371195958099968000,
    6332659870762850625,
]


def write_gauxc_h5_from_pyscf(filename: str, mol: gto.Mole, dm: np.ndarray) -> None:
    data = pyscf_to_gauxc_h5(mol, dm)
    with h5py.File(filename, "w") as fd:
        for key, value in data.items():
            fd.create_dataset(key, data=value)


def pyscf_to_gauxc_h5(mol: gto.Mole, dm: np.ndarray) -> dict[str, np.ndarray]:
    molecule = np.array(
        [
            (number, *coords)
            for number, coords in zip(mol.atom_charges(), mol.atom_coords(unit="Bohr"), strict=True)
        ],
        dtype=MOLECULE_DTYPE,
    )
    basis = np.array(
        [
            format_basis(
                func[0],
                mol.cart,
                [pair[0] for pair in func[1:]],
                [pair[prim] for pair in func[1:]],
                coord,
            )
            for atom, coord in mol._atom
            for func in mol._basis[atom]
            for prim in range(1, len(func[1]))
        ],
        dtype=BASIS_DTYPE,
    )
    dm_scalar = dm if dm.ndim == 2 else dm[0] + dm[1]
    dm_z = np.zeros_like(dm) if dm.ndim == 2 else dm[0] - dm[1]

    return {
        "MOLECULE": molecule,
        "BASIS": basis,
        "DENSITY_SCALAR": dm_scalar,
        "DENSITY_Z": dm_z,
    }


def norm(coeff: list[float], alpha: list[float], l: int) -> list[float]:
    """
    Normalize contraction coefficients for a given angular momentum and exponents
    using libint normalization conventions.
    """
    alpha = np.asarray(alpha)
    two_alpha = 2 * alpha
    two_alpha_to_am32 = two_alpha ** (l + 1) * np.sqrt(two_alpha)
    normalization_factor = np.sqrt(2**l * two_alpha_to_am32 / (SQRT_PI_CUBED * K_MINUS_1[2 * l]))
    gamma = alpha[:, np.newaxis] + alpha[np.newaxis, :]
    aa = K_MINUS_1[2 * l] * SQRT_PI_CUBED / (2**l * gamma ** (l + 1) * np.sqrt(gamma))
    coeff = coeff * normalization_factor
    normalization_factor = 1.0 / np.sqrt(np.einsum("i,j,ij->", coeff, coeff, aa))
    return (coeff * normalization_factor).tolist()


def format_basis(
    l: int,
    cart: bool,
    alpha: list[float],
    coeff: list[float],
    coord: list[float],
    padv: float = 0.0,
    padl: int = 16,
) -> tuple[int, int, int, list[float], list[float], list[float]]:
    return (
        len(alpha),
        l,
        0 if cart or l == 1 else 1,
        alpha + [padv] * (padl - len(alpha)),
        norm(coeff, alpha, l) + [padv] * (padl - len(coeff)),
        coord,
    )
