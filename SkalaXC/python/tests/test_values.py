import pytest
import skalaxc


def test_molecule_items_are_independent_values() -> None:
    molecule = skalaxc.Molecule()
    molecule.append(skalaxc.Atom(1, 1.0, 2.0, 3.0))
    atom = molecule[0]
    atom.x = 4.0
    assert molecule[0].x == 1.0

    molecule[0] = atom
    atom.x = 5.0
    assert molecule[0].x == 4.0
    for _ in range(128):
        molecule.append(skalaxc.Atom(2, 0.0, 0.0, 0.0))
    assert (atom.atomic_number, atom.x, atom.y, atom.z) == (1, 5.0, 2.0, 3.0)
    del molecule
    assert atom.x == 5.0


def test_basis_items_are_independent_values() -> None:
    basis = skalaxc.BasisSet()
    basis.append(skalaxc.Shell(0, True, [1.0], [0.5], [0.0, 1.0, 2.0]))
    shell = basis[0]
    basis[0] = skalaxc.Shell(1, True, [2.0], [0.75], [3.0, 4.0, 5.0])
    assert shell.angular_momentum == 0
    assert basis[0].angular_momentum == 1
    for _ in range(128):
        basis.append(shell)
    del basis
    assert shell.nprim == 1
    assert shell.exponents == [1.0]
    assert shell.coefficients == [0.5]
    assert shell.center == [0.0, 1.0, 2.0]


def test_container_index_bounds() -> None:
    molecule = skalaxc.Molecule()
    basis = skalaxc.BasisSet()
    atom = skalaxc.Atom(1, 0.0, 0.0, 0.0)
    shell = skalaxc.Shell(0, True, [1.0], [1.0], [0.0, 0.0, 0.0])
    with pytest.raises(IndexError):
        _ = molecule[0]
    with pytest.raises(IndexError):
        molecule[0] = atom
    with pytest.raises(IndexError):
        _ = basis[0]
    with pytest.raises(IndexError):
        basis[0] = shell
    molecule.append(atom)
    basis.append(shell)
    with pytest.raises(IndexError):
        _ = molecule[1]
    with pytest.raises(IndexError):
        molecule[1] = atom
    with pytest.raises(IndexError):
        _ = basis[1]
    with pytest.raises(IndexError):
        basis[1] = shell
