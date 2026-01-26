from pyscf import gto

from skala.gauxc.export import write_gauxc_h5_from_pyscf
from skala.pyscf import SkalaRKS

mol = gto.M(atom="He 0 0 0", basis="def2-svp", unit="Bohr", spin=0)
ks = SkalaRKS(mol, xc="pbe")
ks.kernel()

dm = ks.make_rdm1()
exc = ks.scf_summary["exc"]
_, _, vxc = ks._numint.nr_rks(ks.mol, ks.grids, ks.xc, dm)

write_gauxc_h5_from_pyscf("He_def2-svp.h5", mol, dm=dm, exc=exc, vxc=vxc)
