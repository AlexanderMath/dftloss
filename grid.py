# alternate implementation of pyscf grid with a few modifications; slightly faster. 
# could just us pycsf version, but this is slightly faster. 
from pyscf.data.elements import charge as elements_proton
from pyscf.dft import gen_grid, radi
import numpy as np 
import jax.numpy as jnp 
import time 

def treutler_atomic_radii_adjust(mol, atomic_radii):
  charges = [elements_proton(x) for x in mol.elements]
  rad = np.sqrt(atomic_radii[charges]) + 1e-200
  rr = rad.reshape(-1, 1) * (1. / rad)
  a = .25 * (rr.T - rr)

  a[a < -0.5] = -0.5
  a[a > 0.5]  = 0.5
  a = jnp.array(a)

  def fadjust(i, j, g):
    g1 = g**2
    g1 -= 1.
    g1 *= -a[i, j]
    g1 += g
    return g1

  return fadjust


def inter_distance(coords):
  rr = np.linalg.norm(coords.reshape(-1, 1, 3) - coords, axis=2)
  rr[np.diag_indices(rr.shape[0])] = 0 
  return rr 

def original_becke(g):
  g = (3 - g**2) * g * .5
  g = (3 - g**2) * g * .5
  g = (3 - g**2) * g * .5
  return g

def gen_grid_partition(coords, atom_coords, natm, atm_dist, elements, 
                       atomic_radii,  becke_scheme=original_becke,):
    ngrids = coords.shape[0]
    dc = coords[None] - atom_coords[:, None]
    grid_dist = np.sqrt(np.einsum('ijk,ijk->ij', dc, dc))  

    ix, jx = np.tril_indices(natm, k=-1)

    natm, ngrid = grid_dist.shape 
    g_ = -1 / (atm_dist.reshape(natm, natm, 1) + np.eye(natm).reshape(natm, natm,1)) * (grid_dist.reshape(1, natm, ngrid) - grid_dist.reshape(natm, 1, ngrid))

    def pbecke_g(i, j):
      g = g_[i, j]
      charges = [elements_proton(x) for x in elements]
      rad = np.sqrt(atomic_radii[charges]) + 1e-200
      rr = rad.reshape(-1, 1) * (1. / rad)
      a = .25 * (rr.T - rr)
      a[a < -0.5] = -0.5
      a[a > 0.5]  = 0.5
      g1 = g**2
      g1 -= 1.
      g1 *= -a[i, j].reshape(-1, 1)
      g1 += g
      return g1

    g = pbecke_g(ix, jx)
    g = np.copy(becke_scheme(g))
    gp2 = (1+g)/2
    gm2 = (1-g)/2

    t0 = time.time()
    pbecke = np.ones((natm, ngrids))  
    c = 0 
    for i in range(natm): 
        for j in range(i): 
            pbecke[i] *= gm2[c]
            pbecke[j] *= gp2[c]
            c += 1
    return pbecke


def get_partition(
  mol,
  atom_coords,
  atom_grids_tab,
  radii_adjust=treutler_atomic_radii_adjust,
  atomic_radii=radi.BRAGG_RADII,
  becke_scheme=original_becke,
  concat=True, state=None
):
  t0 = time.time()
  atm_dist = inter_distance(atom_coords)  # [natom, natom]

  coords_all = []
  weights_all = []

  for ia in range(mol.natm):
    coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
    coords = coords + atom_coords[ia]  # [ngrid, 3]
    pbecke  = gen_grid_partition(coords, atom_coords, mol.natm, atm_dist, mol.elements, atomic_radii)  # [natom, ngrid]
    weights = vol * pbecke[ia] / np.sum(pbecke, axis=0) 
    coords_all.append(coords)
    weights_all.append(weights)

  if concat:
    coords_all = np.vstack(coords_all)
    weights_all = np.hstack(weights_all)

  coords = (coords_all, weights_all)
  return coords_all, weights_all


class DifferentiableGrids(gen_grid.Grids):
  """Differentiable alternative to the original pyscf.gen_grid.Grids."""

  def build(self, atom_coords, state=None) :
    t0 = time.time()
    mol = self.mol

    atom_grids_tab = self.gen_atomic_grids(
      mol, self.atom_grid, self.radi_method, self.level, 
      self.prune, 
      #False, # WARNING: disables self.prune; allows changing C->O and F->N in same compute graph, but makes sizes of everythign larger/slower. 
    )

    coords, weights = get_partition(
      mol,
      atom_coords,
      atom_grids_tab,
      treutler_atomic_radii_adjust,
       self.atomic_radii,
      original_becke,
      state=state,
    )

    self.coords = coords
    self.weights = weights 
    return coords, weights
