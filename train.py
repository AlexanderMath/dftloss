import os
os.environ['OMP_NUM_THREADS'] = '29'
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pyscf
import chex
import optax
from tqdm import tqdm 
import time 
from transformer import transformer, transformer_init
import math 
from functools import partial 
import pickle 
import random 
from sparse_symmetric_ERI import sparse_symmetric_einsum
random.seed(42)
np.random.seed(42)

unit = "angstrom"  # pdf files are angstrom
#unit = "bohr"
cfg, HARTREE_TO_EV, EPSILON_B3LYP, HYB_B3LYP = None, 27.2114079527, 1e-20, 0.2
angstrom_to_bohr, bohr_to_angstrom, nm_to_bohr, bohr_to_nm = 1.8897, 0.529177, 18.8973, 0.0529177249
B, BxNxN, BxNxK = None, None, None

# Only need to recompute: L_inv, grid_AO, grid_weights, H_core, ERI and E_nuc. 
def dm_energy(W, state, nn, cfg=None, opts=None): 
    # Cast to f32. 
    if opts.nn_f32: state.pos, state.H_core, state.L_inv = state.pos.astype(jnp.float32), state.H_core.astype(jnp.float32), state.L_inv.astype(jnp.float32)

    # Convert initial guess "dm_init" into "H_init" (density matrix -> Hamiltonian). 
    density_matrix = state.dm_init
    B,N, _ = density_matrix.shape
    diff_JK    = JK(density_matrix, state, opts.foriloop, opts.eri_f32, opts.bs)        
    E_xc, V_xc = jax.vmap(explicit_exchange_correlation)( density_matrix, state.grid_AO, state.grid_weights)
    H          = state.H_core + diff_JK + V_xc  

    # Add neural network correection to "H_init". 
    nn = jax.vmap(transformer, in_axes=(None, None, 0, 0, 0, 0,0,0,0,0), out_axes=(0))
    pred = nn(cfg, W, state.ao_types, state.pos, state.H_core, state.L_inv, state.dm_init, diff_JK, V_xc.reshape(B, N,N), H).astype(jnp.float64)
    H = H + pred/10

    # Given H compute new (dm, H) 
    L_inv_Q        = state.L_inv_T @ jnp.linalg.eigh(state.L_inv @ H @ state.L_inv_T)[1]   
    density_matrix = 2 * (L_inv_Q*state.mask) @ jnp.transpose(L_inv_Q, (0,2,1)) 
    diff_JK        = JK(density_matrix, state, opts.foriloop, opts.eri_f32, opts.bs)        
    E_xc, V_xc     = jax.vmap(explicit_exchange_correlation)( density_matrix, state.grid_AO, state.grid_weights)
    H              = state.H_core + diff_JK + V_xc  

    # density mixing (aka residual connection) 
    '''L_inv_Q        = state.L_inv_T @ jnp.linalg.eigh(state.L_inv @ H @ state.L_inv_T)[1]   
    density_matrix = (density_matrix + 2 * (L_inv_Q*state.mask) @ jnp.transpose(L_inv_Q, (0,2,1)) )/2
    diff_JK        = JK(density_matrix, state, opts.foriloop, opts.eri_f32, opts.bs)        
    E_xc, V_xc     = jax.vmap(explicit_exchange_correlation)( density_matrix, state.grid_AO, state.grid_weights)
    H              = state.H_core + diff_JK + V_xc  '''

    # fix idempotency dm@dm=dm. 
    '''L_inv_Q        = state.L_inv_T @ jnp.linalg.eigh(state.L_inv @ H @ state.L_inv_T)[1]   
    density_matrix = 2 * (L_inv_Q*state.mask) @ jnp.transpose(L_inv_Q, (0,2,1)) 
    diff_JK        = JK(density_matrix, state, opts.foriloop, opts.eri_f32, opts.bs)        
    E_xc, V_xc     = jax.vmap(explicit_exchange_correlation)( density_matrix, state.grid_AO, state.grid_weights)
    H              = state.H_core + diff_JK + V_xc '''

    energies       = E_xc + state.E_nuc + jnp.sum((density_matrix * (state.H_core + diff_JK/2)).reshape(density_matrix.shape[0], -1), axis=-1) 
    energy         = jnp.sum(energies)  
    loss           = energy 
    return loss, (energies, (energy, 0, 0, 0, 0), E_xc, density_matrix, W, H)

from exchange_correlation.b3lyp import vxc_b3lyp as b3lyp
def explicit_exchange_correlation(density_matrix, grid_AO, grid_weights):
    grid_AO_dm = grid_AO[0] @ density_matrix                                                    
    grid_AO_dm = jnp.expand_dims(grid_AO_dm, axis=0)                                            
    mult = grid_AO_dm * grid_AO  
    rho = jnp.sum(mult, axis=2)                                                                 
    E_xc, vrho, vgamma = b3lyp(rho, EPSILON_B3LYP)                                              
    E_xc = jnp.sum(rho[0] * grid_weights * E_xc)                                                
    rho = jnp.concatenate([vrho.reshape(1, -1)/2, 4*vgamma*rho[1:4]], axis=0) * grid_weights    
    grid_AO_T = grid_AO[0].T                                                                    
    rho = jnp.expand_dims(rho, axis=2)                                                          
    grid_AO_rho = grid_AO * rho                                                                 
    sum_grid_AO_rho = jnp.sum(grid_AO_rho, axis=0)                                              
    V_xc = grid_AO_T @ sum_grid_AO_rho                                                          
    V_xc = V_xc + V_xc.T                                                                        
    return E_xc, V_xc   


def JK(density_matrix, state, jax_foriloop, eri_f32, bs): 
    B, N, N = density_matrix.shape
    if eri_f32: density_matrix = density_matrix.astype(jnp.float32)
    diff_JK = jax.vmap(sparse_symmetric_einsum, in_axes=(None, None, 0, None))(
        state.nonzero_distinct_ERI[0].astype(density_matrix.dtype), 
        state.nonzero_indices[0], 
        density_matrix, 
        jax_foriloop
    )
    if B == 1: return diff_JK
    else: return diff_JK - jax.vmap(sparse_symmetric_einsum, in_axes=(0, None, 0, None))(\
        state.diffs_ERI, 
        state.indxs, 
        density_matrix, 
        jax_foriloop).astype(jnp.float64)

def nao(atom, basis):
    m = pyscf.gto.Mole(atom='%s 0 0 0; %s 0 0 1;'%(atom, atom), basis=basis, unit=unit)
    m.build()
    return m.nao_nr()//2

def batched_state(mol_str, opts, bs, wiggle_num=0, 
                  do_pyscf=True, validation=False, 
                  extrapolate=False,
                  pad_electrons=45, 
                  pad_diff_ERIs=50000,
                  pad_distinct_ERIs=120000,
                  pad_grid_AO=25000,
                  pad_nonzero_distinct_ERI=200000,
                  pad_sparse_diff_grid=200000, 
                  mol_idx=42,
                  train=True, input_angles=None,
                  ): 
    # Set seed to ensure different rotation. 
    np.random.seed(mol_idx)
    train = not validation

    start_time = time.time()
    do_print = opts.do_print 
    if do_print: print("\t[%.4fs] start of 'batched_state'. "%(time.time()-start_time))
    max_pad_electrons, max_pad_diff_ERIs, max_pad_distinct_ERIs, max_pad_grid_AO, max_pad_nonzero_distinct_ERI, max_pad_sparse_diff_grid = \
        -1, -1, -1, -1, -1, -1

    if opts.alanine: 
        pad_electrons = 70
        padding_estimate = [ 100000 ,150000    ,176370     ]
        padding_estimate = [int(a*1.05) for a in padding_estimate]
        pad_diff_ERIs, pad_distinct_ERIs, pad_nonzero_distinct_ERI= [int(a*8/opts.eri_bs) for a in padding_estimate]

    mol = build_mol(mol_str, opts.basis)
    pad_electrons = min(pad_electrons, mol.nao_nr())
        
    if opts.alanine:
        phi, psi = [float(a) for a in np.random.uniform(0, opts.rotate_deg, 2)]

    states = []
    import copy 
    for iteration in range(bs):
        new_str = copy.deepcopy(mol_str)

        if do_print: print("\t[%.4fs] initializing state %i. "%(time.time()-start_time, iteration))
        if opts.alanine: #  and train: 
            from rdkit import Chem
            from rdkit.Chem import AllChem
            pdb_file = 'alanine.pdb'
            molecule = Chem.MolFromPDBFile(pdb_file, removeHs=False)
            phi_atoms = [4, 6, 8, 14]  # indices for phi dihedral
            psi_atoms = [6, 8, 14, 16]  # indices for psi dihedral

            def xyz(atom): return np.array([atom.x, atom.y, atom.z]).reshape(1, 3)
            def get_atom_positions(mol):
                conf = mol.GetConformer()
                return np.concatenate([xyz(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], axis=0)

            AllChem.SetDihedralDeg(molecule.GetConformer(), *phi_atoms, phi)
            angle = psi + float(np.random.uniform(0, opts.rotate_deg, 1)) 
            angle = angle % opts.rotate_deg 

            AllChem.SetDihedralDeg(molecule.GetConformer(), *psi_atoms, angle )
            pos = get_atom_positions(molecule)

            for j in range(len(new_str)): new_str[j][1] = tuple(pos[j])

        if iteration == 0: 
            state = init_dft(new_str, opts, do_pyscf=do_pyscf, pad_electrons=pad_electrons)
            c, w = state.grid_coords, state.grid_weights
        elif iteration <= 1 or not opts.prof:  # when profiling (for development only) create fake molecule to skip waiting on dataloader
            state = init_dft(new_str, opts, c, w, do_pyscf=do_pyscf and iteration < 3, state=state, pad_electrons=pad_electrons)
        states.append(state)

    state = cats(states)
    N = state.N[0]
    if do_print: print("\t[%.4fs] concatenated states. "%(time.time()-start_time))

    # Compute ERI sparsity. 
    nonzero = []
    for e, i in zip(state.nonzero_distinct_ERI, state.nonzero_indices):
        abs = np.abs(e)
        indxs = abs < opts.eri_threshold #1e-10 
        e[indxs] = 0 
        nonzero.append(np.nonzero(e)[0])

    if do_print: print("\t[%.4fs] got sparsity. "%(time.time()-start_time))

    # Merge nonzero indices and prepare (ij, kl).
    # rep is the number of repetitions we include in the sparse representation. 
    union = nonzero[0]
    for i in range(1, len(nonzero)): 
        union = np.union1d(union, nonzero[i])
    nonzero_indices = union.astype(np.uint32)
    if do_print: print("\t[%.4fs] got union of sparsity. "%(time.time()-start_time))

    from sparse_symmetric_ERI import get_i_j, num_repetitions_fast
    ij, kl               = get_i_j(nonzero_indices)
    if do_print: print("\t[%.4fs] got (ij,kl). "%(time.time()-start_time)) 
    rep= num_repetitions_fast(ij.reshape(-1), kl.reshape(-1))
    if do_print: print("\t[%.4fs] got reps. "%(time.time()-start_time)) 

    batches = opts.eri_bs
    es = []
    for e,_ in zip(state.nonzero_distinct_ERI, state.nonzero_indices):
        nonzero_distinct_ERI = e[nonzero_indices] / rep
        remainder            = nonzero_indices.shape[0] % (batches)
        if remainder != 0: nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,batches-remainder))

        nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(batches, -1)
        es.append(nonzero_distinct_ERI)

    state.nonzero_distinct_ERI = np.concatenate([np.expand_dims(a, axis=0) for a in es])

    if do_print: print("\t[%.4fs] padded ERI and nonzero_indices. . "%(time.time()-start_time))
    i, j = get_i_j(ij.reshape(-1))
    k, l = get_i_j(kl.reshape(-1))
    if do_print: print("\t[%.4fs] got ijkl. "%(time.time()-start_time))

    if remainder != 0:
        i = np.pad(i, ((0,batches-remainder)))
        j = np.pad(j, ((0,batches-remainder)))
        k = np.pad(k, ((0,batches-remainder)))
        l = np.pad(l, ((0,batches-remainder)))
    nonzero_indices = np.vstack([i,j,k,l]).T.reshape(batches, -1, 4).astype(np.int32) 
    state.nonzero_indices = nonzero_indices  
    if do_print: print("\t[%.4fs] padded and vstacked ijkl. "%(time.time()-start_time))

    # use the same sparsity pattern across a batch.
    #print(train, opts.bs, state.nonzero_distinct_ERI.shape)
    if train and opts.bs > 1 and state.nonzero_distinct_ERI.shape[0] > 1: 
        diff_ERIs  = state.nonzero_distinct_ERI[:1] - state.nonzero_distinct_ERI
        diff_indxs = state.nonzero_indices.reshape(1, batches, -1, 4)
        #nzr        = np.abs(diff_ERIs[1]).reshape(batches, -1) > 1e-10
        nzr        = np.abs(diff_ERIs[1]).reshape(batches, -1) > opts.eri_threshold 

        diff_ERIs  = diff_ERIs[:, nzr].reshape(bs, -1)
        diff_indxs = diff_indxs[:, nzr].reshape(-1, 4)

        remainder = np.sum(nzr) % batches
        if remainder != 0:
            diff_ERIs = np.pad(diff_ERIs, ((0,0),(0,batches-remainder)))
            diff_indxs = np.pad(diff_indxs, ((0,batches-remainder),(0,0)))

        diff_ERIs = diff_ERIs.reshape(bs, batches, -1)
        diff_indxs = diff_indxs.reshape(batches, -1, 4)
        # this is the key savings resulting from batching! 
        if do_print: print(diff_ERIs.shape, diff_indxs.shape, state.nonzero_distinct_ERI.shape)

        if pad_diff_ERIs == -1: 
            state.indxs=diff_indxs
            state.diffs_ERI=diff_ERIs
            assert False, "deal with precomputed_indxs; only added in else branch below"
        else: 
            max_pad_diff_ERIs = diff_ERIs.shape[2]
            if do_print: print("\t[%.4fs] max_pad_diff_ERIs=%i"%(time.time()-start_time, max_pad_diff_ERIs))
            # pad ERIs with 0 and indices with -1 so they point to 0. 
            assert diff_indxs.shape[1] == diff_ERIs.shape[2]
            pad = pad_diff_ERIs - diff_indxs.shape[1]
            assert pad > 0, (pad_diff_ERIs, diff_indxs.shape[1])
            state.indxs     = np.pad(diff_indxs, ((0,0), (0, pad), (0, 0)), 'constant', constant_values=(-1))
            state.diffs_ERI = np.pad(diff_ERIs,  ((0,0), (0, 0),   (0, pad))) # pad zeros 

    #state.grid_AO = state.grid_AO[:1]
    state.nonzero_distinct_ERI = state.nonzero_distinct_ERI[:1]
    state.nonzero_indices = np.expand_dims(state.nonzero_indices, axis=0)

    if pad_distinct_ERIs != -1: 
        max_pad_distinct_ERIs = state.nonzero_distinct_ERI.shape[2]
        if do_print: print("\t[%.4fs] max_pad_distinct_ERIs=%i"%(time.time()-start_time, max_pad_diff_ERIs))
        assert state.nonzero_distinct_ERI.shape[2] == state.nonzero_indices.shape[2]
        pad = pad_distinct_ERIs - state.nonzero_distinct_ERI.shape[2]
        assert pad > 0, (pad_distinct_ERIs, state.nonzero_distinct_ERI.shape[2])
        state.nonzero_indices      = np.pad(state.nonzero_indices,      ((0,0), (0,0), (0, pad), (0,0)), 'constant', constant_values=(-1))
        state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, ((0,0), (0,0),  (0, pad))) # pad zeros 

    indxs = np.abs(state.nonzero_distinct_ERI ) > opts.eri_threshold #1e-9 
    state.nonzero_distinct_ERI = state.nonzero_distinct_ERI[indxs]
    state.nonzero_indices = state.nonzero_indices[indxs]
    remainder = state.nonzero_indices.shape[0] % batches

    if remainder != 0:
        state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, (0,batches-remainder))
        state.nonzero_indices = np.pad(state.nonzero_indices, ((0,batches-remainder), (0,0)))
    state.nonzero_distinct_ERI = state.nonzero_distinct_ERI.reshape(1, batches, -1)
    state.nonzero_indices = state.nonzero_indices.reshape(1, batches, -1, 4)

    if pad_nonzero_distinct_ERI != -1: 
        max_pad_nonzero_distinct_ERI = state.nonzero_distinct_ERI.shape[2]
        if do_print: print("\t[%.4fs] max_pad_nonzero_distinct_ERI=%i"%(time.time()-start_time, max_pad_nonzero_distinct_ERI))

        assert state.nonzero_distinct_ERI.shape[2] == state.nonzero_indices.shape[2]
        pad = pad_nonzero_distinct_ERI - state.nonzero_distinct_ERI.shape[2]
        assert pad >= 0, (pad_nonzero_distinct_ERI, state.nonzero_distinct_ERI.shape[2])
        state.nonzero_distinct_ERI = np.pad(state.nonzero_distinct_ERI, ((0,0),(0,0),(0,pad)))
        state.nonzero_indices = np.pad(state.nonzero_indices, ((0,0),(0,0),(0,pad), (0,0)), 'constant', constant_values=(-1))

    B = state.grid_AO.shape[0]
    state.pad_sizes = np.concatenate([np.array([
        max_pad_diff_ERIs, max_pad_distinct_ERIs, max_pad_grid_AO, 
        max_pad_nonzero_distinct_ERI, max_pad_sparse_diff_grid]).reshape(1, -1) for _ in range(B)])

    if opts.eri_f32: 
        state.nonzero_distinct_ERI = state.nonzero_distinct_ERI.astype(jnp.float32)
        state.diffs_ERI = state.diffs_ERI.astype(jnp.float32)

    if opts.xc_f32: 
        state.main_grid_AO = state.main_grid_AO.astype(jnp.float32)
        state.grid_AO = state.grid_AO.astype(jnp.float32)

    return state 

def train_QPT(mol_str, opts):
    start_time = time.time()
    print()
    if opts.wandb: 
        import wandb 
        run = wandb.init(project='dftloss')
        opts.name = run.name
        wandb.log(vars(opts))
    else:
        opts.name = "%i"%time.time()

    rnd_key = jax.random.PRNGKey(42)
    n_vocab = nao("C", opts.basis) + nao("N", opts.basis) + \
              nao("O", opts.basis) + nao("F", opts.basis) + \
              nao("H", opts.basis)  

    global cfg
    if opts.tiny:  # 5M 
        d_model= 192
        n_heads = 6
        n_layers = 12
    if opts.small:
        d_model= 384
        n_heads = 6
        n_layers = 12
    if opts.base: 
        d_model= 768
        n_heads = 12
        n_layers = 12
    if opts.medium: 
        d_model= 1024
        n_heads = 16
        n_layers = 24
    if opts.large:  # this is 600M; 
        d_model= 1280 
        n_heads = 16
        n_layers = 36
    if opts.largep:  # interpolated between large and largep. 
        d_model= 91*16 # halway from 80 to 100 
        n_heads = 16*1 # this is 1.3B; decrease parameter count 30%. 
        n_layers = 43
    if opts.xlarge:  
        d_model= 1600 
        n_heads = 25 
        n_layers = 48

    if opts.nn: 
        rnd_key, cfg, params, total_params = transformer_init(
            rnd_key,
            n_vocab,
            d_model =d_model,
            n_layers=n_layers,
            n_heads =n_heads,
            d_ff    =d_model*4,
        )
        print("[%.4fs] initialized transformer. "%(time.time()-start_time) )
        if opts.nn_f32: params = params.to_float32()

        from natsort import natsorted 
        if opts.resume: 
            all = os.listdir("checkpoints")
            candidates = natsorted([a for a in all if opts.resume in a])
            print(candidates)
            print("found candidates", candidates)

            to_load = candidates[-1].replace("_model.pickle", "").replace("_adam_state.pickle", "")
            print("choose candidate ", to_load)
            opts.resume = to_load 

        if opts.resume: 
            print("loading checkpoint")
            params = pickle.load(open("checkpoints/%s_model.pickle"%opts.resume, "rb"))
            if opts.nn_f32: params = params.to_float32()
            else:  params = params.to_float64()
            print("done loading. ")

    if opts.nn: 
        #https://arxiv.org/pdf/1706.03762.pdf see 5.3 optimizer 
        def custom_schedule(it, learning_rate=opts.lr, min_lr=opts.min_lr, warmup_iters=opts.warmup_iters, lr_decay_iters=opts.lr_decay): 
            cond1 = (it < warmup_iters) * learning_rate * it / warmup_iters
            cond2 = (it > lr_decay_iters) * min_lr
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio)) 
            cond3 = (it >= warmup_iters) * (it <= lr_decay_iters) * (min_lr + coeff * (learning_rate - min_lr))
            return cond1 + cond2 + cond3 
 
        adam = optax.chain(
            optax.clip_by_global_norm(1),
            optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-12),
            optax.add_decayed_weights(0.1),
            optax.scale_by_schedule(custom_schedule),
            optax.scale(-1),
        )
        w = params 

        print(jax.devices())

        from torch.utils.data import DataLoader, Dataset
        class OnTheFly(Dataset):
            # prepares dft tensors with pyscf "on the fly". 
            # dataloader is very keen on throwing segfaults (e.g. using jnp in dataloader throws segfaul). 
            #
            # problem: second epoch always gives segfault. 
            # hacky fix; make __len__ = real_length*num_epochs and __getitem__ do idx%real_num_examples 
            def __init__(self, opts, train=True, num_epochs=10**9, extrapolate=False):
                self.num_epochs = num_epochs
                self.opts = opts 
                self.validation = not train 
                self.extrapolate = extrapolate
                self.do_pyscf = self.validation or self.extrapolate

                self.train = train 
                self.mol_strs = mol_str

                self.H = [0 for _ in self.mol_strs] 
                self.E = [0 for _ in self.mol_strs]

                if train: self.bs = opts.bs 
                else: self.bs = opts.val_bs
                    
            def __len__(self): return len(self.mol_strs)*self.num_epochs

            def __getitem__(self, idx):
                return batched_state(self.mol_strs[idx%len(self.mol_strs)], self.opts, self.bs, \
                    wiggle_num=0, do_pyscf=self.do_pyscf, validation=False, \
                        extrapolate=self.extrapolate, mol_idx=idx, train=self.train), self.H[idx%len(self.mol_strs)], self.E[idx%len(self.mol_strs)]

        print("[%.4fs] initialized datasets. "%(time.time()-start_time) )
        valid = OnTheFly(opts, train=False)
        print("[%.4fs] initialized datasets. "%(time.time()-start_time) )

        if opts.precompute:   
            val_state = valid[0]
            exit()

        train = OnTheFly(opts, train=True)
        print("[%.4fs] initialized datasets. "%(time.time()-start_time) )
        if opts.workers != 0: train_dataloader = DataLoader(train, batch_size=1, pin_memory=True, shuffle=False, drop_last=True, num_workers=opts.workers, prefetch_factor=2, collate_fn=lambda x: x[0])
        else:                 train_dataloader = DataLoader(train, batch_size=1, pin_memory=True, shuffle=False, drop_last=True, num_workers=opts.workers,  collate_fn=lambda x: x[0])
        pbar = tqdm(train_dataloader)
        print("[%.4fs] initialized dataloaders. "%(time.time()-start_time) ) 

        if opts.test_dataloader:
            t0 = time.time()
            for iteration, (state, H, E) in enumerate(pbar):
                    if iteration == 0: summary(state) 
                    print(time.time()-t0)
                    t0 = time.time()
                    print(state.pad_sizes.reshape(1, -1))

            exit()

    vandg = jax.jit(jax.value_and_grad(dm_energy, has_aux=True), backend=opts.backend, static_argnames=('nn', "cfg", "opts"))
    valf = jax.jit(dm_energy, backend=opts.backend, static_argnames=('nn', "cfg", "opts"))
    adam_state = adam.init(w)
    print("[%.4fs] jitted vandg and valf."%(time.time()-start_time) )

    if opts.resume: 
        print("loading adam state")
        adam_state = pickle.load(open("checkpoints/%s_adam_state.pickle"%opts.resume, "rb"))
        print("done")

    w, adam_state = jax.device_put(w), jax.device_put(adam_state)
    print("[%.4fs] jax.device_put(w,adam_state)."%(time.time()-start_time) )

    @partial(jax.jit, backend=opts.backend)
    def update(w, adam_state, accumulated_grad):
        accumulated_grad = jax.tree_map(lambda x: x / opts.bs, accumulated_grad)
        updates, adam_state = adam.update(accumulated_grad, adam_state, w)
        w = optax.apply_updates(w, updates)
        return w, adam_state

    if opts.wandb: 
        if not opts.nn: total_params = -1 
        wandb.log({'total_params': total_params, 'batch_size': opts.bs, 'lr': opts.lr })

    min_val, min_dm, mins, valid_str, step, val_states, ext_state = 0, 0, np.ones(opts.bs)*1e6, "", 0, [], None
    t0, load_time, train_time, val_time, plot_time = time.time(), 0, 0, 0, 0

    paddings = []
    states   = []

    print("[%.4fs] first iteration."%(time.time()-start_time) )

    for iteration, (state, H, E) in enumerate(pbar):
        if iteration == 0: summary(state) 
        state = jax.device_put(state) 

        # Molecules have differ number of nonzero electron repulsion integrals. 
        # To not JIT 1000s different compute graphs we estimate max padding. 
        if iteration < 100 and opts.do_print: 
            paddings.append(state.pad_sizes.reshape(1, -1))
            _paddings = np.concatenate(paddings, axis=0)
            print(np.max(_paddings, 0))

        dct = {}
        dct["iteraton"] = iteration 

        states.append(state)
        if len(states) > opts.mol_repeats: states.pop(0)

        load_time, t0 = time.time()-t0, time.time()

        for j, state in enumerate(states):
            print(". ", end="", flush=True) 
            if j == 0: _t0 =time.time()
            (val, (vals, losses, E_xc, density_matrix, _W, _)), grad = vandg(w, state, opts.nn, cfg, opts)
            print(",", end="", flush=True)
            if j == 0: time_step1 = time.time()-_t0

            print("#", end="", flush=True)
            w, adam_state = update(w, adam_state, grad)
                    
            if opts.checkpoint != -1 and adam_state[1].count % opts.checkpoint == 0 and adam_state[1].count > 0:
                t0 = time.time()
                try: 
                    name = opts.name.replace("-", "_")
                    path_model = "checkpoints/%s_%i_model.pickle"%(name, iteration)
                    path_adam = "checkpoints/%s_%i_adam_state.pickle"%(name, iteration)
                    print("trying to checkpoint to %s and %s"%(path_model, path_adam))
                    pickle.dump(jax.device_get(w), open(path_model, "wb"))
                    pickle.dump(jax.device_get(adam_state), open(path_adam, "wb"))
                    print("done!")
                    print("\t-resume \"%s\""%(path_model.replace("_model.pickle", "")))
                except: 
                    print("fail!")
                    pass 
                print("tried saving model took %fs"%(time.time()-t0))  
                save_time, t0 = time.time()-t0, time.time()

        global_batch_size = len(states)*opts.bs
        if opts.wandb: dct["global_batch_size"] = global_batch_size
        if opts.wandb: 
            dct["energy"] = -losses[0]
            dct["norm_errvec"] = losses[1]
            dct["H_loss"] = losses[2]
            dct["E_loss"] = losses[3]
            dct["dm_loss"] = losses[4]

        train_time, t0 = time.time()-t0, time.time() 
        update_time, t0 = time.time()-t0, time.time() 

        pbar.set_description("train=%.4f"%(vals[0]*HARTREE_TO_EV) + "[eV] "+ valid_str + "time=%.1f %.1f %.1f %.1f %.1f %.1f"%(load_time, time_step1, train_time, update_time, val_time, plot_time))

        if opts.wandb:
            dct["time_load"]  = load_time 
            dct["time_step1"]  = time_step1
            dct["time_train"]  = train_time
            dct["time_val"]  = val_time 

            dct["train_E"] = np.abs(E*HARTREE_TO_EV)
            dct["train_E_pred"] = np.abs(vals[0]*HARTREE_TO_EV)

        step = adam_state[1].count

        plot_time, t0 = time.time()-t0, time.time() 

        if opts.nn and (iteration < 250 or iteration % 10 == 0): 
            lr = custom_schedule(step)
            dct["scheduled_lr"] = lr

            val_idxs = [0]
            for j, val_idx in enumerate(val_idxs):
                if len(val_states) < len(val_idxs): val_states.append(jax.device_put(valid[val_idx]))
                val_state, val_H, val_E = val_states[j]
                _, (valid_vals, losses, _, vdensity_matrix, vW, H) = valf(w, val_state, opts.nn, cfg, opts)

                pickle.dump([H, valid_vals[0]*HARTREE_TO_EV, val_E*HARTREE_TO_EV], open("visualize/%i.pkl"%step, "wb"))

                dct['val_E_%i'%val_idx] = np.abs(valid_vals[0]*HARTREE_TO_EV-val_E*HARTREE_TO_EV )

                for i in range(0, 1):
                    dct['valid_l%i_%i'%(val_idx, i) ] = np.abs(valid_vals[i]*HARTREE_TO_EV-val_state.pyscf_E[i])
                    dct['valid_E%i_%i'%(val_idx, i) ] = np.abs(valid_vals[i]*HARTREE_TO_EV)
                    dct['valid_pyscf%i_%i'%(val_idx, i) ] = np.abs(val_state.pyscf_E[i])
            
                valid_str =  "lr=%.3e"%lr 

                valid_str += " |pyscf_E - QPT_E|=%.5e[eV]"%(valid_vals[0]*HARTREE_TO_EV-val_state.pyscf_E[0])

                if opts.alanine: 
                    dct["valid_l0"] = np.abs(valid_vals[0]*HARTREE_TO_EV - val_state.pyscf_E[0])

        
        if opts.wandb: 
            dct["step"] = step 
            wandb.log(dct)
        val_time, t0 = time.time()-t0, time.time()
    


@chex.dataclass
class IterationState:
    mask: np.array
    init: np.array
    E_nuc: np.array
    L_inv: np.array
    L_inv_T: np.array
    H_core: np.array
    grid_AO: np.array
    grid_weights: np.array
    grid_coords: np.array
    pyscf_E: np.array
    N: int 
    ERI: np.array
    nonzero_distinct_ERI: list 
    nonzero_indices: list
    diffs_ERI: np.array
    main_grid_AO: np.array
    diffs_grid_AO: np.array
    indxs: np.array
    sparse_diffs_grid_AO: np.array
    rows: np.array
    cols: np.array
    pos: np.array
    ao_types: np.array
    pad_sizes: np.array
    precomputed_nonzero_indices: np.array
    precomputed_indxs: np.array
    forces: np.array
    O: np.array
    dm_init: np.array



def init_dft(mol_str, opts, _coords=None, _weights=None, first=False, do_pyscf=True, state=None, pad_electrons=-1):
    do_print = False 
    mol = build_mol(mol_str, opts.basis)
    if do_pyscf: pyscf_E, pyscf_hlgap, pyscf_forces = reference(mol_str, opts)
    else:        pyscf_E, pyscf_hlgap, pyscf_forces = np.zeros(1), np.zeros(1), np.zeros(1)

    N                = mol.nao_nr()                                 
    n_electrons_half = mol.nelectron//2                             
    E_nuc            = mol.energy_nuc()                             

    if do_print: print("grid", end="", flush=True)

    from pyscf import dft
    #grids            = pyscf.dft.gen_grid.Grids(mol)
    from grid import DifferentiableGrids
    grids            = DifferentiableGrids(mol)
    grids.level      = opts.level
    #grids.build()
    grids.build(np.concatenate([np.array(a[1]).reshape(1, 3) for a in mol._atom]), state=state)

    grid_weights    = grids.weights                                 
    grid_coords     = grids.coords
    coord_str       = 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1'
    grid_AO         = mol.eval_gto(coord_str, grids.coords, 4)      

    if do_print: print("int1e", end="", flush=True)

    kinetic         = mol.intor_symmetric('int1e_kin')              
    nuclear         = mol.intor_symmetric('int1e_nuc')              
    O               = mol.intor_symmetric('int1e_ovlp')             
    L               = np.linalg.cholesky(O)
    L_inv           = np.linalg.inv(L)          
    dm_init = pyscf.scf.hf.init_guess_by_minao(mol)

    if pad_electrons == -1: 
        init = np.eye(N)[:, :n_electrons_half] 
        mask = np.ones((1, n_electrons_half))
    else: 
        assert pad_electrons > n_electrons_half, (pad_electrons, n_electrons_half)
        init = np.eye(N)[:, :pad_electrons] 
        mask = np.zeros((1, pad_electrons))
        mask[:, :n_electrons_half] = 1

    # todo: rewrite int2e_sph to only recompute changing atomic orbitals (~ will be N times faster). 
    if do_print: print("int2e",end ="", flush=True)
    nonzero_distinct_ERI = mol.intor("int2e_sph", aosym="s8")
    ERI = np.zeros(1)
    if do_print: print(nonzero_distinct_ERI.shape, nonzero_distinct_ERI.nbytes/10**9)
        
    def e(x): return np.expand_dims(x, axis=0)

    n_C = nao('C', opts.basis)
    n_N = nao('N', opts.basis)
    n_O = nao('O', opts.basis)
    n_F = nao('F', opts.basis)
    n_H = nao('H', opts.basis)
    n_vocab = n_C + n_N + n_O + n_F + n_H
    start, stop = 0, n_C
    c = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_N
    n = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_O
    o = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_F
    f = list(range(n_vocab))[start:stop]
    start, stop = stop, stop+n_H
    h = list(range(n_vocab))[start:stop]
    types = []
    pos = []
    for a, p in mol_str:
        if a.lower() == 'h': 
            types += h
            pos += [np.array(p).reshape(1, -1)]*len(h)
        elif a.lower() == 'c': 
            types += c
            pos += [np.array(p).reshape(1, -1)]*len(c)
        elif a.lower() == 'n': 
            types += n
            pos += [np.array(p).reshape(1, -1)]*len(n)
        elif a.lower() == 'o': 
            types += o
            pos += [np.array(p).reshape(1, -1)]*len(o)
        elif a.lower() == 'f': 
            types += f
            pos += [np.array(p).reshape(1, -1)]*len(f)
        else: raise Exception()
    ao_types = np.array(types)
    pos = np.concatenate(pos)
    pad_sizes = np.zeros(1)


    state = IterationState(
        diffs_ERI = np.zeros((1,1)),
        main_grid_AO = np.zeros((1,1)),
        diffs_grid_AO = np.zeros((1,1)),
        indxs = np.zeros((1,1)),
        sparse_diffs_grid_AO = np.zeros((1,1)),
        rows = np.zeros((1,1)),
        cols = np.zeros((1,1)),
        pos=e(pos),
        ao_types=e(ao_types),
        init = e(init), 
        E_nuc=e(E_nuc), 
        ERI=e(ERI),  
        nonzero_distinct_ERI=[nonzero_distinct_ERI],
        nonzero_indices=[0],
        H_core=e(nuclear+kinetic),
        L_inv=e(L_inv), 
        L_inv_T = e(L_inv.T),
        grid_AO=e(grid_AO), 
        grid_weights=e(grid_weights), 
        grid_coords=e(grid_coords),
        pyscf_E=e(pyscf_E[-1:]), 
        N=e(mol.nao_nr()),
        mask=e(mask),
        pad_sizes=e(pad_sizes),
        precomputed_nonzero_indices=np.zeros((1,1)),
        precomputed_indxs=np.zeros((1,1)),
        forces=e(pyscf_forces),
        O = e(O),
        dm_init = e(dm_init),
    )

    return state


def summary(state): 
    if state is None: return 
    print("_"*100)
    total = 0
    for field_name, field_def in state.__dataclass_fields__.items():
        field_value = getattr(state, field_name)
        try: 
            print("%35s %24s %20s %20s"%(field_name,getattr(field_value, 'shape', None), getattr(field_value, "nbytes", None)/10**9, getattr(field_value, "dtype", None) ))
            total += getattr(field_value, "nbytes", None)/10**9

        except: 
            try: 
                print("%35s %25s %20s"%(field_name,getattr(field_value[0], 'shape', None), getattr(field_value[0], "nbytes", None)/10**9))
                total += getattr(field_value, "nbytes", None)/10**9
            except: 
                print("BROKE FOR ", field_name)
        

    print("%35s %25s %20s"%("-", "total", total))
    try:
        print(state.pyscf_E[:, -1])
    except:
        pass 
    print("_"*100)

def _cat(x,y,name):
    if "list" in str(type(x)):
        return x + y 
    else: 
        return np.concatenate([x,y])


def cat(dc1, dc2, axis=0):
    concatenated_fields = {
        field: _cat(getattr(dc1, field), getattr(dc2, field), field)
        for field in dc1.__annotations__
    }
    return IterationState(**concatenated_fields)

def _cats(xs):
    if "list" in str(type(xs[0])):
        return sum(xs, [])#x + y 
    else: 
        return np.concatenate(xs)


def cats(dcs):
    concatenated_fields = {
        field: _cats([getattr(dc, field) for dc in dcs])
        for field in dcs[0].__annotations__
    }
    return IterationState(**concatenated_fields)

def pyscf_reference(mol_str, opts):
    from pyscf import __config__
    mol = build_mol(mol_str, opts.basis)
    mol.max_cycle = 50 
    mf = pyscf.scf.RKS(mol)
    mf.xc = 'B3LYP5'
    mf.verbose = 0 # put this to 4 and it prints DFT options set here. 
    mf.diis_space = 8
    mf.grids.level = opts.level
    pyscf_energies = []
    pyscf_hlgaps = []
    lumo         = mol.nelectron//2
    homo         = lumo - 1
    t0 = time.time()
    def callback(envs):
        pyscf_energies.append(envs["e_tot"]*HARTREE_TO_EV)
        hl_gap_hartree = np.abs(envs["mo_energy"][homo] - envs["mo_energy"][lumo]) * HARTREE_TO_EV
        pyscf_hlgaps.append(hl_gap_hartree)
        print("PYSCF: ", pyscf_energies[-1], "[eV]", time.time()-t0)
    mf.callback = callback
    mf.kernel()
    print("")
    if opts.forces: 
        forces = mf.nuc_grad_method().kernel()
    else: forces = 0 
    return np.array(pyscf_energies), np.array(pyscf_hlgaps), np.array(forces)


def build_mol(mol_str, basis_name):
    mol = pyscf.gto.mole.Mole()
    mol.build(atom=mol_str, unit=unit, basis=basis_name, spin=0, verbose=0)
    return mol

def reference(mol_str, opts):
    import pickle 
    import hashlib 
    filename = "precomputed/%s.pkl"%hashlib.sha256((str(mol_str) + str(opts.basis) + str(opts.level) + unit + str(opts.forces)).encode('utf-8')).hexdigest() 
    print(filename)
    if not os.path.exists(filename):
        pyscf_E, pyscf_hlgap, pyscf_forces = pyscf_reference(mol_str, opts)
        with open(filename, "wb") as file: 
            pickle.dump([pyscf_E, pyscf_hlgap, pyscf_forces, unit], file)
    else: 
        pyscf_E, pyscf_hlgap, pyscf_forces, _ = pickle.load(open(filename, "rb"))
    return pyscf_E, pyscf_hlgap, pyscf_forces


if __name__ == "__main__":
    import os
    import argparse 

    parser = argparse.ArgumentParser()
    # DFT options 
    parser.add_argument('-basis',   type=str,   default="sto3g")  
    parser.add_argument('-level',   type=int,   default=3)

    # GD options 
    parser.add_argument('-backend', type=str,       default="cpu") 
    parser.add_argument('-lr',      type=str,     default="5e-4") 
    parser.add_argument('-min_lr',      type=str,     default="1e-7")
    parser.add_argument('-warmup_iters',      type=float,     default=1000)
    parser.add_argument('-lr_decay',      type=float,     default=200000)
    parser.add_argument('-steps',   type=int,       default=100000)
    parser.add_argument('-bs',      type=int,       default=1)
    parser.add_argument('-val_bs',      type=int,   default=1)
    parser.add_argument('-mol_repeats',  type=int,  default=6) # How many time to optimize wrt each molecule. 

    # energy computation speedups 
    parser.add_argument('-foriloop',  action="store_true") # whether to use jax.lax.foriloop for sparse_symmetric_eri (faster compile time but slower training. ) nice for iterating quickly
    parser.add_argument('-xc_f32',   action="store_true") 
    parser.add_argument('-eri_f32',  action="store_true") 
    parser.add_argument('-nn_f32',  action="store_true") 
    parser.add_argument('-eri_bs',  type=int, default=8) 

    parser.add_argument('-wandb',      action="store_true") 
    parser.add_argument('-prof',       action="store_true") 

    # dataset 
    parser.add_argument('-benzene',      action="store_true") 
    parser.add_argument('-alanine',      action="store_true") 
    parser.add_argument('-do_print',     action="store_true")  # useful for debugging. 
    parser.add_argument('-states',       type=int,   default=1)
    parser.add_argument('-workers',      type=int,   default=5) 
    parser.add_argument('-precompute',   action="store_true")  # precompute labels; only run once for data{set/augmentation}.
    parser.add_argument('-eri_threshold',  type=float,   default=1e-10, help="loss function threshold only")
    parser.add_argument('-rotate_deg',     type=float,   default=90, help="how many degrees to rotate")
    parser.add_argument('-test_dataloader',     action="store_true", help="no training, just test/loop through dataloader. ")


    # models 
    parser.add_argument('-nn',       action="store_true", help="train nn, defaults to GD") 
    parser.add_argument('-tiny',     action="store_true") 
    parser.add_argument('-small',    action="store_true") 
    parser.add_argument('-base',     action="store_true") 
    parser.add_argument('-medium',   action="store_true") 
    parser.add_argument('-large',    action="store_true") 
    parser.add_argument('-xlarge',   action="store_true") 
    parser.add_argument('-largep',   action="store_true")  # large "plus"
    parser.add_argument('-forces',   action="store_true")  
    parser.add_argument("-checkpoint", default=-1, type=int, help="which iteration to save model (default -1 = no saving)") # checkpoint model 
    parser.add_argument("-resume",   default="", help="path to checkpoint pickle file") # resume saved (checkpointed) model
    opts = parser.parse_args()
    # trick to allow 1e-4/math.sqrt(16) when reducing bs by 16. 
    opts.lr = eval(opts.lr)
    opts.min_lr = eval(opts.min_lr)
    if opts.tiny or opts.small or opts.base or opts.large or opts.xlarge: opts.nn = True 


    class HashableNamespace:
      def __init__(self, namespace): self.__dict__.update(namespace.__dict__)
      def __hash__(self): return hash(tuple(sorted(self.__dict__.items())))
    opts = HashableNamespace(opts)

    args_dict = vars(opts)
    print(args_dict)

    # benzene 
    if opts.benzene: 
        mol_strs = [[
                ["C", ( 0.0000,  0.0000, 0.0000)],
                ["C", ( 1.4000,  0.0000, 0.0000)],
                ["C", ( 2.1000,  1.2124, 0.0000)],
                ["C", ( 1.4000,  2.4249, 0.0000)],
                ["C", ( 0.0000,  2.4249, 0.0000)],
                ["C", (-0.7000,  1.2124, 0.0000)],
                ["H", (-0.5500, -0.9526, 0.0000)],
                ["H", (-0.5500,  3.3775, 0.0000)],
                ["H", ( 1.9500, -0.9526, 0.0000)], 
                ["H", (-1.8000,  1.2124, 0.0000)],
                ["H", ( 3.2000,  1.2124, 0.0000)],
                ["H", ( 1.9500,  3.3775, 0.0000)]
            ]]

    if opts.alanine: 
        mol_strs = [[
            ["H", (2.000,  1.000, -0.000)],
            ["C", (2.000,  2.090,  0.000)],
            ["H", (1.486,  2.454,  0.890)],
            ["H", (1.486,  2.454, -0.890)],
            ["C", (3.427,  2.641, -0.000)],
            ["O", (4.391,  1.877, -0.000)],
            ["N", (3.555,  3.970, -0.000)],
            ["H", (2.733,  4.556, -0.000)],
            ["C", (4.853,  4.614, -0.000)],
            ["H", (5.408,  4.316,  0.890)],
            ["C", (5.661,  4.221, -1.232)],
            ["H", (5.123,  4.521, -2.131)],
            ["H", (6.630,  4.719, -1.206)],
            ["H", (5.809,  3.141, -1.241)],
            ["C", (4.713,  6.129,  0.000)],
            ["O", (3.601,  6.653,  0.000)],
            ["N", (5.846,  6.835,  0.000)],
            ["H", (6.737,  6.359, -0.000)],
            ["C", (5.846,  8.284,  0.000)],
            ["H", (4.819,  8.648,  0.000)],
            ["H", (6.360,  8.648,  0.890)],
            ["H", (6.360,  8.648, -0.890)],
        ]]

    train_QPT(mol_strs, opts)
