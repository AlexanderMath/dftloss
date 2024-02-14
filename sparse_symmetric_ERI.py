import pyscf 
import numpy as np 
import jax 
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp 
HYB_B3LYP = 0.2

def get_i_j(val):
    i = (np.sqrt(1 + 8*val.astype(np.uint64)) - 1)//2 
    j = (((val - i) - (i**2 - val))//2)
    return i, j

def _ijkl(value, symmetry, N, f):
    i, j, k, l = value[0], value[1], value[2], value[3]
    return f(i,j,k,l,symmetry,N)
ijkl = jax.vmap(_ijkl, in_axes=(0, None, None, None))

def np_ijkl(value, symmetry, N, f):
    i, j, k, l = value[:, 0], value[:, 1], value[:, 2], value[:, 3]
    return f(i,j,k,l,symmetry,N)


def num_repetitions_fast(ij, kl):
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)

    repetitions = 2**(
        np.equal(i,j).astype(np.uint64) + 
        np.equal(k,l).astype(np.uint64) + 
        (1 - ((1 - np.equal(k,i) * np.equal(l,j)) * 
        (1- np.equal(k,j) * np.equal(l,i))).astype(np.uint64))
    )
    return repetitions

indices_func = lambda i,j,k,l,symmetry,N: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k,
                                                     k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i,
                                                     k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k,
                                                     i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]

def _indices_func(i, j, k, l, symmetry, N):
    if symmetry == 0: return i * N + j
    elif symmetry == 1: return j * N + i
    elif symmetry == 2: return i * N + j
    elif symmetry == 3: return j * N + i
    elif symmetry == 4: return k * N + l
    elif symmetry == 5: return l * N + k
    elif symmetry == 6: return k * N + l
    elif symmetry == 7: return l * N + k
    elif symmetry == 8 or symmetry == 9: return k * N + l
    elif symmetry == 10 or symmetry == 11: return l * N + k
    elif symmetry == 12 or symmetry == 13: return i * N + j
    elif symmetry == 14 or symmetry == 15: return j * N + i
    elif symmetry == 16: return k * N + j
    elif symmetry == 17: return k * N + i
    elif symmetry == 18: return l * N + j
    elif symmetry == 19: return l * N + i
    elif symmetry == 20: return i * N + l
    elif symmetry == 21: return i * N + k
    elif symmetry == 22: return j * N + l
    elif symmetry == 23: return j * N + k
    elif symmetry == 24: return i * N + l 
    elif symmetry == 25: return j*N+l 
    elif symmetry == 26: return i*N+k
    elif symmetry == 27: return j*N+k
    elif symmetry == 28: return k * N + j
    elif symmetry == 29: return l * N + j
    elif symmetry == 30: return k * N + i
    elif symmetry == 31: return l * N + i


def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm, foriloop):
    dm = dm.reshape(-1)
    dtype = dm.dtype 
    diff_JK = jnp.zeros(dm.shape, dtype=dtype)
    N = int(np.sqrt(dm.shape[0]))
    Z = jnp.zeros((N**2,), dtype=dtype)

    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(), 
        collapsed_slice_dims=(0,),
        start_index_map=(0,))
    scatter_dnums = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(), 
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,))

    def iteration(symmetry, vals): 
        diff_JK = vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            diff_JK = vals 
            indices = nonzero_indices[i]
            eris    = nonzero_distinct_ERI[i]

            dm_indices = ijkl(indices, symmetry+is_K_matrix*8, N, indices_func).reshape(-1, 1)
            dm_values = jax.lax.gather(dm, dm_indices, dimension_numbers=dnums, slice_sizes=(1,), mode=jax.lax.GatherScatterMode.FILL_OR_DROP)
            dm_values = dm_values * eris  
            
            ss_indices = ijkl(indices, symmetry+8+is_K_matrix*8, N, indices_func) .reshape(-1,1)
            diff_JK = diff_JK + jax.lax.scatter_add(Z,
                                            ss_indices, dm_values, 
                                            scatter_dnums, indices_are_sorted=True, unique_indices=False, mode=jax.lax.GatherScatterMode.FILL_OR_DROP)\
                                *(-HYB_B3LYP/2)**is_K_matrix
            
            return diff_JK

        batches = nonzero_indices.shape[0] 

        if foriloop: 
            diff_JK = jax.lax.fori_loop(0, batches, sequentialized_iter, diff_JK) 
        else:
            for i in range(batches):
                diff_JK = sequentialized_iter(i, diff_JK)
        return diff_JK

    if foriloop: 
        diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    else:
        for i in range(0, 16): 
            diff_JK = iteration(i, diff_JK)
    return diff_JK.reshape(N, N)
    
    