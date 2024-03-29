import jax
from jax import vmap
import jax.numpy as jnp
from functools import partial
import jax.experimental.host_callback
import math 
import numpy as np 

def rand(rng, f, shape, **kwargs):
    rng, rng1 = jax.random.split(rng)
    return rng, f(rng1, shape, **kwargs)

def linear_init_uniform(rng: jax.random.KeyArray, in_features: int, out_features: int):
    params = ParamsDict()
    rnd_range = 1 / in_features**0.5
    rng, params.weight = rand( rng, jax.random.uniform, (in_features, out_features), minval=-rnd_range, maxval=rnd_range,)
    params.bias = jnp.zeros((out_features,))
    return rng, params, (in_features, out_features)

def elementwise_linear_init_identity(shape): return ParamsDict(gain=jnp.ones(shape), bias=jnp.zeros(shape))

def linear(params, x: jnp.ndarray): return x @ params.weight + params.bias[None, :]

def elementwise_linear(params, x: jnp.ndarray): return params.gain[None, :] * x + params.bias[None, :]

def standardize(x, eps=1e-5): return (x - x.mean()) / (x.std() + eps)

def transformer_init(
    rng: jax.random.KeyArray,
    n_vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    max_len=4096,
):
    total_params = 0 

    config = ParamsDict()
    config.heads = n_heads
    if True: 
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0
    else:
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0

    params = ParamsDict()

    print("_"*100)

    rng, params.embeddings = rand(rng, jax.random.normal, (n_vocab, d_model))
    total_params += np.prod(params.embeddings.shape)
    print("%26s %26s %26s"%("params.embeddings",params.embeddings.shape, np.prod(params.embeddings.shape)))

    rng, params.project_positions, shape = linear_init_uniform(rng, 123, d_model)
    total_params += np.prod(shape)
    print("%26s %26s %26s"%("params.project_positions",shape, np.prod(shape)))

    params.layers = []
    for i in range(n_layers):
        layer = ParamsDict()
        layer.norm_self_attn = elementwise_linear_init_identity(d_model)
        total_params += np.prod(d_model*2)
        print("%26s %26s %26s"%("layer%i.norm_self_attn"%i, (d_model,2), np.prod((d_model, 2))))

        rng, layer.kqv, shape = linear_init_uniform(rng, d_model, d_model*3)
        total_params += np.prod(shape) 
        print("%26s %26s %26s"%("layer%i.kqv"%i, shape, np.prod(shape)))

        rng, layer.c_proj, shape = linear_init_uniform(rng, d_model, d_model)
        total_params += np.prod(shape) 
        print("%26s %26s %26s"%("layer%i.c_proj"%i, shape, np.prod(shape)))

        layer.norm_ff = elementwise_linear_init_identity(d_model)
        total_params += np.prod(d_model*2)
        print("%26s %26s %26s"%("layer%i.norm_ff"%i, (d_model,2), np.prod((d_model, 2))))

        rng, layer.ffn1, shape = linear_init_uniform(rng, d_model, d_ff)
        total_params += np.prod(shape)
        print("%26s %26s %26s"%("layer%i.ffn1"%i, shape, np.prod(shape))) 

        rng, layer.ffn2, shape = linear_init_uniform(rng, d_ff, d_model)
        total_params += np.prod(shape)
        print("%26s %26s %26s"%("layer%i.ffn2"%i, shape, np.prod(shape)))

        params.layers.append(layer)

    print("total: ", total_params)

    return rng, config, params, total_params


@partial(jax.jit, static_argnums=0)
def transformer(cfg, params, x: jnp.ndarray, pos: jnp.ndarray, H_core: jnp.ndarray, L_inv, dm_init,diff_JK, V_xc, H_init):
    embeddings = cfg.lambda_e * params.embeddings[x, :]  # L x Dm
    L, Dm      = embeddings.shape
    nheads     = cfg.heads

    # todo: apply the same translation/rotation to hamiltonian. 
    # Roughly get f( {R@ri+t}_i ) = f( {r_i}_i )
    pos      = pos - jnp.mean(pos, axis=0).reshape(1, 3) # makes jnp.mean(position, axis=0) = [0,0,0]
    cov      = jnp.cov(pos.T)
    eigvects = jnp.linalg.eigh(cov)[1] 
    pos      = pos @ eigvects # makes jnp.cov(pos.T)=jnp.eye(3) 

    # Mix of sin/cos and 3d point cloud transformers. 
    pos = jnp.concatenate([pos] +  \
                                [jnp.cos(pos*f/20*2*np.pi) for f in range(20)] + \
                                [jnp.sin(pos*f/20*2*np.pi) for f in range(20)], 
                                axis=1) #(N,3) -> (N,3+60+60) = (N, 123)
    pos = linear(params.project_positions, pos)                         # L x Dm
    all_pairs_dot = pos.reshape(-1, Dm) @ pos.reshape(-1, Dm).T  # this is L x L 
    x = embeddings + pos                                                     #  L x Dm 
    
    def block(x, layer_num, layer):
        # Layer-normalize 
        t1 = vmap(standardize)(x)                           # L x Dm 
        t1 = elementwise_linear(layer.norm_self_attn, t1)   # L x Dm

        qkv     = linear(layer.kqv, t1)                     # L x 3*Dm
        q,k,v = jnp.split(qkv, 3, axis=1)                   # (L x Dm,)*3
        q = jnp.transpose(q.reshape(L, nheads, Dm//nheads), (1, 0, 2)) # nheads x L x Dm//nheads
        k = jnp.transpose(k.reshape(L, nheads, Dm//nheads), (1, 0, 2))
        v = jnp.transpose(v.reshape(L, nheads, Dm//nheads), (1, 0, 2))

        score = (q @ jnp.transpose(k, (0, 2, 1)))  
        score = score 
        score = score / math.sqrt(Dm/nheads)                # B x L x L 

        # quantum biased attention 
        if True:  
            score = score.at[:2].add( H_core / jnp.max(jnp.abs(H_core)) )
            score  = score.at[2:4].add( all_pairs_dot / jnp.max(jnp.abs(all_pairs_dot)) )
            M = L_inv @ H_core @ L_inv.T   
            score  = score.at[4:6].add(  M / jnp.max(jnp.abs(M)) )
            score = score.at[6:8].add( dm_init / jnp.max(jnp.abs(dm_init))) 
            score = score.at[8:10].add(diff_JK / jnp.max(jnp.abs(diff_JK)))
            score = score.at[10:12].add(V_xc / jnp.max(jnp.abs(V_xc)))
            score = score.at[12:14].add(H_init / jnp.max(jnp.abs(H_init)))

        attn = jax.nn.softmax(score, axis=1) 
        y = attn @ v 
        y = y.swapaxes(0,1).reshape(L, Dm)  
        y = linear(layer.c_proj, y)
        x = x + y 

        # Layer-normalize 
        t2 = vmap(standardize)(x)
        t2 = elementwise_linear(layer.norm_ff, t2)          # L x Dm

        # Feedforward fully connected
        t2 = linear(layer.ffn1, t2)                         # L x Dm*4
        t2 = jax.nn.gelu(t2)
        t2 = linear(layer.ffn2, t2)                         # L x Dm

        # Residual connection 
        x = x + t2
        return x

    # Apply all but the last transformer block. 
    for layer_num, layer in enumerate(params.layers[:-1]):
        x = jax.checkpoint(block)(x, layer_num, layer)
        #x = block(x, layer_num, layer)

    layer = params.layers[-1]
    # Prediction is last attention (without nhead = 1), and q=k so score is symmetric! 
    nheads = 1 
    t1    = vmap(standardize)(x)                           # L x Dm 
    t1    = elementwise_linear(layer.norm_self_attn, t1)   # L x Dm
    qkv   = linear(layer.kqv, t1)
    q,k,v = jnp.split(qkv, 3, axis=1)
    q     = jnp.transpose(q.reshape(L, 1, Dm//1), (1, 0, 2))[0]
    k     = jnp.transpose(k.reshape(L, 1, Dm//1), (1, 0, 2))[0] 
    v     = jnp.transpose(v.reshape(L, 1, Dm//1), (1, 0, 2))[0] 
    # give model access to directly learn eigenvalues; 
    # training error drops much faster. 
    score = (q @ q.T) #/ math.sqrt(Dm*1)   # scale dosn't matter, qr is scale invariant! 
    Q= jnp.linalg.qr(score)[0]
    M = Q @ jnp.diag(k[:, 0]) @ Q.T # perhaps scale the eigenvalues to [-10, 10] or so? 
    #M = q @ q.T / math.sqrt(Dm)
    return M 

import types
import json
import jax

import numbers

def is_simple_type(x):
    return isinstance(x, (numbers.Number, bool, str))

@jax.tree_util.register_pytree_node_class
class ParamsDict(types.SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.__dict__, lambda a: a is not self.__dict__) 

    @classmethod
    def tree_unflatten(cls, aux, values):
        return ParamsDict(**jax.tree_util.tree_unflatten(aux, values))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __hash__(self):
        return hash(tuple(hash(x) for (_,x) in self.__dict__.items()))

    def print(self, path = ''):
        for (k,v) in self.items(path):
            print(k + ':',v)

    @classmethod
    def labels_aux(cls, path, obj):
        if isinstance(obj, (list, tuple)) and any(not is_simple_type(x) for x in obj):
            for i,vi in enumerate(obj):
                yield from cls.labels_aux(f'{path}[{i}]', vi)
        elif isinstance(obj, dict):
            for (k,v) in obj.items():
                yield from cls.labels_aux(path + '/' + k, v)
        elif isinstance(obj, ParamsDict):
            yield from cls.labels_aux(path, obj.__dict__)
        else:
            yield (path, obj)

    def items(self, path = ''):
        yield from self.labels_aux(path, self)

    def to_float32(self):
        def convert_to_float32(x):
            if isinstance(x, jnp.ndarray) and x.dtype == jnp.float64:
                return x.astype(jnp.float32)
            return x

        new_dict = jax.tree_map(convert_to_float32, self.__dict__)
        return ParamsDict(**new_dict)

    def to_float64(self):
        def convert_to_float64(x):
            if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
                return x.astype(jnp.float64)
            return x

        new_dict = jax.tree_map(convert_to_float64, self.__dict__)
        return ParamsDict(**new_dict)

