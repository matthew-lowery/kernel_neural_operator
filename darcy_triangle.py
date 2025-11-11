import jax
import optax
from jax import numpy as jnp, random as jr
from numpy.polynomial.legendre import leggauss
from utils import *
import equinox as eqx
from kernels import *
from quadratures import triangle_quad_rule
from models import KNO_DARCY_TRIANGLE as model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=96)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=10)
parser.add_argument('--input-kernel', type=str, default='a_g', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--output-kernel', type=str, default='a_g', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--quadrature-res', type=int, default=8)
parser.add_argument('--print-every', type=int, default=1)
parser.add_argument('--eval-every', type=int, default=5)


args = parser.parse_args()
print(args)

DTYPE = jnp.float32
key = jr.PRNGKey(args.seed)

### load data
fp = './datasets/darcy_triangular.npz'
data = jnp.load(fp)
x_grid = data["bc_coords"]

x_mu,x_std = x_grid.mean(axis=0),x_grid.std(axis=0)
x_grid = (x_grid - x_mu) / x_std

x = data["k"]
y_grid = data["mesh_grid"]
y_grid = (y_grid - x_mu) / x_std
y = data["h"]

print(x_grid.shape, x.shape, y.shape, y_grid.shape)
y = y.reshape(2000, -1)
x = x.reshape(2000, -1, 1)

domain_dims = 2
codomain_dims = 1
ntrain = 1900
ntest = 100

x_train, x_test = x[: ntrain], x[-ntest:]
y_train, y_test = y[: ntrain], y[-ntest:]

num_train_batches = len(x_train) // args.batch_size
num_steps = args.epochs * num_train_batches

## kernel setup
integration_kernel = kernels[args.int_kernel]
input_kernel = kernels[args.input_kernel]
output_kernel = kernels[args.output_kernel]
input_kernel = partial(input_kernel, ndims=domain_dims)
output_kernel = partial(output_kernel, ndims=domain_dims)
integration_kernel = partial(integration_kernel, ndims=domain_dims)

### preprocess data
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)

in_feats = domain_dims + codomain_dims
model = model(input_kernel,
                     output_kernel,
                     integration_kernel, 
                     args.lift_dim, 
                     args.depth,  
                     in_feats,
                     key=key) 


lr_schedule = cosine_annealing(num_steps, peak_value=args.lr_max, down=1e5)
optimizer=optax.adam(lr_schedule)
opt_state = optimizer.init(eqx.filter([model], is_trainable))

# load quadrature rules
q_nodes,q_weights = triangle_quad_rule(args.quadrature_res, leggauss)
q_nodes = (q_nodes - x_mu) / x_std
q_weights = jnp.prod(x_std) * q_weights
print(f'{len(q_nodes)=}')

param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
print(f'{param_count=}')

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, ):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,
                                                x_grid,
                                                y_grid,
                                                q_nodes,q_weights))(x)
        y_pred = y_pred.reshape(args.batch_size, -1)
        y_pred = y_normalizer.decode(y_pred)
        l2 =  ((y - y_pred)**2).sum(axis=-1).mean()
        rel_l2 = (jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
        return l2, rel_l2
    (train_loss,rel_l2), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model)
    updates,opt_state = optimizer.update([grads], 
                                            opt_state, 
                                            eqx.filter([model], is_trainable))
    model = eqx.apply_updates(model, updates[0])
    return model, opt_state, train_loss, rel_l2

@eqx.filter_jit
def eval(model, batch,):
    x,y = batch
    def loss(model):
        y_pred = jax.lax.map(lambda x: model(x,
                                                x_grid,
                                                y_grid,
                                                q_nodes,q_weights), x, batch_size=args.test_batch_size)
        y_pred = y_pred.reshape(ntest,-1)
        y_pred = y_normalizer.decode(y_pred)
        return (jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
    
    rel_l2 = loss(model)
    return rel_l2

for epoch in range(args.epochs):
    key,_ = jr.split(key)

    for i in range(num_train_batches): 
        batch = get_batch(key, (x_train, y_train), i, args.batch_size)
        model, opt_state, train_loss, rel_l2 = train_step(model, opt_state, optimizer, batch)
        # print(rel_l2)
    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_rel_l2 = eval(model, (x_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')