import jax
from jax import numpy as jnp, random as jr
jax.config.update('jax_enable_x64', True)
import optax
from utils import *
from kernels import *
import equinox as eqx
from models import KNO_DIFFUSION_REACTION as model
from scipy.io import loadmat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10_000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=32)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--test-batch-size', type=int, default=10)
parser.add_argument('--output-kernel', type=str, default='a_g', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--time', action='store_true')
parser.add_argument('--quad-res', default=729, type=int)
parser.add_argument('--print-every', type=int, default=1)
parser.add_argument('--eval-every', type=int, default=5)


args = parser.parse_args()

key = jr.PRNGKey(args.seed)

### load data
DTYPE = jnp.float32
data = jnp.load('./datasets/diffrec_3d.npz')
### input_function, input function locations, output function, output function locations
x, x_grid, y, y_grid = data["x"].astype(DTYPE), data["x_grid"].astype(DTYPE), data["y"].astype(DTYPE), data["y_grid"].astype(DTYPE)

### each input function is a constant so no point to interpolate to quadrature nodes, but we do need to interpolate to y_grid
const = x[:,0]
x = jnp.ones((1200, args.quad_res, 1)) * const.reshape(-1,1,1) 
y = y.reshape(1200, -1)

domain_dims = 3 
codomain_dims = 1

ntrain = 1000
ntest = 200
x_train, x_test = x[:ntrain], x[-ntest:]
y_train, y_test = y[:ntrain], y[-ntest:]

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

num_train_batches = len(x_train) // args.batch_size
num_steps = args.epochs * num_train_batches

### kernel setup
integration_kernel = kernels[args.int_kernel]
output_kernel = kernels[args.output_kernel]
output_kernel = partial(output_kernel, ndims=domain_dims)
integration_kernel = partial(integration_kernel, ndims=domain_dims)


### preprocess data
x_normalizer = UnitGaussianNormalizer(x_train) ## cus of this? 
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)
in_feats = domain_dims + codomain_dims

model = model(output_kernel,
              integration_kernel, 
              args.lift_dim, 
              args.depth,   
              in_feats,
              key=key) 

lr_schedule = cosine_annealing(num_steps, peak_value=args.lr_max, gamma=0.2, num_cycles=2)
optimizer=optax.adam(lr_schedule)
opt_state = optimizer.init(eqx.filter([model], eqx.is_array))

qr = loadmat(f'./datasets/n_{args.quad_res}.mat')
q_nodes,q_weights = qr['t'], qr['w']

param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
print(f'{param_count=}')

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, ):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,
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
                                            eqx.filter([model], eqx.is_array))
    model = eqx.apply_updates(model, updates[0])
    return model, opt_state, train_loss, rel_l2

@eqx.filter_jit
def eval(model, batch,):
    x,y = batch
    def loss(model):
        y_pred = jax.lax.map(lambda x: model(x,y_grid,
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

    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_rel_l2 = eval(model, (x_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')