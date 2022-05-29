import jax
import jax.numpy as jnp

def sqfn(x):
    return x**2

gradsqfn = jax.grad(sqfn)

vec = jnp.array([1, 3, 7])

print('sqfn:', sqfn(vec))

print('gradsqfn:', gradsqfn(vec))