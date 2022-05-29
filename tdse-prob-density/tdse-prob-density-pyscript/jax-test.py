import jax
import jax.numpy as jnp

def sqfn(x):
    return x**2

jacosqfn = jax.jacobian(sqfn)

vec = jnp.array([1.0, 3.0, 7.0])

print('sqfn:', sqfn(vec))

print('gradsqfn:', jacosqfn(vec), sep='\n')