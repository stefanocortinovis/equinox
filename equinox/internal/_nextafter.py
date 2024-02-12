import jax
import jax.numpy as jnp
from jaxtyping import Array


@jax.custom_jvp
def nextafter(x: Array) -> Array:
    """Returns the floating point number after `x`."""
    # Flush denormal to normal.
    # Our use for these is to handle jumps in the vector field. Typically that means
    # there will be an "if x > cond" condition somewhere. However JAX uses DAZ
    # (denormals-are-zero), which will cause this check to fail near zero:
    # `jnp.nextafter(0, jnp.inf) > 0` gives `False`.
    return x + jnp.finfo(x).smallest_normal


@nextafter.defjvp
def nextafter_jvp(primals, tangents):
    (x,) = primals
    (tx,) = tangents
    return nextafter(x), tx


@jax.custom_jvp
def prevbefore(x: Array) -> Array:
    """Returns the floating point number before `x`."""
    return x - jnp.finfo(x).smallest_normal


@prevbefore.defjvp
def prevbefore_jvp(primals, tangents):
    (x,) = primals
    (tx,) = tangents
    return prevbefore(x), tx
