import math
from typing import Optional, TypeVar, Callable

import jax
import jax.random as jrandom

from ..custom_types import Array
from ..module import Module, static_field

import functools as ft

def init_xavier_uniform(key, shape, in_features, out_features):
    lim = math.sqrt(6 / (in_features+out_features))
    params = jrandom.uniform(key, shape, minval=-lim, maxval=lim)
    return params


class Linear(Module):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: int = static_field()
    out_features: int = static_field()
    use_bias: bool = static_field()
    init_func_weights: Optional[Callable]
    init_func_bias: Optional[Callable]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        init_func_weights: Optional[Callable] = None, 
        init_func_bias: Optional[Callable] = None,
        *,
        key: "jax.random.PRNGKey"
    ):
        """**Arguments:**

        - `in_features`: The input size.
        - `out_features`: The output size.
        - `use_bias`: Whether to add on a bias as well.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)

        if init_func_weights is None:
            init_func_weights = ft.partial(init_xavier_uniform, in_features=in_features, out_features=out_features)

        self.weight = init_func_weights(wkey, (out_features, in_features))
        if use_bias:
            if init_func_bias is None:
                init_func_bias = ft.partial(init_xavier_uniform, in_features=in_features, out_features=out_features)
            self.bias = init_func_bias(bkey, (out_features,))
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.init_func_weights = init_func_weights
        self.init_func_bias = init_func_bias

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring batch dimensions) then use
            `jax.vmap`. For example, for an input `x` of shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`
        """

        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x


_T = TypeVar("T")


class Identity(Module):
    """Identity operation that does nothing. Sometimes useful as a placeholder for
    another Module.
    """

    def __init__(self, *args, **kwargs):
        """Consumes arbitrary `*args` and `**kwargs` but ignores them."""
        # Ignores args and kwargs
        super().__init__()

    def __call__(self, x: _T, *, key: Optional["jax.random.PRNGKey"] = None) -> _T:
        """**Arguments:**

        - `x`: The input, of any type.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The input, unchanged.
        """
        return x
