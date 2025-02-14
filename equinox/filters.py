from typing import Any, Callable, List, Optional, Tuple
from typing_extensions import get_args

import jax
import jax.numpy as jnp

from .custom_types import Array, MoreArrays, PyTree, TreeDef


_array_types = get_args(Array)
_morearray_types = get_args(MoreArrays)
_arraylike_types = _morearray_types + (int, float, complex, bool)


# TODO: not sure if this is the best way to do this? In light of:
# https://github.com/google/jax/commit/258ae44303b1539eff6253263694ec768b8803f0#diff-de759f969102e9d64b54a299d11d5f0e75cfe3052dc17ffbcd2d43b250719fb0
def is_array(element: Any) -> bool:
    return isinstance(element, _array_types)


# Does _not_ do a try/except on jnp.asarray(element) because that's very slow.
def is_array_like(element: Any) -> bool:
    return isinstance(element, _arraylike_types)


def is_inexact_array(element: Any) -> bool:
    return is_array(element) and jnp.issubdtype(element.dtype, jnp.inexact)


def is_inexact_array_like(element: Any) -> bool:
    return (
        isinstance(element, _morearray_types)
        and jnp.issubdtype(element.dtype, jnp.inexact)
    ) or isinstance(element, (float, complex))


def split(
    pytree: PyTree,
    filter_fn: Optional[Callable[[Any], bool]] = None,
    filter_tree: Optional[PyTree] = None,
) -> Tuple[List[Any], List[Any], List[bool], TreeDef]:

    validate_filters("split", filter_fn, filter_tree)
    flat, treedef = jax.tree_flatten(pytree)
    flat_true = []
    flat_false = []

    if filter_fn is None:
        which, treedef_filter = jax.tree_flatten(filter_tree)
        if treedef != treedef_filter:
            raise ValueError(
                "filter_tree must have the same tree structure as the PyTree being split."
            )
        for f, w in zip(flat, which):
            if w:
                flat_true.append(f)
            else:
                flat_false.append(f)
    else:
        which = []
        for f in flat:
            if filter_fn(f):
                flat_true.append(f)
                which.append(True)
            else:
                flat_false.append(f)
                which.append(False)

    return flat_true, flat_false, which, treedef


def merge(
    flat_true: List[Any], flat_false: List[Any], which: List[bool], treedef: TreeDef
):
    flat = []
    flat_true = iter(flat_true)
    flat_false = iter(flat_false)
    for element in which:
        if element:
            flat.append(next(flat_true))
        else:
            flat.append(next(flat_false))
    return jax.tree_unflatten(treedef, flat)


def validate_filters(fn_name, filter_fn, filter_tree):
    if (filter_fn is None and filter_tree is None) or (
        filter_fn is not None and filter_tree is not None
    ):
        raise ValueError(
            f"Precisely one of `filter_fn` and `filter_tree` should be passed to {fn_name}"
        )
