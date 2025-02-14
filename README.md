<h1 align='center'>Equinox</h1>
<h2 align='center'>Callable PyTrees and filtered JIT/grad transformations<br>=> neural networks in JAX</h2>

Equinox brings more power to your model building in [JAX](https://github.com/google/jax).<br>
Represent *parameterised functions as data*, and use *filtered transformations* for powerful fine-grained control of the model-building process.

Equinox is half tech-demo, half neural network library.

## Equinox in brief

### Building neural networks

Build models using a PyTorch-like class based API *without* sacrificing JAX-like functional programming.

In particular, *without* extra complexity like class-to-functional transformations, or custom notions of parameter groups.

Equinox is a tiny library -- no behind-the-scenes magic, guaranteed.
The elegance of Equinox is its selling point in a world that already has [Haiku](https://github.com/deepmind/dm-haiku), [Flax](https://github.com/google/flax) etc.

### Technical contributions

1. Equinox represents *parameterised functions as [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html)*. This is done by subclassing `equinox.Module`. (For example a neural network is a function parameterised by its weights, biases, etc.) They key point is that you can now JIT/grad a higher-order function (like a loss function) with respect to a parameterised function as its input (like a model).
2. Equinox provides *filtered transformations*. These allow you to JIT/grad just some of the leaves of a PyTree -- not just a whole argument. This is provided by `equinox.jitf` and `equinox.gradf`.

Point 1 allows you to represent models as PyTrees. (No separate passing around of parameters and forward passes.)<br>
Point 2 is a more fine-grained way to JIT/grad PyTrees.<br>
They are completely independent of each other -- but synergise very nicely.
- If your model-as-a-PyTree only consists of things you want to JIT/differentiate, then just use `jax.jit` or `jax.grad` as normal. Equinox is JAX-friendly.
- If you parameterise your model by something that isn't JIT/grad-able, then use `equinox.jitf` or `equinox.gradf`. For example you might want to store an arbitrary Python function as an activation function.

### Installation

```
pip install equinox
```
Requires Python 3.7+ and JAX 0.2.18+.

### Quick example

```python
import equinox as eqx
import functools as ft, jax, jax.numpy as jnp, jax.random as jrandom

# Define our model. `Module` subclasses are both functions and data, so we can pass them into higher
# order functions like vmap/jit/grad or a loss function.
# There's no magic here. `Module` just registers your class as a PyTree node.
class LinearOrIdentity(eqx.Module):
    weight: jnp.ndarray
    flag: bool

    def __init__(self, in_features, out_features, flag, key):
        self.weight = jrandom.normal(key, (out_features, in_features))
        self.flag = flag

    def __call__(self, x):
        if self.flag:
            return x
        return self.weight @ x

# We use the fact that our model is data, by passing it in as an argument to the loss.
# There's no magic here. `model` is a PyTree like any other; nothing about it being a `Module` is
# special-cased.
#
# We use filtered transformations to unpack its data and select just the leaves we want to 
# JIT+differentiate. (In this case, all JAX arrays -- so `weight` but not `flag`.)
@ft.partial(eqx.jitf, filter_fn=eqx.is_array)
@ft.partial(eqx.gradf, filter_fn=eqx.is_array)
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

modelkey, xkey, ykey = jrandom.split(jrandom.PRNGKey(0), 3)
model = LinearOrIdentity(2, 3, flag=False, key=modelkey)
x, y = jrandom.normal(xkey, (100, 2)), jrandom.normal(ykey, (100, 3))
grads = loss(model, x, y)
```

### Integrates smoothly with JAX

There's nothing special about Equinox modules. They're just PyTrees.

There's nothing special about filtered transformations. They just operate on PyTrees.

Equinox is all just regular JAX -- PyTrees and transformations! Together, these two pieces allow us to specify complex models in JAX-friendly ways.

In particular note that `equinox.jitf` and `equinox.gradf` are *not* "a way to make JIT/grad work with Modules" like many libraries have. They are general operations on PyTrees, and nothing about `Module` is special-cased. (And indeed `Module` itself is just a PyTree like any other.)

## Examples

- [`train_mlp.py`](./examples/train_mlp.py) gives a short example that introduces `equinox.jitf` and `equinox.gradf`. These will be used to select the parameters of an MLP and train them.
 
- [`frozen_layer.py`](./examples/frozen_layer.py) demonstrates how this approach really shines: some of the parameters will be trained, some of them will be frozen, but *all* of them will be efficiently JIT-traced.

- [`build_model.py`](./examples/build_model.py) demonstrates how to build parameterised-functions-as-data using `equinox.Module`. In particular we'll construct an MLP from scratch, and then pass it into higher-order functions like JIT and grad in order to train it. This allows us to produce models using a familiar class-based syntax, that are also functional and integrate directly with JAX's JIT/autograd.

- [`train_rnn.py`](./examples/train_rnn.py) trains an RNN on a toy clockwise/anticlockwise spiral classification problem.

- [`modules_to_initapply.py`](./examples/modules_to_initapply.py) demonstrates how to use Equinox in an init/apply-style way, which some JAX libraries have been built around. (e.g. Stax)

## API

### Full API list
```python
# Module                         # Utilities        
equinox.Module                   equinox.apply_updates
                                 equinox.tree_at      
# Filtered transformations       equinox.tree_equal   
equinox.jitf                     
equinox.gradf                    # Neural networks
equinox.value_and_grad_f         equinox.nn.Linear
                                 equinox.nn.Identity
# Filters                        equinox.nn.Dropout
equinox.is_array                 equinox.nn.GRUCell
equinox.is_array_like            equinox.nn.LSTMCell
equinox.is_inexact_array         equinox.nn.Sequential
equinox.is_inexact_array_like    equinox.nn.MLP

# Splitting/merging
equinox.split                    
equinox.merge                    
                                 
```

### Module

```python
equinox.Module
```
Base class; create your model by inheriting from this.

Specify all its attributes at the class level (identical to [dataclasses](https://docs.python.org/3/library/dataclasses.html)). This defines its children in the PyTree.

```python
class MyModule(equinox.Module):
    weight: typing.Any
    bias: typing.Any
    submodule: Module
```

In this case a default `__init__` method is provided, which just fills in these attributes with the argments passed: `MyModule(weight, bias, submodule)`. Alternatively you can provide an `__init__` method yourself. (For example to specify dimension sizes instead of raw weights.) By the end of `__init__`, every attribute must have been assigned.

```python
class AnotherModule(equinox.Module):
    weight: Any

    def __init__(self, input_size, output_size, key):
        self.weight = jax.random.normal(key, (output_size, input_size))
```

After initialisation then attributes cannot be modified: models are immutable as per functional programming. (Parameter updates are made by creating a new model, not by mutating parameters in-place; see for example [`train_mlp.py`](./examples/train_mlp.py).)

It is typical to also create some methods on the class. As `self` will be an input parameter -- treated as a PyTree -- then these methods will get access to the attributes of the instance. Defining `__call__` gives an easy way to define a forward pass for a model (although any method can be used, and no methods are special-cased):

```python
class LinearWithoutBias(equinox.Module):
    weight: Any

    def __call__(self, x):
        return self.weight @ x
```

If defining a method `meth`, then take care not to write `instance = MyModule(...); jax.jit(instance.meth)(...)`. (Or similarly with `jax.grad`, `equinox.jitf` etc.) This is because `instance.meth` is not a pure function as it already has the `self` parameter passed implicitly. Instead do
```python
@jax.jit
def func(instance, args):
    instance.meth(args)
    # Also use this pattern with instance(args) if you defined `__call__` instead of `meth`.
```

### Filtered transformations

```python
equinox.jitf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.jit`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is inputted to `fun`. If it returns `True`, the leaf will be traced. It returns `False`, the leaf with be treated as static. Mutually exclusive with `filter_tree`.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs. (Or if `static_argnums` is passed, the number of inputs not already marked static via `static_argnums`.) It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be traced; it it is `False` the leaf will be treated as static. Mutually exclusive with `filter_tree`.
- `**kwargs` are the usual other arguments to `jax.jit`, like `static_argnums`. In particular, a leaf will be marked static if either (a) it is filtered as being so, *or* (b) it is part of a PyTree that is marked through `static_argnums`.

Precisely one of `filter_fn` or `filter_tree` must be passed.<br>
See also `equinox.is_array`, which is usually a good choice of `filter_fn`. This will trace every JAX array, and make the rest static.<br>
See also `equinox.tree_at` for an easy way to create the `filter_tree` argument.

```python
equinox.gradf(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.grad`.

- `fun` is a pure function to JIT compile.
- `filter_fn` is a callable `Any -> bool`. It will be called on every leaf of every PyTree that is marked as potentially requiring gradient via `argnums`. If it returns `True`, the leaf will be differentiated. If it returns `False`, the leaf will not be differentiated. Mutually exclusive with `filter_tree`.
- `filter_tree` is a tree, or tuple of trees, of the same length as the number of inputs marked as potentially requiring gradient via `argnums`. It must have the exact same tree structure as the inputs. Every leaf must be either `True` or `False`. Each leaf of `filter_tree` is matched up against the corresponding input: if it is `True` the leaf will be differentiated; if it is `False` the leaf will not be differentiated. Mutually exclusive with `filter_fn`.
- `**kwargs` are the usual other argments to `jax.grad`, like `argnums`. In particular, a leaf will only be differentiated if (a) it is filtered as being so, *and* (b) it is part of a PyTree that is marked through `argnums`.

Precisely one of `filter_fn` or `filter_tree` must be passed.<br>
See also `equinox.is_inexact_array`, which is usually a good choice of `filter_fn`. This will differentiate all floating-point JAX arrays.<br>
See also `equinox.tree_at` for an easy way to create the `filter_tree` argument.

Note that as the returned gradients must have the same structure as the inputs, then all nondifferentiable components of the input PyTrees will have gradient `None`. 
Doing a simple `jax.tree_map(lambda m, g: m - lr * g, model, grad)` will fail. 
As such Equinox provides `equinox.apply_updates` as a simple convenience: it will only apply the update if the gradient is not `None`. See below.

```python
equinox.value_and_grad_f(fun, *, filter_fn=None, filter_tree=None, **kwargs)
```
Wraps `jax.value_and_grad`. Arguments are as `equinox.gradf`.

### Filters

Any function `Any -> bool` can be used as a filter. We provide some convenient common choices.

```python
equinox.is_array(element)
```
Returns `True` if `element` is a JAX array (not but a NumPy array).

```python
equinox.is_array_like(element)
```
Returns `True` if `element` is a JAX array, NumPy array, or a Python float/int/bool/complex.

```python
equinox.is_inexact_array(element)
```
Returns `True` if `element` is a floating point JAX array (but not a NumPy array).

```python
equinox.is_inexact_array_like(element)
```
Returns `True` if `element` is a floating point JAX array, floating point NumPy array, or a Python float or complex.

### Splitting/merging

Filters can also be used to organise the contents of PyTrees, if needed.

```python
equinox.split(pytree, filter_fn=None, filter_tree=None)
```
Partitions the leaves of a PyTree into two groups.

- `pytree` is any PyTree
- `filter_fn` is any function `Leaf -> bool` to call on each of its leaves.
- `filter_tree` is a PyTree with the same structure as `pytree`, with every leaf either `True` or `False`.

Precisely one of `filter_fn` or `filter_tree` may be passed.

Returns a 4-tuple of `(flat_true, flat_false, which, treedef)`.

- `flat_true` will be a list of leaves for which `filter_fn`/`filter_tree` was `True`.
- `flat_false` will be a list of leaves for which `filter_fn`/`filter_tree` was `False`.
- `which` and `treedef` specify the input PyTree. `treedef` is a `PyTreeDef` (like `jax.tree_flatten` returns). `which` is a list of `True`/`False` specifying which leaves were truthy or falsey.

See also `equinox.merge` to reconstitute the PyTree again.

This function is useful when working with JAX libraries that only support PyTrees of trainable parameters, and not more general PyTrees: the model can be split into its trainable and nontrainable components and passed into the library that way. See the example [`modules_to_initapply.py`](./examples/modules_to_initapply.py).

```python
equinox.merge(flat_true, flat_false, which, treedef)
```

The inverse of `equinox.split`.

### Utilities

```python
equinox.apply_updates(model, updates)
```
Performs a training update to a model.
- `model` must be a PyTree;
- `updates` must be a PyTree with the same structure.

It essentially performs `jax.tree_map(lambda m, u: m + u, model, updates)`. However anywhere `updates` is `None` then no update is made at all, so as to handle nondifferentiable parts of `model`.

The returned value is the updated model. (`model` is *not* mutated in place, as is usual in JAX and functional programming.)

To produce `updates`, it is typical to take the gradients from the loss function, and then adjust them according to any standard optimiser; for example [Optax](https://github.com/deepmind/optax) provides `optax.sgd` or `optax.adam`.

```python
equinox.tree_at(where, pytree, replace=_sentinel, replace_fn=_sentinel)
```
Modifies an existing tree, and returns the modified tree. (Like `.at` for "in place modifications" of JAX arrays.)

- `where` is a callable `PyTree -> Leaf` or `PyTree -> Tuple[Leaf, ...]`. It should consume a PyTree of the same shape as `pytree`, and return the leaf or leaves that should be replaced. For example `where=lambda mlp: mlp.layers[-1].linear.weight`.
- `pytree` is the existing PyTree to modify.
- `replace` should either be a single element, or a tuple of the same length as returned by `where`. This specifies the replacements to make at the locations specified by `where`. Mutually exclusive with `replace_fn`.
- `replace_fn` should be a function `Leaf -> Any`. It will be called on every leaf replaced using `where`. The return value from `replace_fn` will be used in its place. Mutually exclusive with `replace`.

For example this can be used to specify the weights of a model to train or not train:
```python
trainable = jax.tree_map(lambda _: False, model)
trainable = equinox.tree_at(lambda mlp: mlp.layers[-1].linear.weight, model, replace=True)
equinox.gradf(..., filter_tree=trainable)
```

```python
equinox.tree_equal(*pytrees)
```
Returns `True` if all PyTrees in the list are equal. All arrays must have the same shape, dtype, and values. JAX arrays and NumPy arrays are not considered equal.

### Neural network library

Equinox includes a small neural network library, mostly as a tech demo for how the rest of the library can be used. Its API is modelled after PyTorch.

```python
equinox.nn.Linear(in_features, out_features, use_bias=True, *, key)(input)
equinox.nn.Identity(*args, **kwargs)(input)  # args and kwargs are ignored
equinox.nn.Dropout(p=0.5, deterministic=False)(input, *, key=None, deterministic=None)
equinox.nn.GRUCell(input_size, hidden_size, bias=True, *, key)(input, hidden)
equinox.nn.LSTMCell(input_size, hidden_size, bias=True, *, key)(input, hidden)
equinox.nn.Sequential(layers)(input, *, key=None)
equinox.nn.MLP(in_size, out_size, width_size, depth,
               activation=jax.nn.relu, final_activation=lambda x: x, *, key)(input)
```
These all behave in the way you expect. The `key` arguments are used to generate the random initial weights, or to generate randomness on the forward pass of stochastic layers like `Dropout`.

The `Dropout(deterministic=...)(deterministic=...)` options determines whether to have the layer act as the identity function, as is commonly done with dropout during inference time. The call-time `deterministic` takes precendence if it passed; otherwise the init-time `deterministic` is used. (Note that because models are PyTrees, you can modify the init-time `deterministic` flag using `equinox.tree_at`. This is perfectly fine, and might be handy if it's easier than using the call-time flag.)

The `MLP(final_activation=...)` option determines any final activation function to apply after the last layer. (In some cases it is desirable for this to be different to the activation used in the main part of the network.)
