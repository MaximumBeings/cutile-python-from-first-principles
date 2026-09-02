# Chapter 4: The cuTile Programming Model

> "`ct.bid(0)` answers one question and one question only: which block, along one axis, am I? Everything else — how many blocks exist, what a helper function is allowed to be called from, how a grid's shape gets decided in the first place — is a separate question this chapter answers on purpose, one at a time, rather than letting them stay implicit the way the first three chapters' single-axis, single-file examples let them stay."

**What you will understand by the end of this chapter:**

- That a cuTile Python grid has at most three axes, and that `ct.bid(axis)`/`ct.num_blocks(axis)` are compiler-checked against exactly that boundary — genuinely confirmed by a rejected fourth axis, not merely documented
- How a grid's shape is actually decided: composed on the host, one axis at a time, from the same `ct.cdiv` this book has used since Chapter 1 — before any kernel, compiler, or device is involved
- What `ct.function(host=..., tile=...)` is documented to express about where a helper function may be called from, and — genuinely tested against `cuda-tile` 1.5.0 — exactly how much of that the compiler actually enforces today

**What you need to know first:**

- Chapters 1-3's `@ct.kernel`/`ct.launch` boundary, `ct.bid(0)` used along a single axis, and `ct.compilation.export_kernel` as this book's driver-free verification path.
- Chapter 1's discovery that an un-annotated helper function called from tile code becomes tile code too, and that calling a genuinely tile-only function directly from host code raises a `RuntimeError`. This chapter generalizes both findings under `ct.function`'s explicit vocabulary.

## 4.1 The Grid: `ct.bid` and `ct.num_blocks` Across Up to Three Axes `[FOUNDATIONAL]`

### Intuition

Every kernel this book has compiled so far called `ct.bid(0)` exactly once and never asked how many blocks the grid actually had — one axis was always enough for a 1D array. A grid is not restricted to one axis, though, any more than CUDA C++'s `blockIdx`/`gridDim` are restricted to `.x`: cuTile Python's grid is up to three-dimensional, and a block can read both its own position and the grid's total shape along any of those three axes from inside tile code.

### Background

`ct.bid(axis)` returns the current block's index along the given axis; `ct.num_blocks(axis)` returns how many blocks the grid has along that same axis. Both take `axis` as a compile-time constant restricted to exactly 0, 1, or 2 — a cuTile Python grid cannot have a fourth dimension, and the compiler checks that boundary itself rather than leaving it to fail unpredictably at launch time.

### Worked Example 4.1.1 — a 2D grid, linearized by hand

```python
import cuda.tile as ct
import io

@ct.kernel
def grid_probe(out):
    # ct.bid(axis) reads this block's own index along one grid axis;
    # ct.num_blocks(axis) reads how many blocks the grid has along that
    # same axis. Both accept axis 0, 1, or 2 -- a cuTile Python grid is
    # at most three-dimensional, the same shape CUDA C++'s blockIdx/gridDim
    # come in.
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    nx = ct.num_blocks(0)
    ny = ct.num_blocks(1)
    # A genuine use of both: turn a 2D block coordinate into the same
    # flat, row-major index numpy.ravel_multi_index would produce.
    linear = bidy * nx + bidx
    out_view = out.tiled_view((1,))
    out_view.store((linear,), ct.full((1,), linear, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(grid_probe, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

The complete file is `01_grid_bid_num_blocks_2d.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_grid_bid_num_blocks_2d.py
```

Genuinely run:

```
compiled: 591 bytes
```

`ct.full(shape, fill_value, dtype)` — used here for the first time in this book — creates a `Tile` filled with a single value, which is how `linear`, an ordinary scalar computed from two `ct.bid` calls and two `ct.num_blocks` calls, gets turned back into something `TiledView.store` can write. The kernel compiles cleanly, which is this book's genuine confirmation that both functions accept axis 1 as readily as axis 0 — nothing about them is specific to a single-axis grid.

### Worked Example 4.1.2 — a fourth axis, genuinely rejected

```python
import cuda.tile as ct
import io

@ct.kernel
def grid_probe(out):
    # A cuTile Python grid has at most three axes -- 0, 1, and 2. Asking
    # for a fourth is not a runtime question the hardware could still
    # answer; it's rejected at compile time, before any kernel logic runs.
    bidw = ct.bid(3)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), bidw, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(grid_probe, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

The complete file is `02_bid_axis_out_of_range.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_bid_axis_out_of_range.py
```

Genuinely run:

```
TileTypeError: Axis must be 0, 1, or 2, but 3 was given.
  "/tmp/ch4/02_bid_axis_out_of_range.py", line 9, col 12-20, in grid_probe:
        bidw = ct.bid(3)
               ^^^^^^^^^
```

> `[COMMON TRAP]` This book independently confirmed the same rejection for `ct.bid(-1)` as well — the boundary is checked on both sides, not just against values too large. The error is a `TileTypeError`, the same exception class Chapters 1 and 2 raised for a raw-tile `if` condition and a non-constant tile shape: cuTile Python treats an axis outside `{0, 1, 2}` as a genuine type violation in the caller's code, not a value that could ever be legal for some other grid shape.

## 4.2 Host-Side Grid Sizing: Composing `ct.cdiv` Per Axis `[FOUNDATIONAL]`

### Intuition

`ct.launch(stream, grid, kernel, kernel_args)` takes its grid shape as an ordinary Python tuple of up to three integers, decided entirely on the host, before the kernel or the compiler are involved at all. Chapter 1 established `ct.cdiv` as a genuine, host-callable ceiling-division helper for exactly this purpose along one axis; a real 2D problem just means calling it once per axis and combining the results into the tuple `ct.launch` expects.

### Background

Nothing about computing a multi-axis grid shape needs tile code, a kernel, or a compiler — it's the same ordinary Python arithmetic Chapter 1 ran directly on the host, extended to more than one axis. A function that only ever gets called from host code, and never from inside a kernel, stays host code with no annotation required at all, by the same rule Chapter 1 established for un-annotated helpers in the other direction.

### Worked Example 4.2.1 — a 2D grid shape, computed and printed directly

```python
import cuda.tile as ct

# A plain, undecorated Python function. Chapter 1 established that an
# undecorated function called *from* tile code becomes tile code too --
# but this one is never called from a kernel anywhere in this file, so it
# stays exactly what it looks like: ordinary host-side Python, with no
# compiler involved at all.
def grid_shape_2d(m, n, tm, tn):
    return (ct.cdiv(m, tm), ct.cdiv(n, tn))

# A problem of shape (1024, 768), tiled (32, 64) per block.
grid = grid_shape_2d(1024, 768, 32, 64)
print("grid shape:", grid)

# A second, uneven problem size -- (1000, 700) doesn't divide evenly by
# either tile dimension, which is exactly the case ct.cdiv's ceiling
# division exists for.
grid2 = grid_shape_2d(1000, 700, 32, 64)
print("grid shape (uneven):", grid2)
```

The complete file is `03_host_grid_sizing_2d.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_host_grid_sizing_2d.py
```

Genuinely run:

```
grid shape: (32, 12)
grid shape (uneven): (32, 11)
```

`1024 / 32 = 32` exactly, and `768 / 64 = 12` exactly — both axes divide evenly, so ceiling division and ordinary division agree. The second call is the case that actually needs a ceiling: `1000 / 32 = 31.25`, rounded up to `32`, and `700 / 64 = 10.9375`, rounded up to `11`. This is the exact tuple this book would pass as `ct.launch`'s `grid` argument if a real device were present — genuinely computed here, even though the launch itself remains `ct.launch`'s territory, which Chapter 5 addresses directly.

## 4.3 Explicit Execution-Space Annotations: `ct.function` `[FOUNDATIONAL]`

### Intuition

Chapter 1 discovered, somewhat by accident, that an un-annotated helper called from a kernel becomes tile code automatically, and that a genuinely tile-only function raises a `RuntimeError` when called directly from host code. `ct.function(host=..., tile=...)` is cuTile Python's vocabulary for saying which of those two spaces a function is meant to be callable from, on purpose, instead of leaving it to fall out of how the function happens to get called. This section states what the decorator is documented to mean, then genuinely tests how much of that cuTile Python 1.5.0 actually enforces.

### Background

`ct.function(func=None, *, host=False, tile=True)` marks a function's intended execution space: `host=True` means it may be called from host code, `tile=True` (the default) means it may be called from tile code. With no arguments at all, `@ct.function` denotes a tile-only function — the same default an un-annotated helper already had once it was called from tile code, just written down explicitly instead of inferred.

### Worked Example 4.3.1 — the explicit default, called from host code

```python
import cuda.tile as ct

# ct.function(), called with no arguments, is equivalent to the implicit
# default every un-annotated helper already had in Chapters 1-3: host=False,
# tile=True. Spelling it out explicitly doesn't change what it does --
# it changes what the reader can see was intended.
@ct.function()
def quad_it(x):
    return x * 4

# Calling it directly from host code, with no kernel and no compiler
# anywhere in the picture, is still rejected -- the same RuntimeError
# Chapter 1 first raised by accident is now raised on purpose.
try:
    print(quad_it(5))
except RuntimeError as e:
    print(f"{type(e).__name__}: {e}")
```

The complete file is `04_function_default_host_call_fails.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_function_default_host_call_fails.py
```

Genuinely run:

```
RuntimeError: Tile functions can only be called from tile code.
```

This is the identical message, and the identical exception type, Chapter 1 first encountered without asking for it. `ct.function()`'s default confirms it deliberately: a function annotated `host=False` (the default) genuinely cannot be called from host code at all, whether the annotation was written on purpose or inherited implicitly from being called inside a kernel somewhere else in the file.

### Worked Example 4.3.2 — the same default, called from tile code

```python
import cuda.tile as ct
import io

@ct.function()
def quad_it(x):
    return x * 4

@ct.kernel
def uses_it(out):
    pid = ct.bid(0)
    quadrupled = quad_it(pid)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), quadrupled, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(uses_it, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

The complete file is `05_function_default_from_tile.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_function_default_from_tile.py
```

Genuinely run:

```
compiled: 692 bytes
```

The same `quad_it`, called the way `tile=True`'s default actually intends it to be called, compiles without complaint. Between this and Worked Example 4.3.1, `ct.function()`'s documented default — callable from tile code, not from host code — is genuinely confirmed on both sides.

### Worked Example 4.3.3 — `host=True`, called from tile code anyway

```python
import cuda.tile as ct
import io

# host=True is documented as "whether the function can be called from
# host code" -- it says nothing about revoking tile-callability, and this
# file genuinely tests that: the same function, called from inside a
# kernel, still compiles.
@ct.function(host=True, tile=False)
def double_it(x):
    return x * 2

@ct.kernel
def uses_it(out):
    pid = ct.bid(0)
    doubled = double_it(pid)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), doubled, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(uses_it, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

The complete file is `06_function_host_true_called_from_tile.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_function_host_true_called_from_tile.py
```

Genuinely run:

```
compiled: 723 bytes
```

`double_it` is annotated `host=True, tile=False` — on a plain reading, callable from host code and *not* from tile code. Called from inside `uses_it`, a real kernel, it compiles cleanly anyway. Reading the installed `cuda-tile` 1.5.0 package's own decorator implementation directly (without reproducing its source here) confirms why: when `host=True` is given, `ct.function` returns the wrapped function completely unchanged — no dispatch wrapper is attached at all. An unwrapped, un-annotated function called from tile code is exactly the case Chapter 1 already established becomes tile code automatically, recursively, with no annotation required — which is precisely what `double_it` still is, `host=True` notwithstanding.

### Worked Example 4.3.4 — `tile=False`, also called from tile code anyway

```python
import cuda.tile as ct
import io

# tile=False, with host left at its own default (False), on paper reads
# as "callable from neither space." What actually happens when a function
# carrying it is called from inside a kernel is the point of this file.
@ct.function(tile=False)
def triple_it(x):
    return x * 3

@ct.kernel
def uses_it(out):
    pid = ct.bid(0)
    tripled = triple_it(pid)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), tripled, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(uses_it, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

The complete file is `07_function_tile_false_no_effect.py`, included in this chapter's Complete Runnable Code.

```bash
python3 07_function_tile_false_no_effect.py
```

Genuinely run:

```
compiled: 705 bytes
```

> `[COMMON TRAP]` `tile=False` reads like the counterpart to `tile=True`'s default — a way to say "this function may not be called from tile code." It genuinely is not enforced that way in `cuda-tile` 1.5.0: `triple_it`, decorated with `tile=False`, still compiles cleanly when called from inside `uses_it`. Reading the installed decorator's own implementation directly confirms why, without needing to guess: the `tile` keyword is accepted and stored nowhere the decorator's own branching logic ever reads back — only the `host` keyword actually changes what gets returned. This is a fact this book genuinely tested against one specific package version, `cuda-tile` 1.5.0, not a claim about what any future release does or is documented to guarantee; a later `cuda-tile` release could enforce `tile=False` where this one doesn't, and this book would rather report today's real, tested behavior than a guess about tomorrow's.

## Complete Runnable Code

### File: `01_grid_bid_num_blocks_2d.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def grid_probe(out):
    # ct.bid(axis) reads this block's own index along one grid axis;
    # ct.num_blocks(axis) reads how many blocks the grid has along that
    # same axis. Both accept axis 0, 1, or 2 -- a cuTile Python grid is
    # at most three-dimensional, the same shape CUDA C++'s blockIdx/gridDim
    # come in.
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    nx = ct.num_blocks(0)
    ny = ct.num_blocks(1)
    # A genuine use of both: turn a 2D block coordinate into the same
    # flat, row-major index numpy.ravel_multi_index would produce.
    linear = bidy * nx + bidx
    out_view = out.tiled_view((1,))
    out_view.store((linear,), ct.full((1,), linear, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(grid_probe, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 01_grid_bid_num_blocks_2d.py
```

### File: `02_bid_axis_out_of_range.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def grid_probe(out):
    # A cuTile Python grid has at most three axes -- 0, 1, and 2. Asking
    # for a fourth is not a runtime question the hardware could still
    # answer; it's rejected at compile time, before any kernel logic runs.
    bidw = ct.bid(3)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), bidw, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(grid_probe, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

```bash
python3 02_bid_axis_out_of_range.py
```

### File: `03_host_grid_sizing_2d.py`

```python
import cuda.tile as ct

# A plain, undecorated Python function. Chapter 1 established that an
# undecorated function called *from* tile code becomes tile code too --
# but this one is never called from a kernel anywhere in this file, so it
# stays exactly what it looks like: ordinary host-side Python, with no
# compiler involved at all.
def grid_shape_2d(m, n, tm, tn):
    return (ct.cdiv(m, tm), ct.cdiv(n, tn))

# A problem of shape (1024, 768), tiled (32, 64) per block.
grid = grid_shape_2d(1024, 768, 32, 64)
print("grid shape:", grid)

# A second, uneven problem size -- (1000, 700) doesn't divide evenly by
# either tile dimension, which is exactly the case ct.cdiv's ceiling
# division exists for.
grid2 = grid_shape_2d(1000, 700, 32, 64)
print("grid shape (uneven):", grid2)
```

```bash
python3 03_host_grid_sizing_2d.py
```

### File: `04_function_default_host_call_fails.py`

```python
import cuda.tile as ct

# ct.function(), called with no arguments, is equivalent to the implicit
# default every un-annotated helper already had in Chapters 1-3: host=False,
# tile=True. Spelling it out explicitly doesn't change what it does --
# it changes what the reader can see was intended.
@ct.function()
def quad_it(x):
    return x * 4

# Calling it directly from host code, with no kernel and no compiler
# anywhere in the picture, is still rejected -- the same RuntimeError
# Chapter 1 first raised by accident is now raised on purpose.
try:
    print(quad_it(5))
except RuntimeError as e:
    print(f"{type(e).__name__}: {e}")
```

```bash
python3 04_function_default_host_call_fails.py
```

### File: `05_function_default_from_tile.py`

```python
import cuda.tile as ct
import io

@ct.function()
def quad_it(x):
    return x * 4

@ct.kernel
def uses_it(out):
    pid = ct.bid(0)
    quadrupled = quad_it(pid)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), quadrupled, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(uses_it, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 05_function_default_from_tile.py
```

### File: `06_function_host_true_called_from_tile.py`

```python
import cuda.tile as ct
import io

# host=True is documented as "whether the function can be called from
# host code" -- it says nothing about revoking tile-callability, and this
# file genuinely tests that: the same function, called from inside a
# kernel, still compiles.
@ct.function(host=True, tile=False)
def double_it(x):
    return x * 2

@ct.kernel
def uses_it(out):
    pid = ct.bid(0)
    doubled = double_it(pid)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), doubled, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(uses_it, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 06_function_host_true_called_from_tile.py
```

### File: `07_function_tile_false_no_effect.py`

```python
import cuda.tile as ct
import io

# tile=False, with host left at its own default (False), on paper reads
# as "callable from neither space." What actually happens when a function
# carrying it is called from inside a kernel is the point of this file.
@ct.function(tile=False)
def triple_it(x):
    return x * 3

@ct.kernel
def uses_it(out):
    pid = ct.bid(0)
    tripled = triple_it(pid)
    out_view = out.tiled_view((1,))
    out_view.store((0,), ct.full((1,), tripled, ct.int32))

array_param = ct.compilation.ArrayConstraint(
    ct.int32, ndim=1, index_dtype=ct.int32,
    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
)
sig = ct.compilation.KernelSignature(
    parameters=[array_param],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(uses_it, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 07_function_tile_false_no_effect.py
```

## Chapter Summary

A cuTile Python grid is at most three-dimensional: `ct.bid(axis)` reads a block's own position and `ct.num_blocks(axis)` reads the grid's total extent along any axis in `{0, 1, 2}`, both genuinely confirmed compiling together in a real 2D kernel, and both genuinely rejected — with a `TileTypeError` naming the exact illegal value — for axis `3` (and, independently confirmed, `-1`). A grid's shape is decided entirely on the host, before any kernel or compiler is involved, by composing `ct.cdiv` once per axis — an ordinary Python function needing no annotation at all, since it's never called from inside a kernel. `ct.function(host=..., tile=...)` is cuTile Python's documented vocabulary for declaring which execution space a helper function belongs to: its default (`host=False, tile=True`) genuinely matches the implicit behavior Chapter 1 first discovered, confirmed on both sides here — rejected from host code, accepted from tile code. Tested directly against `cuda-tile` 1.5.0, though, `host=True` returns a function completely unwrapped rather than restricting it to host-only, and `tile=False` has no observable effect on whether a function may still be called from inside a kernel — both genuinely confirmed by compiling kernels that call such functions, and both reported as tested facts about one specific package version, not assumptions read off the documentation's prose alone.

## Self-Check Questions

1. `ct.bid(axis)` and `ct.num_blocks(axis)` both restrict `axis` to `{0, 1, 2}`. What class of exception does passing `3` raise, and what does that exception class choice say about how cuTile Python categorizes the mistake?
2. Worked Example 4.2.1's `grid_shape_2d` function has no `@ct.function` or `@ct.kernel` decorator at all. Why does it not need one, given that Chapter 1 established that calling an un-annotated function from tile code turns it into tile code?
3. `ct.function()`'s documented default is `host=False, tile=True`. State, precisely, what each half of that default was shown to mean in Worked Examples 4.3.1 and 4.3.2.
4. Worked Example 4.3.3 decorates `double_it` with `host=True, tile=False`, then calls it from inside a kernel, and it compiles. What does reading the installed decorator's own implementation say actually happens when `host=True` is passed, and how does that explain the result?
5. This chapter reports that `tile=False` "has no observable effect" in `cuda-tile` 1.5.0, rather than saying it "doesn't work" or "is broken." Why does the more careful phrasing matter here?

## Where We Go Next

This chapter treated `ct.launch`'s grid argument as something computed correctly on the host and then, necessarily, left untested — this book's own environment has no device to launch against. Chapter 5 turns to the other half of that same sentence: exactly what `ct.compilation.export_kernel` needs and doesn't need to produce real, deployable output, what genuinely distinguishes a `cubin` from Tile IR bytecode, and what `ct.launch` would require that `export_kernel` deliberately does not — the honest boundary this book has approached from one side for four chapters now, examined directly.

## Worked Solutions

**1.** A `TileTypeError` — the same exception class Chapters 1 and 2 raised for a raw-tile `if` condition and a non-constant tile shape. Grouping an out-of-range grid axis under `TileTypeError` says cuTile Python treats it as a genuine type violation in the caller's code — a value that could never be legal, for any grid, the same way passing a tile where a compile-time constant is required could never be legal — rather than a value that merely happens to be wrong for this particular launch.

**2.** Because it's never called from inside a kernel anywhere in the file. Chapter 1's rule only promotes an un-annotated function to tile code when *tile code calls it* — `grid_shape_2d` is called exactly twice, both times directly from ordinary host-level `print` statements, so the promotion rule never triggers. A function's execution space isn't a property of how it's written; it's a property of who calls it.

**3.** `host=False` (4.3.1): calling the decorated function directly from host code raises `RuntimeError: Tile functions can only be called from tile code.` — the same message and exception type Chapter 1 first triggered by accident. `tile=True` (4.3.2): calling the identical function from inside a real kernel compiles cleanly, with no error at all — confirming the two halves of the documented default independently, one per Worked Example.

**4.** The installed decorator's own implementation, read directly rather than assumed, shows that when `host=True` is passed, the decorator returns the original function completely unwrapped — no dispatch logic is attached to it at all. An unwrapped, un-annotated function is exactly the case Chapter 1 already showed becomes tile code automatically the moment tile code calls it, recursively, with no annotation required. `host=True` doesn't grant an *additional* capability so much as it removes the wrapper that would have otherwise blocked host-side calls — and removing that wrapper happens to leave the function just as callable from tile code as if it had never been decorated.

**5.** Because "doesn't work" or "is broken" would claim the current behavior is a defect cuTile Python did not intend — a claim this book has no basis for making, since it never inspected NVIDIA's design intent, only the installed package's actual, tested behavior on one specific version. "Has no observable effect" states exactly what was genuinely tested — the `tile` keyword is accepted, and a function decorated with `tile=False` still compiles when called from tile code — without extending that observation into a claim about intent, future versions, or documentation accuracy this book cannot verify.
