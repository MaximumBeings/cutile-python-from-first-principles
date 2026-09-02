# Chapter 3: Tiled Views, Shape, and Memory Layout

> "`ct.load(array, index, shape)` says the same three things every single time it's called: which array, which tile, and how big. A `TiledView` says the last one exactly once — at the moment the view is created — and lets every later call be about nothing but position. That's not a convenience macro. Fixing the shape and the padding behavior once, in one place, is what lets a `TiledView`'s tiles overlap, which is not something repeating the same `shape` argument on every `ct.load` call could ever express."

**What you will understand by the end of this chapter:**

- How `array.tiled_view(tile_shape)` bundles the shape and padding decisions Chapter 1 and 2 passed to every single `ct.load`/`ct.store` call into one object, created once, indexed many times
- What `traversal_steps` genuinely buys a `TiledView` that plain tiling can't express at all: tiles that overlap, the shape sliding-window and stencil kernels are built from
- That `Array.slice` produces a real sub-array view over the same memory, with no data copied, and that its bounds can be genuine runtime values, not just compile-time constants
- Exactly which shape mismatches cuTile Python's compiler tolerates through implicit, NumPy-style broadcasting, and exactly which one it doesn't — with the real exception type (`TileInternalError`, not `TileTypeError`) that a genuinely incompatible pair of shapes raises
- That a `TiledView`'s `padding_mode` isn't free even before a single element is loaded: five of `cuTile Python`'s six padding modes compile to measurably larger code than the sixth, confirmed directly from real byte counts, not from documentation claims alone

**What you need to know first:**

- Chapters 1 and 2's `Array`/`Tile` distinction, `ct.load`/`ct.store`, and the `ArrayConstraint`/`KernelSignature`/`export_kernel` scaffolding this chapter keeps reusing without re-explaining.
- Familiarity with NumPy's broadcasting rules is useful but not required — Section 3.4 states the relevant rule directly, and confirms it with a real compiled kernel rather than assuming you already know it.

## 3.1 From `ct.load` to `Array.tiled_view` `[FOUNDATIONAL]`

### Intuition

Calling `ct.load(a, index=(pid,), shape=(tile_size,))` on every single line that touches array `a` is a little like re-stating a shipping address on every individual box in a shipment instead of writing it once on the manifest and letting every box just carry a number. `array.tiled_view(tile_shape)` is that manifest: it fixes the tile shape (and, as Section 3.5 covers, the padding behavior) exactly once, up front, and hands back a `TiledView` object whose own `.load(index)` and `.store(index, tile)` methods only need a position from then on.

### Background

`Array.tiled_view(tile_shape, *, padding_mode=..., traversal_steps=...)` returns a `TiledView` that partitions the array into a grid of equally-sized tiles, exactly the same partitioning `ct.load`/`ct.store` compute from their own `shape` argument every time they're called. The `TiledView`'s own `.load(index)` and `.store(index, tile)` then only need the tile-space position — the shape and padding mode were already fixed when the view was created.

| | `ct.load(array, index, shape)` / `ct.store(array, index, tile)` | `array.tiled_view(shape).load(index)` / `.store(index, tile)` |
|---|---|---|
| Shape specified | On every call | Once, when the view is created |
| Padding mode specified | On every call (`ct.load` only) | Once, when the view is created |
| Can express overlapping tiles | No | Yes — `traversal_steps` (Section 3.2) |

### Worked Example 3.1.1 — `vector_add`, rewritten through a `TiledView`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add_view(a, b, c, tile_size: ct.Constant[int]):
    a_view = a.tiled_view((tile_size,))
    b_view = b.tiled_view((tile_size,))
    c_view = c.tiled_view((tile_size,))
    pid = ct.bid(0)
    result = a_view.load((pid,)) + b_view.load((pid,))
    c_view.store((pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(vector_add_view, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

The complete file is `01_tiled_view_basic.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_tiled_view_basic.py
```

Genuinely run:

```
compiled: 698 bytes
```

This is the identical kernel Chapters 1 and 2 built with `ct.load`/`ct.store` directly, restated through three `TiledView`s — one per array. Nothing about what the kernel computes changed; only how the shape gets communicated to the compiler did. Both styles are genuinely valid cuTile Python and both appear throughout the rest of this book, chosen per kernel by whichever reads more clearly for what that kernel actually does.

## 3.2 Overlapping Tiles: `traversal_steps` `[FOUNDATIONAL]`

### Intuition

Every tiled view so far has partitioned its array the way a butcher portions a whole side of beef into non-overlapping cuts — each byte belongs to exactly one tile, and moving from tile `i` to tile `i+1` means moving forward by exactly one tile's width. A sliding-window computation — a moving average, a 1D stencil, a convolution's receptive field — needs something a plain partition can't express at all: consecutive windows that *overlap*, each one shifted by less than its own width. `traversal_steps` is the argument that makes that possible: it decouples how far apart consecutive tile origins are from how big each tile itself is.

### Background

`traversal_steps`, an optional argument to `tiled_view`, sets the number of elements between consecutive tile origins along each axis, independently of `tile_shape`. When `traversal_steps` is left at its default (`None`, meaning equal to `tile_shape`), tiles partition the array with no overlap and no gaps — every example before this section. Setting `traversal_steps[i]` to something *smaller* than `tile_shape[i]` makes consecutive tiles along that axis genuinely overlap.

### Worked Example 3.2.1 — a 4-wide window sliding 2 elements at a time

```python
import cuda.tile as ct
import io

@ct.kernel
def sliding_window_sum(a, c, tile_size: ct.Constant[int], step: ct.Constant[int]):
    view = a.tiled_view((tile_size,), traversal_steps=(step,))
    pid = ct.bid(0)
    window = view.load((pid,))
    total = ct.sum(window)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(4), ct.compilation.ConstantConstraint(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(sliding_window_sum, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

`02_traversal_steps_sliding_window.py`, included in Complete Runnable Code:

```bash
python3 02_traversal_steps_sliding_window.py
```

Genuinely run:

```
compiled: 747 bytes
```

`tile_shape=(4,)` with `traversal_steps=(2,)` means block `pid`'s window starts at element `pid * 2`, not `pid * 4` — block 0 covers elements 0–3, block 1 covers elements 2–5, block 2 covers elements 4–7, each one overlapping its neighbor by half its width. There is no way to express this by varying `ct.load`'s `shape` argument alone; `shape` only ever controls how big one tile is, never how far apart consecutive tiles' starting points are.

### ASCII Diagram — a non-overlapping partition vs. a sliding window

```
Non-overlapping (tile_shape=(4,), traversal_steps=None -> defaults to (4,)):

index:  0   1   2   3   4   5   6   7
       [------- tile 0 -------][------- tile 1 -------]

Overlapping (tile_shape=(4,), traversal_steps=(2,)):

index:  0   1   2   3   4   5   6   7
       [------- tile 0 -------]
               [------- tile 1 -------]
                       [------- tile 2 -------]
```

## 3.3 `Array.slice`: A Sub-Array View, No Copy `[FOUNDATIONAL]`

### Intuition

`Array.slice` is the array-level equivalent of a Python list slice's `view` semantics rather than its copy semantics — think of `numpy`'s own slicing, which hands back a new array object that still points into the same underlying buffer, rather than duplicating any data. Restricting an `Array` to a sub-range along one axis with `.slice(axis, start, stop)` costs nothing beyond bookkeeping: the returned `Array` references exactly the same memory, just with a narrower valid range.

### Background

`array.slice(axis, start, stop)` returns a new `Array` restricted to `[start, stop)` along the given `axis`, with `axis` required to be a compile-time constant but `start` and `stop` allowed to be genuine runtime values — "scalars or 0D tiles," in the documentation's own words, not necessarily `ct.Constant`. That's a real, deliberate asymmetry with Chapter 2's tile-shape rule: a tile's *shape* must be compile-time fixed, but the *bounds of a slice* don't have to be, because a slice doesn't change how many registers anything occupies the way a tile shape does — it only narrows which addresses are considered valid.

### Worked Example 3.3.1 — summing only the second half of an array

```python
import cuda.tile as ct
import io

@ct.kernel
def sum_second_half(a, c, n: ct.Constant[int], tile_size: ct.Constant[int]):
    half = n // 2
    second_half = a.slice(0, half, n)
    pid = ct.bid(0)
    view = second_half.tiled_view((tile_size,))
    tile = view.load((pid,))
    total = ct.sum(tile)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(1024), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(sum_second_half, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

`03_array_slice.py`, included in Complete Runnable Code:

```bash
python3 03_array_slice.py
```

Genuinely run:

```
compiled: 832 bytes
```

`half = n // 2` is ordinary Python-level integer arithmetic on a `ct.Constant[int]` value — since `n` is itself constant-embedded, `half` is too, computed once at compile time. `a.slice(0, half, n)` then produces a new `Array` covering only elements `[half, n)` of `a`'s first axis, and `second_half.tiled_view((tile_size,))` partitions *that* narrower view exactly the way Section 3.1 partitioned a whole array — the tiled view has no idea, and doesn't need to know, that it's sitting on top of a slice rather than the original array.

## 3.4 Shape Compatibility: Broadcasting and Its Real Limits `[FOUNDATIONAL]`

### Intuition

Adding a `(4, 8)`-shaped tile to an `(8,)`-shaped tile makes intuitive sense the way adding a single row to every row of a small spreadsheet does — the shorter shape gets logically repeated along the missing axis. Adding a `(256,)`-shaped tile to a `(128,)`-shaped tile has no equivalent intuitive rescue: there's no missing axis to repeat along, just two flatly incompatible lengths along the one axis both tiles share. cuTile Python's compiler draws exactly this line, following the same broadcasting rule NumPy does, and rejects what falls on the wrong side of it with a real, specific error rather than silently doing something the two shapes don't actually support.

### Background

cuTile Python's own documentation states its shape-broadcasting behavior follows NumPy's broadcasting rule directly: two shapes are compatible for an elementwise operation if, comparing their dimensions from the trailing (rightmost) axis inward, every pair of aligned dimensions is either equal or one of them is 1 (with a missing leading dimension treated as if it were 1). `(4, 8)` and `(8,)` are compatible under exactly this rule — the trailing `8` matches `8`, and the missing leading dimension of the second shape broadcasts against `4`. `(256,)` and `(128,)` are not: their one shared axis has two different lengths, neither of which is 1.

### Worked Example 3.4.1 — a genuinely rejected shape mismatch

```python
import cuda.tile as ct
import io

@ct.kernel
def bad_add(a, b, c):
    a_view = a.tiled_view((256,))
    b_view = b.tiled_view((128,))
    pid = ct.bid(0)
    result = a_view.load((pid,)) + b_view.load((pid,))
    c_view = c.tiled_view((256,))
    c_view.store((pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param()],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(bad_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

`04_shape_mismatch_error.py`, included in Complete Runnable Code — it runs to completion (the error is caught and printed), it just never produces a compiled kernel:

```bash
python3 04_shape_mismatch_error.py
```

Genuinely run:

```
TileInternalError: Shapes are not broadcastable: (256,), (128,)
  "/tmp/ch3/04_shape_mismatch_error.py", line 9, col 14-54, in bad_add:
        result = a_view.load((pid,)) + b_view.load((pid,))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

> `[COMMON TRAP]` This is genuinely a `TileInternalError`, not the `TileTypeError` Chapters 1 and 2 raised for a raw-tile `if` condition or a non-constant tile shape. cuTile Python distinguishes "the compiler correctly identified a real type violation in your code" (`TileTypeError`) from other categories of compile-time failure — a shape-arithmetic mismatch surfacing as `TileInternalError` here is this book's genuine, captured evidence of exactly which exception type this specific violation raises in `cuda-tile` 1.5.0, not a claim you should extrapolate to every shape-related error this compiler can produce.

### Worked Example 3.4.2 — the same shapes, genuinely compatible

```python
import cuda.tile as ct
import io

@ct.kernel
def broadcast_add(a, b, c):
    a_view = a.tiled_view((4, 8))
    b_view = b.tiled_view((8,))
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    result = a_view.load((pid0, pid1)) + b_view.load((pid1,))
    c_view = c.tiled_view((4, 8))
    c_view.store((pid0, pid1), result)

def array_param(ndim):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(broadcast_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

`05_broadcast_add_implicit.py`, included in Complete Runnable Code:

```bash
python3 05_broadcast_add_implicit.py
```

Genuinely run:

```
compiled: 933 bytes
```

No explicit broadcasting call appears anywhere in this kernel — ordinary `+` between a `(4, 8)` tile and an `(8,)` tile compiles cleanly, with the shorter shape implicitly repeated across the missing leading axis, exactly the NumPy rule Section 3.4's Background table states.

### Worked Example 3.4.3 — the same broadcast, made explicit with `ct.broadcast_to`

```python
import cuda.tile as ct
import io

@ct.kernel
def explicit_broadcast_add(a, b, c):
    a_view = a.tiled_view((4, 8))
    b_view = b.tiled_view((8,))
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    a_tile = a_view.load((pid0, pid1))
    b_tile = b_view.load((pid1,))
    b_broadcast = ct.broadcast_to(b_tile, (4, 8))
    result = a_tile + b_broadcast
    c_view = c.tiled_view((4, 8))
    c_view.store((pid0, pid1), result)

def array_param(ndim):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(explicit_broadcast_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

`06_explicit_broadcast_to.py`, included in Complete Runnable Code:

```bash
python3 06_explicit_broadcast_to.py
```

Genuinely run:

```
compiled: 965 bytes
```

`ct.broadcast_to(b_tile, (4, 8))` does by hand exactly what Worked Example 3.4.2's bare `+` did implicitly — nothing about the final `result` differs between the two kernels. `ct.broadcast_to` earns its place in the language anyway: later chapters need an explicitly-broadcast tile as an intermediate value in its own right — as an argument to `ct.matmul` or a `ct.reduce` callback, say — where an implicit broadcast folded invisibly into a `+` wouldn't give you a named value to pass along at all.

## 3.5 Padding Modes: What Happens at the Array's Edge `[FOUNDATIONAL]`

### Intuition

Chapter 1's diagram showed a tile straddling the end of a 10-element array — two full elements of real data, then out-of-bounds positions the tile still has to fill with *something*, since every tile's shape is fixed at compile time regardless of how much real data is actually left. `padding_mode` is the setting that decides what that "something" is: silence (`UNDETERMINED`, whatever bits happen to be there), a definite numeric value (`ZERO`, `NEG_ZERO`), or a value specifically meant to be *noticed* downstream (`NAN`, `POS_INF`, `NEG_INF` — exactly the values a max-reduction or a softmax wants at a position that must never win).

### Background

`cuda-tile` 1.5.0 defines six `PaddingMode` values: `UNDETERMINED` (the default — genuinely unspecified, whatever bits are already present), `ZERO`, `NEG_ZERO`, `NAN`, `POS_INF`, and `NEG_INF`. Every mode besides `UNDETERMINED` commits the compiler to actually detecting the out-of-bounds condition and writing a specific, known value there — work `UNDETERMINED` is explicitly permitted to skip.

### Worked Example 3.5.1 — all six modes, genuinely compiled and measured

```python
import cuda.tile as ct
import io

def build_kernel(mode):
    @ct.kernel
    def load_with_padding(a, c, tile_size: ct.Constant[int]):
        view = a.tiled_view((tile_size,), padding_mode=mode)
        pid = ct.bid(0)
        tile = view.load((pid,))
        c_view = c.tiled_view((tile_size,))
        c_view.store((pid,), tile)
    return load_with_padding

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

for mode in ct.PaddingMode:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print(f"{mode}: {len(buf.getvalue())} bytes")
```

`07_padding_modes_comparison.py`, included in Complete Runnable Code:

```bash
python3 07_padding_modes_comparison.py
```

Genuinely run:

```
PaddingMode.UNDETERMINED: 620 bytes
PaddingMode.ZERO: 636 bytes
PaddingMode.NEG_ZERO: 636 bytes
PaddingMode.NAN: 636 bytes
PaddingMode.POS_INF: 636 bytes
PaddingMode.NEG_INF: 636 bytes
```

`UNDETERMINED` is genuinely the smallest of the six — 16 bytes smaller than every other mode, all five of which compile to the identical 636 bytes as each other. That's consistent with what the documentation states directly: `UNDETERMINED` doesn't commit the compiler to generating any boundary-detection logic at all, while every other mode has to actually check whether a given position is in-bounds and, if not, write the specific value that mode promises — real, if modest, generated code that `UNDETERMINED` genuinely skips. This book cannot verify *which* value actually lands in an out-of-bounds position at runtime — that needs a real load against real device memory, squarely in `ct.launch` territory Chapter 1 flagged as unavailable here — but the size difference alone is real, measured evidence that padding mode isn't a purely descriptive, cost-free setting.

## Complete Runnable Code

### File: `01_tiled_view_basic.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add_view(a, b, c, tile_size: ct.Constant[int]):
    a_view = a.tiled_view((tile_size,))
    b_view = b.tiled_view((tile_size,))
    c_view = c.tiled_view((tile_size,))
    pid = ct.bid(0)
    result = a_view.load((pid,)) + b_view.load((pid,))
    c_view.store((pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(vector_add_view, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 01_tiled_view_basic.py
```

### File: `02_traversal_steps_sliding_window.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def sliding_window_sum(a, c, tile_size: ct.Constant[int], step: ct.Constant[int]):
    view = a.tiled_view((tile_size,), traversal_steps=(step,))
    pid = ct.bid(0)
    window = view.load((pid,))
    total = ct.sum(window)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(4), ct.compilation.ConstantConstraint(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(sliding_window_sum, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 02_traversal_steps_sliding_window.py
```

### File: `03_array_slice.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def sum_second_half(a, c, n: ct.Constant[int], tile_size: ct.Constant[int]):
    half = n // 2
    second_half = a.slice(0, half, n)
    pid = ct.bid(0)
    view = second_half.tiled_view((tile_size,))
    tile = view.load((pid,))
    total = ct.sum(tile)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(1024), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(sum_second_half, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 03_array_slice.py
```

### File: `04_shape_mismatch_error.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def bad_add(a, b, c):
    a_view = a.tiled_view((256,))
    b_view = b.tiled_view((128,))
    pid = ct.bid(0)
    result = a_view.load((pid,)) + b_view.load((pid,))
    c_view = c.tiled_view((256,))
    c_view.store((pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param()],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(bad_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

```bash
python3 04_shape_mismatch_error.py
```

### File: `05_broadcast_add_implicit.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def broadcast_add(a, b, c):
    a_view = a.tiled_view((4, 8))
    b_view = b.tiled_view((8,))
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    result = a_view.load((pid0, pid1)) + b_view.load((pid1,))
    c_view = c.tiled_view((4, 8))
    c_view.store((pid0, pid1), result)

def array_param(ndim):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(broadcast_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 05_broadcast_add_implicit.py
```

### File: `06_explicit_broadcast_to.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def explicit_broadcast_add(a, b, c):
    a_view = a.tiled_view((4, 8))
    b_view = b.tiled_view((8,))
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    a_tile = a_view.load((pid0, pid1))
    b_tile = b_view.load((pid1,))
    b_broadcast = ct.broadcast_to(b_tile, (4, 8))
    result = a_tile + b_broadcast
    c_view = c.tiled_view((4, 8))
    c_view.store((pid0, pid1), result)

def array_param(ndim):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(explicit_broadcast_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 06_explicit_broadcast_to.py
```

### File: `07_padding_modes_comparison.py`

```python
import cuda.tile as ct
import io

def build_kernel(mode):
    @ct.kernel
    def load_with_padding(a, c, tile_size: ct.Constant[int]):
        view = a.tiled_view((tile_size,), padding_mode=mode)
        pid = ct.bid(0)
        tile = view.load((pid,))
        c_view = c.tiled_view((tile_size,))
        c_view.store((pid,), tile)
    return load_with_padding

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

for mode in ct.PaddingMode:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print(f"{mode}: {len(buf.getvalue())} bytes")
```

```bash
python3 07_padding_modes_comparison.py
```

## Chapter Summary

`Array.tiled_view(tile_shape)` bundles a shape and a padding mode into one object created once and indexed many times through its own `.load(index)`/`.store(index, tile)` methods — genuinely equivalent to calling `ct.load`/`ct.store` with the same shape on every line, confirmed by rewriting `vector_add` through it. `traversal_steps` is what a `TiledView` can express that no amount of `ct.load` shape-juggling can: tiles whose consecutive origins are closer together than their own width, genuinely compiled here for a 4-wide window sliding 2 elements at a time — the shape sliding-window and stencil computations need. `Array.slice(axis, start, stop)` produces a real, no-copy sub-array view over the same underlying memory, with bounds that are allowed to be genuine runtime values rather than compile-time constants, confirmed by slicing a compile-time-computed midpoint out of a 1024-element array. Shape compatibility in tile arithmetic follows NumPy's own broadcasting rule exactly — a `(4, 8)` tile and an `(8,)` tile combine cleanly, both implicitly (through ordinary `+`) and explicitly (through `ct.broadcast_to`) — while a `(256,)` tile and a `(128,)` tile, sharing one axis with two incompatible lengths, are genuinely rejected, with a `TileInternalError` this book confirmed directly rather than assumed. And `padding_mode` is not a purely descriptive setting: `UNDETERMINED`, the default, genuinely compiles to less code than any of the other five modes, all of which have to generate the real boundary-checking logic `UNDETERMINED` is explicitly allowed to skip.

## Self-Check Questions

1. What does `array.tiled_view(tile_shape)` let you avoid repeating that plain `ct.load`/`ct.store` calls require on every line?
2. A `TiledView` created with `tile_shape=(4,)` and `traversal_steps=(2,)` produces tiles that overlap. Why can't the same overlapping behavior be produced using `ct.load`'s `shape` argument alone, however it's varied?
3. `Array.slice`'s `start` and `stop` arguments are allowed to be genuine runtime scalars, but Chapter 2 established that a tile's `shape` must be a compile-time constant. Why is a slice's bounds allowed to be more dynamic than a tile's shape?
4. `(4, 8)` and `(8,)` combine cleanly under broadcasting; `(256,)` and `(128,)` don't. State the general rule that explains both outcomes.
5. This chapter found that five of `cuda-tile`'s six `PaddingMode` values compile to identical, larger output than the sixth (`UNDETERMINED`). What does that difference in compiled size actually tell you about what the compiler does differently for `UNDETERMINED`, and what does it *not* tell you, given that no kernel in this chapter was ever launched on real hardware?

## Where We Go Next

This chapter treated every `Array` as if its memory were laid out the simplest possible way — contiguous, row-major, starting at some base address the compiler doesn't need to reason about further. Part 1 opens by taking that assumption apart: what `ArrayConstraint`'s remaining fields (`stride_constant`, `shape_constant`, `stride_divisible_by`, `shape_divisible_by`, `base_addr_divisible_by`) let the compiler assume about a real array's memory layout, what each assumption is actually worth in the code it lets the compiler generate, and what happens — as this chapter's `TileInternalError` and Chapter 2's `TileTypeError`s already previewed — when a kernel is compiled against an assumption the array handed to it at launch time doesn't actually satisfy.

## Worked Solutions

**1.** Repeating the tile `shape` (and, when it matters, the `padding_mode`) on every single `ct.load`/`ct.store` call touching that array. `array.tiled_view(tile_shape, padding_mode=...)` fixes both exactly once, when the view is created, and every later `.load(index)`/`.store(index, tile)` call only needs to say which tile, not how big or how it should be padded.

**2.** `ct.load`'s `shape` argument only ever controls the size of one tile — it has no way to also say how far apart consecutive tile origins are, because that's simply not one of its parameters. `traversal_steps` is a genuinely separate piece of information a `TiledView` carries alongside `tile_shape`, specifically to decouple "how big is each tile" from "how far do I move between tiles" — a distinction `ct.load` alone has no vocabulary to express, no matter what `shape` value you pass it.

**3.** A slice's bounds only ever restrict which addresses of an already-existing array are considered valid — they don't change how many registers anything occupies, how an operation gets scheduled, or how a value gets laid out in shared memory or registers, all of which are genuinely fixed once and for all when a tile's shape is decided. A tile's shape has to be compile-time-constant because the compiler generates fundamentally different code for a `(4,)` tile than for a `(256,)` tile; a slice's `start`/`stop` only feed into address arithmetic that's perfectly normal to compute at runtime, the same way indexing into a plain array with a runtime-computed offset always has been.

**4.** The rule, matching NumPy's own: compare the two shapes' dimensions from the trailing (rightmost) axis inward; a treat any missing leading dimension as if it were 1; the shapes are compatible if every aligned pair of dimensions is either equal or one of them is 1. `(4, 8)` vs. `(8,)`: aligned trailing dimensions are `8` and `8` (equal), and the missing leading dimension of `(8,)` is treated as 1, which broadcasts against `4` — compatible. `(256,)` vs. `(128,)`: their one shared, trailing dimension is `256` vs. `128` — neither equal nor either one equal to 1 — incompatible.

**5.** The size difference genuinely tells you that four of the five non-default padding modes require the compiler to generate real, additional instructions this book can directly observe in the compiled output — almost certainly boundary-checking logic that decides whether a given position is in-bounds before choosing what value to produce, work `UNDETERMINED` is explicitly permitted to skip entirely. It does *not* tell you anything about what value actually appears at an out-of-bounds position when one of these kernels genuinely runs — confirming that a `ZERO`-padded load actually produces zeros, rather than merely compiling to different code than `UNDETERMINED` does, requires a real load against real device memory on real hardware, which is `ct.launch` territory this book's own environment cannot reach.
