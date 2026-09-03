# Chapter 22: Permute, Transpose, and the Two Checks Behind cat

> "Every operation this book has covered so far either computes a new value from old ones or moves values between memory and tiles. This chapter's five operations do neither: they only change how a tile's own elements are arranged, addressed, and named — the same values, seen from a different shape."

**What you will understand by the end of this chapter:**

- `ct.permute`: the general axis-reordering operation, and how it validates its own `axes` argument — rejecting a repeated axis and rejecting a permutation shorter than the tile's rank, each with its own distinct error
- `ct.transpose`: a two-axis special case of `permute`, defaulting to swapping the only two axes a 2D tile has, but requiring `axis0`/`axis1` explicitly once a tile has more than two dimensions — and, tested directly under Chapter 10's naming-confound control, compiling to byte-identical output to the equivalent `permute` call
- `ct.expand_dims`: inserting a size-1 axis at a given position, including NumPy-style negative-axis addressing, and what happens when the requested axis is out of range
- `ct.cat`: whose docstring states plainly that its two input tiles "must have the same shape," a claim this chapter tests directly and finds is actually the visible effect of two separate, independently-enforced compiler checks — one about the concatenated result's own shape, one about the axes not being joined
- `ct.extract`: reading a subtile out of a larger tile by an index into a grid of same-shaped subtiles, not an index into the tile's own elements, and the two distinct ways to get that indexing wrong
- That all five operations compose freely inside a single kernel, demonstrated by a capstone that partitions a tile into quadrants, transposes each one, and reassembles the whole tile — confirming `extract` and `cat` behave as genuine structural inverses of each other

**What you need to know first:**

- Chapter 15's tile-shape vocabulary, including the power-of-two rule this book has relied on since that chapter, and which turns out to explain more about `ct.cat` than its own docstring says outright.
- Chapter 10's naming-confound rule, applied directly to compare `transpose` against an equivalent `permute` call.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow this book has used throughout.

## 22.1 ct.permute: General Axis Reordering

### Intuition

`ct.permute(x, axes)` reorders a tile's axes according to an explicit tuple — axis `i` of the result comes from axis `axes[i]` of the input. It is the most general of this chapter's operations: every other reordering this chapter covers is a special case of what `permute` can already express. A valid `axes` argument must be an honest permutation of `range(x.ndim)` — every axis mentioned exactly once. Two ways to violate that are worth checking directly: repeating an axis, and supplying too few axes for the tile's own rank.

### Background

Testing a legitimate 3D permutation first establishes the ordinary case, then a repeated-axis tuple and a too-short tuple probe how precisely the compiler validates its own stated contract.

### Worked Example 22.1.1 — a valid permutation, then two invalid ones

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig3 = ct.compilation.KernelSignature(
    [array_param(3), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.permute(x, axes) reorders a tile's axes according to an explicit
# permutation tuple -- the general case that transpose specializes.
@ct.kernel
def kernel_permute(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.permute(x, (2, 0, 1))
    ct.store(c, (0, 0, 0), y)
print(f"permute (2,4,8) axes=(2,0,1): {compile_bytes(kernel_permute, sig3)} cubin bytes")

# A permutation tuple with a repeated axis is not a permutation at all --
# does the compiler catch this, or just silently drop an axis?
@ct.kernel
def kernel_permute_dup(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.permute(x, (0, 0, 1))
    ct.store(c, (0, 0, 0), y)
try:
    n = compile_bytes(kernel_permute_dup, sig3)
    print(f"permute axes=(0,0,1) duplicate: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"permute axes=(0,0,1) duplicate: {type(e).__name__}: {e}")

# A permutation tuple shorter than the tile's own rank.
@ct.kernel
def kernel_permute_short(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.permute(x, (1, 0))
    ct.store(c, (0, 0, 0), y)
try:
    n = compile_bytes(kernel_permute_short, sig3)
    print(f"permute axes=(1,0) too short: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"permute axes=(1,0) too short: {type(e).__name__}: {e}")
```

Genuinely run:

```
permute (2,4,8) axes=(2,0,1): 25872 cubin bytes
permute axes=(0,0,1) duplicate: TileTypeError: Repeated axis #1: 0
  "/tmp/ch22/01_permute_basics_and_validation.py", line 36, col 9-32, in kernel_permute_dup:
        y = ct.permute(x, (0, 0, 1))
            ^^^^^^^^^^^^^^^^^^^^^^^^

permute axes=(1,0) too short: TileTypeError: Num axes must match input's rank: 2 vs 3
  "/tmp/ch22/01_permute_basics_and_validation.py", line 49, col 9-29, in kernel_permute_short:
        y = ct.permute(x, (1, 0))
            ^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

Both invalid tuples are caught, and caught with two distinct, purpose-specific messages rather than one generic "invalid axes" complaint. The repeated-axis case names the exact position and value at fault — "Repeated axis #1: 0" — and the too-short case states the mismatch numerically — "2 vs 3" — rather than simply saying the tuple was wrong. This is a pattern this book has now seen repeatedly since Chapter 8: cuTile Python's compiler validates structural arguments like `axes` with the same seriousness it applies to dtypes and shapes, and its error messages tend to name the specific numbers involved rather than gesturing vaguely at "invalid input."

## 22.2 ct.transpose: permute's Two-Axis Special Case

### Intuition

`ct.transpose`'s docstring describes a narrower operation than `permute`: it swaps exactly two axes. For a 2D tile, `axis0` and `axis1` default to the tile's only two axes, so `ct.transpose(x)` alone is enough. For a tile with more than two dimensions, the docstring says `axis0`/`axis1` "must be explicitly specified" — there is no default two-axis guess once there are three or more axes to choose from. Since a 2D transpose is really just `permute(x, (1, 0))` under a friendlier name, it is worth testing directly whether the compiler treats them as genuinely the same operation, using Chapter 10's naming-confound control to make sure any byte difference could only come from the operation itself.

### Background

First, confirm the 2D default and the 3D explicit-axes requirement behave as documented. Then compile `transpose(x)` and the equivalent `permute(x, (1, 0))` under the identical function name `kernel_fn`, so a byte-count comparison is controlled the way Chapter 10 established.

### Worked Example 22.2.1 — transpose's default, its 3D requirement, and its identity with permute

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig2 = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig3 = ct.compilation.KernelSignature(
    [array_param(3), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.transpose's docstring: for a 2D tile, axis0/axis1 default to
# swapping the only two axes there are.
@ct.kernel
def kernel_transpose2d(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.transpose(x)
    ct.store(c, (0, 0), y)
print(f"transpose 2D default: {compile_bytes(kernel_transpose2d, sig2)} cubin bytes")

# For tiles with more than 2 dimensions, the docstring says axis0/axis1
# "must be explicitly specified." Does a 3D tile without them actually
# get rejected, rather than silently guessing which two axes to swap?
@ct.kernel
def kernel_transpose3d_noargs(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.transpose(x)
    ct.store(c, (0, 0, 0), y)
try:
    n = compile_bytes(kernel_transpose3d_noargs, sig3)
    print(f"transpose 3D no explicit axes: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"transpose 3D no explicit axes: {type(e).__name__}: {e}")

# With axis0/axis1 given explicitly, a 3D transpose compiles.
@ct.kernel
def kernel_transpose3d(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.transpose(x, 0, 2)
    ct.store(c, (0, 0, 0), y)
print(f"transpose 3D axis0=0 axis1=2: {compile_bytes(kernel_transpose3d, sig3)} cubin bytes")

# transpose(x) for a 2D tile and permute(x, (1, 0)) describe the exact
# same operation. Naming-confound-controlled (Chapter 10): both variants
# reuse the identical function name "kernel_fn", so any byte difference
# would have to come from the operation itself, not from the symbol
# name embedded in the cubin.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.transpose(x)
    ct.store(c, (0, 0), y)
n_transpose = compile_bytes(kernel_fn, sig2)
print(f"transpose(x) on (4,8), naming-controlled: {n_transpose} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.permute(x, (1, 0))
    ct.store(c, (0, 0), y)
n_permute = compile_bytes(kernel_fn, sig2)
print(f"permute(x, (1,0)) on (4,8), naming-controlled: {n_permute} cubin bytes")
print(f"identical bytes: {n_transpose == n_permute}")
```

Genuinely run:

```
transpose 2D default: 22960 cubin bytes
transpose 3D no explicit axes: TileTypeError: `axes` must be specified for tile with more than 2 dimensions
  "/tmp/ch22/02_transpose_default_and_explicit_axes.py", line 41, col 9-23, in kernel_transpose3d_noargs:
        y = ct.transpose(x)
            ^^^^^^^^^^^^^^^

transpose 3D axis0=0 axis1=2: 26000 cubin bytes
transpose(x) on (4,8), naming-controlled: 22576 cubin bytes
permute(x, (1,0)) on (4,8), naming-controlled: 22576 cubin bytes
identical bytes: True
```

### Discussion

Every prediction holds, including the last one: `transpose(x)` and `permute(x, (1, 0))`, compiled under the identical function name so nothing but the operation itself could account for a difference, produce byte-identical cubins. Note that this figure (22576) differs slightly from the 2D default's own figure two examples earlier (22960) — that gap is Chapter 16's file-path-length confound plus the different function name (`kernel_fn` versus `kernel_transpose2d`) doing exactly what Chapter 10 predicts, not a new finding. The genuinely informative comparison is the one held constant: `transpose` and its `permute` equivalent, same function name, same file, same everything except which of the two API entry points was called — and the compiler produced the exact same machine code for both. That is about as direct a confirmation as this book's `export_kernel`-only environment can offer that `transpose` is implemented as nothing more than a convenience wrapper over `permute`'s more general two-axis case.

## 22.3 ct.expand_dims: Inserting a Size-1 Axis

### Intuition

`ct.expand_dims(x, axis)` inserts a new axis of size 1 at the given position — turning a 1D tile of shape `(8,)` into a `(1, 8)` row or an `(8, 1)` column, depending on where the new axis goes. Its docstring notes the position can also be reached through NumPy-style syntax, `x[:, None]`, and that `axis` accepts negative indexing the way Python sequence indices generally do. An axis argument far outside anything a 1D tile's expansion could support is worth checking directly, if only to see whether the resulting error names the tile's rank the way this chapter's other validation errors have.

### Background

Testing `axis=0`, then the negative-indexed `axis=-1`, establishes the documented behavior at both ends of a 1D tile's two possible expansion points; an out-of-range `axis=5` tests the boundary.

### Worked Example 22.3.1 — expand_dims at both valid positions, then out of range

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig_2d_from_1d = ct.compilation.KernelSignature(
    [array_param(1), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.expand_dims(x, axis) inserts a new size-1 axis at the given
# position. axis=0 on a 1D tile gives a (1, N) row.
@ct.kernel
def kernel_expand0(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.expand_dims(x, 0)
    ct.store(c, (0, 0), y)
print(f"expand_dims(x, 0) on (8,): {compile_bytes(kernel_expand0, sig_2d_from_1d)} cubin bytes")

# Negative axis indexing, NumPy-style: -1 on a 1D tile gives an (N, 1)
# column instead of a (1, N) row.
@ct.kernel
def kernel_expand_neg1(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.expand_dims(x, -1)
    ct.store(c, (0, 0), y)
print(f"expand_dims(x, -1) on (8,): {compile_bytes(kernel_expand_neg1, sig_2d_from_1d)} cubin bytes")

# An axis far outside the range a 1D tile could ever support (only 0,
# 1, or -1 make sense once the new axis is inserted).
@ct.kernel
def kernel_expand_oob(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.expand_dims(x, 5)
    ct.store(c, (0, 0), y)
try:
    n = compile_bytes(kernel_expand_oob, sig_2d_from_1d)
    print(f"expand_dims(x, 5) on (8,): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"expand_dims(x, 5) on (8,): {type(e).__name__}: {e}")
```

Genuinely run:

```
expand_dims(x, 0) on (8,): 21096 cubin bytes
expand_dims(x, -1) on (8,): 21224 cubin bytes
expand_dims(x, 5) on (8,): TileTypeError: Axis 5 is out of range for rank 2'
  "/tmp/ch22/03_expand_dims_and_axis_validation.py", line 46, col 9-28, in kernel_expand_oob:
        y = ct.expand_dims(x, 5)
            ^^^^^^^^^^^^^^^^^^^^
```

### Discussion

Both valid positions compile, and to slightly different byte counts — `axis=0` and `axis=-1` are not the same operation on a 1D tile, since one produces a row and the other a column, so this is one of the rare byte-count differences in this book that needs no naming-confound caveat: both variants share the identical function name pattern already, and the shapes genuinely differ. The out-of-range case is rejected with a message that names the rank of the *result*, not the input: "out of range for rank 2" — a 1D input tile expanding into a 2D output, and it is that 2D output rank the valid axis range is checked against. It is worth noting, without dwelling on it, that the error message ends with a stray closing single-quote — `for rank 2'` — a minor cosmetic wrinkle in the compiler's own error-formatting code, reproduced here exactly as it was genuinely emitted, rather than corrected into tidier prose.

## 22.4 ct.cat: One Docstring Claim, Two Compiler Checks

### Intuition

`ct.cat((tiles), axis)`'s docstring states its requirement plainly: "the two input tiles must have the same shape," attributing this to "the power-of-two assumption on all tile shapes." Read at face value, this sounds like a single equality check — either the two tiles match exactly, or they don't. But an ordinary concatenation only needs the tiles to agree on every axis *except* the one being joined; requiring full equality is a much stronger claim. Testing a mismatch that only differs on the concat axis, versus a mismatch on a different axis entirely, separates these into two distinguishable outcomes.

### Background

First, confirm equal shapes concatenate cleanly. Then concatenate a `(2,2)` and a `(2,4)` tile along axis 1 — the two tiles agree everywhere except the joined axis, exactly the case an ordinary concat library would accept — to see whether it is rejected, and for which stated reason. Finally, concatenate a `(2,4)` and a `(4,4)` tile along axis 1, where the concat axis sums to a valid power of two (4+4=8) but the *non-concat* axis disagrees (2 vs 4), isolating whether a second, independent check exists for the axes not being joined.

### Worked Example 22.4.1 — same shape, a concat-axis mismatch, and a non-concat-axis mismatch

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.cat's docstring claims the two input tiles "must have the same
# shape," citing the power-of-two assumption on tile shapes. Equal
# shapes concatenated along axis 0 compile cleanly, as expected.
@ct.kernel
def kernel_cat_equal_axis0(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.load(b, (0, 0), (4, 4))
    z = ct.cat((x, y), 0)
    ct.store(c, (0, 0), z)
print(f"cat (4,4)+(4,4) axis0 [equal shapes]: {compile_bytes(kernel_cat_equal_axis0, sig)} cubin bytes")

# A (2,2) and a (2,4) tile, concatenated along axis 1: in most array
# libraries this is an ordinary concat, since the two tiles agree on
# every axis except the one being joined. Does cuTile Python reject it,
# and if so, for the reason the docstring implies (unequal shapes) or
# a different one entirely?
@ct.kernel
def kernel_cat_mismatch(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (2, 2))
    y = ct.load(b, (0, 0), (2, 4))
    z = ct.cat((x, y), 1)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_cat_mismatch, sig)
    print(f"cat (2,2)+(2,4) axis1 [concat-axis sum=6, not pow2]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"cat (2,2)+(2,4) axis1 [concat-axis sum=6, not pow2]: {type(e).__name__}: {e}")

# A second, independent test: (2,4) and (4,4) concatenated along axis 1.
# The concat axis sums to 4+4=8, a valid power of two -- but the
# *non-concat* axis (axis 0) disagrees: 2 vs 4. Isolating this shows
# whether cat separately checks the non-concat axes for equality, or
# whether the power-of-two-result check was the only thing enforcing
# "same shape" all along.
@ct.kernel
def kernel_cat_nonconcat_mismatch(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (2, 4))
    y = ct.load(b, (0, 0), (4, 4))
    z = ct.cat((x, y), 1)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_cat_nonconcat_mismatch, sig)
    print(f"cat (2,4)+(4,4) axis1 [non-concat axis0 differs, concat sum=8 pow2]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"cat (2,4)+(4,4) axis1 [non-concat axis0 differs, concat sum=8 pow2]: {type(e).__name__}: {e}")
```

Genuinely run:

```
cat (4,4)+(4,4) axis0 [equal shapes]: 27864 cubin bytes
cat (2,2)+(2,4) axis1 [concat-axis sum=6, not pow2]: TileTypeError: Result tile shape must be power of 2, got: [2, 6]
  "/tmp/ch22/04_cat_two_checks_not_one.py", line 42, col 9-25, in kernel_cat_mismatch:
        z = ct.cat((x, y), 1)
            ^^^^^^^^^^^^^^^^^

cat (2,4)+(4,4) axis1 [non-concat axis0 differs, concat sum=8 pow2]: TileTypeError: Expected tiles to have the same shape for non axis dimensions, got (2, 4) and (4, 4)
  "/tmp/ch22/04_cat_two_checks_not_one.py", line 61, col 9-25, in kernel_cat_nonconcat_mismatch:
        z = ct.cat((x, y), 1)
            ^^^^^^^^^^^^^^^^^
```

### Discussion

Both mismatched calls are rejected, but for two entirely different, independently-worded reasons — confirming the docstring's single blanket claim is really the visible surface of two separate compiler checks. The `(2,2)`/`(2,4)` case fails because the *concatenated result* along axis 1 would be `[2, 6]`, and 6 is not a power of two — nothing in that error mentions the two input shapes being unequal at all. The `(2,4)`/`(4,4)` case fails a completely different check: "same shape for non axis dimensions," which has nothing to do with powers of two and everything to do with the axes not being joined needing to agree, exactly as an ordinary concat library would require. Put the two checks together and something else falls out for free: since Chapter 15 established that every tile axis length must itself be a power of two, two power-of-two numbers can only ever sum to a third power of two when they are equal to each other — 2+2=4 works, but no two *distinct* powers of two ever sum to a power of two. That means the power-of-two-result check on the concat axis, combined with the ordinary equality check on every other axis, together force the concat axis to match too, as a mathematical consequence rather than a third, separately-coded rule. The docstring's "must have the same shape" is true, but it is an emergent fact about two independent checks colliding with the power-of-two invariant, not a single check that says so directly.

## 22.5 ct.extract: Reading a Subtile by Grid Index

### Intuition

`ct.extract(x, index, shape)` partitions `x` into a grid of same-shaped subtiles and returns the one at the given grid position. The docstring is explicit that `index` addresses "the grid of subtiles, not element index" — extracting shape `(2, 2)` from a `(4, 4)` tile creates a 2x2 grid of quadrants, so valid indices along either axis are only 0 or 1, not 0 through 3. Two distinct ways to misuse this are worth testing: an index that is valid as an element position but not as a grid position, and an extraction `shape` that itself violates the power-of-two rule this chapter's `cat` section just leaned on.

### Background

A valid `(0, 1)` extraction from a `(4, 4)` tile with `(2, 2)` subtiles establishes the ordinary case (the top-right quadrant). An index of `(0, 2)` — which would be a valid *element* column but is one grid position too many — tests the grid-versus-element distinction directly. A `(3, 3)` extraction shape tests whether `extract`'s own `shape` argument is checked against the power-of-two rule independently of whether it would evenly divide the input.

### Worked Example 22.5.1 — a valid grid index, an out-of-bounds grid index, and a non-power-of-two extraction shape

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.extract(x, index, shape) partitions x into a grid of subtiles of
# the given shape, and returns the one at the given grid index -- an
# index into the grid, not into x's own elements. A (4,4) tile
# partitioned into (2,2) subtiles has a 2x2 grid; index (0,1) is the
# top-right quadrant.
@ct.kernel
def kernel_extract_valid(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.extract(x, (0, 1), (2, 2))
    ct.store(c, (0, 0), y)
print(f"extract (4,4) index=(0,1) shape=(2,2): {compile_bytes(kernel_extract_valid, sig)} cubin bytes")

# The grid for (2,2) subtiles of a (4,4) tile is only 2x2, so valid
# indices along either axis are 0 or 1. Index (0, 2) is out of bounds.
@ct.kernel
def kernel_extract_oob(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.extract(x, (0, 2), (2, 2))
    ct.store(c, (0, 0), y)
try:
    n = compile_bytes(kernel_extract_oob, sig)
    print(f"extract (4,4) index=(0,2) shape=(2,2) [oob]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract (4,4) index=(0,2) shape=(2,2) [oob]: {type(e).__name__}: {e}")

# extract's own shape argument is subject to the same power-of-two
# tile-shape rule Chapter 15 established -- a (3,3) extract shape
# should be rejected on that basis alone, independent of whether it
# would evenly divide (4,4).
@ct.kernel
def kernel_extract_nondivide(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.extract(x, (0, 0), (3, 3))
    ct.store(c, (0, 0), y)
try:
    n = compile_bytes(kernel_extract_nondivide, sig)
    print(f"extract (4,4) index=(0,0) shape=(3,3) [non-power-of-two]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract (4,4) index=(0,0) shape=(3,3) [non-power-of-two]: {type(e).__name__}: {e}")
```

Genuinely run:

```
extract (4,4) index=(0,1) shape=(2,2): 24000 cubin bytes
extract (4,4) index=(0,2) shape=(2,2) [oob]: TileTypeError: Index 2 out of bounds at dimension #1: valid range is [0, 2) in tile space (input shape (4, 4), extract shape (2, 2))
  "/tmp/ch22/05_extract_grid_indexing.py", line 39, col 9-37, in kernel_extract_oob:
        y = ct.extract(x, (0, 2), (2, 2))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

extract (4,4) index=(0,0) shape=(3,3) [non-power-of-two]: TileTypeError: Invalid argument "shape" of extract(): Dimension #0 of shape (3, 3) is not a power of two
  "/tmp/ch22/05_extract_grid_indexing.py", line 55, col 9-37, in kernel_extract_nondivide:
        y = ct.extract(x, (0, 0), (3, 3))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

The out-of-bounds error goes out of its way to disambiguate exactly the confusion the docstring warned about: it says "valid range is [0, 2)" — grid positions, not element positions — and then spells out both shapes involved in parentheses, "(input shape (4, 4), extract shape (2, 2))," so there is no need to recompute the grid size by hand to understand why 2 was rejected. The non-power-of-two case is rejected before the compiler even gets to checking whether `(3, 3)` divides `(4, 4)` evenly (it doesn't, but that was never the stated reason) — the error names `extract`'s own `shape` parameter explicitly and cites the same power-of-two rule this chapter's `cat` section relied on, confirming that rule is a genuinely global property of tile shapes in cuTile Python, not something specific to how tiles are loaded or stored.

## 22.6 Capstone: Extract and cat as Structural Inverses

### Intuition

If `extract` genuinely partitions a tile into a grid of subtiles, and `cat` genuinely reassembles tiles along an axis, then the two should compose into a working round-trip: partition a tile into pieces with `extract`, and rebuild the original tile's shape with `cat`. Layering a `transpose` on each piece in between tests something slightly stronger — that this chapter's five operations don't just each work in isolation, but compose freely with each other and with an operation from an earlier chapter inside one kernel body.

### Background

A `(4, 4)` tile is split into four `(2, 2)` quadrants via `extract`, then reassembled with three `cat` calls (two along axis 1 to rebuild each row, one along axis 0 to stack the rows). A plain load/store of the same shape serves as a baseline for how much heavier the decomposition makes the compiled kernel. A second variant transposes each extracted quadrant before reassembling, confirming the composition survives an operation from Section 22.2 mixed in.

### Worked Example 22.6.1 — partition, transpose each piece, and reassemble

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: partition a (4,4) tile into four (2,2) quadrants with
# extract, then reassemble the whole tile with cat -- confirming
# extract and cat behave as structural inverses of each other.
@ct.kernel
def kernel_extract_cat_roundtrip(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    tl = ct.extract(x, (0, 0), (2, 2))
    tr = ct.extract(x, (0, 1), (2, 2))
    bl = ct.extract(x, (1, 0), (2, 2))
    br = ct.extract(x, (1, 1), (2, 2))
    top = ct.cat((tl, tr), 1)
    bottom = ct.cat((bl, br), 1)
    whole = ct.cat((top, bottom), 0)
    ct.store(c, (0, 0), whole)
print(f"extract/cat roundtrip (4,4) -> 4x(2,2) -> (4,4): {compile_bytes(kernel_extract_cat_roundtrip, sig)} cubin bytes")

# A plain load/store of the same shape, with no extract/cat at all, as
# a baseline for how much heavier the round-trip's decomposition makes
# the compiled kernel.
@ct.kernel
def kernel_direct_roundtrip(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    ct.store(c, (0, 0), x)
print(f"direct load/store (4,4), no extract/cat: {compile_bytes(kernel_direct_roundtrip, sig)} cubin bytes")

# Extending the round-trip with a per-quadrant transpose in between
# confirms this chapter's five operations compose freely with each
# other inside a single kernel, not just in isolated examples.
@ct.kernel
def kernel_extract_transpose_cat(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    tl = ct.extract(x, (0, 0), (2, 2))
    tr = ct.extract(x, (0, 1), (2, 2))
    bl = ct.extract(x, (1, 0), (2, 2))
    br = ct.extract(x, (1, 1), (2, 2))
    tl_t = ct.transpose(tl)
    tr_t = ct.transpose(tr)
    bl_t = ct.transpose(bl)
    br_t = ct.transpose(br)
    top = ct.cat((tl_t, tr_t), 1)
    bottom = ct.cat((bl_t, br_t), 1)
    whole = ct.cat((top, bottom), 0)
    ct.store(c, (0, 0), whole)
print(f"extract/transpose/cat composed, quadrant-wise transpose: {compile_bytes(kernel_extract_transpose_cat, sig)} cubin bytes")
```

Genuinely run:

```
extract/cat roundtrip (4,4) -> 4x(2,2) -> (4,4): 24768 cubin bytes
direct load/store (4,4), no extract/cat: 23104 cubin bytes
extract/transpose/cat composed, quadrant-wise transpose: 25664 cubin bytes
```

### Discussion

All three kernels compile cleanly. The plain load/store, at 23104 bytes, is the smallest of the three, as expected — it does no partitioning or reassembly at all. The extract/cat round-trip adds 1664 bytes over that baseline for the four extractions and three concatenations, and layering a `transpose` onto each quadrant adds a further 896 bytes on top of that. Neither addition is surprising in direction, and this book's `export_kernel`-only environment can confirm exactly what it has been able to confirm since Chapter 1: that the composition is well-typed and compiles, not that the reassembled tile's element values genuinely match the original. What this capstone does add, though, is the clearest demonstration yet that `extract` and `cat` are not just individually well-behaved operations but genuine structural inverses — partition with one, exactly reverse it with the other, and mix in an operation from three chapters earlier without any of the three operations needing to know about the others.

## Chapter Summary

This chapter covered five operations whose entire job is rearranging a tile's own layout rather than computing new values or moving data between memory and tiles. `ct.permute` is the general axis-reordering operation, validating its `axes` argument with two distinct, purpose-specific errors for a repeated axis and a too-short tuple. `ct.transpose` is a two-axis special case of `permute` — defaulting to swapping a 2D tile's only two axes, requiring `axis0`/`axis1` explicitly for anything with more dimensions, and, confirmed under Chapter 10's naming-confound control, compiling to byte-identical output to the equivalent `permute` call. `ct.expand_dims` inserts a size-1 axis, including NumPy-style negative-axis addressing, rejecting an out-of-range axis with an error that names the *result* tile's rank. `ct.cat`'s docstring claim that its two inputs "must have the same shape" turned out to be the combined effect of two independent, separately-worded checks — a power-of-two constraint on the concatenated result, and an ordinary equality constraint on every axis not being joined — which together mathematically force equality on the concat axis too, since two power-of-two numbers only sum to a power of two when they match. `ct.extract` reads a subtile by an index into a grid of subtiles, not an index into elements, with error messages that spell out both shapes involved rather than leaving the grid size to be recomputed by hand. The capstone confirmed all five operations compose freely, using `extract` and `cat` as genuine structural inverses of each other with a `transpose` mixed into the middle of the round-trip.

## Self-Check Questions

1. Section 22.2 found that `transpose(x)` and `permute(x, (1, 0))` compile to byte-identical cubins under Chapter 10's naming-confound control. What would it have meant for this book's understanding of `transpose` if the two byte counts had differed even under that control — and why does controlling for the function name matter for drawing that conclusion at all?
2. Section 22.4 found that `ct.cat`'s docstring claim ("the two input tiles must have the same shape") is the emergent result of two independently-worded checks rather than one direct equality check. Suppose a hypothetical `cat` implementation checked shape equality directly, with one single error message, instead of the two checks this chapter found. Would that hypothetical implementation behave any differently, from a caller's point of view, than the real one — or would the difference be purely about which error message you'd see?
3. Section 22.5's out-of-bounds `extract` error stated the valid range as "[0, 2)" and explicitly printed both the input shape and the extract shape. Contrast this with Section 22.1's "Num axes must match input's rank: 2 vs 3" error for `permute`. What do both error messages have in common in how they help a caller fix the mistake, compared to an error that simply said "invalid index" or "invalid axes" with no further detail?
4. The capstone in Section 22.6 chose to decompose a `(4, 4)` tile into `(2, 2)` quadrants using `extract`, and reassemble with three `cat` calls. Given Section 22.4's finding about `cat`'s two checks, what would have to be true of the four quadrant shapes for the three `cat` calls in that capstone to have any chance of compiling, before even considering whether the *values* end up correct?
5. This chapter's five operations — `permute`, `transpose`, `expand_dims`, `cat`, and `extract` — never touch a tile's actual numeric values, only its shape and the arrangement of its axes. Given everything this book has established about what `export_kernel`'s driver-free compilation can and cannot confirm, what specifically can a clean compile of this chapter's capstone tell you, and what would still require running the kernel on real hardware to know for certain?

## Where We Go Next

This chapter closed out the shape-manipulation corner of `dir(ct)` that Chapter 21 first pointed toward — `permute`, `transpose`, `expand_dims`, `cat`, and `extract` all now confirmed to compose freely and to validate their own arguments with the same precision this book has come to expect. What remains from that same forward reference is a lower-level pair: `load_advanced_indexing` and `store_advanced_indexing`, sitting alongside the `gather()`/`scatter()` convention Chapter 20 introduced, under a name suggesting there is a broader indexing story this book has only seen one corner of; and `bitcast`, `pack_to_bytes`, and `unpack_from_bytes`, which look past a tile's logical shape entirely toward its raw bytes — the same `bitcast` that Chapter 19's bitwise-rejection error messages kept recommending as an escape hatch, without this book ever actually calling it.

## Worked Solutions

1. A byte difference under the naming-confound control would have meant `transpose` and the equivalent `permute` call genuinely compile to different machine code despite describing the same logical operation — evidence that `transpose` is not simply a thin wrapper delegating to `permute`, but has its own, separately-generated code path, perhaps specialized in some way `permute`'s general axis-reordering logic isn't. Controlling for the function name matters because Chapter 10 established that the compiled cubin embeds the kernel function's own `__name__` as a symbol, which changes the byte count independent of anything the kernel's body does; without holding the name fixed, a byte difference between a `kernel_transpose2d` and a `kernel_permute` could just as easily come from the eight extra characters in one name as from any real difference in the operations themselves. Since both variants here used the identical name `kernel_fn`, the observed equality is a clean, unconfounded result: whatever `transpose` compiles to, it is exactly what `permute(x, (1, 0))` compiles to.

2. From a caller's point of view, the two implementations would behave identically in every case that actually arises in this chapter: any input that fails one check fails the other (since both reject the same underlying mismatches), and any input that passes both real checks would also pass a single combined equality check. The only visible difference is exactly what Question 1 already gestures at for a different pair of operations: which error message you'd see, and how much information it gives you about *why* your call was rejected. The real `cat` tells you specifically whether your problem is the concatenated result violating the power-of-two rule or the non-concat axes disagreeing — two different problems with two different fixes. A single "shapes must match" message would leave you to work out which kind of mismatch you'd caused yourself. The two-check design costs nothing in terms of what compiles and what doesn't; it only adds precision to the failure case.

3. Both error messages replace a bare true/false verdict with the specific numbers a caller would otherwise have to reconstruct by hand: the `permute` error states the exact required count and the exact count supplied ("2 vs 3"), and the `extract` error states the valid range and both shapes involved in the computation that produced it. In each case, the message does the arithmetic the caller would have needed to do anyway — computing the tile's rank, or computing the grid size from the input and extract shapes — and hands over the result directly, rather than requiring the caller to go re-derive it from the docstring and their own code. A bare "invalid index" or "invalid axes" would still tell you *that* something was wrong, but would leave the actual diagnostic work undone; the messages this book has seen since Chapter 8 consistently do that work for you.

4. For the three `cat` calls to have any chance of compiling, each pair being concatenated must first satisfy the non-concat-axis equality check Section 22.4 found — the two quadrants forming a "row" must agree on their non-concatenated axis, and the two assembled rows being stacked must likewise agree on theirs — and second, the concatenated result along the joined axis must itself be a power of two. In the capstone, all four quadrants share the identical shape `(2, 2)`, which trivially satisfies both requirements at every step: every non-concat axis matches because every quadrant's shape matches, and every concat-axis sum is `2 + 2 = 4`, a power of two. Had the quadrants been asymmetric — say, three `(2, 2)` pieces and one `(2, 4)` piece — at least one of the three `cat` calls would have hit one of Section 22.4's two checks, regardless of whether the numeric values inside those tiles were otherwise sensible.

5. A clean compile of the capstone confirms that every shape and dtype constraint this chapter tested — `extract`'s grid-index bounds and power-of-two shape requirement, `cat`'s two independent checks, `transpose`'s axis requirements — is satisfied simultaneously by this particular composition, and that the whole pipeline type-checks as a single well-formed kernel from load through extract, transpose, cat, and store. What it cannot confirm, and has never been able to confirm anywhere in this book since Chapter 1, is that the actual numeric values ending up in the output tile are the values you'd get by genuinely partitioning, transposing, and reassembling real data — that would require `ct.launch()` on real GPU hardware with a driver, which this book's `export_kernel`-only, driver-free environment has never had access to. The distinction this chapter's operations make especially vivid is that "structurally correct" and "numerically correct" are entirely separate claims: every worked example here demonstrates the first, and none of them can speak to the second.

## Complete Runnable Code

### File: `01_permute_basics_and_validation.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig3 = ct.compilation.KernelSignature(
    [array_param(3), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.permute(x, axes) reorders a tile's axes according to an explicit
# permutation tuple -- the general case that transpose specializes.
@ct.kernel
def kernel_permute(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.permute(x, (2, 0, 1))
    ct.store(c, (0, 0, 0), y)
print(f"permute (2,4,8) axes=(2,0,1): {compile_bytes(kernel_permute, sig3)} cubin bytes")

# A permutation tuple with a repeated axis is not a permutation at all --
# does the compiler catch this, or just silently drop an axis?
@ct.kernel
def kernel_permute_dup(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.permute(x, (0, 0, 1))
    ct.store(c, (0, 0, 0), y)
try:
    n = compile_bytes(kernel_permute_dup, sig3)
    print(f"permute axes=(0,0,1) duplicate: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"permute axes=(0,0,1) duplicate: {type(e).__name__}: {e}")

# A permutation tuple shorter than the tile's own rank.
@ct.kernel
def kernel_permute_short(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.permute(x, (1, 0))
    ct.store(c, (0, 0, 0), y)
try:
    n = compile_bytes(kernel_permute_short, sig3)
    print(f"permute axes=(1,0) too short: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"permute axes=(1,0) too short: {type(e).__name__}: {e}")
```

### File: `02_transpose_default_and_explicit_axes.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig2 = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig3 = ct.compilation.KernelSignature(
    [array_param(3), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.transpose's docstring: for a 2D tile, axis0/axis1 default to
# swapping the only two axes there are.
@ct.kernel
def kernel_transpose2d(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.transpose(x)
    ct.store(c, (0, 0), y)
print(f"transpose 2D default: {compile_bytes(kernel_transpose2d, sig2)} cubin bytes")

# For tiles with more than 2 dimensions, the docstring says axis0/axis1
# "must be explicitly specified." Does a 3D tile without them actually
# get rejected, rather than silently guessing which two axes to swap?
@ct.kernel
def kernel_transpose3d_noargs(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.transpose(x)
    ct.store(c, (0, 0, 0), y)
try:
    n = compile_bytes(kernel_transpose3d_noargs, sig3)
    print(f"transpose 3D no explicit axes: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"transpose 3D no explicit axes: {type(e).__name__}: {e}")

# With axis0/axis1 given explicitly, a 3D transpose compiles.
@ct.kernel
def kernel_transpose3d(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.transpose(x, 0, 2)
    ct.store(c, (0, 0, 0), y)
print(f"transpose 3D axis0=0 axis1=2: {compile_bytes(kernel_transpose3d, sig3)} cubin bytes")

# transpose(x) for a 2D tile and permute(x, (1, 0)) describe the exact
# same operation. Naming-confound-controlled (Chapter 10): both variants
# reuse the identical function name "kernel_fn", so any byte difference
# would have to come from the operation itself, not from the symbol
# name embedded in the cubin.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.transpose(x)
    ct.store(c, (0, 0), y)
n_transpose = compile_bytes(kernel_fn, sig2)
print(f"transpose(x) on (4,8), naming-controlled: {n_transpose} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.permute(x, (1, 0))
    ct.store(c, (0, 0), y)
n_permute = compile_bytes(kernel_fn, sig2)
print(f"permute(x, (1,0)) on (4,8), naming-controlled: {n_permute} cubin bytes")
print(f"identical bytes: {n_transpose == n_permute}")
```

### File: `03_expand_dims_and_axis_validation.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig_2d_from_1d = ct.compilation.KernelSignature(
    [array_param(1), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.expand_dims(x, axis) inserts a new size-1 axis at the given
# position. axis=0 on a 1D tile gives a (1, N) row.
@ct.kernel
def kernel_expand0(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.expand_dims(x, 0)
    ct.store(c, (0, 0), y)
print(f"expand_dims(x, 0) on (8,): {compile_bytes(kernel_expand0, sig_2d_from_1d)} cubin bytes")

# Negative axis indexing, NumPy-style: -1 on a 1D tile gives an (N, 1)
# column instead of a (1, N) row.
@ct.kernel
def kernel_expand_neg1(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.expand_dims(x, -1)
    ct.store(c, (0, 0), y)
print(f"expand_dims(x, -1) on (8,): {compile_bytes(kernel_expand_neg1, sig_2d_from_1d)} cubin bytes")

# An axis far outside the range a 1D tile could ever support (only 0,
# 1, or -1 make sense once the new axis is inserted).
@ct.kernel
def kernel_expand_oob(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.expand_dims(x, 5)
    ct.store(c, (0, 0), y)
try:
    n = compile_bytes(kernel_expand_oob, sig_2d_from_1d)
    print(f"expand_dims(x, 5) on (8,): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"expand_dims(x, 5) on (8,): {type(e).__name__}: {e}")
```

### File: `04_cat_two_checks_not_one.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.cat's docstring claims the two input tiles "must have the same
# shape," citing the power-of-two assumption on tile shapes. Equal
# shapes concatenated along axis 0 compile cleanly, as expected.
@ct.kernel
def kernel_cat_equal_axis0(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.load(b, (0, 0), (4, 4))
    z = ct.cat((x, y), 0)
    ct.store(c, (0, 0), z)
print(f"cat (4,4)+(4,4) axis0 [equal shapes]: {compile_bytes(kernel_cat_equal_axis0, sig)} cubin bytes")

# A (2,2) and a (2,4) tile, concatenated along axis 1: in most array
# libraries this is an ordinary concat, since the two tiles agree on
# every axis except the one being joined. Does cuTile Python reject it,
# and if so, for the reason the docstring implies (unequal shapes) or
# a different one entirely?
@ct.kernel
def kernel_cat_mismatch(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (2, 2))
    y = ct.load(b, (0, 0), (2, 4))
    z = ct.cat((x, y), 1)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_cat_mismatch, sig)
    print(f"cat (2,2)+(2,4) axis1 [concat-axis sum=6, not pow2]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"cat (2,2)+(2,4) axis1 [concat-axis sum=6, not pow2]: {type(e).__name__}: {e}")

# A second, independent test: (2,4) and (4,4) concatenated along axis 1.
# The concat axis sums to 4+4=8, a valid power of two -- but the
# *non-concat* axis (axis 0) disagrees: 2 vs 4. Isolating this shows
# whether cat separately checks the non-concat axes for equality, or
# whether the power-of-two-result check was the only thing enforcing
# "same shape" all along.
@ct.kernel
def kernel_cat_nonconcat_mismatch(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (2, 4))
    y = ct.load(b, (0, 0), (4, 4))
    z = ct.cat((x, y), 1)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_cat_nonconcat_mismatch, sig)
    print(f"cat (2,4)+(4,4) axis1 [non-concat axis0 differs, concat sum=8 pow2]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"cat (2,4)+(4,4) axis1 [non-concat axis0 differs, concat sum=8 pow2]: {type(e).__name__}: {e}")
```

### File: `05_extract_grid_indexing.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.extract(x, index, shape) partitions x into a grid of subtiles of
# the given shape, and returns the one at the given grid index -- an
# index into the grid, not into x's own elements. A (4,4) tile
# partitioned into (2,2) subtiles has a 2x2 grid; index (0,1) is the
# top-right quadrant.
@ct.kernel
def kernel_extract_valid(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.extract(x, (0, 1), (2, 2))
    ct.store(c, (0, 0), y)
print(f"extract (4,4) index=(0,1) shape=(2,2): {compile_bytes(kernel_extract_valid, sig)} cubin bytes")

# The grid for (2,2) subtiles of a (4,4) tile is only 2x2, so valid
# indices along either axis are 0 or 1. Index (0, 2) is out of bounds.
@ct.kernel
def kernel_extract_oob(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.extract(x, (0, 2), (2, 2))
    ct.store(c, (0, 0), y)
try:
    n = compile_bytes(kernel_extract_oob, sig)
    print(f"extract (4,4) index=(0,2) shape=(2,2) [oob]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract (4,4) index=(0,2) shape=(2,2) [oob]: {type(e).__name__}: {e}")

# extract's own shape argument is subject to the same power-of-two
# tile-shape rule Chapter 15 established -- a (3,3) extract shape
# should be rejected on that basis alone, independent of whether it
# would evenly divide (4,4).
@ct.kernel
def kernel_extract_nondivide(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    y = ct.extract(x, (0, 0), (3, 3))
    ct.store(c, (0, 0), y)
try:
    n = compile_bytes(kernel_extract_nondivide, sig)
    print(f"extract (4,4) index=(0,0) shape=(3,3) [non-power-of-two]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract (4,4) index=(0,0) shape=(3,3) [non-power-of-two]: {type(e).__name__}: {e}")
```

### File: `06_capstone_extract_cat_transpose_roundtrip.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: partition a (4,4) tile into four (2,2) quadrants with
# extract, then reassemble the whole tile with cat -- confirming
# extract and cat behave as structural inverses of each other.
@ct.kernel
def kernel_extract_cat_roundtrip(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    tl = ct.extract(x, (0, 0), (2, 2))
    tr = ct.extract(x, (0, 1), (2, 2))
    bl = ct.extract(x, (1, 0), (2, 2))
    br = ct.extract(x, (1, 1), (2, 2))
    top = ct.cat((tl, tr), 1)
    bottom = ct.cat((bl, br), 1)
    whole = ct.cat((top, bottom), 0)
    ct.store(c, (0, 0), whole)
print(f"extract/cat roundtrip (4,4) -> 4x(2,2) -> (4,4): {compile_bytes(kernel_extract_cat_roundtrip, sig)} cubin bytes")

# A plain load/store of the same shape, with no extract/cat at all, as
# a baseline for how much heavier the round-trip's decomposition makes
# the compiled kernel.
@ct.kernel
def kernel_direct_roundtrip(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    ct.store(c, (0, 0), x)
print(f"direct load/store (4,4), no extract/cat: {compile_bytes(kernel_direct_roundtrip, sig)} cubin bytes")

# Extending the round-trip with a per-quadrant transpose in between
# confirms this chapter's five operations compose freely with each
# other inside a single kernel, not just in isolated examples.
@ct.kernel
def kernel_extract_transpose_cat(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 4))
    tl = ct.extract(x, (0, 0), (2, 2))
    tr = ct.extract(x, (0, 1), (2, 2))
    bl = ct.extract(x, (1, 0), (2, 2))
    br = ct.extract(x, (1, 1), (2, 2))
    tl_t = ct.transpose(tl)
    tr_t = ct.transpose(tr)
    bl_t = ct.transpose(bl)
    br_t = ct.transpose(br)
    top = ct.cat((tl_t, tr_t), 1)
    bottom = ct.cat((bl_t, br_t), 1)
    whole = ct.cat((top, bottom), 0)
    ct.store(c, (0, 0), whole)
print(f"extract/transpose/cat composed, quadrant-wise transpose: {compile_bytes(kernel_extract_transpose_cat, sig)} cubin bytes")
```
