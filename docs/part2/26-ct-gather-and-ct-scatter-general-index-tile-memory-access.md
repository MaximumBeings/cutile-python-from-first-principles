# Chapter 26: ct.gather and ct.scatter — General Index-Tile Memory Access

> "Chapter 23 gave this book one dimension of freedom: a single runtime index tile, with every other axis pinned down at compile time. This chapter removes the pin entirely — every axis can move independently, at the cost of a restriction Chapter 23 never had to think about at all."

**What you will understand by the end of this chapter:**

- `ct.gather(array, indices)`: reading a tile from `array` where every axis is addressed by its own runtime integer tile, with the result's shape determined by broadcasting those per-axis index tiles against each other
- The 1-dimensional convenience form — passing a bare tile instead of a length-1 tuple — confirmed byte-identical to the explicit tuple form under naming control
- How `mask`, `padding_value`, and `check_bounds` interact: a custom mask alone measurably changes `gather`'s compiled size, while a non-default `padding_value` on top of an existing mask does not
- That this book's `export_kernel`-only, driver-free method can confirm `gather` accepts a compile-time-constant negative index without complaint, but cannot confirm the docstring's separate claim about what such an index would actually read at runtime
- `latency`: an undocumented-in-the-docstring but compiler-validated scheduling hint, constrained to the integer range 1-10, that in this book's own tests never once changed a compiled byte count
- `ct.scatter(array, indices, value)`: gather's write counterpart, with its own validator and no `padding_value` parameter at all, since a masked-out or out-of-bounds write simply doesn't happen
- A genuine, measured asymmetry: adding an otherwise-identical custom `mask` changes `gather`'s compiled size but leaves `scatter`'s completely unchanged
- Why `ct.gather`/`ct.scatter` can express a fully general remap — every axis independently dynamic — that Chapter 23's `ct.load_advanced_indexing`/`ct.store_advanced_indexing` structurally cannot, confirmed by attempting the identical task with both and watching one of them fail

**What you need to know first:**

- Chapter 23's `ct.load_advanced_indexing`/`ct.store_advanced_indexing`, and its rule that exactly one axis may be a runtime 1D integer tile while every other axis is a compile-time `ct.Slice` — this chapter's central comparison is against that restriction specifically.
- Chapter 10's naming-confound control, used throughout this chapter's byte-count comparisons.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow this book has used throughout.

## 26.1 ct.gather: Reading From Independently-Computed Coordinates

### Intuition

`ct.gather(array, indices)` takes a tuple of per-axis integer index tiles — one per dimension of `array` — and reads `array[ind0[...], ind1[...], ...]` at every position implied by broadcasting those index tiles against each other. Nothing in that description requires any axis to be compile-time-fixed the way Chapter 23's `load_advanced_indexing` requires all but one to be; every index tile can, in principle, be computed from something entirely dynamic. For a 1-dimensional array, the docstring also permits a shorthand: passing a bare tile instead of a length-1 tuple, described as "strictly equivalent" — worth confirming under this book's usual naming-confound control rather than taking on faith.

### Background

A 2-dimensional gather with two broadcastable index tiles of shapes `(4, 1, 1)` and `(1, 1, 8)` produces a result of the broadcast shape `(4, 1, 8)`, matching the docstring's own worked example exactly. Then, on a 1-dimensional array, the same gather is compiled twice under Chapter 10's naming-confound control (both kernels named `kernel_fn`): once passing the index as an explicit length-1 tuple, once passing it bare.

### Worked Example 26.1.1 — broadcasting per-axis indices, and the 1D convenience form

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig2to3 = ct.compilation.KernelSignature(
    [array_param(2), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.gather(array, indices) reads array[ind0[...], ind1[...]] for a
# tuple of per-axis index tiles, one per array dimension. Broadcastable
# index shapes (4,1,1) and (1,1,8) produce a result of the broadcasted
# shape (4,1,8) -- every element of the result can, in principle, come
# from an entirely independent (row, col) pair, unlike Chapter 23's
# load_advanced_indexing, which requires all but one axis to be a
# compile-time ct.Slice.
@ct.kernel
def kernel_gather_basic(a, c, tile_size: ct.Constant[int]):
    ind0 = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1, 1))
    ind1 = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 1, 8))
    t = ct.gather(a, (ind0, ind1))
    ct.store(c, (0, 0, 0), t)
print(f"gather basic (broadcast (4,1,1) x (1,1,8) -> (4,1,8)): {compile_bytes(kernel_gather_basic, sig2to3)} cubin bytes")

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# For a 1-D array, indices may be passed as a bare tile instead of a
# length-1 tuple -- documented as "strictly equivalent." Naming-
# controlled comparison (both variants named "kernel_fn") confirms this
# equivalence at the byte level, not just by reading the docstring.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, (ind,))
    ct.store(c, (0,), t)
n_tuple = compile_bytes(kernel_fn, sig1)
print(f"gather 1D with (ind,) tuple: {n_tuple} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_bare = compile_bytes(kernel_fn, sig1)
print(f"gather 1D with bare ind tile: {n_bare} cubin bytes")
print(f"identical bytes: {n_tuple == n_bare}")
```

Genuinely run:

```
gather basic (broadcast (4,1,1) x (1,1,8) -> (4,1,8)): 23864 cubin bytes
gather 1D with (ind,) tuple: 20000 cubin bytes
gather 1D with bare ind tile: 20000 cubin bytes
identical bytes: True
```

### Discussion

The basic gather compiles without incident, confirming this book can build the exact broadcasting pattern the docstring describes. The convenience-form comparison is a clean, direct confirmation rather than an assumption: passing a bare tile and passing an explicit length-1 tuple produce byte-for-byte identical cubins under naming control, exactly matching the docstring's claim that the two are "strictly equivalent" rather than merely similar. What makes `gather` structurally different from anything in Chapter 23 is not visible in either of these two examples individually — both could, in principle, have been expressed with `load_advanced_indexing` too, since each only has one axis genuinely varying. That difference is worth deferring to this chapter's capstone, where both axes vary independently at once.

## 26.2 Masking, Padding, Bounds, and a Negative Index This Book Can't Fully Verify

### Intuition

`gather`'s docstring describes three interacting controls: a custom `mask` (substituting `padding_value` where the mask is `False`), automatic bounds-checking (also substituting `padding_value` for an out-of-bounds index), and `check_bounds=False` to disable that automatic check entirely. It also states, in passing, that negative indices are always out of bounds — explicitly not Python's negative-indexing convention. That last claim describes what a *running* kernel would compute, which is exactly the kind of claim this book's `export_kernel`-only, driver-free method has never been able to confirm directly, going back to Chapter 8's `RoundingMode`.

### Background

Five kernels probe this incrementally: a plain gather with every option left at its default, the identical gather with a custom mask added (default `padding_value`), the same mask with a non-default `padding_value` added on top, the plain gather with `check_bounds=False`, and finally a gather using an all-negative compile-time-constant index tile — checking only whether the compiler accepts it, not what it would compute.

### Worked Example 26.2.1 — isolating what each control actually changes

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_plain = compile_bytes(kernel_fn, sig1)
print(f"gather plain (defaults): {n_plain} cubin bytes")

# A custom mask alone (default padding_value=0) already compiles larger
# than the plain case: gather has to generate code to select between a
# loaded value and the padding value, which a plain gather (mask always
# true) apparently doesn't need.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    t = ct.gather(a, ind, mask=mask)
    ct.store(c, (0,), t)
n_mask_only = compile_bytes(kernel_fn, sig1)
print(f"gather with mask only (default padding_value): {n_mask_only} cubin bytes")
print(f"plain == mask_only: {n_plain == n_mask_only}")

# Adding a non-default padding_value on top of the same mask.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    t = ct.gather(a, ind, mask=mask, padding_value=-1)
    ct.store(c, (0,), t)
n_mask_pad = compile_bytes(kernel_fn, sig1)
print(f"gather with mask + padding_value=-1: {n_mask_pad} cubin bytes")

# check_bounds=False disables the bounds-checking mask entirely --
# naming-controlled comparison against the plain (check_bounds=True by
# default) case, to see whether skipping that check changes the
# compiled structure at all. It does: it compiles SMALLER than the
# default, confirming bounds-checking code is genuinely present by
# default and genuinely removed when disabled.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, check_bounds=False)
    ct.store(c, (0,), t)
n_no_bounds = compile_bytes(kernel_fn, sig1)
print(f"gather check_bounds=False: {n_no_bounds} cubin bytes")
print(f"plain == check_bounds=False: {n_plain == n_no_bounds}")

# A negative index, as a compile-time constant tile. The docstring says
# negative indices are always out of bounds (NOT Python's negative-
# index convention) -- but that is a claim about the VALUE gather would
# compute at runtime, which this book's export_kernel-only, driver-free
# environment cannot execute. What it CAN report is whether the
# compiler accepts a negative index at all when building the kernel.
@ct.kernel
def kernel_gather_negative_index(a, c, tile_size: ct.Constant[int]):
    ind = ct.full((8,), -1, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
try:
    n = compile_bytes(kernel_gather_negative_index, sig1)
    print(f"gather with all-negative-constant index tile: {n} cubin bytes")
except Exception as e:
    print(f"gather with all-negative-constant index tile: {type(e).__name__}: {e}")
```

Genuinely run:

```
gather plain (defaults): 20000 cubin bytes
gather with mask only (default padding_value): 20128 cubin bytes
plain == mask_only: False
gather with mask + padding_value=-1: 20128 cubin bytes
gather check_bounds=False: 19872 cubin bytes
plain == check_bounds=False: False
gather with all-negative-constant index tile: 19888 cubin bytes
```

### Discussion

Two of the three controls have a directly observable effect on compiled size, and one does not. Adding a custom `mask` costs 128 bytes over the plain case, presumably because the compiler now has to generate code to select between a loaded value and `padding_value` at every element, rather than unconditionally trusting the load. Changing `padding_value` from its default `0` to `-1` on top of that same mask costs nothing further — `20128` bytes either way — which makes sense once the mask-selection code is already understood to exist: which specific constant gets substituted doesn't change how much code substituting it requires. Disabling bounds-checking entirely compiles *smaller* than the default, confirming directly that the default `check_bounds=True` genuinely adds code of its own, separate from any user-supplied mask, and that turning it off is not a no-op. The negative-index case is this section's reminder of a boundary this book has met before: the compiler accepts an all-`-1` index tile without complaint, which confirms only that nothing about a negative *value*, specifically, is caught at compile time — it says nothing about whether the resulting kernel would actually read out-of-bounds memory, return `padding_value`, or do something else entirely, since that would require a real launch this book's `export_kernel`-only environment cannot perform.

## 26.3 The latency Hint: A Validated Parameter With No Visible Effect

### Intuition

`gather`'s signature includes a `latency` parameter that its own docstring never mentions — but the compiler validates it anyway, which is itself informative: a parameter with no documented purpose but a hard-enforced range is presumably a scheduling hint of some kind, intended to influence when the compiler issues the underlying memory operation relative to other work, rather than to change what gets computed.

### Background

A `latency=5` gather is compiled and compared, under naming control, against a gather with no `latency` argument at all, to see whether supplying the hint changes anything observable from outside the compiler. Two out-of-range values — `11` and `0` — are then tried, to find the actual boundaries of whatever range the compiler enforces.

### Worked Example 26.3.1 — a validated hint that never moves a byte count

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# gather's latency parameter is a scheduling hint, not documented in
# its own docstring but validated by the compiler: must be an integer
# between 1 and 10, inclusive.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, latency=5)
    ct.store(c, (0,), t)
n_latency5 = compile_bytes(kernel_fn, sig1)
print(f"gather latency=5: {n_latency5} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_no_latency = compile_bytes(kernel_fn, sig1)
print(f"gather with no latency hint: {n_no_latency} cubin bytes")
print(f"latency=5 == no latency hint: {n_latency5 == n_no_latency}")

@ct.kernel
def kernel_gather_latency_too_high(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, latency=11)
    ct.store(c, (0,), t)
try:
    n = compile_bytes(kernel_gather_latency_too_high, sig1)
    print(f"gather latency=11: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"gather latency=11: {type(e).__name__}: {e}")

@ct.kernel
def kernel_gather_latency_too_low(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, latency=0)
    ct.store(c, (0,), t)
try:
    n = compile_bytes(kernel_gather_latency_too_low, sig1)
    print(f"gather latency=0: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"gather latency=0: {type(e).__name__}: {e}")
```

Genuinely run:

```
gather latency=5: 20000 cubin bytes
gather with no latency hint: 20000 cubin bytes
latency=5 == no latency hint: True
gather latency=11: TileValueError: Latency must be between 1 and 10, got 11
  "/tmp/ch26/03_gather_latency_hint_validation.py", line 43, col 9-37, in kernel_gather_latency_too_high:
        t = ct.gather(a, ind, latency=11)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

gather latency=0: TileValueError: Latency must be between 1 and 10, got 0
  "/tmp/ch26/03_gather_latency_hint_validation.py", line 54, col 9-36, in kernel_gather_latency_too_low:
        t = ct.gather(a, ind, latency=0)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

`latency=5` and no `latency` argument at all compile to the identical byte count, confirming that this hint, whatever it does, does not change the generated code's *size* — consistent with it being a scheduling hint (perhaps influencing instruction ordering or pipelining choices the assembler makes at a finer grain than this book's byte-count method can distinguish) rather than a structural one like `mask` or `check_bounds`. The valid range turns out to be exactly `1` through `10` inclusive, and both boundary violations raise `TileValueError` with a message that states the actual value received alongside the valid range — the same exception type this book first triggered in Chapter 25 from an entirely unrelated tuple-unpacking mismatch, here validating a numeric scheduling parameter instead. This is a useful, if modest, reminder that a compiled byte count is not a universal detector for "did this argument do anything" — it can only ever confirm that something changed the generated *code*, and `latency` is direct evidence that a real, validated, presumably load-bearing parameter can still leave no trace this book's method is equipped to see.

## 26.4 ct.scatter: Gather's Write Counterpart, and Its Own Validator

### Intuition

`ct.scatter(array, indices, value)` mirrors `gather`'s addressing scheme exactly — the same per-axis index tuple, the same broadcasting rule, the same 1D convenience form — but writes `value` into `array` at the computed positions instead of reading from it. One asymmetry is visible directly in its signature: there is no `padding_value` parameter at all. That absence makes sense once the underlying operations are compared: a masked-out or out-of-bounds *read* has to return something, so `gather` needs a substitute value; a masked-out or out-of-bounds *write* simply doesn't have to happen at all, so there is nothing to substitute.

### Background

A 2-dimensional scatter with broadcastable index shapes `(4, 1)` and `(1, 8)`, storing a `(4, 8)` value tile, matches the docstring's own worked example. A 1-dimensional scatter is then compiled plain and with `check_bounds=False`, to see whether disabling bounds-checking affects `scatter`'s size the same way Section 26.2 found it affects `gather`'s. Finally, passing a length-1 indices tuple to a rank-2 array tests whether `scatter` validates its own tuple-length-versus-rank rule independently of `gather`'s.

### Worked Example 26.4.1 — scatter's basic form, bounds-checking, and rank validation

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
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

# ct.scatter(array, indices, value) is gather's write counterpart:
# broadcastable per-axis index shapes (4,1) and (1,8), value shape
# (4, 8) -- matching the docstring's own example exactly, with `value`
# broadcast to the shared (4, 8) shape of the indices.
@ct.kernel
def kernel_scatter_basic(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0, 0), (4, 8))
    ind0 = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1))
    ind1 = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 8))
    ct.scatter(c, (ind0, ind1), x)
print(f"scatter basic (2D indices, no mask): {compile_bytes(kernel_scatter_basic, sig2)} cubin bytes")

# scatter has no padding_value parameter at all, unlike gather -- a
# masked-out or out-of-bounds element simply isn't stored, so there's
# nothing to substitute in its place. Its mask and check_bounds
# parameters otherwise mirror gather's.
sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, ind, x)
n_plain = compile_bytes(kernel_fn, sig1)
print(f"scatter 1D plain (defaults): {n_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, ind, x, check_bounds=False)
n_no_bounds = compile_bytes(kernel_fn, sig1)
print(f"scatter 1D check_bounds=False: {n_no_bounds} cubin bytes")
print(f"plain == check_bounds=False: {n_plain == n_no_bounds}")

# scatter validates that indices' tuple length matches the array's
# rank, the same rule Chapter 23 found shared between
# load_advanced_indexing and store_advanced_indexing -- but this is a
# genuinely different validator (gather/scatter's own), reporting the
# mismatch in its own words.
sig1to2 = ct.compilation.KernelSignature(
    [array_param(1), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_scatter_wrong_rank(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, (ind,), x)
try:
    n = compile_bytes(kernel_scatter_wrong_rank, sig1to2)
    print(f"scatter indices tuple length 1 into rank-2 array: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"scatter indices tuple length 1 into rank-2 array: {type(e).__name__}: {e}")
```

Genuinely run:

```
scatter basic (2D indices, no mask): 22848 cubin bytes
scatter 1D plain (defaults): 20256 cubin bytes
scatter 1D check_bounds=False: 20128 cubin bytes
plain == check_bounds=False: False
scatter indices tuple length 1 into rank-2 array: TileTypeError: For array of rank 2, `indices` must be a tuple of length 2. However, `indices` has type Tuple[Tile[int32,(8)]].
  "/tmp/ch26/04_scatter_basics_and_validation.py", line 71, col 5-28, in kernel_scatter_wrong_rank:
        ct.scatter(c, (ind,), x)
        ^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

`scatter`'s basic form and its rank-length validator both work exactly as `gather`'s own documentation would predict, confirming the two operations really do share a common addressing scheme rather than merely resembling each other in their written descriptions. `check_bounds=False` reduces `scatter`'s compiled size just as it does `gather`'s, confirming the bounds-check code that gets removed is a genuine, shared piece of machinery rather than something `gather` alone carries. The rank-mismatch error message is worth comparing against Chapter 23's own tuple-arity error for `load_advanced_indexing`/`store_advanced_indexing` ("index length 1 does not match array rank 2"): the wording is different enough — "must be a tuple of length 2. However, `indices` has type Tuple[...]" — to confirm this book's earlier finding that Chapter 23's validator is shared code was specific to that one family; `gather`/`scatter` clearly implements its own separate check, in its own words, even though the two families are solving a similar-sounding problem.

## 26.5 A Genuine Asymmetry: Why gather's Mask Costs Bytes and scatter's Doesn't

### Intuition

Section 26.2 found that adding a custom `mask` to `gather` costs measurable bytes over the unmasked case. `scatter` accepts an identically-shaped `mask` parameter, described in near-identical language in its own docstring — "where the mask is False, no store occurs." Whether that costs `scatter` the same kind of bytes `gather` paid is not something either docstring answers on its own; it has to be measured directly, side by side, in one script.

### Background

`scatter`'s plain and masked-only forms are compiled under naming control, exactly mirroring Section 26.2's `gather` comparison, and the two results — `gather`'s and `scatter`'s — are placed in the same script so the comparison is genuinely apples-to-apples rather than two numbers read from different files days apart.

### Worked Example 26.5.1 — the same mask addition, on both operations, in one script

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Section 26.2 found that a custom mask makes gather compile larger
# than the plain, unmasked case. Does the identical isolated change --
# adding a mask, nothing else -- do the same thing to scatter?
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, ind, x)
n_scatter_plain = compile_bytes(kernel_fn, sig1)
print(f"scatter plain (defaults): {n_scatter_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    ct.scatter(c, ind, x, mask=mask)
n_scatter_mask = compile_bytes(kernel_fn, sig1)
print(f"scatter with mask only: {n_scatter_mask} cubin bytes")
print(f"scatter plain == scatter with mask: {n_scatter_plain == n_scatter_mask}")

# The same comparison for gather, repeated here for a direct side-by-
# side rather than relying on Section 26.2's numbers from a different
# script.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_gather_plain = compile_bytes(kernel_fn, sig1)
print(f"gather plain (defaults): {n_gather_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    t = ct.gather(a, ind, mask=mask)
    ct.store(c, (0,), t)
n_gather_mask = compile_bytes(kernel_fn, sig1)
print(f"gather with mask only: {n_gather_mask} cubin bytes")
print(f"gather plain == gather with mask: {n_gather_plain == n_gather_mask}")
```

Genuinely run:

```
scatter plain (defaults): 20256 cubin bytes
scatter with mask only: 20256 cubin bytes
scatter plain == scatter with mask: True
gather plain (defaults): 20000 cubin bytes
gather with mask only: 20128 cubin bytes
gather plain == gather with mask: False
```

### Discussion

The asymmetry is real and directly measured, not an artifact of comparing numbers across two different scripts run days apart: in the same file, under the same naming control, adding a mask changes `gather`'s byte count and does not change `scatter`'s at all. A plausible, though unverifiable-from-this-book's-method, explanation follows from Section 26.2's own finding that `check_bounds=True` (the default for both operations) already generates its own masking code — a conditional that decides, per element, whether to actually perform the memory access. For `gather`, an *additional* custom mask has to be combined with that existing bounds-check condition and then used to select between the loaded value and `padding_value` — genuinely new code, since there was no "substitute a value" logic before. For `scatter`, a custom mask can potentially be logically ANDed directly into the exact same conditional-write mechanism the bounds-check was already using — no store happens either way once a location is excluded, whether it was excluded by the mask or by the bounds check, so there may simply be no new *kind* of code left to generate. This book's `export_kernel`-only environment cannot inspect the compiler's intermediate representation to confirm this mechanism directly, but the measured asymmetry itself is not in question — only its explanation carries this appropriate degree of hedging.

## 26.6 Capstone: A Fully General 2D Remap, and Why load_advanced_indexing Can't Do It

### Intuition

Every example so far in this chapter has technically had only one axis varying dynamically at a time, which means every one of them could, in principle, have been expressed with Chapter 23's `load_advanced_indexing` instead. The genuine reason to reach for `gather`/`scatter` only shows up once *every* axis needs to vary independently and simultaneously — a fully general remap where each output element's source coordinates are computed from entirely separate, unrelated expressions per axis. This capstone builds exactly that, and then attempts the identical task with `load_advanced_indexing` directly, to confirm the restriction is real rather than assumed.

### Background

A `(4, 8)` tile is remapped using two full `(4, 8)` index tiles — `row_idx`, computed as `(row + col) % 4`, and `col_idx`, computed as `(col + 1) % 8` — each varying across both dimensions simultaneously rather than being broadcast from a single 1D range along one axis. `ct.gather` reads the remapped values, and `ct.scatter` writes them back using the identical two index tiles. The same two fully-2D index tiles are then passed to `ct.load_advanced_indexing`, which requires its sparse index to be strictly 1-dimensional, to confirm directly that it rejects what `gather` accepts.

### Worked Example 26.6.1 — a remap ct.gather/ct.scatter can express and ct.load_advanced_indexing cannot

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
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

# Capstone: a fully general 2D remap, where every output element reads
# from and writes to an independently-computed (row, col) pair -- both
# axes vary per element simultaneously. This is something Chapter 23's
# ct.load_advanced_indexing/ct.store_advanced_indexing structurally
# cannot express, since they require all but one axis to be addressed
# by a compile-time ct.Slice. ct.gather and ct.scatter place no such
# restriction on how many axes are dynamic at once.
@ct.kernel
def kernel_gather_scatter_remap(a, c, tile_size: ct.Constant[int]):
    row = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1))
    col = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 8))
    row_idx = ct.broadcast_to((row + col) % 4, (4, 8))
    col_idx = ct.broadcast_to((col + 1) % 8, (4, 8))
    remapped = ct.gather(a, (row_idx, col_idx))
    ct.scatter(c, (row_idx, col_idx), remapped)
n = compile_bytes(kernel_gather_scatter_remap, sig)
print(f"gather+scatter fully-general 2D remap: {n} cubin bytes")

# Attempting the identical remap with ct.load_advanced_indexing directly
# confirms the restriction is real, not just an assumption carried over
# from Chapter 23: passing two independently-varying 2D index tiles,
# where load_advanced_indexing expects exactly one 1D "sparse" index
# tile and compile-time ct.Slice objects everywhere else, is rejected.
@ct.kernel
def kernel_advanced_indexing_two_dynamic_axes(a, c, tile_size: ct.Constant[int]):
    row = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1))
    col = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 8))
    row_idx = ct.broadcast_to((row + col) % 4, (4, 8))
    col_idx = ct.broadcast_to((col + 1) % 8, (4, 8))
    t = ct.load_advanced_indexing(a, (row_idx, col_idx))
    ct.store(c, (0, 0), t)
try:
    n = compile_bytes(kernel_advanced_indexing_two_dynamic_axes, sig)
    print(f"load_advanced_indexing with two fully-dynamic axes: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing with two fully-dynamic axes: {type(e).__name__}: {e}")
```

Genuinely run:

```
gather+scatter fully-general 2D remap: 22720 cubin bytes
load_advanced_indexing with two fully-dynamic axes: TileTypeError: Sparse index at dim 0 must be a 1D integer tile, got 2D
  "/tmp/ch26/06_capstone_general_2d_remap_vs_advanced_indexing.py", line 49, col 9-56, in kernel_advanced_indexing_two_dynamic_axes:
        t = ct.load_advanced_indexing(a, (row_idx, col_idx))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

The `gather`+`scatter` remap compiles cleanly, confirming that both operations happily accept fully-independent, per-axis-and-per-element index tiles built from entirely different expressions — `row_idx` depends on both `row` and `col`, `col_idx` depends only on `col`, and neither one is a plain broadcast of a single 1D range the way every earlier example in this chapter used. The `load_advanced_indexing` attempt on the identical index tiles is rejected, and its error message names the exact structural mismatch: "Sparse index at dim 0 must be a 1D integer tile, got 2D." This is not the same rejection wording Chapter 23 catalogued for a wrong number of sparse dimensions ("found 2 at dims [...]") — that check fires when more than one axis is *marked* as sparse via a bare tile rather than a `ct.Slice`. Here, both axes are already being addressed by 2D tiles from the very first argument position, so the validator's first complaint is simply that the tile it's looking at isn't 1-dimensional at all. Either way, the conclusion holds: `load_advanced_indexing`'s restriction to one sparse axis is a genuine structural limit, not a documentation artifact this book merely assumed, and `gather`/`scatter` exist specifically to cover the case that restriction excludes.

## Chapter Summary

This chapter introduced `ct.gather` and `ct.scatter`, a pair of memory operations that generalize Chapter 23's `load_advanced_indexing`/`store_advanced_indexing` by removing its central restriction: every axis can be addressed by its own independently-computed runtime integer tile, rather than requiring all but one axis to be a compile-time `ct.Slice`. `gather`'s 1D convenience form (a bare tile instead of a length-1 tuple) was confirmed byte-identical to the explicit form under naming control. A custom `mask` measurably increases `gather`'s compiled size, while changing `padding_value` on top of an existing mask does not, and disabling `check_bounds` measurably decreases it — three controls, three different observable effects, one of them null. The `latency` scheduling hint, though genuinely validated to the range 1-10 by the compiler, never once changed a compiled byte count in this chapter's tests, a useful reminder of the difference between "this parameter does something" and "this parameter changes what this book's byte-count method can see." `scatter` shares `gather`'s addressing scheme and bounds-checking machinery, but has no `padding_value` at all, since a suppressed write has nothing to substitute — and, measured directly and side by side, adding a mask to `scatter` costs no bytes at all, a real asymmetry with `gather`'s otherwise-parallel behavior. The capstone closed the chapter's central comparison directly: a remap where both axes vary independently compiles cleanly under `gather`/`scatter` and is rejected outright, with a specific and legible error, under `load_advanced_indexing` — confirming Chapter 23's one-sparse-axis restriction is exactly as absolute as this chapter assumed going in.

## Self-Check Questions

1. Section 26.2 found that a custom `mask` costs `gather` measurable bytes, but a changed `padding_value` on top of an existing mask costs nothing further. What does this suggest about whether `padding_value` is best understood as a second, independent feature of `gather`, or as more like a configuration constant fed into machinery the mask parameter already required?
2. Section 26.3 found that `latency` is validated (rejecting `0` and `11`) but never changes a compiled byte count. Suppose a future chapter needed to determine whether `latency` actually changes anything about the compiled kernel at all, given that this book's method cannot launch kernels or measure real execution time. What kind of evidence, short of a real GPU launch, might distinguish "does nothing" from "does something this book's byte-count method can't detect"?
3. Section 26.5 proposed a hypothesis for why `scatter`'s mask costs no bytes while `gather`'s does — that a custom mask can be ANDed directly into `scatter`'s existing bounds-check conditional, while `gather` needs genuinely new code to select a padding value. What observation, if this book's `export_kernel`-only method could make it, would most directly confirm or refute this specific hypothesis (as opposed to some other explanation for the same asymmetry)?
4. Section 26.6's `load_advanced_indexing` rejection said "Sparse index at dim 0 must be a 1D integer tile, got 2D" — a different message from Chapter 23's "found 2 at dims [...]" wording for a different kind of violation. Both messages ultimately arise from the same underlying rule (exactly one sparse, 1D axis). Why might a single validator report a violation of one consistent rule using two structurally different messages, rather than one generic one?
5. This chapter's capstone specifically avoided treating the difference between the `gather`+`scatter` remap's byte count (`22720`) and any `load_advanced_indexing`-based alternative as a "cost" comparison. Given that `load_advanced_indexing` cannot even express the same remap, what would have to be different about the comparison for a byte-count difference between the two approaches to mean anything at all?

## Where We Go Next

This chapter closed the loop on Chapter 23's advanced-indexing restriction by naming and demonstrating the more general mechanism sitting right next to it in `dir(ct)`. Two threads from Chapter 25's own forward-looking research remain untouched: the `ct.tune` module, glimpsed only by name so far, which points toward autotuning — letting the compiler search over implementation choices rather than committing to one the way every kernel in this book so far has. And `ct.function(host=True)`, read in Chapter 25's docstring research but never once exercised, which marks a boundary this book has not yet crossed: every kernel so far has run entirely on the device side of the host/device line, and a host-callable tile function would be the first genuine step across it.

## Worked Solutions

**1.** The evidence points toward `padding_value` being configuration fed into machinery the mask already requires, rather than an independent feature of its own. If `padding_value` triggered genuinely separate code — its own selection logic, say, distinct from whatever the mask itself needs — changing its value from `0` to `-1` would be expected to at least occasionally shift the byte count, the way changing which arithmetic operation a combining function performs shifted `ct.reduce`'s byte count back in Chapter 24. Instead, the mask alone accounts for the entire jump from the plain case, and the specific constant substituted afterward is free. That is consistent with a single mechanism — "select between the loaded value and some constant" — where the mask determines whether that mechanism exists at all, and `padding_value` merely supplies which constant the already-existing mechanism reaches for.

**2.** Without a real launch, the most direct evidence available would be disassembling the compiled cubin itself (with a tool like `cuobjdump`, external to cuTile Python) rather than only counting its total byte length. Two cubins can be identical in overall size while differing in their actual instruction sequence — different padding, alignment, or instruction *selection* could keep the byte count constant even if `latency` changes which specific memory-access instruction variant gets emitted, or where the compiler's scheduler places it relative to surrounding instructions. This book's method has, from the start, treated "the compiled byte count" as its lowest-friction genuine signal precisely because it requires nothing beyond `export_kernel` — but it was never claimed to be a *complete* view of everything a compiler decision can change, and `latency` looks like a case where that gap is real: a parameter the compiler visibly validates and presumably acts on, that this book's chosen method simply isn't fine-grained enough to see moving.

**3.** The hypothesis specifically claims that `scatter`'s custom mask gets combined into the *same* conditional-write instruction the bounds check already produces, rather than needing a separate mechanism. The most direct way to confirm or refute that specific claim — again, without a launch — would be to disassemble both the plain and the masked `scatter` cubins and check whether they contain the same number and kind of predicate-guarded store instructions, with only the predicate's own computation differing (a combined AND of two conditions instead of one). If the masked version turned out to contain an entirely separate branch or store instruction alongside the original bounds-checked one, that would refute the "same conditional, combined predicate" hypothesis even though the total byte count happened to stay flat — a coincidence rather than the proposed mechanism. Byte-count equality alone is consistent with the hypothesis, but doesn't rule out other explanations that also happen to leave the size unchanged.

**4.** A single validator checking one overall rule can still fail at different points in its own reasoning, and each point naturally has different information available to report. Checking whether a specific tile is 1-dimensional is a property of that one tile alone, checkable the moment the validator looks at it — reporting "got 2D" the instant it finds a disqualifying tile is more immediately useful than continuing to scan the rest of the tuple first. Counting how many tiles across the whole tuple qualify as legitimate "sparse" candidates, by contrast, requires having already confirmed each tile's own shape is acceptable, and only then can the validator report a count like "found 2 at dims [...]" — a fact about the tuple as a whole, not about any single tile in isolation. Two different messages, then, aren't evidence of two different underlying rules; they're evidence that the one rule is checked in stages, and a validator reporting the earliest concrete problem it actually found is more informative than forcing every violation through one generic, least-common-denominator message.

**5.** For a byte-count difference to mean anything as a "cost" comparison, both sides of the comparison would first need to be capable of computing the *same* result — otherwise a smaller or larger byte count is just describing two different programs that happen to share a byte-count axis, not two different prices for the identical computation. Since `load_advanced_indexing` cannot express this particular remap at all, the only way to make the comparison meaningful would be to compare `gather`+`scatter` against some other approach that genuinely *can* express the same fully-independent, two-axis-dynamic computation — for instance, an explicitly unrolled sequence of single-element loads and stores built with `ct.static_iter`, one per possible index combination. Even then, this book's own established discipline would apply in full: identical kernel naming (Chapter 10), identical file-path length (Chapter 16), and identical source-line position (Chapter 24) would all need controlling for before any byte-count gap between the two capable-of-the-same-thing approaches could be attributed to one being genuinely more expensive to compile than the other, rather than to any of the three confounds this book has already had to learn to control for.

## Complete Runnable Code

### File: `01_gather_basics_and_convenience_form.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig2to3 = ct.compilation.KernelSignature(
    [array_param(2), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.gather(array, indices) reads array[ind0[...], ind1[...]] for a
# tuple of per-axis index tiles, one per array dimension. Broadcastable
# index shapes (4,1,1) and (1,1,8) produce a result of the broadcasted
# shape (4,1,8) -- every element of the result can, in principle, come
# from an entirely independent (row, col) pair, unlike Chapter 23's
# load_advanced_indexing, which requires all but one axis to be a
# compile-time ct.Slice.
@ct.kernel
def kernel_gather_basic(a, c, tile_size: ct.Constant[int]):
    ind0 = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1, 1))
    ind1 = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 1, 8))
    t = ct.gather(a, (ind0, ind1))
    ct.store(c, (0, 0, 0), t)
print(f"gather basic (broadcast (4,1,1) x (1,1,8) -> (4,1,8)): {compile_bytes(kernel_gather_basic, sig2to3)} cubin bytes")

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# For a 1-D array, indices may be passed as a bare tile instead of a
# length-1 tuple -- documented as "strictly equivalent." Naming-
# controlled comparison (both variants named "kernel_fn") confirms this
# equivalence at the byte level, not just by reading the docstring.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, (ind,))
    ct.store(c, (0,), t)
n_tuple = compile_bytes(kernel_fn, sig1)
print(f"gather 1D with (ind,) tuple: {n_tuple} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_bare = compile_bytes(kernel_fn, sig1)
print(f"gather 1D with bare ind tile: {n_bare} cubin bytes")
print(f"identical bytes: {n_tuple == n_bare}")
```

### File: `02_gather_mask_padding_bounds_and_negative_index.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_plain = compile_bytes(kernel_fn, sig1)
print(f"gather plain (defaults): {n_plain} cubin bytes")

# A custom mask alone (default padding_value=0) already compiles larger
# than the plain case: gather has to generate code to select between a
# loaded value and the padding value, which a plain gather (mask always
# true) apparently doesn't need.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    t = ct.gather(a, ind, mask=mask)
    ct.store(c, (0,), t)
n_mask_only = compile_bytes(kernel_fn, sig1)
print(f"gather with mask only (default padding_value): {n_mask_only} cubin bytes")
print(f"plain == mask_only: {n_plain == n_mask_only}")

# Adding a non-default padding_value on top of the same mask.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    t = ct.gather(a, ind, mask=mask, padding_value=-1)
    ct.store(c, (0,), t)
n_mask_pad = compile_bytes(kernel_fn, sig1)
print(f"gather with mask + padding_value=-1: {n_mask_pad} cubin bytes")

# check_bounds=False disables the bounds-checking mask entirely --
# naming-controlled comparison against the plain (check_bounds=True by
# default) case, to see whether skipping that check changes the
# compiled structure at all. It does: it compiles SMALLER than the
# default, confirming bounds-checking code is genuinely present by
# default and genuinely removed when disabled.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, check_bounds=False)
    ct.store(c, (0,), t)
n_no_bounds = compile_bytes(kernel_fn, sig1)
print(f"gather check_bounds=False: {n_no_bounds} cubin bytes")
print(f"plain == check_bounds=False: {n_plain == n_no_bounds}")

# A negative index, as a compile-time constant tile. The docstring says
# negative indices are always out of bounds (NOT Python's negative-
# index convention) -- but that is a claim about the VALUE gather would
# compute at runtime, which this book's export_kernel-only, driver-free
# environment cannot execute. What it CAN report is whether the
# compiler accepts a negative index at all when building the kernel.
@ct.kernel
def kernel_gather_negative_index(a, c, tile_size: ct.Constant[int]):
    ind = ct.full((8,), -1, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
try:
    n = compile_bytes(kernel_gather_negative_index, sig1)
    print(f"gather with all-negative-constant index tile: {n} cubin bytes")
except Exception as e:
    print(f"gather with all-negative-constant index tile: {type(e).__name__}: {e}")
```

### File: `03_gather_latency_hint_validation.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# gather's latency parameter is a scheduling hint, not documented in
# its own docstring but validated by the compiler: must be an integer
# between 1 and 10, inclusive.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, latency=5)
    ct.store(c, (0,), t)
n_latency5 = compile_bytes(kernel_fn, sig1)
print(f"gather latency=5: {n_latency5} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_no_latency = compile_bytes(kernel_fn, sig1)
print(f"gather with no latency hint: {n_no_latency} cubin bytes")
print(f"latency=5 == no latency hint: {n_latency5 == n_no_latency}")

@ct.kernel
def kernel_gather_latency_too_high(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, latency=11)
    ct.store(c, (0,), t)
try:
    n = compile_bytes(kernel_gather_latency_too_high, sig1)
    print(f"gather latency=11: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"gather latency=11: {type(e).__name__}: {e}")

@ct.kernel
def kernel_gather_latency_too_low(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind, latency=0)
    ct.store(c, (0,), t)
try:
    n = compile_bytes(kernel_gather_latency_too_low, sig1)
    print(f"gather latency=0: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"gather latency=0: {type(e).__name__}: {e}")
```

### File: `04_scatter_basics_and_validation.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
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

# ct.scatter(array, indices, value) is gather's write counterpart:
# broadcastable per-axis index shapes (4,1) and (1,8), value shape
# (4, 8) -- matching the docstring's own example exactly, with `value`
# broadcast to the shared (4, 8) shape of the indices.
@ct.kernel
def kernel_scatter_basic(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0, 0), (4, 8))
    ind0 = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1))
    ind1 = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 8))
    ct.scatter(c, (ind0, ind1), x)
print(f"scatter basic (2D indices, no mask): {compile_bytes(kernel_scatter_basic, sig2)} cubin bytes")

# scatter has no padding_value parameter at all, unlike gather -- a
# masked-out or out-of-bounds element simply isn't stored, so there's
# nothing to substitute in its place. Its mask and check_bounds
# parameters otherwise mirror gather's.
sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, ind, x)
n_plain = compile_bytes(kernel_fn, sig1)
print(f"scatter 1D plain (defaults): {n_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, ind, x, check_bounds=False)
n_no_bounds = compile_bytes(kernel_fn, sig1)
print(f"scatter 1D check_bounds=False: {n_no_bounds} cubin bytes")
print(f"plain == check_bounds=False: {n_plain == n_no_bounds}")

# scatter validates that indices' tuple length matches the array's
# rank, the same rule Chapter 23 found shared between
# load_advanced_indexing and store_advanced_indexing -- but this is a
# genuinely different validator (gather/scatter's own), reporting the
# mismatch in its own words.
sig1to2 = ct.compilation.KernelSignature(
    [array_param(1), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_scatter_wrong_rank(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, (ind,), x)
try:
    n = compile_bytes(kernel_scatter_wrong_rank, sig1to2)
    print(f"scatter indices tuple length 1 into rank-2 array: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"scatter indices tuple length 1 into rank-2 array: {type(e).__name__}: {e}")
```

### File: `05_mask_asymmetry_between_gather_and_scatter.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Section 26.2 found that a custom mask makes gather compile larger
# than the plain, unmasked case. Does the identical isolated change --
# adding a mask, nothing else -- do the same thing to scatter?
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    ct.scatter(c, ind, x)
n_scatter_plain = compile_bytes(kernel_fn, sig1)
print(f"scatter plain (defaults): {n_scatter_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    ct.scatter(c, ind, x, mask=mask)
n_scatter_mask = compile_bytes(kernel_fn, sig1)
print(f"scatter with mask only: {n_scatter_mask} cubin bytes")
print(f"scatter plain == scatter with mask: {n_scatter_plain == n_scatter_mask}")

# The same comparison for gather, repeated here for a direct side-by-
# side rather than relying on Section 26.2's numbers from a different
# script.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    t = ct.gather(a, ind)
    ct.store(c, (0,), t)
n_gather_plain = compile_bytes(kernel_fn, sig1)
print(f"gather plain (defaults): {n_gather_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    ind = ct.arange(8, dtype=ct.int32)
    mask = ind < 4
    t = ct.gather(a, ind, mask=mask)
    ct.store(c, (0,), t)
n_gather_mask = compile_bytes(kernel_fn, sig1)
print(f"gather with mask only: {n_gather_mask} cubin bytes")
print(f"gather plain == gather with mask: {n_gather_plain == n_gather_mask}")
```

### File: `06_capstone_general_2d_remap_vs_advanced_indexing.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
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

# Capstone: a fully general 2D remap, where every output element reads
# from and writes to an independently-computed (row, col) pair -- both
# axes vary per element simultaneously. This is something Chapter 23's
# ct.load_advanced_indexing/ct.store_advanced_indexing structurally
# cannot express, since they require all but one axis to be addressed
# by a compile-time ct.Slice. ct.gather and ct.scatter place no such
# restriction on how many axes are dynamic at once.
@ct.kernel
def kernel_gather_scatter_remap(a, c, tile_size: ct.Constant[int]):
    row = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1))
    col = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 8))
    row_idx = ct.broadcast_to((row + col) % 4, (4, 8))
    col_idx = ct.broadcast_to((col + 1) % 8, (4, 8))
    remapped = ct.gather(a, (row_idx, col_idx))
    ct.scatter(c, (row_idx, col_idx), remapped)
n = compile_bytes(kernel_gather_scatter_remap, sig)
print(f"gather+scatter fully-general 2D remap: {n} cubin bytes")

# Attempting the identical remap with ct.load_advanced_indexing directly
# confirms the restriction is real, not just an assumption carried over
# from Chapter 23: passing two independently-varying 2D index tiles,
# where load_advanced_indexing expects exactly one 1D "sparse" index
# tile and compile-time ct.Slice objects everywhere else, is rejected.
@ct.kernel
def kernel_advanced_indexing_two_dynamic_axes(a, c, tile_size: ct.Constant[int]):
    row = ct.reshape(ct.arange(4, dtype=ct.int32), (4, 1))
    col = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 8))
    row_idx = ct.broadcast_to((row + col) % 4, (4, 8))
    col_idx = ct.broadcast_to((col + 1) % 8, (4, 8))
    t = ct.load_advanced_indexing(a, (row_idx, col_idx))
    ct.store(c, (0, 0), t)
try:
    n = compile_bytes(kernel_advanced_indexing_two_dynamic_axes, sig)
    print(f"load_advanced_indexing with two fully-dynamic axes: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing with two fully-dynamic axes: {type(e).__name__}: {e}")
```
