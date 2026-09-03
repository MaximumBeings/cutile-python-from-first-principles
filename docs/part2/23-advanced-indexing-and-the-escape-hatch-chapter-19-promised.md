# Chapter 23: Advanced Indexing and the Escape Hatch Chapter 19 Promised

> "Every load and store this book has written until now addressed memory through a single contiguous block-offset, the same `(pid,)`-plus-`tile_size` convention since Chapter 1. This chapter opens a second addressing model — one runtime-indexed dimension mixed with any number of ordinary contiguous ones — and then goes further still, to the raw bytes underneath a tile's dtype entirely."

**What you will understand by the end of this chapter:**

- `ct.load_advanced_indexing` and `ct.store_advanced_indexing`: a addressing convention distinct from both this book's ordinary `ct.load`/`ct.store` and Chapter 20's `gather()`/`scatter()` — exactly one dimension addressed by a runtime 1D integer tile (the sparse dim), every other dimension addressed by a compile-time-sized contiguous `ct.Slice` (a dense dim)
- That both functions share a single validator, revealed directly by an error message that names both functions by name regardless of which one you called — confirmed by testing zero sparse dims, two sparse dims, and a too-short `indices` tuple
- That `store_advanced_indexing`'s stored tile shape is checked against the shape the indices themselves imply, the same store-shape discipline this book has seen since Chapter 3, now applied to a shape computed from a tuple of mixed sparse/dense descriptors rather than declared upfront
- `ct.bitcast`: reinterpreting a tile's raw bits as a different, equal-width dtype, and — tested directly for the first time in this book — that it is genuinely usable as Chapter 19's own error messages promised, letting `bitwise_and` operate on `float32` data by bitcasting through `uint32` and back
- `ct.pack_to_bytes` and `ct.unpack_from_bytes`: a documented inverse pair that flattens a tile to its raw `uint8` bytes and back, with their own independent validation — a rejected `bool_` input, a rejected 2D input, and a bit-width divisibility check distinct from `bitcast`'s equal-width requirement

**What you need to know first:**

- Chapter 19's bitwise-operation rejection messages, which recommended "an explicit `cuda.tile.bitcast()` for non-integer operands" without this book ever calling it — this chapter closes that thread.
- Chapter 20's `gather()`/`scatter()` indexing convention and Chapter 15's power-of-two tile-shape rule, both of which reappear here in a new context.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow this book has used throughout.

## 23.1 ct.load_advanced_indexing: One Sparse Dim, Many Dense Slices

### Intuition

`ct.load_advanced_indexing(array, indices)` reads a tile from `array` using a mixed addressing scheme: `indices` must have exactly `array.ndim` entries, and exactly one of them must be a 1D integer tile — the "sparse dim," whose values are runtime element-space indices selecting which rows (or columns) of that axis to gather. Every other entry must be a `ct.Slice(start, length)` — a "dense dim," an ordinary contiguous range where `length` is a compile-time power of two. This is a third addressing convention alongside this book's `ct.load`/`ct.store` block-offset addressing and Chapter 20's `gather()`/`scatter()` full-tile-of-indices addressing — here, only one axis gets to be runtime-indexed at all.

### Background

Testing the sparse dim in both possible positions of a 2D array establishes the ordinary case. Then, three deliberately invalid `indices` tuples — one too short, one with no sparse dim at all, one with two — probe how the "exactly one" requirement is actually enforced.

### Worked Example 23.1.1 — sparse dim in either position, then three ways to violate "exactly one"

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

# ct.load_advanced_indexing(array, indices) reads a tile from
# non-contiguous slices: exactly one entry in indices must be a 1D
# integer tile (the sparse dim), and the rest must be ct.Slice objects
# (dense dims). Basic 2D case: sparse dim 0, dense dim 1.
@ct.kernel
def kernel_load_basic(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(2, 4)))
    ct.store(c, (0, 0), x)
print(f"load_advanced_indexing sparse=dim0, dense=dim1: {compile_bytes(kernel_load_basic, sig2)} cubin bytes")

# Sparse dim in the other position: sparse dim 1, dense dim 0.
@ct.kernel
def kernel_load_basic_swapped(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    col_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (ct.Slice(2, 4), col_indices))
    ct.store(c, (0, 0), x)
print(f"load_advanced_indexing sparse=dim1, dense=dim0: {compile_bytes(kernel_load_basic_swapped, sig2)} cubin bytes")

# indices tuple shorter than array.ndim.
@ct.kernel
def kernel_load_short_indices(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices,))
    ct.store(c, (0, 0), x)
try:
    n = compile_bytes(kernel_load_short_indices, sig2)
    print(f"load_advanced_indexing indices len=1 (array ndim=2): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing indices len=1 (array ndim=2): {type(e).__name__}: {e}")

# Zero sparse dims: both entries are Slice objects, no 1D tile at all.
@ct.kernel
def kernel_load_no_sparse(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load_advanced_indexing(a, (ct.Slice(0, 4), ct.Slice(0, 4)))
    ct.store(c, (0, 0), x)
try:
    n = compile_bytes(kernel_load_no_sparse, sig2)
    print(f"load_advanced_indexing zero sparse dims (both Slice): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing zero sparse dims (both Slice): {type(e).__name__}: {e}")

# Two sparse dims: both entries are 1D tiles.
@ct.kernel
def kernel_load_two_sparse(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    col_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, col_indices))
    ct.store(c, (0, 0), x)
try:
    n = compile_bytes(kernel_load_two_sparse, sig2)
    print(f"load_advanced_indexing two sparse dims (both Tile): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing two sparse dims (both Tile): {type(e).__name__}: {e}")
```

Genuinely run:

```
load_advanced_indexing sparse=dim0, dense=dim1: 22448 cubin bytes
load_advanced_indexing sparse=dim1, dense=dim0: 22848 cubin bytes
load_advanced_indexing indices len=1 (array ndim=2): TileTypeError: load_advanced_indexing/store_advanced_indexing index length 1 does not match array rank 2
  "/tmp/ch23/01_load_advanced_indexing_basics_and_validation.py", line 46, col 9-52, in kernel_load_short_indices:
        x = ct.load_advanced_indexing(a, (row_indices,))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

load_advanced_indexing zero sparse dims (both Slice): TileTypeError: load_advanced_indexing/store_advanced_indexing: exactly one index must be a 1D integer Tile (the sparse dim); none found
  "/tmp/ch23/01_load_advanced_indexing_basics_and_validation.py", line 58, col 9-70, in kernel_load_no_sparse:
        x = ct.load_advanced_indexing(a, (ct.Slice(0, 4), ct.Slice(0, 4)))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

load_advanced_indexing two sparse dims (both Tile): TileTypeError: load_advanced_indexing/store_advanced_indexing: exactly one index must be a 1D integer Tile (the sparse dim); found 2 at dims [0, 1]
  "/tmp/ch23/01_load_advanced_indexing_basics_and_validation.py", line 72, col 9-64, in kernel_load_two_sparse:
        x = ct.load_advanced_indexing(a, (row_indices, col_indices))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

All three invalid cases are caught, and every one of their messages opens with the identical phrase: "load_advanced_indexing/store_advanced_indexing." That shared prefix is a small but genuine implementation detail surfacing through the error text — both functions are validated by one common routine, not two independently-written ones, and that routine reports itself under both names regardless of which one the caller actually invoked. The two-sparse-dims case also goes further than a bare rejection: it reports "found 2 at dims [0, 1]," naming the exact positions responsible, in keeping with this book's now-familiar pattern of error messages that do the caller's diagnostic arithmetic for them. The valid cases succeed with either axis playing the sparse role, confirming the convention is genuinely position-agnostic — nothing about dim 0 or dim 1 is privileged, only whichever entry happens to be a tile.

## 23.2 ct.store_advanced_indexing: The Same Convention, Enforced Together

### Intuition

`ct.store_advanced_indexing(array, indices, tile)` uses the identical `indices` convention as `load_advanced_indexing`, in reverse: the tile being written must have a shape that exactly matches what `indices` implies — the sparse dim's length equal to the index tile's own length, the dense dims equal to their `Slice.length` values. This is the same store-shape discipline this book has enforced since Chapter 3's earliest examples, just computed from a more elaborate description of the destination than a fixed declared shape. `ct.Slice`'s own docstring separately requires `length` to be a compile-time power of two — a rule this chapter can test directly, since Section 23.1 already confirmed both functions share one validator.

### Background

A matching-shape store establishes the ordinary case. A shape mismatch — storing a `(4, 8)` tile where the indices imply `(4, 4)` — tests the store-shape check. A `ct.Slice` with a non-power-of-two length tests whether that rule is enforced even before the store-shape question comes up.

### Worked Example 23.2.1 — a matching store, a shape mismatch, and a non-power-of-two Slice

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
    [array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.store_advanced_indexing(array, indices, tile) uses the same
# convention in reverse: one sparse dim, the rest dense Slices, and
# the stored tile's shape must exactly match the shape the indices imply.
@ct.kernel
def kernel_store_basic(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32) + 1
    t = ct.full((4, 4), 1, dtype=a.dtype)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), t)
print(f"store_advanced_indexing matching shape (4,4): {compile_bytes(kernel_store_basic, sig2)} cubin bytes")

# A tile whose shape does not match what the indices imply: indices
# imply (4, 4), but the tile being stored is (4, 8).
@ct.kernel
def kernel_store_shape_mismatch(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32) + 1
    t = ct.full((4, 8), 1, dtype=a.dtype)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), t)
try:
    n = compile_bytes(kernel_store_shape_mismatch, sig2)
    print(f"store_advanced_indexing mismatched tile shape (4,8) vs implied (4,4): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"store_advanced_indexing mismatched tile shape (4,8) vs implied (4,4): {type(e).__name__}: {e}")

# ct.Slice's own docstring says length "must be a power of two." A
# Slice with a non-power-of-two length should be rejected -- the same
# validation load_advanced_indexing and store_advanced_indexing share.
@ct.kernel
def kernel_load_slice_nonpow2(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(0, 3)))
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 3)), x)
try:
    n = compile_bytes(kernel_load_slice_nonpow2, sig2)
    print(f"load_advanced_indexing Slice(0, 3) non-power-of-two length: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing Slice(0, 3) non-power-of-two length: {type(e).__name__}: {e}")
```

Genuinely run:

```
store_advanced_indexing matching shape (4,4): 19432 cubin bytes
store_advanced_indexing mismatched tile shape (4,8) vs implied (4,4): TileTypeError: Tile shape (4, 8) does not match the shape implied by indices (4, 4)
  "/tmp/ch23/02_store_advanced_indexing_and_shape_matching.py", line 38, col 5-67, in kernel_store_shape_mismatch:
        ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), t)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

load_advanced_indexing Slice(0, 3) non-power-of-two length: TileTypeError: Index at dim 1 has size 3; must be a power of two
  "/tmp/ch23/02_store_advanced_indexing_and_shape_matching.py", line 52, col 9-67, in kernel_load_slice_nonpow2:
        x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(0, 3)))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

The shape-mismatch error states both shapes plainly — "Tile shape (4, 8) does not match the shape implied by indices (4, 4)" — rather than a bare dimension count, letting the caller see exactly which dimension is off without recomputing what the indices imply by hand. The non-power-of-two `Slice` is rejected before the call ever reaches the sparse-versus-dense counting logic from Section 23.1, since the error names dimension 1's size specifically rather than complaining about the tuple's structure — confirming `Slice.length`'s power-of-two requirement is checked independently, and early, in the same validation pipeline both functions share.

## 23.3 padding_mode: A Parameter This Book Can't Yet See Do Anything

### Intuition

`load_advanced_indexing`'s docstring describes `padding_mode` as controlling "the fill value for OOB elements on both sparse and dense dims," defaulting to `PaddingMode.UNDETERMINED`. Chapter 8 established that this book's compile-time-only, driver-free environment can confirm a `RoundingMode` enum value is *accepted*, but never confirm what it actually changes about computed results, since no kernel here is ever launched. `padding_mode` raises the same question in a sharper form: for a load the compiler can prove is always in-bounds, does choosing a specific fill value like `ZERO` over the default `UNDETERMINED` change the compiled kernel at all?

### Background

Compiling the identical load/store round trip under `padding_mode=PaddingMode.UNDETERMINED` and `padding_mode=PaddingMode.ZERO`, using Chapter 10's naming-confound control (the identical function name `kernel_fn` for both), isolates whether the two modes produce different code.

### Worked Example 23.3.1 — two padding modes, one function name

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
    [array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# padding_mode controls the fill value for out-of-bounds elements on
# both the sparse and dense dims. Does varying it change the compiled
# byte count for a load that a compiler could, in principle, prove is
# always in-bounds? Naming-confound-controlled (Chapter 10): both
# variants reuse the identical function name "kernel_fn".
@ct.kernel
def kernel_fn(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), padding_mode=ct.PaddingMode.UNDETERMINED)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), x)
n_undetermined = compile_bytes(kernel_fn, sig2)
print(f"load_advanced_indexing padding_mode=UNDETERMINED, naming-controlled: {n_undetermined} cubin bytes")

@ct.kernel
def kernel_fn(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), padding_mode=ct.PaddingMode.ZERO)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), x)
n_zero = compile_bytes(kernel_fn, sig2)
print(f"load_advanced_indexing padding_mode=ZERO, naming-controlled: {n_zero} cubin bytes")
print(f"identical bytes: {n_undetermined == n_zero}")
```

Genuinely run:

```
load_advanced_indexing padding_mode=UNDETERMINED, naming-controlled: 19416 cubin bytes
load_advanced_indexing padding_mode=ZERO, naming-controlled: 19416 cubin bytes
identical bytes: True
```

### Discussion

The two modes compile to byte-identical cubins. This is consistent with, though it does not prove, the idea that this particular load — four rows selected from an unconstrained-shape array, at a compile-time-fixed dense offset — gives the compiler no static guarantee of being in-bounds or out-of-bounds either way, so both padding modes would only diverge in behavior at runtime, on data this book's `export_kernel`-only environment never sees. This is exactly the same epistemic wall Chapter 8 hit with `RoundingMode`: this book can confirm `padding_mode` is *accepted* as a valid keyword argument for both enum values tested, and that neither one is rejected outright, but confirming that the two modes actually produce different runtime behavior — different fill values landing in an out-of-bounds read — remains outside what a driver-free compilation pass can ever show. The naming-confound control at least rules out one alternative explanation for the identical byte counts: it isn't that the two kernels merely happen to share incidental structure, since everything about them, including the function name itself, is held fixed except the one keyword argument in question.

## 23.4 ct.bitcast: Reinterpreting Bits Without Casting Values

### Intuition

`ct.bitcast(x, dtype)` reinterprets a tile's raw bits as a different dtype, without converting the underlying value the way `ct.astype` would — its own docstring example bitcasts the float `1.0` to `uint32` and prints `0x3f800000`, the literal IEEE-754 bit pattern for `1.0`, not some integer conversion of the number one. Since a bit-for-bit reinterpretation only makes sense between dtypes of the same width, testing a width-changing bitcast directly should show whether that requirement is enforced, or whether `bitcast` silently truncates or pads instead.

### Background

A `float32`-to-`uint32` bitcast (both 32 bits) and an `int32`-to-`uint32` bitcast (also both 32 bits, differing only in signedness) establish the ordinary same-width case. A `float32`-to-`uint8` bitcast (32 bits to 8 bits) tests the width-changing case directly. Finally, this section returns to a genuinely four-chapters-old open thread: Chapter 19 found that `bitwise_and`, `bitwise_or`, and `bitwise_xor` reject `float32` operands outright, with an error message recommending "an explicit `cuda.tile.bitcast()` for non-integer operands" — advice this book never actually tried. Bitcasting both operands to `uint32`, calling `bitwise_and`, and bitcasting the result back to `float32` tests whether that recommendation genuinely works.

### Worked Example 23.4.1 — same-width bitcasts, a width-changing rejection, and Chapter 19's escape hatch tested directly

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

sig_f32_to_u32 = ct.compilation.KernelSignature(
    [array_param(1, ct.float32), array_param(1, ct.uint32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.bitcast(x, dtype) reinterprets a tile's raw bits as a different
# dtype -- float32 and uint32 share the same 32-bit width, the
# documented use case.
@ct.kernel
def kernel_bitcast_f32_to_u32(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.bitcast(x, ct.uint32)
    ct.store(c, (0,), y)
print(f"bitcast float32 -> uint32 (same width): {compile_bytes(kernel_bitcast_f32_to_u32, sig_f32_to_u32)} cubin bytes")

# Does bitcast permit a WIDTH-CHANGING reinterpretation -- float32 (32
# bits) to uint8 (8 bits) -- or does it require equal bit widths,
# unlike pack_to_bytes which explicitly changes the element count?
sig_f32_to_u8 = ct.compilation.KernelSignature(
    [array_param(1, ct.float32), array_param(1, ct.uint8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_bitcast_f32_to_u8(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.bitcast(x, ct.uint8)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_bitcast_f32_to_u8, sig_f32_to_u8)
    print(f"bitcast float32 -> uint8 (width-changing): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitcast float32 -> uint8 (width-changing): {type(e).__name__}: {e}")

# bitcast between two same-width but differently-signed integer
# dtypes: int32 -> uint32.
sig_i32_to_u32 = ct.compilation.KernelSignature(
    [array_param(1, ct.int32), array_param(1, ct.uint32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_bitcast_i32_to_u32(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.bitcast(x, ct.uint32)
    ct.store(c, (0,), y)
print(f"bitcast int32 -> uint32 (same width): {compile_bytes(kernel_bitcast_i32_to_u32, sig_i32_to_u32)} cubin bytes")

# This is the very escape hatch Chapter 19's bitwise-rejection error
# messages kept recommending: "Use an explicit cuda.tile.bitcast() for
# non-integer operands." bitwise_and rejected float32/float32 outright
# in Chapter 19 -- does bitcasting both operands to uint32 first, then
# calling bitwise_and, actually work as the error message implied?
sig_bitwise_via_bitcast = ct.compilation.KernelSignature(
    [array_param(1, ct.float32), array_param(1, ct.float32), array_param(1, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_bitwise_and_via_bitcast(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.load(b, (0,), (8,))
    xi = ct.bitcast(x, ct.uint32)
    yi = ct.bitcast(y, ct.uint32)
    zi = ct.bitwise_and(xi, yi)
    z = ct.bitcast(zi, ct.float32)
    ct.store(c, (0,), z)
print(f"bitwise_and(float32, float32) via bitcast-to-uint32 round trip: {compile_bytes(kernel_bitwise_and_via_bitcast, sig_bitwise_via_bitcast)} cubin bytes")
```

Genuinely run:

```
bitcast float32 -> uint32 (same width): 20784 cubin bytes
bitcast float32 -> uint8 (width-changing): TileTypeError: Cannot bitcast from float32 to uint8: bit width is different (32 vs. 8)
  "/tmp/ch23/04_bitcast_width_and_bitwise_escape_hatch.py", line 42, col 9-31, in kernel_bitcast_f32_to_u8:
        y = ct.bitcast(x, ct.uint8)
            ^^^^^^^^^^^^^^^^^^^^^^^

bitcast int32 -> uint32 (same width): 20784 cubin bytes
bitwise_and(float32, float32) via bitcast-to-uint32 round trip: 23216 cubin bytes
```

### Discussion

The width-changing bitcast is rejected outright, and precisely: "bit width is different (32 vs. 8)," confirming `bitcast` is a strict equal-width reinterpretation, not a truncating or padding one — that job belongs to `pack_to_bytes` and `unpack_from_bytes`, tested next. The real payoff of this section is the last line: `bitwise_and(float32, float32)`, which Chapter 19 found was rejected outright with a `TileTypeError` recommending exactly this bitcast pattern, genuinely compiles once both operands are bitcast to `uint32` first and the result is bitcast back to `float32` afterward. This is the first time in this book that a piece of advice embedded in an error message has been tested end to end rather than just read — Chapter 19 could report what the error said, but had no operation available yet to check whether following that advice actually worked. It does.

## 23.5 ct.pack_to_bytes and ct.unpack_from_bytes: The Byte-Level Round Trip

### Intuition

Where `bitcast` reinterprets a tile in place at a fixed element count, `ct.pack_to_bytes(x)` flattens a tile and reinterprets its raw bytes as a 1D `uint8` tile of a *different* element count — a `(4,)` `int32` tile, 128 bits total, becomes 16 `uint8` elements. `ct.unpack_from_bytes(x, dtype)` is its documented inverse, requiring a 1D `uint8` input and dividing evenly into the target dtype's bit width. Both functions state their own divisibility rules explicitly in their docstrings, and both are worth testing at their boundaries: an input `unpack_from_bytes` can't reasonably accept (a 2D tile), an input that fails the divisibility rule cleanly, and — a genuine open question the docstrings never address — whether `pack_to_bytes` accepts a `bool_` tile at all, given that `bool_`'s own bit width in cuTile Python has never been established in this book.

### Background

A basic `pack_to_bytes`/`unpack_from_bytes` pair in opposite directions establishes ordinary use. A 2D input to `unpack_from_bytes` tests its stated 1D requirement. A `(2,)` `uint8` tile — 16 bits, which does not divide evenly into `int32`'s 32 bits — tests the divisibility rule with a genuinely power-of-two-shaped input, avoiding Chapter 15's separate power-of-two tile-shape rule from masking the result. A `bool_` tile tests `pack_to_bytes` at a boundary its docstring never mentions. A full pack/unpack round trip closes the section by confirming the two functions are structural inverses, the way Chapter 22's `extract`/`cat` were.

### Worked Example 23.5.1 — pack, unpack, two rejections, and a round trip

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ct.pack_to_bytes(x) flattens a tile and reinterprets its raw bytes
# as uint8 elements -- a (4,) int32 tile (128 bits total) becomes a
# 16-element uint8 tile.
sig_pack = ct.compilation.KernelSignature(
    [array_param(1, ct.int32), array_param(1, ct.uint8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_pack(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (4,))
    y = ct.pack_to_bytes(x)
    ct.store(c, (0,), y)
print(f"pack_to_bytes (4,) int32 -> uint8: {compile_bytes(kernel_pack, sig_pack)} cubin bytes")

# ct.unpack_from_bytes(x, dtype) is pack_to_bytes's documented inverse.
sig_unpack = ct.compilation.KernelSignature(
    [array_param(1, ct.uint8), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_unpack(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (16,))
    y = ct.unpack_from_bytes(x, ct.int32)
    ct.store(c, (0,), y)
print(f"unpack_from_bytes (16,) uint8 -> int32: {compile_bytes(kernel_unpack, sig_unpack)} cubin bytes")

# unpack_from_bytes requires a 1D input tile. Does a 2D uint8 tile
# get rejected?
sig_unpack_2d = ct.compilation.KernelSignature(
    [array_param(2, ct.uint8), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_unpack_2d(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (2, 8))
    y = ct.unpack_from_bytes(x, ct.int32)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_unpack_2d, sig_unpack_2d)
    print(f"unpack_from_bytes on 2D uint8 tile: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack_from_bytes on 2D uint8 tile: {type(e).__name__}: {e}")

# unpack_from_bytes's own divisibility rule: total bits must be
# divisible by the target dtype's bit width. A (2,) uint8 tile is 16
# bits total, which does not divide evenly into int32's 32-bit width.
sig_unpack_notdivisible = ct.compilation.KernelSignature(
    [array_param(1, ct.uint8), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_unpack_notdivisible(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (2,))
    y = ct.unpack_from_bytes(x, ct.int32)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_unpack_notdivisible, sig_unpack_notdivisible)
    print(f"unpack_from_bytes (2,) uint8 [16 bits] -> int32 [32 bits]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack_from_bytes (2,) uint8 [16 bits] -> int32 [32 bits]: {type(e).__name__}: {e}")

# pack_to_bytes's own stated rule is that total bits must be divisible
# by 8 -- but bool_ is rejected outright, before that rule even comes
# into play.
sig_pack_bool = ct.compilation.KernelSignature(
    [array_param(1, ct.bool_), array_param(1, ct.uint8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_pack_bool(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (4,))
    y = ct.pack_to_bytes(x)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_pack_bool, sig_pack_bool)
    print(f"pack_to_bytes (4,) bool_ -> uint8: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"pack_to_bytes (4,) bool_ -> uint8: {type(e).__name__}: {e}")

# Round trip: pack an int32 tile to bytes, then unpack it straight
# back to int32, confirming pack_to_bytes and unpack_from_bytes are
# genuine structural inverses, the way Chapter 22's extract/cat were.
sig_roundtrip = ct.compilation.KernelSignature(
    [array_param(1, ct.int32), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_pack_unpack_roundtrip(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (4,))
    packed = ct.pack_to_bytes(x)
    y = ct.unpack_from_bytes(packed, ct.int32)
    ct.store(c, (0,), y)
print(f"pack_to_bytes/unpack_from_bytes roundtrip (4,) int32: {compile_bytes(kernel_pack_unpack_roundtrip, sig_roundtrip)} cubin bytes")
```

Genuinely run:

```
pack_to_bytes (4,) int32 -> uint8: 21152 cubin bytes
unpack_from_bytes (16,) uint8 -> int32: 21536 cubin bytes
unpack_from_bytes on 2D uint8 tile: TileTypeError: unpack_from_bytes requires a 1D tile, got 2D tile with shape (2, 8)
  "/tmp/ch23/05_pack_to_bytes_and_unpack_from_bytes.py", line 53, col 9-41, in kernel_unpack_2d:
        y = ct.unpack_from_bytes(x, ct.int32)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

unpack_from_bytes (2,) uint8 [16 bits] -> int32 [32 bits]: TileTypeError: Cannot unpack tile Tile[uint8,(2)] to int32: total bits (2 * 8) not divisible by 32
  "/tmp/ch23/05_pack_to_bytes_and_unpack_from_bytes.py", line 72, col 9-41, in kernel_unpack_notdivisible:
        y = ct.unpack_from_bytes(x, ct.int32)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pack_to_bytes (4,) bool_ -> uint8: TileTypeError: pack_to_bytes from a bool_ tile is not supported
  "/tmp/ch23/05_pack_to_bytes_and_unpack_from_bytes.py", line 91, col 9-27, in kernel_pack_bool:
        y = ct.pack_to_bytes(x)
            ^^^^^^^^^^^^^^^^^^^

pack_to_bytes/unpack_from_bytes roundtrip (4,) int32: 20784 cubin bytes
```

### Discussion

Every prediction holds, and the `bool_` case answers the question this section opened with directly: `pack_to_bytes` does not attempt to reason about how many bits a `bool_` element occupies at all — it refuses the dtype outright, "not supported," rather than picking some assumed width (1 bit, 8 bits, or anything else) and possibly getting it wrong. The divisibility error spells out the arithmetic in the message itself — "total bits (2 * 8) not divisible by 32" — continuing this book's pattern of error text that shows its work. The round trip's byte count, 20784, is worth comparing to Section 23.4's `bitcast` round trip figures (also 20784, for both the `float32`-to-`uint32` and `int32`-to-`uint32` bitcasts) — though without a naming-confound control tying these three kernels together, that match is suggestive rather than conclusive, a reminder from Chapter 10 not to read too much into an unconfounded coincidence in byte counts.

## 23.6 Capstone: Advanced Indexing Meets the Byte-Level View

### Intuition

This chapter's two halves — sparse/dense addressing and byte-level reinterpretation — have nothing to do with each other on the surface, but nothing stops them from appearing in the same kernel. A capstone that gathers rows with `load_advanced_indexing`, routes the result through a `pack_to_bytes`/`unpack_from_bytes` round trip, and writes it back out with `store_advanced_indexing` at a different set of rows exercises every operation this chapter introduced in one pipeline.

### Background

Four source rows are gathered via `load_advanced_indexing`, packed to raw bytes, unpacked straight back to the original dtype, reshaped back to two dimensions (since `unpack_from_bytes` only produces 1D tiles), and written to four different destination rows via `store_advanced_indexing`. A baseline version without the byte-level detour compiles the same gather-then-scatter shape for comparison.

### Worked Example 23.6.1 — a full pipeline combining both halves of the chapter

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

# Capstone: load a non-contiguous row-gathered tile with
# load_advanced_indexing, pack it to its raw bytes, unpack the bytes
# straight back to the original dtype, and write the result back with
# store_advanced_indexing at a *different* set of sparse rows -- one
# kernel using every operation from both halves of this chapter.
@ct.kernel
def kernel_capstone(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    src_rows = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (src_rows, ct.Slice(0, 4)))
    packed = ct.pack_to_bytes(x)
    unpacked = ct.unpack_from_bytes(packed, ct.int32)
    y = ct.reshape(unpacked, (4, 4))
    dst_rows = ct.arange(4, dtype=ct.int32) + 4
    ct.store_advanced_indexing(b, (dst_rows, ct.Slice(0, 4)), y)
print(f"capstone: load_advanced_indexing -> pack -> unpack -> store_advanced_indexing: {compile_bytes(kernel_capstone, sig)} cubin bytes")

# Baseline: the same round trip without the pack/unpack detour, to
# see how much the byte-level detour actually costs.
@ct.kernel
def kernel_capstone_direct(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    src_rows = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (src_rows, ct.Slice(0, 4)))
    dst_rows = ct.arange(4, dtype=ct.int32) + 4
    ct.store_advanced_indexing(b, (dst_rows, ct.Slice(0, 4)), x)
print(f"capstone baseline: load_advanced_indexing -> store_advanced_indexing, no byte detour: {compile_bytes(kernel_capstone_direct, sig)} cubin bytes")
```

Genuinely run:

```
capstone: load_advanced_indexing -> pack -> unpack -> store_advanced_indexing: 22448 cubin bytes
capstone baseline: load_advanced_indexing -> store_advanced_indexing, no byte detour: 22592 cubin bytes
```

### Discussion

Both kernels compile cleanly, confirming `load_advanced_indexing`, `pack_to_bytes`, `unpack_from_bytes`, `ct.reshape`, and `store_advanced_indexing` compose without friction in a single kernel body. The byte counts are worth reading carefully rather than at face value: the pipeline with the pack/unpack detour compiles slightly *smaller* (22448 bytes) than the baseline without it (22592 bytes), despite doing strictly more in the source code. This is not a naming-confound-controlled comparison in Chapter 10's sense — the two kernels have genuinely different bodies, and the baseline lacks the `ct.reshape` call the byte-level detour requires (since `unpack_from_bytes` only ever produces a 1D tile), so the two are not identical up to one substitution the way this book's controlled comparisons have been. The honest conclusion is narrower than "pack/unpack is free": this book's `export_kernel`-only environment can say the composition compiles, and can report the two numbers, but cannot respect a reading that treats an uncontrolled byte-count difference as a claim about relative cost, which is exactly the discipline Chapter 10 established this comparison would need to earn.

## Chapter Summary

This chapter introduced a third memory-addressing convention alongside this book's ordinary `ct.load`/`ct.store` and Chapter 20's `gather()`/`scatter()`: `ct.load_advanced_indexing` and `ct.store_advanced_indexing`, which require exactly one dimension addressed by a runtime 1D integer tile (the sparse dim) and every other dimension addressed by a compile-time-sized contiguous `ct.Slice` (a dense dim). Both functions share a single validator, confirmed directly by error messages that name both functions regardless of which one was called, and both enforce `ct.Slice`'s power-of-two length requirement and a store-shape check consistent with this book's discipline since Chapter 3. Testing `padding_mode` found no byte-level difference between `UNDETERMINED` and `ZERO` for a load this book's `export_kernel`-only environment cannot prove is ever actually out of bounds — the same epistemic wall Chapter 8 hit with `RoundingMode`. The chapter's second half moved to raw bytes: `ct.bitcast` reinterprets a tile's bits as an equal-width dtype, rejecting any width-changing attempt outright, and — tested directly for the first time — genuinely enables Chapter 19's recommended workaround, letting `bitwise_and` operate on `float32` data via a `uint32` round trip. `ct.pack_to_bytes` and `ct.unpack_from_bytes` extend that byte-level view to a width-changing, structurally-inverse pair, with their own independent validation: a rejected `bool_` input, a rejected 2D input, and a divisibility check distinct from `bitcast`'s equal-width rule. The capstone combined both halves in a single kernel, confirming they compose freely, while being explicit about which of its byte-count comparisons Chapter 10's naming-confound discipline actually licenses.

## Self-Check Questions

1. Section 23.1 found that both `load_advanced_indexing` and `store_advanced_indexing` produce error messages beginning with the identical phrase "load_advanced_indexing/store_advanced_indexing," regardless of which function was actually called. What does this suggest about how the two functions are implemented, and how is this similar to or different from Section 23.4's finding that `bitcast`'s width-check error names the specific two dtypes and bit widths involved?
2. Section 23.3 found that `padding_mode=UNDETERMINED` and `padding_mode=ZERO` compile to byte-identical cubins for a specific load, and connected this to Chapter 8's identical finding about `RoundingMode`. What kind of test — one this book's `export_kernel`-only environment could never perform — would be needed to determine whether the two padding modes actually produce different results at runtime?
3. Section 23.4 confirmed that Chapter 19's recommended `bitcast` workaround for `bitwise_and(float32, float32)` genuinely compiles. Chapter 19 itself never attempted this, reporting only what the error message said. What does the fact that this took until Chapter 23 to test suggest about the difference between an error message quoting an operation by name and a book actually verifying that operation works?
4. Section 23.5 found that `pack_to_bytes` rejects `bool_` tiles outright, rather than picking an assumed bit width (1 bit, 8 bits, or something else) and proceeding. Given what Section 23.5 also found about `unpack_from_bytes`'s explicit divisibility check ("total bits ... not divisible by 32"), why might a library author choose to hard-reject an ambiguous case like `bool_`'s bit width rather than picking a documented default and letting the divisibility check catch any resulting problems downstream?
5. Section 23.6's capstone found the pack/unpack-detour kernel compiling to fewer bytes than the simpler baseline kernel, and the Discussion explicitly declined to conclude that the detour is "free" as a result. Restate, in your own words, exactly what would have needed to be true about the two kernels for that byte-count comparison to license a conclusion about relative cost, drawing on Chapter 10's naming-confound rule.

## Where We Go Next

This chapter closed the `dir(ct)` items Chapter 21 first flagged as untouched — advanced indexing and the byte-level view are now both explored, alongside a four-chapter-old thread from Chapter 19 finally tested to completion. What remains largely unexplored is a cluster of operations this book has referenced only by name so far: `ct.reduce` and `ct.scan` accept an arbitrary caller-supplied reduction or prefix-scan function rather than a fixed operation, a generality this book's `ct.sum`/`ct.max`/`ct.cumsum` examples never needed; `ct.argmax` and `ct.argmin` return positions rather than values, a different kind of reduction than any this book has covered; and `ct.static_assert`, `ct.static_eval`, and `ct.static_iter` point toward compile-time metaprogramming — code that runs while a kernel is being compiled rather than while it executes — a layer beneath everything this book has written so far, which has always treated a kernel's Python source as something compiled once and never inspected by itself.

## Worked Solutions

1. Both functions reporting themselves under the same combined name — "load_advanced_indexing/store_advanced_indexing" — strongly suggests one shared internal validation routine handles the `indices` convention for both, and that routine was written to identify itself by both public names rather than by whichever one the caller actually used, presumably because the validation logic (counting sparse dims, checking tuple length) is identical for both directions of the operation. This is a coarser-grained disclosure than Section 23.4's `bitcast` error, which names the exact two dtypes and their exact two bit widths involved in that specific call — a message generated fresh from the actual arguments passed, rather than a fixed string shared across two related but distinct entry points. Both are still informative, but at different levels: one confirms an implementation-sharing fact about the library, the other confirms a fact about the specific call that failed.

2. Determining whether `padding_mode=UNDETERMINED` and `padding_mode=ZERO` produce different runtime behavior would require actually executing a kernel using each mode on real GPU hardware — via `ct.launch()`, with a real CUDA stream, real device memory, and an input array where the load provably reads out of bounds — and then comparing the two runs' output arrays element by element to see whether the out-of-bounds positions differ. That is precisely the one class of test this book's `export_kernel`-only, driver-free compilation workflow has never been able to perform anywhere in twenty-three chapters: everything this book confirms is about whether a kernel compiles and what its compiled size is, never about what values come out the other end of an actual execution.

3. The gap between Chapter 19 and Chapter 23 suggests that quoting an operation's name inside an error message is a much cheaper claim for a library to make than a book actually confirming that operation behaves as claimed — the error message costs the library authors nothing beyond writing the string, while testing it required this book to wait until `bitcast` itself became the chapter's own subject, with a worked example built specifically to combine it with `bitwise_and` in the way the error message implied. It is a useful reminder that this book's own standing rule — never let a claim into the text without genuinely running the code — applies just as much to a third party's advice embedded in an error string as to a documented rule in a docstring; both are claims made by the library about itself, and both needed independent verification before this book was willing to report them as fact.

4. A hard rejection for `bool_` avoids committing to a bit-width assumption that later code, or later versions of the library, might quietly get wrong or need to change — if `pack_to_bytes` had picked, say, 1 bit per `bool_` element and proceeded, any caller relying on that assumption would silently break if a future version of cuTile Python changed how `bool_` is represented in memory. The divisibility check on `unpack_from_bytes`, by contrast, is a check on a genuinely well-defined property — bit widths of concrete numeric dtypes are unambiguous and unlikely to change — so letting the arithmetic catch a downstream problem (an odd byte count relative to a target dtype) is safe precisely because the two numbers going into that arithmetic were never in question. `bool_`'s bit-width ambiguity is a different kind of problem, one arithmetic checking downstream can't catch because the ambiguity is upstream, in what the input's own size should even be counted as.

5. For the byte-count comparison to license a conclusion about relative cost, the two kernels would need to be identical in every respect except the presence of the `pack_to_bytes`/`unpack_from_bytes` detour itself — Chapter 10's naming-confound rule specifically requires the same function name so the compiled symbol name contributes nothing to the difference, and by extension the same everything else: same argument order, same intervening operations, same final store call. The capstone's baseline kernel instead omits the `ct.reshape` call entirely, because `unpack_from_bytes` only ever returns a 1D tile and the destination store expects two dimensions — meaning the two kernels differ by more than just "has the detour or doesn't," and the smaller byte count in the detour version could be explained by any number of confounded factors, not necessarily anything about the cost of `pack_to_bytes`/`unpack_from_bytes` themselves. Without controlling for that difference the way Section 23.3's `padding_mode` comparison controlled for the function name, the honest reading of the numbers is simply that both kernels compiled, with no further claim available about which one is "more expensive."

## Complete Runnable Code

### File: `01_load_advanced_indexing_basics_and_validation.py`

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

# ct.load_advanced_indexing(array, indices) reads a tile from
# non-contiguous slices: exactly one entry in indices must be a 1D
# integer tile (the sparse dim), and the rest must be ct.Slice objects
# (dense dims). Basic 2D case: sparse dim 0, dense dim 1.
@ct.kernel
def kernel_load_basic(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(2, 4)))
    ct.store(c, (0, 0), x)
print(f"load_advanced_indexing sparse=dim0, dense=dim1: {compile_bytes(kernel_load_basic, sig2)} cubin bytes")

# Sparse dim in the other position: sparse dim 1, dense dim 0.
@ct.kernel
def kernel_load_basic_swapped(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    col_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (ct.Slice(2, 4), col_indices))
    ct.store(c, (0, 0), x)
print(f"load_advanced_indexing sparse=dim1, dense=dim0: {compile_bytes(kernel_load_basic_swapped, sig2)} cubin bytes")

# indices tuple shorter than array.ndim.
@ct.kernel
def kernel_load_short_indices(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices,))
    ct.store(c, (0, 0), x)
try:
    n = compile_bytes(kernel_load_short_indices, sig2)
    print(f"load_advanced_indexing indices len=1 (array ndim=2): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing indices len=1 (array ndim=2): {type(e).__name__}: {e}")

# Zero sparse dims: both entries are Slice objects, no 1D tile at all.
@ct.kernel
def kernel_load_no_sparse(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load_advanced_indexing(a, (ct.Slice(0, 4), ct.Slice(0, 4)))
    ct.store(c, (0, 0), x)
try:
    n = compile_bytes(kernel_load_no_sparse, sig2)
    print(f"load_advanced_indexing zero sparse dims (both Slice): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing zero sparse dims (both Slice): {type(e).__name__}: {e}")

# Two sparse dims: both entries are 1D tiles.
@ct.kernel
def kernel_load_two_sparse(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    col_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, col_indices))
    ct.store(c, (0, 0), x)
try:
    n = compile_bytes(kernel_load_two_sparse, sig2)
    print(f"load_advanced_indexing two sparse dims (both Tile): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing two sparse dims (both Tile): {type(e).__name__}: {e}")
```

### File: `02_store_advanced_indexing_and_shape_matching.py`

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
    [array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.store_advanced_indexing(array, indices, tile) uses the same
# convention in reverse: one sparse dim, the rest dense Slices, and
# the stored tile's shape must exactly match the shape the indices imply.
@ct.kernel
def kernel_store_basic(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32) + 1
    t = ct.full((4, 4), 1, dtype=a.dtype)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), t)
print(f"store_advanced_indexing matching shape (4,4): {compile_bytes(kernel_store_basic, sig2)} cubin bytes")

# A tile whose shape does not match what the indices imply: indices
# imply (4, 4), but the tile being stored is (4, 8).
@ct.kernel
def kernel_store_shape_mismatch(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32) + 1
    t = ct.full((4, 8), 1, dtype=a.dtype)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), t)
try:
    n = compile_bytes(kernel_store_shape_mismatch, sig2)
    print(f"store_advanced_indexing mismatched tile shape (4,8) vs implied (4,4): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"store_advanced_indexing mismatched tile shape (4,8) vs implied (4,4): {type(e).__name__}: {e}")

# ct.Slice's own docstring says length "must be a power of two." A
# Slice with a non-power-of-two length should be rejected -- the same
# validation load_advanced_indexing and store_advanced_indexing share.
@ct.kernel
def kernel_load_slice_nonpow2(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(0, 3)))
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 3)), x)
try:
    n = compile_bytes(kernel_load_slice_nonpow2, sig2)
    print(f"load_advanced_indexing Slice(0, 3) non-power-of-two length: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"load_advanced_indexing Slice(0, 3) non-power-of-two length: {type(e).__name__}: {e}")
```

### File: `03_padding_mode_naming_controlled.py`

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
    [array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# padding_mode controls the fill value for out-of-bounds elements on
# both the sparse and dense dims. Does varying it change the compiled
# byte count for a load that a compiler could, in principle, prove is
# always in-bounds? Naming-confound-controlled (Chapter 10): both
# variants reuse the identical function name "kernel_fn".
@ct.kernel
def kernel_fn(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), padding_mode=ct.PaddingMode.UNDETERMINED)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), x)
n_undetermined = compile_bytes(kernel_fn, sig2)
print(f"load_advanced_indexing padding_mode=UNDETERMINED, naming-controlled: {n_undetermined} cubin bytes")

@ct.kernel
def kernel_fn(a, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row_indices = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), padding_mode=ct.PaddingMode.ZERO)
    ct.store_advanced_indexing(a, (row_indices, ct.Slice(0, 4)), x)
n_zero = compile_bytes(kernel_fn, sig2)
print(f"load_advanced_indexing padding_mode=ZERO, naming-controlled: {n_zero} cubin bytes")
print(f"identical bytes: {n_undetermined == n_zero}")
```

### File: `04_bitcast_width_and_bitwise_escape_hatch.py`

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

sig_f32_to_u32 = ct.compilation.KernelSignature(
    [array_param(1, ct.float32), array_param(1, ct.uint32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.bitcast(x, dtype) reinterprets a tile's raw bits as a different
# dtype -- float32 and uint32 share the same 32-bit width, the
# documented use case.
@ct.kernel
def kernel_bitcast_f32_to_u32(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.bitcast(x, ct.uint32)
    ct.store(c, (0,), y)
print(f"bitcast float32 -> uint32 (same width): {compile_bytes(kernel_bitcast_f32_to_u32, sig_f32_to_u32)} cubin bytes")

# Does bitcast permit a WIDTH-CHANGING reinterpretation -- float32 (32
# bits) to uint8 (8 bits) -- or does it require equal bit widths,
# unlike pack_to_bytes which explicitly changes the element count?
sig_f32_to_u8 = ct.compilation.KernelSignature(
    [array_param(1, ct.float32), array_param(1, ct.uint8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_bitcast_f32_to_u8(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.bitcast(x, ct.uint8)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_bitcast_f32_to_u8, sig_f32_to_u8)
    print(f"bitcast float32 -> uint8 (width-changing): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitcast float32 -> uint8 (width-changing): {type(e).__name__}: {e}")

# bitcast between two same-width but differently-signed integer
# dtypes: int32 -> uint32.
sig_i32_to_u32 = ct.compilation.KernelSignature(
    [array_param(1, ct.int32), array_param(1, ct.uint32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_bitcast_i32_to_u32(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.bitcast(x, ct.uint32)
    ct.store(c, (0,), y)
print(f"bitcast int32 -> uint32 (same width): {compile_bytes(kernel_bitcast_i32_to_u32, sig_i32_to_u32)} cubin bytes")

# This is the very escape hatch Chapter 19's bitwise-rejection error
# messages kept recommending: "Use an explicit cuda.tile.bitcast() for
# non-integer operands." bitwise_and rejected float32/float32 outright
# in Chapter 19 -- does bitcasting both operands to uint32 first, then
# calling bitwise_and, actually work as the error message implied?
sig_bitwise_via_bitcast = ct.compilation.KernelSignature(
    [array_param(1, ct.float32), array_param(1, ct.float32), array_param(1, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_bitwise_and_via_bitcast(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    y = ct.load(b, (0,), (8,))
    xi = ct.bitcast(x, ct.uint32)
    yi = ct.bitcast(y, ct.uint32)
    zi = ct.bitwise_and(xi, yi)
    z = ct.bitcast(zi, ct.float32)
    ct.store(c, (0,), z)
print(f"bitwise_and(float32, float32) via bitcast-to-uint32 round trip: {compile_bytes(kernel_bitwise_and_via_bitcast, sig_bitwise_via_bitcast)} cubin bytes")
```

### File: `05_pack_to_bytes_and_unpack_from_bytes.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ct.pack_to_bytes(x) flattens a tile and reinterprets its raw bytes
# as uint8 elements -- a (4,) int32 tile (128 bits total) becomes a
# 16-element uint8 tile.
sig_pack = ct.compilation.KernelSignature(
    [array_param(1, ct.int32), array_param(1, ct.uint8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_pack(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (4,))
    y = ct.pack_to_bytes(x)
    ct.store(c, (0,), y)
print(f"pack_to_bytes (4,) int32 -> uint8: {compile_bytes(kernel_pack, sig_pack)} cubin bytes")

# ct.unpack_from_bytes(x, dtype) is pack_to_bytes's documented inverse.
sig_unpack = ct.compilation.KernelSignature(
    [array_param(1, ct.uint8), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_unpack(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (16,))
    y = ct.unpack_from_bytes(x, ct.int32)
    ct.store(c, (0,), y)
print(f"unpack_from_bytes (16,) uint8 -> int32: {compile_bytes(kernel_unpack, sig_unpack)} cubin bytes")

# unpack_from_bytes requires a 1D input tile. Does a 2D uint8 tile
# get rejected?
sig_unpack_2d = ct.compilation.KernelSignature(
    [array_param(2, ct.uint8), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_unpack_2d(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (2, 8))
    y = ct.unpack_from_bytes(x, ct.int32)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_unpack_2d, sig_unpack_2d)
    print(f"unpack_from_bytes on 2D uint8 tile: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack_from_bytes on 2D uint8 tile: {type(e).__name__}: {e}")

# unpack_from_bytes's own divisibility rule: total bits must be
# divisible by the target dtype's bit width. A (2,) uint8 tile is 16
# bits total, which does not divide evenly into int32's 32-bit width.
sig_unpack_notdivisible = ct.compilation.KernelSignature(
    [array_param(1, ct.uint8), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_unpack_notdivisible(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (2,))
    y = ct.unpack_from_bytes(x, ct.int32)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_unpack_notdivisible, sig_unpack_notdivisible)
    print(f"unpack_from_bytes (2,) uint8 [16 bits] -> int32 [32 bits]: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack_from_bytes (2,) uint8 [16 bits] -> int32 [32 bits]: {type(e).__name__}: {e}")

# pack_to_bytes's own stated rule is that total bits must be divisible
# by 8 -- but bool_ is rejected outright, before that rule even comes
# into play.
sig_pack_bool = ct.compilation.KernelSignature(
    [array_param(1, ct.bool_), array_param(1, ct.uint8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_pack_bool(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (4,))
    y = ct.pack_to_bytes(x)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_pack_bool, sig_pack_bool)
    print(f"pack_to_bytes (4,) bool_ -> uint8: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"pack_to_bytes (4,) bool_ -> uint8: {type(e).__name__}: {e}")

# Round trip: pack an int32 tile to bytes, then unpack it straight
# back to int32, confirming pack_to_bytes and unpack_from_bytes are
# genuine structural inverses, the way Chapter 22's extract/cat were.
sig_roundtrip = ct.compilation.KernelSignature(
    [array_param(1, ct.int32), array_param(1, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_pack_unpack_roundtrip(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (4,))
    packed = ct.pack_to_bytes(x)
    y = ct.unpack_from_bytes(packed, ct.int32)
    ct.store(c, (0,), y)
print(f"pack_to_bytes/unpack_from_bytes roundtrip (4,) int32: {compile_bytes(kernel_pack_unpack_roundtrip, sig_roundtrip)} cubin bytes")
```

### File: `06_capstone_advanced_indexing_meets_byte_reinterpretation.py`

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

# Capstone: load a non-contiguous row-gathered tile with
# load_advanced_indexing, pack it to its raw bytes, unpack the bytes
# straight back to the original dtype, and write the result back with
# store_advanced_indexing at a *different* set of sparse rows -- one
# kernel using every operation from both halves of this chapter.
@ct.kernel
def kernel_capstone(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    src_rows = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (src_rows, ct.Slice(0, 4)))
    packed = ct.pack_to_bytes(x)
    unpacked = ct.unpack_from_bytes(packed, ct.int32)
    y = ct.reshape(unpacked, (4, 4))
    dst_rows = ct.arange(4, dtype=ct.int32) + 4
    ct.store_advanced_indexing(b, (dst_rows, ct.Slice(0, 4)), y)
print(f"capstone: load_advanced_indexing -> pack -> unpack -> store_advanced_indexing: {compile_bytes(kernel_capstone, sig)} cubin bytes")

# Baseline: the same round trip without the pack/unpack detour, to
# see how much the byte-level detour actually costs.
@ct.kernel
def kernel_capstone_direct(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    src_rows = ct.arange(4, dtype=ct.int32)
    x = ct.load_advanced_indexing(a, (src_rows, ct.Slice(0, 4)))
    dst_rows = ct.arange(4, dtype=ct.int32) + 4
    ct.store_advanced_indexing(b, (dst_rows, ct.Slice(0, 4)), x)
print(f"capstone baseline: load_advanced_indexing -> store_advanced_indexing, no byte detour: {compile_bytes(kernel_capstone_direct, sig)} cubin bytes")
```
