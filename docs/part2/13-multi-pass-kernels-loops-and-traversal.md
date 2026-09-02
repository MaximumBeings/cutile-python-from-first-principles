# Chapter 13: Multi-Pass Kernels — Loops and Traversal

> "Chapter 11's loops were all Python-trace-time constructs: a fixed, compile-time-known trip count that the tracer simply followed and unrolled, one call per iteration. Every kernel this book has compiled has processed exactly one tile per array, known in full before compilation ever started. Real programs are rarely that simple — a row of data can be any length, known only once a real array actually arrives. This chapter finds cuTile Python's other kind of loop: one whose trip count is a genuine runtime value, compiled as a real loop rather than unrolled, and combines it with a second real primitive, tiled overlap, into a kernel that processes its data in two genuinely separate passes."

**What you will understand by the end of this chapter:**

- That `for k in range(view.num_tiles(axis))` compiles to a genuine, native loop with a runtime trip count — categorically different from Chapter 11's Python-trace-time-unrolled loops, since `num_tiles` depends on the array's real runtime shape and has no concrete Python value to unroll against
- That a real, compiled loop is measurably more expensive than fully unrolling the same number of iterations at compile time — confirmed by comparing genuinely equivalent work compiled both ways
- That the dynamic loop's compiled size varies with `tile_size` non-monotonically, the same "don't assume a gradient" caution this book has applied to other hints, while the unrolled loop's size grows monotonically with its own trip count, a genuinely different shape of relationship
- That `Array.tiled_view`'s `traversal_steps` parameter genuinely produces overlapping or gapped tile traversal, is enforced against its own rank and positivity requirements with real, distinct `TileTypeError` messages, and produces measurably different compiled code only at its most extreme tested value
- How to combine a runtime-trip-count loop with Chapter 8's masking technique into one kernel that processes the same data in two genuinely separate passes

**What you need to know first:**

- Chapter 8's masking technique — computing per-element offsets and comparing them against `Array.shape` to build a boundary mask with `ct.where`.
- Chapter 11's Python-trace-time loop unrolling, which this chapter's dynamic loop is deliberately contrasted against.
- Chapter 7's and Chapter 10's established caution that a hint's effect on compiled size is often a boundary step, not a smooth gradient.

## 13.1 A Second Kind of Loop: A Genuine Runtime Trip Count `[FOUNDATIONAL]`

### Intuition

`TiledView.num_tiles(axis)` is documented to return a genuine `int32` — Chapter 8 already established that `Array.shape` is a real, runtime-accessible property, and `num_tiles` is computed from it. A Python `for` loop over a value with no concrete number until runtime cannot be unrolled by the tracer the way Chapter 11's loops were. This section checks directly whether the compiler instead handles it as a genuine, native loop.

### Background

Nothing about `view.num_tiles(0)` is knowable when `export_kernel` traces the kernel body — the actual array passed at launch determines it, and this book's `export_kernel`-only environment never launches anything. If this compiles at all, it can only be because the compiler is generating real loop control code that evaluates the trip count on the device, not because the tracer resolved a concrete number in Python.

### Worked Example 13.1.1 — a data-dependent loop, compiled with no data at all

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# view.num_tiles(0) is a genuine, runtime int32 value -- it depends on the
# array's real shape (a Chapter 8 finding), which is not known at compile
# time. "for k in range(n)" with n a runtime value cannot be unrolled by the
# tracer the way Chapter 11's "for _ in range(n)" was, since n has no
# concrete Python value to unroll against. This checks whether the compiler
# instead compiles this into a genuine, native loop.
@ct.kernel
def row_sum(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(row_sum, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"data-dependent 'for k in range(view.num_tiles(0))' loop: {len(buf.getvalue())} cubin bytes")
except Exception as e:
    print(f"data-dependent loop: {type(e).__name__}: {e}")
```

The complete file is `01_dynamic_trip_count_loop.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_dynamic_trip_count_loop.py
```

Genuinely run:

```
data-dependent 'for k in range(view.num_tiles(0))' loop: 67968 cubin bytes
```

This compiles successfully with no array ever having existed, confirming the compiler genuinely supports a loop whose trip count is resolved on the device at runtime, not by the Python tracer ahead of time. Section 13.2 checks what that costs compared to unrolling the equivalent work.

## 13.2 The Real Cost of a Real Loop `[FOUNDATIONAL]`

### Intuition

Section 13.1 confirmed a genuine compiled loop is possible. It said nothing about whether that loop is cheaper, more expensive, or about the same as unrolling the identical number of iterations the way Chapter 11 did. This section compiles both shapes doing the same real work and compares them directly.

### Background

A fully unrolled loop lets the compiler see every iteration's concrete tile index as a literal constant, opening the door to per-iteration optimization the compiler cannot apply the same way across a genuine, data-dependent loop body that has to work for any trip count. A real loop, in exchange, needs actual loop-control instructions: a counter, a comparison, and a branch, evaluated on the device every iteration.

### Worked Example 13.2.1 — the same four tiles, summed two different ways

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig_dynamic = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_unrolled = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64), ct.compilation.ConstantConstraint(4)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def row_sum_dynamic(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

@ct.kernel
def row_sum_unrolled(a, c, tile_size: ct.Constant[int], n: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Both kernels do the same real work for an array whose row happens to be
# exactly 4 tiles wide: sum 4 tiles of the same tile_size. row_sum_dynamic
# does not know that "4" at compile time; row_sum_unrolled is handed it
# directly as a ct.Constant[int], the way Chapter 11 built its unrolled
# loops.
print(f"genuinely compiled loop (trip count from view.num_tiles(0)): {compile_bytes(row_sum_dynamic, sig_dynamic)} cubin bytes")
print(f"Python-unrolled loop (trip count 4, a ct.Constant[int]): {compile_bytes(row_sum_unrolled, sig_unrolled)} cubin bytes")
```

The complete file is `02_dynamic_vs_unrolled_same_work.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_dynamic_vs_unrolled_same_work.py
```

Genuinely run:

```
genuinely compiled loop (trip count from view.num_tiles(0)): 68224 cubin bytes
Python-unrolled loop (trip count 4, a ct.Constant[int]): 25344 cubin bytes
```

> `[COMMON TRAP]` For the identical real work — four tiles, summed — the genuinely compiled loop is nearly three times larger than the fully unrolled version. This is not a small effect, and it runs in the direction that is easy to forget when a data-dependent trip count feels like it should be the more "efficient" choice for handling variable-length data: the flexibility of not needing to know the trip count at compile time is bought with real, substantial compiled-code cost. (These two numbers are not comparable to Section 13.1's 67,968-byte figure — Chapter 1's warning that byte counts are sensitive to exact source text applies here too, since `row_sum_dynamic` and `row_sum` are different function names in different files.)

## 13.3 Two Different Shapes of Growth `[FOUNDATIONAL]`

### Intuition

Section 13.2 compared one specific case. Whether the dynamic loop's size depends on `tile_size` the way earlier hints have shown non-monotonic boundary effects, and whether the unrolled loop's size grows smoothly with its own trip count the way Chapter 11's plain repeated calls did not, are both genuinely open, separate questions.

### Background

The dynamic loop's own trip count has no compile-time representation at all — the only remaining compile-time knob affecting it is `tile_size`, the shape of each individual tile. The unrolled loop, by contrast, has a real compile-time trip count `n`, directly settable and sweepable.

### Worked Example 13.3.1 — the dynamic loop, swept by `tile_size`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

@ct.kernel
def row_sum_dynamic(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

# tile_size is the one real compile-time knob left in the dynamic-loop
# kernel -- the loop's own trip count is a genuine runtime value with no
# compile-time equivalent to sweep.
for tile_size in [32, 64, 128]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(tile_size)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print(f"dynamic-loop kernel, tile_size={tile_size}: {compile_bytes(row_sum_dynamic, sig)} cubin bytes")
```

The complete file is `03_dynamic_loop_tile_size_scaling.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_dynamic_loop_tile_size_scaling.py
```

Genuinely run:

```
dynamic-loop kernel, tile_size=32: 66336 cubin bytes
dynamic-loop kernel, tile_size=64: 68224 cubin bytes
dynamic-loop kernel, tile_size=128: 66560 cubin bytes
```

### Worked Example 13.3.2 — the unrolled loop, swept by trip count

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

@ct.kernel
def row_sum_unrolled(a, c, tile_size: ct.Constant[int], n: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

for n in [1, 2, 4, 8]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(64), ct.compilation.ConstantConstraint(n)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print(f"Python-unrolled loop, n={n}: {compile_bytes(row_sum_unrolled, sig)} cubin bytes")
```

The complete file is `04_unrolled_loop_n_scaling.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_unrolled_loop_n_scaling.py
```

Genuinely run:

```
Python-unrolled loop, n=1: 22912 cubin bytes
Python-unrolled loop, n=2: 23680 cubin bytes
Python-unrolled loop, n=4: 25344 cubin bytes
Python-unrolled loop, n=8: 28416 cubin bytes
```

Two genuinely different shapes of growth, confirmed directly. The dynamic loop's size across `tile_size=32, 64, 128` is 66,336 / 68,224 / 66,560 bytes — not monotonic, the same "boundary effect, not a gradient" caution this book applied to `occupancy` in Chapter 10. The unrolled loop's size across `n=1, 2, 4, 8`, by contrast, grows monotonically every single step — 22,912 / 23,680 / 25,344 / 28,416 bytes — the more intuitive "more unrolled work, more code" relationship, in clean contrast to Chapter 11's non-monotonic repeated-call result for a much smaller helper function. This book has now measured both a hint whose relationship to size is a step, a helper-call count whose relationship is genuinely non-monotonic, and an unrolled loop trip count whose relationship is cleanly monotonic — three different shapes, reported exactly as each one measured.

## 13.4 `traversal_steps`: Overlapping and Gapped Tiles `[FOUNDATIONAL]`

### Intuition

`Array.tiled_view`'s `traversal_steps` parameter is documented to control the spacing between consecutive tile origins independently of the tile's own shape — smaller than `tile_shape` means overlap, larger means gaps. This section checks that the documented range of behavior genuinely compiles, and that its documented restrictions (matching rank, positive values) are genuinely enforced.

### Background

`traversal_steps` defaults to `tile_shape` itself (no overlap or gaps). Chapter 6 and Chapter 7 already established that a documented parameter range does not automatically mean every value in it is enforced, or that different values necessarily produce different compiled code — this section checks `traversal_steps` against that same standard rather than assuming it from the docstring alone.

### Worked Example 13.4.1 — the documented range, and two genuinely invalid values

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def build_kernel(traversal_steps):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO, traversal_steps=traversal_steps)
        tile = view.load((pid,))
        c_view = c.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
        c_view.store((pid,), tile)
    return kernel_fn

def compile_bytes(traversal_steps):
    buf = io.BytesIO()
    ct.compilation.export_kernel(build_kernel(traversal_steps), [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# traversal_steps[i] == tile_shape[i] (or None) means no overlap or gaps;
# traversal_steps[i] < tile_shape[i] means tiles overlap; > means gaps.
for name, steps in [
    ("None (default, no overlap)", None),
    ("(64,) matches tile_shape (no overlap)", (64,)),
    ("(32,) overlap", (32,)),
    ("(128,) gap", (128,)),
    ("(1,) maximal overlap", (1,)),
]:
    try:
        print(f"traversal_steps={name}: {compile_bytes(steps)} cubin bytes")
    except Exception as e:
        print(f"traversal_steps={name}: {type(e).__name__}: {e}")

# Genuinely invalid traversal_steps values.
for name, steps in [
    ("(64, 64) wrong rank for a 1D array", (64, 64)),
    ("(0,) zero step", (0,)),
    ("(-1,) negative step", (-1,)),
]:
    try:
        print(f"traversal_steps={name}: {compile_bytes(steps)} cubin bytes")
    except ct.TileError as e:
        print(f"traversal_steps={name}: {type(e).__name__}: {str(e).splitlines()[0]}")
```

The complete file is `05_traversal_steps.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_traversal_steps.py
```

Genuinely run:

```
traversal_steps=None (default, no overlap): 20896 cubin bytes
traversal_steps=(64,) matches tile_shape (no overlap): 20896 cubin bytes
traversal_steps=(32,) overlap: 20896 cubin bytes
traversal_steps=(128,) gap: 20896 cubin bytes
traversal_steps=(1,) maximal overlap: 20512 cubin bytes
traversal_steps=(64, 64) wrong rank for a 1D array: TileTypeError: Invalid argument "traversal_steps" of tiled_view(): Expected traversal_steps length to be 1, got 2
traversal_steps=(0,) zero step: TileTypeError: Invalid argument "traversal_steps" of tiled_view(): Dimension #0 of traversal_steps (0,) is not positive
traversal_steps=(-1,) negative step: TileTypeError: Invalid argument "traversal_steps" of tiled_view(): Dimension #0 of traversal_steps (-1,) is not positive
```

`None`, matching `tile_shape` explicitly, real overlap, and a real gap all compile to the identical 20,896 bytes for this single-tile-per-block kernel — only the most extreme tested value, `(1,)`, genuinely compiles smaller, the same "only the boundary moves the needle" pattern Section 13.3 already found for `tile_size`. Both documented restrictions are genuinely enforced with distinct, specific `TileTypeError` messages: a rank mismatch is caught separately from a non-positive value, and `0` and `-1` are rejected with the identical "not positive" wording, confirming the check is a single positivity test rather than two separate range checks.

## 13.5 A Genuine Two-Pass Kernel `[FOUNDATIONAL]`

### Intuition

Sections 13.1 through 13.4 established three independent, real pieces: a compiled loop with a runtime trip count, Chapter 8's masking technique for a real array boundary, and `traversal_steps`' control over tile spacing. Combined, they support something no earlier chapter's kernel could express: a single kernel body that processes the same row of data in two genuinely separate passes, each one a full pass over every tile.

### Background

A masked two-pass reduction — finding a row's real maximum in one pass, then its real minimum in a second, independent pass over the identical tiles — needs exactly this chapter's dynamic loop (since a row's length is not known at compile time) and exactly Chapter 8's masking (since the row's last tile may extend past the array's real boundary).

### Worked Example 13.5.1 — row range, in two genuine passes

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def masked_range(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)

    # Pass 1: a genuine, compiler-native loop with a runtime trip count
    # (Section 13.1) finds the row's real maximum, masking each tile's
    # padded tail against the array's real boundary (Chapter 8's technique).
    running_max = ct.full((tile_size,), -3.0e38, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        offsets = k * tile_size + ct.arange(tile_size, dtype=ct.int32)
        mask = offsets < a.shape[0]
        masked_tile = ct.where(mask, tile, ct.full((tile_size,), -3.0e38, ct.float32))
        running_max = ct.where(masked_tile > running_max, masked_tile, running_max)
    row_max = ct.max(running_max)

    # Pass 2: the identical loop shape, run a second time over the same
    # tiles, this time tracking the row's real minimum.
    running_min = ct.full((tile_size,), 3.0e38, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        offsets = k * tile_size + ct.arange(tile_size, dtype=ct.int32)
        mask = offsets < a.shape[0]
        masked_tile = ct.where(mask, tile, ct.full((tile_size,), 3.0e38, ct.float32))
        running_min = ct.where(masked_tile < running_min, masked_tile, running_min)
    row_min = ct.min(running_min)

    c_view = c.tiled_view((1,))
    c_view.store((pid,), row_max - row_min)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(masked_range, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"two-pass masked range kernel: {len(buf.getvalue())} cubin bytes")
except Exception as e:
    print(f"two-pass masked range kernel: {type(e).__name__}: {e}")
```

The complete file is `06_two_pass_masked_range.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_two_pass_masked_range.py
```

Genuinely run:

```
two-pass masked range kernel: 89216 cubin bytes
```

This kernel genuinely compiles, confirming that a runtime-trip-count loop, run twice over the same conceptual data with two independent accumulators, and masked against a real array boundary each time, composes cleanly into one coherent kernel body — no combination of these three, individually-confirmed pieces broke when brought together. Nothing about `pid`, `mask`, or `n` is knowable until a real array and grid actually exist; `export_kernel` compiles this kernel's full logic anyway, exactly as every prior chapter's driver-free verification has.

## Complete Runnable Code

### File: `01_dynamic_trip_count_loop.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# view.num_tiles(0) is a genuine, runtime int32 value -- it depends on the
# array's real shape (a Chapter 8 finding), which is not known at compile
# time. "for k in range(n)" with n a runtime value cannot be unrolled by the
# tracer the way Chapter 11's "for _ in range(n)" was, since n has no
# concrete Python value to unroll against. This checks whether the compiler
# instead compiles this into a genuine, native loop.
@ct.kernel
def row_sum(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(row_sum, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"data-dependent 'for k in range(view.num_tiles(0))' loop: {len(buf.getvalue())} cubin bytes")
except Exception as e:
    print(f"data-dependent loop: {type(e).__name__}: {e}")
```

```bash
python3 01_dynamic_trip_count_loop.py
```

### File: `02_dynamic_vs_unrolled_same_work.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig_dynamic = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_unrolled = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64), ct.compilation.ConstantConstraint(4)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def row_sum_dynamic(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

@ct.kernel
def row_sum_unrolled(a, c, tile_size: ct.Constant[int], n: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Both kernels do the same real work for an array whose row happens to be
# exactly 4 tiles wide: sum 4 tiles of the same tile_size. row_sum_dynamic
# does not know that "4" at compile time; row_sum_unrolled is handed it
# directly as a ct.Constant[int], the way Chapter 11 built its unrolled
# loops.
print(f"genuinely compiled loop (trip count from view.num_tiles(0)): {compile_bytes(row_sum_dynamic, sig_dynamic)} cubin bytes")
print(f"Python-unrolled loop (trip count 4, a ct.Constant[int]): {compile_bytes(row_sum_unrolled, sig_unrolled)} cubin bytes")
```

```bash
python3 02_dynamic_vs_unrolled_same_work.py
```

### File: `03_dynamic_loop_tile_size_scaling.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

@ct.kernel
def row_sum_dynamic(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

# tile_size is the one real compile-time knob left in the dynamic-loop
# kernel -- the loop's own trip count is a genuine runtime value with no
# compile-time equivalent to sweep.
for tile_size in [32, 64, 128]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(tile_size)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print(f"dynamic-loop kernel, tile_size={tile_size}: {compile_bytes(row_sum_dynamic, sig)} cubin bytes")
```

```bash
python3 03_dynamic_loop_tile_size_scaling.py
```

### File: `04_unrolled_loop_n_scaling.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

@ct.kernel
def row_sum_unrolled(a, c, tile_size: ct.Constant[int], n: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    total = ct.full((tile_size,), 0.0, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        total = ct.add(total, tile)
    result = ct.sum(total)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), result)

for n in [1, 2, 4, 8]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(64), ct.compilation.ConstantConstraint(n)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print(f"Python-unrolled loop, n={n}: {compile_bytes(row_sum_unrolled, sig)} cubin bytes")
```

```bash
python3 04_unrolled_loop_n_scaling.py
```

### File: `05_traversal_steps.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def build_kernel(traversal_steps):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO, traversal_steps=traversal_steps)
        tile = view.load((pid,))
        c_view = c.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
        c_view.store((pid,), tile)
    return kernel_fn

def compile_bytes(traversal_steps):
    buf = io.BytesIO()
    ct.compilation.export_kernel(build_kernel(traversal_steps), [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# traversal_steps[i] == tile_shape[i] (or None) means no overlap or gaps;
# traversal_steps[i] < tile_shape[i] means tiles overlap; > means gaps.
for name, steps in [
    ("None (default, no overlap)", None),
    ("(64,) matches tile_shape (no overlap)", (64,)),
    ("(32,) overlap", (32,)),
    ("(128,) gap", (128,)),
    ("(1,) maximal overlap", (1,)),
]:
    try:
        print(f"traversal_steps={name}: {compile_bytes(steps)} cubin bytes")
    except Exception as e:
        print(f"traversal_steps={name}: {type(e).__name__}: {e}")

# Genuinely invalid traversal_steps values.
for name, steps in [
    ("(64, 64) wrong rank for a 1D array", (64, 64)),
    ("(0,) zero step", (0,)),
    ("(-1,) negative step", (-1,)),
]:
    try:
        print(f"traversal_steps={name}: {compile_bytes(steps)} cubin bytes")
    except ct.TileError as e:
        print(f"traversal_steps={name}: {type(e).__name__}: {str(e).splitlines()[0]}")
```

```bash
python3 05_traversal_steps.py
```

### File: `06_two_pass_masked_range.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def masked_range(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    n = view.num_tiles(0)

    # Pass 1: a genuine, compiler-native loop with a runtime trip count
    # (Section 13.1) finds the row's real maximum, masking each tile's
    # padded tail against the array's real boundary (Chapter 8's technique).
    running_max = ct.full((tile_size,), -3.0e38, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        offsets = k * tile_size + ct.arange(tile_size, dtype=ct.int32)
        mask = offsets < a.shape[0]
        masked_tile = ct.where(mask, tile, ct.full((tile_size,), -3.0e38, ct.float32))
        running_max = ct.where(masked_tile > running_max, masked_tile, running_max)
    row_max = ct.max(running_max)

    # Pass 2: the identical loop shape, run a second time over the same
    # tiles, this time tracking the row's real minimum.
    running_min = ct.full((tile_size,), 3.0e38, ct.float32)
    for k in range(n):
        tile = view.load((k,))
        offsets = k * tile_size + ct.arange(tile_size, dtype=ct.int32)
        mask = offsets < a.shape[0]
        masked_tile = ct.where(mask, tile, ct.full((tile_size,), 3.0e38, ct.float32))
        running_min = ct.where(masked_tile < running_min, masked_tile, running_min)
    row_min = ct.min(running_min)

    c_view = c.tiled_view((1,))
    c_view.store((pid,), row_max - row_min)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(masked_range, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"two-pass masked range kernel: {len(buf.getvalue())} cubin bytes")
except Exception as e:
    print(f"two-pass masked range kernel: {type(e).__name__}: {e}")
```

```bash
python3 06_two_pass_masked_range.py
```

## Chapter Summary

`for k in range(view.num_tiles(axis))` compiles to a genuine, native loop with a runtime trip count, confirmed by compiling it successfully with no array ever having existed — categorically different from Chapter 11's Python-trace-time-unrolled loops, since a runtime value has no concrete number for the tracer to unroll against. That flexibility has a real, measured cost: the same four-tile sum compiled through a genuine loop was nearly three times larger than the fully unrolled equivalent. The two loop shapes also grow differently as their respective knobs change — the dynamic loop's size across `tile_size` was non-monotonic, the same boundary-effect caution this book has applied elsewhere, while the unrolled loop's size grew monotonically with its own trip count, a genuinely different and more intuitive shape. `Array.tiled_view`'s `traversal_steps` parameter compiles correctly across its documented range of overlap and gaps, producing a measurably smaller result only at its most extreme tested value, and is genuinely enforced against both its rank and its positivity requirement, each with its own distinct `TileTypeError`. Finally, all three pieces — a runtime-trip-count loop, Chapter 8's masking technique, and two independent accumulators — composed cleanly into one real, two-pass kernel body that `export_kernel` compiled successfully without complaint, confirming this chapter's individually-tested primitives combine the way a reader would hope.

## Self-Check Questions

1. Section 13.1's loop compiles successfully even though no real array, and therefore no real trip count, ever exists in this book's `export_kernel`-only environment. What does that confirm about where the loop's trip count is actually evaluated?
2. Section 13.2 found the genuinely compiled loop nearly three times larger than the unrolled equivalent for the same four-tile sum. Does this mean a data-dependent loop is always the wrong choice? What information would you need that this chapter's environment cannot provide to answer that fully?
3. Section 13.3 found the dynamic loop's size non-monotonic across `tile_size`, while the unrolled loop's size was monotonic across its own trip count `n`. Are these two findings in tension with each other? Why or why not?
4. Section 13.4 found `traversal_steps=(0,)` and `traversal_steps=(-1,)` rejected with the identical "not positive" message. What single, simpler check would produce exactly that pattern of results?
5. Section 13.5's two-pass kernel runs the identical loop shape twice, once for a maximum and once for a minimum. What would you need to change about this kernel to combine both passes into a single loop instead, and what trade-off might that involve given Section 13.2's finding about loop cost?

## Where We Go Next

This chapter confirmed cuTile Python supports two structurally different notions of "loop" — one resolved entirely at trace time by unrolling, and one genuinely compiled with a runtime trip count — and that combining a runtime loop with earlier chapters' masking technique produces a real, multi-pass kernel body. Every primitive this book has tested from Chapter 1 onward has been confirmed one piece at a time, in isolation; Chapter 14 turns to auditing how much of that confirmed surface actually gets exercised together, examining what happens when several of this book's own worked examples are combined into a single larger program and checking, deliberately, for combinations this book has not yet tried.

## Worked Solutions

**1.** It confirms the trip count is evaluated on the device, at kernel launch time, by real generated loop-control instructions — not resolved by the Python tracer during `export_kernel`. If the tracer needed a concrete number to compile this kernel at all, compilation would have failed outright with no array present, the same way Chapter 11's helper-call loops needed a concrete Python int to unroll.

**2.** No — this chapter's `export_kernel`-only environment can only measure compiled size, never actual runtime performance. A larger compiled kernel is not automatically a slower one; the loop-control overhead this chapter measured as extra bytes might be negligible against real memory-bound work on actual hardware, or might not be, and only a genuine `ct.launch` against real hardware, which this book has never had since Chapter 1, could answer that.

**3.** No tension — they are two independent claims about two different knobs on two different kernels. Section 13.3's non-monotonic result is about how the dynamic loop's compiled size responds to `tile_size`; the unrolled loop's monotonic result is about how its compiled size responds to its own trip count `n`, a completely different parameter on a different kernel. Nothing requires two unrelated relationships to share the same shape.

**4.** A single check of the form "is this value greater than zero," applied independently to each dimension, produces exactly this pattern: `0` fails it because zero is not greater than zero, and `-1` fails it for the more obvious reason of being negative, but both failures are reported through the identical message because the underlying test does not distinguish "exactly zero" from "negative" as separate cases.

**5.** You would need to fold the max and min accumulation into the body of a single loop, updating `running_max` and `running_min` together from the same loaded and masked tile each iteration, rather than loading and masking every tile twice across two separate loops. The trade-off is real: Section 13.2 found a genuinely compiled loop measurably more expensive than the equivalent unrolled work, but that finding was about trip count, not about how much work happens inside one iteration — combining the two reductions into one loop body means each iteration does more, and this chapter's environment cannot say without a real compile-and-compare whether that combined loop compiles smaller than two separate ones, only that it is a real, testable question a curious reader could check the same way this chapter checked everything else.
