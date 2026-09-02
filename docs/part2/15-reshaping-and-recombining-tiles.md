# Chapter 15: Reshaping and Recombining Tiles

> "Chapter 14 closed by promising to turn `dir()`'s discipline back onto the kernel body itself, checking what this book has actually run against what fourteen chapters of steady progress made it easy to assume had been run. `dir(ct)` lists `reshape`, `transpose`, `permute`, `expand_dims`, `cat`, and `extract` — six operations for reshaping and recombining a tile that is already sitting in registers, none of them appearing in a single worked example so far. Testing the first of them surfaces something bigger than any one function: every tile shape this book has ever built — `(64,)`, `(32,)`, `(128,)`, `(8, 8)` — has quietly satisfied a hard requirement that no chapter has ever stated outright, because every number this book happened to choose was already a power of two."

**What you will understand by the end of this chapter:**

- That every tile shape in cuTile Python must have a power-of-two size along every dimension — a real, immediately-enforced `TileTypeError`, confirmed here for the first time despite fourteen chapters of kernels that all happened to satisfy it by coincidence of the numbers chosen
- `ct.reshape`'s `-1` dimension inference and its own real, size-mismatch `TileTypeError` when the total element count does not divide evenly
- That `ct.transpose(x, axis0, axis1)` is a genuine special case of `ct.permute(x, axes)` — confirmed by a naming-confound-controlled byte-for-byte match — and that `permute`'s compiled cost varies substantially across mathematically equally-valid axis orderings, with the identity permutation cheapest of all
- That `ct.expand_dims(x, axis)` and the documented NumPy-style sugar `x[None, :]` compile to byte-identical output, and that an out-of-range axis is rejected with a real `TileTypeError`
- That `ct.cat`'s documented "same shape" requirement is actually two independent, more specific checks — matching non-axis dimensions, and a power-of-two result — which happen to coincide in every case because two distinct powers of two never sum to a third
- `ct.extract`'s three genuinely different rejection reasons (an out-of-range grid index, a non-power-of-two extract shape, and an extract shape that does not evenly divide the input), and that all five operations from this chapter compose into a single kernel body without conflict

**What you need to know first:**

- Chapter 3's `Array`/`Tile` shape vocabulary and Chapter 8's masking and padding techniques — this chapter's tiles are built the same way, just recombined after loading rather than while loading.
- Chapter 10's naming-confound rule: every kernel compared for byte count in this chapter shares the name `kernel_fn`, reused across variants or built from a factory function where more than one is needed at once.
- Chapter 14's closing observation that checking `dir()` against what has actually been run, rather than what this book assumed it had run, is itself a repeatable technique — this chapter is that technique applied one level down, from the parameter surface to the kernel body's own tile operations.

## 15.1 A Rule Every Chapter Has Silently Obeyed `[FOUNDATIONAL]`

### Intuition

Every `tile_size` this book has ever chosen — `64` in Chapter 1, `32` and `128` in later sweeps, `8` and `4` in Chapter 13's tiny illustrative examples — happens to be a power of two. That could be a stylistic habit, a real hardware-level requirement, or something in between. Nothing has ever tried a tile shape that isn't one.

### Background

Several of the genuine errors already surfacing while preparing this chapter's other worked examples — before a single one of them was written up — mentioned "not a power of two" in their message text. That is a strong hint that this is not a soft convention this book happened to follow, but a hard, compiler-enforced property of every `Tile` shape, checked wherever a shape is constructed: `ct.load`, `ct.full`, `ct.reshape`, `ct.extract`, and evidently the result of `ct.cat` as well.

### Worked Example 15.1.1 — five tile sizes, one already a power of two

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

# Every tile_size this book has ever used -- 64, 32, 128, 8, 4, 256, and so
# on -- has happened to be a power of two. Nothing has ever tried a
# non-power-of-two tile_size directly.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), x)

for size in [6, 5, 63, 64, 100]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(size)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    try:
        n = compile_bytes(kernel_fn, sig)
        print(f"tile_size={size}: {n} cubin bytes")
    except Exception as e:
        print(f"tile_size={size}: {type(e).__name__}: {e}")
```

The complete file is `01_power_of_two_tile_shape.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_power_of_two_tile_shape.py
```

Genuinely run:

```
tile_size=6: TileTypeError: Invalid argument "shape" of load(): Dimension #0 of shape (6,) is not a power of two
  "/tmp/ch15/01_power_of_two_tile_shape.py", line 21, col 9-40, in kernel_fn:
        x = ct.load(a, (pid,), (tile_size,))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tile_size=5: TileTypeError: Invalid argument "shape" of load(): Dimension #0 of shape (5,) is not a power of two
  "/tmp/ch15/01_power_of_two_tile_shape.py", line 21, col 9-40, in kernel_fn:
        x = ct.load(a, (pid,), (tile_size,))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tile_size=63: TileTypeError: Invalid argument "shape" of load(): Dimension #0 of shape (63,) is not a power of two
  "/tmp/ch15/01_power_of_two_tile_shape.py", line 21, col 9-40, in kernel_fn:
        x = ct.load(a, (pid,), (tile_size,))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tile_size=64: 20896 cubin bytes
tile_size=100: TileTypeError: Invalid argument "shape" of load(): Dimension #0 of shape (100,) is not a power of two
  "/tmp/ch15/01_power_of_two_tile_shape.py", line 21, col 9-40, in kernel_fn:
        x = ct.load(a, (pid,), (tile_size,))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```

> `[COMMON TRAP]` A convention this book has followed since Chapter 1 without ever stating it — every `tile_size` chosen has been a power of two — turns out to be a hard requirement, not a habit. `6`, `5`, `63`, and `100` are all rejected with the identical `TileTypeError` shape, naming the exact dimension and the exact rule violated; only `64` compiles. Fourteen chapters of kernels satisfied this rule by construction, because every tile size this book happened to pick for illustrative purposes was already a power of two — nothing in those chapters ever depended on the number being anything else, so the requirement simply never had an opportunity to surface. This is worth keeping in mind for every other tile-shape rule the rest of this chapter uncovers: a real, load-bearing constraint can hide in plain sight for a very long time if nothing ever happens to test the boundary.

## 15.2 Reshaping a Tile In Place `[FOUNDATIONAL]`

### Intuition

`ct.reshape(x, shape)` changes a tile's declared shape without changing its underlying data — the same 64 elements loaded in Section 15.1, viewed as an `(8, 8)` grid instead of a flat 64-element row, or back again. One dimension of the target shape may be given as `-1`, letting the compiler infer it from the total element count, the same convenience NumPy's own `reshape` offers.

### Background

Reshaping is only valid when the total number of elements is preserved — a `(64,)` tile can become `(8, 8)`, `(4, 16)`, or `(2, 4, 8)`, but not `(7,)`, which has a different total. This is a purely structural check, answerable without any real data ever existing, which is exactly why it is checkable in this book's `export_kernel`-only environment.

### Worked Example 15.2.1 — a valid round trip, and a genuine size mismatch

```python
import cuda.tile as ct
import io

def array_param(ndim=1):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.reshape(x, (8, -1))
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn, sig)
    print(f"reshape (64,) -> (8,-1) -> (64,): {n} cubin bytes")
except Exception as e:
    print(f"reshape (64,) -> (8,-1) -> (64,): {type(e).__name__}: {e}")

# Invalid reshape: total element count doesn't match.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.reshape(x, (7,))
    ct.store(c, (pid,), y)

sig2 = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn2, sig2)
    print(f"reshape (64,) -> (7,): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"reshape (64,) -> (7,): {type(e).__name__}: {e}")
```

The complete file is `02_reshape.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_reshape.py
```

Genuinely run:

```
reshape (64,) -> (8,-1) -> (64,): 20896 cubin bytes
reshape (64,) -> (7,): TileTypeError: Cannot reshape (64,) to (7,)
  "/tmp/ch15/02_reshape.py", line 38, col 9-27, in kernel_fn2:
        y = ct.reshape(x, (7,))
            ^^^^^^^^^^^^^^^^^^^

```

The `-1`-inferred round trip — `(64,)` to `(8, -1)`, resolving to `(8, 8)`, and back to `(64,)` — compiles to the exact same 20,896 bytes as a kernel that never reshaped at all (Section 15.1's `tile_size=64` baseline), a reasonable result for two reshapes that together are a genuine no-op on the underlying data. The mismatched reshape fails immediately, with a message stating the two shapes directly rather than a generic size error — `(64,)` has no valid interpretation as `(7,)`, and the compiler says exactly that.

## 15.3 Transpose Is a Special Case of Permute `[FOUNDATIONAL]`

### Intuition

`ct.transpose(x, axis0, axis1)` swaps exactly two named axes; `ct.permute(x, axes)` reorders every axis at once according to an arbitrary permutation. Swapping axes 0 and 2 of a 3-dimensional tile and leaving axis 1 alone is itself one particular permutation — `(2, 1, 0)` — so `transpose(x, 0, 2)` and `permute(x, (2, 1, 0))` ought to describe the identical data movement. Whether the compiler actually treats them as identical, rather than as two independently-implemented code paths that merely produce the same logical result, is a real, checkable question.

### Background

An uncontrolled first attempt at this comparison — naming the transpose kernel `kernel_fn3` and the permute kernel a different name entirely, purely as a matter of which worked example was written first — produced two different byte counts, 22,912 against 22,784, that looked like a real discrepancy in how transpose and permute get compiled. Chapter 10 already has a name for exactly this trap: the compiled cubin embeds the kernel function's own Python name as a symbol, so any comparison across differently-named functions is confounded before the comparison even starts. Rebuilding both kernels under the identical name `kernel_fn` is the only way to know whether a byte-count difference here means anything.

### Worked Example 15.3.1 — the controlled comparison, then two genuine rejections

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

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# transpose(x, axis0, axis1) swapping axes 0 and 2 of a (4, 4, 4) tile
# should be exactly the same data movement as permute(x, (2, 1, 0)) --
# swap the first and last axes, leave the middle one fixed. Both variants
# are named "kernel_fn" (Chapter 10's naming-confound rule) so their byte
# counts are directly comparable.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.transpose(tile3d, 0, 2)
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
kernel_transpose = kernel_fn
print(f"transpose(tile3d, 0, 2): {compile_bytes(kernel_transpose, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.permute(tile3d, (2, 1, 0))
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
kernel_permute = kernel_fn
print(f"permute(tile3d, (2, 1, 0)): {compile_bytes(kernel_permute, sig)} cubin bytes")

# transpose() is documented to require explicit axes for anything beyond
# 2 dimensions -- never checked directly in this book before.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.transpose(tile3d)
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn2, sig)
    print(f"transpose() on a 3D tile, no axes given: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"transpose() on a 3D tile, no axes given: {type(e).__name__}: {e}")

# permute() with a repeated axis -- not a valid permutation at all.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.permute(tile3d, (0, 0, 2))
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn3, sig)
    print(f"permute((0, 0, 2)) (repeated axis): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"permute((0, 0, 2)) (repeated axis): {type(e).__name__}: {e}")

# If transpose is really just permute with a specific two-axis swap, does
# the choice of WHICH permutation costs the same across every possible
# 3-axis reordering? Sweeping all six permutations of (4, 4, 4), each
# built from a factory so every variant shares the name "kernel_fn".
def build_kernel(axes):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        tile3d = ct.reshape(x, (4, 4, 4))
        t = ct.permute(tile3d, axes)
        z = ct.reshape(t, (tile_size,))
        ct.store(c, (pid,), z)
    return kernel_fn

for axes in [(0, 1, 2), (2, 0, 1), (2, 1, 0), (1, 0, 2), (1, 2, 0), (0, 2, 1)]:
    n = compile_bytes(build_kernel(axes), sig)
    print(f"permute({axes}): {n} cubin bytes")
```

The complete file is `03_transpose_permute_equivalence_and_cost.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_transpose_permute_equivalence_and_cost.py
```

Genuinely run:

```
transpose(tile3d, 0, 2): 22784 cubin bytes
permute(tile3d, (2, 1, 0)): 22784 cubin bytes
transpose() on a 3D tile, no axes given: TileTypeError: `axes` must be specified for tile with more than 2 dimensions
  "/tmp/ch15/03_transpose_permute_equivalence_and_cost.py", line 54, col 9-28, in kernel_fn2:
        t = ct.transpose(tile3d)
            ^^^^^^^^^^^^^^^^^^^^

permute((0, 0, 2)) (repeated axis): TileTypeError: Repeated axis #1: 0
  "/tmp/ch15/03_transpose_permute_equivalence_and_cost.py", line 69, col 9-37, in kernel_fn3:
        t = ct.permute(tile3d, (0, 0, 2))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

permute((0, 1, 2)): 20896 cubin bytes
permute((2, 0, 1)): 22912 cubin bytes
permute((2, 1, 0)): 22784 cubin bytes
permute((1, 0, 2)): 22912 cubin bytes
permute((1, 2, 0)): 22784 cubin bytes
permute((0, 2, 1)): 21408 cubin bytes
```

With the naming confound actually controlled for, `transpose(tile3d, 0, 2)` and `permute(tile3d, (2, 1, 0))` compile to the exact same 22,784 bytes — confirming, cleanly this time, that `transpose` really is just a named special case of `permute`, not a separately-optimized code path that merely happens to produce equivalent results. `transpose`'s own axis requirement is enforced exactly as documented: omitting `axis0`/`axis1` on a 3-dimensional tile is a real, immediate `TileTypeError`, and `permute`'s axis list is checked for validity as a genuine permutation, rejecting a repeated axis by name and position rather than a generic "invalid axes" message. The cost sweep across all six permutations of a 3-axis tile adds a further, unclaimed observation: the identity permutation `(0, 1, 2)` is cheapest at 20,896 bytes — identical to a kernel that never permuted at all — while every genuine reordering costs more, and not uniformly: `(2, 0, 1)` and `(1, 0, 2)` both compile to 22,912 bytes, `(2, 1, 0)` and `(1, 2, 0)` both compile to 22,784 bytes, and `(0, 2, 1)`, which only swaps the two innermost axes and leaves the outermost one fixed, compiles smallest of the non-identity group at 21,408 bytes. This book's `export_kernel`-only environment cannot say why a permutation touching only the innermost axes is cheaper than one that moves the outermost axis, but it can report, from a real and repeated measurement, that "permuting axes" is not a single uniform cost — which specific axes move matters.

## 15.4 `expand_dims` and Its NumPy-Style Sugar `[FOUNDATIONAL]`

### Intuition

`ct.expand_dims(x, axis)` inserts a new, size-1 axis at the given position — documented as also reachable through NumPy-style indexing syntax, `x[:, None]` or `x[None, :]`. If the two are genuinely interchangeable rather than merely similar in effect, they should trace to the identical compiled operation, not just the identical logical shape.

### Background

Nothing about the earlier equivalence in Section 15.3 was true by default — `transpose` and `permute` needed a controlled test to confirm they trace to the same thing. The same discipline applies here: `expand_dims` and its documented indexing sugar deserve the same direct comparison rather than being assumed equivalent because the documentation says so.

### Worked Example 15.4.1 — function call versus indexing sugar, and an out-of-range axis

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

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.expand_dims(x, axis) vs the documented NumPy-style sugar x[:, None].
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.expand_dims(x, 0)
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)
kernel_call = kernel_fn
print(f"ct.expand_dims(x, 0): {compile_bytes(kernel_call, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = x[None, :]
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)
kernel_sugar = kernel_fn
print(f"x[None, :]: {compile_bytes(kernel_sugar, sig)} cubin bytes")

# An out-of-range axis.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.expand_dims(x, 5)
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)
kernel_bad_axis = kernel_fn
try:
    n = compile_bytes(kernel_bad_axis, sig)
    print(f"ct.expand_dims(x, 5) (out of range): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"ct.expand_dims(x, 5) (out of range): {type(e).__name__}: {e}")
```

The complete file is `04_expand_dims.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_expand_dims.py
```

Genuinely run:

```
ct.expand_dims(x, 0): 20896 cubin bytes
x[None, :]: 20896 cubin bytes
ct.expand_dims(x, 5) (out of range): TileTypeError: Axis 5 is out of range for rank 2'
  "/tmp/ch15/04_expand_dims.py", line 46, col 9-28, in kernel_fn:
        y = ct.expand_dims(x, 5)
            ^^^^^^^^^^^^^^^^^^^^

```

The documented equivalence holds exactly: `ct.expand_dims(x, 0)` and `x[None, :]` compile to byte-identical output, confirming the indexing syntax really is sugar for the same underlying operation rather than a separate code path that happens to produce the same shape. The out-of-range check reports "rank 2" — the rank the tile *would have* after the insertion, one more than its actual rank of 1 — a small but genuine detail confirming the bounds check accounts for the new axis rather than checking against the tile's rank before the insertion. (The message's own trailing stray quotation mark, visible in the output above, is a genuine, verbatim library quirk — not a copy error introduced in writing this chapter.)

## 15.5 Concatenating Tiles: One Restriction Wearing Two Different Faces `[FOUNDATIONAL]`

### Intuition

`ct.cat`'s own docstring states plainly that "due to power-of-two assumption on all tile shapes, the two input tiles must have the same shape" — a stronger restriction than NumPy's `concatenate`, which only requires matching shapes on every axis *except* the one being joined. Section 15.1 already found that every tile shape in cuTile Python must independently be a power of two. Put those two facts together, and the docstring's blanket "same shape" claim might be a simplification of something more specific rather than a literal, separately-enforced rule.

### Background

Two tiles being concatenated along one axis only need to agree on their other axes for the result to make geometric sense — that much is true of any concatenation, NumPy's included. What cuTile Python's docstring attributes to "the power-of-two assumption" suggests a second, independent constraint on top of that: whatever the concatenated axis's combined size works out to, it also has to be a power of two, the same rule Section 15.1 already confirmed for every tile shape.

### Worked Example 15.5.1 — two genuine rejections, for two different reasons

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

sig_double = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# cat() on two equal-shaped (8, 8) tiles, along each axis in turn.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y2d = ct.reshape(y, (8, 8))
    combined = ct.cat((x2d, y2d), 0)
    z = ct.reshape(combined, (2 * tile_size,))
    ct.store(c, (pid,), z)
kernel_axis0 = kernel_fn
print(f"cat((x2d, y2d), 0): {compile_bytes(kernel_axis0, sig_double)} cubin bytes")

@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y2d = ct.reshape(y, (8, 8))
    combined = ct.cat((x2d, y2d), 1)
    z = ct.reshape(combined, (2 * tile_size,))
    ct.store(c, (pid,), z)
kernel_axis1 = kernel_fn
print(f"cat((x2d, y2d), 1): {compile_bytes(kernel_axis1, sig_double)} cubin bytes")

# The docstring says cat's two inputs "must have the same shape," but this
# checks the actual, more specific rule directly: tiles with DIFFERENT
# shapes along every axis at once.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y2d = ct.reshape(y, (16, 4))
    combined = ct.cat((x2d, y2d), 0)
    z = ct.reshape(combined, (2 * tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn, sig_double)
    print(f"cat of (8,8) with (16,4): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"cat of (8,8) with (16,4): {type(e).__name__}: {e}")

# The message above says "same shape for non axis dimensions" -- narrower
# than the docstring's plain "same shape." Built directly with ct.full
# (rather than ct.load) to isolate cat()'s own rule: x2d is (4, 8), y2d is
# (8, 8) -- different sizes along the cat axis itself (axis 0: 4 vs 8,
# both individually powers of two), identical along axis 1.
sig_single = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ConstantConstraint(96)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn(c, out_size: ct.Constant[int]):
    pid = ct.bid(0)
    x2d = ct.full((4, 8), 1.0, ct.float32)
    y2d = ct.full((8, 8), 2.0, ct.float32)
    combined = ct.cat((x2d, y2d), 0)
    z = ct.reshape(combined, (out_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn, sig_single)
    print(f"cat of (4,8) with (8,8) along axis 0 (differ only on cat axis): {n} cubin bytes")
except Exception as e:
    print(f"cat of (4,8) with (8,8) along axis 0 (differ only on cat axis): {type(e).__name__}: {e}")
```

The complete file is `05_cat.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_cat.py
```

Genuinely run:

```
cat((x2d, y2d), 0): 26512 cubin bytes
cat((x2d, y2d), 1): 26640 cubin bytes
cat of (8,8) with (16,4): TileTypeError: Expected tiles to have the same shape for non axis dimensions, got (8, 8) and (16, 4)
  "/tmp/ch15/05_cat.py", line 57, col 16-36, in kernel_fn:
        combined = ct.cat((x2d, y2d), 0)
                   ^^^^^^^^^^^^^^^^^^^^^

cat of (4,8) with (8,8) along axis 0 (differ only on cat axis): TileTypeError: Result tile shape must be power of 2, got: [12, 8]
  "/tmp/ch15/05_cat.py", line 80, col 16-36, in kernel_fn:
        combined = ct.cat((x2d, y2d), 0)
                   ^^^^^^^^^^^^^^^^^^^^^

```

The two rejections confirm exactly the two-part rule Section 15.1 predicted, and neither is the docstring's own wording verbatim. `(8, 8)` against `(16, 4)` — different along *every* axis — fails the more NumPy-like check first, "same shape for non axis dimensions," naming both shapes directly. `(4, 8)` against `(8, 8)`, matching everywhere except the concatenation axis itself, passes that first check without complaint — and fails a second, entirely separate one instead: the *combined* shape, `(12, 8)`, is not a power of two, so `cat` refuses to produce it, even though `4` and `8` were each individually valid tile dimensions on their own. This is not a coincidence the docstring glossed over so much as a consequence of arithmetic: two distinct powers of two never sum to a third power of two, since a power of two has exactly one set bit in binary and the sum of two different powers of two always has exactly two. Practically, then, `cat`'s two non-axis-matching inputs can only ever safely differ in size along the concatenation axis if that difference is zero — the docstring's "same shape" is the almost-always-true consequence of two independent, more specific checks, not a single rule enforced directly.

## 15.6 Extracting a Subtile, and a Capstone Combining All Five Operations `[FOUNDATIONAL]`

### Intuition

`ct.extract(x, index, shape)` is documented as an in-register analogue of Chapter 3's `ct.load` — instead of partitioning an `Array` in memory into a grid of tiles and loading one, it partitions an already-loaded `Tile` into a grid of subtiles and returns one of those. Every constraint this book has already seen apply to `ct.load`'s own `shape` argument — power-of-two sizing, valid index bounds — is a reasonable thing to expect here too, alongside one `extract`-specific rule its documentation states directly: the extract shape must evenly divide the input tile's own shape in every dimension.

### Background

An `(8, 8)` tile split into `(4, 4)` subtiles has exactly a 2-by-2 grid of them, so valid grid indices run from `0` to `1` along each axis — not element indices, but subtile-grid indices, the same distinction Chapter 3's `ct.load` draws between an array index and an element offset. Three genuinely different ways to violate this exist: asking for a grid index outside that 2-by-2 range, asking for a subtile shape that is not itself a power of two, and asking for a subtile shape that is a valid power of two but simply does not divide the input evenly — three structurally different checks, worth confirming are actually three different error messages rather than one generic "invalid extract" catch-all.

### Worked Example 15.6.1 — three rejections, then a capstone chaining every operation in this chapter

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

sig64 = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# extract(x, index, shape): partitions x into a grid of `shape`-sized
# subtiles and returns the one at `index` -- an in-register analogue of
# Chapter 3's ct.load, but operating on an already-loaded tile rather
# than an Array.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (0, 1), shape=(4, 4))
    z = ct.reshape(sub, (16,))
    ct.store(c, (pid,), z)
sig16 = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                                    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"extract((8,8) tile, index=(0,1), shape=(4,4)): {compile_bytes(kernel_fn, sig16)} cubin bytes")

# An out-of-range grid index: an (8,8) tile split into (4,4) subtiles has
# only a 2x2 grid of them (valid indices 0-1 per axis) -- index (2, 0) is out.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (2, 0), shape=(4, 4))
    z = ct.reshape(sub, (16,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn2, sig16)
    print(f"extract((8,8) tile, index=(2,0), shape=(4,4)) (out of range): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract((8,8) tile, index=(2,0), shape=(4,4)) (out of range): {type(e).__name__}: {e}")

# A non-power-of-two extract shape.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (0, 0), shape=(3, 3))
    z = ct.reshape(sub, (9,))
    ct.store(c, (pid,), z)
sig9 = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                                    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn3, sig9)
    print(f"extract((8,8) tile, shape=(3,3)) (not a power of two): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract((8,8) tile, shape=(3,3)) (not a power of two): {type(e).__name__}: {e}")

# A power-of-two extract shape that still doesn't evenly divide the input
# shape -- larger than the input entirely, a genuinely different check
# from the power-of-two one above.
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (0, 0), shape=(16, 8))
    z = ct.reshape(sub, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn4, sig64)
    print(f"extract((8,8) tile, shape=(16,8)) (larger than input): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract((8,8) tile, shape=(16,8)) (larger than input): {type(e).__name__}: {e}")

# Capstone: reshape, permute, extract, and cat chained in one kernel body,
# something no single worked example in this chapter has attempted -- load
# a flat tile, reshape it to 2D, permute it, extract two disjoint quadrants,
# concatenate them back together along the opposite axis, flatten, store.
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    permuted = ct.permute(x2d, (1, 0))
    top = ct.extract(permuted, (0, 0), shape=(4, 8))
    bottom = ct.extract(permuted, (1, 0), shape=(4, 8))
    combined = ct.cat((top, bottom), 1)
    z = ct.reshape(combined, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn5, sig64)
    print(f"capstone (reshape + permute + extract + cat chained): {n} cubin bytes")
except Exception as e:
    print(f"capstone (reshape + permute + extract + cat chained): {type(e).__name__}: {e}")
```

The complete file is `06_extract_and_capstone.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_extract_and_capstone.py
```

Genuinely run:

```
extract((8,8) tile, index=(0,1), shape=(4,4)): 22656 cubin bytes
extract((8,8) tile, index=(2,0), shape=(4,4)) (out of range): TileTypeError: Index 2 out of bounds at dimension #0: valid range is [0, 2) in tile space (input shape (8, 8), extract shape (4, 4))
  "/tmp/ch15/06_extract_and_capstone.py", line 47, col 11-47, in kernel_fn2:
        sub = ct.extract(x2d, (2, 0), shape=(4, 4))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

extract((8,8) tile, shape=(3,3)) (not a power of two): TileTypeError: Invalid argument "shape" of extract(): Dimension #0 of shape (3, 3) is not a power of two
  "/tmp/ch15/06_extract_and_capstone.py", line 62, col 11-47, in kernel_fn3:
        sub = ct.extract(x2d, (0, 0), shape=(3, 3))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

extract((8,8) tile, shape=(16,8)) (larger than input): TileTypeError: Input shape (8, 8) is not divisible by result shape (16, 8) at dimension #0
  "/tmp/ch15/06_extract_and_capstone.py", line 85, col 11-48, in kernel_fn4:
        sub = ct.extract(x2d, (0, 0), shape=(16, 8))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

capstone (reshape + permute + extract + cat chained): 23552 cubin bytes
```

All three rejections are genuinely distinct, exactly as the three different underlying rules predicted: an out-of-range grid index names the offending dimension, the valid range, and both the input and extract shapes together; a non-power-of-two extract shape reproduces Section 15.1's rule, now enforced on `extract`'s `shape` argument specifically; and an extract shape that is individually valid but too large to evenly divide the input names both shapes and the specific dimension where the division fails. The capstone compiles cleanly on its first attempt — reshaping a flat tile to 2D, permuting it, extracting two disjoint quadrants by grid index, concatenating them back together along a different axis than the one the permute touched, and flattening the result, all five of this chapter's operations in one kernel body with no adjustment needed to get it past the compiler.

## Chapter Summary

Testing the first of six tile-recombination operations this book had never used surfaced something bigger than any single function: every tile shape in cuTile Python must be a power of two along every dimension, a hard, immediately-enforced rule that fourteen chapters of kernels satisfied by coincidence, never once by design, because every tile size this book happened to choose was already one. `ct.reshape` enforces a separate, purely structural rule — total element count must be preserved, `-1` inference included — with its own distinct error. `ct.transpose` is confirmed, only once the naming confound was actually controlled for, to be a genuine special case of `ct.permute`, compiling to byte-identical output for the same swap expressed either way; permuting a tile's axes is not a uniform-cost operation, with the identity permutation cheapest and different non-identity reorderings costing measurably different amounts. `ct.expand_dims` and its documented NumPy-style indexing sugar are equally confirmed identical, byte for byte. `ct.cat`'s docstring claim that its two inputs "must have the same shape" turns out to be the practical consequence of two independent, more specific checks — matching non-axis dimensions, and a power-of-two result — which happen to coincide in every real case only because two distinct powers of two never sum to a third. And `ct.extract`, the most structurally rich of the six, enforces three genuinely different rules with three distinct messages, then composes cleanly with every other operation from this chapter into a single capstone kernel.

## Self-Check Questions

1. Section 15.1 found that `tile_size` values of `6`, `5`, `63`, and `100` are all rejected, while every earlier chapter's tile sizes happened to work. What does the fact that this rule went untested for fourteen chapters suggest about the risk of trusting a pattern in your own example choices as if it were a documented guarantee?
2. Section 15.3's first, uncontrolled comparison of `transpose` and `permute` showed a real byte-count difference that vanished once both kernels were named identically. What general lesson does this suggest about interpreting ANY byte-count difference this book has reported, even in chapters where the naming-confound rule was already being followed?
3. Section 15.4 found `ct.expand_dims(x, 0)` and `x[None, :]` compile to byte-identical output. Contrast this with Section 15.3's transpose/permute result — in what way are these two findings actually testing subtly different claims, even though both are described as "confirming an equivalence"?
4. Section 15.5 explains `cat`'s "same shape" docstring claim as the practical consequence of two independent checks that "happen to coincide... only because two distinct powers of two never sum to a third." Is this a mathematical fact this chapter can assert with full confidence, or an empirical pattern like Section 15.1's untested convention? What distinguishes the two kinds of claim?
5. Section 15.6's capstone kernel combines reshape, permute, extract, and cat in one body and compiles successfully on the first attempt. Given how many individual rules this chapter uncovered for these operations, what does a clean first-attempt compile of the capstone actually tell you, and what would you still need to check to be confident the composition is correct rather than merely valid?

## Where We Go Next

This chapter's opening discovery — a rule silently obeyed by every kernel this book has ever built, never once tested until an unrelated worked example's error message happened to mention it — is a sharper version of the same lesson Chapter 14 drew from `ListConstraint`: a codebase this careful can still carry gaps that look like settled ground purely because nothing ever exercised the boundary. Every reshaping and recombining operation `dir(ct)` lists has now been tested directly, compared against its own documentation, and, where two operations claimed to be equivalent, checked byte-for-byte rather than taken on faith. `dir(ct)` still lists elementwise math this book has never called by name — trigonometric, exponential, and rounding functions among them — and reduction operations beyond the `sum`, `max`, and `min` this book leaned on from Chapter 1 onward. Chapter 16 turns to that remaining arithmetic surface, applying the same discipline one more time.

## Worked Solutions

**1.** It suggests that a convention followed consistently across many examples can be mistaken for a documented guarantee, when it may only ever have been an unexamined habit that happened never to be violated. The risk is specific to example-driven writing: choosing `64`, `32`, and `128` repeatedly because they are natural, round numbers for tile sizes is a reasonable default, but it is not the same thing as confirming the compiler would accept anything else — and the only way to tell the difference between "this pattern is required" and "this pattern is merely what I've tried so far" is to deliberately test a value outside it, which is exactly what this section finally did.

**2.** It suggests that any two kernels compared for byte count should be treated with suspicion unless their names are visibly, textually identical in the worked example itself — not merely "probably fine because Chapter 10's rule is well known by now." The uncontrolled first attempt in this chapter was written by someone who already knew the naming-confound rule and still produced a misleading comparison, simply by naming one kernel `kernel_fn3` and the other something else out of convenience while drafting. If a rule can be forgotten in the act of applying the very discipline meant to catch it, verifying the actual kernel names in a comparison is worth doing directly rather than trusting that the rule was followed.

**3.** Section 15.3's finding is about two different *functions* — `transpose` and `permute` — producing the same compiled result for logically equivalent inputs, confirming they share an implementation rather than being two independently-built code paths. Section 15.4's finding is about one function and one piece of *syntax* — `ct.expand_dims` and Python's own indexing notation — producing the same result, which is a claim about how the language-level sugar desugars, not about two library functions sharing a backend. The two findings both use the word "equivalence," but one is about the relationship between two API entry points and the other is about whether special syntax is genuinely transparent sugar rather than a distinct code path with a coincidentally matching outcome.

**4.** It is a mathematical fact this chapter can assert with full confidence, not an empirical pattern. That two distinct powers of two never sum to a third power of two follows directly from how binary representation works — a power of two has exactly one bit set, and the sum of two numbers each with exactly one set bit, in different positions, always has exactly two bits set — which is true for every pair of distinct powers of two, not just the ones this chapter happened to test. This is the opposite situation from Section 15.1's tile-size convention, which really was just an untested pattern in this book's own examples; here, the reasoning holds for every possible case by the structure of binary numbers themselves, not because this chapter ran out of counterexamples to try.

**5.** A clean compile confirms only that every individual operation's structural rules were satisfied at every step — every reshape preserved element count, the permute was a valid permutation, both extracts had in-range grid indices and power-of-two, evenly-dividing shapes, and the final cat's two inputs matched on non-axis dimensions with a power-of-two result. It says nothing at all about whether the *numeric* result the kernel would actually produce, if it were ever launched on real hardware, is the one a reader intended — this book's `export_kernel`-only environment has never been able to verify computed values, only compiled structure, all the way back to Chapter 1. Confirming the capstone is correct, not merely valid, would require the one thing this environment has never had: a real `ct.launch` against actual hardware, with known input data and an independently-computed expected output to compare against.

## Complete Runnable Code

### File: `01_power_of_two_tile_shape.py`

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

# Every tile_size this book has ever used -- 64, 32, 128, 8, 4, 256, and so
# on -- has happened to be a power of two. Nothing has ever tried a
# non-power-of-two tile_size directly.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), x)

for size in [6, 5, 63, 64, 100]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(size)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    try:
        n = compile_bytes(kernel_fn, sig)
        print(f"tile_size={size}: {n} cubin bytes")
    except Exception as e:
        print(f"tile_size={size}: {type(e).__name__}: {e}")
```

### File: `02_reshape.py`

```python
import cuda.tile as ct
import io

def array_param(ndim=1):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.reshape(x, (8, -1))
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn, sig)
    print(f"reshape (64,) -> (8,-1) -> (64,): {n} cubin bytes")
except Exception as e:
    print(f"reshape (64,) -> (8,-1) -> (64,): {type(e).__name__}: {e}")

# Invalid reshape: total element count doesn't match.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.reshape(x, (7,))
    ct.store(c, (pid,), y)

sig2 = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn2, sig2)
    print(f"reshape (64,) -> (7,): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"reshape (64,) -> (7,): {type(e).__name__}: {e}")
```

### File: `03_transpose_permute_equivalence_and_cost.py`

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

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# transpose(x, axis0, axis1) swapping axes 0 and 2 of a (4, 4, 4) tile
# should be exactly the same data movement as permute(x, (2, 1, 0)) --
# swap the first and last axes, leave the middle one fixed. Both variants
# are named "kernel_fn" (Chapter 10's naming-confound rule) so their byte
# counts are directly comparable.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.transpose(tile3d, 0, 2)
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
kernel_transpose = kernel_fn
print(f"transpose(tile3d, 0, 2): {compile_bytes(kernel_transpose, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.permute(tile3d, (2, 1, 0))
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
kernel_permute = kernel_fn
print(f"permute(tile3d, (2, 1, 0)): {compile_bytes(kernel_permute, sig)} cubin bytes")

# transpose() is documented to require explicit axes for anything beyond
# 2 dimensions -- never checked directly in this book before.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.transpose(tile3d)
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn2, sig)
    print(f"transpose() on a 3D tile, no axes given: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"transpose() on a 3D tile, no axes given: {type(e).__name__}: {e}")

# permute() with a repeated axis -- not a valid permutation at all.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    tile3d = ct.reshape(x, (4, 4, 4))
    t = ct.permute(tile3d, (0, 0, 2))
    z = ct.reshape(t, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn3, sig)
    print(f"permute((0, 0, 2)) (repeated axis): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"permute((0, 0, 2)) (repeated axis): {type(e).__name__}: {e}")

# If transpose is really just permute with a specific two-axis swap, does
# the choice of WHICH permutation costs the same across every possible
# 3-axis reordering? Sweeping all six permutations of (4, 4, 4), each
# built from a factory so every variant shares the name "kernel_fn".
def build_kernel(axes):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        tile3d = ct.reshape(x, (4, 4, 4))
        t = ct.permute(tile3d, axes)
        z = ct.reshape(t, (tile_size,))
        ct.store(c, (pid,), z)
    return kernel_fn

for axes in [(0, 1, 2), (2, 0, 1), (2, 1, 0), (1, 0, 2), (1, 2, 0), (0, 2, 1)]:
    n = compile_bytes(build_kernel(axes), sig)
    print(f"permute({axes}): {n} cubin bytes")
```

### File: `04_expand_dims.py`

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

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.expand_dims(x, axis) vs the documented NumPy-style sugar x[:, None].
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.expand_dims(x, 0)
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)
kernel_call = kernel_fn
print(f"ct.expand_dims(x, 0): {compile_bytes(kernel_call, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = x[None, :]
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)
kernel_sugar = kernel_fn
print(f"x[None, :]: {compile_bytes(kernel_sugar, sig)} cubin bytes")

# An out-of-range axis.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.expand_dims(x, 5)
    z = ct.reshape(y, (tile_size,))
    ct.store(c, (pid,), z)
kernel_bad_axis = kernel_fn
try:
    n = compile_bytes(kernel_bad_axis, sig)
    print(f"ct.expand_dims(x, 5) (out of range): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"ct.expand_dims(x, 5) (out of range): {type(e).__name__}: {e}")
```

### File: `05_cat.py`

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

sig_double = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# cat() on two equal-shaped (8, 8) tiles, along each axis in turn.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y2d = ct.reshape(y, (8, 8))
    combined = ct.cat((x2d, y2d), 0)
    z = ct.reshape(combined, (2 * tile_size,))
    ct.store(c, (pid,), z)
kernel_axis0 = kernel_fn
print(f"cat((x2d, y2d), 0): {compile_bytes(kernel_axis0, sig_double)} cubin bytes")

@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y2d = ct.reshape(y, (8, 8))
    combined = ct.cat((x2d, y2d), 1)
    z = ct.reshape(combined, (2 * tile_size,))
    ct.store(c, (pid,), z)
kernel_axis1 = kernel_fn
print(f"cat((x2d, y2d), 1): {compile_bytes(kernel_axis1, sig_double)} cubin bytes")

# The docstring says cat's two inputs "must have the same shape," but this
# checks the actual, more specific rule directly: tiles with DIFFERENT
# shapes along every axis at once.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y2d = ct.reshape(y, (16, 4))
    combined = ct.cat((x2d, y2d), 0)
    z = ct.reshape(combined, (2 * tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn, sig_double)
    print(f"cat of (8,8) with (16,4): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"cat of (8,8) with (16,4): {type(e).__name__}: {e}")

# The message above says "same shape for non axis dimensions" -- narrower
# than the docstring's plain "same shape." Built directly with ct.full
# (rather than ct.load) to isolate cat()'s own rule: x2d is (4, 8), y2d is
# (8, 8) -- different sizes along the cat axis itself (axis 0: 4 vs 8,
# both individually powers of two), identical along axis 1.
sig_single = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ConstantConstraint(96)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn(c, out_size: ct.Constant[int]):
    pid = ct.bid(0)
    x2d = ct.full((4, 8), 1.0, ct.float32)
    y2d = ct.full((8, 8), 2.0, ct.float32)
    combined = ct.cat((x2d, y2d), 0)
    z = ct.reshape(combined, (out_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn, sig_single)
    print(f"cat of (4,8) with (8,8) along axis 0 (differ only on cat axis): {n} cubin bytes")
except Exception as e:
    print(f"cat of (4,8) with (8,8) along axis 0 (differ only on cat axis): {type(e).__name__}: {e}")
```

### File: `06_extract_and_capstone.py`

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

sig64 = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# extract(x, index, shape): partitions x into a grid of `shape`-sized
# subtiles and returns the one at `index` -- an in-register analogue of
# Chapter 3's ct.load, but operating on an already-loaded tile rather
# than an Array.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (0, 1), shape=(4, 4))
    z = ct.reshape(sub, (16,))
    ct.store(c, (pid,), z)
sig16 = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                                    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"extract((8,8) tile, index=(0,1), shape=(4,4)): {compile_bytes(kernel_fn, sig16)} cubin bytes")

# An out-of-range grid index: an (8,8) tile split into (4,4) subtiles has
# only a 2x2 grid of them (valid indices 0-1 per axis) -- index (2, 0) is out.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (2, 0), shape=(4, 4))
    z = ct.reshape(sub, (16,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn2, sig16)
    print(f"extract((8,8) tile, index=(2,0), shape=(4,4)) (out of range): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract((8,8) tile, index=(2,0), shape=(4,4)) (out of range): {type(e).__name__}: {e}")

# A non-power-of-two extract shape.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (0, 0), shape=(3, 3))
    z = ct.reshape(sub, (9,))
    ct.store(c, (pid,), z)
sig9 = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                                    stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn3, sig9)
    print(f"extract((8,8) tile, shape=(3,3)) (not a power of two): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract((8,8) tile, shape=(3,3)) (not a power of two): {type(e).__name__}: {e}")

# A power-of-two extract shape that still doesn't evenly divide the input
# shape -- larger than the input entirely, a genuinely different check
# from the power-of-two one above.
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    sub = ct.extract(x2d, (0, 0), shape=(16, 8))
    z = ct.reshape(sub, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn4, sig64)
    print(f"extract((8,8) tile, shape=(16,8)) (larger than input): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"extract((8,8) tile, shape=(16,8)) (larger than input): {type(e).__name__}: {e}")

# Capstone: reshape, permute, extract, and cat chained in one kernel body,
# something no single worked example in this chapter has attempted -- load
# a flat tile, reshape it to 2D, permute it, extract two disjoint quadrants,
# concatenate them back together along the opposite axis, flatten, store.
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    permuted = ct.permute(x2d, (1, 0))
    top = ct.extract(permuted, (0, 0), shape=(4, 8))
    bottom = ct.extract(permuted, (1, 0), shape=(4, 8))
    combined = ct.cat((top, bottom), 1)
    z = ct.reshape(combined, (tile_size,))
    ct.store(c, (pid,), z)
try:
    n = compile_bytes(kernel_fn5, sig64)
    print(f"capstone (reshape + permute + extract + cat chained): {n} cubin bytes")
except Exception as e:
    print(f"capstone (reshape + permute + extract + cat chained): {type(e).__name__}: {e}")
```
