# Chapter 16: Scans, Reductions, and a Second Confound

> "Renaming a file should not change what it compiles to. That assumption held through fifteen chapters of this book — until preparing this chapter's own capstone example, giving a probe script its final, more descriptive filename in place of a disposable placeholder, changed its compiled byte count by exactly 128 bytes. Nothing about the kernel had changed. Only its filename had."

**What you will understand by the end of this chapter:**

- That the compiled cubin's size depends not only on the kernel function's own Python name (Chapter 10) but on the length of the full `__file__` path Python resolves at compile time — a second, independent confound, with a real, precisely-located single-character-boundary jump whose exact threshold length depends on the kernel's own content
- What this second confound does, and does not, undermine about the byte counts this book has already published, and why same-file comparisons — the overwhelming majority of this book's claims — are unaffected by it
- `ct.argmax`/`ct.argmin`: index-returning reductions, their `axis`/`keepdims` behavior, and the documented tuple-of-axes restriction confirmed as a real, immediate `TileTypeError`
- `ct.cumsum`/`ct.cumprod`: inclusive prefix-scan reductions along one axis, the `reverse` flag, and that the choice of axis itself has a measurable compiled cost even for a square, symmetric tile
- `ct.prod` as the multiplicative counterpart to `ct.sum` — byte-identical to it, under Chapter 10's naming control, for the same reduction shape — and Chapter 8's `RoundingMode` "float types only" restriction confirmed to extend to reduction operations
- `ct.scan(x, axis, func, identity)`: the general, user-supplied-function inclusive scan that `cumsum` is documented as one specific instance of, confirmed byte-identical to the built-in under naming control
- `ct.reduce(x, axis, func, identity)`: the equally general reduction that `sum` is documented as one specific instance of, confirmed the same way
- That a custom `ct.scan`-based running maximum — an operation with no named equivalent anywhere in cuTile Python — composes cleanly with `argmax` and a custom `reduce` in a single kernel body

**What you need to know first:**

- Chapter 10's naming-confound rule: kernel function names must match exactly across any two kernels being compared for compiled size.
- Chapter 8's `RoundingMode` and `flush_to_zero` material — this chapter only re-checks whether that restriction generalizes to a reduction context, rather than reintroducing it from scratch.
- Chapter 15's power-of-two tile-shape rule, still enforced on every `reshape` in this chapter's kernels.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow as every chapter before it.

## 16.1 A Renamed File, a Different Byte Count `[FOUNDATIONAL]`

### Intuition

Every byte-count comparison since Chapter 10 has controlled for one confound: the compiled cubin embeds the kernel function's own Python `__name__`, so two kernels compared for size have to share a name or the comparison is meaningless. That rule was applied carefully while drafting this chapter's own capstone example (Section 16.6) — right up until the probe script's placeholder filename was swapped for the descriptive name it would actually carry in the finished book, and the reported byte count changed. The kernel function's name had not changed. Its signature had not changed. Only the string Python records in `__file__` had.

### Background

If the source file's own path length affects the compiled artifact, every prior chapter's reported numbers are, strictly speaking, a joint property of the kernel code and of wherever this book's sandbox happened to place that code — not of the kernel code alone. Before drawing any conclusion from that, the effect needs to be isolated directly: one kernel, copied byte-for-byte across files whose names differ only in length, each compiled in its own subprocess so that each process sees a genuinely different `__file__` value.

The script below does exactly that, and does it without hard-coding a specific length anywhere. It compiles itself as-is first, then copies itself under a filename 200 characters longer and compiles that copy in a fresh subprocess, then binary-searches between the two lengths to find the exact character at which the compiled size changes. Locating the boundary this way, rather than asserting a fixed number, is deliberate: the boundary's exact position depends on how long this file's own absolute path already happens to be on whatever machine runs it, so the only way to report it honestly is to have the script discover its own answer.

### Worked Example 16.1.1 — the same kernel, at different `__file__` lengths

```python
import cuda.tile as ct
import io
import os
import shutil
import subprocess
import sys

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), x)

def compile_trivial():
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_fn, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

if "--worker" in sys.argv:
    # Worker mode: report this process's own __file__ length and the
    # resulting compiled size, then exit.
    print(f"{len(__file__)}\t{compile_trivial()}")
    sys.exit(0)

# Genuine, unmodified compile of this exact file, wherever it happens to live.
this_size = compile_trivial()
print(f"this file (__file__ length {len(__file__)}): {this_size} cubin bytes")

# Chapter 10 found the compiled cubin embeds the KERNEL FUNCTION's own
# Python name. Does the SOURCE FILE's own path matter too? Copy this
# script, byte for byte, to a filename 200 characters longer than its
# own, in the same directory, and compile the copy in a fresh subprocess.
here = os.path.dirname(os.path.abspath(__file__)) or "."
long_name = os.path.join(here, "x" * 196 + "_confound.py")
shutil.copyfile(__file__, long_name)
result = subprocess.run([sys.executable, long_name, "--worker"], capture_output=True, text=True)
long_len_s, long_size_s = result.stdout.strip().split("\t")
os.remove(long_name)
print(f"identical script, __file__ length {long_len_s}: {long_size_s} cubin bytes")

# The jump is real -- and it lands on an exact single-character boundary,
# not a gradual drift. Binary search between this file's own length and
# the much-longer copy's length to find that boundary. No fixed length is
# assumed in advance, since it depends on how long this file's own
# absolute path already happens to be.
def compiled_size_at_length(total_len):
    prefix = here + os.sep
    pad = total_len - len(prefix) - len(".py")
    fname = prefix + "y" * pad + ".py"
    shutil.copyfile(__file__, fname)
    result = subprocess.run([sys.executable, fname, "--worker"], capture_output=True, text=True)
    reported_len, size = result.stdout.strip().split("\t")
    os.remove(fname)
    return int(reported_len), int(size)

lo, lo_size = len(os.path.abspath(__file__)), this_size
hi, hi_size = int(long_len_s), int(long_size_s)
while hi - lo > 1:
    mid = (lo + hi) // 2
    mid_len, mid_size = compiled_size_at_length(mid)
    if mid_size == lo_size:
        lo, lo_size = mid_len, mid_size
    else:
        hi, hi_size = mid_len, mid_size
print(f"boundary located: length {lo} -> {lo_size} bytes, "
      f"length {hi} -> {hi_size} bytes (jump of {hi_size - lo_size} bytes)")
```

Genuinely run:

```
this file (__file__ length 41): 20896 cubin bytes
identical script, __file__ length 218: 21152 cubin bytes
boundary located: length 69 -> 20896 bytes, length 70 -> 21024 bytes (jump of 128 bytes)
```

### Discussion

Three things are genuinely established here. First, the effect is real: a byte-for-byte identical kernel, copied to a longer filename and compiled in a separate process, produces a larger cubin — 20896 bytes at this file's own 41-character path in this book's sandbox, 21152 bytes at a path 200 characters longer. Second, the transition between sizes is not a gradual drift with length; it happens at one exact character. In this sandbox, at this file's own directory, length 69 compiles to 20896 bytes and length 70 — one character longer — compiles to 21024 bytes, a clean 128-byte jump. Third, the far-away copy did not simply land on that same post-jump size: 21152 bytes is larger again than 21024 bytes, meaning there is at least one further boundary beyond the first one, not characterized here, only confirmed to exist. A separate probe kernel combining `cumsum`, `argmax`, and `add` — structurally different from this trivial load-and-store kernel — showed the same phenomenon with the same 128-byte jump magnitude, but at a different threshold length (between 132 and 133 characters in that kernel's case, rather than 69 and 70), confirming the mechanism generalizes across kernel content while its exact trigger point does not.

**What this means for this book.** Every compiled byte count this book has reported, back to Chapter 1, was measured at a specific `__file__` length determined by this sandbox's own `/tmp/chN/` directory convention. Reproducing this book's numbers exactly, from a different absolute path — this repository's real location on a reader's own machine, for instance — is not guaranteed, because that path's length could sit on a different side of a boundary this chapter has only shown exists, not mapped in full. This does not, however, undermine the comparisons this book has actually been making. Every naming-controlled comparison in this book so far — Chapter 10 onward — compiles both variants from within the *same* running script, which means both variants share the exact same `__file__` value and the exact same length. The confound this section isolates only bears on comparing an absolute byte count recorded in one file against an absolute byte count recorded in a different file, or on a reader expecting this book's specific numbers to reproduce verbatim from a different directory depth. It does not touch a same-file, same-run comparison like Section 16.4's `scan`-versus-`cumsum` check or Section 16.5's `reduce`-versus-`sum` check, where both kernels are compiled by the same process at the same `__file__` length and only the operation itself differs.

One retroactive note follows directly from this: Section 16.3 compiles a `cumsum(x2d, axis=0)` kernel using the exact same kernel-body code that Chapter 15's own probe work first used, and reports a different byte count for it — 24064 bytes here, versus 23936 bytes when that comparison was first drafted under a shorter filename during this chapter's own research. Nothing about the kernel changed between those two measurements. The file it lived in did.

## 16.2 Argmax and Argmin: Index-Returning Reductions

### Intuition

Every reduction this book has used since Chapter 1 — `sum`, `max`, `min` — returns a *value*: the largest element, the total, the smallest element. `argmax` and `argmin` return a *position* instead: the index at which that extreme value occurs.

### Background

`ct.argmax(x, axis=None, *, keepdims=False)` and `ct.argmin(x, axis=None, *, keepdims=False)` share the same signature shape as this book's earlier reductions, with `axis=None` collapsing the whole tile to a single index and an explicit axis collapsing just that dimension. Their documentation states two rules worth testing directly rather than assuming: that ties are broken by returning the smallest index among the tied positions, and — despite the signature's own type annotation allowing `axis` to be `None | const int | tuple[const int, ...]` — that a tuple of axes is specifically *not* supported for `argmax`/`argmin`, unlike the more general tuple support `sum` and `prod` allow. The tie-breaking behavior can only be confirmed by actually computing a result, which this book's `export_kernel`-only environment has never been able to do (Chapter 1's founding limitation still applies); the tuple-axis restriction, by contrast, is a compile-time check this book can test directly.

### Worked Example 16.2.1 — full reduction, axis reduction, and the tuple-axis rejection

```python
import cuda.tile as ct
import io

def array_param(size):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# argmax(x, axis=None) reduces the whole tile to a single scalar index.
sig_scalar_out = ct.compilation.KernelSignature(
    [array_param(64), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    idx = ct.argmax(x2d, None)
    ct.store(c, (pid,), ct.astype(idx, ct.float32))
try:
    n = compile_bytes(kernel_fn, sig_scalar_out)
    print(f"argmax(x2d, None): {n} cubin bytes")
except Exception as e:
    print(f"argmax(x2d, None): {type(e).__name__}: {e}")

# argmax(x, axis=1, keepdims=True) reduces one axis, keeping shape (8, 1).
sig_axis_out = ct.compilation.KernelSignature(
    [array_param(64), array_param(8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    idx = ct.argmax(x2d, 1, keepdims=True)
    flat = ct.reshape(idx, (8,))
    ct.store(c, (pid,), ct.astype(flat, ct.float32))
try:
    n = compile_bytes(kernel_fn2, sig_axis_out)
    print(f"argmax(x2d, axis=1, keepdims=True): {n} cubin bytes")
except Exception as e:
    print(f"argmax(x2d, axis=1, keepdims=True): {type(e).__name__}: {e}")

# The docstring says argmax/argmin do NOT support a tuple of axes.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    idx = ct.argmax(x2d, (0, 1))
    ct.store(c, (pid,), ct.astype(idx, ct.float32))
try:
    n = compile_bytes(kernel_fn3, sig_scalar_out)
    print(f"argmax(x2d, (0, 1)) (tuple axis): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"argmax(x2d, (0, 1)) (tuple axis): {type(e).__name__}: {e}")
```

Genuinely run:

```
argmax(x2d, None): 25088 cubin bytes
argmax(x2d, axis=1, keepdims=True): 25856 cubin bytes
argmax(x2d, (0, 1)) (tuple axis): TileTypeError: Invalid argument "axis" of argmax(): Expected an integer constant, but given value has type Tuple[Tile[int32,()],Tile[int32,()]]
  "/tmp/ch16/02_argmax_argmin.py", line 58, col 11-32, in kernel_fn3:
        idx = ct.argmax(x2d, (0, 1))
              ^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

Both accepted forms compile without incident — a full reduction to one scalar index, and an axis-1 reduction that keeps a `(8, 1)` shape before this example reshapes it down to `(8,)` for storage. Since these two kernels use different function names and different output shapes, their byte counts are not a controlled comparison against each other in Chapter 10's sense; each is simply confirmed to compile as documented. The tuple-axis rejection is the more interesting result: the error message frames the failure in terms of the *type system*, not a special-cased runtime check for `argmax` specifically — a tuple of two 0-dimensional int32 tiles is rejected because `axis` is typed to expect "an integer constant," full stop, with no branch in the message acknowledging that a tuple would be meaningful for a different reduction. This suggests the tuple-axis restriction for `argmax`/`argmin` is enforced by giving `axis` a narrower expected type at the entry point for these two functions specifically, rather than by accepting any axis specification generically and rejecting tuples with a dedicated, argmax-aware message afterward.

## 16.3 Cumsum, Cumprod, and Prod: The Rest of the Named Reductions

### Intuition

`ct.sum`, `ct.max`, and `ct.min` have carried this book's reduction needs since Chapter 1. `dir(ct)` lists three more reduction-family operations never used in a worked example before now: `ct.cumsum` and `ct.cumprod`, which reduce a tile *incrementally* rather than all at once, and `ct.prod`, an ordinary all-at-once reduction that simply multiplies instead of adding.

### Background

`ct.cumsum(x, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False)` and `ct.cumprod(...)` share this signature shape. Unlike `sum`, `max`, `min`, `argmax`, and `argmin`, there is no `axis=None` option here — a cumulative scan is inherently a per-axis operation, since scanning "all axes at once" is not a well-defined idea the way reducing all axes to one value is. `reverse=True` runs the scan from the other end of the axis. Both functions accept the same `rounding_mode`/`flush_to_zero` parameters Chapter 8 already covered in depth for elementwise operations; this section's third worked example re-checks, briefly, whether that "float types only" restriction on `rounding_mode` generalizes to a reduction context via `ct.prod`, rather than repeating Chapter 8's fuller treatment. `ct.prod(x, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False)` mirrors `ct.sum`'s own signature almost exactly, swapping addition for multiplication.

### Worked Example 16.3.1 — cumsum, cumprod, axis choice, and prod

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

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 1)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumsum(x2d, axis=1): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 1, reverse=True)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumsum(x2d, axis=1, reverse=True): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumprod(x2d, 1)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumprod(x2d, axis=1): {compile_bytes(kernel_fn3, sig)} cubin bytes")

# cumsum and cumprod along axis 0 instead of axis 1 -- does the axis choice
# itself cost anything for a square (8, 8) tile?
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 0)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumsum(x2d, axis=0): {compile_bytes(kernel_fn4, sig)} cubin bytes")

# prod parallels sum, but multiplicatively. Naming-controlled comparison
# (Chapter 10) against ct.sum for the identical full-reduction shape.
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    r = ct.sum(x2d, None)
    ct.store(c, (pid,), ct.full((tile_size,), r, ct.float32))
kernel_sum = kernel_fn5
print(f"sum(x2d, None): {compile_bytes(kernel_sum, sig)} cubin bytes")

@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    r = ct.prod(x2d, None)
    ct.store(c, (pid,), ct.full((tile_size,), r, ct.float32))
kernel_prod = kernel_fn5
print(f"prod(x2d, None): {compile_bytes(kernel_prod, sig)} cubin bytes")

# The docstring says rounding_mode is "only supported for float types" --
# never checked directly against an integer-dtyped reduction before.
def int_array_param():
    return ct.compilation.ArrayConstraint(
        ct.int32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )
sig_int = ct.compilation.KernelSignature(
    [int_array_param(), int_array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    r = ct.prod(x2d, None, rounding_mode=ct.RoundingMode.RN)
    ct.store(c, (pid,), ct.full((tile_size,), r, ct.int32))
try:
    n = compile_bytes(kernel_fn6, sig_int)
    print(f"prod(int32 tile, rounding_mode=RN): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"prod(int32 tile, rounding_mode=RN): {type(e).__name__}: {e}")
```

Genuinely run:

```
cumsum(x2d, axis=1): 24832 cubin bytes
cumsum(x2d, axis=1, reverse=True): 24960 cubin bytes
cumprod(x2d, axis=1): 24832 cubin bytes
cumsum(x2d, axis=0): 24064 cubin bytes
sum(x2d, None): 24576 cubin bytes
prod(x2d, None): 24576 cubin bytes
prod(int32 tile, rounding_mode=RN): TileTypeError: Rounding mode can only be used for unrestricted float types, but got int32
  "/tmp/ch16/03_cumsum_cumprod_and_prod.py", line 96, col 9-60, in kernel_fn6:
        r = ct.prod(x2d, None, rounding_mode=ct.RoundingMode.RN)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

`cumsum` and `cumprod` compile to the identical 24832 bytes for the same axis-1 scan on this square tile — a coincidence worth noting but not over-reading, since `kernel_fn` and `kernel_fn3` are different function names and this is not a Chapter-10-controlled comparison. `reverse=True` does cost something: 24960 bytes against 24832, a real difference for scanning in the opposite direction along the same axis. Axis choice on a square `(8, 8)` tile costs something too — `cumsum` along axis 0 compiles to 24064 bytes, smaller than the 24832 bytes for axis 1 on this same tile shape, echoing Chapter 15's finding that mathematically equivalent axis choices are not uniformly priced by the compiler even when the tile itself is perfectly symmetric.

`sum` and `prod`, compiled under Chapter 10's naming control (both variants defined as `kernel_fn5`, reassigned in turn, exactly as earlier chapters have done whenever more than one variant needs to share a name), compile to byte-identical 24576-byte cubins for the same full-reduction shape — the same relationship this chapter already found between `cumsum` and its general `ct.scan` equivalent, and will find again in Section 16.5 between `sum` and `ct.reduce`. The rounding-mode rejection on an integer-dtyped `prod` reproduces Chapter 8's "float types only" message exactly, this time triggered through a reduction rather than an elementwise operation — confirming the restriction is enforced at the level of `RoundingMode` itself, not re-implemented separately for every operation that accepts one.

## 16.4 `ct.scan`: The General Mechanism Behind Cumsum

### Intuition

`cumsum` always adds. `cumprod` always multiplies. `ct.scan(x, axis, func, identity)` takes the combining function itself as an argument, and the documentation states plainly that `cumsum` is exactly `ct.scan` with `func=operator.add`, `identity=0`.

### Background

If that documented relationship is accurate, compiling a `cumsum(x2d, 1)` kernel and a `ct.scan(x2d, 1, operator.add, 0.0)` kernel — both defined under the identical function name `kernel_fn`, per Chapter 10's rule — should produce byte-identical cubins. `ct.scan` additionally accepts a `reverse` flag matching `cumsum`'s own, and can operate on a tuple of tiles at once with a matching tuple of combining functions and identities, a generality this section's worked example does not need in order to test the specific, documented `cumsum` equivalence.

### Worked Example 16.4.1 — `ct.cumsum` versus a hand-built `ct.scan`

```python
import cuda.tile as ct
import io
import operator

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

# ct.scan(x, axis, func, identity) applies a custom, user-supplied
# combining function as an inclusive prefix scan -- cumsum is documented
# as exactly this pattern with func=operator.add, identity=0. Naming-
# controlled (Chapter 10) comparison against the built-in ct.cumsum.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 1)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
kernel_builtin = kernel_fn
print(f"ct.cumsum(x2d, 1): {compile_bytes(kernel_builtin, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.scan(x2d, 1, operator.add, 0.0)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
kernel_custom = kernel_fn
try:
    n = compile_bytes(kernel_custom, sig)
    print(f"ct.scan(x2d, 1, operator.add, 0.0): {n} cubin bytes")
except Exception as e:
    print(f"ct.scan(x2d, 1, operator.add, 0.0): {type(e).__name__}: {e}")
```

Genuinely run:

```
ct.cumsum(x2d, 1): 24832 cubin bytes
ct.scan(x2d, 1, operator.add, 0.0): 24832 cubin bytes
```

### Discussion

Byte-identical, under Chapter 10's naming control — both kernels are named `kernel_fn` at definition time, then reassigned to `kernel_builtin`/`kernel_custom` for clarity, exactly as this book has done since Chapter 10 whenever a comparison needs two variants under one shared name. This confirms the documentation's own framing is not just a descriptive analogy but a literal implementation fact from the compiler's perspective: `cumsum` produces exactly the code a hand-written `operator.add` scan would produce, not merely code that computes the same values by a different path.

## 16.5 `ct.reduce`: The General Mechanism Behind Sum

### Intuition

Section 16.4 found `ct.scan` generalizes `cumsum`. `ct.reduce(x, axis, func, identity)` is documented as the equivalent generalization one level up: `sum(x, axis)` is exactly `ct.reduce` with `func=operator.add`, `identity=0`.

### Background

`ct.reduce` additionally takes `keepdims`, matching `sum`'s own parameter of the same name. To keep both variants' output shapes identical without resorting to a mismatched-shape padding trick, this worked example reduces axis 1 of an `(8, 8)` tile down to `(8, 1)` with `keepdims=True`, then broadcasts that back up to `(8, 8)` with `ct.broadcast_to` before storing — a pattern that lets both the built-in and custom variants share one output array shape cleanly.

### Worked Example 16.5.1 — `ct.sum` versus a hand-built `ct.reduce`

```python
import cuda.tile as ct
import io
import operator

def array_param(dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
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

# ct.reduce(x, axis, func, identity) applies a custom, user-supplied
# combining function along one axis -- sum(x, axis) is documented as
# exactly this pattern with func=operator.add, identity=0. Both reduce
# axis 1 of an (8, 8) tile down to shape (8, 1) with keepdims=True, then
# broadcast that back up to fill the (8, 8) output tile so both variants
# can share one output array shape. Naming-controlled (Chapter 10)
# comparison.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.sum(x2d, 1, keepdims=True)
    broadcasted = ct.broadcast_to(y, (8, 8))
    ct.store(c, (pid,), ct.reshape(broadcasted, (tile_size,)))
kernel_builtin = kernel_fn
print(f"ct.sum(x2d, axis=1, keepdims=True): {compile_bytes(kernel_builtin, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True)
    broadcasted = ct.broadcast_to(y, (8, 8))
    ct.store(c, (pid,), ct.reshape(broadcasted, (tile_size,)))
kernel_custom = kernel_fn
try:
    n = compile_bytes(kernel_custom, sig)
    print(f"ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True): {n} cubin bytes")
except Exception as e:
    print(f"ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True): {type(e).__name__}: {e}")
```

Genuinely run:

```
ct.sum(x2d, axis=1, keepdims=True): 26624 cubin bytes
ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True): 26624 cubin bytes
```

### Discussion

Byte-identical again, under the same naming control as Section 16.4. `sum` and `ct.reduce(..., operator.add, 0.0)` produce exactly the same compiled code for the same reduction, exactly as `cumsum` and `ct.scan(..., operator.add, 0.0)` did. Between them, Sections 16.4 and 16.5 confirm both halves of the documented relationship this chapter opened with: the two general mechanisms, `scan` and `reduce`, are not merely convenient user-facing wrappers that happen to compute the right values through some other internal path — they compile to the identical artifact a reader would get by calling the named, specific operation directly.

## 16.6 Capstone: A Custom Scan With No Named Equivalent

### Intuition

`dir(ct)` names `cumsum` and `cumprod` as its only two built-in scans. There is no `cummax` — no named operation anywhere in cuTile Python computes a running maximum. Section 16.4 already showed `ct.scan` can reproduce `cumsum` exactly; this capstone uses that same generality to build an operation that has no named counterpart to reproduce at all, then combines it with `argmax` (Section 16.2) and a custom `ct.reduce` (Section 16.5) in one kernel body.

### Background

This is also the file whose renaming, during this chapter's own drafting, first surfaced Section 16.1's path-length confound — its byte count below reflects its final, actual filename in this book's canonical `06_capstone_running_max_and_custom_ops.py`, not the disposable placeholder name it briefly carried while this chapter's other worked examples were still being drafted.

### Worked Example 16.6.1 — running maximum, peak index, and a custom row-sum in one kernel

```python
import cuda.tile as ct
import io
import operator

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

# cuTile Python has no built-in "running maximum" scan -- only cumsum and
# cumprod are named. ct.scan's generality builds one anyway, something no
# combination of this book's named operations could do directly. This
# capstone combines it with argmax (Section 16.2) and a custom ct.reduce
# (Section 16.5) in a single kernel body.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    running_max = ct.scan(x2d, 1, ct.maximum, -3.0e38)
    peak_index = ct.argmax(x2d, None)
    row_totals = ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True)
    combined = ct.add(running_max, ct.broadcast_to(row_totals, (8, 8)))
    combined = ct.add(combined, ct.astype(peak_index, ct.float32))
    ct.store(c, (pid,), ct.reshape(combined, (tile_size,)))

try:
    n = compile_bytes(kernel_fn, sig)
    print(f"capstone (scan running-max + argmax + custom reduce): {n} cubin bytes")
except Exception as e:
    print(f"capstone (scan running-max + argmax + custom reduce): {type(e).__name__}: {e}")
```

Genuinely run:

```
capstone (scan running-max + argmax + custom reduce): 33792 cubin bytes
```

### Discussion

The kernel compiles cleanly on the first attempt: `ct.scan` with `ct.maximum` as the combining function and a large negative float as the identity element builds a running-maximum scan with no adjustment needed, `argmax` reduces the same source tile to a single peak index, and a custom `ct.reduce` computes per-row totals that are broadcast back up to `(8, 8)` before all three results are combined and stored. Six chapters' worth of operations — reshape, scan, argmax, reduce, broadcast_to, add — compose without conflict, echoing Chapter 15's own capstone finding that a clean first-attempt compile confirms every individual operation's structural rules were satisfied, and nothing more: it says nothing about whether the *values* this kernel would actually produce on real hardware are the ones a reader intended, since this book's `export_kernel`-only environment has never been able to check computed results, only compiled structure, all the way back to Chapter 1.

## Chapter Summary

Preparing this chapter's own capstone example surfaced a second naming confound this book had never suspected: the compiled cubin's size depends not only on the kernel function's own name (Chapter 10) but on the length of the source file's own `__file__` path, with a real, single-character-boundary jump whose exact trigger length is specific to the kernel's own content. That confound is real and worth knowing about when comparing absolute byte counts across different files or reproducing this book's numbers from a different directory depth, but it does not touch this book's actual comparisons, which — Chapter 10 onward — have always compiled both compared variants from within the same running script at the same `__file__` length. With that scoped and understood, this chapter completed the reduction-family operations `dir(ct)` still listed as untested: `argmax`/`argmin` return an index rather than a value, with a documented tuple-of-axes restriction confirmed as a real, immediate type-level rejection rather than a runtime special case; `cumsum`/`cumprod` scan incrementally along one axis, with axis choice itself carrying a measurable compiled cost even on a symmetric tile; `prod` mirrors `sum` byte-for-byte under naming control, and Chapter 8's rounding-mode restriction was confirmed to generalize cleanly to a reduction context. Most significantly, `ct.scan` and `ct.reduce` were both confirmed, byte-for-byte, to be the literal general mechanisms their documentation claims `cumsum` and `sum` are specific instances of — not just descriptively similar operations that happen to compute the same values by some other internal path — and that generality was enough to build a running-maximum scan with no named equivalent anywhere in cuTile Python, composing cleanly with two of this chapter's other operations in one capstone kernel.

## Self-Check Questions

1. Section 16.1 found that renaming a file — with no change to the kernel function's name, body, or signature — changed the compiled byte count. Why does this NOT mean every naming-confound-controlled comparison this book has made since Chapter 10 needs to be redone?
2. Section 16.1's script locates the exact boundary length with a binary search rather than the chapter simply stating a fixed number like "70 characters." Why is a fixed number the wrong thing to assert here, given what Worked Example 16.1.1 actually found about the far-away copy's byte count?
3. Section 16.2 found that `argmax`'s tuple-axis rejection is phrased as a type error ("Expected an integer constant") rather than an argmax-specific message about tuples not being supported. What does that phrasing suggest about how the restriction is likely implemented, compared to a hypothetical alternative where `argmax` accepted any axis specification and then checked for tuples itself?
4. Sections 16.4 and 16.5 each found a custom, user-supplied-function operation (`ct.scan`, `ct.reduce`) compiling byte-for-byte identically to a named built-in (`cumsum`, `sum`). What distinguishes this kind of equivalence from Section 16.3's finding that `cumsum` and `cumprod` happen to compile to the same byte count for the same axis?
5. Section 16.6's capstone combines `ct.scan`, `ct.argmax`, and `ct.reduce` in one kernel that compiles successfully on the first attempt. In what specific way does this differ from a demonstration that the kernel's running maximum, peak index, and row totals are computed correctly?

## Where We Go Next

Chapter 15 closed by pointing to the elementwise math this book had never called by name — trigonometric, exponential, and rounding functions among `dir(ct)`'s remaining entries — as this chapter's destination. This chapter's own drafting process changed that plan: a renamed file producing a different compiled artifact was too significant a finding, with implications for every byte count this book has ever reported, to set aside in favor of the originally planned material. Completing the reduction family that `argmax`, `cumsum`, `prod`, `scan`, and `reduce` belong to filled the rest of this chapter alongside it. `dir(ct)` still lists that same elementwise math surface untouched — trigonometric, exponential, logarithmic, and rounding functions this book has never invoked by name, despite `RoundingMode` itself already being familiar since Chapter 8. Chapter 17 turns to that surface next, carrying forward both of this book's now-confirmed naming confounds — kernel function name, and source file path length — as standing discipline for every comparison it makes.

## Worked Solutions

**1.** Every naming-confound-controlled comparison this book has made compiles both compared kernels from within the *same* running script — meaning both kernels are compiled with the exact same `__file__` value and therefore the exact same path length. Section 16.1's confound only produces a difference when the `__file__` length itself differs between the two things being compared, which never happens within a single script's own execution; it only bears on comparing an absolute number recorded in one file against an absolute number recorded in a different file, which this book's naming-confound comparisons have never done.

**2.** The far-away copy — 200 characters longer than baseline — compiled to 21152 bytes, not the 21024 bytes found just past the first boundary. If the effect were a single one-time step, both would have matched. Since they didn't, there is evidently at least a second boundary somewhere between the first jump and the 200-characters-longer copy, meaning "the boundary is at 70 characters" would be true only for this specific kernel at this specific starting length, and false as a general statement about the mechanism — asserting a fixed number would overstate how much this chapter actually characterized, when what was actually confirmed is that boundaries exist and produce clean 128-byte jumps, not that there is exactly one universal trigger length.

**3.** A type error phrased in terms of what `axis`'s expected type is ("Expected an integer constant") suggests `argmax`/`argmin` give the `axis` parameter a narrower declared type than the more permissive `None | const int | tuple[const int, ...]` the Python-level signature and general reduction functions like `sum`/`prod` suggest — rejecting a tuple before any argmax-specific logic ever runs, simply because a tuple does not satisfy that narrower type. A hypothetical alternative implementation — accept any axis form generically, then check specifically "is this argmax and is axis a tuple" — would more likely have produced a message naming argmax explicitly and explaining why tuples aren't supported for it, rather than a generic type-mismatch message that reads the same way any other integer-constant type violation would.

**4.** Section 16.3's `cumsum`-versus-`cumprod` byte-count match is not a controlled comparison in Chapter 10's sense at all — the two kernels have different function names (`kernel_fn` and `kernel_fn3`), so any conclusion drawn from their matching size would be exactly the kind of naming-confound mistake Chapter 10 first warned against; the shared byte count there is simply an unexplained coincidence, correctly left uninterpreted in the Discussion. Sections 16.4 and 16.5's equivalences are genuine controlled comparisons — both variants in each case are defined under the identical name `kernel_fn` before being reassigned to descriptive names, exactly satisfying Chapter 10's rule — so their byte-for-byte match is actual evidence that the general and specific operations compile to identical code, not a coincidence that happens to survive an uncontrolled comparison.

**5.** A clean first-attempt compile confirms only that every operation's own structural rules were satisfied at every step of this kernel — the scan's combining function and identity were well-typed, the axis argument to argmax was a valid integer, the reduce's output shape broadcast correctly, and so on. It says nothing about whether the specific numeric values this kernel would produce — the actual running maximum, peak index, and row totals for any real input data — are the values a reader intended, because this book's `export_kernel`-only, driver-free environment has never been able to run a kernel against real data and check its output, only compile it and inspect the resulting artifact's structure and size. Confirming correctness rather than mere validity would require `ct.launch` against actual hardware, with known inputs and an independently-computed expected output — the one thing no chapter of this book, including this one, has ever had available.

## Complete Runnable Code

### File: `01_file_path_length_confound.py`

```python
import cuda.tile as ct
import io
import os
import shutil
import subprocess
import sys

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), x)

def compile_trivial():
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_fn, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

if "--worker" in sys.argv:
    # Worker mode: report this process's own __file__ length and the
    # resulting compiled size, then exit.
    print(f"{len(__file__)}\t{compile_trivial()}")
    sys.exit(0)

# Genuine, unmodified compile of this exact file, wherever it happens to live.
this_size = compile_trivial()
print(f"this file (__file__ length {len(__file__)}): {this_size} cubin bytes")

# Chapter 10 found the compiled cubin embeds the KERNEL FUNCTION's own
# Python name. Does the SOURCE FILE's own path matter too? Copy this
# script, byte for byte, to a filename 200 characters longer than its
# own, in the same directory, and compile the copy in a fresh subprocess.
here = os.path.dirname(os.path.abspath(__file__)) or "."
long_name = os.path.join(here, "x" * 196 + "_confound.py")
shutil.copyfile(__file__, long_name)
result = subprocess.run([sys.executable, long_name, "--worker"], capture_output=True, text=True)
long_len_s, long_size_s = result.stdout.strip().split("\t")
os.remove(long_name)
print(f"identical script, __file__ length {long_len_s}: {long_size_s} cubin bytes")

# The jump is real -- and it lands on an exact single-character boundary,
# not a gradual drift. Binary search between this file's own length and
# the much-longer copy's length to find that boundary. No fixed length is
# assumed in advance, since it depends on how long this file's own
# absolute path already happens to be.
def compiled_size_at_length(total_len):
    prefix = here + os.sep
    pad = total_len - len(prefix) - len(".py")
    fname = prefix + "y" * pad + ".py"
    shutil.copyfile(__file__, fname)
    result = subprocess.run([sys.executable, fname, "--worker"], capture_output=True, text=True)
    reported_len, size = result.stdout.strip().split("\t")
    os.remove(fname)
    return int(reported_len), int(size)

lo, lo_size = len(os.path.abspath(__file__)), this_size
hi, hi_size = int(long_len_s), int(long_size_s)
while hi - lo > 1:
    mid = (lo + hi) // 2
    mid_len, mid_size = compiled_size_at_length(mid)
    if mid_size == lo_size:
        lo, lo_size = mid_len, mid_size
    else:
        hi, hi_size = mid_len, mid_size
print(f"boundary located: length {lo} -> {lo_size} bytes, "
      f"length {hi} -> {hi_size} bytes (jump of {hi_size - lo_size} bytes)")
```

### File: `02_argmax_argmin.py`

```python
import cuda.tile as ct
import io

def array_param(size):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# argmax(x, axis=None) reduces the whole tile to a single scalar index.
sig_scalar_out = ct.compilation.KernelSignature(
    [array_param(64), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    idx = ct.argmax(x2d, None)
    ct.store(c, (pid,), ct.astype(idx, ct.float32))
try:
    n = compile_bytes(kernel_fn, sig_scalar_out)
    print(f"argmax(x2d, None): {n} cubin bytes")
except Exception as e:
    print(f"argmax(x2d, None): {type(e).__name__}: {e}")

# argmax(x, axis=1, keepdims=True) reduces one axis, keeping shape (8, 1).
sig_axis_out = ct.compilation.KernelSignature(
    [array_param(64), array_param(8), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    idx = ct.argmax(x2d, 1, keepdims=True)
    flat = ct.reshape(idx, (8,))
    ct.store(c, (pid,), ct.astype(flat, ct.float32))
try:
    n = compile_bytes(kernel_fn2, sig_axis_out)
    print(f"argmax(x2d, axis=1, keepdims=True): {n} cubin bytes")
except Exception as e:
    print(f"argmax(x2d, axis=1, keepdims=True): {type(e).__name__}: {e}")

# The docstring says argmax/argmin do NOT support a tuple of axes.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    idx = ct.argmax(x2d, (0, 1))
    ct.store(c, (pid,), ct.astype(idx, ct.float32))
try:
    n = compile_bytes(kernel_fn3, sig_scalar_out)
    print(f"argmax(x2d, (0, 1)) (tuple axis): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"argmax(x2d, (0, 1)) (tuple axis): {type(e).__name__}: {e}")
```

### File: `03_cumsum_cumprod_and_prod.py`

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

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 1)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumsum(x2d, axis=1): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 1, reverse=True)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumsum(x2d, axis=1, reverse=True): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumprod(x2d, 1)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumprod(x2d, axis=1): {compile_bytes(kernel_fn3, sig)} cubin bytes")

# cumsum and cumprod along axis 0 instead of axis 1 -- does the axis choice
# itself cost anything for a square (8, 8) tile?
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 0)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
print(f"cumsum(x2d, axis=0): {compile_bytes(kernel_fn4, sig)} cubin bytes")

# prod parallels sum, but multiplicatively. Naming-controlled comparison
# (Chapter 10) against ct.sum for the identical full-reduction shape.
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    r = ct.sum(x2d, None)
    ct.store(c, (pid,), ct.full((tile_size,), r, ct.float32))
kernel_sum = kernel_fn5
print(f"sum(x2d, None): {compile_bytes(kernel_sum, sig)} cubin bytes")

@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    r = ct.prod(x2d, None)
    ct.store(c, (pid,), ct.full((tile_size,), r, ct.float32))
kernel_prod = kernel_fn5
print(f"prod(x2d, None): {compile_bytes(kernel_prod, sig)} cubin bytes")

# The docstring says rounding_mode is "only supported for float types" --
# never checked directly against an integer-dtyped reduction before.
def int_array_param():
    return ct.compilation.ArrayConstraint(
        ct.int32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )
sig_int = ct.compilation.KernelSignature(
    [int_array_param(), int_array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    r = ct.prod(x2d, None, rounding_mode=ct.RoundingMode.RN)
    ct.store(c, (pid,), ct.full((tile_size,), r, ct.int32))
try:
    n = compile_bytes(kernel_fn6, sig_int)
    print(f"prod(int32 tile, rounding_mode=RN): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"prod(int32 tile, rounding_mode=RN): {type(e).__name__}: {e}")
```

### File: `04_scan_vs_cumsum.py`

```python
import cuda.tile as ct
import io
import operator

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

# ct.scan(x, axis, func, identity) applies a custom, user-supplied
# combining function as an inclusive prefix scan -- cumsum is documented
# as exactly this pattern with func=operator.add, identity=0. Naming-
# controlled (Chapter 10) comparison against the built-in ct.cumsum.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.cumsum(x2d, 1)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
kernel_builtin = kernel_fn
print(f"ct.cumsum(x2d, 1): {compile_bytes(kernel_builtin, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.scan(x2d, 1, operator.add, 0.0)
    ct.store(c, (pid,), ct.reshape(y, (tile_size,)))
kernel_custom = kernel_fn
try:
    n = compile_bytes(kernel_custom, sig)
    print(f"ct.scan(x2d, 1, operator.add, 0.0): {n} cubin bytes")
except Exception as e:
    print(f"ct.scan(x2d, 1, operator.add, 0.0): {type(e).__name__}: {e}")
```

### File: `05_reduce_vs_sum.py`

```python
import cuda.tile as ct
import io
import operator

def array_param(dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
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

# ct.reduce(x, axis, func, identity) applies a custom, user-supplied
# combining function along one axis -- sum(x, axis) is documented as
# exactly this pattern with func=operator.add, identity=0. Both reduce
# axis 1 of an (8, 8) tile down to shape (8, 1) with keepdims=True, then
# broadcast that back up to fill the (8, 8) output tile so both variants
# can share one output array shape. Naming-controlled (Chapter 10)
# comparison.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.sum(x2d, 1, keepdims=True)
    broadcasted = ct.broadcast_to(y, (8, 8))
    ct.store(c, (pid,), ct.reshape(broadcasted, (tile_size,)))
kernel_builtin = kernel_fn
print(f"ct.sum(x2d, axis=1, keepdims=True): {compile_bytes(kernel_builtin, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    y = ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True)
    broadcasted = ct.broadcast_to(y, (8, 8))
    ct.store(c, (pid,), ct.reshape(broadcasted, (tile_size,)))
kernel_custom = kernel_fn
try:
    n = compile_bytes(kernel_custom, sig)
    print(f"ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True): {n} cubin bytes")
except Exception as e:
    print(f"ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True): {type(e).__name__}: {e}")
```

### File: `06_capstone_running_max_and_custom_ops.py`

```python
import cuda.tile as ct
import io
import operator

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

# cuTile Python has no built-in "running maximum" scan -- only cumsum and
# cumprod are named. ct.scan's generality builds one anyway, something no
# combination of this book's named operations could do directly. This
# capstone combines it with argmax (Section 16.2) and a custom ct.reduce
# (Section 16.5) in a single kernel body.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    running_max = ct.scan(x2d, 1, ct.maximum, -3.0e38)
    peak_index = ct.argmax(x2d, None)
    row_totals = ct.reduce(x2d, 1, operator.add, 0.0, keepdims=True)
    combined = ct.add(running_max, ct.broadcast_to(row_totals, (8, 8)))
    combined = ct.add(combined, ct.astype(peak_index, ct.float32))
    ct.store(c, (pid,), ct.reshape(combined, (tile_size,)))

try:
    n = compile_bytes(kernel_fn, sig)
    print(f"capstone (scan running-max + argmax + custom reduce): {n} cubin bytes")
except Exception as e:
    print(f"capstone (scan running-max + argmax + custom reduce): {type(e).__name__}: {e}")
```
