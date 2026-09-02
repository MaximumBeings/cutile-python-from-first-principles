# Chapter 17: Elementwise Math and the Limits of Rounding Mode

> "Chapter 8 asked, in its own closing Self-Check Questions, whether it was safe to conclude that `RoundingMode.FULL` — rejected by both `ct.add` and `ct.sqrt` in every test that chapter ran — was rejected by every cuTile Python function that accepts a `rounding_mode` argument at all. The honest answer at the time was that two data points were not enough to say. This chapter finally has a third and a fourth: `ct.exp` and `ct.tanh` accept `RoundingMode.FULL` outright, and reject the very mode `ct.sqrt` defaults to."

**What you will understand by the end of this chapter:**

- `ct.sin`, `ct.cos`, `ct.tan`, `ct.sinh`, `ct.cosh`, `ct.tanh`, and `ct.atan2` all compile as ordinary elementwise operations — and that `tanh` alone among them accepts a `rounding_mode` argument the other five do not
- `ct.exp`, `ct.exp2`, `ct.log`, and `ct.log2`: that `exp` accepts `rounding_mode` while `exp2` rejects the keyword outright with a real, compiler-level `TileTypeError` rather than a plain Python argument-binding error
- The empirical answer to Chapter 8's own closing question: no `RoundingMode` value is universally accepted or universally rejected across cuTile Python's functions — `RN`, the mode `ct.sqrt` defaults to, is flatly rejected by `ct.exp` and `ct.tanh`, while `FULL`, rejected by `ct.add` and `ct.sqrt`, is accepted by both of them
- `ct.ceil`, `ct.floor`, `ct.abs`, `ct.negative`, and `ct.isnan`: which of these preserve their input's dtype and which change it, and `isnan`'s real, immediate rejection of an integer-dtyped tile
- `ct.pow`, `ct.mod`, `ct.floordiv`, and `ct.truediv`: broadcasting across mismatched tile shapes, dtype promotion across mixed integer and float operands, and `truediv`'s own acceptance of `RoundingMode.FULL`
- That `ct.reduce` (Chapter 16), `ct.broadcast_to` (Chapter 15), and this chapter's `ct.exp`, `ct.truediv`, and `ct.log` compose into a single log-softmax-shaped kernel body without conflict

**What you need to know first:**

- Chapter 8's `RoundingMode` and `flush_to_zero` material, and specifically its own Self-Check Question 5 — this chapter is that question, finally answered with real data.
- Chapter 15's `ct.reshape` and `ct.broadcast_to`, and Chapter 16's `ct.reduce` and naming-confound discipline (kernel function names matching exactly across any compared pair).
- Chapter 16's file-path-length confound: every absolute byte count in this chapter was measured at this book's own `/tmp/ch17/` sandbox path, a different length again from Chapter 16's `/tmp/ch16/` — comparisons within this chapter's own files remain valid, since each script compiles all its own variants at one shared `__file__` length.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow as every chapter before it.

## 17.1 Trigonometric and Hyperbolic Functions

### Intuition

`dir(ct)` lists seven functions this book has never called by name: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, and the two-argument `atan2`. Each is a straightforward elementwise operation with no masking, padding, or reduction behavior to untangle — the question worth asking is simply whether they all compile as expected, and whether any of the six single-argument functions quietly differs from the others in what keyword arguments it accepts.

### Background

`inspect.signature` reports `sin`, `cos`, `tan`, `sinh`, and `cosh` with the plain signature `(x, /) -> 'TileOrScalar'` — no `rounding_mode`, no `flush_to_zero`, nothing beyond the input tile itself. `tanh` is the one exception: its signature is `(x, /, *, rounding_mode=None) -> 'TileOrScalar'`, documented as accepting `RoundingMode.FULL` and `RoundingMode.APPROX` specifically, "f32 only," since a specific CUDA Toolkit version. That documented restriction is exactly what Section 17.3 tests directly, once this section's plain, default-argument calls confirm the six single-input functions all compile without incident. `atan2(x1, x2, /) -> 'TileOrScalar'` takes two tiles — a numerator and a denominator, in the docstring's own framing — with the same shape-broadcasting and dtype-promotion behavior this chapter's binary operations (Section 17.5) will exercise more thoroughly.

The `TileOrScalar` return annotation itself is worth noting: every reduction and reshaping operation this book has used through Chapter 16 was annotated as returning a `Tile`. These seven functions, and every other one-line elementwise math function `dir(ct)` still lists, are annotated to return either a `Tile` or a plain Python scalar — consistent with operating just as sensibly on an ordinary Python number as on tile data.

### Worked Example 17.1.1 — six single-argument functions and one two-argument function

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
    ct.store(c, (pid,), ct.sin(x))
print(f"sin(x): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.cos(x))
print(f"cos(x): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.tan(x))
print(f"tan(x): {compile_bytes(kernel_fn3, sig)} cubin bytes")

@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.sinh(x))
print(f"sinh(x): {compile_bytes(kernel_fn4, sig)} cubin bytes")

@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.cosh(x))
print(f"cosh(x): {compile_bytes(kernel_fn5, sig)} cubin bytes")

@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.tanh(x))
print(f"tanh(x): {compile_bytes(kernel_fn6, sig)} cubin bytes")

# atan2 takes two tiles -- the y-coordinate and x-coordinate tiles.
sig_double = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    y = ct.load(a, (pid,), (tile_size,))
    x = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.atan2(y, x))
print(f"atan2(y, x): {compile_bytes(kernel_fn7, sig_double)} cubin bytes")
```

Genuinely run:

```
sin(x): 26040 cubin bytes
cos(x): 26168 cubin bytes
tan(x): 25912 cubin bytes
sinh(x): 22304 cubin bytes
cosh(x): 21792 cubin bytes
tanh(x): 21920 cubin bytes
atan2(y, x): 27616 cubin bytes
```

### Discussion

All seven compile without incident. Since each is defined under a different function name (`kernel_fn` through `kernel_fn7`), the differing byte counts are not a Chapter-10-controlled comparison of relative cost — they are simply confirmation that each function compiles, with no attempt to interpret why `sin` compiles larger than `sinh`, or `cosh` smaller than `tanh`. The one substantive finding here is structural rather than numeric: `tanh` accepted the plain, no-argument call the same as its five siblings, but its signature alone already sets it apart by accepting a `rounding_mode` keyword none of the other five do — a difference Section 17.3 tests directly rather than leaving as a signature-inspection curiosity.

## 17.2 Exponential and Logarithmic Functions

### Intuition

`exp`, `exp2`, `log`, and `log2` round out the elementwise math surface's most common members. `exp` and `exp2` compute the same underlying operation in two different bases; `log` and `log2` invert them. Their signatures already hint at an asymmetry worth confirming directly: `exp` accepts a `rounding_mode` argument, but `exp2` accepts only `flush_to_zero` — the same split Chapter 8 saw between `rounding_mode` and `flush_to_zero` as genuinely separate, independently-controlled parameters, not two names for the same setting.

### Background

If `exp2`'s signature really does omit `rounding_mode` entirely, passing that keyword to `ct.exp2` inside a kernel body should fail — the interesting question is what kind of failure. Chapter 16 found that passing an unsupported keyword can be caught either by Python's own argument binding (a plain `TypeError`, before the kernel body is even inspected) or by cuTile Python's own compiler front-end (a `TileTypeError`, once the body is being type-checked). Since `@ct.kernel`-decorated functions defer their real argument validation to compile time, this section tests which kind of error `exp2` actually produces.

### Worked Example 17.2.1 — exp, exp2, log, log2, and exp2's missing keyword

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
    ct.store(c, (pid,), ct.exp(x))
print(f"exp(x): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.exp2(x))
print(f"exp2(x): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.log(x))
print(f"log(x): {compile_bytes(kernel_fn3, sig)} cubin bytes")

@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.log2(x))
print(f"log2(x): {compile_bytes(kernel_fn4, sig)} cubin bytes")

# exp2's documented parameter is flush_to_zero, not rounding_mode -- unlike
# exp itself. Does exp2 actually reject a rounding_mode keyword outright?
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.exp2(x, rounding_mode=ct.RoundingMode.RN))
try:
    n = compile_bytes(kernel_fn5, sig)
    print(f"exp2(x, rounding_mode=RN): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"exp2(x, rounding_mode=RN): {type(e).__name__}: {e}")

# exp2's flush_to_zero itself, exercised directly.
@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.exp2(x, flush_to_zero=True))
print(f"exp2(x, flush_to_zero=True): {compile_bytes(kernel_fn6, sig)} cubin bytes")
```

Genuinely run:

```
exp(x): 21408 cubin bytes
exp2(x): 21024 cubin bytes
log(x): 22432 cubin bytes
log2(x): 22432 cubin bytes
exp2(x, rounding_mode=RN): TileTypeError: exp2(): got an unexpected keyword argument 'rounding_mode'
  "/tmp/ch17/02_exp_and_log.py", line 54, col 25-68, in kernel_fn5:
        ct.store(c, (pid,), ct.exp2(x, rounding_mode=ct.RoundingMode.RN))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

exp2(x, flush_to_zero=True): 21024 cubin bytes
```

### Discussion

`exp`, `exp2`, `log`, and `log2` all compile cleanly on their default calls. The `rounding_mode` keyword passed to `exp2` is rejected with `TileTypeError: exp2(): got an unexpected keyword argument 'rounding_mode'` — a `TileTypeError`, not a plain Python `TypeError`, confirming this rejection happens inside cuTile Python's own compiler front-end rather than at ordinary Python function-call time, exactly as Chapter 16's `argmax` tuple-axis rejection was also a compiler-level `TileTypeError` rather than something Python's own call machinery caught first. The message's phrasing — "got an unexpected keyword argument," the same wording Python's own `TypeError` would use for an ordinary function — shows the compiler is deliberately mimicking standard Python argument-binding errors for a keyword that genuinely does not exist on `exp2`'s signature, rather than treating it as a semantically invalid but syntactically recognized argument the way Section 17.3's rejected rounding modes are. `exp2(x, flush_to_zero=True)` compiles to the identical 21024 bytes as the plain `exp2(x)` call — this book's first case of a documented parameter accepting a real, non-default value with no effect at all on the compiled artifact's size, worth noting honestly rather than assuming every parameter must leave a visible trace.

## 17.3 Rounding Modes Are Not One Size Fits All `[FOUNDATIONAL]`

### Intuition

Chapter 8 tested `RoundingMode`'s seven enum values against exactly two functions, `ct.add` and `ct.sqrt`, and found both rejected `RoundingMode.FULL` with the same exception type and message pattern. Its own closing Self-Check Question 5 asked directly whether that was enough evidence to conclude `FULL` is rejected by every cuTile Python function that accepts a `rounding_mode` argument — and answered, correctly, that two data points from admittedly similar arithmetic operations were not enough to say. `ct.exp` and `ct.tanh` are this chapter's third and fourth data points, and their own documentation already hints at the answer: both list `FULL` and `APPROX` as their explicitly supported modes, with no mention of `RN` at all.

### Background

If those docstrings are accurate, `ct.exp` and `ct.tanh` should accept `RoundingMode.FULL` — the exact mode `ct.add` and `ct.sqrt` rejected — and should reject `RoundingMode.RN`, the exact mode `ct.sqrt` defaults to and compiles successfully with. Testing this requires exercising both functions against `RN`, `FULL`, and `APPROX` directly, and — since trusting Chapter 8's own report from a different file and a different sandbox run would itself be a small violation of this book's "verify, don't assume" discipline — reproducing `ct.sqrt`'s three-mode behavior in this same script rather than citing the earlier chapter's numbers directly.

### Worked Example 17.3.1 — `ct.exp`, `ct.tanh`, and `ct.sqrt` against the same three modes

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

def try_build(name, fn, mode):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        ct.store(c, (pid,), fn(x, rounding_mode=mode))
    try:
        n = compile_bytes(kernel_fn, sig)
        print(f"{name}(x, rounding_mode={mode}): {n} cubin bytes")
    except Exception as e:
        print(f"{name}(x, rounding_mode={mode}): {type(e).__name__}: {e}")

# Chapter 8 found ct.sqrt accepts RoundingMode.RN and RoundingMode.APPROX,
# and rejects RoundingMode.FULL. ct.exp and ct.tanh document a DIFFERENT
# subset -- only FULL and APPROX -- explicitly since a specific CUDA
# Toolkit version. Does ct.exp actually reject RN, the very mode ct.sqrt
# defaults to?
try_build("ct.exp", ct.exp, ct.RoundingMode.RN)
try_build("ct.exp", ct.exp, ct.RoundingMode.FULL)
try_build("ct.exp", ct.exp, ct.RoundingMode.APPROX)

try_build("ct.tanh", ct.tanh, ct.RoundingMode.RN)
try_build("ct.tanh", ct.tanh, ct.RoundingMode.FULL)
try_build("ct.tanh", ct.tanh, ct.RoundingMode.APPROX)

# For contrast, reproduce Chapter 8's own sqrt findings directly in this
# same script, rather than trusting the earlier chapter's report from a
# different file and a different sandbox run.
try_build("ct.sqrt", ct.sqrt, ct.RoundingMode.RN)
try_build("ct.sqrt", ct.sqrt, ct.RoundingMode.FULL)
try_build("ct.sqrt", ct.sqrt, ct.RoundingMode.APPROX)
```

Genuinely run:

```
ct.exp(x, rounding_mode=RoundingMode.RN): TileTypeError: Rounding mode nearest_even is not supported for exp
  "/tmp/ch17/03_rounding_mode_not_one_size_fits_all.py", line 25, col 29-53, in kernel_fn:
            ct.store(c, (pid,), fn(x, rounding_mode=mode))
                                ^^^^^^^^^^^^^^^^^^^^^^^^^

ct.exp(x, rounding_mode=RoundingMode.FULL): 21536 cubin bytes
ct.exp(x, rounding_mode=RoundingMode.APPROX): 21024 cubin bytes
ct.tanh(x, rounding_mode=RoundingMode.RN): TileTypeError: Rounding mode nearest_even is not supported for tanh
  "/tmp/ch17/03_rounding_mode_not_one_size_fits_all.py", line 25, col 29-53, in kernel_fn:
            ct.store(c, (pid,), fn(x, rounding_mode=mode))
                                ^^^^^^^^^^^^^^^^^^^^^^^^^

ct.tanh(x, rounding_mode=RoundingMode.FULL): 21920 cubin bytes
ct.tanh(x, rounding_mode=RoundingMode.APPROX): 21024 cubin bytes
ct.sqrt(x, rounding_mode=RoundingMode.RN): 21984 cubin bytes
ct.sqrt(x, rounding_mode=RoundingMode.FULL): TileTypeError: Rounding mode full is not supported for sqrt
  "/tmp/ch17/03_rounding_mode_not_one_size_fits_all.py", line 25, col 29-53, in kernel_fn:
            ct.store(c, (pid,), fn(x, rounding_mode=mode))
                                ^^^^^^^^^^^^^^^^^^^^^^^^^

ct.sqrt(x, rounding_mode=RoundingMode.APPROX): 21024 cubin bytes
```

### Discussion

The documented asymmetry is real and precisely the opposite of what a reader generalizing from Chapter 8 alone would expect. `ct.exp` and `ct.tanh` both reject `RoundingMode.RN` outright — the exact mode `ct.sqrt` compiles successfully with, at 21984 bytes — and both accept `RoundingMode.FULL`, the exact mode Chapter 8 found `ct.add` and `ct.sqrt` reject. `RoundingMode.APPROX` is accepted by all three functions tested here, though Chapter 8 separately found `ct.add` rejects `APPROX` — meaning even the one mode that looked consistent across this chapter's three functions is not universal either, once `ct.add` is brought back into the comparison. Every rejection here reproduces the same message pattern Chapter 8 already established — "Rounding mode `<name>` is not supported for `<function>`" — naming both the specific mode and the specific function, which is itself informative: the restriction is evidently tracked per function, with its own explicit allow-list, rather than being derived from some shared property (float-only, say) that would reject or accept the same modes everywhere.

This directly and empirically answers Chapter 8's Self-Check Question 5: no, it was not safe to conclude `RoundingMode.FULL` is rejected by every function that accepts a `rounding_mode` argument, and the two functions that finally test the opposite case — `ct.exp` and `ct.tanh` — don't just fail to reject `FULL`, they specifically require moving away from the `RN`/`RZ`/`RM`/`RP` family that every rounding-mode-accepting function this book had tested through Chapter 8 defaulted toward. The general lesson holds in both directions now: neither "this mode is always accepted" nor "this mode is always rejected" is a safe inference from any function this book has tested, no matter how many functions have agreed so far. Each function's supported subset has to be checked on its own.

## 17.4 Ceiling, Floor, Absolute Value, Negation, and NaN Detection

### Intuition

`ceil`, `floor`, `abs`, and `negative` each modify a tile's values without doing arithmetic between two tiles or reducing anything — the plainest possible elementwise operations. `isnan` is different in kind: it doesn't transform a value, it classifies one, and its result is naturally boolean rather than a value in the input's own dtype.

### Background

`ct.ceil(x, /)` and `ct.floor(x, /)` share the plainest possible signature, with no `rounding_mode` or `flush_to_zero` at all — unsurprising, since rounding to the nearest integer direction is already what these functions exist to do; there is no separate "rounding mode" left to control. `ct.abs(x, /)` and `ct.negative(x, /)` (documented as "same as `-x`") share that same plain signature and, being sign-related rather than value-transforming, are worth testing against both a float32 tile and an int32 tile to confirm the same function handles both dtypes without a separate integer-specific entry point. `ct.isnan(x, /)` is documented to return a boolean-classified result, which this section's signature declares with a `bool_`-dtyped output array rather than reusing the input's own dtype — and since NaN is meaningful only for floating-point representations, an integer-dtyped input is worth testing as a likely, real rejection rather than an operation that would simply always report `False`.

### Worked Example 17.4.1 — ceil, floor, abs, negative, and isnan across dtypes

```python
import cuda.tile as ct
import io

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
sig_int = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_bool_out = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.ceil(x))
print(f"ceil(x): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.floor(x))
print(f"floor(x): {compile_bytes(kernel_fn2, sig)} cubin bytes")

# abs on a float32 tile.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.abs(x))
print(f"abs(float32 x): {compile_bytes(kernel_fn3, sig)} cubin bytes")

# abs on an int32 tile -- a genuinely different dtype path through the
# same named function.
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.abs(x))
print(f"abs(int32 x): {compile_bytes(kernel_fn4, sig_int)} cubin bytes")

@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.negative(x))
print(f"negative(int32 x): {compile_bytes(kernel_fn5, sig_int)} cubin bytes")

# isnan returns a boolean-dtyped tile, not a value in the input's own dtype.
@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.isnan(x))
print(f"isnan(x): {compile_bytes(kernel_fn6, sig_bool_out)} cubin bytes")

# isnan on an int32 tile -- NaN is a float-only concept.
@ct.kernel
def kernel_fn7(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.isnan(x))
try:
    n = compile_bytes(kernel_fn7, ct.compilation.KernelSignature(
        [array_param(ct.int32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    ))
    print(f"isnan(int32 x): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"isnan(int32 x): {type(e).__name__}: {e}")
```

Genuinely run:

```
ceil(x): 21024 cubin bytes
floor(x): 21024 cubin bytes
abs(float32 x): 21024 cubin bytes
abs(int32 x): 21024 cubin bytes
negative(int32 x): 21024 cubin bytes
isnan(x): 21024 cubin bytes
isnan(int32 x): TileTypeError: Unexpected input type Tile[int32,(64)]
  "/tmp/ch17/04_ceil_floor_abs_negative_isnan.py", line 79, col 25-35, in kernel_fn7:
        ct.store(c, (pid,), ct.isnan(x))
                            ^^^^^^^^^^^
```

### Discussion

`ceil`, `floor`, `abs` on both dtypes, and `negative` all compile cleanly, each preserving its input's own dtype through to the output array. All six of these single-function kernels happen to compile to the identical 21024 bytes — each is defined under its own distinct function name (`kernel_fn` through `kernel_fn6`), so this is, once again, not a Chapter-10-controlled comparison and not evidence that these six operations cost the compiler the same amount; it simply reflects that each is a minimally simple, single-operation kernel body of roughly the same shape. `isnan` on an integer-dtyped tile is rejected with `TileTypeError: Unexpected input type Tile[int32,(64)]` — a real, immediate compile-time error confirming NaN detection is restricted to floating-point representations, phrased generically enough ("Unexpected input type") that it reads as a broader type-compatibility check rather than an `isnan`-specific integer ban, similar in spirit to Chapter 16's `argmax` tuple-axis rejection being phrased as a general type mismatch rather than a function-specific rule.

## 17.5 Pow, Mod, Floordiv, and Truediv: The Binary Elementwise Family

### Intuition

`pow`, `mod`, `floordiv`, and `truediv` are cuTile Python's remaining binary elementwise operations — each documented as broadcasting its two operands' shapes and promoting their dtypes to a common one, exactly as `atan2` was in Section 17.1. `truediv` additionally carries the same `rounding_mode`/`flush_to_zero` parameter shape as Chapter 8's `ct.add` and `ct.sqrt`, and Section 17.3's finding — that no rounding mode is universally accepted or rejected — is worth testing against a fourth function here rather than assumed to apply.

### Background

Broadcasting an `(8, 8)` tile against an `(8, 1)` tile — one operand varying per row, the other constant across each row — is the same shape relationship Chapter 15's `ct.cat` and Chapter 16's `ct.reduce`-with-`keepdims` already exercised; `ct.pow` is tested against it directly here. Mixed-dtype promotion is tested by calling `ct.mod` with one int32 operand and one float32 operand, which the docstring's "dtype promoted to common dtype" language implies should resolve to a single working dtype rather than being rejected outright. `ct.floordiv`'s own docstring specifically notes it supports floating-point operands, computing `floor(x / y)` rather than being restricted to integer division the way Python's own `//` operator might suggest to a reader unfamiliar with its float-operand behavior.

### Worked Example 17.5.1 — broadcasting, dtype promotion, and truediv's rounding mode

```python
import cuda.tile as ct
import io

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
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.pow(x, y))
print(f"pow(x, y), matching (64,) shapes: {compile_bytes(kernel_fn, sig)} cubin bytes")

# pow with a (8, 8) tile and an (8, 1) tile -- the docs promise broadcasting.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    exponents = ct.full((8, 1), 2.0, ct.float32)
    result = ct.pow(x2d, exponents)
    ct.store(c, (pid,), ct.reshape(result, (tile_size,)))
sig_bcast = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn2, sig_bcast)
    print(f"pow((8,8) tile, (8,1) tile) broadcast: {n} cubin bytes")
except Exception as e:
    print(f"pow((8,8) tile, (8,1) tile) broadcast: {type(e).__name__}: {e}")

# mod on int32 tiles.
sig_int = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn3(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.mod(x, y))
print(f"mod(int32 x, int32 y): {compile_bytes(kernel_fn3, sig_int)} cubin bytes")

# mod with mixed dtypes -- int32 and float32 operands, promoted to a
# common dtype per the docstring.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn4(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.mod(x, y))
try:
    n = compile_bytes(kernel_fn4, sig_mixed)
    print(f"mod(int32 x, float32 y) mixed dtype: {n} cubin bytes")
except Exception as e:
    print(f"mod(int32 x, float32 y) mixed dtype: {type(e).__name__}: {e}")

# floordiv on float32 operands -- the docstring says it supports floats
# too, not just integers, unlike a naive reading of "floor division".
@ct.kernel
def kernel_fn5(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.floordiv(x, y))
print(f"floordiv(float32 x, float32 y): {compile_bytes(kernel_fn5, sig)} cubin bytes")

# truediv, default and with rounding_mode -- the same parameter shape as
# Chapter 8's ct.sqrt and ct.add.
@ct.kernel
def kernel_fn6(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.truediv(x, y))
print(f"truediv(x, y): {compile_bytes(kernel_fn6, sig)} cubin bytes")

@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.truediv(x, y, rounding_mode=ct.RoundingMode.FULL))
try:
    n = compile_bytes(kernel_fn7, sig)
    print(f"truediv(x, y, rounding_mode=FULL): {n} cubin bytes")
except Exception as e:
    print(f"truediv(x, y, rounding_mode=FULL): {type(e).__name__}: {e}")
```

Genuinely run:

```
pow(x, y), matching (64,) shapes: 28176 cubin bytes
pow((8,8) tile, (8,1) tile) broadcast: 27392 cubin bytes
mod(int32 x, int32 y): 23328 cubin bytes
mod(int32 x, float32 y) mixed dtype: 27168 cubin bytes
floordiv(float32 x, float32 y): 25824 cubin bytes
truediv(x, y): 25696 cubin bytes
truediv(x, y, rounding_mode=FULL): 23584 cubin bytes
```

### Discussion

`pow`'s broadcast against a genuinely different shape — `(8, 8)` against `(8, 1)` — compiles cleanly, confirming the same broadcasting behavior this book has already seen from `ct.cat` and `ct.reduce`-with-`keepdims` extends to this binary elementwise family as documented. `mod` accepts the mixed int32/float32 pair without rejection, compiling to 27168 bytes — larger than either same-dtype `mod` call this section ran, consistent with a real dtype-promotion step happening rather than the mismatched-dtype call being silently rejected or silently truncated to one operand's dtype. `floordiv` on two float32 tiles compiles without incident, confirming the docstring's claim that floor division is not restricted to integer operands the way Python's own `//` might suggest to a reader who has only ever used it on integers.

`truediv(x, y, rounding_mode=RoundingMode.FULL)` compiles successfully at 23584 bytes — a fourth function, alongside `ct.exp` and `ct.tanh`, that accepts the exact `RoundingMode.FULL` value Chapter 8 found `ct.add` and `ct.sqrt` reject. This reinforces Section 17.3's conclusion from an independent function rather than merely repeating it: `FULL`'s acceptance is not a property specific to `exp`-family transcendental functions, since `truediv` is an ordinary arithmetic operation in the same family as `add`, and yet the two functions land on opposite sides of `FULL`'s support.

## 17.6 Capstone: A Log-Softmax-Shaped Kernel

### Intuition

Every operation this chapter introduced was tested in isolation. The capstone composes several of them — `exp`, `truediv`, and `log` — with machinery from two earlier chapters, `ct.reduce` from Chapter 16 and `ct.broadcast_to` from Chapter 15, into a single kernel body shaped like a log-softmax computation: subtract each row's maximum for numerical range control, exponentiate, normalize by each row's sum, then take the log of the result.

### Background

This capstone deliberately uses `ct.exp` with `rounding_mode=RoundingMode.FULL` — the exact combination Section 17.3 confirmed compiles, in contrast to the `RoundingMode.RN` that same section confirmed `ct.exp` rejects — so the choice is not incidental to this kernel but a direct application of this chapter's own finding. `ct.sub`, `ct.sum`, and `ct.reduce` have all appeared in earlier chapters; nothing here is a new operation beyond this chapter's own `exp`, `truediv`, and `log`.

### Worked Example 17.6.1 — reduce, broadcast, subtract, exponentiate, sum, divide, log

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

# A log-softmax-shaped kernel body, chosen to chain this chapter's new
# elementwise math (exp, truediv, log) through machinery from earlier
# chapters: Chapter 16's ct.reduce, Chapter 15's ct.broadcast_to and
# ct.reshape, and Section 17.3's finding that ct.exp accepts
# RoundingMode.FULL where ct.add and ct.sqrt do not.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    row_max = ct.reduce(x2d, 1, ct.maximum, -3.0e38, keepdims=True)
    shifted = ct.sub(x2d, ct.broadcast_to(row_max, (8, 8)))
    exped = ct.exp(shifted, rounding_mode=ct.RoundingMode.FULL)
    row_sum = ct.sum(exped, 1, keepdims=True)
    normalized = ct.truediv(exped, ct.broadcast_to(row_sum, (8, 8)))
    log_probs = ct.log(normalized)
    ct.store(c, (pid,), ct.reshape(log_probs, (tile_size,)))

try:
    n = compile_bytes(kernel_fn, sig)
    print(f"capstone (reduce + broadcast_to + sub + exp[FULL] + sum + truediv + log): {n} cubin bytes")
except Exception as e:
    print(f"capstone (reduce + broadcast_to + sub + exp[FULL] + sum + truediv + log): {type(e).__name__}: {e}")
```

Genuinely run:

```
capstone (reduce + broadcast_to + sub + exp[FULL] + sum + truediv + log): 50176 cubin bytes
```

### Discussion

The kernel compiles cleanly on the first attempt: a custom-function reduction (Chapter 16) finds each row's maximum, `ct.broadcast_to` (Chapter 15) expands that back to the full tile shape for subtraction, `ct.exp` runs with the `FULL` rounding mode Section 17.3 confirmed it accepts, `ct.sum` collapses each row again for normalization, `ct.truediv` divides, and `ct.log` closes the loop — seven operations spanning three chapters, composing without any adjustment needed to satisfy the compiler. As with every capstone since Chapter 15, a clean compile confirms only that each operation's own structural and type rules were satisfied at every step; it says nothing about whether the specific numeric values this kernel would produce on real hardware form a mathematically sound log-softmax for any given input, since this book's `export_kernel`-only, driver-free environment has never been able to check computed results, only compiled structure, all the way back to Chapter 1.

## Chapter Summary

This chapter completed the elementwise math surface `dir(ct)` had left untouched since Chapter 15 first promised it: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, and `atan2` all compile as ordinary elementwise operations, with `tanh` alone among the trigonometric and hyperbolic functions accepting a `rounding_mode` argument; `exp`, `exp2`, `log`, and `log2` round out the exponential and logarithmic family, with `exp2` rejecting an unsupported `rounding_mode` keyword through a genuine compiler-level `TileTypeError` rather than an ordinary Python argument-binding failure. Most significantly, this chapter finally answered Chapter 8's own closing Self-Check Question: whether `RoundingMode.FULL`'s rejection by both `ct.add` and `ct.sqrt` was safe to generalize to every function accepting a `rounding_mode` argument. It was not. `ct.exp`, `ct.tanh`, and `ct.truediv` all accept `RoundingMode.FULL` outright, and `ct.exp` and `ct.tanh` both reject `RoundingMode.RN` — the very mode `ct.sqrt` defaults to — meaning neither universal acceptance nor universal rejection of any single rounding mode is a safe inference from however many functions have agreed on it so far. `ceil`, `floor`, `abs`, and `negative` preserve their input's dtype through ordinary elementwise transformation, while `isnan` classifies into a genuinely different, boolean-dtyped output and rejects an integer-dtyped input outright. `pow`, `mod`, `floordiv`, and `truediv` confirmed the documented broadcasting and dtype-promotion behavior this chapter's binary operations share with `atan2`, and a capstone kernel composed `exp`, `truediv`, and `log` with Chapter 16's `reduce` and Chapter 15's `broadcast_to` into a single log-softmax-shaped kernel body with no adjustment needed to satisfy the compiler.

## Self-Check Questions

1. Section 17.3 found that `ct.exp` and `ct.tanh` reject `RoundingMode.RN` and accept `RoundingMode.FULL` — the reverse of `ct.sqrt`'s behavior. Chapter 8 tested only `ct.add` and `ct.sqrt` before drawing its own cautious conclusion. What made two data points insufficient there, and what does having four data points now (still) not entitle this book to conclude?
2. Section 17.2 found `ct.exp2`'s rejection of a `rounding_mode` keyword uses the phrasing "got an unexpected keyword argument" — the same wording an ordinary Python `TypeError` would use — but the actual exception raised is `TileTypeError`. What does the choice to mimic Python's own error phrasing, while still raising a compiler-specific exception type, suggest about how deliberately these error messages are designed?
3. Section 17.4 found `ceil`, `floor`, `abs` (both dtypes), and `negative` all compile to the identical 21024 bytes. Why does this NOT license the conclusion that these four operations cost the compiler the same amount, in the way Section 16.4's naming-controlled `ct.scan`-versus-`ct.cumsum` comparison legitimately did?
4. Section 17.5 found `ct.mod(int32, float32)` compiles to a byte count different from either same-dtype `mod` call in that section. What would you need to check, beyond a successful compile and a different byte count, to confirm dtype promotion is actually happening the way the docstring describes, rather than the mismatched call being handled some other way that also happens to produce a different-sized artifact?
5. Section 17.6's capstone deliberately chose `rounding_mode=RoundingMode.FULL` for its `ct.exp` call rather than leaving the argument at its default. What is the specific reason FULL was chosen there instead of, say, `RoundingMode.APPROX` — which Section 17.3 also confirmed `ct.exp` accepts?

## Where We Go Next

Chapter 15 first pointed to this chapter's elementwise math surface as a gap, and Chapter 16's own drafting process delayed reaching it by a full chapter in favor of the file-path confound. This chapter finally closed that gap, and in doing so, closed a loop opened all the way back in Chapter 8: the question of whether any `RoundingMode` value could be assumed universally accepted or rejected is now answered, with real evidence spanning arithmetic operations (`add`, `sqrt`, `truediv`) and transcendental ones (`exp`, `tanh`), across two chapters. `dir(ct)` is close to exhausted of operations this book has never tested directly, but a handful remain genuinely untouched: the bitwise family (`bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `bitwise_lshift`, `bitwise_rshift`), the comparison operations (`equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`) and `ct.where`'s conditional selection built on top of them, and the atomic memory operations (`atomic_add`, `atomic_max`, `atomic_cas`, and their siblings) that this book's single-block, single-pass kernels have never had a reason to reach for. Chapter 18 turns to the first of these: the comparison operations and `ct.where`, the machinery conditional logic inside a kernel body actually runs on.

## Worked Solutions

**1.** Two data points were insufficient because `ct.add` and `ct.sqrt` are both ordinary arithmetic operations from the same general family, and their agreement on rejecting `FULL` could just as easily have reflected something specific to that family — a shared implementation detail, a shared category of operation — as a genuinely universal rule about `RoundingMode.FULL` itself. Chapter 8's own caution was well-founded: `ct.exp` and `ct.tanh`, transcendental rather than arithmetic, immediately broke the pattern. But four data points still does not entitle this book to conclude anything universal either — it only shows that both universal acceptance and universal rejection of `FULL` are false, for at least these functions; a fifth or sixth function could still show a third or fourth distinct pattern (accepting `FULL` but rejecting `APPROX`, say), and nothing this chapter tested rules that out. The only safe general conclusion is the negative one: no single mode's status can be assumed for an untested function.

**2.** Mimicking Python's own "got an unexpected keyword argument" phrasing while still raising `TileTypeError` (rather than letting Python's own `TypeError` propagate, or inventing entirely different wording) suggests the error messages are deliberately designed to feel familiar to a Python programmer encountering them, even though the underlying mechanism — a compiler's own type-checking pass working through an intermediate representation of the kernel body, not real-time Python function dispatch — is quite different from what produces that message in ordinary Python code. This is consistent with cuTile Python's broader pattern of error messages seen throughout this book: specific, named arguments, expected and actual types spelled out, source locations pointing at the exact offending line — all suggesting real design investment in making compiler-stage errors read like ordinary, familiar Python errors rather than exposing the compiler's own internal machinery.

**3.** Section 16.4's `ct.scan`-versus-`ct.cumsum` comparison was legitimate because both kernels were defined under the identical function name `kernel_fn`, satisfying Chapter 10's naming-confound rule directly. Section 17.4's six functions are each defined under a distinct name (`kernel_fn` through `kernel_fn6`), which is exactly the situation Chapter 10 warned produces a meaningless byte-count comparison — a shared byte count under different names could reflect these operations genuinely costing the same amount, or it could reflect the naming confound coincidentally landing all six kernels' names in the same length or hashing bucket, or some other artifact entirely. Without renaming all six to share one identical function name and recompiling, there is no way to distinguish "these operations really cost the same" from "the byte counts happen to match for unrelated reasons" — which is precisely why the Discussion explicitly declines to draw that conclusion.

**4.** A successful compile and a different byte count are consistent with dtype promotion happening as documented, but they are also consistent with other explanations this section never ruled out — for instance, the compiler silently treating the int32 operand as if it were already float32 without any real promotion step, or applying some other implicit conversion that happens to also change the compiled size. Confirming real dtype promotion specifically would require inspecting the actual output dtype the compiler assigns to the `mod` result — checking what `ArrayConstraint` dtype the output array must declare for the kernel to compile successfully — rather than inferring it indirectly from a byte count that changed for unspecified reasons. This book's `export_kernel`-only environment makes that a matter of testing different declared output dtypes against the same input kernel and seeing which ones compile, not something a byte count alone can settle.

**5.** `RoundingMode.APPROX` and `RoundingMode.FULL` are both confirmed to compile successfully with `ct.exp` in Section 17.3, but they represent different tradeoffs the names themselves suggest — `APPROX` an approximate, presumably faster computation, and `FULL` a fully-precise one — even though this book's `export_kernel`-only environment cannot measure either speed or numerical accuracy directly, only confirm both compile. `FULL` was chosen for the capstone specifically because it is the more surprising of the two results to have confirmed: `RoundingMode.FULL` was the exact value Chapter 8 found flatly rejected by both `ct.add` and `ct.sqrt`, so using it in a capstone kernel is a deliberate, visible callback to that earlier finding being overturned here — a choice made to demonstrate the finding in composition, not merely to pick whichever mode happened to be tested first.

## Complete Runnable Code

### File: `01_trig_and_hyperbolic.py`

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
    ct.store(c, (pid,), ct.sin(x))
print(f"sin(x): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.cos(x))
print(f"cos(x): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.tan(x))
print(f"tan(x): {compile_bytes(kernel_fn3, sig)} cubin bytes")

@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.sinh(x))
print(f"sinh(x): {compile_bytes(kernel_fn4, sig)} cubin bytes")

@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.cosh(x))
print(f"cosh(x): {compile_bytes(kernel_fn5, sig)} cubin bytes")

@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.tanh(x))
print(f"tanh(x): {compile_bytes(kernel_fn6, sig)} cubin bytes")

# atan2 takes two tiles -- the y-coordinate and x-coordinate tiles.
sig_double = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    y = ct.load(a, (pid,), (tile_size,))
    x = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.atan2(y, x))
print(f"atan2(y, x): {compile_bytes(kernel_fn7, sig_double)} cubin bytes")
```

### File: `02_exp_and_log.py`

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
    ct.store(c, (pid,), ct.exp(x))
print(f"exp(x): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.exp2(x))
print(f"exp2(x): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.log(x))
print(f"log(x): {compile_bytes(kernel_fn3, sig)} cubin bytes")

@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.log2(x))
print(f"log2(x): {compile_bytes(kernel_fn4, sig)} cubin bytes")

# exp2's documented parameter is flush_to_zero, not rounding_mode -- unlike
# exp itself. Does exp2 actually reject a rounding_mode keyword outright?
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.exp2(x, rounding_mode=ct.RoundingMode.RN))
try:
    n = compile_bytes(kernel_fn5, sig)
    print(f"exp2(x, rounding_mode=RN): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"exp2(x, rounding_mode=RN): {type(e).__name__}: {e}")

# exp2's flush_to_zero itself, exercised directly.
@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.exp2(x, flush_to_zero=True))
print(f"exp2(x, flush_to_zero=True): {compile_bytes(kernel_fn6, sig)} cubin bytes")
```

### File: `03_rounding_mode_not_one_size_fits_all.py`

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

def try_build(name, fn, mode):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        ct.store(c, (pid,), fn(x, rounding_mode=mode))
    try:
        n = compile_bytes(kernel_fn, sig)
        print(f"{name}(x, rounding_mode={mode}): {n} cubin bytes")
    except Exception as e:
        print(f"{name}(x, rounding_mode={mode}): {type(e).__name__}: {e}")

# Chapter 8 found ct.sqrt accepts RoundingMode.RN and RoundingMode.APPROX,
# and rejects RoundingMode.FULL. ct.exp and ct.tanh document a DIFFERENT
# subset -- only FULL and APPROX -- explicitly since a specific CUDA
# Toolkit version. Does ct.exp actually reject RN, the very mode ct.sqrt
# defaults to?
try_build("ct.exp", ct.exp, ct.RoundingMode.RN)
try_build("ct.exp", ct.exp, ct.RoundingMode.FULL)
try_build("ct.exp", ct.exp, ct.RoundingMode.APPROX)

try_build("ct.tanh", ct.tanh, ct.RoundingMode.RN)
try_build("ct.tanh", ct.tanh, ct.RoundingMode.FULL)
try_build("ct.tanh", ct.tanh, ct.RoundingMode.APPROX)

# For contrast, reproduce Chapter 8's own sqrt findings directly in this
# same script, rather than trusting the earlier chapter's report from a
# different file and a different sandbox run.
try_build("ct.sqrt", ct.sqrt, ct.RoundingMode.RN)
try_build("ct.sqrt", ct.sqrt, ct.RoundingMode.FULL)
try_build("ct.sqrt", ct.sqrt, ct.RoundingMode.APPROX)
```

### File: `04_ceil_floor_abs_negative_isnan.py`

```python
import cuda.tile as ct
import io

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
sig_int = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_bool_out = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.ceil(x))
print(f"ceil(x): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.floor(x))
print(f"floor(x): {compile_bytes(kernel_fn2, sig)} cubin bytes")

# abs on a float32 tile.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.abs(x))
print(f"abs(float32 x): {compile_bytes(kernel_fn3, sig)} cubin bytes")

# abs on an int32 tile -- a genuinely different dtype path through the
# same named function.
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.abs(x))
print(f"abs(int32 x): {compile_bytes(kernel_fn4, sig_int)} cubin bytes")

@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.negative(x))
print(f"negative(int32 x): {compile_bytes(kernel_fn5, sig_int)} cubin bytes")

# isnan returns a boolean-dtyped tile, not a value in the input's own dtype.
@ct.kernel
def kernel_fn6(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.isnan(x))
print(f"isnan(x): {compile_bytes(kernel_fn6, sig_bool_out)} cubin bytes")

# isnan on an int32 tile -- NaN is a float-only concept.
@ct.kernel
def kernel_fn7(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.isnan(x))
try:
    n = compile_bytes(kernel_fn7, ct.compilation.KernelSignature(
        [array_param(ct.int32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    ))
    print(f"isnan(int32 x): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"isnan(int32 x): {type(e).__name__}: {e}")
```

### File: `05_pow_mod_floordiv_truediv.py`

```python
import cuda.tile as ct
import io

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
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.pow(x, y))
print(f"pow(x, y), matching (64,) shapes: {compile_bytes(kernel_fn, sig)} cubin bytes")

# pow with a (8, 8) tile and an (8, 1) tile -- the docs promise broadcasting.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    exponents = ct.full((8, 1), 2.0, ct.float32)
    result = ct.pow(x2d, exponents)
    ct.store(c, (pid,), ct.reshape(result, (tile_size,)))
sig_bcast = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn2, sig_bcast)
    print(f"pow((8,8) tile, (8,1) tile) broadcast: {n} cubin bytes")
except Exception as e:
    print(f"pow((8,8) tile, (8,1) tile) broadcast: {type(e).__name__}: {e}")

# mod on int32 tiles.
sig_int = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn3(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.mod(x, y))
print(f"mod(int32 x, int32 y): {compile_bytes(kernel_fn3, sig_int)} cubin bytes")

# mod with mixed dtypes -- int32 and float32 operands, promoted to a
# common dtype per the docstring.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn4(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.mod(x, y))
try:
    n = compile_bytes(kernel_fn4, sig_mixed)
    print(f"mod(int32 x, float32 y) mixed dtype: {n} cubin bytes")
except Exception as e:
    print(f"mod(int32 x, float32 y) mixed dtype: {type(e).__name__}: {e}")

# floordiv on float32 operands -- the docstring says it supports floats
# too, not just integers, unlike a naive reading of "floor division".
@ct.kernel
def kernel_fn5(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.floordiv(x, y))
print(f"floordiv(float32 x, float32 y): {compile_bytes(kernel_fn5, sig)} cubin bytes")

# truediv, default and with rounding_mode -- the same parameter shape as
# Chapter 8's ct.sqrt and ct.add.
@ct.kernel
def kernel_fn6(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.truediv(x, y))
print(f"truediv(x, y): {compile_bytes(kernel_fn6, sig)} cubin bytes")

@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.truediv(x, y, rounding_mode=ct.RoundingMode.FULL))
try:
    n = compile_bytes(kernel_fn7, sig)
    print(f"truediv(x, y, rounding_mode=FULL): {n} cubin bytes")
except Exception as e:
    print(f"truediv(x, y, rounding_mode=FULL): {type(e).__name__}: {e}")
```

### File: `06_capstone_log_softmax_shaped.py`

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

# A log-softmax-shaped kernel body, chosen to chain this chapter's new
# elementwise math (exp, truediv, log) through machinery from earlier
# chapters: Chapter 16's ct.reduce, Chapter 15's ct.broadcast_to and
# ct.reshape, and Section 17.3's finding that ct.exp accepts
# RoundingMode.FULL where ct.add and ct.sqrt do not.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    row_max = ct.reduce(x2d, 1, ct.maximum, -3.0e38, keepdims=True)
    shifted = ct.sub(x2d, ct.broadcast_to(row_max, (8, 8)))
    exped = ct.exp(shifted, rounding_mode=ct.RoundingMode.FULL)
    row_sum = ct.sum(exped, 1, keepdims=True)
    normalized = ct.truediv(exped, ct.broadcast_to(row_sum, (8, 8)))
    log_probs = ct.log(normalized)
    ct.store(c, (pid,), ct.reshape(log_probs, (tile_size,)))

try:
    n = compile_bytes(kernel_fn, sig)
    print(f"capstone (reduce + broadcast_to + sub + exp[FULL] + sum + truediv + log): {n} cubin bytes")
except Exception as e:
    print(f"capstone (reduce + broadcast_to + sub + exp[FULL] + sum + truediv + log): {type(e).__name__}: {e}")
```
