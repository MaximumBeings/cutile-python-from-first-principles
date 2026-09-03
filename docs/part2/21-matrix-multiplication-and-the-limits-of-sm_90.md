# Chapter 21: Matrix Multiplication and the Limits of sm_90

> "Every kernel this book has written treats a tile's dimensions as independent of each other — an elementwise operation touches one position at a time, a reduction collapses one axis without asking what the others hold. This chapter introduces the first operations where that stops being true, and along the way, discovers that this book's own compilation target has quietly been a variable all along."

**What you will understand by the end of this chapter:**

- `ct.matmul`: the most permissive member of cuTile Python's matrix-multiplication family, accepting 1D (dot product), 2D, and 3D batched-with-broadcast tiles, promoting mixed input dtypes to a common type exactly the way Chapter 17's ordinary arithmetic family does
- `ct.mma(x, y, acc)`: a fused multiply-accumulate that computes `(x @ y) + acc` as one operation, restricted to 2D or 3D inputs only, and — the sharper divergence from `matmul` — one that explicitly refuses to promote mismatched `x`/`y` dtypes, with one documented, verified exception: `int8` paired with `uint8`
- `mma`'s documented input-dtype-to-accumulator-dtype table, confirmed directly across three input dtypes: `float16` accepts either `float16` or `float32` accumulation, while `bfloat16` and `tfloat32` both accept only `float32`
- That `use_fast_acc=True` is a hard, compile-time requirement — `fp8` input dtypes only — enforced with a real `TileTypeError`, not the "silently ignored" behavior its docstring describes for a completely different precondition (which GPU architecture is targeted)
- That this book's very first cross-architecture comparison reveals dtype support itself depends on the `gpu_code` string passed to `export_kernel` — the exact same kernel that compiles cleanly for `sm_90` raises a brand new exception type, `TileUnsupportedFeatureError`, when compiled for `sm_80` or `sm_89`
- That `ct.mma_scaled`'s own docstring marks its examples "skip if not Blackwell or newer," and that this book can confirm that claim directly: the identical block-scaled kernel that fails on `sm_90` compiles cleanly on `sm_100`

**What you need to know first:**

- Chapter 17's broadcasting and dtype-promotion vocabulary, and Chapter 19's finding that a single operation family can reject or accept a given dtype differently depending on exactly which function is called.
- Chapter 10's naming-confound rule, applied throughout this chapter's dtype-table testing.
- No new environment setup for most of this chapter: the same `export_kernel`-only, driver-free compilation workflow as ever. What IS new, for the first time in twenty chapters, is treating `gpu_code` itself as something worth varying rather than a fixed `"sm_90"` passed without a second thought.

## 21.1 ct.matmul: Shapes, Broadcasting, and Promotion

### Intuition

`ct.matmul`'s docstring accepts `x` and `y` as "1D, 2D, or 3D" — the widest shape tolerance of any operation this book has covered. A 2D-by-2D call is ordinary matrix multiplication; a 1D-by-1D call is a dot product, reducing two vectors to a single scalar; a 3D-by-2D call batches multiple matrix multiplications, broadcasting the missing batch dimension the same way Chapter 17's arithmetic family broadcasts missing axes. The docstring also states that mismatched input dtypes "will first be promoted to common dtype" — familiar, Chapter-17-shaped language, worth confirming directly rather than assuming it holds for an operation this structurally different from elementwise arithmetic.

### Background

Testing all three shape arities against `ct.matmul` directly, then testing a mixed-dtype call against two different declared output dtypes — one matching the promoted result, one that would only be correct if no promotion had occurred — establishes both claims at once.

### Worked Example 21.1.1 — matmul across 1D, 2D, 3D, and mixed dtypes

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

# ct.matmul accepts 1D, 2D, or 3D tiles, per its own docstring. 2D x 2D
# is ordinary matrix multiplication.
sig2d = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul2d(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
print(f"matmul 2D (8,16)x(16,8): {compile_bytes(kernel_matmul2d, sig2d)} cubin bytes")

# 1D x 1D is a dot product -- matmul's most permissive shape, one this
# book's other operations never had an analog of.
sig1d = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul1d(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (16,))
    y = ct.load(b, (0,), (16,))
    z = ct.matmul(x, y)
    ct.store(c, (0,), ct.full((1,), z, dtype=ct.float32))
print(f"matmul 1D dot product (16,)x(16,): {compile_bytes(kernel_matmul1d, sig1d)} cubin bytes")

# Batched: 3D x 2D with broadcast, per matmul's own docstring example --
# the batch dimension of x broadcasts against y's lack of one.
sig_batch = ct.compilation.KernelSignature(
    [array_param(3), array_param(2), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul_batch(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0, 0), z)
print(f"matmul batched 3D(2,8,16) x 2D(16,8): {compile_bytes(kernel_matmul_batch, sig_batch)} cubin bytes")

# Mixed dtype: float16 x float32 -- matmul's docstring says operands
# "will first be promoted to common dtype," the same promotion vocabulary
# Chapter 17 documented for the ordinary arithmetic family.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(2, ct.float16), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul_mixed(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
print(f"matmul(float16, float32) -> float32 output: {compile_bytes(kernel_matmul_mixed, sig_mixed)} cubin bytes")

# The same mixed-dtype call, but declaring a float16 output instead: if
# the promoted result is genuinely float32, a float16-declared output
# should now be rejected as a mismatch -- the same store-dtype check
# this book has seen since Chapter 3.
sig_mixed_f16out = ct.compilation.KernelSignature(
    [array_param(2, ct.float16), array_param(2, ct.float32), array_param(2, ct.float16), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul_mixed_f16out(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_matmul_mixed_f16out, sig_mixed_f16out)
    print(f"matmul(float16, float32) -> float16 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"matmul(float16, float32) -> float16 output: {type(e).__name__}: {e}")
```

Genuinely run:

```
matmul 2D (8,16)x(16,8): 31912 cubin bytes
matmul 1D dot product (16,)x(16,): 29472 cubin bytes
matmul batched 3D(2,8,16) x 2D(16,8): 36536 cubin bytes
matmul(float16, float32) -> float32 output: 32040 cubin bytes
matmul(float16, float32) -> float16 output: TileTypeError: Stored tile is incompatible with array's dtype: cannot implicitly cast float32 to float16
Unknown location
  "/tmp/ch21/01_matmul_shapes_broadcast_and_promotion.py", line 90, col 5-26, in kernel_matmul_mixed_f16out:
        ct.store(c, (0, 0), z)
        ^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

All three shapes compile cleanly, and the mixed-dtype call promotes exactly as documented: the `float32`-declared output accepts the result, and the `float16`-declared output is rejected with the same "cannot implicitly cast" store-mismatch error this book has seen for ordinary elementwise mismatches since Chapter 3. `ct.matmul` behaves, in every respect tested here, like a well-mannered member of Chapter 17's broadcasting-and-promoting family — its only real novelty is the two-dimensional structure of what it actually computes, not any new rule about how dtypes or shapes are reconciled. That uneventfulness is itself worth noting going into the next section: `ct.mma`, the next member of this family, is about to break both patterns.

## 21.2 ct.mma: Where the Family Stops Promoting

### Intuition

`ct.mma(x, y, acc)` computes `(x @ y) + acc` as a single fused operation, preserving `acc`'s own dtype in the result. Its docstring restricts `x` and `y` to "2D or 3D" — no 1D dot-product case, unlike `matmul`. More strikingly, it states plainly: "If `x` and `y` have different dtype, they will NOT be promoted to common dtype" — the opposite of what `matmul`'s docstring just said, and worth testing directly rather than taking on faith, especially after Chapters 18 and 19 found operation families whose members diverge from their own shared-sounding documentation in exactly this kind of way.

### Background

Three things to test: whether a 1D call to `mma` is genuinely rejected, whether a mixed-dtype call is genuinely rejected, and — if the rejection message names a specific exception, as several of this book's error messages have — whether that named exception actually holds up under a direct test.

### Worked Example 21.2.1 — mma's shape restriction and its one documented dtype exception

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

# ct.mma computes (x @ y) + acc as a single fused operation, preserving
# acc's dtype. Basic 2D x 2D usage first.
sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_basic(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma 2D basic (float32/float32/float32): {compile_bytes(kernel_mma_basic, sig)} cubin bytes")

# mma's docstring restricts x and y to 2D or 3D -- unlike matmul's
# 1D/2D/3D. Does a 1D call actually get rejected?
sig1d = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_1d(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (16,))
    y = ct.load(b, (0,), (16,))
    acc = ct.load(accsrc, (0,), (1,))
    z = ct.mma(x, y, acc)
    ct.store(c, (0,), z)
try:
    n = compile_bytes(kernel_mma_1d, sig1d)
    print(f"mma 1D: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma 1D: {type(e).__name__}: {e}")

# mma's docstring states plainly: "If x and y have different dtype, they
# will NOT be promoted to common dtype" -- the opposite of matmul's own
# documented behavior for the identical-looking mixed-dtype case.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(2, ct.float16), array_param(2, ct.float32), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_mixed(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_mma_mixed, sig_mixed)
    print(f"mma(float16 x, float32 y) mixed dtype: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(float16 x, float32 y) mixed dtype: {type(e).__name__}: {e}")

# The rejection message above named a specific exception: "unless they
# are int8/uint8". Does mma actually accept x=int8 paired with y=uint8,
# confirming that carved-out exception holds up rather than being an
# unreachable caveat in the error text?
sig_int8_uint8 = ct.compilation.KernelSignature(
    [array_param(2, ct.int8), array_param(2, ct.uint8), array_param(2, ct.int32), array_param(2, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_int8_uint8(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_mma_int8_uint8, sig_int8_uint8)
    print(f"mma(int8 x, uint8 y) -> int32 acc: {n} cubin bytes")
except Exception as e:
    print(f"mma(int8 x, uint8 y) -> int32 acc: {type(e).__name__}: {e}")
```

Genuinely run:

```
mma 2D basic (float32/float32/float32): 36160 cubin bytes
mma 1D: TileTypeError: Expect shape of `x` to be at least 2D, got (16,)
  "/tmp/ch21/02_mma_diverges_from_matmul.py", line 43, col 9-25, in kernel_mma_1d:
        z = ct.mma(x, y, acc)
            ^^^^^^^^^^^^^^^^^

mma(float16 x, float32 y) mixed dtype: TileTypeError: x and y must have the same dtype unless they are int8/uint8, got float16 float32
  "/tmp/ch21/02_mma_diverges_from_matmul.py", line 64, col 9-25, in kernel_mma_mixed:
        z = ct.mma(x, y, acc)
            ^^^^^^^^^^^^^^^^^

mma(int8 x, uint8 y) -> int32 acc: 35904 cubin bytes
```

### Discussion

Every prediction holds. The 1D call is rejected outright, with an error that names the exact shape it received — `(16,)` — and states the requirement in the compiler's own words: "at least 2D." The mixed `float16`/`float32` call is likewise rejected, and the error message does something Chapter 19's rejection messages never did: it names its own exception inline, "unless they are `int8`/`uint8`." Testing that exact carved-out pairing directly confirms it is real, not just a phrase in an error string that nothing ever actually reaches — `mma(int8, uint8)` compiles cleanly against an `int32` accumulator. This is a small but genuine methodological point this book has practiced since Chapter 8's `RoundingMode` testing: an error message's own stated exception is a claim like any other, and claims get tested, not assumed.

## 21.3 The Accumulator Dtype Table, Confirmed

### Intuition

`mma`'s docstring includes a table pairing each input dtype with the accumulator dtypes it allows: `float16` inputs may accumulate into either `float16` or `float32`; `bfloat16` inputs may accumulate only into `float32`; `tfloat32` inputs likewise only into `float32`. Three input dtypes, one docstring table, and — following this chapter's practice so far — worth confirming directly rather than trusting the table's prose.

### Background

For each of the three input dtypes, this section tries both accumulator dtypes the table's `float16` row allows, then confirms the two dtypes the table restricts to `float32`-only reject a `float16` accumulator specifically.

### Worked Example 21.3.1 — float16, bfloat16, and tfloat32 against their allowed accumulator dtypes

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

def make_sig(xy_dtype, acc_dtype):
    return ct.compilation.KernelSignature(
        [array_param(2, xy_dtype), array_param(2, xy_dtype), array_param(2, acc_dtype), array_param(2, acc_dtype), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# mma's docstring table pairs each input dtype with the accumulator
# dtypes it allows. f16 input allows EITHER f16 or f32 acc.
@ct.kernel
def kernel_f16_f16acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(f16, f16) -> f16 acc: {compile_bytes(kernel_f16_f16acc, make_sig(ct.float16, ct.float16))} cubin bytes")

@ct.kernel
def kernel_f16_f32acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(f16, f16) -> f32 acc: {compile_bytes(kernel_f16_f32acc, make_sig(ct.float16, ct.float32))} cubin bytes")

# bf16 input allows ONLY f32 acc per the table -- f16 acc should reject.
@ct.kernel
def kernel_bf16_f32acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(bf16, bf16) -> f32 acc: {compile_bytes(kernel_bf16_f32acc, make_sig(ct.bfloat16, ct.float32))} cubin bytes")

@ct.kernel
def kernel_bf16_f16acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_bf16_f16acc, make_sig(ct.bfloat16, ct.float16))
    print(f"mma(bf16, bf16) -> f16 acc: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(bf16, bf16) -> f16 acc: {type(e).__name__}: {e}")

# tf32 input allows ONLY f32 acc per the table -- the same restriction.
@ct.kernel
def kernel_tf32_f32acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(tf32, tf32) -> f32 acc: {compile_bytes(kernel_tf32_f32acc, make_sig(ct.tfloat32, ct.float32))} cubin bytes")

@ct.kernel
def kernel_tf32_f16acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_tf32_f16acc, make_sig(ct.tfloat32, ct.float16))
    print(f"mma(tf32, tf32) -> f16 acc: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(tf32, tf32) -> f16 acc: {type(e).__name__}: {e}")
```

Genuinely run:

```
mma(f16, f16) -> f16 acc: 37440 cubin bytes
mma(f16, f16) -> f32 acc: 37312 cubin bytes
mma(bf16, bf16) -> f32 acc: 37824 cubin bytes
mma(bf16, bf16) -> f16 acc: TileTypeError: Unsupported acc dtype float16, supported dtypes are (<DType 'float32'>,)
  "/tmp/ch21/03_mma_acc_dtype_table.py", line 60, col 9-25, in kernel_bf16_f16acc:
        z = ct.mma(x, y, acc)
            ^^^^^^^^^^^^^^^^^

mma(tf32, tf32) -> f32 acc: 36416 cubin bytes
mma(tf32, tf32) -> f16 acc: TileTypeError: Unsupported acc dtype float16, supported dtypes are (<DType 'float32'>,)
  "/tmp/ch21/03_mma_acc_dtype_table.py", line 85, col 9-25, in kernel_tf32_f16acc:
        z = ct.mma(x, y, acc)
            ^^^^^^^^^^^^^^^^^
```

### Discussion

The table holds exactly as documented across all five tested combinations. `float16` accepts both accumulator dtypes the table lists for it; `bfloat16` and `tfloat32` both accept `float32` and both reject `float16` with the identical error message, which itself states the supported dtype set as a literal Python tuple: `(<DType 'float32'>,)`. This is a rare case in this book where a documented table survives direct, systematic testing without a single surprise — a useful reminder, after two chapters full of docstring/reality gaps, that not every claim in cuTile Python's documentation is aspirational. Some of them are just accurate.

## 21.4 use_fast_acc: A Hard Requirement, Not a Silent One

### Intuition

`mma`'s `use_fast_acc` parameter docstring packs two separate preconditions into one paragraph: it "requires fp8 input dtypes (`float8_e4m3fn` or `float8_e5m2`)," and it "currently only has an effect on Hopper GPUs; silently ignored on other architectures." Read quickly, both clauses might sound like the same kind of soft, best-effort behavior — try it anywhere, and it either works or quietly does nothing. Testing the dtype precondition directly settles whether that reading is correct.

### Background

Passing `use_fast_acc=True` with ordinary `float32` inputs, then with the documented `float8_e4m3fn` inputs, isolates which of the two preconditions actually gets enforced as a hard compile-time check.

### Worked Example 21.4.1 — use_fast_acc against a non-fp8 dtype, then against fp8

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

sig_f32 = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# mma's docstring says use_fast_acc "requires fp8 input dtypes" and is
# "silently ignored on other architectures." Those are two different
# preconditions in one paragraph -- does the dtype requirement get
# enforced the same silent way, or does it raise a real compile error?
@ct.kernel
def kernel_fast_acc_f32(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc, use_fast_acc=True)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_fast_acc_f32, sig_f32)
    print(f"mma(float32, use_fast_acc=True): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(float32, use_fast_acc=True): {type(e).__name__}: {e}")

# use_fast_acc=True with genuine fp8 (float8_e4m3fn) inputs and f32 acc
# -- the documented, intended use case, compiled for sm_90 (Hopper),
# the one architecture use_fast_acc's docstring says it actually affects.
sig_fp8 = ct.compilation.KernelSignature(
    [array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e4m3fn), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fast_acc_fp8(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc, use_fast_acc=True)
    ct.store(c, (0, 0), z)
print(f"mma(float8_e4m3fn, use_fast_acc=True) -> f32 acc on sm_90: {compile_bytes(kernel_fast_acc_fp8, sig_fp8)} cubin bytes")
```

Genuinely run:

```
mma(float32, use_fast_acc=True): TileTypeError: use_fast_acc is only supported for fp8 input dtypes (float8_e4m3fn, float8_e5m2), got float32
  "/tmp/ch21/04_use_fast_acc_requirement.py", line 30, col 9-44, in kernel_fast_acc_f32:
        z = ct.mma(x, y, acc, use_fast_acc=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mma(float8_e4m3fn, use_fast_acc=True) -> f32 acc on sm_90: 42048 cubin bytes
```

### Discussion

The dtype precondition is enforced as a hard `TileTypeError`, not silently ignored — the "silently ignored" language in `use_fast_acc`'s docstring turns out to apply specifically and only to the architecture precondition (Hopper versus everything else), a claim this book has no way to test yet, since every chapter through Chapter 20 compiled exclusively for `gpu_code="sm_90"` without ever asking whether that choice mattered. `use_fast_acc=True` with genuine `float8_e4m3fn` inputs compiles cleanly on `sm_90` — but the phrase "on `sm_90`" in that sentence is new. No earlier chapter needed to say which architecture a result held for, because none of them had ever varied it. That is exactly what the next section does.

## 21.5 One Kernel, Two Architectures

### Intuition

Twenty chapters of this book passed `gpu_code="sm_90"` to `export_kernel` without treating it as a variable worth investigating — every dtype rule, every shape restriction, every promotion behavior was implicitly "on `sm_90`," a qualifier this book never needed to state because it never changed. `use_fast_acc`'s own docstring just mentioned "Hopper GPUs" and "other architectures" as if compiled behavior could genuinely differ by target. Does it?

### Background

`sm_80` (Ampere) and `sm_89` (Ada Lovelace) both predate `sm_90` (Hopper); `sm_100` (Blackwell) comes after. Compiling the exact same `float8_e4m3fn` kernel from Section 21.4 against all four architecture strings, changing nothing else, isolates whether `gpu_code` alone can change what compiles.

### Worked Example 21.5.1 — the same fp8 kernel across sm_80, sm_89, sm_90, and sm_100

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, gpu_code):
    sig = ct.compilation.KernelSignature(
        [array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e4m3fn), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

# Every chapter before this one compiled exclusively for gpu_code="sm_90"
# and treated dtype/shape acceptance as a fixed property of the compiler.
# This is the first time this book has varied gpu_code itself. The exact
# same kernel -- fp8 inputs, use_fast_acc=True, f32 acc -- compiled
# against four different architecture strings.
@ct.kernel
def kernel_fast_acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc, use_fast_acc=True)
    ct.store(c, (0, 0), z)

for gpu_code in ["sm_80", "sm_89", "sm_90", "sm_100"]:
    try:
        n = compile_bytes(kernel_fast_acc, gpu_code)
        print(f"mma(fp8, use_fast_acc=True) on {gpu_code}: {n} cubin bytes")
    except Exception as e:
        print(f"mma(fp8, use_fast_acc=True) on {gpu_code}: {type(e).__name__}: {e}")
```

Genuinely run:

```
mma(fp8, use_fast_acc=True) on sm_80: TileUnsupportedFeatureError: float8_e4m3fn is not supported on sm_80
  "/tmp/ch21/05_dtype_support_varies_by_architecture.py", lines 25--31, col 1-66, in kernel_fast_acc:
    def kernel_fast_acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mma(fp8, use_fast_acc=True) on sm_89: TileUnsupportedFeatureError: float8_e4m3fn is not supported on sm_89
  "/tmp/ch21/05_dtype_support_varies_by_architecture.py", lines 25--31, col 1-66, in kernel_fast_acc:
    def kernel_fast_acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mma(fp8, use_fast_acc=True) on sm_90: 41792 cubin bytes
mma(fp8, use_fast_acc=True) on sm_100: 50744 cubin bytes
```

### Discussion

This is the first genuinely new kind of finding in this book since Chapter 1: the exact same kernel body, the exact same dtypes, the exact same signature — rejected on `sm_80` and `sm_89`, accepted on `sm_90` and `sm_100`, with nothing changed except a single string argument to `export_kernel`. The rejection is also a new exception type this book has never encountered: `TileUnsupportedFeatureError`, distinct from every `TileTypeError` this book has raised since Chapter 1's first `if`-on-a-tile failure. The message names the actual cause plainly — `float8_e4m3fn is not supported on sm_80` — and it is worth being precise about what this rejection is and isn't about: it has nothing to do with `use_fast_acc` specifically. The `float8_e4m3fn` dtype itself is what `sm_80` and `sm_89` cannot represent; `use_fast_acc` was simply along for the ride in this particular kernel. Every dtype-acceptance finding in this book's first twenty chapters was implicitly scoped to `sm_90` without ever saying so, because `sm_90` was the only architecture ever asked. This section is the first time that scoping became visible.

## 21.6 Capstone: mma_scaled Needs Blackwell, and Two Fused Matmuls

### Intuition

`ct.mma_scaled`'s own docstring marks both of its examples `:skipif: not is_blackwell_or_newer()` — an explicit, machine-checkable claim that this operation needs Blackwell-class hardware, the generation after Hopper. Section 21.5 just established that `gpu_code` can change what compiles; this capstone applies that lesson directly to `mma_scaled`, confirming its architecture requirement rather than accepting the `skipif` annotation as documentation alone. The capstone's second half returns to `mma` itself, composing two calls to demonstrate the fusion `mma` exists for: accumulating the results of two separate matrix multiplications into one tile without an explicit intermediate add.

### Background

`mma_scaled` block-scales `x` and `y` along the K dimension before multiplying, using a scale-factor dtype — `float8_e8m0fnu` — that appears nowhere else in this book. Testing the identical block-scaled kernel against `sm_90` and `sm_100` isolates whether that scale-factor dtype, specifically, is what triggers the Blackwell requirement. The chained-`mma` half of the capstone builds `D = A@B + C@E` as two `ct.mma` calls, the second one's `acc` argument being literally the first call's own return value.

### Worked Example 21.6.1 — mma_scaled across architectures, then a chained mma composition

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code="sm_90"):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

# ct.mma_scaled's own docstring marks its examples ":skipif: not
# is_blackwell_or_newer()" -- an explicit claim that this operation needs
# Blackwell-class hardware. Section 20.5 already found dtype support can
# vary by gpu_code; this confirms mma_scaled's scale-factor dtype,
# float8_e8m0fnu, is unsupported on sm_90 (Hopper) but compiles on
# sm_100 (Blackwell), directly matching the docstring's own condition.
sig_scaled = ct.compilation.KernelSignature(
    [array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e8m0fnu),
     array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e8m0fnu),
     array_param(2, ct.float32), array_param(2, ct.float32),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_scaled(x_arr, xs_arr, y_arr, ys_arr, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_arr, (0, 0), (2, 64))
    xs = ct.load(xs_arr, (0, 0), (2, 2))
    y = ct.load(y_arr, (0, 0), (64, 2))
    ys = ct.load(ys_arr, (0, 0), (2, 2))
    acc = ct.load(accsrc, (0, 0), (2, 2))
    z = ct.mma_scaled(x, xs, y, ys, acc)
    ct.store(c, (0, 0), z)

for gpu_code in ["sm_90", "sm_100"]:
    try:
        n = compile_bytes(kernel_mma_scaled, sig_scaled, gpu_code)
        print(f"mma_scaled(fp8, block=32) on {gpu_code}: {n} cubin bytes")
    except Exception as e:
        print(f"mma_scaled(fp8, block=32) on {gpu_code}: {type(e).__name__}: {e}")

# Capstone proper: D = A@B + C@E, computed as two chained ct.mma calls
# sharing one accumulator, rather than two ct.matmul calls and a
# separate elementwise add. This is exactly the fusion mma exists for:
# the second mma's `acc` argument IS the first mma's own result.
sig_chain = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_chained_mma(a_arr, b_arr, c_arr, e_arr, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a = ct.load(a_arr, (0, 0), (8, 16))
    b = ct.load(b_arr, (0, 0), (16, 8))
    c = ct.load(c_arr, (0, 0), (8, 16))
    e = ct.load(e_arr, (0, 0), (16, 8))
    zero_acc = ct.zeros((8, 8), dtype=ct.float32)
    partial = ct.mma(a, b, zero_acc)
    total = ct.mma(c, e, partial)
    ct.store(out, (0, 0), total)
try:
    n = compile_bytes(kernel_chained_mma, sig_chain)
    print(f"chained mma capstone (A@B + C@E via two mma calls): {n} cubin bytes")
except Exception as e:
    print(f"chained mma capstone (A@B + C@E via two mma calls): {type(e).__name__}: {e}")
```

Genuinely run:

```
mma_scaled(fp8, block=32) on sm_90: TileUnsupportedFeatureError: float8_e8m0fnu is not supported on sm_90
  "/tmp/ch21/06_capstone_mma_scaled_and_chained_mma.py", lines 29--37, col 1-92, in kernel_mma_scaled:
    def kernel_mma_scaled(x_arr, xs_arr, y_arr, ys_arr, accsrc, c, tile_size: ct.Constant[int]):
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

mma_scaled(fp8, block=32) on sm_100: 99168 cubin bytes
chained mma capstone (A@B + C@E via two mma calls): 43992 cubin bytes
```

### Discussion

The docstring's own `skipif` condition is confirmed directly rather than trusted: the identical block-scaled kernel fails on `sm_90` with the same `TileUnsupportedFeatureError` this chapter first met in Section 21.5, this time naming `float8_e8m0fnu` — the scale-factor dtype — as the specific cause, and compiles cleanly on `sm_100`. This is the second architecture-dependent rejection this chapter has found, and both share the identical exception type and the identical shape of explanation: not the operation itself, but one specific dtype it touches, unsupported on the older target. The chained-`mma` half compiles cleanly in one attempt, confirming `mma`'s `acc` parameter composes the way its docstring's own phrase "preserves the dtype of `acc`" implies it should: the second call's accumulator is not a fresh zero tile but the first call's actual returned result, fusing two matrix multiplications and an addition into a dataflow this book's earlier chapters would have needed a separate `ct.matmul` and a separate `ct.add` to express. As with every capstone since Chapter 15, a clean compile confirms the composition is well-typed — not that the specific numeric values `A@B + C@E` would produce on real hardware are correct, which remains, as it has since Chapter 1, beyond what this book's `export_kernel`-only environment can check.

## Chapter Summary

This chapter introduced cuTile Python's matrix-multiplication family: `ct.matmul`, the most permissive member, accepting 1D dot products, 2D matrices, and 3D batched-with-broadcast tiles while promoting mixed input dtypes exactly like Chapter 17's ordinary arithmetic family; and `ct.mma`, which restricts inputs to 2D or 3D only and explicitly refuses to promote mismatched `x`/`y` dtypes, with one verified exception — `int8` paired with `uint8` — named directly in its own rejection message. `mma`'s documented input-to-accumulator dtype table survived direct testing across three input dtypes without a single surprise: `float16` accepts either `float16` or `float32` accumulation, while `bfloat16` and `tfloat32` both accept only `float32`. `use_fast_acc=True` turned out to be a hard, `TileTypeError`-enforced requirement for fp8 input dtypes, distinct from the docstring's separate, still-untested claim about silent architecture-dependent behavior. That untested claim led directly to this book's first cross-architecture comparison: the identical fp8 kernel that compiles on `sm_90` and `sm_100` raises a brand new exception type, `TileUnsupportedFeatureError`, on `sm_80` and `sm_89` — the first time in twenty-one chapters that changing `gpu_code` alone, with nothing else different, changed whether a kernel compiled at all. The capstone confirmed `ct.mma_scaled`'s own documented Blackwell requirement directly, finding the identical `TileUnsupportedFeatureError` pattern one more time, before closing with a chained-`mma` kernel fusing two matrix multiplications and an addition into a single dataflow, using `acc` compositionally for the first time in this book.

## Self-Check Questions

1. Section 21.2 found `mma`'s own error message names its int8/uint8 exception directly, and Section 21.2's testing then confirmed that exception actually works. Contrast this with Chapter 19's bitwise-operation error messages, which never named an exception because none of those operations had one. What does an error message that explicitly names its own exception tell you about the reliability of testing that exception, compared to an error message that states a rule with no stated exceptions at all?
2. Section 21.4 distinguished two preconditions packed into one docstring paragraph for `use_fast_acc`: a dtype requirement (hard-enforced) and an architecture behavior (untested until Section 21.5). Why might a library author choose to enforce one precondition as a compile error while leaving the other as silent, architecture-dependent behavior, rather than enforcing both the same way?
3. Section 21.5 found `TileUnsupportedFeatureError` for the first time in this book, distinct from every `TileTypeError` seen since Chapter 1. Based on the name and the specific case that triggered it, what general category of restriction do you think `TileUnsupportedFeatureError` is reserved for, as opposed to `TileTypeError`?
4. Section 21.5's finding — that `sm_80` and `sm_89` reject `float8_e4m3fn` while `sm_90` and `sm_100` accept it — was tested using exactly one kernel body varied only by `gpu_code`. What would you need to test in addition, to know whether this dtype restriction is specific to `mma` with `use_fast_acc=True`, or whether `float8_e4m3fn` is unsupported on `sm_80`/`sm_89` for every operation that touches it?
5. Section 21.6's capstone chains two `ct.mma` calls, with the second call's `acc` argument being the first call's own returned tile. Every earlier chapter's capstones combined operations from different chapters within a single kernel body, but never fed one call's return value directly into another call of the SAME function. What does the fact that this compiled with no special handling suggest about whether `mma`'s `acc` parameter is restricted to freshly-loaded or freshly-constructed tiles, versus any tile of the right shape and dtype regardless of where it came from?

## Where We Go Next

This chapter's real discovery was not any single dtype rule but the fact that `gpu_code` had been a hidden variable all along — twenty chapters of this book's findings were implicitly scoped to `sm_90`, a scoping that only became visible once `ct.mma`'s more exotic corners (`use_fast_acc`, `mma_scaled`) forced the question. `dir(ct)` still lists operations this book has never opened: `permute`, `transpose`, and `expand_dims` extend Chapter 15's reshaping vocabulary; `cat` and `extract` suggest ways of combining or pulling apart tiles this book has never needed; `load_advanced_indexing` and `store_advanced_indexing` sit alongside the `gather()`/`scatter()` convention Chapter 20 introduced, under a name that suggests there is more to that story; and `bitcast`, `pack_to_bytes`, and `unpack_from_bytes` hint at a lower-level view of tile memory than any operation in this book has taken so far, closer to the "explicit `cuda.tile.bitcast()`" that Chapter 19's bitwise-rejection error messages kept recommending without this book ever actually trying it.

## Worked Solutions

1. An error message that explicitly names its own exception — like `mma`'s "unless they are int8/uint8" — converts an assumption into a specific, falsifiable claim: you know exactly what to test, and a clean compile on that exact pairing is a direct confirmation with nothing left ambiguous. An error message that states a rule with no stated exceptions gives you no such target — you can only confirm the rule holds for the cases you happen to try, and you can never be sure whether an untried case is a silent exception the message simply didn't bother mentioning or a case that would fail exactly as the stated rule predicts. Chapter 19's bitwise operations were in the second position: their error messages stated a requirement (same dtype, integer-only) without ever suggesting a carve-out, so this book's confidence in "no exceptions" there rests on the absence of a counterexample across the cases tried, not on a direct textual claim confirmed the way `mma`'s int8/uint8 exception was.

2. A dtype requirement is a structural property the compiler can check before it ever has to reason about the target GPU: either the input tiles are `float8_e4m3fn`/`float8_e5m2` or they are not, and that fact is knowable from the kernel signature alone, making a hard compile-time rejection cheap and unambiguous to enforce. An architecture-dependent behavior is a different kind of thing — `use_fast_acc` describes a performance path that Hopper's tensor cores support and other architectures simply don't have a fast-path instruction for, so "silently ignored" likely reflects that on non-Hopper targets there is no wrong answer to reject: the flag just has nothing to attach to, and refusing to compile over a no-op performance hint would arguably be more surprising than quietly doing the correct, unaccelerated thing. Enforcing the dtype rule protects against a kernel that would produce wrong results category; ignoring the architecture mismatch protects against penalizing a kernel that would still produce correct results everywhere, just not with the requested acceleration on that target.

3. Every `TileTypeError` this book has raised since Chapter 1 concerned something a fixed compiler could reason about from the kernel's own types and shapes alone — mismatched dtypes, wrong ranks, shapes that don't broadcast, integer-only operations given floats. `TileUnsupportedFeatureError`, by contrast, appeared only when the same kernel body, same dtypes, same shapes, produced a different outcome purely because `gpu_code` changed. That pattern suggests `TileUnsupportedFeatureError` is reserved for capability gaps between the compiler's abstract type system and a specific hardware target's actual instruction set — a dtype or feature that is perfectly well-typed and well-formed as a cuTile Python expression, but that the requested architecture's tensor cores or ALUs simply have no instruction to execute. `TileTypeError` answers "is this kernel well-formed?"; `TileUnsupportedFeatureError` answers "can this specific piece of hardware run a well-formed kernel like this one?" — and this book's `export_kernel`-only environment can observe that distinction directly, without ever running a kernel on real silicon, precisely because both are compile-time checks.

4. This chapter's test varied exactly one thing (`gpu_code`) while holding the kernel body, the dtypes, and the operation (`mma` with `use_fast_acc=True`) fixed — which is enough to show that `float8_e4m3fn` support depends on architecture in at least this one context, but not enough to show how far that dependency generalizes. To separate "this dtype is unsupported on `sm_80`/`sm_89` for every operation" from "this specific combination of `mma` and `use_fast_acc` is what's unsupported there," you would need to compile a kernel that uses `float8_e4m3fn` in a different operation entirely — a plain `ct.load`/`ct.store` round-trip with no `mma` involved, or `ct.matmul` instead of `ct.mma`, or ordinary elementwise arithmetic on `float8_e4m3fn` tiles — against the same `sm_80`/`sm_89` targets. If those simpler kernels also raise `TileUnsupportedFeatureError` naming `float8_e4m3fn`, the restriction is about the dtype itself on that architecture; if they compile cleanly and only the `mma`-with-`use_fast_acc` case fails, the restriction is narrower than this chapter's single test can distinguish.

5. The chained-`mma` capstone compiled with no special handling, no explicit cast, and no intermediate `ct.store`/`ct.load` round-trip between the two calls — the first `mma`'s return value was fed directly into the second `mma`'s `acc` parameter as an ordinary Python value passed between two lines of the same kernel body. That this required no accommodation at all suggests `acc` is not restricted to tiles that were freshly loaded from an array or freshly constructed with something like `ct.zeros` — it accepts any tile of the right shape and dtype, regardless of provenance, exactly the way every other operation in this book that takes a tile argument has behaved since Chapter 1. This book's `export_kernel`-only environment confirms the composition type-checks cleanly; it cannot confirm that the fused computation produces the numerically correct sum on real hardware, which — as with every capstone before it — remains outside what a driver-free compilation pass can check.

## Complete Runnable Code

### File: `01_matmul_shapes_broadcast_and_promotion.py`

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

# ct.matmul accepts 1D, 2D, or 3D tiles, per its own docstring. 2D x 2D
# is ordinary matrix multiplication.
sig2d = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul2d(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
print(f"matmul 2D (8,16)x(16,8): {compile_bytes(kernel_matmul2d, sig2d)} cubin bytes")

# 1D x 1D is a dot product -- matmul's most permissive shape, one this
# book's other operations never had an analog of.
sig1d = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul1d(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (16,))
    y = ct.load(b, (0,), (16,))
    z = ct.matmul(x, y)
    ct.store(c, (0,), ct.full((1,), z, dtype=ct.float32))
print(f"matmul 1D dot product (16,)x(16,): {compile_bytes(kernel_matmul1d, sig1d)} cubin bytes")

# Batched: 3D x 2D with broadcast, per matmul's own docstring example --
# the batch dimension of x broadcasts against y's lack of one.
sig_batch = ct.compilation.KernelSignature(
    [array_param(3), array_param(2), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul_batch(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0, 0), z)
print(f"matmul batched 3D(2,8,16) x 2D(16,8): {compile_bytes(kernel_matmul_batch, sig_batch)} cubin bytes")

# Mixed dtype: float16 x float32 -- matmul's docstring says operands
# "will first be promoted to common dtype," the same promotion vocabulary
# Chapter 17 documented for the ordinary arithmetic family.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(2, ct.float16), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul_mixed(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
print(f"matmul(float16, float32) -> float32 output: {compile_bytes(kernel_matmul_mixed, sig_mixed)} cubin bytes")

# The same mixed-dtype call, but declaring a float16 output instead: if
# the promoted result is genuinely float32, a float16-declared output
# should now be rejected as a mismatch -- the same store-dtype check
# this book has seen since Chapter 3.
sig_mixed_f16out = ct.compilation.KernelSignature(
    [array_param(2, ct.float16), array_param(2, ct.float32), array_param(2, ct.float16), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_matmul_mixed_f16out(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_matmul_mixed_f16out, sig_mixed_f16out)
    print(f"matmul(float16, float32) -> float16 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"matmul(float16, float32) -> float16 output: {type(e).__name__}: {e}")
```

### File: `02_mma_diverges_from_matmul.py`

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

# ct.mma computes (x @ y) + acc as a single fused operation, preserving
# acc's dtype. Basic 2D x 2D usage first.
sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_basic(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma 2D basic (float32/float32/float32): {compile_bytes(kernel_mma_basic, sig)} cubin bytes")

# mma's docstring restricts x and y to 2D or 3D -- unlike matmul's
# 1D/2D/3D. Does a 1D call actually get rejected?
sig1d = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_1d(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (16,))
    y = ct.load(b, (0,), (16,))
    acc = ct.load(accsrc, (0,), (1,))
    z = ct.mma(x, y, acc)
    ct.store(c, (0,), z)
try:
    n = compile_bytes(kernel_mma_1d, sig1d)
    print(f"mma 1D: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma 1D: {type(e).__name__}: {e}")

# mma's docstring states plainly: "If x and y have different dtype, they
# will NOT be promoted to common dtype" -- the opposite of matmul's own
# documented behavior for the identical-looking mixed-dtype case.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(2, ct.float16), array_param(2, ct.float32), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_mixed(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_mma_mixed, sig_mixed)
    print(f"mma(float16 x, float32 y) mixed dtype: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(float16 x, float32 y) mixed dtype: {type(e).__name__}: {e}")

# The rejection message above named a specific exception: "unless they
# are int8/uint8". Does mma actually accept x=int8 paired with y=uint8,
# confirming that carved-out exception holds up rather than being an
# unreachable caveat in the error text?
sig_int8_uint8 = ct.compilation.KernelSignature(
    [array_param(2, ct.int8), array_param(2, ct.uint8), array_param(2, ct.int32), array_param(2, ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_int8_uint8(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_mma_int8_uint8, sig_int8_uint8)
    print(f"mma(int8 x, uint8 y) -> int32 acc: {n} cubin bytes")
except Exception as e:
    print(f"mma(int8 x, uint8 y) -> int32 acc: {type(e).__name__}: {e}")
```

### File: `03_mma_acc_dtype_table.py`

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

def make_sig(xy_dtype, acc_dtype):
    return ct.compilation.KernelSignature(
        [array_param(2, xy_dtype), array_param(2, xy_dtype), array_param(2, acc_dtype), array_param(2, acc_dtype), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# mma's docstring table pairs each input dtype with the accumulator
# dtypes it allows. f16 input allows EITHER f16 or f32 acc.
@ct.kernel
def kernel_f16_f16acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(f16, f16) -> f16 acc: {compile_bytes(kernel_f16_f16acc, make_sig(ct.float16, ct.float16))} cubin bytes")

@ct.kernel
def kernel_f16_f32acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(f16, f16) -> f32 acc: {compile_bytes(kernel_f16_f32acc, make_sig(ct.float16, ct.float32))} cubin bytes")

# bf16 input allows ONLY f32 acc per the table -- f16 acc should reject.
@ct.kernel
def kernel_bf16_f32acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(bf16, bf16) -> f32 acc: {compile_bytes(kernel_bf16_f32acc, make_sig(ct.bfloat16, ct.float32))} cubin bytes")

@ct.kernel
def kernel_bf16_f16acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_bf16_f16acc, make_sig(ct.bfloat16, ct.float16))
    print(f"mma(bf16, bf16) -> f16 acc: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(bf16, bf16) -> f16 acc: {type(e).__name__}: {e}")

# tf32 input allows ONLY f32 acc per the table -- the same restriction.
@ct.kernel
def kernel_tf32_f32acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
print(f"mma(tf32, tf32) -> f32 acc: {compile_bytes(kernel_tf32_f32acc, make_sig(ct.tfloat32, ct.float32))} cubin bytes")

@ct.kernel
def kernel_tf32_f16acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_tf32_f16acc, make_sig(ct.tfloat32, ct.float16))
    print(f"mma(tf32, tf32) -> f16 acc: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(tf32, tf32) -> f16 acc: {type(e).__name__}: {e}")
```

### File: `04_use_fast_acc_requirement.py`

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

sig_f32 = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# mma's docstring says use_fast_acc "requires fp8 input dtypes" and is
# "silently ignored on other architectures." Those are two different
# preconditions in one paragraph -- does the dtype requirement get
# enforced the same silent way, or does it raise a real compile error?
@ct.kernel
def kernel_fast_acc_f32(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc, use_fast_acc=True)
    ct.store(c, (0, 0), z)
try:
    n = compile_bytes(kernel_fast_acc_f32, sig_f32)
    print(f"mma(float32, use_fast_acc=True): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"mma(float32, use_fast_acc=True): {type(e).__name__}: {e}")

# use_fast_acc=True with genuine fp8 (float8_e4m3fn) inputs and f32 acc
# -- the documented, intended use case, compiled for sm_90 (Hopper),
# the one architecture use_fast_acc's docstring says it actually affects.
sig_fp8 = ct.compilation.KernelSignature(
    [array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e4m3fn), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fast_acc_fp8(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc, use_fast_acc=True)
    ct.store(c, (0, 0), z)
print(f"mma(float8_e4m3fn, use_fast_acc=True) -> f32 acc on sm_90: {compile_bytes(kernel_fast_acc_fp8, sig_fp8)} cubin bytes")
```

### File: `05_dtype_support_varies_by_architecture.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, gpu_code):
    sig = ct.compilation.KernelSignature(
        [array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e4m3fn), array_param(2, ct.float32), array_param(2, ct.float32), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

# Every chapter before this one compiled exclusively for gpu_code="sm_90"
# and treated dtype/shape acceptance as a fixed property of the compiler.
# This is the first time this book has varied gpu_code itself. The exact
# same kernel -- fp8 inputs, use_fast_acc=True, f32 acc -- compiled
# against four different architecture strings.
@ct.kernel
def kernel_fast_acc(a, b, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    acc = ct.load(accsrc, (0, 0), (8, 8))
    z = ct.mma(x, y, acc, use_fast_acc=True)
    ct.store(c, (0, 0), z)

for gpu_code in ["sm_80", "sm_89", "sm_90", "sm_100"]:
    try:
        n = compile_bytes(kernel_fast_acc, gpu_code)
        print(f"mma(fp8, use_fast_acc=True) on {gpu_code}: {n} cubin bytes")
    except Exception as e:
        print(f"mma(fp8, use_fast_acc=True) on {gpu_code}: {type(e).__name__}: {e}")
```

### File: `06_capstone_mma_scaled_and_chained_mma.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code="sm_90"):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

# ct.mma_scaled's own docstring marks its examples ":skipif: not
# is_blackwell_or_newer()" -- an explicit claim that this operation needs
# Blackwell-class hardware. Section 20.5 already found dtype support can
# vary by gpu_code; this confirms mma_scaled's scale-factor dtype,
# float8_e8m0fnu, is unsupported on sm_90 (Hopper) but compiles on
# sm_100 (Blackwell), directly matching the docstring's own condition.
sig_scaled = ct.compilation.KernelSignature(
    [array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e8m0fnu),
     array_param(2, ct.float8_e4m3fn), array_param(2, ct.float8_e8m0fnu),
     array_param(2, ct.float32), array_param(2, ct.float32),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_mma_scaled(x_arr, xs_arr, y_arr, ys_arr, accsrc, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_arr, (0, 0), (2, 64))
    xs = ct.load(xs_arr, (0, 0), (2, 2))
    y = ct.load(y_arr, (0, 0), (64, 2))
    ys = ct.load(ys_arr, (0, 0), (2, 2))
    acc = ct.load(accsrc, (0, 0), (2, 2))
    z = ct.mma_scaled(x, xs, y, ys, acc)
    ct.store(c, (0, 0), z)

for gpu_code in ["sm_90", "sm_100"]:
    try:
        n = compile_bytes(kernel_mma_scaled, sig_scaled, gpu_code)
        print(f"mma_scaled(fp8, block=32) on {gpu_code}: {n} cubin bytes")
    except Exception as e:
        print(f"mma_scaled(fp8, block=32) on {gpu_code}: {type(e).__name__}: {e}")

# Capstone proper: D = A@B + C@E, computed as two chained ct.mma calls
# sharing one accumulator, rather than two ct.matmul calls and a
# separate elementwise add. This is exactly the fusion mma exists for:
# the second mma's `acc` argument IS the first mma's own result.
sig_chain = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_chained_mma(a_arr, b_arr, c_arr, e_arr, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a = ct.load(a_arr, (0, 0), (8, 16))
    b = ct.load(b_arr, (0, 0), (16, 8))
    c = ct.load(c_arr, (0, 0), (8, 16))
    e = ct.load(e_arr, (0, 0), (16, 8))
    zero_acc = ct.zeros((8, 8), dtype=ct.float32)
    partial = ct.mma(a, b, zero_acc)
    total = ct.mma(c, e, partial)
    ct.store(out, (0, 0), total)
try:
    n = compile_bytes(kernel_chained_mma, sig_chain)
    print(f"chained mma capstone (A@B + C@E via two mma calls): {n} cubin bytes")
except Exception as e:
    print(f"chained mma capstone (A@B + C@E via two mma calls): {type(e).__name__}: {e}")
```
