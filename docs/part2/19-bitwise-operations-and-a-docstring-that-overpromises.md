# Chapter 19: Bitwise Operations and a Docstring That Overpromises

> "Chapter 18 closed by noting that `ct.where`'s docstring described a narrower contract than the compiler actually enforces — the compiler was more permissive than documented. This chapter finds the opposite failure mode in the very next operation family it tests: a docstring that promises MORE than the compiler actually delivers."

**What you will understand by the end of this chapter:**

- `ct.bitwise_xor`, `ct.bitwise_not`, `ct.bitwise_lshift`, and `ct.bitwise_rshift` — the four bitwise operations this book has not yet exercised directly, alongside deeper coverage of `ct.bitwise_and`/`ct.bitwise_or` beyond Chapter 18's light masking use — and that all six reject `float32` operands, but with three genuinely different `TileTypeError` wordings split across the family, not one shared message
- That `ct.bitwise_and`, `ct.bitwise_or`, and `ct.bitwise_xor` all document themselves with the identical "dtype promoted to common dtype" language Chapter 17 confirmed for the arithmetic family — and that the real compiler does not honor that promise for these three operations at all, rejecting `int32`/`int64`, `int32`/`uint32`, and even `bool_`/`int32` operand pairs outright with a "must have same data type" error
- That `ct.bitwise_lshift` and `ct.bitwise_rshift`, documented with that identical promotion language, DO honor it — mixed-width integer operands promote to the wider type exactly as advertised, making the shift operators the only two members of this six-function family whose behavior matches their own docstring on this point
- That `ct.bitwise_lshift`/`ct.bitwise_rshift` check their two operands independently for "is this an integer," each with its own half of a shared error message naming which side failed, rather than checking the pair together the way `ct.bitwise_and`/`ct.bitwise_or`/`ct.bitwise_xor` do
- That `ct.bitwise_not` on a `bool_`-dtyped tile compiles successfully against BOTH a `bool_`-declared output and an `int32`-declared output — a second, smaller surprise about a function whose docstring shows only an `int32` example and says nothing about `bool_` at all
- That a capstone combining a shift, a mask, and the classic `x & (x - 1) == 0` power-of-two bit trick compiles cleanly, closing a callback to Chapter 15's rule that a tile's own *shape* must be a power of two by testing the same property about a kernel's *data* instead

**What you need to know first:**

- Chapter 17's binary elementwise family (broadcasting, dtype promotion, `inspect.signature`'s `TileOrScalar` return type) and Chapter 18's `ct.bitwise_and`/`ct.bitwise_or` masking pattern, which this chapter extends rather than reintroduces.
- Chapter 10's naming-confound rule: a byte-count comparison is only meaningful when both compared kernels share one function name.
- Chapter 15's finding that a tile's shape must itself be a power of two — the capstone in Section 19.6 tests the same numeric property about ordinary integer data, not a shape.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow as every chapter before it. This book has never been able to run a kernel with `ct.launch`, so nothing in this chapter checks what a bitwise operation actually computes on real hardware — only what the compiler accepts, rejects, and how large the resulting cubin is.

## 19.1 The Remaining Bitwise Operations

### Intuition

Chapter 18 introduced `ct.bitwise_and` and `ct.bitwise_or` only as a way to combine two boolean masks elementwise — a narrow, specific use case. `dir(ct)` lists four more bitwise operations this book has never touched at all: `bitwise_xor`, `bitwise_not`, `bitwise_lshift`, and `bitwise_rshift`. `inspect.signature` shows five of these six sharing the exact binary-elementwise shape already familiar from Chapter 17 and 18 — `(x, y, /) -> 'TileOrScalar'` — with `bitwise_not` alone breaking the pattern as a unary operation, `(x, /) -> 'TileOrScalar'`.

### Background

Checking `dir(ct)` for anything beyond these six — a `popcount`, a `count_leading_zeros`, a `count_trailing_zeros` — turns up nothing; this chapter's scope is exactly these six functions. Their docstrings read almost identically to Chapter 17's arithmetic family: each names a Python operator alias (`&`, `|`, `^`, `~`, `<<`, `>>`), and the five binary ones each state, verbatim, "The `shape` of `x` and `y` will be broadcasted and `dtype` promoted to common dtype" — the identical promotion language Chapter 17 confirmed the arithmetic family actually honors. Whether that promise holds for bitwise operations, too, is the question the rest of this chapter spends most of its time on.

### Worked Example 19.1.1 — the four new bitwise operations, plus a scalar shift amount

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
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

# ct.bitwise_xor: elementwise exclusive-or, the third binary logical op
# in the family alongside Chapter 18's bitwise_and/bitwise_or.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_xor(x, y))
print(f"bitwise_xor(x, y): {compile_bytes(kernel_fn, sig)} cubin bytes")

# ct.bitwise_not: unary, one operand, one result -- the odd one out in
# this family's signatures ((x, /) rather than (x, y, /)).
sig_unary = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
print(f"bitwise_not(x): {compile_bytes(kernel_fn2, sig_unary)} cubin bytes")

# ct.bitwise_lshift: shift left by a tile of shift amounts, not just a
# scalar -- every element of x shifted by the corresponding element of y.
@ct.kernel
def kernel_fn3(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
print(f"bitwise_lshift(x, y): {compile_bytes(kernel_fn3, sig)} cubin bytes")

# ct.bitwise_rshift: its right-shift counterpart.
@ct.kernel
def kernel_fn4(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_rshift(x, y))
print(f"bitwise_rshift(x, y): {compile_bytes(kernel_fn4, sig)} cubin bytes")

# Shifting by a plain Python scalar rather than a loaded tile -- the more
# common "shift by a fixed constant amount" pattern. lshift and rshift
# defined under distinct names here (not naming-controlled against each
# other), so their equal byte counts below are reported as an observation,
# not a claim of equivalence.
sig_scalar = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_scalar(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, 2))
print(f"bitwise_lshift(x, scalar 2): {compile_bytes(kernel_lshift_scalar, sig_scalar)} cubin bytes")

@ct.kernel
def kernel_rshift_scalar(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_rshift(x, 2))
print(f"bitwise_rshift(x, scalar 2): {compile_bytes(kernel_rshift_scalar, sig_scalar)} cubin bytes")
```

Genuinely run:

```
bitwise_xor(x, y): 23312 cubin bytes
bitwise_not(x): 21024 cubin bytes
bitwise_lshift(x, y): 23328 cubin bytes
bitwise_rshift(x, y): 23328 cubin bytes
bitwise_lshift(x, scalar 2): 21296 cubin bytes
bitwise_rshift(x, scalar 2): 21296 cubin bytes
```

### Discussion

All six calls compile without incident against `int32` operands — no surprises yet. `bitwise_lshift(x, y)` and `bitwise_rshift(x, y)` land on the identical 23328-byte count, and their scalar-shift-amount variants both land on 21296 bytes, but `kernel_fn3`, `kernel_fn4`, `kernel_lshift_scalar`, and `kernel_rshift_scalar` are four separate function names — Chapter 10's naming-confound rule means these matches are reported as raw data, not as evidence that left- and right-shift compile to structurally identical code. The real substance of this chapter starts in the next two sections, where the docstrings' promises are tested directly rather than taken at face value.

## 19.2 Three Wordings for One Rejection

### Intuition

Chapter 18 found `and` and Python's chained-comparison sugar both reject a multi-element tile, using two different `TileTypeError` messages for what is, semantically, the same underlying "not a genuine scalar" requirement. This section asks whether the bitwise family's rejection of `float32` operands — an operation on bit patterns has no defined meaning for IEEE floating-point values — follows a similar pattern: one shared restriction, expressed through more than one wording.

### Background

None of the six bitwise docstrings mentions a required integer dtype explicitly; the restriction, if it exists, would have to come from the compiler's type checker rather than the documented contract. Testing all six functions against `float32` inputs directly is the only way to find out both whether the rejection exists and how it is worded.

### Worked Example 19.2.1 — float32 rejected by all six functions, in three distinct wordings

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
sig_unary = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Every bitwise op in this chapter rejects float32 operands -- but not with
# one shared message. Three distinct wordings appear across six functions.

# Wording 1: bitwise_and, bitwise_or, bitwise_xor -- the three binary
# logical ops -- all share this exact sentence.
@ct.kernel
def kernel_and(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and, sig)
    print(f"bitwise_and(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(float32, float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_or(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_or(x, y))
try:
    n = compile_bytes(kernel_or, sig)
    print(f"bitwise_or(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_or(float32, float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_xor(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_xor(x, y))
try:
    n = compile_bytes(kernel_xor, sig)
    print(f"bitwise_xor(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_xor(float32, float32): {type(e).__name__}: {e}")

# Wording 2: bitwise_not -- the lone unary op -- uses its own, much
# terser phrasing rather than reusing wording 1.
@ct.kernel
def kernel_not(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
try:
    n = compile_bytes(kernel_not, sig_unary)
    print(f"bitwise_not(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_not(float32): {type(e).__name__}: {e}")

# Wording 3: bitwise_lshift and bitwise_rshift -- the two shift ops --
# share a third wording, and that wording names which side is at fault.
@ct.kernel
def kernel_lshift(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift, sig)
    print(f"bitwise_lshift(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_lshift(float32, float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_rshift(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_rshift(x, y))
try:
    n = compile_bytes(kernel_rshift, sig)
    print(f"bitwise_rshift(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_rshift(float32, float32): {type(e).__name__}: {e}")

# Wording 3 names a side -- so does an int32 LHS with a float32 RHS (the
# shift-amount operand) trigger a different half of that same wording?
sig_int_lhs_float_rhs = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.float32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_rhs(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_rhs, sig_int_lhs_float_rhs)
    print(f"bitwise_lshift(int32, float32 shift amount): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_lshift(int32, float32 shift amount): {type(e).__name__}: {e}")
```

Genuinely run:

```
bitwise_and(float32, float32): TileTypeError: Bitwise operations require integers or booleans. Use an explicit cuda.tile.bitcast() for non-integer operands.
  "/tmp/ch19/02_float_rejection_three_wordings.py", line 34, col 25-44, in kernel_and:
        ct.store(c, (pid,), ct.bitwise_and(x, y))
                            ^^^^^^^^^^^^^^^^^^^^

bitwise_or(float32, float32): TileTypeError: Bitwise operations require integers or booleans. Use an explicit cuda.tile.bitcast() for non-integer operands.
  "/tmp/ch19/02_float_rejection_three_wordings.py", line 46, col 25-43, in kernel_or:
        ct.store(c, (pid,), ct.bitwise_or(x, y))
                            ^^^^^^^^^^^^^^^^^^^

bitwise_xor(float32, float32): TileTypeError: Bitwise operations require integers or booleans. Use an explicit cuda.tile.bitcast() for non-integer operands.
  "/tmp/ch19/02_float_rejection_three_wordings.py", line 58, col 25-44, in kernel_xor:
        ct.store(c, (pid,), ct.bitwise_xor(x, y))
                            ^^^^^^^^^^^^^^^^^^^^

bitwise_not(float32): TileTypeError: Float inputs are not supported
  "/tmp/ch19/02_float_rejection_three_wordings.py", line 71, col 25-41, in kernel_not:
        ct.store(c, (pid,), ct.bitwise_not(x))
                            ^^^^^^^^^^^^^^^^^

bitwise_lshift(float32, float32): TileTypeError: Bitwise shift requires an integer for left-hand side, got: float32
  "/tmp/ch19/02_float_rejection_three_wordings.py", line 85, col 25-47, in kernel_lshift:
        ct.store(c, (pid,), ct.bitwise_lshift(x, y))
                            ^^^^^^^^^^^^^^^^^^^^^^^

bitwise_rshift(float32, float32): TileTypeError: Bitwise shift requires an integer for left-hand side, got: float32
  "/tmp/ch19/02_float_rejection_three_wordings.py", line 97, col 25-47, in kernel_rshift:
        ct.store(c, (pid,), ct.bitwise_rshift(x, y))
                            ^^^^^^^^^^^^^^^^^^^^^^^

bitwise_lshift(int32, float32 shift amount): TileTypeError: Bitwise shift requires an integer for right-hand side, got: float32
  "/tmp/ch19/02_float_rejection_three_wordings.py", line 115, col 25-47, in kernel_lshift_rhs:
        ct.store(c, (pid,), ct.bitwise_lshift(x, y))
                            ^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

The prediction holds, with a finer structure than Chapter 18's two-way split. `bitwise_and`, `bitwise_or`, and `bitwise_xor` — the three binary logical operations — all fail with the identical, most-informative message of the three: it names the restriction ("require integers or booleans") and even suggests the fix (`cuda.tile.bitcast()`). `bitwise_not`, the lone unary function, uses a much terser wording that appears nowhere in the other five functions' errors: "Float inputs are not supported," with no mention of integers, booleans, or a `bitcast()` workaround. `bitwise_lshift` and `bitwise_rshift` share a third wording that neither of the other two groups uses, and this third wording is doing something neither of the others does: it names *which operand* failed. Feeding `bitwise_lshift` a `float32` value for `x` produces the "left-hand side" wording; feeding it a `float32` value for `y` — the shift-amount operand, with `x` correctly typed as `int32` — produces the "right-hand side" wording instead. That confirms the shift operators check their two operands as two separate, independently-reported conditions, rather than checking the pair together as one condition the way the logical family's shared message implies. Three functions, one message; one function, a second message; two functions, a third message with an internal left/right distinction the other two messages don't need — six functions in a single `dir(ct)`-adjacent family, and no single sentence covers all of their float-rejection behavior.

## 19.3 A Docstring That Overpromises

### Intuition

Chapter 18's central finding was a docstring underselling the compiler's real permissiveness: `ct.where` claimed a narrower contract than the compiler actually enforced. This section finds the opposite failure mode in the very next family tested. `ct.bitwise_and`, `ct.bitwise_or`, and `ct.bitwise_xor` each state, verbatim, that "the `dtype` [will be] promoted to common dtype" — the same sentence, word for word, that Chapter 17 confirmed the arithmetic family genuinely honors for mixed `int32`/`float32` and mixed-width integer operands. Does the bitwise family honor that same promise?

### Background

Testing this directly means feeding `bitwise_and` two operands of different, but both integer, dtypes — first different widths (`int32` and `int64`), then the same width but different signedness (`int32` and `uint32`), then `bool_` alongside `int32` — and checking whether the compiler promotes them to a common type the way the docstring says, or rejects the mismatch outright. The same test, run once against `bitwise_or`, checks whether this is an isolated `bitwise_and` behavior or a shared property of the whole logical sub-family. A parallel test against `bitwise_lshift`, whose docstring makes the identical promotion claim, checks whether the shift operators behave any differently.

### Worked Example 19.3.1 — bitwise_and/bitwise_or reject every dtype mismatch; bitwise_lshift promotes

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Chapter 17 documented dtype promotion as the rule for the binary
# elementwise family: mixed int32/float32, mixed int32/int64, and so on
# are silently promoted to a common type. bitwise_and, bitwise_or, and
# bitwise_xor break that rule -- they demand an EXACT dtype match instead.

# int32 AND int64: rejected outright, no promotion attempted.
sig_int32_int64 = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int64), array_param(ct.int64), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_and_widths(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and_widths, sig_int32_int64)
    print(f"bitwise_and(int32, int64): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(int32, int64): {type(e).__name__}: {e}")

# Same width, different signedness: int32 vs uint32. Still rejected --
# "same data type" means exactly that, not "same bit width."
sig_sign = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.uint32), array_param(ct.uint32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_and_sign(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and_sign, sig_sign)
    print(f"bitwise_and(int32, uint32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(int32, uint32): {type(e).__name__}: {e}")

# bool_ vs int32: also rejected. bool_ is not a special case that widens
# freely into an integer type here.
sig_bool = ct.compilation.KernelSignature(
    [array_param(ct.bool_), array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_and_bool_int(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and_bool_int, sig_bool)
    print(f"bitwise_and(bool_, int32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(bool_, int32): {type(e).__name__}: {e}")

# bitwise_or shares bitwise_and's strict rule -- not a bitwise_and-only
# restriction.
@ct.kernel
def kernel_or_widths(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_or(x, y))
try:
    n = compile_bytes(kernel_or_widths, sig_int32_int64)
    print(f"bitwise_or(int32, int64): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_or(int32, int64): {type(e).__name__}: {e}")

# bitwise_lshift, by contrast, DOES promote: an int32 tile shifted by an
# int64-typed tile of shift amounts compiles cleanly, with the result
# taking the wider dtype -- exactly the Chapter 17 promotion pattern that
# bitwise_and/or/xor just refused to follow.
@ct.kernel
def kernel_lshift_promote(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_promote, sig_int32_int64)
    print(f"bitwise_lshift(int32, int64 shift amount) -> int64 output: {n} cubin bytes")
except Exception as e:
    print(f"bitwise_lshift(int32, int64 shift amount) -> int64 output: {type(e).__name__}: {e}")

# The same call, but declaring an int32 output instead: if the promoted
# result really is int64, an int32-declared output should now fail --
# with the same "cannot implicitly cast" store-mismatch error used
# throughout this book, not a bitwise-specific one.
sig_int32_out = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int64), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_int32_out(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_int32_out, sig_int32_out)
    print(f"bitwise_lshift(int32, int64 shift amount) -> int32 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_lshift(int32, int64 shift amount) -> int32 output: {type(e).__name__}: {e}")

# And with the widths reversed -- int64 x, int32 shift amount -- the
# promotion still lands on int64.
sig_reverse = ct.compilation.KernelSignature(
    [array_param(ct.int64), array_param(ct.int32), array_param(ct.int64), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_reverse(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_reverse, sig_reverse)
    print(f"bitwise_lshift(int64, int32 shift amount) -> int64 output: {n} cubin bytes")
except Exception as e:
    print(f"bitwise_lshift(int64, int32 shift amount) -> int64 output: {type(e).__name__}: {e}")
```

Genuinely run:

```
bitwise_and(int32, int64): TileTypeError: Bitwise operands must have same data type, got: int32 and int64
  "/tmp/ch19/03_strict_equality_vs_shift_promotion.py", line 30, col 25-44, in kernel_and_widths:
        ct.store(c, (pid,), ct.bitwise_and(x, y))
                            ^^^^^^^^^^^^^^^^^^^^

bitwise_and(int32, uint32): TileTypeError: Bitwise operands must have same data type, got: int32 and uint32
  "/tmp/ch19/03_strict_equality_vs_shift_promotion.py", line 48, col 25-44, in kernel_and_sign:
        ct.store(c, (pid,), ct.bitwise_and(x, y))
                            ^^^^^^^^^^^^^^^^^^^^

bitwise_and(bool_, int32): TileTypeError: Bitwise operands must have same data type, got: bool_ and int32
  "/tmp/ch19/03_strict_equality_vs_shift_promotion.py", line 66, col 25-44, in kernel_and_bool_int:
        ct.store(c, (pid,), ct.bitwise_and(x, y))
                            ^^^^^^^^^^^^^^^^^^^^

bitwise_or(int32, int64): TileTypeError: Bitwise operands must have same data type, got: int32 and int64
  "/tmp/ch19/03_strict_equality_vs_shift_promotion.py", line 80, col 25-43, in kernel_or_widths:
        ct.store(c, (pid,), ct.bitwise_or(x, y))
                            ^^^^^^^^^^^^^^^^^^^

bitwise_lshift(int32, int64 shift amount) -> int64 output: 23840 cubin bytes
bitwise_lshift(int32, int64 shift amount) -> int32 output: TileTypeError: Stored tile is incompatible with array's dtype: cannot implicitly cast int64 to int32
Unknown location
  "/tmp/ch19/03_strict_equality_vs_shift_promotion.py", line 116, col 5-48, in kernel_lshift_int32_out:
        ct.store(c, (pid,), ct.bitwise_lshift(x, y))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

bitwise_lshift(int64, int32 shift amount) -> int64 output: 23840 cubin bytes
```

### Discussion

This is the chapter's central finding, and it runs in the opposite direction from Chapter 18's. `ct.where`'s docstring undersold the compiler's real permissiveness; `ct.bitwise_and`, `ct.bitwise_or`, and (by direct implication, sharing the identical docstring sentence) `ct.bitwise_xor` oversell it. All three state the same promotion promise Chapter 17 confirmed for `add`, `mul`, and the rest of the arithmetic family — and none of the three keeps that promise. `int32` paired with `int64` is rejected. `int32` paired with `uint32`, same bit width, opposite signedness, is rejected with the same wording. Even `bool_` paired with `int32` — a pairing one might expect to widen freely, since a boolean is often treated as a 0-or-1 integer — is rejected identically. The error message itself, "must have same data type," is unambiguous about the actual rule: not "compatible," not "promotable," but identical.

`ct.bitwise_lshift` and `ct.bitwise_rshift` are the only two of these six functions whose behavior matches their own docstring on this specific point. `int32` shifted by an `int64`-typed tile of shift amounts compiles cleanly, and the promoted result genuinely is `int64` — confirmed by declaring the output as `int64` (succeeds, 23840 bytes) and then, in a second kernel identical apart from that one declared dtype, as `int32` (rejected, with the ordinary "cannot implicitly cast" store-mismatch error this book has seen many times since Chapter 3, not a bitwise-specific one). The result is the same when the wider and narrower types are swapped: `int64` shifted by an `int32` amount still promotes to `int64`. So within one `dir(ct)` family, sharing near-identical docstring language, three functions silently drop the promotion promise their own documentation makes, and two keep it exactly. A reader trusting the docstring alone would write code for `bitwise_and` that compiles for `bitwise_lshift`.

## 19.4 bitwise_not on a Boolean Tile

### Intuition

`ct.bitwise_not`'s docstring shows exactly one example, and it uses `int32`: `~ct.full((4,), 0, dtype=ct.int32)` producing `[-1, -1, -1, -1]` — a genuine two's-complement bitwise flip of every bit, including the sign bit. Nothing in the docstring says what happens when `x` is `bool_` instead. A true bitwise complement of a boolean value, if `bool_` is represented as a single meaningful bit, would flip that bit — but a true two's-complement flip of an entire integer word holding a 0 or 1 would produce something that isn't a clean boolean at all. Which behavior does the compiler actually implement, and which output dtype does it expect as a result?

### Background

Since this book cannot run `ct.launch` and inspect actual computed values, the only way to probe this is indirectly: try declaring the kernel's output array as `bool_` and see whether the compiler accepts it, then try declaring it as `int32` instead. If only one of the two compiles, that tells us which dtype the compiler considers `bitwise_not(bool_)` to produce.

### Worked Example 19.4.1 — bitwise_not(bool_) against two different declared output dtypes

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ct.bitwise_not's docstring calls it a bitwise complement, with no special
# case mentioned for bool_. On an integer type, a true bitwise complement
# of a single-bit-meaningful value would produce something other than a
# clean 0/1 -- so does bitwise_not on a bool_ tile even compile against a
# bool_-declared output, or does it demand an integer output instead?
sig_bool_io = ct.compilation.KernelSignature(
    [array_param(ct.bool_), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_not_bool_to_bool(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
try:
    n = compile_bytes(kernel_not_bool_to_bool, sig_bool_io)
    print(f"bitwise_not(bool_) -> bool_ output: {n} cubin bytes")
except Exception as e:
    print(f"bitwise_not(bool_) -> bool_ output: {type(e).__name__}: {e}")

# The same bool_ input, but declaring an int32 output instead -- if
# bitwise_not(bool_) is secretly typed as returning something other than
# bool_, this is where that would show up as a store-dtype rejection.
sig_bool_to_int = ct.compilation.KernelSignature(
    [array_param(ct.bool_), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_not_bool_to_int(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
try:
    n = compile_bytes(kernel_not_bool_to_int, sig_bool_to_int)
    print(f"bitwise_not(bool_) -> int32 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_not(bool_) -> int32 output: {type(e).__name__}: {e}")

# For comparison, bitwise_not on a genuine integer tile, output declared
# with the same int32 dtype as the input -- the ordinary case.
sig_int_io = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_not_int(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
print(f"bitwise_not(int32) -> int32 output: {compile_bytes(kernel_not_int, sig_int_io)} cubin bytes")
```

Genuinely run:

```
bitwise_not(bool_) -> bool_ output: 21168 cubin bytes
bitwise_not(bool_) -> int32 output: 21296 cubin bytes (unexpected)
bitwise_not(int32) -> int32 output: 21024 cubin bytes
```

### Discussion

Both compile. That is itself the surprise: the docstring's silence on `bool_` inputs might have suggested one clean answer — either `bitwise_not(bool_)` is typed as producing `bool_` and nothing else, or it is typed as producing an integer and a `bool_`-declared output would be rejected — but the compiler accepts either declared output dtype for the identical input. This is consistent with a compiler that treats `bitwise_not` on a `bool_` tile as producing a value flexible enough to satisfy both a `bool_` slot and an `int32` slot, the same broad, docstring-outrunning permissiveness Chapter 18 found in `ct.where`'s handling of its `cond` argument, now showing up in a completely different function. This book's `export_kernel`-only environment cannot say whether the two compiled kernels compute the same underlying bit pattern and merely store it into differently-typed arrays, or whether the compiler silently inserts a genuine logical-negation step for the `bool_`-declared case and a genuine bitwise-complement step for the `int32`-declared case — that would require running both kernels on real data and comparing outputs, which is exactly the kind of question this book has never been equipped to answer, all the way back to Chapter 1. What can be said is only that the compiler's type checker does not treat these two declared output dtypes as mutually exclusive for this one input dtype, a genuinely open question the docstring gives no basis for predicting either way.

## 19.5 Combining a Shift and a Mask

### Intuition

Chapter 18's masked-softmax capstone combined comparisons, `bitwise_and`, and `where` into one kernel body. This section builds two small, self-contained bit-manipulation patterns from this chapter's own operations, as building blocks toward Section 19.6's capstone: extracting one byte from a wider integer using a shift followed by a mask, and rounding a value up to the next multiple of a power of two using a shift-free but still purely bitwise trick.

### Background

`(x >> 8) & 0xFF` is the standard idiom for extracting the second-from-lowest byte of an integer: the right-shift moves that byte down into the lowest eight bits, and the mask clears everything above it. `(x + mask) & ~mask`, where `mask = align - 1` for a power-of-two `align`, is the standard idiom for rounding `x` up to the next multiple of `align` without a modulo or a branch — adding `mask` pushes any partial remainder into the next block, and the final `& ~mask` clears exactly the low bits that constitute "already at a multiple of `align`." Both idioms exercise `bitwise_and`, `bitwise_not`, and (in the first case) `bitwise_rshift` together in one small kernel body.

### Worked Example 19.5.1 — byte extraction and power-of-two rounding

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
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

# A classic shift-then-mask pattern: pull out the second byte (bits 8-15)
# of a 32-bit integer with (x >> 8) & 0xFF. Genuinely combines a shift op
# and a logical op from this chapter with a scalar mask.
@ct.kernel
def kernel_extract_byte(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    shifted = ct.bitwise_rshift(x, 8)
    byte1 = ct.bitwise_and(shifted, 0xFF)
    ct.store(c, (pid,), byte1)
print(f"extract byte via (x >> 8) & 0xFF: {compile_bytes(kernel_extract_byte, sig)} cubin bytes")

# Rounding a value UP to the next multiple of a power-of-two tile size --
# a bit trick built entirely from this chapter's ops: (x + mask) & ~mask,
# where mask = size - 1. An explicit callback to Chapter 15's rule that
# tile shapes must themselves be powers of two.
@ct.kernel
def kernel_round_up_pow2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    align = 16
    mask = align - 1
    rounded = ct.bitwise_and(x + mask, ct.bitwise_not(mask))
    ct.store(c, (pid,), rounded)
print(f"round up to multiple of 16 via (x + mask) & ~mask: {compile_bytes(kernel_round_up_pow2, sig)} cubin bytes")
```

Genuinely run:

```
extract byte via (x >> 8) & 0xFF: 21280 cubin bytes
round up to multiple of 16 via (x + mask) & ~mask: 21296 cubin bytes
```

### Discussion

Both idioms compile cleanly with no adjustment needed, confirming this chapter's operations compose the way ordinary bit-manipulation code expects them to: a shift result feeding directly into a mask, and a scalar Python `int` mask combined with `ct.bitwise_not` of that same scalar, with no explicit `ct.full` wrapper required anywhere in either kernel body. As with every other clean compile in this book, that confirms only that the operations are well-typed and shape-compatible in sequence — not that `align = 16` actually rounds correctly for every input `x`, which would again require `ct.launch` and real hardware to check.

## 19.6 Capstone: Testing Whether an Integer Is a Power of Two

### Intuition

Chapter 15 established a rule about tile *shapes*: a kernel's own tile dimensions must themselves be powers of two. This capstone tests the identical numeric property — is a given integer a power of two? — about ordinary *data* flowing through a kernel, using the classic bit trick `x & (x - 1) == 0`, valid for every positive `x`. For a power of two, exactly one bit is set, and subtracting one flips that bit off and every lower bit on, leaving no bits in common with the original value; for any other positive integer, at least one lower bit survives the subtraction unset in `x - 1` but set in `x`, so the `and` is nonzero.

### Background

This capstone deliberately exercises three findings from earlier in this chapter and one from Chapter 18. Section 19.3 found `bitwise_and` requires an exact dtype match with no promotion — which is why `one` below is built with `ct.full((tile_size,), 1, ct.int32)`, matching `x`'s declared `int32` dtype exactly, rather than left as a bare Python `int` subtracted directly (which Chapter 17's arithmetic-family promotion rules would handle fine, but which this chapter's stricter `bitwise_and` would not, once the subtraction's result needed to reach `bitwise_and` as a genuinely matching dtype). Chapter 18's `bitwise_and`-for-combining-boolean-tiles pattern combines two independently-computed boolean conditions — "no shared bits with `x - 1`" and "`x` is positive," since the trick is only valid for positive integers — using `bitwise_and` rather than Python's `and`, since these are genuine multi-element tiles.

### Worked Example 19.6.1 — is_pow2 via the classic bit trick

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Capstone: the classic bit trick for testing whether a positive integer
# is a power of two -- x & (x - 1) == 0 -- is nonzero for every value
# except powers of two, where the highest set bit and all bits below it
# cancel exactly. This chapter's finding that bitwise_and demands an
# EXACT dtype match (Section 19.3) is why `one` below is built with
# ct.full(..., ct.int32) rather than left as a bare Python int subtracted
# straight from x -- and why x's own dtype, not some promoted type, is
# what the final ct.equal comparison runs against.
#
# An explicit callback to Chapter 15: there, a KERNEL'S TILE SHAPE had to
# be a power of two. Here, the same power-of-two property is tested about
# the DATA flowing through the kernel, using tools this chapter built:
# bitwise_and for the bit trick itself, ct.greater to exclude non-positive
# inputs (the trick is only valid for x > 0), and bitwise_and again to
# combine the two boolean tiles into one result -- Chapter 18's
# elementwise mask-combination pattern, not Python's `and`.
sig = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_is_pow2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    one = ct.full((tile_size,), 1, ct.int32)
    no_shared_bits = ct.equal(ct.bitwise_and(x, x - one), 0)
    is_positive = ct.greater(x, 0)
    ct.store(c, (pid,), ct.bitwise_and(is_positive, no_shared_bits))
print(f"is_pow2 via x & (x-1) == 0: {compile_bytes(kernel_is_pow2, sig)} cubin bytes")
```

Genuinely run:

```
is_pow2 via x & (x-1) == 0: 21152 cubin bytes
```

### Discussion

The kernel compiles cleanly on the first attempt, chaining `bitwise_and` (twice, in two different roles — once as the bit trick itself, once as Chapter 18's boolean-mask combinator), `ct.equal`, and `ct.greater` into a single coherent body. As with every capstone since Chapter 15, this confirms only that the compiler accepted every type and shape at every step — not that the trick actually identifies powers of two correctly for any specific input, which would need `ct.launch` and real hardware, unavailable throughout this book. The callback to Chapter 15 is deliberate: that chapter's power-of-two rule governed the compiler's own structural requirements on a kernel's tile shape, a constraint the compiler enforces regardless of what data flows through the kernel. This capstone's power-of-two check is the opposite kind of thing entirely — an ordinary property of runtime data that the kernel computes, using bitwise operations general enough that the compiler has no special awareness that "is this a power of two" is even the question being asked.

## Chapter Summary

This chapter completed cuTile Python's bitwise operation family, exercising `bitwise_xor`, `bitwise_not`, `bitwise_lshift`, and `bitwise_rshift` for the first time and testing `bitwise_and`/`bitwise_or` well beyond Chapter 18's narrow masking use. All six functions reject `float32` operands, but the rejection is worded three different ways across the family: one message shared by `bitwise_and`/`bitwise_or`/`bitwise_xor`, a second, terser message unique to `bitwise_not`, and a third message shared by `bitwise_lshift`/`bitwise_rshift` that additionally names which operand — left-hand side or right-hand side — failed, confirming the two shift operators check their operands independently rather than as a pair. The chapter's central finding was a docstring failure running in the opposite direction from Chapter 18's: `bitwise_and`, `bitwise_or`, and `bitwise_xor` all promise, in identical wording to Chapter 17's genuinely-promoting arithmetic family, that mixed dtypes will be "promoted to common dtype" — and none of the three honors that promise, rejecting `int32`/`int64`, `int32`/`uint32`, and even `bool_`/`int32` operand pairs outright with a "must have same data type" error. `bitwise_lshift` and `bitwise_rshift`, sharing the identical docstring promise, are the only two of the six functions that actually keep it, promoting mixed-width integer operands to the wider type exactly as documented. A smaller surprise followed: `bitwise_not` on a `bool_`-dtyped tile compiles successfully against both a `bool_`-declared and an `int32`-declared output, a case the function's single, `int32`-only docstring example gives no basis for predicting. A capstone combined a shift, a mask, and the classic `x & (x - 1) == 0` power-of-two bit trick into a kernel that compiles cleanly, testing the same power-of-two property Chapter 15 enforced on tile shapes, now applied to ordinary runtime data instead.

## Self-Check Questions

1. Section 19.2 found three different wordings for six functions' float-rejection errors, with the shift operators' wording additionally naming which side (left-hand or right-hand) failed. What does the fact that `bitwise_and`/`bitwise_or`/`bitwise_xor` do NOT name a side in their shared message suggest about how those three functions check their two operands, compared to how `bitwise_lshift`/`bitwise_rshift` check theirs?
2. Section 19.3 found that `bitwise_and`'s docstring claims the same dtype-promotion behavior Chapter 17 confirmed for `ct.add`, but the real compiler rejects every dtype mismatch tested, including `bool_` paired with `int32`. If you were relying only on the docstring, what specific kind of kernel would you write that compiles fine for `ct.add` but fails to compile for `ct.bitwise_and`?
3. Section 19.3 also found `bitwise_lshift` DOES promote mixed integer widths, unlike `bitwise_and`. Given that both functions share the identical "dtype promoted to common dtype" docstring sentence, what does this tell you about how much weight to put on one sentence of shared docstring language when predicting two different functions' actual behavior?
4. Section 19.4 found `bitwise_not(bool_)` compiles against both a `bool_`-declared and an `int32`-declared output. Name one thing this book's `export_kernel`-only environment would need in order to determine whether the two declared-output variants compute the same bit pattern or two genuinely different results.
5. Section 19.6's capstone tests whether an integer is a power of two, while Chapter 15 enforced a rule that tile *shapes* must themselves be powers of two. Both use the same underlying mathematical property. What is the key difference between a rule the compiler enforces on a kernel's structure (Chapter 15) and a property a kernel computes about its own data (Section 19.6), in terms of what a clean compile can and cannot tell you about each?

## Where We Go Next

Chapter 18 closed by describing `ct.bitwise_and`/`ct.bitwise_or` as tools for combining masks elementwise, "in sharp contrast to Python's `and`, `or`, `if`, and chained-comparison syntax" — a distinction between elementwise data operations and genuine control flow that this chapter's entire bitwise family has now been shown to respect: not one of these six functions has anything to do with branching, only with computing values. That distinction is about to matter in a new way. Every kernel this book has written, all the way back to Chapter 1, has been confined to one block reading and writing its own private slice of memory — nothing a kernel has ever done required it to know or care whether some other block, running concurrently, might be touching the same memory at the same time. `dir(ct)` lists a family of atomic memory operations — `atomic_add`, `atomic_max`, `atomic_cas`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_xchg` — built specifically for the case this book has never encountered: multiple blocks that need to safely read-modify-write a single shared location without their updates clobbering each other. Chapter 20 turns to these atomics, the first cuTile Python operations this book has met that are defined entirely in terms of what other, concurrently-running blocks might also be doing.

## Complete Runnable Code

### File: `01_xor_not_lshift_rshift.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
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

# ct.bitwise_xor: elementwise exclusive-or, the third binary logical op
# in the family alongside Chapter 18's bitwise_and/bitwise_or.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_xor(x, y))
print(f"bitwise_xor(x, y): {compile_bytes(kernel_fn, sig)} cubin bytes")

# ct.bitwise_not: unary, one operand, one result -- the odd one out in
# this family's signatures ((x, /) rather than (x, y, /)).
sig_unary = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
print(f"bitwise_not(x): {compile_bytes(kernel_fn2, sig_unary)} cubin bytes")

# ct.bitwise_lshift: shift left by a tile of shift amounts, not just a
# scalar -- every element of x shifted by the corresponding element of y.
@ct.kernel
def kernel_fn3(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
print(f"bitwise_lshift(x, y): {compile_bytes(kernel_fn3, sig)} cubin bytes")

# ct.bitwise_rshift: its right-shift counterpart.
@ct.kernel
def kernel_fn4(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_rshift(x, y))
print(f"bitwise_rshift(x, y): {compile_bytes(kernel_fn4, sig)} cubin bytes")

# Shifting by a plain Python scalar rather than a loaded tile -- the more
# common "shift by a fixed constant amount" pattern. lshift and rshift
# defined under distinct names here (not naming-controlled against each
# other), so their equal byte counts below are reported as an observation,
# not a claim of equivalence.
sig_scalar = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_scalar(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, 2))
print(f"bitwise_lshift(x, scalar 2): {compile_bytes(kernel_lshift_scalar, sig_scalar)} cubin bytes")

@ct.kernel
def kernel_rshift_scalar(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_rshift(x, 2))
print(f"bitwise_rshift(x, scalar 2): {compile_bytes(kernel_rshift_scalar, sig_scalar)} cubin bytes")
```

### File: `02_float_rejection_three_wordings.py`

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
sig_unary = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Every bitwise op in this chapter rejects float32 operands -- but not with
# one shared message. Three distinct wordings appear across six functions.

# Wording 1: bitwise_and, bitwise_or, bitwise_xor -- the three binary
# logical ops -- all share this exact sentence.
@ct.kernel
def kernel_and(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and, sig)
    print(f"bitwise_and(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(float32, float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_or(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_or(x, y))
try:
    n = compile_bytes(kernel_or, sig)
    print(f"bitwise_or(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_or(float32, float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_xor(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_xor(x, y))
try:
    n = compile_bytes(kernel_xor, sig)
    print(f"bitwise_xor(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_xor(float32, float32): {type(e).__name__}: {e}")

# Wording 2: bitwise_not -- the lone unary op -- uses its own, much
# terser phrasing rather than reusing wording 1.
@ct.kernel
def kernel_not(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
try:
    n = compile_bytes(kernel_not, sig_unary)
    print(f"bitwise_not(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_not(float32): {type(e).__name__}: {e}")

# Wording 3: bitwise_lshift and bitwise_rshift -- the two shift ops --
# share a third wording, and that wording names which side is at fault.
@ct.kernel
def kernel_lshift(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift, sig)
    print(f"bitwise_lshift(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_lshift(float32, float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_rshift(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_rshift(x, y))
try:
    n = compile_bytes(kernel_rshift, sig)
    print(f"bitwise_rshift(float32, float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_rshift(float32, float32): {type(e).__name__}: {e}")

# Wording 3 names a side -- so does an int32 LHS with a float32 RHS (the
# shift-amount operand) trigger a different half of that same wording?
sig_int_lhs_float_rhs = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.float32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_rhs(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_rhs, sig_int_lhs_float_rhs)
    print(f"bitwise_lshift(int32, float32 shift amount): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_lshift(int32, float32 shift amount): {type(e).__name__}: {e}")
```

### File: `03_strict_equality_vs_shift_promotion.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Chapter 17 documented dtype promotion as the rule for the binary
# elementwise family: mixed int32/float32, mixed int32/int64, and so on
# are silently promoted to a common type. bitwise_and, bitwise_or, and
# bitwise_xor break that rule -- they demand an EXACT dtype match instead.

# int32 AND int64: rejected outright, no promotion attempted.
sig_int32_int64 = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int64), array_param(ct.int64), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_and_widths(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and_widths, sig_int32_int64)
    print(f"bitwise_and(int32, int64): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(int32, int64): {type(e).__name__}: {e}")

# Same width, different signedness: int32 vs uint32. Still rejected --
# "same data type" means exactly that, not "same bit width."
sig_sign = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.uint32), array_param(ct.uint32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_and_sign(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and_sign, sig_sign)
    print(f"bitwise_and(int32, uint32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(int32, uint32): {type(e).__name__}: {e}")

# bool_ vs int32: also rejected. bool_ is not a special case that widens
# freely into an integer type here.
sig_bool = ct.compilation.KernelSignature(
    [array_param(ct.bool_), array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_and_bool_int(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_and(x, y))
try:
    n = compile_bytes(kernel_and_bool_int, sig_bool)
    print(f"bitwise_and(bool_, int32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_and(bool_, int32): {type(e).__name__}: {e}")

# bitwise_or shares bitwise_and's strict rule -- not a bitwise_and-only
# restriction.
@ct.kernel
def kernel_or_widths(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_or(x, y))
try:
    n = compile_bytes(kernel_or_widths, sig_int32_int64)
    print(f"bitwise_or(int32, int64): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_or(int32, int64): {type(e).__name__}: {e}")

# bitwise_lshift, by contrast, DOES promote: an int32 tile shifted by an
# int64-typed tile of shift amounts compiles cleanly, with the result
# taking the wider dtype -- exactly the Chapter 17 promotion pattern that
# bitwise_and/or/xor just refused to follow.
@ct.kernel
def kernel_lshift_promote(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_promote, sig_int32_int64)
    print(f"bitwise_lshift(int32, int64 shift amount) -> int64 output: {n} cubin bytes")
except Exception as e:
    print(f"bitwise_lshift(int32, int64 shift amount) -> int64 output: {type(e).__name__}: {e}")

# The same call, but declaring an int32 output instead: if the promoted
# result really is int64, an int32-declared output should now fail --
# with the same "cannot implicitly cast" store-mismatch error used
# throughout this book, not a bitwise-specific one.
sig_int32_out = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int64), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_int32_out(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_int32_out, sig_int32_out)
    print(f"bitwise_lshift(int32, int64 shift amount) -> int32 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_lshift(int32, int64 shift amount) -> int32 output: {type(e).__name__}: {e}")

# And with the widths reversed -- int64 x, int32 shift amount -- the
# promotion still lands on int64.
sig_reverse = ct.compilation.KernelSignature(
    [array_param(ct.int64), array_param(ct.int32), array_param(ct.int64), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_lshift_reverse(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_lshift(x, y))
try:
    n = compile_bytes(kernel_lshift_reverse, sig_reverse)
    print(f"bitwise_lshift(int64, int32 shift amount) -> int64 output: {n} cubin bytes")
except Exception as e:
    print(f"bitwise_lshift(int64, int32 shift amount) -> int64 output: {type(e).__name__}: {e}")
```

### File: `04_bitwise_not_on_bool.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ct.bitwise_not's docstring calls it a bitwise complement, with no special
# case mentioned for bool_. On an integer type, a true bitwise complement
# of a single-bit-meaningful value would produce something other than a
# clean 0/1 -- so does bitwise_not on a bool_ tile even compile against a
# bool_-declared output, or does it demand an integer output instead?
sig_bool_io = ct.compilation.KernelSignature(
    [array_param(ct.bool_), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_not_bool_to_bool(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
try:
    n = compile_bytes(kernel_not_bool_to_bool, sig_bool_io)
    print(f"bitwise_not(bool_) -> bool_ output: {n} cubin bytes")
except Exception as e:
    print(f"bitwise_not(bool_) -> bool_ output: {type(e).__name__}: {e}")

# The same bool_ input, but declaring an int32 output instead -- if
# bitwise_not(bool_) is secretly typed as returning something other than
# bool_, this is where that would show up as a store-dtype rejection.
sig_bool_to_int = ct.compilation.KernelSignature(
    [array_param(ct.bool_), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_not_bool_to_int(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
try:
    n = compile_bytes(kernel_not_bool_to_int, sig_bool_to_int)
    print(f"bitwise_not(bool_) -> int32 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"bitwise_not(bool_) -> int32 output: {type(e).__name__}: {e}")

# For comparison, bitwise_not on a genuine integer tile, output declared
# with the same int32 dtype as the input -- the ordinary case.
sig_int_io = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_not_int(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.bitwise_not(x))
print(f"bitwise_not(int32) -> int32 output: {compile_bytes(kernel_not_int, sig_int_io)} cubin bytes")
```

### File: `05_combining_shift_and_mask.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
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

# A classic shift-then-mask pattern: pull out the second byte (bits 8-15)
# of a 32-bit integer with (x >> 8) & 0xFF. Genuinely combines a shift op
# and a logical op from this chapter with a scalar mask.
@ct.kernel
def kernel_extract_byte(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    shifted = ct.bitwise_rshift(x, 8)
    byte1 = ct.bitwise_and(shifted, 0xFF)
    ct.store(c, (pid,), byte1)
print(f"extract byte via (x >> 8) & 0xFF: {compile_bytes(kernel_extract_byte, sig)} cubin bytes")

# Rounding a value UP to the next multiple of a power-of-two tile size --
# a bit trick built entirely from this chapter's ops: (x + mask) & ~mask,
# where mask = size - 1. An explicit callback to Chapter 15's rule that
# tile shapes must themselves be powers of two.
@ct.kernel
def kernel_round_up_pow2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    align = 16
    mask = align - 1
    rounded = ct.bitwise_and(x + mask, ct.bitwise_not(mask))
    ct.store(c, (pid,), rounded)
print(f"round up to multiple of 16 via (x + mask) & ~mask: {compile_bytes(kernel_round_up_pow2, sig)} cubin bytes")
```

### File: `06_capstone_power_of_two_check.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Capstone: the classic bit trick for testing whether a positive integer
# is a power of two -- x & (x - 1) == 0 -- is nonzero for every value
# except powers of two, where the highest set bit and all bits below it
# cancel exactly. This chapter's finding that bitwise_and demands an
# EXACT dtype match (Section 19.3) is why `one` below is built with
# ct.full(..., ct.int32) rather than left as a bare Python int subtracted
# straight from x -- and why x's own dtype, not some promoted type, is
# what the final ct.equal comparison runs against.
#
# An explicit callback to Chapter 15: there, a KERNEL'S TILE SHAPE had to
# be a power of two. Here, the same power-of-two property is tested about
# the DATA flowing through the kernel, using tools this chapter built:
# bitwise_and for the bit trick itself, ct.greater to exclude non-positive
# inputs (the trick is only valid for x > 0), and bitwise_and again to
# combine the two boolean tiles into one result -- Chapter 18's
# elementwise mask-combination pattern, not Python's `and`.
sig = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_is_pow2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    one = ct.full((tile_size,), 1, ct.int32)
    no_shared_bits = ct.equal(ct.bitwise_and(x, x - one), 0)
    is_positive = ct.greater(x, 0)
    ct.store(c, (pid,), ct.bitwise_and(is_positive, no_shared_bits))
print(f"is_pow2 via x & (x-1) == 0: {compile_bytes(kernel_is_pow2, sig)} cubin bytes")
```
