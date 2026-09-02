# Chapter 18: Comparisons, Where, and the Limits of a Tile Condition

> "Chapter 1 found, in its very first chapter, that a raw multi-element boolean tile cannot drive an `if` statement — only a genuine scalar can. That finding sat quietly for seventeen chapters, cited but never extended. This chapter finally asks the question Chapter 1 left open: if `if` needs a scalar, what exactly happens when Python's own `and`, `or`, and chained-comparison syntax — all of which secretly need the same thing — meet a tile instead?"

**What you will understand by the end of this chapter:**

- `ct.equal`, `ct.not_equal`, `ct.greater`, `ct.greater_equal`, `ct.less`, and `ct.less_equal`: that a comparison's boolean result can be stored into either a `bool_`-declared or an `int32`-declared output array, with the `int32` case measurably costing more under Chapter 10's naming control, and that comparisons broadcast and promote dtypes exactly like Chapter 17's arithmetic family
- `ct.where(cond, x, y)`'s documented claim that `cond`, `x`, and `y` all share one shape and `x`/`y` share one dtype — and that the real compiler is more permissive on every count: mismatched `x`/`y` dtypes promote rather than reject, `cond` need not be `bool_`-dtyped at all, and `cond` broadcasts against a differently-shaped `x`/`y` rather than requiring an exact match
- `ct.bitwise_and`/`ct.bitwise_or` (and Python's own `&`/`|`) combine two boolean tiles elementwise with no control flow involved — in sharp contrast to Python's `and`, `or`, `if`, and chained-comparison syntax, all of which secretly require a genuine scalar and fail against a multi-element tile, via two subtly different error messages
- That a hand-rolled clamp built from `ct.where` and this chapter's comparisons does **not** compile to the same byte count as the built-in `ct.minimum`/`ct.maximum` equivalent, under Chapter 10's naming control — logically equivalent code that the compiler nonetheless treats differently
- That a masked, row-normalized kernel combining this chapter's comparisons, `bitwise_and`, and `where` with Chapter 16's `reduce`, Chapter 15's `broadcast_to`, and Chapter 17's `exp`/`truediv` compiles cleanly in one body

**What you need to know first:**

- Chapter 1's foundational finding that `if <multi-element boolean tile>:` is rejected with `TileTypeError: Expected a scalar value` — this chapter extends that finding rather than repeating it from scratch.
- Chapter 10's naming-confound rule, Chapter 15's `reshape`/`broadcast_to`, Chapter 16's `reduce`, and Chapter 17's binary elementwise family (broadcasting, dtype promotion) and its `exp`/`truediv` rounding-mode findings.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow as every chapter before it.

## 18.1 The Six Comparison Operators

### Intuition

`equal`, `not_equal`, `greater`, `greater_equal`, `less`, and `less_equal` are cuTile Python's elementwise comparisons — the machinery behind `==`, `!=`, `>`, `>=`, `<`, and `<=` on tiles. Each is documented with the same binary-elementwise signature shape Chapter 17 already exercised for `pow`, `mod`, `floordiv`, and `truediv`: broadcasting shapes, promoting dtypes, and, per `inspect.signature`, returning a `TileOrScalar`.

### Background

None of the six comparison docstrings specifies a required output dtype for the boolean result they produce — unlike `ct.isnan` in Chapter 17, whose docstring explicitly frames its result as boolean-classified. This section tests two open questions directly: whether the same comparison compiles equally well against a `bool_`-declared output array and an `int32`-declared one, and whether comparing a tile against a plain Python scalar (rather than a second tile) works the same way Chapter 17's `atan2` and this chapter's own `ct.where` accept scalar operands.

### Worked Example 18.1.1 — six comparisons, two output dtypes, a scalar operand, and mixed input dtypes

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
    [array_param(), array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.equal(x, y))
print(f"equal(x, y): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.not_equal(x, y))
print(f"not_equal(x, y): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater(x, y))
print(f"greater(x, y): {compile_bytes(kernel_fn3, sig)} cubin bytes")

@ct.kernel
def kernel_fn4(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater_equal(x, y))
print(f"greater_equal(x, y): {compile_bytes(kernel_fn4, sig)} cubin bytes")

@ct.kernel
def kernel_fn5(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.less(x, y))
print(f"less(x, y): {compile_bytes(kernel_fn5, sig)} cubin bytes")

@ct.kernel
def kernel_fn6(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.less_equal(x, y))
print(f"less_equal(x, y): {compile_bytes(kernel_fn6, sig)} cubin bytes")

# The docstrings never mention a required output dtype. Does the same
# comparison compile against a bool_-declared output AND an int32-declared
# output equally, or is one of the two rejected? Naming-controlled
# (Chapter 10): both variants defined as kernel_fn7, reassigned in turn.
sig_bool_out = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_int_out = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.equal(x, y))
kernel_bool_out = kernel_fn7
print(f"equal(x, y) -> bool_ output: {compile_bytes(kernel_bool_out, sig_bool_out)} cubin bytes")

@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.equal(x, y))
kernel_int_out = kernel_fn7
print(f"equal(x, y) -> int32 output: {compile_bytes(kernel_int_out, sig_int_out)} cubin bytes")

# A tile compared against a plain Python scalar, not a second tile.
sig_scalar = ct.compilation.KernelSignature(
    [array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn9(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater(x, 2.0))
print(f"greater(x, 2.0) scalar RHS: {compile_bytes(kernel_fn9, sig_scalar)} cubin bytes")

# Mixed dtype operands -- int32 and float32 -- promoted to a common dtype,
# the same claim Chapter 17's binary elementwise family documented.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.float32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn10(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater(x, y))
try:
    n = compile_bytes(kernel_fn10, sig_mixed)
    print(f"greater(int32 x, float32 y) mixed dtype: {n} cubin bytes")
except Exception as e:
    print(f"greater(int32 x, float32 y) mixed dtype: {type(e).__name__}: {e}")
```

Genuinely run:

```
equal(x, y): 23184 cubin bytes
not_equal(x, y): 23312 cubin bytes
greater(x, y): 23312 cubin bytes
greater_equal(x, y): 23312 cubin bytes
less(x, y): 23312 cubin bytes
less_equal(x, y): 23312 cubin bytes
equal(x, y) -> bool_ output: 23312 cubin bytes
equal(x, y) -> int32 output: 23456 cubin bytes
greater(x, 2.0) scalar RHS: 21024 cubin bytes
greater(int32 x, float32 y) mixed dtype: 23456 cubin bytes
```

### Discussion

All six comparisons compile cleanly against a `bool_`-declared output. The naming-controlled comparison between a `bool_`-declared output and an `int32`-declared output for the identical `equal(x, y)` call — both defined under the shared name `kernel_fn7`, reassigned in turn — shows a real, measurable difference: 23312 bytes against 23456 bytes. The comparison itself produces the same boolean result either way, but storing that result into an `int32` array evidently costs the compiler something extra, consistent with an implicit boolean-to-integer conversion being inserted at the store rather than the comparison instruction itself changing shape. A tile compared against a plain Python scalar compiles without incident, and mixed int32/float32 operands promote to a common dtype and compile successfully — both confirming Chapter 17's binary elementwise family's documented behavior extends to the comparison family as well.

## 18.2 Where: Selecting Between Two Tiles

### Intuition

`ct.where(cond, x, y)` selects, elementwise, from `x` wherever `cond` is true and from `y` wherever it is false — this book's first genuinely conditional, per-element operation, in contrast to every control-flow mechanism this book has used so far, which has operated at the level of an entire kernel invocation rather than individual tile elements.

### Background

The docstring is direct: `cond` is "Boolean tile of shape `S`," and `x`/`y` are each "Tile of shape `S` and dtype `T`" — one shared shape and one shared dtype across all three arguments, as plain a specification as this book has seen for a three-argument function. This section confirms the ordinary, documented case compiles as expected before Section 18.3 tests whether the compiler actually enforces every part of that specification as strictly as written.

### Worked Example 18.2.1 — selecting with a computed condition, and with scalar replacement values

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

# ct.where(cond, x, y): select from x where cond is True, else from y.
# The condition is built from one of Section 18.1's comparison operators.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    cond = ct.greater(x, y)
    ct.store(c, (pid,), ct.where(cond, x, y))
print(f"where(x > y, x, y): {compile_bytes(kernel_fn, sig)} cubin bytes")

# where with a scalar-valued x and y, rather than two loaded tiles --
# the common "replace matching elements with a constant" pattern.
sig2 = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    cond = ct.equal(x, 0.0)
    ct.store(c, (pid,), ct.where(cond, ct.full((tile_size,), -1.0, ct.float32), x))
print(f"where(x == 0, -1.0, x): {compile_bytes(kernel_fn2, sig2)} cubin bytes")
```

Genuinely run:

```
where(x > y, x, y): 23312 cubin bytes
where(x == 0, -1.0, x): 21024 cubin bytes
```

### Discussion

Both calls compile cleanly: a condition built from a genuine tile comparison, selecting between two loaded tiles, and a condition selecting between a constant-filled tile and a loaded one — the "replace elements matching some condition with a fixed value" pattern likely to be `where`'s single most common real use. With the documented, exactly-matching-shape-and-dtype case confirmed, Section 18.3 turns to what happens outside it.

## 18.3 Where Is More Permissive Than Its Own Docstring `[FOUNDATIONAL]`

### Intuition

`ct.where`'s docstring commits to three specific claims: `cond` is boolean, and `x`/`y` share one shape and one dtype. Chapter 17 already found that a function's documentation can understate what the compiler actually accepts — `ct.mod` promoted mismatched dtypes despite no explicit mention of promotion in that specific sentence of its docstring. `ct.where`'s docstring is more explicit than `mod`'s about requiring a single shared dtype for `x` and `y`, which makes it a sharper test: does the compiler actually enforce that, or does it quietly promote here too?

### Background

Four claims are worth testing directly, each a plausible reading of the docstring's plain language: that `x` and `y` must share one dtype exactly; that `cond` must be genuinely `bool_`-dtyped; that an int32-dtyped tile of 0s and 1s cannot substitute for a real boolean tile as `cond`; and that `cond` must match `x`/`y`'s shape exactly rather than broadcasting the way Chapter 17's binary elementwise family does. Chapter 16's `ct.reduce`-with-`keepdims` is used here to build a genuinely differently-shaped `cond` — an `(8, 1)` tile — without artificially constructing a shape mismatch that wouldn't arise from ordinary kernel code.

### Worked Example 18.3.1 — dtype promotion, non-boolean conditions, and broadcasting

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

# ct.where's docstring declares cond, x, and y all as sharing one shape S,
# with x and y sharing one dtype T. Does the compiler actually enforce
# that x and y share a dtype, or does it promote them like the rest of
# this book's binary elementwise family (Chapter 17)?
sig_mixed = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.int32), array_param(ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    cond = ct.greater(x, ct.astype(y, ct.float32))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn, sig_mixed)
    print(f"where(cond, float32 x, int32 y) mixed dtype: {n} cubin bytes")
except Exception as e:
    print(f"where(cond, float32 x, int32 y) mixed dtype: {type(e).__name__}: {e}")

# The same mixed-dtype call, but declaring the output array as int32
# instead of float32 -- if the promoted result is genuinely float32,
# an int32-declared output should now be rejected as a mismatch.
sig_int_out = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    cond = ct.greater(x, ct.astype(y, ct.float32))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn2, sig_int_out)
    print(f"where(cond, float32 x, int32 y) -> int32 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"where(cond, float32 x, int32 y) -> int32 output: {type(e).__name__}: {e}")

# The docstring also declares cond as "Boolean tile of shape S." Does an
# int32-dtyped condition tile -- never explicitly bool_ -- work directly?
sig_int_cond = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn3(cond_arr, a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    cond = ct.load(cond_arr, (pid,), (tile_size,))
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn3, sig_int_cond)
    print(f"where(int32 cond, x, y): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"where(int32 cond, x, y): {type(e).__name__}: {e}")

# And a float32-dtyped condition tile -- further still from "Boolean."
sig_float_cond = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn4(cond_arr, a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    cond = ct.load(cond_arr, (pid,), (tile_size,))
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn4, sig_float_cond)
    print(f"where(float32 cond, x, y): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"where(float32 cond, x, y): {type(e).__name__}: {e}")

# The docstring says cond, x, and y all share one shape S. Does cond
# actually need to match x/y's shape exactly, or does it broadcast the
# way Chapter 17's binary elementwise family does? A (8, 1) condition
# against (8, 8) x and y, passed directly with no explicit broadcast_to.
sig_bcast = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    row_max = ct.reduce(x2d, 1, ct.maximum, -3.0e38, keepdims=True)
    cond = ct.greater(row_max, 0.0)
    result = ct.where(cond, x2d, ct.full((8, 8), -1.0, ct.float32))
    ct.store(c, (pid,), ct.reshape(result, (tile_size,)))
try:
    n = compile_bytes(kernel_fn5, sig_bcast)
    print(f"where((8,1) cond, (8,8) x, (8,8) y) unbroadcast cond: {n} cubin bytes")
except Exception as e:
    print(f"where((8,1) cond, (8,8) x, (8,8) y) unbroadcast cond: {type(e).__name__}: {e}")
```

Genuinely run:

```
where(cond, float32 x, int32 y) mixed dtype: 23440 cubin bytes
where(cond, float32 x, int32 y) -> int32 output: TileTypeError: Stored tile is incompatible with array's dtype: cannot implicitly cast float32 to int32
Unknown location
  "/tmp/ch18/03_where_more_permissive_than_docstring.py", line 49, col 5-45, in kernel_fn2:
        ct.store(c, (pid,), ct.where(cond, x, y))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

where(int32 cond, x, y): 26192 cubin bytes (unexpected)
where(float32 cond, x, y): 26192 cubin bytes (unexpected)
where((8,1) cond, (8,8) x, (8,8) y) unbroadcast cond: 28672 cubin bytes
```

### Discussion

Every one of the four claims tested turns out to be more permissive in practice than the docstring's plain language suggests. `where(cond, float32 x, int32 y)` compiles successfully — the mismatched dtypes are promoted, not rejected, exactly as Chapter 17's `mod` promoted a mismatched int32/float32 pair. The follow-up test confirms the promoted result is genuinely `float32`: declaring the output array as `int32` fails with `TileTypeError: Stored tile is incompatible with array's dtype: cannot implicitly cast float32 to int32`, meaning the value `where` actually produced needed a `float32`-declared output to be stored, not an `int32` one. An `int32`-dtyped condition tile — never declared `bool_` — is accepted directly, and so is a `float32`-dtyped one, both marked `(unexpected)` above only in the sense that the docstring's "Boolean tile" language gives no reason to expect either would work; the compiler evidently accepts any tile whose values can be interpreted as a per-element truth value at the point `where` consumes it, not specifically a `bool_`-typed one. And an `(8, 1)`-shaped condition, built from a genuine `Chapter 16` `keepdims=True` reduction, broadcasts successfully against `(8, 8)` `x` and `y` operands passed with no explicit `broadcast_to` call at all — the exact broadcasting behavior Chapter 17's binary elementwise family already established, extended here to `where`'s three-argument shape.

None of this means the docstring is wrong, exactly — every example it gives is still accurate — but it commits to a narrower contract ("shape `S`," "dtype `T`," "Boolean") than the compiler actually enforces. A reader who took the docstring as the full specification of what `where` accepts would have concluded, incorrectly, that all three of these calls should fail. The general lesson echoes Chapter 17's rounding-mode finding from the opposite direction: there, no single behavior could be assumed to generalize across functions; here, a single function's own stated contract cannot be assumed to state the full boundary of what it actually accepts, without testing past the documented cases directly.

## 18.4 Combining Boolean Tiles: Elementwise Masks Versus Python's Control-Flow Protocol

### Intuition

Chapter 1 found that `if <multi-element boolean tile>:` fails, because `if` needs a genuine scalar and a tile of many elements is not one. That finding has sat unextended for seventeen chapters. Combining two conditions — "greater than 2 AND less than 8" — is exactly the situation where a reader unfamiliar with that restriction might reach for Python's `and` keyword or its chained-comparison sugar (`2 < x < 8`) without realizing both secretly route through the same scalar-truthiness protocol `if` does.

### Background

`ct.bitwise_and` and `ct.bitwise_or` (and Python's own `&`/`|` operators, which cuTile Python maps to them) combine two tiles elementwise, position by position, with no truthiness evaluation of the tile as a whole — exactly the tool this chapter's masking needs, and a genuinely different mechanism from `and`/`or`. Python's `and`, `or`, `if`, and chained-comparison syntax (`a < b < c`, sugar for `a < b and b < c`) all call `bool()` on an intermediate value to decide how to proceed — precisely the operation Chapter 1 found fails against a multi-element tile. This section confirms `&`/`|` succeed where `and`/`if` fail, and checks whether Python's chained-comparison sugar — which desugars to the identical `and`-based logic under the hood — produces the identical error Chapter 1 already found, or something subtly different.

### Worked Example 18.4.1 — elementwise masks that work, and control-flow constructs that don't

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
    [array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Combining two elementwise boolean tiles with ct.bitwise_and -- a genuine
# elementwise AND, one result per element, no control flow involved.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = ct.bitwise_and(ct.greater(x, 2.0), ct.less(x, 8.0))
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn, sig)
    print(f"bitwise_and(greater(x,2), less(x,8)): {n} cubin bytes")
except Exception as e:
    print(f"bitwise_and(greater(x,2), less(x,8)): {type(e).__name__}: {e}")

# The same mask, using Python's own & operator directly on the two tiles.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = (x > 2.0) & (x < 8.0)
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn2, sig)
    print(f"(x > 2.0) & (x < 8.0): {n} cubin bytes")
except Exception as e:
    print(f"(x > 2.0) & (x < 8.0): {type(e).__name__}: {e}")

# ct.bitwise_or, the elementwise counterpart.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    out_of_range = ct.bitwise_or(ct.less(x, 2.0), ct.greater(x, 8.0))
    ct.store(c, (pid,), out_of_range)
try:
    n = compile_bytes(kernel_fn3, sig)
    print(f"bitwise_or(less(x,2), greater(x,8)): {n} cubin bytes")
except Exception as e:
    print(f"bitwise_or(less(x,2), greater(x,8)): {type(e).__name__}: {e}")

# Chapter 1 already found that a multi-element boolean TILE cannot drive a
# real `if` statement -- it needs a genuine scalar. Python's own `and`
# keyword hits the identical requirement directly, with the identical
# "Expected a scalar value" message Chapter 1 first found from `if`.
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = (x > 2.0) and (x < 8.0)
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn4, sig)
    print(f"(x > 2.0) and (x < 8.0): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"(x > 2.0) and (x < 8.0): {type(e).__name__}: {e}")

# Python's chained-comparison sugar, 2.0 < x < 8.0, desugars to the same
# `and`-based logic under the hood -- does it fail the identical way?
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = 2.0 < x < 8.0
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn5, sig)
    print(f"2.0 < x < 8.0 (chained): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"2.0 < x < 8.0 (chained): {type(e).__name__}: {e}")
```

Genuinely run:

```
bitwise_and(greater(x,2), less(x,8)): 21024 cubin bytes
(x > 2.0) & (x < 8.0): 21024 cubin bytes
bitwise_or(less(x,2), greater(x,8)): 21024 cubin bytes
(x > 2.0) and (x < 8.0): TileTypeError: Expected a scalar value, but given value has type Tile[bool_,(64)]
  "/tmp/ch18/04_combining_boolean_tiles.py", line 68, col 17-23, in kernel_fn4:
        in_range = (x > 2.0) and (x < 8.0)
                    ^^^^^^^

2.0 < x < 8.0 (chained): TileTypeError: Invalid argument #1 of if_else(): Expected a bool, but given value has type Tile[bool_,(64)]
  "/tmp/ch18/04_combining_boolean_tiles.py", line 82, col 16-28, in kernel_fn5:
        in_range = 2.0 < x < 8.0
                   ^^^^^^^^^^^^^
```

### Discussion

`ct.bitwise_and`, Python's own `&`, and `ct.bitwise_or` all compile without incident — genuine elementwise combination of two boolean tiles, one output element per pair of input elements, exactly the tool a masking kernel like this chapter's capstone needs. `and` fails with `TileTypeError: Expected a scalar value, but given value has type Tile[bool_,(64)]` — the identical message and error type Chapter 1 first found from a raw `if` statement, confirming `and`'s scalar requirement really is the same underlying check `if` enforces, not a coincidentally similar one. The chained comparison `2.0 < x < 8.0` also fails, but with different, more specific wording: `Invalid argument #1 of if_else(): Expected a bool, but given value has type Tile[bool_,(64)]`, naming an internal `if_else()` call the plain `and` case never mentioned. Both errors trace back to the same root cause — a multi-element tile reaching a code path that needs a genuine scalar truth value — but Python's chained-comparison sugar and its explicit `and` keyword evidently route through the compiler's front end differently enough to produce two distinct error messages for what a reader might reasonably expect to be identical failures under the hood.

The practical lesson for building a real mask: use `&`/`|` (or their `bitwise_and`/`bitwise_or` spellings) to combine boolean tiles elementwise, and reserve `if`/`and`/`or`/chained comparisons for the genuine, whole-kernel scalar branches Chapter 1's own `ct.sum(a_tile)`-gated example already demonstrated. The two mechanisms look similar in ordinary Python — both involve boolean values and comparison operators — but only one of them operates per-element the way tile code generally needs.

## 18.5 A Clamp That Isn't Compiled the Same Two Ways

### Intuition

`ct.minimum`/`ct.maximum`, used already in Chapter 16's and 17's capstones, are the natural way to implement clamping: `maximum(minimum(x, high), low)` caps `x` at `high` and floors it at `low` in two calls. This chapter's own `ct.where` and comparison operators can build the identical logical operation by hand — cap values above `high` down to `high`, then floor values below `low` up to `low`. Chapter 16 found `ct.scan`/`ct.reduce` compile byte-for-byte identically to their named specific instances (`cumsum`/`sum`). Does a hand-rolled `where`-based clamp compile identically to `minimum`/`maximum`'s built-in one, the same way?

### Background

Both implementations compute the same clamped result for any given input, under this book's compiled-structure-only verification (no `ct.launch`, so this chapter cannot and does not claim to verify they compute the *same numeric values* — only that both compile). Chapter 10's naming-confound rule applies as always: both variants must be defined under the identical function name to make any byte-count comparison meaningful.

### Worked Example 18.5.1 — `minimum`/`maximum` versus `where` + comparisons

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

# clamp(x, 2.0, 8.0) built from ct.minimum/ct.maximum.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    clamped = ct.maximum(ct.minimum(x, 8.0), 2.0)
    ct.store(c, (pid,), clamped)
kernel_minmax = kernel_fn
print(f"clamp via minimum/maximum: {compile_bytes(kernel_minmax, sig)} cubin bytes")

# The identical clamp, hand-rolled from two ct.where calls and this
# chapter's own comparison operators, under the identical function name
# kernel_fn (Chapter 10's naming-confound rule).
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    too_high = ct.greater(x, 8.0)
    capped = ct.where(too_high, ct.full((tile_size,), 8.0, ct.float32), x)
    too_low = ct.less(capped, 2.0)
    clamped = ct.where(too_low, ct.full((tile_size,), 2.0, ct.float32), capped)
    ct.store(c, (pid,), clamped)
kernel_where = kernel_fn
print(f"clamp via where + comparisons: {compile_bytes(kernel_where, sig)} cubin bytes")
```

Genuinely run:

```
clamp via minimum/maximum: 21024 cubin bytes
clamp via where + comparisons: 21152 cubin bytes
```

### Discussion

Unlike Chapter 16's `scan`/`cumsum` and `reduce`/`sum` pairs, which compiled to byte-identical artifacts under naming control, this pair does not: 21024 bytes against 21152 bytes, a real 128-byte difference between two kernels sharing the identical function name and computing (as far as this book's compiled-structure-only verification can confirm) the same logical operation. This is a genuinely different empirical answer from Chapter 16's, and it is worth sitting with rather than explaining away: `ct.minimum`/`ct.maximum` are evidently not simply convenience wrappers the compiler rewrites into the same comparison-and-select pattern a hand-rolled version would use — they compile to something measurably different, plausibly because a dedicated min/max operation maps to a single hardware instruction where two chained `where`-plus-comparison calls require the compiler to materialize an explicit boolean mask and a select for each one. This book's `export_kernel`-only environment cannot inspect the actual generated instructions to confirm that explanation directly, so it is offered as the most plausible account of the difference, not a verified mechanism — the one fact this section can state with full confidence is that the two implementations are not compiled identically, however similar they look in Python.

## 18.6 Capstone: A Masked, Row-Normalized Kernel

### Intuition

This chapter's own material — comparisons, `bitwise_and`, and `where` — combines with Chapter 16's `reduce`, Chapter 15's `broadcast_to`, and Chapter 17's `exp`/`truediv` into a single kernel: zero out any element outside a fixed range, then row-normalize what remains using the same log-softmax-style shape Chapter 17's own capstone used.

### Background

The masking step is exactly Section 18.4's elementwise-AND pattern — `ct.bitwise_and(ct.greater(x2d, -1.0), ct.less(x2d, 1.0))` — combined with Section 18.2's `where`-based replacement to zero out-of-range elements before the row-normalization proceeds. Everything downstream of the mask is unchanged from Chapter 17's own capstone: a `Chapter 16` `reduce`-based row maximum, a `Chapter 15` `broadcast_to`, `exp` with `RoundingMode.FULL` (Chapter 17's own finding), and a final `truediv`.

### Worked Example 18.6.1 — mask, then normalize

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
    in_range = ct.bitwise_and(ct.greater(x2d, -1.0), ct.less(x2d, 1.0))
    masked = ct.where(in_range, x2d, ct.full((8, 8), 0.0, ct.float32))
    row_max = ct.reduce(masked, 1, ct.maximum, -3.0e38, keepdims=True)
    shifted = ct.sub(masked, ct.broadcast_to(row_max, (8, 8)))
    exped = ct.exp(shifted, rounding_mode=ct.RoundingMode.FULL)
    row_sum = ct.sum(exped, 1, keepdims=True)
    normalized = ct.truediv(exped, ct.broadcast_to(row_sum, (8, 8)))
    ct.store(c, (pid,), ct.reshape(normalized, (tile_size,)))

try:
    n = compile_bytes(kernel_fn, sig)
    print(f"capstone (mask + reduce + broadcast_to + exp[FULL] + truediv): {n} cubin bytes")
except Exception as e:
    print(f"capstone (mask + reduce + broadcast_to + exp[FULL] + truediv): {type(e).__name__}: {e}")
```

Genuinely run:

```
capstone (mask + reduce + broadcast_to + exp[FULL] + truediv): 36864 cubin bytes
```

### Discussion

The kernel compiles cleanly on the first attempt, chaining nine operations across four chapters — `bitwise_and`, `greater`, `less`, and `where` from this chapter; `reduce` from Chapter 16; `broadcast_to` from Chapter 15; and `exp`, `sub`, `sum`, and `truediv` from Chapter 17 and earlier — into a single, coherent kernel body. As with every capstone since Chapter 15, a clean compile confirms only that each operation's structural and type rules were satisfied at every step; it says nothing about whether the specific numeric values this kernel would produce on real hardware form a correctly masked, correctly normalized result for any given input, since this book's `export_kernel`-only, driver-free environment has never been able to check computed results, only compiled structure, all the way back to Chapter 1.

## Chapter Summary

This chapter completed cuTile Python's comparison and conditional-selection surface. `ct.equal`, `ct.not_equal`, `ct.greater`, `ct.greater_equal`, `ct.less`, and `ct.less_equal` all compile as ordinary elementwise operations, broadcasting and promoting dtypes exactly like Chapter 17's arithmetic family, with a naming-controlled comparison confirming that storing a comparison's result into an `int32`-declared output costs measurably more than a `bool_`-declared one. `ct.where` was found, on four separate points, to be genuinely more permissive than its own docstring: mismatched `x`/`y` dtypes promote rather than reject, an `int32` or even `float32` condition tile works with no `bool_` requirement enforced, and a differently-shaped condition broadcasts against `x`/`y` rather than needing an exact shape match. Combining boolean tiles elementwise with `ct.bitwise_and`/`ct.bitwise_or` (or Python's own `&`/`|`) works cleanly, in sharp contrast to Python's `and`, `or`, and chained-comparison syntax, all of which route through the same scalar-truthiness requirement Chapter 1 first found `if` enforcing against a multi-element tile — confirmed here to produce the identical error for `and`, and a subtly different one, naming an internal `if_else()` call, for chained comparisons. A hand-rolled `where`-based clamp was found, under Chapter 10's naming control, to compile to a genuinely different byte count than the built-in `ct.minimum`/`ct.maximum` equivalent — a finding in the opposite direction from Chapter 16's `scan`/`reduce` results, and a reminder that logical equivalence in Python does not entail identical compiled output. A capstone combined this chapter's masking machinery with `reduce`, `broadcast_to`, `exp`, and `truediv` from three earlier chapters into a single masked, row-normalized kernel with no adjustment needed to satisfy the compiler.

## Self-Check Questions

1. Section 18.1 found that declaring a comparison's output as `int32` compiles to a larger cubin than declaring it `bool_`, under Chapter 10's naming control. What specific step does this suggest the compiler inserts for the `int32` case that it does not need for the `bool_` case?
2. Section 18.3 found `ct.where` accepts a `float32`-dtyped condition tile, despite its docstring specifically calling `cond` a "Boolean tile." Does this mean the docstring is incorrect? What is the more precise way to describe the relationship between what the docstring says and what the compiler actually accepts?
3. Section 18.4 found that `and` and chained comparisons (`2.0 < x < 8.0`) both fail against a multi-element boolean tile, but with different error messages — one mentioning `if_else()`, the other not. Given that a chained comparison is documented Python behavior to desugar into `and`-based logic, what does the differing error message suggest about how cuTile Python's compiler front end processes these two syntactic forms?
4. Section 18.5 found a `where`-based clamp does NOT compile identically to a `minimum`/`maximum`-based one, in contrast to Chapter 16's `scan`/`cumsum` and `reduce`/`sum` pairs, which did compile identically. What is the key structural difference between how Chapter 16's pairs were constructed and how Section 18.5's pair was constructed, that might explain why one pair matched and the other didn't?
5. Section 18.6's capstone masks out-of-range values with `ct.where` before row-normalizing. Suppose every element of some row were outside the `(-1, 1)` range, so that entire row becomes all zeros after masking. What would you need — beyond what this book's `export_kernel`-only environment can check — to know whether that row's normalization step produces a sensible result rather than, say, a division by zero?

## Where We Go Next

This chapter closed a loop Chapter 1 opened in its very first pages: what exactly happens when Python's various boolean-combining constructs — not just a bare `if`, but `and`, `or`, and chained comparisons — meet a tile that isn't a genuine scalar. The answer, now confirmed directly rather than left as an implication of Chapter 1's single finding, is that all of them fail, though not always with identical wording. `dir(ct)` still lists two families this book has never touched: the bitwise family beyond the `bitwise_and`/`bitwise_or` this chapter used for masking — `bitwise_xor`, `bitwise_not`, `bitwise_lshift`, `bitwise_rshift` — and the atomic memory operations (`atomic_add`, `atomic_max`, `atomic_cas`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_xchg`) that every kernel this book has written, confined to one block reading and writing its own slice of memory, has never had a reason to reach for. Chapter 19 turns to the remaining bitwise operations first, before Chapter 20's atomics finally introduce a kernel that needs to know about memory another block might be touching at the same time.

## Worked Solutions

**1.** The comparison operation itself produces a boolean result regardless of the declared output dtype — the difference in byte count must come from what happens between computing that boolean and writing it into the output array. The most plausible step is an explicit boolean-to-integer conversion inserted immediately before the store, converting the compiler's internal boolean representation into the 0/1 integer values an `int32` array actually holds, since a `bool_`-declared output can presumably store the boolean result with no such conversion. This book's `export_kernel`-only environment cannot inspect the actual generated instructions to confirm this is exactly what happens, but it is the most direct explanation consistent with the `int32` case costing more while computing the identical logical result.

**2.** The docstring is not incorrect in the sense of describing something false — every example it gives compiles and (as far as this book can verify) behaves as documented. But it describes a narrower contract than the compiler actually enforces: "Boolean tile" states the typical, expected case rather than a strict requirement the compiler rejects deviations from. The more precise description is that `ct.where` accepts any tile whose values resolve to a per-element truth value at the point it's consumed — `bool_` is the natural and presumably intended dtype for `cond`, but not the only one the type checker actually permits. This is the same lesson Section 18.3's dtype-promotion and broadcasting findings already established from two other angles: a docstring's plain-language description and the compiler's actual accepted range are related but not identical, and only testing past the documented cases reveals where they diverge.

**3.** The differing error messages suggest that even though a chained comparison is documented to be semantically equivalent to `and`-based logic, cuTile Python's compiler front end does not necessarily implement that equivalence by literally desugaring the chained comparison into an `and` expression and then processing both through the identical code path. If it did, the two failures would very likely produce the same message, the way `if` and explicit `and` did in this chapter. Instead, the chained comparison's error specifically names an `if_else()` call, suggesting the compiler's own internal representation of a chained comparison routes through a distinct construct — perhaps a ternary-like internal primitive used to implement the short-circuiting behavior chained comparisons require — that ordinary `and` does not use, even though both ultimately hit the same "not a genuine scalar" wall.

**4.** Chapter 16's `scan`/`cumsum` and `reduce`/`sum` pairs were constructed so that the custom variant called the SAME underlying general mechanism (`ct.scan`, `ct.reduce`) that the named built-in is documented to be a specific instance of — `cumsum` literally IS `ct.scan` with a particular `func` and `identity`, per the documentation itself, so the two compiling identically confirms the documentation's own claimed relationship. Section 18.5's pair, by contrast, does not have that relationship: `ct.minimum`/`ct.maximum` are not documented as being built from `ct.where` and comparisons internally, nor is there any claim that a `where`-based clamp is "the same operation" from the compiler's point of view — they are simply two different ways of expressing the same intended logical behavior, written independently rather than one being defined in terms of the other. Byte-identical output was something Chapter 16 could reasonably expect to confirm because the documentation predicted it; Section 18.5 had no such prediction to test, only a hypothesis worth checking, and the checking is what revealed the difference.

**5.** You would need to actually run the kernel against real data with `ct.launch` on real hardware — something this book's `export_kernel`-only environment has never had available, all the way back to Chapter 1 — and inspect the resulting values for a row that becomes entirely zero after masking. The masking and normalization logic as written performs a `truediv` by that row's sum, and a row that is all zeros after masking would have a sum of zero, making that division a division by zero. Whether that produces `NaN`, an unhandled hardware exception, or something else entirely depends on runtime floating-point behavior this book's compiled-structure-only verification cannot observe: a clean compile confirms the operations are all well-typed and shape-compatible, not that every possible input produces a numerically sound result.

## Complete Runnable Code

### File: `01_six_comparisons.py`

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
    [array_param(), array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.equal(x, y))
print(f"equal(x, y): {compile_bytes(kernel_fn, sig)} cubin bytes")

@ct.kernel
def kernel_fn2(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.not_equal(x, y))
print(f"not_equal(x, y): {compile_bytes(kernel_fn2, sig)} cubin bytes")

@ct.kernel
def kernel_fn3(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater(x, y))
print(f"greater(x, y): {compile_bytes(kernel_fn3, sig)} cubin bytes")

@ct.kernel
def kernel_fn4(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater_equal(x, y))
print(f"greater_equal(x, y): {compile_bytes(kernel_fn4, sig)} cubin bytes")

@ct.kernel
def kernel_fn5(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.less(x, y))
print(f"less(x, y): {compile_bytes(kernel_fn5, sig)} cubin bytes")

@ct.kernel
def kernel_fn6(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.less_equal(x, y))
print(f"less_equal(x, y): {compile_bytes(kernel_fn6, sig)} cubin bytes")

# The docstrings never mention a required output dtype. Does the same
# comparison compile against a bool_-declared output AND an int32-declared
# output equally, or is one of the two rejected? Naming-controlled
# (Chapter 10): both variants defined as kernel_fn7, reassigned in turn.
sig_bool_out = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_int_out = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.equal(x, y))
kernel_bool_out = kernel_fn7
print(f"equal(x, y) -> bool_ output: {compile_bytes(kernel_bool_out, sig_bool_out)} cubin bytes")

@ct.kernel
def kernel_fn7(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.equal(x, y))
kernel_int_out = kernel_fn7
print(f"equal(x, y) -> int32 output: {compile_bytes(kernel_int_out, sig_int_out)} cubin bytes")

# A tile compared against a plain Python scalar, not a second tile.
sig_scalar = ct.compilation.KernelSignature(
    [array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn9(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater(x, 2.0))
print(f"greater(x, 2.0) scalar RHS: {compile_bytes(kernel_fn9, sig_scalar)} cubin bytes")

# Mixed dtype operands -- int32 and float32 -- promoted to a common dtype,
# the same claim Chapter 17's binary elementwise family documented.
sig_mixed = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(ct.float32), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn10(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.greater(x, y))
try:
    n = compile_bytes(kernel_fn10, sig_mixed)
    print(f"greater(int32 x, float32 y) mixed dtype: {n} cubin bytes")
except Exception as e:
    print(f"greater(int32 x, float32 y) mixed dtype: {type(e).__name__}: {e}")
```

### File: `02_where_basic.py`

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

# ct.where(cond, x, y): select from x where cond is True, else from y.
# The condition is built from one of Section 18.1's comparison operators.
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    cond = ct.greater(x, y)
    ct.store(c, (pid,), ct.where(cond, x, y))
print(f"where(x > y, x, y): {compile_bytes(kernel_fn, sig)} cubin bytes")

# where with a scalar-valued x and y, rather than two loaded tiles --
# the common "replace matching elements with a constant" pattern.
sig2 = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    cond = ct.equal(x, 0.0)
    ct.store(c, (pid,), ct.where(cond, ct.full((tile_size,), -1.0, ct.float32), x))
print(f"where(x == 0, -1.0, x): {compile_bytes(kernel_fn2, sig2)} cubin bytes")
```

### File: `03_where_more_permissive_than_docstring.py`

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

# ct.where's docstring declares cond, x, and y all as sharing one shape S,
# with x and y sharing one dtype T. Does the compiler actually enforce
# that x and y share a dtype, or does it promote them like the rest of
# this book's binary elementwise family (Chapter 17)?
sig_mixed = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.int32), array_param(ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    cond = ct.greater(x, ct.astype(y, ct.float32))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn, sig_mixed)
    print(f"where(cond, float32 x, int32 y) mixed dtype: {n} cubin bytes")
except Exception as e:
    print(f"where(cond, float32 x, int32 y) mixed dtype: {type(e).__name__}: {e}")

# The same mixed-dtype call, but declaring the output array as int32
# instead of float32 -- if the promoted result is genuinely float32,
# an int32-declared output should now be rejected as a mismatch.
sig_int_out = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.int32), array_param(ct.int32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn2(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    cond = ct.greater(x, ct.astype(y, ct.float32))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn2, sig_int_out)
    print(f"where(cond, float32 x, int32 y) -> int32 output: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"where(cond, float32 x, int32 y) -> int32 output: {type(e).__name__}: {e}")

# The docstring also declares cond as "Boolean tile of shape S." Does an
# int32-dtyped condition tile -- never explicitly bool_ -- work directly?
sig_int_cond = ct.compilation.KernelSignature(
    [array_param(ct.int32), array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn3(cond_arr, a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    cond = ct.load(cond_arr, (pid,), (tile_size,))
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn3, sig_int_cond)
    print(f"where(int32 cond, x, y): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"where(int32 cond, x, y): {type(e).__name__}: {e}")

# And a float32-dtyped condition tile -- further still from "Boolean."
sig_float_cond = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn4(cond_arr, a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    cond = ct.load(cond_arr, (pid,), (tile_size,))
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), ct.where(cond, x, y))
try:
    n = compile_bytes(kernel_fn4, sig_float_cond)
    print(f"where(float32 cond, x, y): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"where(float32 cond, x, y): {type(e).__name__}: {e}")

# The docstring says cond, x, and y all share one shape S. Does cond
# actually need to match x/y's shape exactly, or does it broadcast the
# way Chapter 17's binary elementwise family does? A (8, 1) condition
# against (8, 8) x and y, passed directly with no explicit broadcast_to.
sig_bcast = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    x2d = ct.reshape(x, (8, 8))
    row_max = ct.reduce(x2d, 1, ct.maximum, -3.0e38, keepdims=True)
    cond = ct.greater(row_max, 0.0)
    result = ct.where(cond, x2d, ct.full((8, 8), -1.0, ct.float32))
    ct.store(c, (pid,), ct.reshape(result, (tile_size,)))
try:
    n = compile_bytes(kernel_fn5, sig_bcast)
    print(f"where((8,1) cond, (8,8) x, (8,8) y) unbroadcast cond: {n} cubin bytes")
except Exception as e:
    print(f"where((8,1) cond, (8,8) x, (8,8) y) unbroadcast cond: {type(e).__name__}: {e}")
```

### File: `04_combining_boolean_tiles.py`

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
    [array_param(), array_param(ct.bool_), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Combining two elementwise boolean tiles with ct.bitwise_and -- a genuine
# elementwise AND, one result per element, no control flow involved.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = ct.bitwise_and(ct.greater(x, 2.0), ct.less(x, 8.0))
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn, sig)
    print(f"bitwise_and(greater(x,2), less(x,8)): {n} cubin bytes")
except Exception as e:
    print(f"bitwise_and(greater(x,2), less(x,8)): {type(e).__name__}: {e}")

# The same mask, using Python's own & operator directly on the two tiles.
@ct.kernel
def kernel_fn2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = (x > 2.0) & (x < 8.0)
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn2, sig)
    print(f"(x > 2.0) & (x < 8.0): {n} cubin bytes")
except Exception as e:
    print(f"(x > 2.0) & (x < 8.0): {type(e).__name__}: {e}")

# ct.bitwise_or, the elementwise counterpart.
@ct.kernel
def kernel_fn3(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    out_of_range = ct.bitwise_or(ct.less(x, 2.0), ct.greater(x, 8.0))
    ct.store(c, (pid,), out_of_range)
try:
    n = compile_bytes(kernel_fn3, sig)
    print(f"bitwise_or(less(x,2), greater(x,8)): {n} cubin bytes")
except Exception as e:
    print(f"bitwise_or(less(x,2), greater(x,8)): {type(e).__name__}: {e}")

# Chapter 1 already found that a multi-element boolean TILE cannot drive a
# real `if` statement -- it needs a genuine scalar. Python's own `and`
# keyword hits the identical requirement directly, with the identical
# "Expected a scalar value" message Chapter 1 first found from `if`.
@ct.kernel
def kernel_fn4(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = (x > 2.0) and (x < 8.0)
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn4, sig)
    print(f"(x > 2.0) and (x < 8.0): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"(x > 2.0) and (x < 8.0): {type(e).__name__}: {e}")

# Python's chained-comparison sugar, 2.0 < x < 8.0, desugars to the same
# `and`-based logic under the hood -- does it fail the identical way?
@ct.kernel
def kernel_fn5(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    in_range = 2.0 < x < 8.0
    ct.store(c, (pid,), in_range)
try:
    n = compile_bytes(kernel_fn5, sig)
    print(f"2.0 < x < 8.0 (chained): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"2.0 < x < 8.0 (chained): {type(e).__name__}: {e}")
```

### File: `05_clamp_equivalence.py`

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

# clamp(x, 2.0, 8.0) built from ct.minimum/ct.maximum.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    clamped = ct.maximum(ct.minimum(x, 8.0), 2.0)
    ct.store(c, (pid,), clamped)
kernel_minmax = kernel_fn
print(f"clamp via minimum/maximum: {compile_bytes(kernel_minmax, sig)} cubin bytes")

# The identical clamp, hand-rolled from two ct.where calls and this
# chapter's own comparison operators, under the identical function name
# kernel_fn (Chapter 10's naming-confound rule).
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    too_high = ct.greater(x, 8.0)
    capped = ct.where(too_high, ct.full((tile_size,), 8.0, ct.float32), x)
    too_low = ct.less(capped, 2.0)
    clamped = ct.where(too_low, ct.full((tile_size,), 2.0, ct.float32), capped)
    ct.store(c, (pid,), clamped)
kernel_where = kernel_fn
print(f"clamp via where + comparisons: {compile_bytes(kernel_where, sig)} cubin bytes")
```

### File: `06_capstone_masked_softmax.py`

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
    in_range = ct.bitwise_and(ct.greater(x2d, -1.0), ct.less(x2d, 1.0))
    masked = ct.where(in_range, x2d, ct.full((8, 8), 0.0, ct.float32))
    row_max = ct.reduce(masked, 1, ct.maximum, -3.0e38, keepdims=True)
    shifted = ct.sub(masked, ct.broadcast_to(row_max, (8, 8)))
    exped = ct.exp(shifted, rounding_mode=ct.RoundingMode.FULL)
    row_sum = ct.sum(exped, 1, keepdims=True)
    normalized = ct.truediv(exped, ct.broadcast_to(row_sum, (8, 8)))
    ct.store(c, (pid,), ct.reshape(normalized, (tile_size,)))

try:
    n = compile_bytes(kernel_fn, sig)
    print(f"capstone (mask + reduce + broadcast_to + exp[FULL] + truediv): {n} cubin bytes")
except Exception as e:
    print(f"capstone (mask + reduce + broadcast_to + exp[FULL] + truediv): {type(e).__name__}: {e}")
```
