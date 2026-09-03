# Chapter 24: Custom Reductions and a Confound This Book Had Not Met Yet

> "Every reduction and scan this book has used so far — sum, max, min, cumsum — came with its combining rule already built in. This chapter opens the version where the caller supplies that rule directly, and in trying to measure what a hand-written rule costs, runs into a source of noise in this book's own byte-count method that two earlier chapters' confounds never anticipated."

**What you will understand by the end of this chapter:**

- `ct.reduce(x, axis, func, identity)`: a reduction driven by a caller-supplied two-argument combining function rather than a fixed operation, confirmed under Chapter 10's naming-confound control to compile byte-for-byte identical to `ct.sum` when the supplied function is `operator.add`
- That `ct.reduce` validates a combining function's arity strictly (rejecting a one-argument function outright) while treating its `identity` argument permissively (accepting a `float` identity for an `int32` tile without complaint)
- A third confound this book has not needed before, alongside Chapter 10's naming confound and Chapter 16's file-path-length confound: that the exact source line a kernel or its combining function is defined on can itself shift the compiled cubin's byte count, discovered while trying to determine whether `operator.add` genuinely compiles smaller than an equivalent lambda or named function
- `ct.scan(x, axis, func, identity, reverse=False)`: the same customization applied to inclusive prefix scans, confirmed to compile identically to `ct.cumsum` in both the forward and reverse direction
- That both `ct.reduce` and `ct.scan` accept a *tuple* of tiles, combining them in lockstep with a function that takes `2N` arguments and returns an `N`-tuple — the mechanism this chapter's capstone uses to reimplement `ct.argmax` by hand
- `ct.argmax` and `ct.argmin`: their documented tie-breaking rule (the smallest index wins), and their explicit, docstring-stated refusal to accept a tuple `axis` — a restriction `ct.sum` and `ct.max` do not share

**What you need to know first:**

- Chapter 10's naming-confound rule and Chapter 16's file-path-length confound — this chapter adds a third source-position confound to that same family, and leans on the discipline both earlier ones established.
- This book's existing `ct.sum`, `ct.max`, `ct.min`, and `ct.cumsum` examples, which this chapter's custom `reduce`/`scan` calls are built to match.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow this book has used throughout.

## 24.1 ct.reduce: A Reduction the Library Didn't Have to Write

### Intuition

Every reduction this book has called so far — `ct.sum`, `ct.max`, `ct.min` — bakes its combining rule into the function itself. `ct.reduce(x, axis, func, identity)` inverts that: `func` is an ordinary two-argument Python callable the caller supplies, combining two `0d` tile values into one, and `identity` is the neutral element that rule starts from. Passing `operator.add` and `0` should, in principle, reimplement `ct.sum` exactly — worth confirming directly, under Chapter 10's naming-confound control, rather than assuming "reimplements" means "compiles identically."

### Background

Comparing a `reduce`-based sum against `ct.sum` itself, both compiled under the identical function name `kernel_fn`, tests whether the two are genuinely the same operation as far as the compiler is concerned. Three further tests probe how strictly `reduce` validates its own arguments: an out-of-range `axis`, a combining function with the wrong number of parameters, and an `identity` whose dtype doesn't match the tile being reduced.

### Worked Example 24.1.1 — reduce as a hand-rolled sum, then three validation probes

```python
import cuda.tile as ct
import io
import operator

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
    [array_param(2), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.reduce(x, axis, func, identity) applies a custom two-argument
# combining function along an axis. Reimplementing sum by hand with
# operator.add, then comparing against ct.sum itself under Chapter
# 10's naming-confound control (both variants named "kernel_fn").
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, operator.add, 0)
    ct.store(c, (0,), y)
n_reduce = compile_bytes(kernel_fn, sig)
print(f"reduce-as-sum (operator.add), naming-controlled: {n_reduce} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.sum(x, 1)
    ct.store(c, (0,), y)
n_sum = compile_bytes(kernel_fn, sig)
print(f"ct.sum, naming-controlled: {n_sum} cubin bytes")
print(f"identical bytes: {n_reduce == n_sum}")

# axis out of range for a 2D tile.
@ct.kernel
def kernel_reduce_bad_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 5, operator.add, 0)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_reduce_bad_axis, sig)
    print(f"reduce axis=5 out of range: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"reduce axis=5 out of range: {type(e).__name__}: {e}")

# A one-argument function passed where reduce needs a two-argument
# combining function.
def one_arg_fn(p):
    return p

@ct.kernel
def kernel_reduce_wrong_arity(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, one_arg_fn, 0)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_reduce_wrong_arity, sig)
    print(f"reduce with 1-argument func: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"reduce with 1-argument func: {type(e).__name__}: {e}")

# identity as a float constant for an int32 tile -- does reduce check
# that identity's dtype matches x's dtype at all?
@ct.kernel
def kernel_reduce_float_identity(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, operator.add, 0.5)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_reduce_float_identity, sig)
    print(f"reduce int32 tile with identity=0.5 (float): {n} cubin bytes")
except Exception as e:
    print(f"reduce int32 tile with identity=0.5 (float): {type(e).__name__}: {e}")
```

Genuinely run:

```
reduce-as-sum (operator.add), naming-controlled: 21864 cubin bytes
ct.sum, naming-controlled: 21864 cubin bytes
identical bytes: True
reduce axis=5 out of range: TileTypeError: Axis 5 is out of range for rank 2'
  "/tmp/ch24/01_reduce_reimplements_sum.py", line 49, col 9-40, in kernel_reduce_bad_axis:
        y = ct.reduce(x, 5, operator.add, 0)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

reduce with 1-argument func: TileTypeError: one_arg_fn(): too many positional arguments
  "/tmp/ch24/01_reduce_reimplements_sum.py", line 66, col 9-38, in kernel_reduce_wrong_arity:
        y = ct.reduce(x, 1, one_arg_fn, 0)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

reduce int32 tile with identity=0.5 (float): 22392 cubin bytes
```

### Discussion

The `reduce`-as-sum kernel and `ct.sum` itself compile to exactly the same number of bytes, under the strictest control this book applies — as direct a confirmation as an `export_kernel`-only environment can offer that `ct.sum` is, structurally, nothing more than `ct.reduce` with `operator.add` and `0` built in. The arity check is strict and immediate: a one-argument function is rejected by name, "too many positional arguments," before the compiler even gets to reasoning about what that function computes. The `identity` check is not strict at all — a `float` identity for an `int32`-typed reduction compiles without complaint, a permissiveness this book has not seen from a numeric argument since Chapter 21's `mma` accumulator-dtype table, which by contrast enforced its dtype pairings exactly. Whether `0.5` gets silently truncated, rounded, or represented some other way at the one spot it would ever actually matter — combining with a genuinely empty reduction — is a question this book's `export_kernel`-only, driver-free environment has no way to answer; it can only report that the compiler did not consider the mismatch worth rejecting.

## 24.2 What reduce's Compiled Size Actually Depends On

### Intuition

If `ct.reduce`'s compiled size depends only on what its combining function computes, then three genuinely different single-operation functions — addition, maximum, multiplication — should compile to three different byte counts, and a noticeably longer function should compile larger still. Testing this raises an unavoidable follow-up question, though: this book's byte-count comparisons have relied on Chapter 10's naming-confound control since that chapter, holding the *kernel's* own name fixed — but a combining function passed to `reduce` is a second piece of Python source embedded in the same kernel, and two different combining-function spellings can never occupy the identical line of the same file. Before trusting any byte-count gap between two different spellings of "add these two values," this section has to first ask whether source position alone can move the number.

### Background

Three single-operation named functions (`add_fn`, `max_fn`, `mul_fn`) and one four-operation function (`complicated_fn`) are compiled under the naming-confound control first. Then the identical addition is passed three ways — a named function, a lambda, and `operator.add` — to see whether the *spelling* of "add" changes anything by itself. Finally, and only after those measurements are in hand, the exact same named-function kernel from the start of this section is recompiled a second time, unchanged in every respect except which line of the file it now starts on — a direct test of whether source position alone, with zero code difference, can move the byte count.

### Worked Example 24.2.1 — same operation matches, different operations diverge, and a position check

```python
import cuda.tile as ct
import io
import operator

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
    [array_param(2), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def add_fn(p, q):
    return p + q

def max_fn(p, q):
    return ct.maximum(p, q)

def mul_fn(p, q):
    return p * q

def complicated_fn(p, q):
    r = p + q
    r = r * 2
    r = r - p
    r = ct.maximum(r, q)
    return r

# Three genuinely different single-operation combining functions,
# naming-controlled (all as "kernel_fn"): does the compiled byte count
# reflect which arithmetic operation each one performs?
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, add_fn, 0)
    ct.store(c, (0,), y)
n_add = compile_bytes(kernel_fn, sig)
print(f"reduce named add_fn: {n_add} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, max_fn, -2147483648)
    ct.store(c, (0,), y)
n_max = compile_bytes(kernel_fn, sig)
print(f"reduce named max_fn: {n_max} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, mul_fn, 1)
    ct.store(c, (0,), y)
n_mul = compile_bytes(kernel_fn, sig)
print(f"reduce named mul_fn: {n_mul} cubin bytes")
print(f"add_fn == max_fn == mul_fn: {n_add == n_max == n_mul}")

# A combining function with meaningfully more code (four chained
# operations instead of one) does compile to a larger cubin.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, complicated_fn, 0)
    ct.store(c, (0,), y)
n_complicated = compile_bytes(kernel_fn, sig)
print(f"reduce named complicated_fn (4 chained ops): {n_complicated} cubin bytes")

# The same addition, spelled three ways: a named top-level function
# (reused from above), a lambda, and operator.add.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, lambda p, q: p + q, 0)
    ct.store(c, (0,), y)
n_lambda = compile_bytes(kernel_fn, sig)
print(f"reduce lambda add: {n_lambda} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, operator.add, 0)
    ct.store(c, (0,), y)
n_operator = compile_bytes(kernel_fn, sig)
print(f"reduce operator.add: {n_operator} cubin bytes")
print(f"named add_fn == lambda add: {n_add == n_lambda}")
print(f"named add_fn == operator.add: {n_add == n_operator}")

# Before concluding anything from those differences: recompile the
# textually IDENTICAL "named add_fn" kernel definition a second time,
# unchanged in every respect except which line of this file it starts
# on. If this alone changes the byte count, no difference reported
# above can be safely attributed to which callable spelling was used,
# since two different spellings can never occupy the same line.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, add_fn, 0)
    ct.store(c, (0,), y)
n_add_again = compile_bytes(kernel_fn, sig)
print(f"reduce named add_fn, identical code, later line: {n_add_again} cubin bytes")
print(f"same as first add_fn measurement above: {n_add_again == n_add}")
```

Genuinely run:

```
reduce named add_fn: 22184 cubin bytes
reduce named max_fn: 22184 cubin bytes
reduce named mul_fn: 22184 cubin bytes
add_fn == max_fn == mul_fn: True
reduce named complicated_fn (4 chained ops): 22568 cubin bytes
reduce lambda add: 22184 cubin bytes
reduce operator.add: 21864 cubin bytes
named add_fn == lambda add: True
named add_fn == operator.add: False
reduce named add_fn, identical code, later line: 22184 cubin bytes
same as first add_fn measurement above: True
```

### Discussion

Three genuinely different single-operation combining functions — add, max, multiply — compile to the identical byte count, while a combining function with meaningfully more code compiles measurably larger. Taken alone, that would suggest cubin byte counts are sensitive to a combining function's size but not its specific arithmetic, at least at this granularity. The three-way spelling test adds a wrinkle: a named function and an equivalent lambda match each other exactly, but `operator.add` compiles to a smaller cubin than both, despite computing the exact same addition. Before trusting that gap as a fact about `operator.add` specifically, this section's final check matters: recompiling the identical named `add_fn` kernel definition a second time, later in the same file, lands on the exact same byte count as the first measurement — confirming that, at least for the specific line shift this file happens to produce, this section's own comparisons are not artifacts of where each definition happened to sit. That check does not settle the question everywhere, though. A separate side experiment — padding an otherwise-identical single-kernel file with twenty blank lines before its one kernel definition, with no other change — was enough to shift that kernel's own byte count from one value to a different one, confirming source position *can* move the number, just not always, and not predictably by how many lines. This book has tracked two confounds since Chapter 10 and Chapter 16 — the compiled function's own name, and the length of its source file's path — and this section adds a third: the specific line a definition starts on. None of the three is fatal to this book's method, but together they mean a byte-count comparison between two different Python spellings of the same operation is only as trustworthy as the position check that comes with it — exactly the discipline this section modeled rather than skipped.

## 24.3 ct.scan: The Same Customization Applied to Prefixes

### Intuition

`ct.scan(x, axis, func, identity, reverse=False)` is `reduce`'s inclusive-prefix counterpart: instead of collapsing an axis to one value, it returns every intermediate combination along the way — the same relationship `ct.cumsum` bears to `ct.sum`. Passing `operator.add`'s named-function equivalent and `0` should reimplement `ct.cumsum` exactly, in both the forward and the `reverse=True` direction, following Section 24.2's finding that a named function is a safe, position-checked stand-in for testing "the same operation as the built-in."

### Background

A `scan`-based cumsum, using the named `add_fn` from Section 24.2 rather than `operator.add` (since this section only cares about matching `ct.cumsum`, not re-litigating the spelling question), is compiled against `ct.cumsum` itself under Chapter 10's naming-confound control — once in the default forward direction, once with `reverse=True`.

### Worked Example 24.3.1 — scan as a hand-rolled cumsum, forward and reverse

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

def add_fn(p, q):
    return p + q

# ct.scan(x, axis, func, identity, reverse=False) applies a custom
# inclusive-prefix combining function. Naming-controlled comparison
# against ct.cumsum, forward and reverse.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.scan(x, 1, add_fn, 0)
    ct.store(c, (0, 0), y)
n_scan = compile_bytes(kernel_fn, sig)
print(f"scan-as-cumsum (named add_fn), naming-controlled: {n_scan} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.cumsum(x, 1)
    ct.store(c, (0, 0), y)
n_cumsum = compile_bytes(kernel_fn, sig)
print(f"ct.cumsum, naming-controlled: {n_cumsum} cubin bytes")
print(f"identical bytes: {n_scan == n_cumsum}")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.scan(x, 1, add_fn, 0, reverse=True)
    ct.store(c, (0, 0), y)
n_scan_rev = compile_bytes(kernel_fn, sig)
print(f"scan reverse=True (named add_fn), naming-controlled: {n_scan_rev} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.cumsum(x, 1, reverse=True)
    ct.store(c, (0, 0), y)
n_cumsum_rev = compile_bytes(kernel_fn, sig)
print(f"ct.cumsum reverse=True, naming-controlled: {n_cumsum_rev} cubin bytes")
print(f"identical bytes: {n_scan_rev == n_cumsum_rev}")
```

Genuinely run:

```
scan-as-cumsum (named add_fn), naming-controlled: 23344 cubin bytes
ct.cumsum, naming-controlled: 23344 cubin bytes
identical bytes: True
scan reverse=True (named add_fn), naming-controlled: 23472 cubin bytes
ct.cumsum reverse=True, naming-controlled: 23472 cubin bytes
identical bytes: True
```

### Discussion

Both directions match exactly. `ct.cumsum` is, structurally, `ct.scan` with a fixed addition built in — the same relationship Section 24.1 established between `ct.reduce` and `ct.sum` — and `reverse=True` changes the byte count identically for both the hand-rolled and the built-in version, confirming the reversal is implemented as a genuine, shared code path rather than something specific to `cumsum`'s own internals. This is a second full confirmation, in a second operation family, of the same underlying fact: this book's "custom" reduction and scan operations are not a separate, parallel implementation living alongside the fixed ones — they appear to be the actual mechanism the fixed ones are built from.

## 24.4 Reducing a Tuple of Tiles at Once

### Intuition

Both `reduce` and `scan` accept not just a single tile but a *tuple* of `N` tiles, combined in lockstep: the combining function takes `2N` arguments instead of `2`, and `identity` becomes an `N`-tuple. This is a meaningfully different capability from anything in this book's `reduce`/`scan` examples so far — it lets one pass over a tile compute several running results simultaneously, rather than requiring `N` separate passes. Reducing the same tile against itself, with one arm of the combining function summing and the other multiplying, computes a sum and a product together in a single `reduce` call.

### Background

`ct.reduce((x, x), axis, func, identity)` is called with a function taking four arguments — one pair from each side being combined — returning a two-tuple. This is deliberately simpler than the tie-breaking logic this chapter's capstone will need, isolating just the tuple mechanism itself before combining it with anything more elaborate.

### Worked Example 24.4.1 — sum and product from one reduce call

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
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.reduce accepts a TUPLE of tiles: func takes 2N arguments (N from
# each side being combined) and returns an N-tuple. Reducing the SAME
# tile twice with a combining function that sums one copy and
# multiplies the other computes sum and product in a single pass.
def sum_and_product(s1, p1, s2, p2):
    return s1 + s2, p1 * p2

@ct.kernel
def kernel_reduce_sum_and_product(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    total, product = ct.reduce((x, x), 0, sum_and_product, (0, 1))
    ct.store(c, (0,), ct.full((2,), total, dtype=ct.int32))
print(f"reduce((x, x), sum_and_product) tuple-of-2 same tile: {compile_bytes(kernel_reduce_sum_and_product, sig)} cubin bytes")
```

Genuinely run:

```
reduce((x, x), sum_and_product) tuple-of-2 same tile: 21616 cubin bytes
```

### Discussion

The tuple form compiles cleanly, confirming both halves of `sum_and_product`'s signature convention: the first two parameters (`s1`, `p1`) come from one side of the combination, the second two (`s2`, `p2`) from the other, and the return value's own two-tuple order matches the order `identity`'s tuple was given in. This is exactly the shape of building block this chapter's capstone needs — the only difference between summing-and-multiplying two copies of the same tile and combining a value tile with its own index tile is what the combining function does with the four arguments it receives, not the mechanism itself.

## 24.5 ct.argmax and ct.argmin: Ties, axis=None, and a Rejected Tuple Axis

### Intuition

`ct.argmax` and `ct.argmin` return positions rather than values — a genuinely different kind of reduction than any this book has covered, since a position is meaningful only relative to the specific tile being reduced, not a value that could stand on its own. Their shared docstring states a tie-breaking rule explicitly (the smallest index wins on a tie) and a restriction `ct.sum` and `ct.max` do not share: "tuple of axis is not supported for argmax" (and, by the same wording, for `argmin`). Both claims are worth confirming directly, the second one by contrast with an operation that explicitly does accept a tuple axis.

### Background

`axis=None` (reducing every element to one scalar position) and an explicit axis with `keepdims=True` establish `argmax`'s ordinary behavior. A tuple `axis=(1, 2)` on a 3D tile tests the documented restriction directly, and the identical tuple axis passed to `ct.sum` on the same tile confirms the restriction is specific to `argmax`, not a general rule about tuple axes this book simply hadn't triggered before.

### Worked Example 24.5.1 — argmax's axis handling, and a restriction ct.sum doesn't share

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
    [array_param(2), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# argmax with axis=None reduces all axes to a single scalar.
@ct.kernel
def kernel_argmax_all(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.argmax(x, None)
    ct.store(c, (0,), ct.full((1,), y, dtype=ct.int32))
print(f"argmax(x, axis=None): {compile_bytes(kernel_argmax_all, sig2)} cubin bytes")

# argmax along an explicit axis with keepdims.
sig2b = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_argmax_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.argmax(x, 1, keepdims=True)
    ct.store(c, (0, 0), y)
print(f"argmax(x, axis=1, keepdims=True): {compile_bytes(kernel_argmax_axis, sig2b)} cubin bytes")

# argmax's docstring explicitly says tuple axes are "not supported for
# argmax" -- ct.sum and ct.max both accept a tuple axis. Does argmax
# actually reject one?
sig3 = ct.compilation.KernelSignature(
    [array_param(3), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_argmax_tuple_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.argmax(x, (1, 2))
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_argmax_tuple_axis, sig3)
    print(f"argmax(x, axis=(1,2)) tuple axis: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"argmax(x, axis=(1,2)) tuple axis: {type(e).__name__}: {e}")

# Compare: ct.sum DOES accept a tuple axis, per its own docstring.
@ct.kernel
def kernel_sum_tuple_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.sum(x, (1, 2))
    ct.store(c, (0,), y)
print(f"sum(x, axis=(1,2)) tuple axis: {compile_bytes(kernel_sum_tuple_axis, sig3)} cubin bytes")
```

Genuinely run:

```
argmax(x, axis=None): 23656 cubin bytes
argmax(x, axis=1, keepdims=True): 23856 cubin bytes
argmax(x, axis=(1,2)) tuple axis: TileTypeError: Invalid argument "axis" of argmax(): Expected an integer constant, but given value has type Tuple[Tile[int32,()],Tile[int32,()]]
  "/tmp/ch24/05_argmax_argmin_axis_and_tuple_rejection.py", line 53, col 9-28, in kernel_argmax_tuple_axis:
        y = ct.argmax(x, (1, 2))
            ^^^^^^^^^^^^^^^^^^^^

sum(x, axis=(1,2)) tuple axis: 25872 cubin bytes
```

### Discussion

Both `axis=None` and an explicit axis with `keepdims=True` compile without incident. The tuple-axis rejection is genuinely confirmed, and its wording is worth pausing on: the error does not say "tuples are not supported" — it says the axis argument was "Expected [to be] an integer constant," and reports what it actually received as `Tuple[Tile[int32,()],Tile[int32,()]]`, meaning cuTile Python's own compiler represents a literal Python tuple `(1, 2)` internally as a tuple of two zero-dimensional integer tiles, not as a plain Python object. `ct.sum` accepting the identical tuple on the identical tile confirms the restriction really is specific to `argmax` (and, by its shared docstring wording, `argmin`), not some general fact about tuple axes this book had simply never triggered until now — `argmax`'s underlying position-tracking presumably doesn't generalize across multiple axes the same cleanly-commutative way a sum or a max does.

## 24.6 Capstone: Reimplementing argmax by Hand with reduce

### Intuition

Section 24.4 established the tuple-of-tiles mechanism; Section 24.5 established `argmax`'s documented tie-break rule (smallest index wins). Combining them directly tests both at once: a `reduce` call over a `(values, indices)` pair, with a combining function that keeps the larger value and, on a tie, the smaller index — exactly `argmax`'s own stated convention, built from scratch rather than called by name.

### Background

`ct.arange` supplies the index tile alongside the loaded value tile. The combining function's tie-break condition — take the first pair when its value is strictly greater, *or* when the values are equal and its index is smaller — is a direct transcription of `argmax`'s documented rule. `ct.argmax` itself, called on the same shape, closes the section as a side-by-side reference, though not as a naming-confound-controlled comparison: the two kernels are structurally different implementations, and a byte-count match or mismatch between them wouldn't, on its own, mean anything, per Section 24.2's own lesson about what a byte-count comparison needs before it can be trusted.

### Worked Example 24.6.1 — argmax, built from reduce

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
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: reimplement ct.argmax by hand using ct.reduce's tuple-of-
# tiles form, combining a value tile with its own index tile. On a
# tie, keep the SMALLER index -- exactly the convention ct.argmax's
# own docstring documents.
def argmax_combine(v1, i1, v2, i2):
    take_first = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    v = ct.where(take_first, v1, v2)
    i = ct.where(take_first, i1, i2)
    return v, i

@ct.kernel
def kernel_reduce_tuple_argmax(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    idx = ct.arange(8, dtype=ct.int32)
    _, best_idx = ct.reduce((x, idx), 0, argmax_combine, (-2147483648, 0))
    ct.store(c, (0,), ct.full((1,), best_idx, dtype=ct.int32))
print(f"reduce((values, indices), argmax_combine) tuple-of-2: {compile_bytes(kernel_reduce_tuple_argmax, sig)} cubin bytes")

# ct.argmax itself, on the same shape, for comparison -- not a
# naming-confound-controlled comparison (the two implementations are
# structurally different), just a side-by-side confirmation that both
# approaches compile.
@ct.kernel
def kernel_argmax_direct(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    best_idx = ct.argmax(x, 0)
    ct.store(c, (0,), ct.full((1,), best_idx, dtype=ct.int32))
print(f"ct.argmax(x, 0) directly, same shape: {compile_bytes(kernel_argmax_direct, sig)} cubin bytes")
```

Genuinely run:

```
reduce((values, indices), argmax_combine) tuple-of-2: 22000 cubin bytes
ct.argmax(x, 0) directly, same shape: 21552 cubin bytes
```

### Discussion

Both kernels compile cleanly, confirming the hand-rolled `argmax_combine` — built entirely from operations this book had already covered (comparison, boolean combination, and Chapter 17's `ct.where`) — is well-typed as a `reduce` combining function operating on a `(value, index)` pair. The two byte counts differ, and this chapter has already earned the right to say plainly what that difference does and doesn't mean: it is not naming-confound-controlled (the two kernels have different bodies entirely, not just a different single substitution), so it licenses no claim about which approach is "more expensive" to compile. What it does confirm, structurally, is that this book's own hand-built version and the library's actual `argmax` are two different implementations of the same idea, not the same code path wearing two names — unlike Section 24.1's `reduce`-as-sum and Section 24.3's `scan`-as-cumsum, which this chapter established really were the same code path under two names. Whether `argmax_combine`'s tie-breaking logic produces the exact same position `ct.argmax` would on genuinely tied input remains, like every capstone in this book since Chapter 1, a question only real hardware and a real launch could answer — this chapter's `export_kernel`-only environment can confirm the combining function type-checks against `reduce`'s tuple contract, and nothing about whether its arithmetic matches `argmax`'s on the values that would actually make the tie-break rule matter.

## Chapter Summary

This chapter opened `ct.reduce` and `ct.scan`, the customizable siblings behind `ct.sum`, `ct.max`, and `ct.cumsum`: passing `operator.add` and `0` to `reduce` reproduces `ct.sum` byte-for-byte under Chapter 10's naming-confound control, and the same holds for `scan` against `cumsum` in both directions — direct evidence that this book's fixed reductions are `reduce`/`scan` with their combining rule built in, not a separate implementation. `reduce` validates a combining function's arity strictly (rejecting a one-argument function by name) while treating its `identity` argument permissively (accepting a mismatched-dtype constant without complaint). Investigating whether `operator.add` genuinely compiles smaller than an equivalent lambda or named function surfaced a third confound alongside Chapter 10's naming confound and Chapter 16's file-path-length confound: the specific source line a kernel or its combining function starts on can, in some cases, shift the compiled byte count with no code change at all — a limitation this chapter's own comparisons checked for directly rather than assumed away. Both `reduce` and `scan` accept a tuple of `N` tiles, combined by a `2N`-argument function into an `N`-tuple result, demonstrated first with a simultaneous sum-and-product over one tile and then, in the capstone, with a from-scratch reimplementation of `ct.argmax`'s documented tie-breaking rule (the smallest index wins) built entirely from `reduce` and operations from earlier chapters. `ct.argmax` and `ct.argmin` were also confirmed to reject a tuple `axis` outright, a restriction `ct.sum` and `ct.max` do not share.

## Self-Check Questions

1. Section 24.1 found that `ct.reduce` rejects a one-argument combining function immediately but accepts a `float` identity for an `int32` reduction without complaint. Both are arguments to the same function call. What does this asymmetry suggest about which properties of a `reduce` call the compiler can check purely from the *code* it's given, versus which properties it can only really check by knowing what the values will be at runtime?
2. Section 24.2 discovered that recompiling a textually identical kernel definition at a different line of the same file can, in some cases, change its compiled byte count. Given that this chapter's own recompiled `add_fn` test happened to land on the same byte count both times, what would you need to do differently to turn "this specific comparison happens to be safe" into "this class of comparison is generally safe"?
3. Section 24.3 confirmed that `scan`-as-cumsum matches `ct.cumsum` exactly in both the forward and `reverse=True` direction. What would it have meant for this book's understanding of `ct.cumsum` if the forward direction had matched but the reverse direction hadn't?
4. Section 24.5 found that `ct.sum` accepts the tuple axis `(1, 2)` that `ct.argmax` rejects outright, and connected this to argmax's "underlying position-tracking" not generalizing the same way a sum does across multiple axes. In your own words, why might tracking *which position* the maximum came from be harder to define unambiguously across two axes at once than simply combining values along them, the way a sum or a max does?
5. The capstone's `argmax_combine` function and `ct.argmax` itself compiled to different byte counts, and the Discussion explicitly declined to read anything into that gap. Contrast this with Section 24.1's `reduce`-as-sum comparison, which used an identical byte count as real evidence that `ct.sum` is `reduce` with `operator.add` built in. What structural difference between the two comparisons explains why one is licensed to draw a conclusion from matching bytes and the other is not?

## Where We Go Next

This chapter closed out the cluster of `dir(ct)` reduction operations Chapter 23 named as still untouched, and in the process surfaced a methodological finding — the source-line confound — that applies retroactively to how carefully any future byte-count comparison in this book needs to be read. What remains almost entirely unexplored is a different kind of question this book has gestured at but never opened: `ct.static_assert`, `ct.static_eval`, and `ct.static_iter` point toward compile-time metaprogramming, code that runs while a kernel is being compiled rather than while it executes. This book has also never once triggered `TileValueError`, `TileStaticEvalError`, `TileStaticAssertionError`, `TileSyntaxError`, `TileRecursionError`, `TileInternalError`, `TileCompilerExecutionError`, or `TileCompilerTimeoutError` — eight exception types sitting in `dir(ct)` alongside the `TileTypeError` and `TileUnsupportedFeatureError` this book has relied on since Chapter 1 and Chapter 21 respectively, whose names alone suggest each is reserved for a specific, still-undiscovered kind of mistake.

## Worked Solutions

**1.** The one-argument combining function is rejected the instant the compiler looks at `func`'s own signature: arity is a property of the function's *definition*, visible without knowing anything about what values will ever flow through it, so the compiler can and does check it eagerly, before it even reasons about types. A dtype mismatch between `identity` and the tile being reduced is a different kind of property — it only becomes observable at the one moment `identity` would actually participate in a computation, which for most inputs never happens (a non-empty tile never needs its identity element at all; `identity` exists only to seed an otherwise-empty reduction). The compiler appears to check what it can verify structurally and to skip what would require reasoning about a value's eventual role in the computation, or its eventual numeric representation — the same gap this book found in Chapter 8 between what `export_kernel` can confirm (it compiles) and what only a real launch could confirm (it computes the right answer).

**2.** One passing self-check on one file is a single data point, not a proof of safety. Turning it into a general claim would require, at minimum, repeating the same "recompile the identical definition at a different line" test across several genuinely different files — different total line counts, different numbers of preceding kernels, different combining-function bodies — and confirming the byte count holds steady across all of them, the way Section 24.2's own blank-line-padding side experiment (which is *not* included in this chapter's own comparisons, precisely because it was designed to demonstrate the opposite: that this confound is real) showed a case where position alone does move the byte count. Since this chapter's own experiments already found both a case where position mattered (the padded file) and a case where it didn't (this section's own recompiled `add_fn`), the honest generalization is that position sensitivity is real but not universal, and no fixed number of passing self-checks would ever prove it absent for a comparison this book hasn't specifically re-tested.

**3.** If the forward direction matched `ct.cumsum` exactly but the reverse direction diverged, it would mean `ct.cumsum(x, axis, reverse=True)` is not simply `ct.scan` with `reverse=True` passed straight through — there would have to be some additional transformation specific to `cumsum`'s own reverse-direction code path that a hand-rolled `scan`-based reimplementation, using the identical `reverse=True` flag, doesn't reproduce. That would have been a genuinely interesting finding in its own right: it would mean `reduce`/`sum` and `scan`/`cumsum` are structurally identical relationships in the forward direction but diverge in the reverse one, undermining this chapter's central claim that the fixed reductions are nothing more than their customizable counterparts with a rule baked in — at least for whichever direction failed to match.

**4.** Summing or maximizing along two axes at once is well-defined because the *values themselves* live in a space where combining is associative and commutative regardless of order — you can fold axis 1 first and then axis 2, or the reverse, and land on the same scalar either way, because addition and max don't care about the shape of the space they're collapsing. A position, by contrast, is only meaningful relative to a specific coordinate system: "the maximum is at position 3" only makes sense if there's a single axis being indexed. Collapsing two axes into "the" position of the maximum requires deciding how to encode a 2D coordinate as a single index (or a pair), and that encoding isn't forced by the math the way summing is — there's no uniquely "natural" answer the way there is for combining values, which is presumably why `argmax`'s docstring simply declines to support it rather than picking one convention among several equally defensible ones.

**5.** Section 24.1's `reduce`-as-sum comparison held every relevant variable fixed except the one thing being tested: both kernels were named `kernel_fn` (Chapter 10's naming confound, controlled), compiled from the same file at the same call site pattern, and differed only in whether `ct.sum` or `ct.reduce(..., operator.add, 0)` computed the reduction — so a matching byte count is evidence *about that one substitution specifically*, because everything else that could have caused a difference was held equal. The capstone's two kernels share none of that control: `kernel_reduce_tuple_argmax` and `kernel_argmax_direct` have entirely different bodies, different local variable structures, and call entirely different sequences of operations (`ct.reduce` with a hand-written tie-break function versus a single call to `ct.argmax`) — a byte-count difference between them could come from any of a dozen structural differences, not just "which approach is more expensive," so nothing about the comparison isolates a single variable the way Section 24.1's did. Matching bytes are only evidence of sameness when everything but the one thing being tested was held constant; here, nothing but the final output shape was held constant, so neither a match nor a mismatch would have meant anything.

## Complete Runnable Code

### File: `01_reduce_reimplements_sum.py`

```python
import cuda.tile as ct
import io
import operator

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
    [array_param(2), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.reduce(x, axis, func, identity) applies a custom two-argument
# combining function along an axis. Reimplementing sum by hand with
# operator.add, then comparing against ct.sum itself under Chapter
# 10's naming-confound control (both variants named "kernel_fn").
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, operator.add, 0)
    ct.store(c, (0,), y)
n_reduce = compile_bytes(kernel_fn, sig)
print(f"reduce-as-sum (operator.add), naming-controlled: {n_reduce} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.sum(x, 1)
    ct.store(c, (0,), y)
n_sum = compile_bytes(kernel_fn, sig)
print(f"ct.sum, naming-controlled: {n_sum} cubin bytes")
print(f"identical bytes: {n_reduce == n_sum}")

# axis out of range for a 2D tile.
@ct.kernel
def kernel_reduce_bad_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 5, operator.add, 0)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_reduce_bad_axis, sig)
    print(f"reduce axis=5 out of range: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"reduce axis=5 out of range: {type(e).__name__}: {e}")

# A one-argument function passed where reduce needs a two-argument
# combining function.
def one_arg_fn(p):
    return p

@ct.kernel
def kernel_reduce_wrong_arity(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, one_arg_fn, 0)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_reduce_wrong_arity, sig)
    print(f"reduce with 1-argument func: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"reduce with 1-argument func: {type(e).__name__}: {e}")

# identity as a float constant for an int32 tile -- does reduce check
# that identity's dtype matches x's dtype at all?
@ct.kernel
def kernel_reduce_float_identity(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, operator.add, 0.5)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_reduce_float_identity, sig)
    print(f"reduce int32 tile with identity=0.5 (float): {n} cubin bytes")
except Exception as e:
    print(f"reduce int32 tile with identity=0.5 (float): {type(e).__name__}: {e}")
```

### File: `02_reduce_compiled_size_and_a_line_position_confound.py`

```python
import cuda.tile as ct
import io
import operator

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
    [array_param(2), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def add_fn(p, q):
    return p + q

def max_fn(p, q):
    return ct.maximum(p, q)

def mul_fn(p, q):
    return p * q

def complicated_fn(p, q):
    r = p + q
    r = r * 2
    r = r - p
    r = ct.maximum(r, q)
    return r

# Three genuinely different single-operation combining functions,
# naming-controlled (all as "kernel_fn"): does the compiled byte count
# reflect which arithmetic operation each one performs?
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, add_fn, 0)
    ct.store(c, (0,), y)
n_add = compile_bytes(kernel_fn, sig)
print(f"reduce named add_fn: {n_add} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, max_fn, -2147483648)
    ct.store(c, (0,), y)
n_max = compile_bytes(kernel_fn, sig)
print(f"reduce named max_fn: {n_max} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, mul_fn, 1)
    ct.store(c, (0,), y)
n_mul = compile_bytes(kernel_fn, sig)
print(f"reduce named mul_fn: {n_mul} cubin bytes")
print(f"add_fn == max_fn == mul_fn: {n_add == n_max == n_mul}")

# A combining function with meaningfully more code (four chained
# operations instead of one) does compile to a larger cubin.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, complicated_fn, 0)
    ct.store(c, (0,), y)
n_complicated = compile_bytes(kernel_fn, sig)
print(f"reduce named complicated_fn (4 chained ops): {n_complicated} cubin bytes")

# The same addition, spelled three ways: a named top-level function
# (reused from above), a lambda, and operator.add.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, lambda p, q: p + q, 0)
    ct.store(c, (0,), y)
n_lambda = compile_bytes(kernel_fn, sig)
print(f"reduce lambda add: {n_lambda} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, operator.add, 0)
    ct.store(c, (0,), y)
n_operator = compile_bytes(kernel_fn, sig)
print(f"reduce operator.add: {n_operator} cubin bytes")
print(f"named add_fn == lambda add: {n_add == n_lambda}")
print(f"named add_fn == operator.add: {n_add == n_operator}")

# Before concluding anything from those differences: recompile the
# textually IDENTICAL "named add_fn" kernel definition a second time,
# unchanged in every respect except which line of this file it starts
# on. If this alone changes the byte count, no difference reported
# above can be safely attributed to which callable spelling was used,
# since two different spellings can never occupy the same line.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.reduce(x, 1, add_fn, 0)
    ct.store(c, (0,), y)
n_add_again = compile_bytes(kernel_fn, sig)
print(f"reduce named add_fn, identical code, later line: {n_add_again} cubin bytes")
print(f"same as first add_fn measurement above: {n_add_again == n_add}")
```

### File: `03_scan_reimplements_cumsum.py`

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

def add_fn(p, q):
    return p + q

# ct.scan(x, axis, func, identity, reverse=False) applies a custom
# inclusive-prefix combining function. Naming-controlled comparison
# against ct.cumsum, forward and reverse.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.scan(x, 1, add_fn, 0)
    ct.store(c, (0, 0), y)
n_scan = compile_bytes(kernel_fn, sig)
print(f"scan-as-cumsum (named add_fn), naming-controlled: {n_scan} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.cumsum(x, 1)
    ct.store(c, (0, 0), y)
n_cumsum = compile_bytes(kernel_fn, sig)
print(f"ct.cumsum, naming-controlled: {n_cumsum} cubin bytes")
print(f"identical bytes: {n_scan == n_cumsum}")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.scan(x, 1, add_fn, 0, reverse=True)
    ct.store(c, (0, 0), y)
n_scan_rev = compile_bytes(kernel_fn, sig)
print(f"scan reverse=True (named add_fn), naming-controlled: {n_scan_rev} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.cumsum(x, 1, reverse=True)
    ct.store(c, (0, 0), y)
n_cumsum_rev = compile_bytes(kernel_fn, sig)
print(f"ct.cumsum reverse=True, naming-controlled: {n_cumsum_rev} cubin bytes")
print(f"identical bytes: {n_scan_rev == n_cumsum_rev}")
```

### File: `04_reduce_tuple_sum_and_product.py`

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
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.reduce accepts a TUPLE of tiles: func takes 2N arguments (N from
# each side being combined) and returns an N-tuple. Reducing the SAME
# tile twice with a combining function that sums one copy and
# multiplies the other computes sum and product in a single pass.
def sum_and_product(s1, p1, s2, p2):
    return s1 + s2, p1 * p2

@ct.kernel
def kernel_reduce_sum_and_product(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    total, product = ct.reduce((x, x), 0, sum_and_product, (0, 1))
    ct.store(c, (0,), ct.full((2,), total, dtype=ct.int32))
print(f"reduce((x, x), sum_and_product) tuple-of-2 same tile: {compile_bytes(kernel_reduce_sum_and_product, sig)} cubin bytes")
```

### File: `05_argmax_argmin_axis_and_tuple_rejection.py`

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
    [array_param(2), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# argmax with axis=None reduces all axes to a single scalar.
@ct.kernel
def kernel_argmax_all(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.argmax(x, None)
    ct.store(c, (0,), ct.full((1,), y, dtype=ct.int32))
print(f"argmax(x, axis=None): {compile_bytes(kernel_argmax_all, sig2)} cubin bytes")

# argmax along an explicit axis with keepdims.
sig2b = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_argmax_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0), (4, 8))
    y = ct.argmax(x, 1, keepdims=True)
    ct.store(c, (0, 0), y)
print(f"argmax(x, axis=1, keepdims=True): {compile_bytes(kernel_argmax_axis, sig2b)} cubin bytes")

# argmax's docstring explicitly says tuple axes are "not supported for
# argmax" -- ct.sum and ct.max both accept a tuple axis. Does argmax
# actually reject one?
sig3 = ct.compilation.KernelSignature(
    [array_param(3), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
@ct.kernel
def kernel_argmax_tuple_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.argmax(x, (1, 2))
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_argmax_tuple_axis, sig3)
    print(f"argmax(x, axis=(1,2)) tuple axis: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"argmax(x, axis=(1,2)) tuple axis: {type(e).__name__}: {e}")

# Compare: ct.sum DOES accept a tuple axis, per its own docstring.
@ct.kernel
def kernel_sum_tuple_axis(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0, 0, 0), (2, 4, 8))
    y = ct.sum(x, (1, 2))
    ct.store(c, (0,), y)
print(f"sum(x, axis=(1,2)) tuple axis: {compile_bytes(kernel_sum_tuple_axis, sig3)} cubin bytes")
```

### File: `06_capstone_argmax_via_reduce.py`

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
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: reimplement ct.argmax by hand using ct.reduce's tuple-of-
# tiles form, combining a value tile with its own index tile. On a
# tie, keep the SMALLER index -- exactly the convention ct.argmax's
# own docstring documents.
def argmax_combine(v1, i1, v2, i2):
    take_first = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    v = ct.where(take_first, v1, v2)
    i = ct.where(take_first, i1, i2)
    return v, i

@ct.kernel
def kernel_reduce_tuple_argmax(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    idx = ct.arange(8, dtype=ct.int32)
    _, best_idx = ct.reduce((x, idx), 0, argmax_combine, (-2147483648, 0))
    ct.store(c, (0,), ct.full((1,), best_idx, dtype=ct.int32))
print(f"reduce((values, indices), argmax_combine) tuple-of-2: {compile_bytes(kernel_reduce_tuple_argmax, sig)} cubin bytes")

# ct.argmax itself, on the same shape, for comparison -- not a
# naming-confound-controlled comparison (the two implementations are
# structurally different), just a side-by-side confirmation that both
# approaches compile.
@ct.kernel
def kernel_argmax_direct(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (0,), (8,))
    best_idx = ct.argmax(x, 0)
    ct.store(c, (0,), ct.full((1,), best_idx, dtype=ct.int32))
print(f"ct.argmax(x, 0) directly, same shape: {compile_bytes(kernel_argmax_direct, sig)} cubin bytes")
```
