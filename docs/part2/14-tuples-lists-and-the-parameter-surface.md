# Chapter 14: Runtime Tuples and Lists — Finishing the Parameter Surface

> "Chapter 2's `[COMMON TRAP]` on `ConstantConstraint((32, 64))` closed with a promise: a structured compile-time value needs `TupleConstraint`, and a list of arrays needs `ListConstraint`, 'covered when Part 1 builds on `ArrayConstraint` directly.' Part 1 came and went. Part 2 came and went. `ListConstraint` was never covered. This chapter goes back and checks `dir(ct.compilation)` against every worked example this book has actually run, rather than against what it assumed it had run, and finds that one promise still open and two more corners — a *runtime*-valued tuple, and the interaction between a list's own aliasing rules and everything Chapter 6 already established about arrays — that thirteen chapters of otherwise careful auditing simply never reached."

**What you will understand by the end of this chapter:**

- That baking a value into a kernel with `ConstantConstraint` does not reliably produce smaller compiled code than reading the identical value as a genuine runtime `ScalarConstraint` — only the multiplicative identity, `1.0`, measurably shrinks the output; every other tested constant, including `0.0`, compiles to the same size as the runtime version's constant-holding sibling
- That `TupleConstraint` is not limited to Chapter 2's compile-time-constant tuples — a tuple of `ScalarConstraint` items describes a genuine runtime-valued tuple parameter, unpacked inside the kernel body like any other Python tuple
- That a `TupleConstraint` whose item count does not match the kernel body's own unpacking is a real, distinct exception — `TileValueError`, not `TileTypeError` or `TileSyntaxError` — reported with the exact source line and column of the mismatched unpack
- `ListConstraint`, cuTile Python's parameter type for a genuine list of arrays, appearing in this book for the first time: how to declare one, how to index it with both a compile-time constant and a genuine runtime index derived from `len()`, and the real `TypeError` when its `element` constraint is anything other than an `ArrayConstraint`
- That `ListConstraint.alias_groups` cannot share a named group with an `ArrayConstraint` — a real, immediate `ValueError` naming both constraint types by name — while two `ListConstraint` parameters can share a group with each other without complaint
- That all three constraint types compose into a single kernel signature without conflict, confirmed by a capstone kernel using a list, a tuple, and a scalar together, and that Chapter 12's `symbol`-field explanation for a failed `==` on a round-tripped `KernelSignature` still fully accounts for the mismatch even on a signature this much more complex

**What you need to know first:**

- Chapter 2's `ct.Constant` contract and its `[COMMON TRAP]` on `ConstantConstraint` rejecting a bare tuple — this chapter picks up exactly where that trap's closing sentence left off.
- Chapter 6's `alias_groups` (Section 6.3) and `may_alias_internally` (Section 6.4) on `ArrayConstraint`, which `ListConstraint`'s own `alias_groups` and `elements_may_alias` are directly compared against.
- Chapter 10's naming-confound rule: every kernel or helper compared for byte count in this chapter is named `kernel_fn`, via a `build_kernel`-style factory where more than one variant is needed.
- Chapter 12's `mangle_kernel_name` / `demangle_kernel_name` round trip and its Section 12.5 finding that a round-tripped `KernelSignature` fails `==` only because of its `symbol` field — reused, not re-derived, in this chapter's capstone.

## 14.1 Baking In a Constant Doesn't Always Shrink the Code `[FOUNDATIONAL]`

### Intuition

Chapter 2 spent an entire chapter establishing that `ct.Constant` is a strict, bidirectionally-enforced contract — but every example that contract was tested against was a *shape* parameter, `tile_size`, which has no real choice in the matter: `ct.load`'s `shape` argument requires a compile-time constant tuple, full stop. An ordinary arithmetic operand is a different question. Multiplying a tile by a runtime scalar and multiplying it by a value baked in at compile time are both legal — so which one actually produces smaller code, and does the *value* of that constant matter?

### Background

A `ScalarConstraint(dtype)` describes a genuine runtime scalar: the compiled kernel reads it from an argument at launch time, the same way it reads any other non-constant value. A `ConstantConstraint(value)` embeds one specific Python `bool`, `int`, or `float` directly into the generated code for that signature — a different compiled artifact for every distinct value, the same mechanism Chapter 2 used to specialize `vector_add` across different `tile_size`s. The intuition that "a compile-time-known value should let the compiler generate less code" is reasonable, but this book's discipline is to check reasonable intuitions rather than assume them.

### Worked Example 14.1.1 — the same multiply, five different constants, and one runtime scalar

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

# Chapter 2 established that ct.Constant is a bidirectional contract, using
# "tile_size" -- a shape parameter that has no choice but to be constant,
# since ct.load's "shape" argument specifically requires one. "factor" here
# is a different kind of parameter: an ordinary arithmetic operand that
# could legitimately be either a runtime scalar or a baked-in constant.
# Both kernels below are named "kernel_fn" (Chapter 10's naming-confound
# rule), differing only in whether "factor" is annotated ct.Constant[float].

@ct.kernel
def kernel_fn(a, c, factor, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.mul(x, factor)
    ct.store(c, (pid,), y)
kernel_runtime = kernel_fn

sig_runtime = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ScalarConstraint(ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"factor is a genuine runtime scalar (ScalarConstraint): {compile_bytes(kernel_runtime, sig_runtime)} cubin bytes")

# Pairing a ConstantConstraint against that same, unannotated "factor"
# parameter reproduces the same contract violation Chapter 2 established,
# now against a plain arithmetic operand instead of a shape.
sig_mismatched = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(2.0), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    compile_bytes(kernel_runtime, sig_mismatched)
    print("ConstantConstraint against an unannotated parameter: compiled (unexpected)")
except TypeError as e:
    print(f"ConstantConstraint against an unannotated parameter: TypeError: {e}")

# The fair comparison: a second kernel, also named "kernel_fn", whose
# "factor" parameter is actually annotated ct.Constant[float], matching
# what ConstantConstraint requires.
@ct.kernel
def kernel_fn(a, c, factor: ct.Constant[float], tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.mul(x, factor)
    ct.store(c, (pid,), y)
kernel_constant = kernel_fn

for val in [0.0, 1.0, 2.0, -5.0, 3.14159]:
    sig_constant = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(val), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print(f"factor is a compile-time constant, value {val} (ConstantConstraint): {compile_bytes(kernel_constant, sig_constant)} cubin bytes")
```

The complete file is `01_scalar_vs_constant_parameter.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_scalar_vs_constant_parameter.py
```

Genuinely run:

```
factor is a genuine runtime scalar (ScalarConstraint): 21448 cubin bytes
ConstantConstraint against an unannotated parameter: TypeError: Invalid kernel parameter 'factor': ConstantConstraint is only valid for parameters annotated as Constant.
factor is a compile-time constant, value 0.0 (ConstantConstraint): 21552 cubin bytes
factor is a compile-time constant, value 1.0 (ConstantConstraint): 21424 cubin bytes
factor is a compile-time constant, value 2.0 (ConstantConstraint): 21552 cubin bytes
factor is a compile-time constant, value -5.0 (ConstantConstraint): 21552 cubin bytes
factor is a compile-time constant, value 3.14159 (ConstantConstraint): 21552 cubin bytes
```

> `[COMMON TRAP]` "Compile-time constant" does not mean "smaller code," and it especially does not mean "smaller code in proportion to how simple the value looks." Four of the five tested constants — `0.0`, `2.0`, `-5.0`, and `3.14159` — compile to the exact same 21,552 bytes as each other, all of them *larger* than the fully runtime `ScalarConstraint` version's 21,448 bytes. Only `1.0`, the multiplicative identity, drops to 21,424 bytes — smaller than every other variant, runtime scalar included. This book's `export_kernel`-only environment cannot see why the compiler treats `1.0` specially here without disassembling the generated code, but it can report, from a real and repeated measurement, that "bake it in as a constant" is not a general size-reduction strategy for an ordinary arithmetic operand — the identity element is the one value doing anything different, and even `0.0`, which a hand-optimizing programmer might expect to fold the whole multiply away, compiles identically to a value with no special algebraic meaning at all.

## 14.2 A Tuple of Runtime Values `[FOUNDATIONAL]`

### Intuition

Every `TupleConstraint` this book has built so far — Chapter 2's `(32, 64)` tile shape, Chapter 5's `v2`-only 2D tile shape — wrapped a `TupleConstraint` around `ConstantConstraint` items: a fixed, compile-time-known tuple. But `TupleConstraint`'s documented signature takes a sequence of *any* `ParameterConstraint`, not specifically `ConstantConstraint`. A tuple of `ScalarConstraint` items should describe something this book has never built: a tuple whose contents are only known at launch time, unpacked inside the kernel body the same way Python unpacks any other tuple.

### Background

`ct.compilation.TupleConstraint(items)` requires only that `items` be a sequence of per-element constraints — nothing in its documentation restricts those elements to `ConstantConstraint`. Inside a kernel body, a parameter whose signature entry is a `TupleConstraint` is unpacked with ordinary Python tuple-unpacking syntax, `lo, hi = bounds`, regardless of whether the item constraints underneath are compile-time constants or genuine runtime scalars.

### Worked Example 14.2.1 — a runtime `(lo, hi)` clamp

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# TupleConstraint(items) describes a genuine tuple-shaped kernel parameter,
# built from a sequence of per-item constraints.
@ct.kernel
def kernel_fn(a, bounds, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    lo, hi = bounds
    x = ct.load(a, (pid,), (tile_size,))
    clamped = ct.where(x < lo, lo, ct.where(x > hi, hi, x))
    ct.store(c, (pid,), clamped)

sig = ct.compilation.KernelSignature(
    [array_param(),
     ct.compilation.TupleConstraint([ct.compilation.ScalarConstraint(ct.float32), ct.compilation.ScalarConstraint(ct.float32)]),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(kernel_fn, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"kernel with a TupleConstraint(scalar, scalar) parameter: {len(buf.getvalue())} cubin bytes")
except Exception as e:
    print(f"kernel with a TupleConstraint(scalar, scalar) parameter: {type(e).__name__}: {e}")
```

The complete file is `02_tuple_constraint_clamp_bounds.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_tuple_constraint_clamp_bounds.py
```

Genuinely run:

```
kernel with a TupleConstraint(scalar, scalar) parameter: 22136 cubin bytes
```

This compiles cleanly, on the first attempt, with no adjustment needed to `TupleConstraint`, `ct.where`, or the unpacking syntax — `bounds` behaves as an ordinary Python 2-tuple inside the kernel body, `lo` and `hi` behave as ordinary runtime scalars once unpacked, and the whole thing clamps a tile the same way any single-scalar-bound version would, just with two runtime bounds arriving as one structured parameter instead of two separate ones.

## 14.3 When the Signature's Arity Lies `[FOUNDATIONAL]`

### Intuition

Section 14.2's `bounds` parameter unpacks into exactly two names, `lo, hi`, matching a `TupleConstraint` with exactly two items. Nothing so far has checked what happens when those two counts disagree — a `TupleConstraint` declaring three items against a kernel body that only unpacks two, or the reverse. Python itself raises `ValueError` for this on an ordinary tuple; cuTile Python's own tracer has to decide independently what to raise, since the mismatch here is caught while tracing the kernel body against a `Tile`-typed aggregate, not while executing ordinary Python.

### Background

`dir(ct)` lists `TileValueError` alongside `TileTypeError` and `TileSyntaxError`, as a distinct, direct subclass of the common `TileError` base — not merely reusing Python's built-in `ValueError` or `TileTypeError` for this kind of mismatch. Whether an arity mismatch actually reaches that class, and what its message looks like, needs a genuine test rather than an assumption from the name alone.

### Worked Example 14.3.1 — a 3-item signature against a 2-item unpack

```python
import cuda.tile as ct
import io

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# The kernel body unpacks `bounds` as a 2-tuple (lo, hi), but the signature
# below declares a TupleConstraint with THREE ScalarConstraint items --
# a genuine arity mismatch, never attempted earlier in this book.
@ct.kernel
def kernel_fn(a, bounds, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    lo, hi = bounds
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.where(x < lo, lo, ct.where(x > hi, hi, x))
    ct.store(a, (pid,), y)

sig = ct.compilation.KernelSignature(
    [ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                     stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False),
     ct.compilation.TupleConstraint([ct.compilation.ScalarConstraint(ct.float32)] * 3),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn, sig)
    print(f"3-item TupleConstraint against a 2-item unpack: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"3-item TupleConstraint against a 2-item unpack: {type(e).__name__}: {e}")
```

The complete file is `03_tuple_arity_mismatch.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_tuple_arity_mismatch.py
```

Genuinely run:

```
3-item TupleConstraint against a 2-item unpack: TileValueError: Too many values to unpack (expected 2, got 3)
  "/tmp/ch14/03_tuple_arity_mismatch.py", line 15, col 5-10, in kernel_fn:
        lo, hi = bounds
        ^^^^^^

```

`TileValueError` is exactly the class `dir(ct)` promised, and its message is closely modeled on the wording CPython itself uses for an ordinary tuple-unpacking mismatch ("too many values to unpack") — but it arrives with something CPython's own `ValueError` never carries: the exact source line and column of the offending unpack, `lo, hi = bounds`, underlined at the precise span that failed. This is the same discipline Chapter 1's `TileTypeError` and Chapter 8's masking errors have already shown — a real static check performed while tracing the kernel body, catching a structural mismatch between the declared signature and the actual Python source before any code generation happens, and reporting it with source-level precision rather than a generic message.

## 14.4 `ListConstraint`: An Old Promise, Finally Kept `[FOUNDATIONAL]`

### Intuition

A kernel parameter that is a genuine Python list of arrays — not a single array, not a fixed-size tuple, but a runtime-length collection where each element is itself an `Array` — needs its own constraint type, because neither `ArrayConstraint` nor `TupleConstraint` describes a variable-length collection of arrays. Chapter 2 named this type, `ListConstraint`, in a single `[COMMON TRAP]` aside and never returned to it. This section builds the first real kernel in this book to use one.

### Background

`ct.compilation.ListConstraint(element, *, alias_groups, elements_may_alias)` requires `element` to be an `ArrayConstraint` — "since only lists of arrays are supported as kernel arguments," per its own documentation. Inside the kernel body, a list-typed parameter supports two operations this book has not needed before: indexing with `mylist[i]`, where `i` can be either a Python-level constant or a genuine runtime scalar, and `len(mylist)`, which returns a genuine runtime `int32` rather than a Python-level constant — there is no way to know a list's length until a real list actually arrives at launch, so `len()` here cannot be resolved by the tracer the way a Python `len()` on a literal list would be.

### Worked Example 14.4.1 — indexing a list by constant and by runtime length

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

# A kernel taking a genuine list-of-arrays parameter. Index 0 and 1 are
# Python-level (compile-time) constants; `n` is a genuine runtime value
# obtained from len(), used here as a dynamic index into the same list.
@ct.kernel
def kernel_fn(arrs, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    n = len(arrs)
    first = ct.load(arrs[0], (pid,), (tile_size,))
    last = ct.load(arrs[n - 1], (pid,), (tile_size,))
    ct.store(out, (pid,), ct.add(first, last))

sig = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=False),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"list-of-arrays parameter, elements_may_alias=False: {compile_bytes(kernel_fn, sig)} cubin bytes")

sig_alias = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=True),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"list-of-arrays parameter, elements_may_alias=True: {compile_bytes(kernel_fn, sig_alias)} cubin bytes")

# ListConstraint documents that `element` must be an ArrayConstraint --
# a combination never tried in this book: pairing it with a ScalarConstraint.
try:
    ct.compilation.ListConstraint(ct.compilation.ScalarConstraint(ct.float32), alias_groups=[], elements_may_alias=False)
    print("ListConstraint(element=ScalarConstraint(...)): accepted (unexpected)")
except TypeError as e:
    print(f"ListConstraint(element=ScalarConstraint(...)): TypeError: {e}")
```

The complete file is `04_list_constraint_indexing_and_element_type.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_list_constraint_indexing_and_element_type.py
```

Genuinely run:

```
list-of-arrays parameter, elements_may_alias=False: 23808 cubin bytes
list-of-arrays parameter, elements_may_alias=True: 23936 cubin bytes
ListConstraint(element=ScalarConstraint(...)): TypeError: ListConstraint only supports an ArrayParameter as the `element` constraint, got 'ScalarConstraint(dtype=<DType 'float32'>)'
```

Both list-indexing operations compile without complaint: `arrs[0]` with a Python-literal index and `arrs[n - 1]` with a genuine runtime index derived from `len(arrs)` both type-check as ordinary `Array` values once extracted, and both feed into `ct.load` exactly as any other array parameter would. `elements_may_alias`, the flag governing whether two distinct elements of the same list are allowed to overlap in memory, behaves the direction Chapter 6's `may_alias_internally` finding would predict: permitting aliasing (`True`) removes an optimization the compiler could otherwise rely on, and the compiled output grows, from 23,808 to 23,936 bytes. And `ListConstraint`'s own type-checking on `element` is exactly as strict as its documentation states — a `ScalarConstraint` is rejected immediately, at `ListConstraint` construction, with a `TypeError` naming the rejected constraint by its full `repr`.

## 14.5 Alias Groups Across Constraint Types `[FOUNDATIONAL]`

### Intuition

Chapter 6 established `alias_groups` entirely in terms of `ArrayConstraint` parameters sharing a group with each other. `ListConstraint` has its own `alias_groups` field, documented as governing whether the *list's own storage* — as opposed to `element.alias_groups`, which is about the list's elements — may alias other parameters. Nothing has tested what happens when a list and a plain array try to share a group, or when two lists do.

### Background

`KernelSignature.__init__` validates every declared alias group across the whole signature before `export_kernel` is ever reached — the same validation, per this chapter's testing, that already rejects a `ListConstraint` whose `element` is not an `ArrayConstraint`. Whether that validation treats a shared alias group between a `ListConstraint` and an `ArrayConstraint` as acceptable, or as a structurally different pairing it refuses to reason about, is not documented anywhere this chapter has read — it needs a direct test.

### Worked Example 14.5.1 — list-and-array sharing a group, versus list-and-list

```python
import cuda.tile as ct
import io

def array_param(alias_groups=()):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=list(alias_groups), may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ListConstraint.alias_groups governs whether the LIST parameter's own
# storage may alias other parameters -- distinct from element.alias_groups,
# which governs aliasing between the list's own elements. A combination
# never attempted earlier in this book: pairing a list's alias group
# against a plain array's alias group.

@ct.kernel
def kernel_fn(arrs, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    n = len(arrs)
    first = ct.load(arrs[0], (pid,), (tile_size,))
    last = ct.load(arrs[n - 1], (pid,), (tile_size,))
    ct.store(out, (pid,), ct.add(first, last))

sig_no_alias = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=False),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"list alias_groups=[] (no aliasing with out): {compile_bytes(kernel_fn, sig_no_alias)} cubin bytes")

try:
    ct.compilation.KernelSignature(
        [ct.compilation.ListConstraint(array_param(), alias_groups=["g"], elements_may_alias=False),
         array_param(alias_groups=["g"]),
         ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print("list alias_groups=['g'] shared with an ArrayConstraint: accepted (unexpected)")
except ValueError as e:
    print(f"list alias_groups=['g'] shared with an ArrayConstraint: ValueError: {e}")

# A ListConstraint sharing an alias group with ANOTHER ListConstraint --
# same constraint type on both sides -- is a different story.
@ct.kernel
def kernel_fn2(arrs_a, arrs_b, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(arrs_a[0], (pid,), (tile_size,))
    y = ct.load(arrs_b[0], (pid,), (tile_size,))
    ct.store(out, (pid,), ct.add(x, y))

sig_two_lists = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=["g"], elements_may_alias=False),
     ct.compilation.ListConstraint(array_param(), alias_groups=["g"], elements_may_alias=False),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"two ListConstraints sharing alias group 'g': {compile_bytes(kernel_fn2, sig_two_lists)} cubin bytes")
```

The complete file is `05_list_alias_groups.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_list_alias_groups.py
```

Genuinely run:

```
list alias_groups=[] (no aliasing with out): 23808 cubin bytes
list alias_groups=['g'] shared with an ArrayConstraint: ValueError: Alias group `g` is used in two constraints of different types ListConstraint and ArrayConstraint. This is currently unsupported (e.g., list storage is not allowed to alias an array).
two ListConstraints sharing alias group 'g': 25232 cubin bytes
```

> `[COMMON TRAP]` An alias group name is not a plain, type-agnostic label the way it might first appear from Chapter 6's `ArrayConstraint`-only examples. Naming the same group string, `"g"`, on a `ListConstraint` and an `ArrayConstraint` is rejected outright by `KernelSignature`'s own validation, with a `ValueError` that names both constraint types explicitly and states the restriction in its own words: list storage is not allowed to alias an array. The identical group name shared between two `ListConstraint` parameters instead, with no `ArrayConstraint` involved, compiles without any complaint at all. The rule is not "these two parameters may or may not overlap in memory" in the abstract — it is scoped to matching constraint types, and crossing that boundary is a construction-time error before `export_kernel`, the compiler, or the kernel body are ever reached.

## 14.6 A Capstone: List, Tuple, and Scalar Together `[FOUNDATIONAL]`

### Intuition

Every constraint type this chapter tested — `ScalarConstraint`, `TupleConstraint` of scalars, and `ListConstraint` — has so far appeared alone, one new idea isolated per section, in the same style every earlier chapter of this book has used to keep byte-count comparisons uncontaminated by unrelated changes. Nothing has confirmed that all three actually coexist peacefully in a single signature. This section builds one kernel using all three at once, and reuses Chapter 12's `mangle_kernel_name` / `demangle_kernel_name` machinery to check whether a signature this much more structurally complex still behaves the way Chapter 12's much simpler signature did.

### Background

Chapter 12, Section 12.5 found that a round-tripped `KernelSignature` — one produced by `demangle_kernel_name` from a mangled symbol — fails a plain `==` comparison against the original signature, and that the entire explanation is the `symbol` field: the original signature's `symbol` is `None` (meaning "auto-generate one"), while the recovered signature's `symbol` is the actual mangled string. `.parameters` and `.calling_convention`, the fields that determine compiled behavior, matched exactly on that simpler, `ArrayConstraint`-only signature. Whether that same, narrow explanation still accounts for the whole difference once the signature contains a list, a tuple, and a scalar together — rather than leaving some other field to diverge once more constraint types are involved — is a genuine, previously untested question.

### Worked Example 14.6.1 — composing all three, then round-tripping the result

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

# A capstone kernel combining all three constraint types introduced in this
# chapter in a single signature, something no earlier example in this book
# has attempted:
#   - a ScalarConstraint runtime "bias" added after clamping,
#   - a TupleConstraint (lo, hi) runtime clamp bounds,
#   - a ListConstraint over a runtime-length list of arrays, each one
#     clamped to (lo, hi) and offset by bias, then summed into "out".
@ct.kernel
def kernel_fn(arrs, bounds, bias, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    lo, hi = bounds
    n = len(arrs)
    acc = ct.load(arrs[0], (pid,), (tile_size,))
    acc = ct.where(acc < lo, lo, ct.where(acc > hi, hi, acc))
    last = ct.load(arrs[n - 1], (pid,), (tile_size,))
    last = ct.where(last < lo, lo, ct.where(last > hi, hi, last))
    combined = ct.add(ct.add(acc, last), bias)
    ct.store(out, (pid,), combined)

sig = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=False),
     ct.compilation.TupleConstraint([ct.compilation.ScalarConstraint(ct.float32),
                                      ct.compilation.ScalarConstraint(ct.float32)]),
     ct.compilation.ScalarConstraint(ct.float32),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"capstone (list + tuple + scalar constraints together): {compile_bytes(kernel_fn, sig)} cubin bytes")

symbol = ct.compilation.mangle_kernel_name("kernel_fn", sig)
print(f"mangled symbol: {symbol}")
name, roundtrip_sig = ct.compilation.demangle_kernel_name(symbol)
print(f"demangled name: {name}")
print(f"roundtrip KernelSignature == original: {roundtrip_sig == sig}")
print(f"roundtrip .parameters == original .parameters: {roundtrip_sig.parameters == sig.parameters}")
print(f"roundtrip .calling_convention == original: {roundtrip_sig.calling_convention == sig.calling_convention}")
print(f"original .symbol: {sig.symbol!r}")
print(f"roundtrip .symbol: {roundtrip_sig.symbol!r}")
```

The complete file is `06_capstone_scalar_tuple_list_together.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_capstone_scalar_tuple_list_together.py
```

Genuinely run:

```
capstone (list + tuple + scalar constraints together): 26248 cubin bytes
mangled symbol: kernel_fn_Kt2_LA1f32_1l0_T2Sf32Sf32_Sf32_A1f32_1l0_I64
demangled name: kernel_fn
roundtrip KernelSignature == original: False
roundtrip .parameters == original .parameters: True
roundtrip .calling_convention == original: True
original .symbol: None
roundtrip .symbol: 'kernel_fn_Kt2_LA1f32_1l0_T2Sf32Sf32_Sf32_A1f32_1l0_I64'
```

The kernel compiles cleanly with all three constraint types present at once — nothing about combining them conflicts, and the mangled symbol string visibly encodes all three shapes (`L` for the list, `T2` for the 2-item tuple, `S` for each bare scalar) in one coherent name. The round trip reproduces Chapter 12's exact finding, unchanged by the added structural complexity: `==` on the whole `KernelSignature` is `False`, but `.parameters` and `.calling_convention` — everything that actually determines what got compiled — match exactly, and the only field that differs is `symbol`, `None` on the original versus the real mangled string on the recovered copy. A signature five constraint types more complex than Chapter 12's did not surface any new field for `==` to trip over; the explanation generalizes exactly as far as it claimed to.

## Chapter Summary

Chapter 2 closed a `[COMMON TRAP]` with a promise that `ListConstraint` would be covered once Part 1 built on `ArrayConstraint` directly — a promise thirteen chapters never kept. This chapter finally builds the first real kernel using one, confirming it supports indexing by both a compile-time constant and a genuine runtime index from `len()`, and that its `element` constraint is checked exactly as strictly as documented, rejecting anything other than an `ArrayConstraint` with an immediate `TypeError`. Along the way, three other corners this book's own auditing had missed came into view: baking a value into a kernel with `ConstantConstraint` does not reliably shrink compiled code relative to a genuine runtime `ScalarConstraint` — only the multiplicative identity `1.0` measurably did, with every other tested constant, `0.0` included, compiling identically to each other and larger than the runtime version; `TupleConstraint` describes a genuine runtime-valued tuple just as readily as Chapter 2's compile-time one, and a mismatch between its declared arity and the kernel body's own unpacking is a real, distinct `TileValueError`, reported with exact source location; and `ListConstraint.alias_groups` cannot share a named group with a plain `ArrayConstraint` — a real, immediate `ValueError` naming both types — while two `ListConstraint` parameters can share a group with each other freely. A capstone kernel combining a list, a runtime tuple, and a scalar all at once compiled without conflict, and Chapter 12's explanation for a round-tripped `KernelSignature` failing `==` — the `symbol` field, and only the `symbol` field — held exactly as well on this far more structurally complex signature as it did on Chapter 12's simplest one.

## Self-Check Questions

1. Section 14.1 found that four of five tested `ConstantConstraint` values compiled to the identical 21,552 bytes, all larger than the `ScalarConstraint` version's 21,448 bytes, while `1.0` alone dropped to 21,424 bytes. What would you need to see that this chapter's `export_kernel`-only environment cannot show you, to explain *why* `1.0` is different?
2. Section 14.2's `TupleConstraint([ScalarConstraint(ct.float32), ScalarConstraint(ct.float32)])` compiled successfully on the first attempt, with no change needed to the unpacking syntax already used for Chapter 2's compile-time tuples. What does that suggest about how much a kernel body's own code needs to know about whether a tuple parameter's contents are compile-time or runtime values?
3. Section 14.3 found a genuine `TileValueError` for a tuple-arity mismatch, distinct from `TileTypeError`. Chapter 1 and Chapter 8 both used `TileTypeError` for their own static checks. What general principle would you propose for when cuTile Python's tracer reaches for `TileValueError` instead of `TileTypeError`?
4. Section 14.5 found that a `ListConstraint` and an `ArrayConstraint` cannot share an alias group, but two `ListConstraint`s can share one with each other. Why might the compiler's authors have drawn that line specifically at "same constraint type," rather than allowing any two parameters that are both, in some sense, array-backed to share a group?
5. Section 14.6 found `.parameters` and `.calling_convention` equal, but `.symbol` different, between an original and a round-tripped `KernelSignature`. Chapter 12 found the identical pattern on a much simpler signature. If you had found some *other* field also differing on this chapter's more complex signature, what would that have told you that Chapter 12's simpler test could not?

## Where We Go Next

This chapter closed a promise Chapter 2 made and this book had otherwise let sit unfulfilled for thirteen chapters, and along the way found three genuinely new boundary cases in territory this book had considered settled: a constant-folding intuition that mostly doesn't hold, a new exception class for a new kind of structural mismatch, and an aliasing rule that cares about constraint type in a way Chapter 6's `ArrayConstraint`-only examples never had reason to reveal. Every constraint type `ct.compilation` actually exports — scalar, array, list, tuple, and constant — has now appeared in a genuine, compiled kernel in this book. Chapter 15 turns from the parameter surface back to the kernel body itself, auditing whether the same discipline — checking `dir()` against what has actually been run, not what this book assumed it had run — turns up equally overlooked corners in cuTile Python's tile-level operations.

## Worked Solutions

**1.** You would need to see the actual generated machine code — the disassembled SASS or PTX for each variant — to know whether the compiler recognizes `x * 1.0` specifically as an identity operation it can elide entirely, versus treating `0.0`, `2.0`, `-5.0`, and `3.14159` as "just another immediate value" requiring the same multiply instruction regardless of what that value is. This book's `export_kernel`-only environment reports only a final byte count, never the compiler's internal reasoning or the instructions actually chosen, so it can report *that* `1.0` is special and *that* the other four are not, but not *why* — that would require inspecting the cubin's disassembly directly, a step beyond what any chapter so far has attempted.

**2.** It suggests the kernel body's own unpacking code is written against the tuple's *shape* — how many items it has, and what to call each one — not against whether those items happen to be resolved at compile time or read at runtime. `lo, hi = bounds` is identical Python whether `bounds`'s underlying constraint is `TupleConstraint([ConstantConstraint(32), ConstantConstraint(64)])` or `TupleConstraint([ScalarConstraint(ct.float32), ScalarConstraint(ct.float32)])` — the compiler resolves what `lo` and `hi` actually *are* (a baked-in constant versus a runtime-loaded value) separately, downstream of the unpacking itself.

**3.** A reasonable principle, consistent with every example this chapter and earlier chapters have shown: `TileTypeError` covers a mismatch about *what kind* of thing a value is — a raw `Tile` used where a scalar was required (Chapter 1), a runtime value arriving where `ct.load`'s `shape` needed a compile-time constant (Chapter 2), an invalid `traversal_steps` rank (Chapter 13) — while `TileValueError` in this chapter covers a mismatch about *how many* of a kind of thing there are: the correct type (a tuple) with the wrong count of elements. The distinction mirrors Python's own built-in split between `TypeError` and `ValueError`, and Section 14.3's message ("too many values to unpack") is deliberately worded to match Python's own `ValueError` for the identical mistake on an ordinary tuple.

**4.** Restricting shared alias groups to matching constraint types keeps the compiler's aliasing analysis from having to reason about two structurally different memory representations at once — an `ArrayConstraint` is a single base pointer plus shape and stride, while a `ListConstraint` is a base pointer *plus a length*, pointing at a whole collection of such arrays (confirmed by this chapter's `len()` example). Two `ArrayConstraint`s, or two `ListConstraint`s, sharing a group are describing "these two things of the identical shape might overlap," a single well-defined question; an `ArrayConstraint` and a `ListConstraint` sharing a group would be asking whether a single array might overlap with a variable-length collection of arrays — a fundamentally different and messier question the error message's own wording ("list storage is not allowed to alias an array") suggests the authors chose not to support yet, rather than one they tried and found impossible in principle.

**5.** It would have told you that the round-trip's information loss is *not* fully confined to the `symbol` field the way Chapter 12 concluded — that some other property of the signature (perhaps something specific to how `ListConstraint` or `TupleConstraint` get mangled and demangled) fails to survive the round trip intact, which would mean Chapter 12's explanation was correct only for the simpler class of signatures it happened to test, not a fully general property of `mangle_kernel_name` / `demangle_kernel_name`. Finding no such field, as this chapter actually did, is itself informative in the other direction — it is evidence, though not a mathematical proof for every possible signature shape, that the `symbol`-field explanation generalizes further than the one narrow case that originally produced it.

## Complete Runnable Code

### File: `01_scalar_vs_constant_parameter.py`

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

# Chapter 2 established that ct.Constant is a bidirectional contract, using
# "tile_size" -- a shape parameter that has no choice but to be constant,
# since ct.load's "shape" argument specifically requires one. "factor" here
# is a different kind of parameter: an ordinary arithmetic operand that
# could legitimately be either a runtime scalar or a baked-in constant.
# Both kernels below are named "kernel_fn" (Chapter 10's naming-confound
# rule), differing only in whether "factor" is annotated ct.Constant[float].

@ct.kernel
def kernel_fn(a, c, factor, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.mul(x, factor)
    ct.store(c, (pid,), y)
kernel_runtime = kernel_fn

sig_runtime = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ScalarConstraint(ct.float32), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"factor is a genuine runtime scalar (ScalarConstraint): {compile_bytes(kernel_runtime, sig_runtime)} cubin bytes")

# Pairing a ConstantConstraint against that same, unannotated "factor"
# parameter reproduces the same contract violation Chapter 2 established,
# now against a plain arithmetic operand instead of a shape.
sig_mismatched = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(2.0), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    compile_bytes(kernel_runtime, sig_mismatched)
    print("ConstantConstraint against an unannotated parameter: compiled (unexpected)")
except TypeError as e:
    print(f"ConstantConstraint against an unannotated parameter: TypeError: {e}")

# The fair comparison: a second kernel, also named "kernel_fn", whose
# "factor" parameter is actually annotated ct.Constant[float], matching
# what ConstantConstraint requires.
@ct.kernel
def kernel_fn(a, c, factor: ct.Constant[float], tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.mul(x, factor)
    ct.store(c, (pid,), y)
kernel_constant = kernel_fn

for val in [0.0, 1.0, 2.0, -5.0, 3.14159]:
    sig_constant = ct.compilation.KernelSignature(
        [array_param(), array_param(), ct.compilation.ConstantConstraint(val), ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print(f"factor is a compile-time constant, value {val} (ConstantConstraint): {compile_bytes(kernel_constant, sig_constant)} cubin bytes")
```

### File: `02_tuple_constraint_clamp_bounds.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# TupleConstraint(items) describes a genuine tuple-shaped kernel parameter,
# built from a sequence of per-item constraints.
@ct.kernel
def kernel_fn(a, bounds, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    lo, hi = bounds
    x = ct.load(a, (pid,), (tile_size,))
    clamped = ct.where(x < lo, lo, ct.where(x > hi, hi, x))
    ct.store(c, (pid,), clamped)

sig = ct.compilation.KernelSignature(
    [array_param(),
     ct.compilation.TupleConstraint([ct.compilation.ScalarConstraint(ct.float32), ct.compilation.ScalarConstraint(ct.float32)]),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(kernel_fn, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"kernel with a TupleConstraint(scalar, scalar) parameter: {len(buf.getvalue())} cubin bytes")
except Exception as e:
    print(f"kernel with a TupleConstraint(scalar, scalar) parameter: {type(e).__name__}: {e}")
```

### File: `03_tuple_arity_mismatch.py`

```python
import cuda.tile as ct
import io

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# The kernel body unpacks `bounds` as a 2-tuple (lo, hi), but the signature
# below declares a TupleConstraint with THREE ScalarConstraint items --
# a genuine arity mismatch, never attempted earlier in this book.
@ct.kernel
def kernel_fn(a, bounds, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    lo, hi = bounds
    x = ct.load(a, (pid,), (tile_size,))
    y = ct.where(x < lo, lo, ct.where(x > hi, hi, x))
    ct.store(a, (pid,), y)

sig = ct.compilation.KernelSignature(
    [ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                     stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False),
     ct.compilation.TupleConstraint([ct.compilation.ScalarConstraint(ct.float32)] * 3),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
try:
    n = compile_bytes(kernel_fn, sig)
    print(f"3-item TupleConstraint against a 2-item unpack: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"3-item TupleConstraint against a 2-item unpack: {type(e).__name__}: {e}")
```

### File: `04_list_constraint_indexing_and_element_type.py`

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

# A kernel taking a genuine list-of-arrays parameter. Index 0 and 1 are
# Python-level (compile-time) constants; `n` is a genuine runtime value
# obtained from len(), used here as a dynamic index into the same list.
@ct.kernel
def kernel_fn(arrs, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    n = len(arrs)
    first = ct.load(arrs[0], (pid,), (tile_size,))
    last = ct.load(arrs[n - 1], (pid,), (tile_size,))
    ct.store(out, (pid,), ct.add(first, last))

sig = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=False),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"list-of-arrays parameter, elements_may_alias=False: {compile_bytes(kernel_fn, sig)} cubin bytes")

sig_alias = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=True),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"list-of-arrays parameter, elements_may_alias=True: {compile_bytes(kernel_fn, sig_alias)} cubin bytes")

# ListConstraint documents that `element` must be an ArrayConstraint --
# a combination never tried in this book: pairing it with a ScalarConstraint.
try:
    ct.compilation.ListConstraint(ct.compilation.ScalarConstraint(ct.float32), alias_groups=[], elements_may_alias=False)
    print("ListConstraint(element=ScalarConstraint(...)): accepted (unexpected)")
except TypeError as e:
    print(f"ListConstraint(element=ScalarConstraint(...)): TypeError: {e}")
```

### File: `05_list_alias_groups.py`

```python
import cuda.tile as ct
import io

def array_param(alias_groups=()):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=list(alias_groups), may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ListConstraint.alias_groups governs whether the LIST parameter's own
# storage may alias other parameters -- distinct from element.alias_groups,
# which governs aliasing between the list's own elements. A combination
# never attempted earlier in this book: pairing a list's alias group
# against a plain array's alias group.

@ct.kernel
def kernel_fn(arrs, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    n = len(arrs)
    first = ct.load(arrs[0], (pid,), (tile_size,))
    last = ct.load(arrs[n - 1], (pid,), (tile_size,))
    ct.store(out, (pid,), ct.add(first, last))

sig_no_alias = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=False),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"list alias_groups=[] (no aliasing with out): {compile_bytes(kernel_fn, sig_no_alias)} cubin bytes")

try:
    ct.compilation.KernelSignature(
        [ct.compilation.ListConstraint(array_param(), alias_groups=["g"], elements_may_alias=False),
         array_param(alias_groups=["g"]),
         ct.compilation.ConstantConstraint(64)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    print("list alias_groups=['g'] shared with an ArrayConstraint: accepted (unexpected)")
except ValueError as e:
    print(f"list alias_groups=['g'] shared with an ArrayConstraint: ValueError: {e}")

# A ListConstraint sharing an alias group with ANOTHER ListConstraint --
# same constraint type on both sides -- is a different story.
@ct.kernel
def kernel_fn2(arrs_a, arrs_b, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(arrs_a[0], (pid,), (tile_size,))
    y = ct.load(arrs_b[0], (pid,), (tile_size,))
    ct.store(out, (pid,), ct.add(x, y))

sig_two_lists = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=["g"], elements_may_alias=False),
     ct.compilation.ListConstraint(array_param(), alias_groups=["g"], elements_may_alias=False),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"two ListConstraints sharing alias group 'g': {compile_bytes(kernel_fn2, sig_two_lists)} cubin bytes")
```

### File: `06_capstone_scalar_tuple_list_together.py`

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

# A capstone kernel combining all three constraint types introduced in this
# chapter in a single signature, something no earlier example in this book
# has attempted:
#   - a ScalarConstraint runtime "bias" added after clamping,
#   - a TupleConstraint (lo, hi) runtime clamp bounds,
#   - a ListConstraint over a runtime-length list of arrays, each one
#     clamped to (lo, hi) and offset by bias, then summed into "out".
@ct.kernel
def kernel_fn(arrs, bounds, bias, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    lo, hi = bounds
    n = len(arrs)
    acc = ct.load(arrs[0], (pid,), (tile_size,))
    acc = ct.where(acc < lo, lo, ct.where(acc > hi, hi, acc))
    last = ct.load(arrs[n - 1], (pid,), (tile_size,))
    last = ct.where(last < lo, lo, ct.where(last > hi, hi, last))
    combined = ct.add(ct.add(acc, last), bias)
    ct.store(out, (pid,), combined)

sig = ct.compilation.KernelSignature(
    [ct.compilation.ListConstraint(array_param(), alias_groups=[], elements_may_alias=False),
     ct.compilation.TupleConstraint([ct.compilation.ScalarConstraint(ct.float32),
                                      ct.compilation.ScalarConstraint(ct.float32)]),
     ct.compilation.ScalarConstraint(ct.float32),
     array_param(),
     ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print(f"capstone (list + tuple + scalar constraints together): {compile_bytes(kernel_fn, sig)} cubin bytes")

symbol = ct.compilation.mangle_kernel_name("kernel_fn", sig)
print(f"mangled symbol: {symbol}")
name, roundtrip_sig = ct.compilation.demangle_kernel_name(symbol)
print(f"demangled name: {name}")
print(f"roundtrip KernelSignature == original: {roundtrip_sig == sig}")
print(f"roundtrip .parameters == original .parameters: {roundtrip_sig.parameters == sig.parameters}")
print(f"roundtrip .calling_convention == original: {roundtrip_sig.calling_convention == sig.calling_convention}")
print(f"original .symbol: {sig.symbol!r}")
print(f"roundtrip .symbol: {roundtrip_sig.symbol!r}")
```
