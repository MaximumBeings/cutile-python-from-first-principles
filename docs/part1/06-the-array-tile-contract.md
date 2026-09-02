# Chapter 6: The Array-Tile Contract

> "Every `ArrayConstraint` this book has built since Chapter 1 has passed `stride_lower_bound_incl=0` and `alias_groups=[]` without asking what either one actually costs, or what either one actually buys. Both are answers to a question a kernel's Python-level signature can't ask on its own: what is the compiler allowed to assume about memory it has never seen, for arrays it won't meet until a real launch this book still can't attempt? Part 0 treated that question as settled. Part 1 opens by taking it apart."

**What you will understand by the end of this chapter:**

- That `stride_lower_bound_incl` isn't a stylistic default this book happened to pick — the installed compiler genuinely rejects any assumption that permits a negative stride, which is the real reason every prior chapter's `ArrayConstraint` set it to `0`
- That `index_dtype`'s choice between `int32` and `int64` is a genuine, compiled trade-off, not a documentation footnote — confirmed by real, differently-sized cubin output, and genuinely rejected for any dtype outside that pair
- What `alias_groups` actually declares about two array parameters, and that declaring them allowed to alias — or forbidden to — produces measurably different compiled output, in a direction this chapter reports honestly rather than assumes
- That `may_alias_internally` is a distinct question from `alias_groups` — about one array's own indices, not two different arrays — and that setting it `True` genuinely disables an optimization the compiled output shows missing

**What you need to know first:**

- Chapters 1-5's `ArrayConstraint(dtype, ndim, index_dtype=..., stride_lower_bound_incl=..., alias_groups=..., may_alias_internally=...)` construction, used identically in every `KernelSignature` so far without examining any field beyond `dtype` and `ndim`.
- Chapter 5's distinction between a failure at `KernelSignature`/`ArrayConstraint` construction time (a plain Python exception, before the compiler runs at all) and a failure genuinely raised by the compiler itself.

## 6.1 `stride_lower_bound_incl`: Why Every Prior Chapter Used `0` `[FOUNDATIONAL]`

### Intuition

`stride_lower_bound_incl` is documented as an optional, per-dimension inclusive lower bound on an array's stride — pass `0` to declare all strides non-negative, or leave dimensions unconstrained. "Optional" and "unconstrained" suggest a real choice exists. This section tests whether that choice is actually open today.

### Background

An array's stride is how many elements to skip, in memory, to move one position along a given axis — ordinary for a forward-iterating array, negative for one that's been reversed or transposed in a way that walks memory backward. `stride_lower_bound_incl=None` documents "no assumption," which should, in principle, include arrays with negative strides.

### Worked Example 6.1.1 — the bound the compiler actually requires

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(bound):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=bound, alias_groups=[], may_alias_internally=False,
    )

# Every ArrayConstraint since Chapter 1 has passed stride_lower_bound_incl=0
# without asking why. This file tests what actually happens at the two
# ends of that choice: no assumption at all (None), and an explicit
# negative bound.
for name, bound in [("None (no assumption)", None), ("0 (non-negative)", 0),
                     ("1 (positive)", 1), ("-5", -5)]:
    sig = ct.compilation.KernelSignature(
        [array_param(bound), array_param(bound), array_param(bound), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"stride_lower_bound_incl={name}: {len(buf.getvalue())} cubin bytes")
    except TypeError as e:
        print(f"stride_lower_bound_incl={name}: {type(e).__name__}: {e}")
```

The complete file is `01_stride_lower_bound_required.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_stride_lower_bound_required.py
```

Genuinely run:

```
stride_lower_bound_incl=None (no assumption): TypeError: Invalid kernel parameter 'a': Negative strides are currently not supported: please specify stride_lower_bound_incl=0
stride_lower_bound_incl=0 (non-negative): 24736 cubin bytes
stride_lower_bound_incl=1 (positive): 24736 cubin bytes
stride_lower_bound_incl=-5: TypeError: Invalid kernel parameter 'a': Negative strides are currently not supported: please specify stride_lower_bound_incl=0
```

> `[COMMON TRAP]` `None` — the parameter's own documented default, meaning "no assumption" — is genuinely rejected, with the exact same error a literal negative bound produces. This is a direct answer to a question this book never asked out loud until now: every `ArrayConstraint` since Chapter 1 passed `stride_lower_bound_incl=0` not as an arbitrary stylistic choice, but because the installed compiler currently has no support for negative strides at all, and any bound that doesn't rule them out — including "no assumption" — is rejected before compilation reaches the kernel body. `0` and `1` both compile, and to identical output for this kernel, because neither permits a negative stride.

## 6.2 `index_dtype`: A Real, Compiled Trade-off `[FOUNDATIONAL]`

### Intuition

Every `ArrayConstraint` so far has set `index_dtype=ct.int32` without asking what `ct.int64` would cost. The documentation states plainly why the choice exists at all: `int64` supports arrays whose shape or stride would overflow a 32-bit integer. That's a real capability with a real cost, not a free-form annotation — this section measures the cost directly.

### Background

`index_dtype` is documented as accepting exactly two values: `ct.int32` and `ct.int64`. It sets the data type used to represent an array's shape, strides, and indices internally — a 32-bit index computation is smaller and typically cheaper than a 64-bit one, wherever a kernel actually has to do index arithmetic against it.

### Worked Example 6.2.1 — `int32` against `int64`, and the boundary itself

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(index_dtype):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=index_dtype,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# int64 exists specifically for arrays whose shape or stride would
# overflow a 32-bit integer -- a real capability, not a free-form choice,
# so it's worth confirming it genuinely changes what gets compiled.
for name, dt in [("int32", ct.int32), ("int64", ct.int64)]:
    sig = ct.compilation.KernelSignature(
        [array_param(dt), array_param(dt), array_param(dt), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"index_dtype={name}: {len(buf.getvalue())} cubin bytes")

# index_dtype is documented as accepting only int32 or int64 -- a plain
# float dtype is genuinely rejected, at ArrayConstraint construction.
try:
    ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.float32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )
except ValueError as e:
    print(f"index_dtype=float32: {type(e).__name__}: {e}")
```

The complete file is `02_index_dtype_int32_vs_int64.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_index_dtype_int32_vs_int64.py
```

Genuinely run:

```
index_dtype=int32: 24736 cubin bytes
index_dtype=int64: 24888 cubin bytes
index_dtype=float32: ValueError: `index_dtype` must be int32 or int64, got float32
```

`int64` genuinely compiles to more code than `int32` for the identical kernel — 152 bytes more, in this measurement — consistent with wider index arithmetic actually being wider in the generated output, not merely wider on paper. The rejection of `ct.float32` confirms the documented restriction directly: `index_dtype` isn't "any numeric dtype that seems reasonable," it's exactly the two integer widths the compiler's index arithmetic is built to represent.

## 6.3 `alias_groups`: What Two Arrays Are Allowed to Share `[FOUNDATIONAL]`

### Intuition

Every `ArrayConstraint` since Chapter 1 has passed `alias_groups=[]`, declaring that array forbidden from aliasing every other parameter — the safest possible assumption, and also the one that gives the compiler the least excuse to be conservative about reads and writes it can otherwise treat as independent. Two arrays that share a named alias group are allowed to overlap in memory; this section measures what declaring that possibility actually changes.

### Background

`alias_groups` is documented as a sequence of arbitrary strings: an empty sequence forbids the parameter from aliasing anything else, and two parameters may alias each other if and only if they share at least one group name in common. This is a declaration about the *relationship between different array parameters* — a separate question from what Section 6.4 covers next, which is about a single array's own indices.

### Worked Example 6.3.1 — disjoint, fully shared, and partially shared

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(groups):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=groups, may_alias_internally=False,
    )

# Case 1: no array may alias any other -- alias_groups=[] for all three,
# the default this book has used everywhere since Chapter 1.
sig_noalias = ct.compilation.KernelSignature(
    [array_param([]), array_param([]), array_param([]), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig_noalias], buf, gpu_code="sm_90", output_format="cubin")
print(f"disjoint (alias_groups=[] for all): {len(buf.getvalue())} cubin bytes")

# Case 2: all three arrays share one alias group -- any pair may alias.
sig_allalias = ct.compilation.KernelSignature(
    [array_param(["g"]), array_param(["g"]), array_param(["g"]), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig_allalias], buf, gpu_code="sm_90", output_format="cubin")
print(f"shared alias group (all in 'g'): {len(buf.getvalue())} cubin bytes")

# Case 3: a and b share a group; c stays disjoint from both.
sig_partial = ct.compilation.KernelSignature(
    [array_param(["g"]), array_param(["g"]), array_param([]), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig_partial], buf, gpu_code="sm_90", output_format="cubin")
print(f"partial (a,b share 'g'; c disjoint): {len(buf.getvalue())} cubin bytes")
```

The complete file is `03_alias_groups_disjoint_vs_shared.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_alias_groups_disjoint_vs_shared.py
```

Genuinely run:

```
disjoint (alias_groups=[] for all): 24736 cubin bytes
shared alias group (all in 'g'): 24480 cubin bytes
partial (a,b share 'g'; c disjoint): 24864 cubin bytes
```

All three configurations compile — `alias_groups` never rejects a signature the way `stride_lower_bound_incl` did in Section 6.1, it only changes what gets generated. All three also produce genuinely different byte counts, and not in a single obvious direction: the fully-disjoint declaration (24,736 bytes) isn't the smallest of the three, and the fully-shared one (24,480 bytes) isn't the largest. This book reports exactly that, and no more — the declared aliasing relationship between parameters measurably changes the compiled kernel, confirmed directly, without asserting *which* specific optimization each byte-count difference corresponds to, since nothing in this chapter's evidence identifies that mechanism directly.

## 6.4 `may_alias_internally`: One Array, Not Two `[FOUNDATIONAL]`

### Intuition

`alias_groups` is a question about two different array *parameters*. `may_alias_internally` is a different question entirely, about one array on its own: can two distinct, in-bounds indices into that single array land on the same memory location — the way a zero-stride array can make every index along one axis resolve to the same address? The documentation states plainly that setting this `True` may disable certain load/store optimizations; this section confirms that in real, measured output.

### Background

`may_alias_internally=False` — the default, and the value every prior chapter's `ArrayConstraint` has used — asserts that an array behaves the way ordinary tensor-library output does: distinct indices land on distinct addresses. Setting it `True` tells the compiler it cannot assume that, which the documentation says may cost it some load/store optimizations.

### Worked Example 6.4.1 — the documented optimization loss, measured directly

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(internal):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=internal,
    )

# may_alias_internally is documented to matter for arrays where two
# distinct in-bounds indices can point to the same memory location (a
# zero stride, for instance). Setting it True is documented to disable
# certain load/store optimizations -- checked directly here.
for name, val in [("False", False), ("True", True)]:
    sig = ct.compilation.KernelSignature(
        [array_param(val), array_param(val), array_param(val), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"may_alias_internally={name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `04_may_alias_internally.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_may_alias_internally.py
```

Genuinely run:

```
may_alias_internally=False: 24736 cubin bytes
may_alias_internally=True: 24864 cubin bytes
```

`True` genuinely compiles to more code than `False` — 128 bytes more, for this kernel — which is at least consistent with the documented claim that permitting internal aliasing disables some load/store optimization: the compiler can no longer assume a load and a later store to the same array's different indices are touching different memory, so it has less room to reorder or cache around them.

## Complete Runnable Code

### File: `01_stride_lower_bound_required.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(bound):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=bound, alias_groups=[], may_alias_internally=False,
    )

# Every ArrayConstraint since Chapter 1 has passed stride_lower_bound_incl=0
# without asking why. This file tests what actually happens at the two
# ends of that choice: no assumption at all (None), and an explicit
# negative bound.
for name, bound in [("None (no assumption)", None), ("0 (non-negative)", 0),
                     ("1 (positive)", 1), ("-5", -5)]:
    sig = ct.compilation.KernelSignature(
        [array_param(bound), array_param(bound), array_param(bound), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"stride_lower_bound_incl={name}: {len(buf.getvalue())} cubin bytes")
    except TypeError as e:
        print(f"stride_lower_bound_incl={name}: {type(e).__name__}: {e}")
```

```bash
python3 01_stride_lower_bound_required.py
```

### File: `02_index_dtype_int32_vs_int64.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(index_dtype):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=index_dtype,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# int64 exists specifically for arrays whose shape or stride would
# overflow a 32-bit integer -- a real capability, not a free-form choice,
# so it's worth confirming it genuinely changes what gets compiled.
for name, dt in [("int32", ct.int32), ("int64", ct.int64)]:
    sig = ct.compilation.KernelSignature(
        [array_param(dt), array_param(dt), array_param(dt), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"index_dtype={name}: {len(buf.getvalue())} cubin bytes")

# index_dtype is documented as accepting only int32 or int64 -- a plain
# float dtype is genuinely rejected, at ArrayConstraint construction.
try:
    ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.float32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )
except ValueError as e:
    print(f"index_dtype=float32: {type(e).__name__}: {e}")
```

```bash
python3 02_index_dtype_int32_vs_int64.py
```

### File: `03_alias_groups_disjoint_vs_shared.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(groups):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=groups, may_alias_internally=False,
    )

# Case 1: no array may alias any other -- alias_groups=[] for all three,
# the default this book has used everywhere since Chapter 1.
sig_noalias = ct.compilation.KernelSignature(
    [array_param([]), array_param([]), array_param([]), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig_noalias], buf, gpu_code="sm_90", output_format="cubin")
print(f"disjoint (alias_groups=[] for all): {len(buf.getvalue())} cubin bytes")

# Case 2: all three arrays share one alias group -- any pair may alias.
sig_allalias = ct.compilation.KernelSignature(
    [array_param(["g"]), array_param(["g"]), array_param(["g"]), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig_allalias], buf, gpu_code="sm_90", output_format="cubin")
print(f"shared alias group (all in 'g'): {len(buf.getvalue())} cubin bytes")

# Case 3: a and b share a group; c stays disjoint from both.
sig_partial = ct.compilation.KernelSignature(
    [array_param(["g"]), array_param(["g"]), array_param([]), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig_partial], buf, gpu_code="sm_90", output_format="cubin")
print(f"partial (a,b share 'g'; c disjoint): {len(buf.getvalue())} cubin bytes")
```

```bash
python3 03_alias_groups_disjoint_vs_shared.py
```

### File: `04_may_alias_internally.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(internal):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=internal,
    )

# may_alias_internally is documented to matter for arrays where two
# distinct in-bounds indices can point to the same memory location (a
# zero stride, for instance). Setting it True is documented to disable
# certain load/store optimizations -- checked directly here.
for name, val in [("False", False), ("True", True)]:
    sig = ct.compilation.KernelSignature(
        [array_param(val), array_param(val), array_param(val), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"may_alias_internally={name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 04_may_alias_internally.py
```

## Chapter Summary

Four `ArrayConstraint` fields every prior chapter used without examining turned out to each carry a real, testable answer. `stride_lower_bound_incl` genuinely must rule out negative strides — the installed compiler rejects `None` (its own documented "no assumption" default) and any explicit negative bound with the identical `TypeError`, which is the real reason every prior chapter's constraint passed `0` rather than an arbitrary stylistic habit. `index_dtype`'s choice between `int32` and `int64` is a real, measured trade-off — `int64` genuinely compiles to more code for the same kernel — and is genuinely restricted to exactly those two values, rejecting a plain `ct.float32` immediately at construction. `alias_groups` declares which array parameters are allowed to overlap in memory, and changing that declaration — disjoint, fully shared, or partially shared — produced three genuinely different, deterministic byte counts, reported here without overclaiming which specific optimization each difference reflects. And `may_alias_internally`, a distinct question about one array's own indices rather than a relationship between two arrays, genuinely cost more compiled code when set `True`, consistent with its documented claim that doing so disables certain load/store optimizations.

## Self-Check Questions

1. Every `ArrayConstraint` since Chapter 1 has passed `stride_lower_bound_incl=0`. This chapter found that `None` — the parameter's own documented default — is rejected. What does that tell you about the relationship between a parameter's *documented* default and what the *installed compiler* actually currently supports?
2. `index_dtype=ct.int64` compiled to a larger cubin than `index_dtype=ct.int32` for the identical kernel. State, in your own words, why a wider index type would be expected to cost more compiled code, and under what real circumstance you'd have to accept that cost anyway.
3. What is the precise difference between what `alias_groups` and `may_alias_internally` each declare — one is about a relationship between which two things, and the other is about which single thing?
4. Section 6.3 reported that the fully-disjoint `alias_groups=[]` configuration was neither the smallest nor the largest of the three tested. Why does this chapter stop short of explaining *which* specific compiler optimization produces each byte-count difference?
5. If you were compiling a kernel where two of its three array parameters are genuinely allowed to be the exact same underlying buffer at launch time, which `ArrayConstraint` field would you need to change from every prior chapter's default, and how would you set it?

## Where We Go Next

This chapter treated an array's shape, stride, and base address as entirely unknown quantities the compiler has to reason about defensively — the most conservative possible starting point, and the one every field examined here still assumed by default. Chapter 7 turns to the opposite side of `ArrayConstraint`: `stride_constant`, `shape_constant`, `stride_divisible_by`, `shape_divisible_by`, and `base_addr_divisible_by` — the fields that let a kernel author hand the compiler real, verifiable assumptions about a specific array's memory layout, and what those assumptions are actually worth in the code they let the compiler generate, genuinely measured the same way every assumption in this chapter was.

## Worked Solutions

**1.** It tells you the two aren't guaranteed to move together: a parameter's own docstring can describe a general design intent (`None` meaning "no assumption is made, permitting any stride including negative ones") that the specific, installed version of the compiler hasn't actually implemented support for yet. Trusting a parameter's documented default without testing it against the real, installed toolchain is exactly the gap this book's whole verification discipline exists to catch.

**2.** A 64-bit index requires wider registers and wider arithmetic instructions than a 32-bit one for every index computation the kernel actually performs — more bits to load, add, and compare, which shows up directly as more generated instructions. You'd have to accept that cost when an array's shape or stride values genuinely can't fit in a 32-bit integer — the exact, documented reason `int64` exists as an option at all, rather than a default nobody would choose without a specific need.

**3.** `alias_groups` is about the relationship *between two different array parameters* — whether array `a` and array `b` are allowed to occupy overlapping memory. `may_alias_internally` is about *one array's own indices* — whether two distinct, in-bounds positions inside that single array (such as two different indices into a zero-stride axis) are allowed to resolve to the same memory location. One is a question about two things; the other is a question about one thing's own internal structure.

**4.** Because nothing this chapter actually tested identifies which specific optimization pass produces each measured difference — only `export_kernel`'s final byte count, not any intermediate compiler stage or disassembly, was available to inspect. Asserting a specific mechanism ("the disjoint case is bigger because the compiler duplicated this particular load") without evidence that specifically shows that mechanism at work would be exactly the kind of unverified claim this book's standing rule against fabricated output exists to prevent — reporting the measured, deterministic difference honestly, without inventing the reason behind it, is the more defensible choice.

**5.** `alias_groups` — assign the two array parameters that are genuinely allowed to be the same buffer a shared, non-empty group string (for example, both set to `["g"]`), rather than leaving both at the default `[]`, which asserts they may never overlap. Declaring `alias_groups=[]` for arrays that can actually alias at launch time would hand the compiler a false assumption about your program's real memory layout — the kind of mismatch between a compiled assumption and launch-time reality this book has flagged as a genuine risk since Chapter 3's `TileInternalError`.
