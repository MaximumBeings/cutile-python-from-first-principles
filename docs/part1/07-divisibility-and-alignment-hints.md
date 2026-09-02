# Chapter 7: Divisibility and Alignment Hints

> "Chapter 6 asked what the compiler has to assume when it's told almost nothing about an array's memory. This chapter asks the opposite question: what does it cost, and what does it buy, when a kernel author tells the compiler something true and specific — that a stride is always 1, that a shape is always a multiple of 8, that a base address always lands on a 256-byte boundary? Every one of those is a genuine, checkable claim about real memory this book still can't allocate. What can be verified, without a GPU, is whether making that claim changes what the compiler actually produces — and by how much."

**What you will understand by the end of this chapter:**

- That `stride_constant` — a real, per-dimension guarantee about an array's exact stride — measurably shrinks compiled output for the one case ordinary tensor libraries actually produce: a C-contiguous last dimension
- That `shape_constant`, `stride_divisible_by`, `shape_divisible_by`, and `base_addr_divisible_by` each genuinely change compiled output too, but not always in the direction "more information, smaller code" would predict — reported honestly, the same way Chapter 2's `opt_level` and Chapter 6's `alias_groups` were
- That `shape_constant` is validated against `shape_divisible_by` for genuine mathematical consistency at `ArrayConstraint` construction — a real, checkable claim ("1024 is divisible by 7") rejected immediately when it's false
- That `shape_constant`, like Chapter 5's tuple-shaped `ct.Constant`, requires `cutile_python_v2` and is genuinely rejected under `v1`

**What you need to know first:**

- Chapter 6's `ArrayConstraint(dtype, ndim, index_dtype=..., stride_lower_bound_incl=..., alias_groups=..., may_alias_internally=...)` fields, and its finding that a compiled-output difference doesn't always move in the direction naive intuition predicts.
- Chapter 5's `cutile_python_v1`/`v2` calling-convention split and its genuine rejection of a capability `v1` doesn't support.

## 7.1 `stride_constant`: A Real Layout Guarantee `[FOUNDATIONAL]`

### Intuition

Chapter 6's `stride_lower_bound_incl` only ever bounded a stride — never pinned it. `stride_constant` goes further: it lets a kernel author declare a dimension's stride as one exact, known value. The most common real case is the one this section tests directly — a freshly-allocated, row-major array, whose last dimension genuinely always has stride 1.

### Background

`stride_constant` takes a per-dimension sequence of optional integers, or `None` if no dimension's stride is known. A value of `1` for the last dimension is documented as the case that "can enable certain optimizations of loads and stores" — the layout NumPy, CuPy, and PyTorch all produce by default for a new, C-contiguous array.

### Worked Example 7.1.1 — unknown, contiguous, and strided

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(stride_const):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        stride_constant=stride_const,
    )

# stride_constant lets a kernel author hand the compiler a real, known
# stride value per dimension -- [1] declares a C-contiguous last axis,
# the layout ordinary tensor libraries produce for a fresh array.
for name, sc in [("None (unknown)", None), ("[1] (contiguous)", [1]), ("[2] (strided by 2)", [2])]:
    sig = ct.compilation.KernelSignature(
        [array_param(sc), array_param(sc), array_param(sc), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"stride_constant={name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `01_stride_constant.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_stride_constant.py
```

Genuinely run:

```
stride_constant=None (unknown): 24736 cubin bytes
stride_constant=[1] (contiguous): 23840 cubin bytes
stride_constant=[2] (strided by 2): 24096 cubin bytes
```

Declaring the real, contiguous case genuinely produces the smallest output of the three — 896 bytes smaller than leaving the stride unknown, and smaller still than declaring a real but non-unit stride of `2`. This is the one result in this chapter that matches the "more known information, less generated code" intuition directly: a stride the compiler knows is always exactly `1` needs no runtime stride multiplication for address computation at all, an optimization a stride of `2` — known or not — still can't skip.

## 7.2 `shape_constant`: A Genuine Trade-off, and a Real Restriction `[FOUNDATIONAL]`

### Intuition

`shape_constant` looks like the natural extension of Section 7.1's idea to an array's length instead of its stride — and the documentation says plainly it requires `cutile_python_v2`, the same restriction Chapter 5 found for a tuple-shaped `ct.Constant`. What's worth testing directly is whether pinning a shape actually shrinks compiled output the way pinning a stride did.

### Background

`shape_constant` takes a per-dimension sequence of optional integers, declaring an array's exact length along that axis. Unlike `stride_constant`, it's explicitly documented as requiring `cutile_python_v2` — a real capability boundary, checkable the same way Chapter 5 checked `v1`'s rejection of tuple parameters.

### Worked Example 7.2.1 — a known shape, and the `v1` boundary

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(shape_const):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        shape_constant=shape_const,
    )

v2 = ct.compilation.CallingConvention.cutile_python_v2()
v1 = ct.compilation.CallingConvention.cutile_python_v1()

# shape_constant hands the compiler a known, fixed length for a
# dimension -- here, that the array genuinely always has 1024 elements.
for name, shc in [("None (unknown)", None), ("[1024] (known)", [1024])]:
    sig = ct.compilation.KernelSignature(
        [array_param(shc), array_param(shc), array_param(shc), ct.compilation.ConstantConstraint(256)],
        calling_convention=v2,
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"shape_constant={name} (v2): {len(buf.getvalue())} cubin bytes")

# Documented to require cutile_python_v2, the same restriction Chapter 5
# found for tuple-shaped ct.Constant parameters.
try:
    sig = ct.compilation.KernelSignature(
        [array_param([1024]), array_param([1024]), array_param([1024]), ct.compilation.ConstantConstraint(256)],
        calling_convention=v1,
    )
except ValueError as e:
    print(f"v1 with shape_constant: {type(e).__name__}: {e}")
```

The complete file is `02_shape_constant.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_shape_constant.py
```

Genuinely run:

```
shape_constant=None (unknown) (v2): 24736 cubin bytes
shape_constant=[1024] (known) (v2): 25264 cubin bytes
v1 with shape_constant: ValueError: Static array shapes are not supported by calling convention cutile_python_v1; version >= 2 is required
```

`v1`'s rejection is exactly what the documentation predicts, worded almost identically to Chapter 5's tuple-parameter rejection: `shape_constant` is a `v2`-only capability, confirmed directly.

> `[COMMON TRAP]` The byte-count result is not what Section 7.1's `stride_constant` result would suggest. Pinning a *known, exact shape* genuinely produced *more* compiled code than leaving it unknown — 528 bytes more — the opposite direction from pinning a known, exact stride. This book draws only the conclusion its own evidence supports: `shape_constant` is a real, measurable trade-off, not a straightforward size reduction, and this chapter reports that honestly rather than assuming every "more information" hint moves compiled output the same direction the way Chapter 2's `opt_level` and Chapter 6's `alias_groups` findings already cautioned against.

## 7.3 `stride_divisible_by` and `shape_divisible_by`: Weaker Than Constant, Still Real `[FOUNDATIONAL]`

### Intuition

`stride_constant` and `shape_constant` both pin a dimension to one exact value. `stride_divisible_by` and `shape_divisible_by` make a weaker, more broadly true claim instead: not "this stride is exactly 4," but "this stride, whatever it is, is a multiple of 4." That weaker claim is still real information — this section measures what it's actually worth.

### Background

Both hints are documented in array elements, not bytes, and both default to `1` — no assumption. A `float16` array declared `stride_divisible_by=8` is really claiming 16-byte divisibility, since each element is 2 bytes wide; the compiler is free to use that weaker guarantee for whatever optimizations don't need the exact value, such as choosing a wider vectorized memory instruction.

### Worked Example 7.3.1 — `stride_divisible_by`, tested at several values

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        stride_divisible_by=div,
    )

# stride_divisible_by declares that a dimension's stride, whatever its
# exact value, is a multiple of the given factor -- weaker than
# stride_constant, but still a real assumption the compiler can use.
for name, div in [("1 (no assumption)", 1), ("2", 2), ("4", 4), ("8", 8)]:
    sig = ct.compilation.KernelSignature(
        [array_param(div), array_param(div), array_param(div), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"stride_divisible_by={name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `03_stride_divisible_by.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_stride_divisible_by.py
```

Genuinely run:

```
stride_divisible_by=1 (no assumption): 24736 cubin bytes
stride_divisible_by=2: 24864 cubin bytes
stride_divisible_by=4: 24864 cubin bytes
stride_divisible_by=8: 24864 cubin bytes
```

Any divisibility assumption beyond `1` costs the identical 128 bytes here, regardless of whether that assumption is `2`, `4`, or `8` — a real, measured fact this book reports without inventing a mechanism it didn't independently confirm: something about *having a divisibility assumption at all* apparently changes what gets generated for this kernel, while the specific factor chosen, among the three values tested, doesn't.

### Worked Example 7.3.2 — `shape_divisible_by`, where the specific factor does matter

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        shape_divisible_by=div,
    )

# shape_divisible_by declares that a dimension's length is a multiple of
# the given factor, without pinning it to one exact value the way
# shape_constant does.
for name, div in [("1 (no assumption)", 1), ("256 (matches tile_size)", 256),
                   ("7 (does not divide 256 evenly)", 7)]:
    sig = ct.compilation.KernelSignature(
        [array_param(div), array_param(div), array_param(div), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"shape_divisible_by={name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `04_shape_divisible_by.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_shape_divisible_by.py
```

Genuinely run:

```
shape_divisible_by=1 (no assumption): 24736 cubin bytes
shape_divisible_by=256 (matches tile_size): 24992 cubin bytes
shape_divisible_by=7 (does not divide 256 evenly): 24864 cubin bytes
```

Unlike `stride_divisible_by`, `shape_divisible_by`'s specific factor genuinely does change the byte count here: `256` (a multiple of the kernel's own `tile_size` constant) and `7` (which isn't) produce two different, non-`1` byte counts, not one shared value the way `stride_divisible_by`'s `2`/`4`/`8` did. This book reports the contrast as observed — two hints that sound structurally identical don't necessarily behave identically under the same compiler, for the same kernel.

## 7.4 `base_addr_divisible_by`: Alignment on the Array as a Whole `[FOUNDATIONAL]`

### Intuition

Every hint so far in this chapter has been per-dimension. `base_addr_divisible_by` is different: it's a single value per array, declaring an alignment guarantee — in bytes — on where the array's data genuinely starts in memory, not on any one axis's shape or stride.

### Background

`base_addr_divisible_by` defaults to `1` (no assumption) and is given directly in bytes, unlike `stride_divisible_by`/`shape_divisible_by`'s element-based units. A real allocator commonly guarantees some minimum alignment for any allocation it returns — 16, 128, or 256 bytes are typical real values — which is exactly the kind of fact this parameter exists to hand the compiler.

### Worked Example 7.4.1 — three alignment guarantees, genuinely compiled

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        base_addr_divisible_by=div,
    )

# base_addr_divisible_by declares an alignment guarantee on the array's
# starting address itself, in bytes -- a single value per array, not
# per-dimension the way the other hints in this chapter are.
for name, div in [("1 (no assumption)", 1), ("16 (bytes)", 16), ("256 (bytes)", 256)]:
    sig = ct.compilation.KernelSignature(
        [array_param(div), array_param(div), array_param(div), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"base_addr_divisible_by={name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `05_base_addr_divisible_by.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_base_addr_divisible_by.py
```

Genuinely run:

```
base_addr_divisible_by=1 (no assumption): 24736 cubin bytes
base_addr_divisible_by=16 (bytes): 25120 cubin bytes
base_addr_divisible_by=256 (bytes): 25264 cubin bytes
```

Both alignment guarantees genuinely compile to more code than making no assumption at all, and `256` compiles to more than `16` — a real, monotonic relationship in this chapter's measurements, but reported as exactly that and nothing more: this book's evidence doesn't say *why* a stronger alignment guarantee costs more code here, only that it genuinely, reproducibly does for this kernel.

### Worked Example 7.4.2 — `shape_constant` and `shape_divisible_by`, checked for consistency

```python
import cuda.tile as ct

def array_param(shape_const, shape_div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        shape_constant=shape_const, shape_divisible_by=shape_div,
    )

# Documented: when shape_constant[i] is set, shape_divisible_by[i] is
# redundant and must be compatible with it -- 1024 is genuinely divisible
# by 8, so this is accepted even though shape_divisible_by is then
# ignored in favor of the exact known value.
try:
    array_param([1024], 8)
    print("compatible (shape_constant=1024, shape_divisible_by=8): accepted")
except ValueError as e:
    print(f"compatible: {type(e).__name__}: {e}")

# 1024 is genuinely not divisible by 7 -- a real, checkable
# inconsistency between the two hints, rejected at construction.
try:
    array_param([1024], 7)
    print("incompatible (shape_constant=1024, shape_divisible_by=7): accepted (unexpected)")
except ValueError as e:
    print(f"incompatible: {type(e).__name__}: {e}")
```

The complete file is `06_shape_constant_divisibility_compatibility.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_shape_constant_divisibility_compatibility.py
```

Genuinely run:

```
compatible (shape_constant=1024, shape_divisible_by=8): accepted
incompatible: ValueError: shape_constant[0] is set to 1024, which is not divisible by 7
```

This is documented behavior, genuinely confirmed: `shape_divisible_by` becomes redundant once `shape_constant` pins an exact value, but redundant doesn't mean unchecked — supplying two hints that mathematically contradict each other (`1024` is not a multiple of `7`) is caught immediately at `ArrayConstraint` construction, the same construction-time validation Chapter 5 found for a signature's `calling_convention` mismatch and Chapter 6 found for `index_dtype`.

## Complete Runnable Code

### File: `01_stride_constant.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(stride_const):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        stride_constant=stride_const,
    )

# stride_constant lets a kernel author hand the compiler a real, known
# stride value per dimension -- [1] declares a C-contiguous last axis,
# the layout ordinary tensor libraries produce for a fresh array.
for name, sc in [("None (unknown)", None), ("[1] (contiguous)", [1]), ("[2] (strided by 2)", [2])]:
    sig = ct.compilation.KernelSignature(
        [array_param(sc), array_param(sc), array_param(sc), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"stride_constant={name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 01_stride_constant.py
```

### File: `02_shape_constant.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(shape_const):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        shape_constant=shape_const,
    )

v2 = ct.compilation.CallingConvention.cutile_python_v2()
v1 = ct.compilation.CallingConvention.cutile_python_v1()

# shape_constant hands the compiler a known, fixed length for a
# dimension -- here, that the array genuinely always has 1024 elements.
for name, shc in [("None (unknown)", None), ("[1024] (known)", [1024])]:
    sig = ct.compilation.KernelSignature(
        [array_param(shc), array_param(shc), array_param(shc), ct.compilation.ConstantConstraint(256)],
        calling_convention=v2,
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"shape_constant={name} (v2): {len(buf.getvalue())} cubin bytes")

# Documented to require cutile_python_v2, the same restriction Chapter 5
# found for tuple-shaped ct.Constant parameters.
try:
    sig = ct.compilation.KernelSignature(
        [array_param([1024]), array_param([1024]), array_param([1024]), ct.compilation.ConstantConstraint(256)],
        calling_convention=v1,
    )
except ValueError as e:
    print(f"v1 with shape_constant: {type(e).__name__}: {e}")
```

```bash
python3 02_shape_constant.py
```

### File: `03_stride_divisible_by.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        stride_divisible_by=div,
    )

# stride_divisible_by declares that a dimension's stride, whatever its
# exact value, is a multiple of the given factor -- weaker than
# stride_constant, but still a real assumption the compiler can use.
for name, div in [("1 (no assumption)", 1), ("2", 2), ("4", 4), ("8", 8)]:
    sig = ct.compilation.KernelSignature(
        [array_param(div), array_param(div), array_param(div), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"stride_divisible_by={name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 03_stride_divisible_by.py
```

### File: `04_shape_divisible_by.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        shape_divisible_by=div,
    )

# shape_divisible_by declares that a dimension's length is a multiple of
# the given factor, without pinning it to one exact value the way
# shape_constant does.
for name, div in [("1 (no assumption)", 1), ("256 (matches tile_size)", 256),
                   ("7 (does not divide 256 evenly)", 7)]:
    sig = ct.compilation.KernelSignature(
        [array_param(div), array_param(div), array_param(div), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"shape_divisible_by={name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 04_shape_divisible_by.py
```

### File: `05_base_addr_divisible_by.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param(div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        base_addr_divisible_by=div,
    )

# base_addr_divisible_by declares an alignment guarantee on the array's
# starting address itself, in bytes -- a single value per array, not
# per-dimension the way the other hints in this chapter are.
for name, div in [("1 (no assumption)", 1), ("16 (bytes)", 16), ("256 (bytes)", 256)]:
    sig = ct.compilation.KernelSignature(
        [array_param(div), array_param(div), array_param(div), ct.compilation.ConstantConstraint(256)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"base_addr_divisible_by={name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 05_base_addr_divisible_by.py
```

### File: `06_shape_constant_divisibility_compatibility.py`

```python
import cuda.tile as ct

def array_param(shape_const, shape_div):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        shape_constant=shape_const, shape_divisible_by=shape_div,
    )

# Documented: when shape_constant[i] is set, shape_divisible_by[i] is
# redundant and must be compatible with it -- 1024 is genuinely divisible
# by 8, so this is accepted even though shape_divisible_by is then
# ignored in favor of the exact known value.
try:
    array_param([1024], 8)
    print("compatible (shape_constant=1024, shape_divisible_by=8): accepted")
except ValueError as e:
    print(f"compatible: {type(e).__name__}: {e}")

# 1024 is genuinely not divisible by 7 -- a real, checkable
# inconsistency between the two hints, rejected at construction.
try:
    array_param([1024], 7)
    print("incompatible (shape_constant=1024, shape_divisible_by=7): accepted (unexpected)")
except ValueError as e:
    print(f"incompatible: {type(e).__name__}: {e}")
```

```bash
python3 06_shape_constant_divisibility_compatibility.py
```

## Chapter Summary

Every divisibility and alignment hint this chapter tested genuinely changed compiled output, but not all in the same direction, and this book reports each honestly rather than forcing a single tidy story. `stride_constant=[1]` — a real C-contiguous last dimension, the layout ordinary tensor libraries actually produce — genuinely shrank compiled output, the one result matching naive "more information, less code" intuition directly. `shape_constant`, by contrast, genuinely grew compiled output when a shape was pinned exactly, and is confirmed, like Chapter 5's tuple-shaped `ct.Constant`, to require `cutile_python_v2` and be rejected under `v1`. `stride_divisible_by` produced one identical byte count across three different nontrivial factors (`2`, `4`, `8`), while `shape_divisible_by`'s specific factor did change the result — two structurally similar hints, genuinely confirmed to behave differently for this kernel. `base_addr_divisible_by` grew compiled output monotonically with a stronger alignment guarantee. And `shape_constant`/`shape_divisible_by` are validated together for real mathematical consistency at `ArrayConstraint` construction — a genuinely incompatible pair (`1024` and `7`) rejected immediately, before a signature is ever built or a kernel is compiled.

## Self-Check Questions

1. `stride_constant=[1]` produced smaller compiled output than leaving the stride unknown, but `shape_constant=[1024]` produced larger compiled output than leaving the shape unknown. What does this pair of results tell you about assuming every "the compiler now knows more" hint shrinks generated code?
2. `stride_divisible_by`'s three tested nontrivial values (`2`, `4`, `8`) all produced the identical byte count, while `shape_divisible_by`'s tested values (`256` and `7`) produced two different byte counts. Is it safe to conclude that `stride_divisible_by`'s specific factor never matters, based on this chapter's evidence alone? Why or why not?
3. Why does `shape_constant` require `cutile_python_v2`, based on what Chapter 5 already established about that calling convention?
4. `array_param([1024], 7)` in Worked Example 7.4.2 is rejected with a `ValueError` at `ArrayConstraint` construction, before any `KernelSignature` or `export_kernel` call. What relationship between `shape_constant` and `shape_divisible_by` does that rejection genuinely verify?
5. This chapter measured `base_addr_divisible_by=256` compiling to more code than `base_addr_divisible_by=1`. Would it be accurate for this book to state "declaring stronger alignment on an array's base address always increases cuTile Python's compiled output"? Why or why not?

## Where We Go Next

This chapter and the one before it treated an array's shape, stride, alignment, and aliasing as facts declared once, at compile time, and trusted from then on — exactly the trust a real launch would have to honor or violate. Chapter 8 turns to what happens at a tile's actual edges: masking a load or store against a boundary the tile's shape doesn't evenly divide, the padding-mode groundwork Chapter 3 introduced, and the rounding modes that govern how a value that doesn't fit exactly gets converted when precision, not just shape, is the boundary being crossed.

## Worked Solutions

**1.** It tells you the direction of a "more compile-time information" hint's effect on output size isn't something you can assume in advance for every field — each one has to be tested on its own terms, the same lesson Chapter 2's `opt_level` and Chapter 6's `alias_groups` already established for different fields. `stride_constant` and `shape_constant` sound structurally parallel (both pin a per-dimension value to a known constant), but this chapter's own measurements show they don't move compiled output the same direction for this kernel.

**2.** No — three tested values producing the same result doesn't establish that no value ever would. `stride_divisible_by`'s specific factor genuinely might matter for a different kernel, a different array dtype, or a divisor outside the small range this chapter actually tested (`2`, `4`, `8`); this chapter's evidence only supports "these three specific factors produced identical output for this specific kernel," not a general claim about every possible factor.

**3.** Chapter 5 established that `cutile_python_v1` rejects tuple-shaped `ct.Constant` parameters, accepting only `cutile_python_v2` for that capability. `shape_constant` is a structurally similar addition — a compile-time-fixed, structured piece of information about a kernel parameter that the original `v1` calling convention's argument-passing scheme was never built to carry — so it falls under the same `v2`-only restriction, for the same underlying reason.

**4.** It verifies that `shape_constant` and `shape_divisible_by`, when both set for the same dimension, are checked for genuine mathematical consistency with each other — not merely accepted as two independent, unrelated hints. `1024` is not evenly divisible by `7`, a real, checkable arithmetic fact, and cuTile Python catches that inconsistency immediately rather than silently ignoring `shape_divisible_by` (as the documentation says happens when the two hints agree) or letting a genuinely contradictory pair of assumptions reach the compiler.

**5.** No. This chapter's evidence supports "for this specific kernel, `base_addr_divisible_by` values of `1`, `16`, and `256` produced these three specific, increasing byte counts" — a measured fact about one kernel, one dtype, and three specific alignment values. Generalizing that into "always increases" for every kernel, every array dtype, and every alignment value would be exactly the kind of untested extrapolation this book's verification discipline exists to avoid, the same caution Chapter 2 applied to `opt_level` and this chapter already applied to `shape_divisible_by`'s factor-dependence.
