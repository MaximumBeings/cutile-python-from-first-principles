# Chapter 2: Kernel Parameterization

> "`ct.Constant` is not a type hint in the way `int` or `float` are in ordinary Python — it doesn't describe what kind of data a parameter holds, it describes when the compiler is allowed to know its value. A `Constant[int]` parameter is not an integer that happens to be known early; it's a promise, checked by the compiler in both directions, that the value will be baked directly into the generated code before a single instruction exists — which is exactly why a tile's shape has to be one, and a running total from `ct.sum` never can be."

**What you will understand by the end of this chapter:**

- That `ct.Constant` is a compiler-enforced contract, not decoration — annotating a parameter `Constant` and then supplying a `ScalarConstraint` for it at compile time is a real, genuinely-triggered `TypeError`, and so is the reverse: supplying a `ConstantConstraint` for a parameter that isn't annotated `Constant` at all
- Why `ct.load`'s `shape` argument specifically requires a compile-time constant tuple, genuinely confirmed by the exact `TileTypeError` the compiler raises when a tile size arrives as an ordinary runtime scalar instead
- How to declare a genuinely multi-dimensional compile-time constant — a `(32, 64)` tile shape — using `TupleConstraint`, and why `ConstantConstraint` alone rejects a Python tuple outright
- That one kernel function can be ahead-of-time compiled for *several* signatures in a single `export_kernel` call — real, differently-shaped specializations of the same source, bundled into one genuinely growing output — which is what makes a single cuTile Python kernel portable across tile sizes the same way Chapter 1's multi-architecture export made it portable across GPUs
- That `opt_level`, a real keyword argument to `@ct.kernel`, genuinely changes the compiled binary's size in this book's own environment — and that the relationship isn't the simple "higher is always smaller" story you might expect

**What you need to know first:**

- Chapter 1's vocabulary: `Array` vs. `Tile`, the kernel boundary, and `ct.compilation.export_kernel` as this book's ahead-of-time, driver-free verification method. This chapter reuses that same `ArrayConstraint`/`KernelSignature` scaffolding without re-explaining it from scratch.
- Nothing about compile-time constants in cuTile Python is unique to GPU programming as such — the closest everyday analogy is a C or C++ template parameter, or a Rust const generic: a value the compiler needs settled before it can generate code, as opposed to a value that only exists once the program is actually running.

## 2.1 `ct.Constant`: A Contract, Checked in Both Directions `[FOUNDATIONAL]`

### Intuition

Think of a kernel parameter's type annotation not as a label but as a promise made to a notary, and the `KernelSignature` passed to `export_kernel` as the notarized document that has to match it exactly. If the promise says "this value will be known and fixed before any code is generated" (`ct.Constant`), the document had better actually supply a fixed value (`ConstantConstraint`) — a promise to notarize a *variable* amount instead is refused. And the reverse holds too: notarizing a fixed amount against a promise that never claimed to be about a fixed value at all is refused just as firmly. cuTile Python's compiler is that notary, and it checks the match in both directions, not just one.

### Background

`ct.Constant` (usable bare, or subscripted as `ct.Constant[int]`, `ct.Constant[tuple[int, int]]`, and similar) marks a kernel parameter as *constant-embedded*: its value is baked directly into the compiled code for one specific signature, the same value every time that compiled specialization runs, rather than read from memory or a register at launch time. Every kernel parameter's `KernelSignature` entry has to agree with its Python-level annotation about this:

| Python annotation | Compatible `KernelSignature` constraint | Incompatible constraint |
|---|---|---|
| `tile_size: ct.Constant[int]` | `ConstantConstraint(256)` | `ScalarConstraint(ct.int32)` |
| `n` (no `Constant` annotation) | `ScalarConstraint(ct.int32)` | `ConstantConstraint(256)` |

### Worked Example 2.1.1 — the baseline: annotation and constraint agree

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled with tile_size embedded as 256:", len(buf.getvalue()), "bytes")
```

The complete file is `01_vector_add_with_constant.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_vector_add_with_constant.py
```

Genuinely run:

```
compiled with tile_size embedded as 256: 716 bytes
```

`tile_size` never appears as a runtime value anywhere in the compiled output — it's `ct.Constant[int]` in the Python annotation, and `ConstantConstraint(256)` in the signature, and the two agree, so the compiler bakes the literal `256` into the specialization it produces for this one signature. (As Chapter 1's `[COMMON TRAP]` noted, the exact byte count here — 716, not the 712 Chapter 1 got for what reads like the identical kernel — reflects this file's own exact source text, not some difference in what the kernel does.)

### Worked Example 2.1.2 — a genuinely rejected mismatch: `Constant` claimed, never supplied

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add_no_annotation(a, b, c, tile_size):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add_no_annotation, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except TypeError as e:
    print(f"{type(e).__name__}: {e}")
```

The complete file, `02_constant_annotation_missing.py`, is included in Complete Runnable Code — it runs to completion (the `TypeError` is caught and printed), it just never produces a compiled kernel.

```bash
python3 02_constant_annotation_missing.py
```

Genuinely run:

```
TypeError: Invalid kernel parameter 'tile_size': ConstantConstraint is only valid for parameters annotated as Constant.
```

`tile_size` here carries no annotation at all — an ordinary, un-typed Python parameter, which cuTile Python treats as a ordinary scalar parameter by default. Supplying a `ConstantConstraint` for it anyway is rejected immediately, before compilation even starts on the kernel's body — this is a signature-level check, not something the compiler discovers partway through translating `ct.load`.

### Worked Example 2.1.3 — the mismatch in the other direction

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ScalarConstraint(ct.int32)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except TypeError as e:
    print(f"{type(e).__name__}: {e}")
```

`03_constant_constraint_missing.py`, included in Complete Runnable Code:

```bash
python3 03_constant_constraint_missing.py
```

Genuinely run:

```
TypeError: Invalid kernel parameter 'tile_size': Expected a scalar/tuple constant, as implied by the Constant annotation.
```

This is the same `vector_add` from Worked Example 2.1.1, unchanged — only the signature's constraint for `tile_size` changed, from `ConstantConstraint(256)` to `ScalarConstraint(ct.int32)`. The Python-level annotation still says `ct.Constant[int]`, and the compiler holds the signature to that promise just as strictly as it held Worked Example 2.1.2's signature to its absence.

> `[COMMON TRAP]` It's easy to assume a type annotation in Python is advisory — a hint a checker like `mypy` might use, but that has no effect on what actually runs. `ct.Constant` is not that kind of annotation. Both directions of mismatch above are rejected before the kernel's body is even translated, which means the annotation is load-bearing: change `tile_size: ct.Constant[int]` to plain `tile_size` (or vice versa) and every existing `KernelSignature` written against the old annotation stops compiling, exactly as Worked Examples 2.1.2 and 2.1.3 demonstrate from either side.

## 2.2 Why a Tile's Shape Specifically Has to Be Constant `[FOUNDATIONAL]`

### Intuition

A `Tile`'s shape isn't stored anywhere at runtime the way an `Array`'s shape is — there's no hidden `shape` field the hardware reads before deciding how many registers a tile occupies. The compiler has to know a tile's exact shape while it's still generating code, because that shape determines how many registers get allocated, how a `matmul` or `reduce` gets scheduled, and how a tile's memory access pattern gets laid out — all decisions made once, ahead of time, not re-made every time the kernel runs. `ct.load`'s `shape` argument is exactly the parameter that fixes a new tile's shape, so it's the one place this requirement is enforced most directly.

### Background

`ct.load`'s signature marks `shape` as `Constant[Shape]` — the same `ct.Constant` mechanism Section 2.1 covered, applied to a built-in operation instead of a user-defined kernel parameter. Passing a value the compiler can't resolve at compile time — a plain runtime scalar, unannotated — into that position produces a real compile-time rejection, not a shape determined "as late as possible" the way a NumPy array's shape can be.

### Worked Example 2.2.1 — a genuinely rejected runtime tile shape

```python
import cuda.tile as ct
import io

@ct.kernel
def bad_shape(a, c, tile_size):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ScalarConstraint(ct.int32)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(bad_shape, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

`04_nonconstant_tile_shape.py`, included in Complete Runnable Code:

```bash
python3 04_nonconstant_tile_shape.py
```

Genuinely run:

```
TileTypeError: Invalid argument "shape" of load(): Expected a constant integer tuple, but given value is not constant
  "/tmp/ch2/04_nonconstant_tile_shape.py", line 7, col 14-57, in bad_shape:
        a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Here `tile_size` is an ordinary parameter — no `ct.Constant` annotation at all, and its signature entry is a plain `ScalarConstraint`, so the compiler genuinely doesn't know its value ahead of time. `ct.load` names exactly which argument it's unhappy with (`"shape"`) and exactly what it needed (`a constant integer tuple`) — the same error shape Chapter 1's `TileTypeError` for a raw-tile `if` condition used, naming the argument and the expectation precisely rather than failing with a generic type mismatch.

## 2.3 Compile-Time Tuples: Multi-Dimensional Constants `[FOUNDATIONAL]`

### Intuition

A one-dimensional tile size is a single compile-time integer, but plenty of real kernels — anything working on a 2D array, like an image or a matrix — need a compile-time *pair* of integers instead, one per axis. `ConstantConstraint` alone isn't the right tool for that: it's built for a single scalar constant value, not a structured value like a tuple. `TupleConstraint` is the tool that says "this whole parameter is a fixed-length tuple, and here is a separate compile-time constraint for each of its elements" — genuinely a different constraint type, not just `ConstantConstraint` handed a tuple.

### Background

Passing a Python `tuple` directly to `ConstantConstraint` is rejected outright — it's built to hold one scalar constant value (an `int`, `float`, or `bool`), not a structured one. `TupleConstraint` takes a sequence of per-item constraints instead — in this case, one `ConstantConstraint` per axis of a 2D tile shape.

### Worked Example 2.3.1 — a genuine 2D compile-time tile shape

```python
import cuda.tile as ct
import io

@ct.kernel
def load_2d_tile(a, c, tile_shape: ct.Constant[tuple[int, int]]):
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    a_tile = ct.load(a, index=(pid0, pid1), shape=tile_shape)
    ct.store(c, index=(pid0, pid1), tile=a_tile)

def array_param(ndim):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

tile_shape_constraint = ct.compilation.TupleConstraint(
    [ct.compilation.ConstantConstraint(32), ct.compilation.ConstantConstraint(64)]
)
sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), tile_shape_constraint],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(load_2d_tile, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled with tile_shape embedded as (32, 64):", len(buf.getvalue()), "bytes")
```

`05_tuple_constant_2d_tile.py`, included in Complete Runnable Code:

```bash
python3 05_tuple_constant_2d_tile.py
```

Genuinely run:

```
compiled with tile_shape embedded as (32, 64): 730 bytes
```

`tile_shape` is a single Python parameter — one tuple, `(32, 64)` — but its `KernelSignature` entry is a `TupleConstraint` wrapping *two* separate `ConstantConstraint`s, one per element. `ct.load`'s `shape=tile_shape` then receives a fully compile-time-resolved 2-tuple, satisfying the same `Constant[Shape]` requirement Section 2.2 covered, just with two axes fixed instead of one.

> `[COMMON TRAP]` `ct.compilation.ConstantConstraint((32, 64))` — reaching straight for `ConstantConstraint` with a tuple argument, since that's what worked for the scalar `256` in Worked Example 2.1.1 — is a genuine, immediate `TypeError` (`Unexpected constant value type <class 'tuple'>`), raised the moment `ConstantConstraint` is constructed, before a `KernelSignature` or `export_kernel` call is even reachable. `ConstantConstraint` is specifically for one scalar constant; a structured compile-time value needs `TupleConstraint` (for tuples) or `ListConstraint` (for lists of arrays, covered when Part 1 builds on `ArrayConstraint` directly) instead.

## 2.4 One Kernel, Many Specializations `[FOUNDATIONAL]`

### Intuition

`export_kernel` in every example so far has taken a list containing exactly one `KernelSignature` — but it's a list for a real reason, not a stylistic artifact. A single cuTile Python kernel function is source code, not yet a compiled binary; compiling it against several signatures at once produces several genuinely distinct specializations — one exact fit for `tile_size=128`, another for `tile_size=256` — bundled together into a single output, the same way Chapter 1's multi-architecture export bundled `sm_80`, `sm_90`, and `sm_100` code paths from one Python function.

### Background

Each additional `KernelSignature` passed to `export_kernel` asks the compiler to produce one more real, independently-specialized version of the kernel, embedded into the same output buffer. This is genuinely additive work for the compiler — not a single generic binary that branches on `tile_size` at runtime, but multiple separately-compiled, constant-folded specializations, one per signature.

### Worked Example 2.4.1 — three tile sizes, one export call

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def sig_for(tile_size):
    return ct.compilation.KernelSignature(
        [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(tile_size)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

for sizes in ([128], [128, 256], [128, 256, 512]):
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig_for(s) for s in sizes], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print(f"{len(sizes)} signature(s) {sizes}: {len(buf.getvalue())} bytes")
```

`06_multi_signature_specialization.py`, included in Complete Runnable Code:

```bash
python3 06_multi_signature_specialization.py
```

Genuinely run:

```
1 signature(s) [128]: 728 bytes
2 signature(s) [128, 256]: 1165 bytes
3 signature(s) [128, 256, 512]: 1602 bytes
```

Every additional signature genuinely grows the output — each of the three runs compiles the exact same `vector_add` source, so the growth isn't new logic, it's a new, separately-specialized compilation of the same logic for a different constant-embedded `tile_size`. A host program shipping this kernel could pick whichever specialization matches the array size it actually has at launch time, without recompiling — the same portability story as Chapter 1's multi-architecture cubin, aimed at tile size instead of GPU generation.

### ASCII Diagram — one source, several bundled specializations

```
   vector_add(a, b, c, tile_size: ct.Constant[int])      <- one Python function
                        |
                        |  export_kernel(vector_add, [sig(128), sig(256), sig(512)], ...)
                        v
   +----------------------------------------------------------+
   |  compiled output (1602 bytes)                             |
   |  +------------------+  +------------------+  +----------+ |
   |  | tile_size = 128  |  | tile_size = 256  |  | ts = 512 | |
   |  | (separately      |  | (separately      |  | (separa- | |
   |  |  specialized)    |  |  specialized)    |  |  tely)   | |
   |  +------------------+  +------------------+  +----------+ |
   +----------------------------------------------------------+
```

## 2.5 Compilation Options: `opt_level` `[SUPPORTING]`

### Intuition

`@ct.kernel` isn't a bare decorator with no dials to turn — it's documented as accepting real compilation options, `opt_level` among them, the same idea as `rustc -O` or `nvcc -O3`: a knob that trades compile effort for the quality of what comes out the other end. What's worth confirming directly, rather than assuming from the name, is whether turning that knob actually changes anything measurable in this book's own environment — and, if it does, whether the relationship is as simple as "higher number, smaller or faster code" or something less tidy.

### Background

`ct.kernel`'s documented parameters include `opt_level`, an integer from 0 to 3 (default 3), alongside `num_ctas`, `occupancy`, and `num_worker_warps` — all real, documented tuning knobs this book does not exhaustively exercise here, since most of them only matter once real launch performance is measurable on real hardware. `opt_level` is the exception: it affects what `export_kernel` actually produces, ahead of time, with no GPU required to observe it.

### Worked Example 2.5.1 — four optimization levels, genuinely compiled

```python
import cuda.tile as ct
import io

def build_kernel(opt_level):
    @ct.kernel(opt_level=opt_level)
    def vector_add(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
        b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
        result = a_tile + b_tile
        ct.store(c, index=(pid,), tile=result)
    return vector_add

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

for level in (0, 1, 2, 3):
    kernel = build_kernel(level)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"opt_level={level}: {len(buf.getvalue())} cubin bytes")
```

`07_opt_level_comparison.py`, included in Complete Runnable Code:

```bash
python3 07_opt_level_comparison.py
```

Genuinely run:

```
opt_level=0: 29088 cubin bytes
opt_level=1: 24608 cubin bytes
opt_level=2: 24736 cubin bytes
opt_level=3: 24736 cubin bytes
```

Two things are genuinely true here, and only one of them is the "obvious" story. `opt_level=0` really does produce the largest binary of the four — 29,088 bytes, noticeably more than any optimized level, consistent with an unoptimized compile keeping more redundant instructions around. But the levels above 0 don't shrink monotonically the way "higher number, smaller output" would predict: `opt_level=1` is genuinely the *smallest* of the four (24,608 bytes), and `opt_level=2` and `opt_level=3` produce identical output for this particular kernel (24,736 bytes each). This book draws only the conclusion its own evidence supports — `opt_level` genuinely changes the compiled cubin for this kernel, and the difference between "unoptimized" and "optimized at all" is large and real — without extrapolating a general "3 is always best" rule this one kernel's numbers don't actually establish; a kernel with more arithmetic to fuse or more loops to unroll could easily show a different ordering.

## Complete Runnable Code

### File: `01_vector_add_with_constant.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled with tile_size embedded as 256:", len(buf.getvalue()), "bytes")
```

```bash
python3 01_vector_add_with_constant.py
```

### File: `02_constant_annotation_missing.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add_no_annotation(a, b, c, tile_size):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add_no_annotation, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except TypeError as e:
    print(f"{type(e).__name__}: {e}")
```

```bash
python3 02_constant_annotation_missing.py
```

### File: `03_constant_constraint_missing.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ScalarConstraint(ct.int32)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except TypeError as e:
    print(f"{type(e).__name__}: {e}")
```

```bash
python3 03_constant_constraint_missing.py
```

### File: `04_nonconstant_tile_shape.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def bad_shape(a, c, tile_size):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ScalarConstraint(ct.int32)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(bad_shape, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

```bash
python3 04_nonconstant_tile_shape.py
```

### File: `05_tuple_constant_2d_tile.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def load_2d_tile(a, c, tile_shape: ct.Constant[tuple[int, int]]):
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    a_tile = ct.load(a, index=(pid0, pid1), shape=tile_shape)
    ct.store(c, index=(pid0, pid1), tile=a_tile)

def array_param(ndim):
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

tile_shape_constraint = ct.compilation.TupleConstraint(
    [ct.compilation.ConstantConstraint(32), ct.compilation.ConstantConstraint(64)]
)
sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), tile_shape_constraint],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(load_2d_tile, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled with tile_shape embedded as (32, 64):", len(buf.getvalue()), "bytes")
```

```bash
python3 05_tuple_constant_2d_tile.py
```

### File: `06_multi_signature_specialization.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def sig_for(tile_size):
    return ct.compilation.KernelSignature(
        [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(tile_size)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

for sizes in ([128], [128, 256], [128, 256, 512]):
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig_for(s) for s in sizes], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print(f"{len(sizes)} signature(s) {sizes}: {len(buf.getvalue())} bytes")
```

```bash
python3 06_multi_signature_specialization.py
```

### File: `07_opt_level_comparison.py`

```python
import cuda.tile as ct
import io

def build_kernel(opt_level):
    @ct.kernel(opt_level=opt_level)
    def vector_add(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
        b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
        result = a_tile + b_tile
        ct.store(c, index=(pid,), tile=result)
    return vector_add

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

for level in (0, 1, 2, 3):
    kernel = build_kernel(level)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"opt_level={level}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 07_opt_level_comparison.py
```

## Chapter Summary

`ct.Constant` is a real, bidirectionally-enforced contract between a kernel's Python-level parameter annotation and the `KernelSignature` used to compile it — genuinely confirmed here from both directions, with a `TypeError` naming the exact mismatched parameter whether a signature supplies a `ConstantConstraint` for a parameter that was never annotated `Constant`, or a `ScalarConstraint` for one that was. That contract exists because a `Tile`'s shape has no runtime representation at all — `ct.load`'s `shape` argument specifically requires a compile-time constant tuple, and a genuine `TileTypeError` confirms exactly that when a plain runtime scalar reaches it instead. Structured compile-time values, like a 2D tile shape, need `TupleConstraint` wrapping one `ConstantConstraint` per element — `ConstantConstraint` alone rejects a bare Python tuple immediately, before a signature or an export call is ever reachable. A single kernel function isn't tied to one compiled specialization: `export_kernel` genuinely compiles several signatures from the same source into one bundled output, each one a real, independently-specialized version of the kernel for a different constant-embedded `tile_size` — the same kind of portability Chapter 1 demonstrated across GPU architectures, here demonstrated across tile sizes instead. And `@ct.kernel`'s `opt_level` argument genuinely changes what gets compiled, confirmed directly with real cubin byte counts across all four levels — with the honest caveat that the relationship isn't simply "higher is smaller": `opt_level=0` was clearly the largest of the four for this kernel, but `1` was smaller than both `2` and `3`, which compiled identically to each other.

## Self-Check Questions

1. Worked Examples 2.1.2 and 2.1.3 both raise a `TypeError` about `tile_size`, but for opposite reasons. What's the direction of the mismatch in each one?
2. Why does `ct.load`'s `shape` argument specifically require a compile-time constant, when other arguments to `ct.load` (like `index`) don't have to be?
3. `ct.compilation.ConstantConstraint((32, 64))` fails immediately with a `TypeError`. What's the actual fix for describing a compile-time 2-tuple, and why doesn't `ConstantConstraint` just accept a tuple directly?
4. When `export_kernel` is given a list of three `KernelSignature`s instead of one, what does the compiler actually produce — one generic kernel that branches on `tile_size` at runtime, or something else? What evidence from this chapter supports your answer?
5. Worked Example 2.5.1 found that `opt_level=1` produced a smaller cubin than both `opt_level=2` and `opt_level=3`. Is it safe to conclude "`opt_level=1` is generally the best choice for cuTile Python kernels" from this one result? Why or why not?

## Where We Go Next

This chapter treated `ArrayConstraint` as a fixed, unexamined ingredient in every `KernelSignature` — always the same four fields (`index_dtype`, `stride_lower_bound_incl`, `alias_groups`, `may_alias_internally`), repeated without asking what each one actually buys the compiler or costs a real kernel that violates its assumptions. Part 1 opens by taking `ArrayConstraint` apart field by field — what aliasing groups genuinely permit or forbid, what a wrong stride or shape assumption actually does when it's violated, and how the compile-time-hint fields (`stride_divisible_by`, `base_addr_divisible_by`, and others this chapter never touched) trade a stronger promise from the caller for real, measurable freedom in what the compiler is allowed to generate.

## Worked Solutions

**1.** Worked Example 2.1.2's `tile_size` carries no `ct.Constant` annotation in the Python source, but its `KernelSignature` entry is a `ConstantConstraint(256)` anyway — a signature claiming a fixed compile-time value for a parameter the kernel itself never promised to treat that way. Worked Example 2.1.3 is the mirror image: `tile_size: ct.Constant[int]` in the Python source promises a compile-time-fixed value, but its signature supplies a `ScalarConstraint(ct.int32)` — a genuine runtime value — reneging on that promise from the signature side instead.

**2.** A tile's shape has no runtime representation anywhere — unlike an `Array`, which genuinely carries runtime shape and stride information the compiler doesn't need in advance, a `Tile`'s shape has to be fully resolved before the compiler can decide how many registers it occupies, how operations on it get scheduled, and how it gets laid out in memory, all of which happen once, at compile time, not freshly on every kernel launch. `index`, by contrast, only says *which* tile to load — a genuinely runtime-varying position that changes on every block, with no effect on the shape or layout decisions the compiler has to make ahead of time.

**3.** The fix is `ct.compilation.TupleConstraint([ct.compilation.ConstantConstraint(32), ct.compilation.ConstantConstraint(64)])` — a `TupleConstraint` wrapping one `ConstantConstraint` per element of the tuple. `ConstantConstraint` doesn't accept a tuple directly because it's specifically built to describe one scalar constant value (an `int`, `float`, or `bool`); a structured value like a 2-tuple needs its own constraint type that itself holds a list of per-element constraints, which is exactly what `TupleConstraint` is for.

**4.** It produces several genuinely separate, independently-specialized compilations of the same source — not one generic kernel with a runtime branch. The evidence is the growing byte count in Worked Example 2.4.1: one signature compiled to 728 bytes, two signatures to 1,165 bytes, and three to 1,602 bytes — each additional signature added real, substantial size to the output, consistent with a fully separate specialization being compiled and embedded for each one, not a small runtime dispatch table added to a single shared body.

**5.** No — one kernel's byte counts at four optimization levels is not enough evidence to generalize a rule for "cuTile Python kernels" as a category. What this chapter's result actually supports is narrower and fully warranted by the evidence: for this specific `vector_add` kernel, in this book's own environment, `opt_level=0` produced clearly the largest binary, and `1` happened to produce a smaller one than `2` or `3`. A kernel with more to optimize — more arithmetic to fuse, loops to unroll, or redundant loads to eliminate — could easily show optimization levels ordered differently, or show real size reductions at higher levels that this small elementwise kernel simply has no room to benefit from.
