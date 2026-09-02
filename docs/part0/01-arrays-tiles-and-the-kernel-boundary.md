# Chapter 1: Arrays, Tiles, and the Kernel Boundary

> "A CUDA C++ kernel answers the question `threadIdx.x` puts in front of every line: what does *this one thread* do? A cuTile Python kernel is never asked that question — the compiler generates the threads for you, underneath a program that only ever talks about tiles. Learning cuTile Python honestly starts with learning exactly where that boundary sits, and refusing to pretend the thread grid CUDA C++ makes you type out by hand has actually gone away — it hasn't. It's just no longer yours to write."

**What you will understand by the end of this chapter:**

- What a `@ct.kernel` function actually is in cuTile Python — a Python object that cannot be called directly, only queued for execution over a grid with `ct.launch`, and what that boundary between "host code" and "tile code" really separates
- Why cuTile Python has two genuinely distinct data types where Triton has one: an `Array` (mutable, strided, lives in GPU global memory, supports almost nothing but `load`/`store`) and a `Tile` (immutable, a compile-time-fixed shape, and the type every arithmetic, reduction, and comparison operation actually works on)
- Why a kernel body is not quite Python — it's a restricted subset the compiler statically analyzes, and which real, genuinely-triggered compiler errors (`TileSyntaxError`, `TileTypeError`) reveal the boundaries of, shown here exactly as `cuda-tile` reports them
- Why `if a_tile > 0:` on a raw `Tile` is a compile error, but `if ct.sum(a_tile) > 0:` on the same tile compiles cleanly — the real distinction between a tile value and a scalar value that Python's own `if` statement forces the compiler to enforce
- How a kernel can be genuinely, verifiably compiled — to real cubin or real Tile IR bytecode, for a real named target architecture — without any NVIDIA GPU, driver, or `nvidia-smi` anywhere in the environment, which is what makes every "Worked Example" in this book real evidence rather than an assertion

**What you need to know first:**

- General Python fluency — functions, decorators, `if`/`while`/`for`, and basic NumPy-style array thinking. No prior GPU programming experience is assumed.
- If you've read this series' Triton edition, this chapter will feel structurally familiar and conceptually different at the same time. Triton also gives you a tile-shaped programming model instead of CUDA C++'s per-thread one — but Triton has exactly one array-like type (a "pointer plus block shape"), while cuTile Python has two: `Array` and `Tile`, kept deliberately distinct by the compiler's own type system. That distinction, introduced in this chapter, has no equivalent in the Triton book at all.
- No GPU content is silently assumed to "just work" here or anywhere else in this book. Every claim in this chapter that something compiles is something this book's own environment genuinely compiled; every claim that something would need real hardware to go further says so explicitly.

## 1.1 The Kernel Boundary: `@ct.kernel` and What Runs Where `[FOUNDATIONAL]`

### Intuition

Think of a `@ct.kernel` function the way you'd think of a sealed work order handed to a factory floor, not a phone call to one specific worker. You don't get to walk onto the floor and personally hand a task to worker #47 — you fill out the work order once, specifying what should happen to a batch of material, and a dispatcher (the grid) hands identical copies of that order to every station (every block) at once, each one stamped with its own station number. `@ct.kernel` marks a Python function as exactly that kind of work order: code destined to run once per block across a grid, never callable directly from ordinary Python, only ever queued for the whole grid at once through `ct.launch`.

### Background

A `@ct.kernel`-decorated function belongs to what cuTile Python's own documentation calls *tile code* — an execution space distinct from *host code*, the ordinary Python that sets arrays up and eventually launches kernels. A kernel function itself cannot be called like a normal Python function; attempting `vector_add(a, b, c, 256)` directly, instead of going through `ct.launch`, is not what this book's own examples do anywhere, precisely because it isn't how the language is meant to be used — kernels are entry points for a grid, not ordinary callables. Helper functions are different: an ordinary, undecorated Python function called *from inside* a kernel is automatically treated as tile code too, recursively, with no annotation required — this book confirms that directly in Section 1.3.

| | Host code (ordinary Python) | Tile code (`@ct.kernel` body) |
|---|---|---|
| Runs where | The CPU process that imports `cuda.tile` | Compiled ahead-of-time or JIT-specialized, then executed once per block on the GPU |
| Callable directly? | Yes — ordinary Python | No — only queued via `ct.launch` |
| Python subset | Full Python | A restricted subset the compiler statically analyzes (Section 1.3) |
| What values look like | NumPy/CuPy/PyTorch arrays, Python scalars | `Array` and `Tile` objects (Section 1.2) |

### Worked Example 1.1.1 — defining a kernel costs nothing GPU-related

```python
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

print(type(vector_add))
print(type(vector_add).__module__ + "." + type(vector_add).__qualname__)
```

The complete, genuinely run file is `01_vector_add_kernel.py`, shown in full in this chapter's Complete Runnable Code section.

```bash
python3 01_vector_add_kernel.py
```

Genuinely run:

```
<class 'cuda.tile._execution.kernel'>
cuda.tile._execution.kernel
```

`@ct.kernel` doesn't compile anything at the moment it's applied — it wraps `vector_add` in a `cuda.tile._execution.kernel` object and defers all real compilation to whenever that kernel is actually launched or, as this book relies on throughout, ahead-of-time exported. Nothing above touched a GPU, a driver, or the `tileiras` compiler at all; it's pure Python object construction, and it's genuinely reproducible in an environment with no NVIDIA hardware whatsoever, which is exactly what produced the output above.

`ct.bid(0)` — used inside the kernel body — reads the current block's index along grid axis 0, the tile-code equivalent of CUDA C++'s `blockIdx.x`, except that (as Section 1.4 shows) it hands back an ordinary-feeling integer that participates in real Python-level `if`/`while` control flow, not a value you thread through `threadIdx`-style arithmetic by hand.

## 1.2 Arrays and Tiles: Two Distinct Types, On Purpose `[FOUNDATIONAL]`

### Intuition

An `Array` in cuTile Python is a loading dock: a large, fixed structure sitting in global memory, addressed by strides, that you can put things into or take things out of — but you don't do arithmetic on the loading dock itself. A `Tile` is the pallet you actually carry into the workshop: a fixed-size, immutable batch of values you loaded off the dock, which you can add, multiply, compare, reduce, and transform freely — and when you're done, you carry a new pallet back out and set it down at a dock location with `store`. The dock and the pallet are never the same kind of object, and cuTile Python's compiler enforces that difference at every line, not just as a matter of style.

### Background

| | `Array` | `Tile` |
|---|---|---|
| Mutable? | Yes — `store` writes into it | No — every operation on a `Tile` produces a new `Tile` |
| Shape known at compile time? | No — shape and strides are runtime values | Yes — every `Tile`'s shape is a compile-time constant |
| Lives where | GPU global memory (or host memory for host-side use) | Registers/shared memory, for the duration of the kernel |
| What you can do with it | Almost nothing besides `ct.load` and `ct.store` | Arithmetic, comparisons, `ct.sum`/`ct.reduce`, `ct.matmul`, shape manipulation, and more |

`ct.load(array, index, shape)` is the only way a `Tile` comes into existence from an `Array`: it partitions the array into a grid of equally-sized tiles of the given `shape`, then returns the one tile at position `index` in that grid. `ct.store(array, index, tile)` is the reverse: it writes a `Tile`'s values back into the array at the tile-space position `index`. Both functions are genuinely documented as partitioning the array into what cuTile Python calls a *tile space* — for a 2D array of shape `(M, N)` loaded with tile shape `(tm, tn)`, the tile space itself has shape `(ceil(M/tm), ceil(N/tn))`, using the same ceiling-division helper, `ct.cdiv`, this book uses directly below.

### Worked Example 1.2.1 — `ct.cdiv` and the shape of a tile space, genuinely run on the host

`ct.cdiv` is explicitly documented as usable from host code, not just inside a kernel — so this is a real, host-only Python session, no kernel or compiler involved at all:

```python
import cuda.tile as ct

print(ct.cdiv(10, 4))
print(ct.cdiv(9, 4))
print(ct.cdiv(8, 4))
```

The complete file is `00_cdiv_host.py`, included in this chapter's Complete Runnable Code section.

```bash
python3 00_cdiv_host.py
```

Genuinely run:

```
3
3
2
```

An array of length 10, loaded with a tile size of 4, has a tile space of `ct.cdiv(10, 4) = 3` tiles along that axis: two full tiles of 4 elements and one partial tile of the remaining 2 — `ct.load`'s documentation is explicit that a tile straddling the array's boundary this way still comes back at the requested `shape`, with the out-of-bounds positions filled according to a `padding_mode` rather than shrunk to fit. A length of 8 divides evenly, giving exactly `ct.cdiv(8, 4) = 2` full tiles with no padding needed at all.

### ASCII Diagram — one array, partitioned into a tile space

```
Array `a`, length 10, tile_size = 4:

index:   0   1   2   3   4   5   6   7   8   9
        +---+---+---+---+---+---+---+---+---+---+
value:  | . | . | . | . | . | . | . | . | . | . |
        +---+---+---+---+---+---+---+---+---+---+
        \___________ tile 0 __________/\_____/
                (pid = 0)                 tile 1 continues...

Tile space along this axis: ct.cdiv(10, 4) = 3 tiles
  pid=0 -> ct.load(a, index=(0,), shape=(4,))  -> elements 0..3   (full tile)
  pid=1 -> ct.load(a, index=(1,), shape=(4,))  -> elements 4..7   (full tile)
  pid=2 -> ct.load(a, index=(2,), shape=(4,))  -> elements 8..9,  padded (partial tile)
```

`pid`, in this diagram and in Worked Example 1.1.1, is exactly `ct.bid(0)` — the grid dispatches one block per tile-space position, and each block's own `ct.bid(0)` call tells it which tile it's responsible for.

### Worked Example 1.2.2 — a genuinely compiled elementwise kernel

Extending Worked Example 1.1.1, `02_tile_arithmetic_export.py` (shown in full below) defines the same `vector_add` kernel and ahead-of-time compiles it — with no GPU present — to real Tile IR bytecode for a named target architecture:

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
print("vector_add compiled for sm_90:", len(buf.getvalue()), "bytes of Tile IR bytecode")
```

```bash
python3 02_tile_arithmetic_export.py
```

Genuinely run:

```
vector_add compiled for sm_90: 712 bytes of Tile IR bytecode
```

`a_tile + b_tile` is ordinary Python `+` syntax, but the values on both sides are `Tile` objects, not NumPy arrays or Python numbers — cuTile Python overloads elementwise arithmetic directly on `Tile`, and the compiler resolved that `+` into real, exportable Tile IR with no error. `ArrayConstraint` is how this book tells the compiler what to assume about each `Array` parameter's dtype, rank, and aliasing behavior before it can compile a kernel for a fixed signature at all — Chapter 6 covers every field of `ArrayConstraint` in depth; here, the four kept simplest (a non-negative-stride, non-aliasing, `float32`, rank-1 array with `int32` indices) are exactly enough to compile this kernel.

> `[COMMON TRAP]` The exact byte count above is not a portable fact about the `vector_add` kernel in the abstract — it's genuinely sensitive to the *exact source text* of the file being compiled. Re-running the identical logic copied into a differently-named file, or with different variable names or line breaks, produces a different byte count, because Tile IR bytecode embeds source-location debug information (the same information NVIDIA's own documentation says lets Nsight Compute profile a cuTile Python kernel the same way it profiles a SIMT CUDA kernel). This book always compiles the exact file shown in each chapter's Complete Runnable Code section from its own name, so the byte counts shown are genuinely reproducible — just not something you should expect to match if you retype the same kernel under a different filename.

## 1.3 The Restricted Python Subset `[FOUNDATIONAL]`

### Intuition

A kernel body looks like Python and uses Python's own keywords, but the compiler reading it is not a Python interpreter — it's a static compiler that has to fully resolve every value's type and every tile's shape before it can generate a single GPU instruction. Exception handling is built around the idea that control flow can be interrupted unpredictably at runtime; class definitions are built around the idea that new, arbitrary types can appear during execution. Both ideas are fundamentally incompatible with a compiler that must know everything about a kernel's structure ahead of time — so cuTile Python's compiler doesn't attempt either, and says so immediately and specifically when it encounters them, rather than silently miscompiling or falling back to something slower.

### Background

cuTile Python's own documentation describes the kernel body as operating over a restricted subset of Python, statically analyzed by the compiler into what its internals (visible directly in the error output below) call an HIR — a high-level intermediate representation — before any GPU code is generated at all. Two constructs this book genuinely confirmed are rejected outright:

| Construct | Genuinely rejected? | Real exception type |
|---|---|---|
| `try` / `except` | Yes | `TileSyntaxError` |
| `class` definitions | Yes | `TileSyntaxError` |
| Calling a plain, undecorated Python function from a kernel | No — genuinely compiles | — |
| A `for` loop bounded by a compile-time `ct.Constant` | No — genuinely compiles | — |
| A `while` loop bounded by a runtime scalar value | No — genuinely compiles | — |

The middle and bottom rows matter as much as the top two: this is not a language that only accepts a tiny, static-only fragment of Python. Ordinary helper functions and genuinely dynamic, runtime-value-dependent loop bounds both compile without complaint, as the next two Worked Examples show directly.

### Worked Example 1.3.1 — a genuinely rejected `try`/`except`

```python
import cuda.tile as ct
import io

@ct.kernel
def guarded_load(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    try:
        a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    except Exception:
        a_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile)

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
    ct.compilation.export_kernel(guarded_load, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

The complete file, `03_tryexcept_unsupported.py`, is not included in this chapter's Complete Runnable Code section, since it deliberately does not compile — it's shown here in full instead.

```bash
python3 03_tryexcept_unsupported.py
```

Genuinely run:

```
TileSyntaxError: Unsupported syntax
  "/tmp/ch1/03_tryexcept_unsupported.py", lines 7--10, col 5-8, in guarded_load:
        try:
        ^^^^
```

The compiler rejects `try` at the point it starts parsing the kernel's structure, before it even gets far enough to notice that the `try` body calls `ct.load` on a value the `except` branch also assigns — there's no runtime exception-handling path to compile to, so cuTile Python doesn't pretend one exists. (The absolute file path in the error is exactly what `cuda-tile`'s own compiler reports for whatever file it was given — this book's own build environment, not something edited out.)

### Worked Example 1.3.2 — a genuinely rejected `class` definition

```python
import cuda.tile as ct
import io

@ct.kernel
def defines_class(a, b, c, tile_size: ct.Constant[int]):
    class Scratch:
        pass
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile)

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
    ct.compilation.export_kernel(defines_class, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

Also not included in Complete Runnable Code, for the same reason — `04_class_def_unsupported.py`, shown here in full:

```bash
python3 04_class_def_unsupported.py
```

Genuinely run:

```
TileSyntaxError: Unsupported syntax
  "/tmp/ch1/04_class_def_unsupported.py", lines 6--7, col 5-18, in defines_class:
        class Scratch:
        ^^^^^^^^^^^^^^
```

Same exception type, same shape of error, for an unrelated construct — `TileSyntaxError` is cuTile Python's answer to any statement its compiler's frontend doesn't have a translation rule for at all, independent of what that statement would have tried to do at runtime.

### Worked Example 1.3.3 — what genuinely *does* compile: a plain helper function

`ct.function`'s own documentation states that an unannotated function called from tile code automatically becomes tile code too, recursively, with no explicit decorator required. `09_plain_helper_function.py`, included in this chapter's Complete Runnable Code, calls an ordinary, undecorated `elementwise_add` from inside a kernel:

```python
import cuda.tile as ct
import io

def elementwise_add(x, y):
    return x + y

@ct.kernel
def vector_add_via_helper(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = elementwise_add(a_tile, b_tile)
    ct.store(c, index=(pid,), tile=result)
```

```bash
python3 09_plain_helper_function.py
```

Genuinely run:

```
compiled: 826 bytes
```

`elementwise_add` carries no `@ct.function` or `@ct.kernel` decorator at all, and the compiler accepts the call from inside `vector_add_via_helper` without complaint — exactly what the documentation describes: calling an unannotated function from tile code recursively extends that function's own execution space to include tile code, with nothing further required at the call site.

### Worked Example 1.3.4 — a runtime-bounded loop

`07_scalar_dynamic_loop.py`, also in Complete Runnable Code, confirms the other half of Section 1.3's Background table: a `while` loop bounded by a genuine runtime scalar (not a compile-time constant) compiles cleanly too:

```bash
python3 07_scalar_dynamic_loop.py
```

Genuinely run:

```
compiled: 931 bytes
```

`accumulate_n_tiles`'s `while i < n:` loop is bounded by `n`, declared with `ct.compilation.ScalarConstraint(ct.int32)` — a true runtime value, unknown at compile time, not a `ct.Constant`. The compiler doesn't need the trip count fixed ahead of time to generate this loop; it only needs `n` to genuinely be a scalar, the distinction Section 1.4 covers next.

## 1.4 Scalars vs. Tiles in Control Flow `[FOUNDATIONAL]`

### Intuition

Every element of a `Tile` might satisfy `> 0` while others don't — asking "is this tile greater than zero" the way you'd ask "is this number greater than zero" doesn't have one true-or-false answer to give Python's `if` statement, which fundamentally needs exactly one. A single scalar value — one int, one float, one bool, extracted or reduced down from a tile or read directly as a kernel parameter — always has exactly one answer, which is exactly what `if` and `while` require. cuTile Python's compiler enforces this distinction as a real type error, not a warning, the moment a raw `Tile` reaches a context that demands a single boolean.

### Background

| | Raw `Tile` (e.g. `a_tile > 0`) | Scalar (e.g. `ct.sum(a_tile) > 0`, or a `ScalarConstraint` parameter) |
|---|---|---|
| Result of a comparison | A `Tile` of `bool_`, same shape as the input | A single scalar `bool` |
| Usable directly in `if`/`while`? | **No** — `TileTypeError` | Yes |
| How to get there from a `Tile` | Reduce it first (`ct.sum`, `ct.reduce`, and similar) | Already scalar |

### Worked Example 1.4.1 — a genuinely rejected tile condition

```python
import cuda.tile as ct
import io

@ct.kernel
def abs_by_raw_tile(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    if a_tile > 0:
        result = a_tile
    else:
        result = -a_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(abs_by_raw_tile, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print("compiled:", len(buf.getvalue()), "bytes")
except ct.TileError as e:
    print(f"{type(e).__name__}: {e}")
```

Not included in Complete Runnable Code, for the same reason as Section 1.3's two examples — `05_raw_tile_condition_error.py`, shown here in full:

```bash
python3 05_raw_tile_condition_error.py
```

Genuinely run:

```
TileTypeError: Expected a scalar value, but given value has type Tile[bool_,(256)]
  "/tmp/ch1/05_raw_tile_condition_error.py", line 8, col 8-17, in abs_by_raw_tile:
        if a_tile > 0:
           ^^^^^^^^^^
```

The error names the exact type it received — `Tile[bool_,(256)]`, a 256-element tile of booleans, one per element of `a_tile` — and states plainly what it needed instead: a scalar. Nothing about the *value* of `a_tile` matters here; this is a compile-time rejection based purely on shape and type, before any data exists at all.

### Worked Example 1.4.2 — the fix: reduce to a scalar first

```python
import cuda.tile as ct
import io

@ct.kernel
def abs_by_sum_gate(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    total = ct.sum(a_tile)
    if total > 0:
        result = a_tile
    else:
        result = -a_tile
    ct.store(c, index=(pid,), tile=result)
```

The complete file — with the same `ArrayConstraint`/`KernelSignature`/`export_kernel` scaffolding Worked Example 1.2.2 introduced — is `06_sum_condition_fixed.py`, shown in full further below in Complete Runnable Code.

```bash
python3 06_sum_condition_fixed.py
```

Genuinely run:

```
compiled: 798 bytes
```

`ct.sum(a_tile)` with no `axis` argument reduces every element of the tile down to one value — genuinely accepted directly by `if total > 0:` with no further conversion. (This chapter uses this pattern purely to demonstrate the scalar/tile distinction; using a whole tile's sum to decide which branch every element of the *same* tile takes isn't a realistic elementwise kernel — Part 2's reduction chapter builds `ct.sum` and `ct.reduce` into genuinely useful kernels instead.)

> `[COMMON TRAP]` It's tempting to assume any reduction automatically produces something "scalar enough" for control flow, but that's not the rule the compiler enforces — the rule is specifically about the value's resolved type being a scalar, not a `Tile` of any shape, including a 1-element one. `ct.sum(a_tile)` with `axis=None` (the default, used above) is genuinely documented as reducing *all* of a tile's elements, which is what makes its result usable this way in this book's own genuinely-run example — a partial reduction with an explicit `axis` on a multi-dimensional tile still returns a `Tile`, not a scalar, and would need a further reduction or an explicit extraction before any `if` could accept it.

## 1.5 Verifying a Kernel Without a GPU: Ahead-of-Time Compilation `[FOUNDATIONAL]`

### Intuition

CUDA C++'s `nvcc` has always been able to compile a `.cu` file to a real binary without a GPU plugged into the machine doing the compiling — the GPU is only needed later, to actually run what got compiled. cuTile Python inherits exactly that separation, and this book leans on it directly: `ct.compilation.export_kernel` resolves a kernel's real, compiler-checked structure — including every `TileSyntaxError` and `TileTypeError` shown so far in this chapter — into a genuine cubin or Tile IR bytecode file for a named target architecture, with no driver, no `nvidia-smi`, and no GPU anywhere in the process. Only `ct.launch`, which this book has not called anywhere in this chapter, actually needs live hardware.

### Background

| Step | Needs a real GPU/driver? | What this book does with it |
|---|---|---|
| `@ct.kernel` definition | No | Every kernel in this chapter |
| `ct.compilation.export_kernel(...)` | No | Every *successful* compile in this chapter, genuinely run |
| `ct.launch(stream, grid, kernel, args)` | Yes — needs a real CUDA stream | Not attempted anywhere in this book's own environment; explicitly tagged **UNVERIFIED — pending real-GPU test** wherever it would matter |

### Worked Example 1.5.1 — one kernel, three real target architectures

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

for arch in ("sm_80", "sm_90", "sm_100"):
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code=arch, output_format="cubin")
    print(f"{arch}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 08_export_multi_arch.py
```

Genuinely run:

```
sm_80: 23856 cubin bytes
sm_90: 24736 cubin bytes
sm_100: 30768 cubin bytes
```

Three real, differently-sized cubin binaries — Ampere/Ada (`sm_80`), Hopper (`sm_90`), and Blackwell (`sm_100`) — compiled from the exact same Python function, in the exact same process, with no GPU present at all. `output_format="cubin"` produces a finished CUDA binary; swapping it for `"tileir_bytecode"`, as Worked Example 1.2.2 did, produces NVIDIA's own intermediate representation instead — both genuinely exercise the same real compiler this book's `pip install "cuda-tile[tileiras]"` installed with no CUDA Toolkit and no driver required, as [Getting Started](../getting-started.md) covers.

## Complete Runnable Code

### File: `00_cdiv_host.py`

```python
import cuda.tile as ct

print(ct.cdiv(10, 4))
print(ct.cdiv(9, 4))
print(ct.cdiv(8, 4))
```

```bash
python3 00_cdiv_host.py
```

### File: `01_vector_add_kernel.py`

```python
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

print(type(vector_add))
print(type(vector_add).__module__ + "." + type(vector_add).__qualname__)
```

```bash
python3 01_vector_add_kernel.py
```

### File: `02_tile_arithmetic_export.py`

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
print("vector_add compiled for sm_90:", len(buf.getvalue()), "bytes of Tile IR bytecode")
```

```bash
python3 02_tile_arithmetic_export.py
```

### File: `03_tryexcept_unsupported.py`

This file is intentionally not included here in runnable form — it exists only to demonstrate a compile-time rejection, and is shown in full inside this chapter's Worked Example 1.3.1.

### File: `04_class_def_unsupported.py`

Also intentionally not included here in runnable form, for the same reason — shown in full inside Worked Example 1.3.2.

### File: `05_raw_tile_condition_error.py`

Also intentionally not included here in runnable form, for the same reason — shown in full inside Worked Example 1.4.1.

### File: `06_sum_condition_fixed.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def abs_by_sum_gate(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    total = ct.sum(a_tile)
    if total > 0:
        result = a_tile
    else:
        result = -a_tile
    ct.store(c, index=(pid,), tile=result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(abs_by_sum_gate, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 06_sum_condition_fixed.py
```

### File: `07_scalar_dynamic_loop.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def accumulate_n_tiles(a, b, c, n, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    acc = ct.load(a, index=(pid,), shape=(tile_size,))
    i = 0
    while i < n:
        acc = acc + ct.load(b, index=(pid,), shape=(tile_size,))
        i = i + 1
    ct.store(c, index=(pid,), tile=acc)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(),
     ct.compilation.ScalarConstraint(ct.int32),
     ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

buf = io.BytesIO()
ct.compilation.export_kernel(accumulate_n_tiles, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 07_scalar_dynamic_loop.py
```

### File: `08_export_multi_arch.py`

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

for arch in ("sm_80", "sm_90", "sm_100"):
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code=arch, output_format="cubin")
    print(f"{arch}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 08_export_multi_arch.py
```

### File: `09_plain_helper_function.py`

```python
import cuda.tile as ct
import io

def elementwise_add(x, y):
    return x + y

@ct.kernel
def vector_add_via_helper(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = elementwise_add(a_tile, b_tile)
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
ct.compilation.export_kernel(vector_add_via_helper, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print("compiled:", len(buf.getvalue()), "bytes")
```

```bash
python3 09_plain_helper_function.py
```

```bash
python3 08_export_multi_arch.py
```

## Chapter Summary

A `@ct.kernel` function is a work order for a whole grid, not an ordinary callable — defining one costs nothing GPU-related, and it can only ever be executed by queuing it, grid-wide, through `ct.launch`. Inside that kernel, cuTile Python enforces a real, compiler-checked distinction Triton doesn't have: an `Array` is the mutable, strided, global-memory structure you `load` tiles from and `store` tiles into, while a `Tile` is the immutable, compile-time-shaped value every arithmetic operation, reduction, and comparison actually operates on — genuinely demonstrated here by `ct.cdiv` computing a real tile-space size on the host, with no compiler involved at all, and by a real elementwise kernel compiling cleanly once its `Array` parameters' constraints are declared. The kernel body itself is a restricted subset of Python, not full Python — `try`/`except` and `class` definitions are both genuinely rejected with a `TileSyntaxError` naming the exact unsupported statement and its location, while an ordinary helper function and a `while` loop bounded by a genuine runtime scalar both compile without complaint, so the restriction is specific, not a blanket ban on dynamism. That specificity extends to control flow directly: a raw `Tile`'s comparison result cannot drive an `if` or `while` — genuinely rejected as a `TileTypeError` naming the tile's exact shape and dtype — until it's reduced to an actual scalar with something like `ct.sum`. Every one of this chapter's successful compiles ran through `ct.compilation.export_kernel`, cuTile Python's ahead-of-time path, which resolves a kernel to real cubin or Tile IR bytecode for a named target architecture with no GPU, driver, or `nvidia-smi` present anywhere — genuinely demonstrated compiling the same kernel for Ampere/Ada, Hopper, and Blackwell in a single run. Only `ct.launch`, against a real CUDA stream, actually needs hardware this book's own environment doesn't have.

## Self-Check Questions

1. Why can't a `@ct.kernel`-decorated function be called directly, the way an ordinary Python function can? What has to happen to it instead?
2. An `Array` and a `Tile` both hold numeric data, but the compiler treats them completely differently. Name two concrete differences between them, and say which one you'd use to accumulate a running elementwise result inside a kernel.
3. `try`/`except` and `class` definitions are both rejected with a `TileSyntaxError`, while a `while` loop bounded by a runtime scalar compiles cleanly. What distinguishes the constructs that get rejected from the constructs that don't?
4. `if a_tile > 0:` fails to compile with a `TileTypeError` naming `Tile[bool_,(256)]`, but `if ct.sum(a_tile) > 0:` on the same `a_tile` compiles cleanly. Explain, precisely, what changed between the two.
5. This chapter compiled every kernel with `ct.compilation.export_kernel` rather than `ct.launch`. What does each of those two calls actually require from the environment, and why does only one of them work here?

## Where We Go Next

This chapter established the two types every later chapter builds on — `Array` and `Tile` — and the one boundary every kernel in this book respects: host code sets arrays up and launches or exports kernels, tile code is a restricted, statically-analyzed subset of Python that operates on tiles and scalars. Chapter 2 stays inside that same boundary and looks at kernel parameterization in depth: what exactly `ct.Constant` buys a kernel at compile time, how a tile's shape has to be known before the compiler can generate a single instruction for it, and why a kernel's real, exportable signature — the same `KernelSignature`/`ArrayConstraint` machinery this chapter used to compile `vector_add` — is itself something this book treats as genuinely tested surface area, not paperwork bolted on afterward.

## Worked Solutions

**1.** A `@ct.kernel` function belongs to *tile code*, an execution space the documentation describes as reachable only by being queued for a whole grid — it isn't a plain callable the way an undecorated Python function is. Calling it has to go through `ct.launch(stream, grid, kernel, kernel_args)`, which dispatches the kernel once per block across the given grid; there is no direct-call path around that, because a kernel is meant to describe what one block does, not what one Python-level invocation does.

**2.** Two concrete differences: an `Array`'s shape and strides are runtime values the compiler doesn't need to know ahead of time, while every `Tile`'s shape is fixed at compile time; and an `Array` is mutable in place (`ct.store` writes into it) while every operation on a `Tile` produces a brand-new `Tile` rather than mutating one. To accumulate a running elementwise result inside a kernel, you'd use a `Tile` — Worked Example 1.3.4's `accumulate_n_tiles` does exactly this, reassigning `acc` to a new `Tile` on every loop iteration, never mutating an `Array` in the loop body itself.

**3.** `try`/`except` and `class` are both genuinely rejected by the compiler's frontend — Worked Examples 1.3.1 and 1.3.2 show the identical `TileSyntaxError` shape for both, naming the exact unsupported statement. What distinguishes them from the accepted constructs isn't "how dynamic they look" in the abstract: a runtime-scalar-bounded `while` loop is genuinely dynamic (the trip count isn't known until the kernel actually runs) and still compiles, because the compiler can generate a real loop with a runtime-checked condition for it. `try`/`except` and `class` don't have any equivalent lowering at all in this language — there's no GPU-side notion of an exception to unwind to, or a new type to define at kernel-compile time — so the compiler doesn't attempt one.

**4.** `a_tile > 0` compares every element of `a_tile` against zero independently, producing a new `Tile` of booleans the same shape as `a_tile` — `Tile[bool_,(256)]`, exactly as the genuine error names it. Python's `if` statement needs exactly one true-or-false answer, which a 256-element tile of independent answers can't provide, so the compiler rejects it as a type error rather than guessing which element (or some combination of all of them) the programmer meant. `ct.sum(a_tile)` with no `axis` argument reduces all 256 of those elements down to a single value first; `ct.sum(a_tile) > 0` then compares that one scalar against zero, producing a genuine scalar `bool` `if` can consume directly.

**5.** `ct.compilation.export_kernel` only needs the `tileiras` compiler — a set of pure-Python wheels this book installed with `pip install "cuda-tile[tileiras]"` and no CUDA Toolkit, no NVIDIA driver, and no GPU at all — because compiling a kernel to cubin or Tile IR bytecode for a named target architecture is fundamentally an offline, ahead-of-time step, the same way `nvcc` can compile a `.cu` file without a GPU plugged into the compiling machine. `ct.launch`, by contrast, actually dispatches a compiled kernel onto a live device over a real CUDA stream — it needs a genuine NVIDIA GPU, its driver, and typically a library like CuPy or PyTorch to construct that stream, none of which exist in this book's own build environment, which is exactly why every successful compile in this chapter used `export_kernel` and no call to `ct.launch` appears anywhere in it.
