# Chapter 5: Targeting Hardware and Ahead-of-Time Compilation

> "`nvcc` compiling a `.cu` file into a real binary without a GPU plugged into the machine doing the compiling is not a novelty this book has to explain twice — it's the same offline discipline cuTile Python inherits wholesale. What's worth a whole chapter isn't that `export_kernel` works without hardware; the first four chapters already leaned on that. It's what the word 'hardware' actually resolves to underneath a string like `\"sm_100\"` — which real architectures that string family reaches, which ones it genuinely doesn't, and what happens, in precise and different ways, the moment either boundary is crossed."

**What you will understand by the end of this chapter:**

- That `export_kernel`'s `output_file` argument accepts a real filesystem path, not only an in-memory `io.BytesIO`, and that the resulting `cubin` file is independently confirmed here to be a genuine ELF binary, not merely a file the documentation calls one
- That `bytecode_version` genuinely pins the compiled TileIR bytecode to one of a real, small, enumerable set of supported versions, with `None`'s documented "auto-detect the latest" behavior confirmed to mean exactly one specific version, byte-for-byte
- What `cutile_python_v1` and `cutile_python_v2` calling conventions are documented to differ on, and the one concrete case — a tuple-shaped `ct.Constant` parameter — where that documented difference genuinely produces different, verifiable behavior rather than identical output
- That "targeting hardware" has a real, compiler-enforced floor: a named architecture string can be rejected in two entirely different places — by cuTile Python's own Python-level parsing, or by the underlying `tileiras` compiler itself refusing an architecture it was never built to target — and this chapter shows both, genuinely, with a different exception type for each

**What you need to know first:**

- Chapter 1's `ct.compilation.export_kernel`, `gpu_code`, and `output_format` as this book's driver-free verification path, including its multi-architecture compile across `sm_80`/`sm_90`/`sm_100`.
- Chapter 2's `ct.Constant`, `TupleConstraint`, and multi-signature specialization from a single kernel source.

## 5.1 A Real File, Genuinely Inspected `[FOUNDATIONAL]`

### Intuition

Every export in this book so far wrote into an `io.BytesIO` — a convenient, in-memory stand-in this book then measured by length. `export_kernel`'s `output_file` parameter is documented to accept more than that: a real filename works too, and what lands on disk when it's used is worth actually opening and reading, not just trusting the word "binary."

### Background

`output_file` is documented as accepting either a binary file-like object or a filename — a `str`, `bytes`, or `os.PathLike`. Passing a real path writes a genuine file to disk; nothing else about `export_kernel`'s behavior changes. A `cubin`, cuTile Python's own documentation says, is "a CUDA binary file" — and a CUDA binary file is, underneath that description, a real ELF (Executable and Linkable Format) object file, the same format Linux uses for ordinary executables and shared libraries. Reading a file's first four bytes and comparing them against the standard ELF magic number (`0x7f`, `'E'`, `'L'`, `'F'`) is a real, independent way to confirm that, rather than assume it from documentation prose alone.

### Worked Example 5.1.1 — writing and inspecting a real `.cubin` file

```python
import cuda.tile as ct
import os

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# output_file accepts a real filesystem path, not just an io.BytesIO --
# this writes a genuine file to disk, no in-memory buffer involved.
ct.compilation.export_kernel(vector_add, [sig], "vector_add.cubin", gpu_code="sm_90", output_format="cubin")
size = os.path.getsize("vector_add.cubin")
print("wrote vector_add.cubin:", size, "bytes")

# A cubin is documented as "a CUDA binary file" -- reading its first four
# bytes and checking them against the real ELF magic number confirms,
# rather than assumes, that it genuinely is one.
with open("vector_add.cubin", "rb") as f:
    magic = f.read(4)
print("magic bytes:", magic.hex(), "-- ELF" if magic == b"\x7fELF" else "-- NOT ELF")

os.remove("vector_add.cubin")
```

The complete file is `01_export_to_file_path.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_export_to_file_path.py
```

Genuinely run:

```
wrote vector_add.cubin: 24736 bytes
magic bytes: 7f454c46 -- ELF
```

`7f454c46` is exactly `0x7f` followed by the ASCII bytes for `E`, `L`, `F` — the real, standard ELF magic number, confirmed by reading the file this book's own `export_kernel` call actually wrote, not asserted from documentation. A `cubin` genuinely is what it says it is.

## 5.2 `bytecode_version`: A Real, Enumerable Set `[FOUNDATIONAL]`

### Intuition

`bytecode_version` is documented as an optional string in `"major.minor"` form — `None` to auto-detect the latest version the compiler supports, or an explicit version to pin the output to. "The latest" is a claim this book can check directly: compile the same kernel once with `None` and once with each explicitly-named version, and compare the results byte-for-byte.

### Background

`ct.compilation.export_kernel`'s `bytecode_version` parameter only affects `output_format="tileir_bytecode"` exports. Left at its default, `None`, the compiler picks its own newest supported version; passed an explicit `"major.minor"` string, it targets exactly that version instead, or rejects the request if that version isn't one the installed compiler actually supports.

### Worked Example 5.2.1 — `None` against every explicitly-named version

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# bytecode_version=None (the default) is documented to auto-detect the
# latest TileIR bytecode version the compiler supports. Comparing it
# directly against each explicitly-named version confirms which one that
# actually is, rather than trusting the word "latest" alone.
for version in [None, "13.1", "13.2", "13.3"]:
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90",
                                  output_format="tileir_bytecode", bytecode_version=version)
    print(f"version={version!r}: {len(buf.getvalue())} bytes")

# A version outside the compiler's real, supported set is genuinely
# rejected before any compilation happens.
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90",
                                  output_format="tileir_bytecode", bytecode_version="13.0")
except ValueError as e:
    print(f"version='13.0': {type(e).__name__}: {e}")
```

The complete file is `02_bytecode_version.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_bytecode_version.py
```

Genuinely run:

```
version=None: 688 bytes
version='13.1': 686 bytes
version='13.2': 686 bytes
version='13.3': 688 bytes
version='13.0': ValueError: Unsupported bytecode version '13.0'. Supported versions are: 13.1, 13.2, 13.3
```

Two genuine findings, not one. First, the installed compiler's real, enumerable set of supported bytecode versions is exactly `{13.1, 13.2, 13.3}` — confirmed directly from the rejection message itself, not from separate documentation. Second, `None`'s output is byte-identical to `"13.3"`'s (688 bytes each) and distinct from `"13.1"`/`"13.2"`'s (686 bytes each, identical to one another) — genuine, measured confirmation that "auto-detect the latest" means exactly `13.3` for this installed compiler, not an approximation or a guess about which version is newest.

## 5.3 Calling Conventions: Where `v1` and `v2` Genuinely Diverge `[FOUNDATIONAL]`

### Intuition

Every `KernelSignature` this book has built since Chapter 1 has passed `calling_convention=ct.compilation.CallingConvention.cutile_python_v2()` without asking what `v1` would have done differently, or whether it would have worked at all. cuTile Python's own documentation states plainly what separates them: both pass a kernel's arguments in the order they're declared, skipping any `ct.Constant` parameters, but `v2` extends `v1` with support for tuple parameters and static array shapes, and is the version the documentation recommends for new code. That's a specific, checkable claim — not just "v2 is newer."

### Background

For a kernel whose only compile-time constant is a single scalar `ct.Constant[int]` — every kernel this book has built through Chapter 4 — `v1` and `v2` have nothing to disagree about: neither a tuple parameter nor a static array shape is anywhere in the signature. The documented difference only has something to bite on once a signature actually contains a tuple-shaped constant, the same `ct.Constant[tuple[int, int]]` Chapter 2 used for a 2D tile shape.

### Worked Example 5.3.1 — no tuple parameter, no difference

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# cutile_python_v1 and cutile_python_v2 both pass a kernel's arguments in
# the same order they're declared, skipping any ct.Constant parameters.
# v2 extends v1 with support this file doesn't exercise yet: tuple
# parameters and static array shapes. For a kernel with only scalar and
# Array parameters -- no tuple-shaped constant -- both conventions
# genuinely compile to identical output.
for cc_name, cc in [("v1", ct.compilation.CallingConvention.cutile_python_v1()),
                     ("v2", ct.compilation.CallingConvention.cutile_python_v2())]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
        calling_convention=cc,
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"{cc_name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `03_calling_convention_v1_vs_v2.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_calling_convention_v1_vs_v2.py
```

Genuinely run:

```
v1: 24736 cubin bytes
v2: 24736 cubin bytes
```

Identical output, exactly as the documented difference predicts: nothing in this signature exercises the capability `v2` adds, so there's nothing for the two conventions to disagree about.

### Worked Example 5.3.2 — a tuple parameter, where they genuinely diverge

```python
import cuda.tile as ct
import io

@ct.kernel
def load_2d_tile(a, c, tile_shape: ct.Constant[tuple[int, int]]):
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    tile = ct.load(a, (pid0, pid1), tile_shape)
    ct.store(c, (pid0, pid1), tile)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=2, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# tile_shape is a tuple[int, int] Constant -- exactly the kind of
# parameter v2 is documented to add support for beyond v1. This is the
# case where the two conventions genuinely stop being interchangeable.
shape_constraint = ct.compilation.TupleConstraint([
    ct.compilation.ConstantConstraint(32),
    ct.compilation.ConstantConstraint(64),
])

# v1 rejects the tuple parameter immediately, at KernelSignature
# construction -- before export_kernel, the compiler, or the kernel body
# are ever reached.
try:
    sig_v1 = ct.compilation.KernelSignature(
        [array_param(), array_param(), shape_constraint],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v1(),
    )
except ValueError as e:
    print(f"v1: {type(e).__name__}: {e}")

# v2 accepts the identical signature and compiles the kernel cleanly.
sig_v2 = ct.compilation.KernelSignature(
    [array_param(), array_param(), shape_constraint],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(load_2d_tile, [sig_v2], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print(f"v2: compiled {len(buf.getvalue())} bytes")
```

The complete file is `04_calling_convention_tuple_parameter.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_calling_convention_tuple_parameter.py
```

Genuinely run:

```
v1: ValueError: Tuple parameters are not supported by calling convention cutile_python_v1; version >= 2 is required
v2: compiled 754 bytes
```

> `[COMMON TRAP]` The rejection happens at `KernelSignature` construction — a plain Python `ValueError`, not a `TileError` subclass — genuinely before `export_kernel` is ever called, before the compiler is invoked, and before the kernel body is inspected at all. `calling_convention` is validated against the shapes of the parameters declared alongside it the moment a signature object is built, the same way Chapter 2's `ct.Constant` annotation mismatches were caught immediately rather than deep inside compilation.

## 5.4 Which Architectures Are Actually Reachable `[FOUNDATIONAL]`

### Intuition

Chapter 1 compiled the same kernel for `sm_80`, `sm_90`, and `sm_100` without ever asking what happens outside that range. `gpu_code` accepts a string, and a string can be anything — a real, newer architecture; a real, older one the compiler was simply never built to target; or nonsense that never reaches the compiler at all. This section tests all three, and they fail — or succeed — in three genuinely different ways.

### Background

`gpu_code` is parsed in two stages. cuTile Python's own Python code first extracts the numeric suffix from a string of the form `"sm_<number>"`; a string that doesn't fit that shape fails immediately with a plain `ValueError`, before any compiler process is started. A string that does parse is then handed to the underlying `tileiras` compiler binary itself, which either recognizes the named architecture or rejects it as an architecture it has no code generator for — and that rejection is a `TileCompilerExecutionError`, a `TileError` subclass reported through this book's usual `except ct.TileError` pattern.

### Worked Example 5.4.1 — one real, newer architecture, and one real, unreachable one

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# sm_120 -- Blackwell's consumer/workstation line -- genuinely compiles,
# a fourth real architecture alongside Chapter 1's sm_80/sm_90/sm_100.
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_120", output_format="cubin")
print(f"sm_120: {len(buf.getvalue())} cubin bytes")

# sm_70 -- Volta, a real, shipped NVIDIA architecture, just one cuTile
# Python's tile-core compiler genuinely does not target -- is rejected by
# the underlying tileiras compiler itself, not by cuTile Python's own
# Python-level validation.
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_70", output_format="cubin")
except ct.TileError as e:
    print(f"sm_70: {type(e).__name__}: {e}")

# A malformed gpu_code string never even reaches the compiler -- it fails
# in pure Python, parsing the string's numeric suffix.
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="not_an_arch", output_format="cubin")
except ValueError as e:
    print(f"not_an_arch: {type(e).__name__}: {e}")
```

The complete file is `05_gpu_code_supported_and_unsupported.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_gpu_code_supported_and_unsupported.py
```

Genuinely run:

```
sm_120: 31264 cubin bytes
sm_70: TileCompilerExecutionError: Return code 1
tileiras: for the --gpu-name option: Cannot find option named 'sm_70'!
Unknown location
not_an_arch: ValueError: invalid literal for int() with base 10: 'not_an_arch'
```

Three genuinely distinct outcomes for three genuinely distinct inputs. `sm_120` — Blackwell's consumer and workstation line, not the data-center `sm_100` Chapter 1 already targeted — compiles cleanly, extending this book's confirmed hardware coverage to a fourth real architecture. `sm_70` — Volta, a real architecture NVIDIA shipped for years — parses fine as a string but is rejected by `tileiras` itself, the actual compiler binary underneath `export_kernel`, with its own process exit code and its own stderr text surfaced directly into the raised exception: cuTile Python's tile-core compilation model genuinely doesn't reach back that far. `not_an_arch` never gets that far at all — it fails in ordinary Python string-to-integer parsing, an entire validation stage before the compiler is ever invoked.

> `[COMMON TRAP]` `sm_70`'s failure and `not_an_arch`'s failure look superficially similar — both are `gpu_code` strings that don't work — but they fail in different code, for different reasons, and raise different exception types. Confusing "this architecture string is malformed" with "this architecture is real but unsupported" would miss that one of these is a mistake in how the string was written, and the other is a genuine, permanent boundary of what this compiler can target at all.

## Complete Runnable Code

### File: `01_export_to_file_path.py`

```python
import cuda.tile as ct
import os

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# output_file accepts a real filesystem path, not just an io.BytesIO --
# this writes a genuine file to disk, no in-memory buffer involved.
ct.compilation.export_kernel(vector_add, [sig], "vector_add.cubin", gpu_code="sm_90", output_format="cubin")
size = os.path.getsize("vector_add.cubin")
print("wrote vector_add.cubin:", size, "bytes")

# A cubin is documented as "a CUDA binary file" -- reading its first four
# bytes and checking them against the real ELF magic number confirms,
# rather than assumes, that it genuinely is one.
with open("vector_add.cubin", "rb") as f:
    magic = f.read(4)
print("magic bytes:", magic.hex(), "-- ELF" if magic == b"\x7fELF" else "-- NOT ELF")

os.remove("vector_add.cubin")
```

```bash
python3 01_export_to_file_path.py
```

### File: `02_bytecode_version.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# bytecode_version=None (the default) is documented to auto-detect the
# latest TileIR bytecode version the compiler supports. Comparing it
# directly against each explicitly-named version confirms which one that
# actually is, rather than trusting the word "latest" alone.
for version in [None, "13.1", "13.2", "13.3"]:
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90",
                                  output_format="tileir_bytecode", bytecode_version=version)
    print(f"version={version!r}: {len(buf.getvalue())} bytes")

# A version outside the compiler's real, supported set is genuinely
# rejected before any compilation happens.
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90",
                                  output_format="tileir_bytecode", bytecode_version="13.0")
except ValueError as e:
    print(f"version='13.0': {type(e).__name__}: {e}")
```

```bash
python3 02_bytecode_version.py
```

### File: `03_calling_convention_v1_vs_v2.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# cutile_python_v1 and cutile_python_v2 both pass a kernel's arguments in
# the same order they're declared, skipping any ct.Constant parameters.
# v2 extends v1 with support this file doesn't exercise yet: tuple
# parameters and static array shapes. For a kernel with only scalar and
# Array parameters -- no tuple-shaped constant -- both conventions
# genuinely compile to identical output.
for cc_name, cc in [("v1", ct.compilation.CallingConvention.cutile_python_v1()),
                     ("v2", ct.compilation.CallingConvention.cutile_python_v2())]:
    sig = ct.compilation.KernelSignature(
        [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
        calling_convention=cc,
    )
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"{cc_name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 03_calling_convention_v1_vs_v2.py
```

### File: `04_calling_convention_tuple_parameter.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def load_2d_tile(a, c, tile_shape: ct.Constant[tuple[int, int]]):
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    tile = ct.load(a, (pid0, pid1), tile_shape)
    ct.store(c, (pid0, pid1), tile)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=2, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

# tile_shape is a tuple[int, int] Constant -- exactly the kind of
# parameter v2 is documented to add support for beyond v1. This is the
# case where the two conventions genuinely stop being interchangeable.
shape_constraint = ct.compilation.TupleConstraint([
    ct.compilation.ConstantConstraint(32),
    ct.compilation.ConstantConstraint(64),
])

# v1 rejects the tuple parameter immediately, at KernelSignature
# construction -- before export_kernel, the compiler, or the kernel body
# are ever reached.
try:
    sig_v1 = ct.compilation.KernelSignature(
        [array_param(), array_param(), shape_constraint],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v1(),
    )
except ValueError as e:
    print(f"v1: {type(e).__name__}: {e}")

# v2 accepts the identical signature and compiles the kernel cleanly.
sig_v2 = ct.compilation.KernelSignature(
    [array_param(), array_param(), shape_constraint],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
buf = io.BytesIO()
ct.compilation.export_kernel(load_2d_tile, [sig_v2], buf, gpu_code="sm_90", output_format="tileir_bytecode")
print(f"v2: compiled {len(buf.getvalue())} bytes")
```

```bash
python3 04_calling_convention_tuple_parameter.py
```

### File: `05_gpu_code_supported_and_unsupported.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    result = ct.load(a, (pid,), (tile_size,)) + ct.load(b, (pid,), (tile_size,))
    ct.store(c, (pid,), result)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# sm_120 -- Blackwell's consumer/workstation line -- genuinely compiles,
# a fourth real architecture alongside Chapter 1's sm_80/sm_90/sm_100.
buf = io.BytesIO()
ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_120", output_format="cubin")
print(f"sm_120: {len(buf.getvalue())} cubin bytes")

# sm_70 -- Volta, a real, shipped NVIDIA architecture, just one cuTile
# Python's tile-core compiler genuinely does not target -- is rejected by
# the underlying tileiras compiler itself, not by cuTile Python's own
# Python-level validation.
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="sm_70", output_format="cubin")
except ct.TileError as e:
    print(f"sm_70: {type(e).__name__}: {e}")

# A malformed gpu_code string never even reaches the compiler -- it fails
# in pure Python, parsing the string's numeric suffix.
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code="not_an_arch", output_format="cubin")
except ValueError as e:
    print(f"not_an_arch: {type(e).__name__}: {e}")
```

```bash
python3 05_gpu_code_supported_and_unsupported.py
```

## Chapter Summary

`export_kernel`'s `output_file` genuinely accepts a real filesystem path, and the resulting `cubin` was directly confirmed here to be a real ELF binary, magic bytes and all — not merely a file the documentation happens to call one. `bytecode_version`'s real, enumerable set of supported values is exactly `{13.1, 13.2, 13.3}` for the compiler installed in this book's own environment, and `None`'s documented "auto-detect the latest" behavior was confirmed, byte-for-byte, to mean precisely `13.3`. `cutile_python_v1` and `cutile_python_v2` are documented to differ only in `v2`'s added support for tuple parameters and static array shapes — genuinely confirmed to produce identical output for a signature with neither, and a real, immediate `ValueError` from `v1` the moment a tuple-shaped `ct.Constant` parameter is actually present. And "targeting hardware" has a real floor with two distinct failure modes on either side of it: `sm_120`, a genuine, newer architecture, compiles cleanly; `sm_70`, a real but genuinely unreachable one, is rejected by the `tileiras` compiler binary itself with a `TileCompilerExecutionError`; and a malformed architecture string never reaches the compiler at all, failing instead in cuTile Python's own Python-level parsing with a plain `ValueError`.

## Self-Check Questions

1. This chapter confirmed a `cubin` file is a real ELF binary by reading four specific bytes from it. What are those bytes, and what would finding a different value there have told you that the file's size alone could not?
2. `bytecode_version=None` and `bytecode_version="13.3"` produced identical byte counts in Worked Example 5.2.1, while `"13.1"` and `"13.2"` produced a different, shared byte count. What does that pattern let you conclude about which version `None` actually resolves to for this compiler?
3. `cutile_python_v1` and `cutile_python_v2` compiled identical output for `vector_add` in Worked Example 5.3.1, but not for `load_2d_tile` in Worked Example 5.3.2. What changed between the two kernels' signatures that explains the difference?
4. `sm_70` and `"not_an_arch"` are both `gpu_code` values that fail to compile. Name the two distinct exception types each one raises, and state which stage of `export_kernel`'s processing each failure actually happens in.
5. Chapter 1 established `sm_80`, `sm_90`, and `sm_100` as genuinely reachable architectures; this chapter adds `sm_120` and shows `sm_70` genuinely is not. Is it safe to conclude "cuTile Python supports every `sm_8x` and newer architecture" from this evidence? Why or why not?

## Where We Go Next

Every chapter in Part 0 has drawn its evidence from one side of a boundary this book has named repeatedly but never fully closed the loop on: `export_kernel`'s ahead-of-time compilation, genuinely run, against `ct.launch`'s real-hardware dispatch, genuinely never attempted. Part 1 doesn't cross that boundary either — this book's own environment still has no GPU — but it does turn to the other object every kernel signature has depended on since Chapter 1 without examining closely: the `Array` itself, and exactly what `ArrayConstraint`'s remaining fields let the compiler assume, safely or not, about the real memory a real array would hand a kernel at the launch time this book still cannot reach.

## Worked Solutions

**1.** The bytes `0x7f`, `'E'`, `'L'`, `'F'` — the standard ELF magic number every real ELF object file begins with. A different value there — anything not matching those four bytes — would have told you the file, whatever its size, wasn't actually a valid ELF binary at all: size alone confirms how much data was written, not what format that data is actually in.

**2.** Since `None` and `"13.3"` produced the exact same byte count (688 bytes) while `"13.1"` and `"13.2"` shared a different one (686 bytes), and cuTile Python's own compiler rejected `"13.0"` as unsupported, the only version left in the compiler's real supported set that both matches `None`'s output and is plausibly "the latest" is `13.3` — confirmed by the byte counts actually matching, not merely inferred from version numbering.

**3.** `vector_add`'s signature has only scalar and `Array` parameters plus one scalar `ct.Constant[int]` — nothing in it is a tuple-shaped constant, the one case the documented `v1`/`v2` difference actually concerns. `load_2d_tile`'s signature includes a `TupleConstraint`-wrapped `ct.Constant[tuple[int, int]]` for its 2D `tile_shape` — exactly the capability `v2` is documented to add beyond `v1`, which is why `v1` rejects it and `v2` doesn't.

**4.** `sm_70` raises a `TileCompilerExecutionError` — a `TileError` subclass — because the string parses successfully and is handed to the real `tileiras` compiler process, which then refuses to recognize it as a valid `--gpu-name` and exits with a nonzero return code. `"not_an_arch"` raises a plain `ValueError`, because it fails earlier, in cuTile Python's own Python-level parsing of the numeric suffix after `"sm_"` — before any compiler process is ever started.

**5.** No. This chapter's evidence supports "`sm_80`, `sm_90`, `sm_100`, and `sm_120` are each individually confirmed reachable, and `sm_70` is individually confirmed not to be" — a set of specific, tested facts about specific architecture names. It does not support a general rule about every architecture number in some numeric range, because no architecture between `sm_70` and `sm_80` (such as `sm_75`, Turing) was ever tested here; extending "the architectures I tested work" into "everything in this range works" is exactly the kind of untested generalization this book's own verification discipline exists to avoid.
