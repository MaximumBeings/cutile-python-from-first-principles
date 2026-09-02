# Chapter 11: Composing Kernels — Tile Functions and Inlining

> "Part 1 tested single primitives in isolation: one load, one store, one atomic, one compiler hint at a time. Real programs are built by composing primitives together, and cuTile Python's documentation makes a specific, testable claim about how that composition works: an ordinary, undecorated Python function called from tile code automatically becomes tile code itself, no annotation required. This chapter takes that claim at its word and asks what actually happens underneath it — whether a called helper is compiled once and shared, or copied at every call site, and exactly where composing functions stops working at all."

**What you will understand by the end of this chapter:**

- That an entirely undecorated Python helper function, called from inside a `@ct.kernel`, compiles identically to the same helper wrapped in `@ct.function` — confirming cuTile Python's documented claim that explicit annotation is genuinely unnecessary
- That Chapter 4's finding about `ct.function(tile=False)` having no observable effect extends to this new context: even a helper explicitly marked `tile=False` compiles and runs fine when called from a kernel
- That calling the same helper function multiple times from one kernel body genuinely duplicates real, generated code at each call site — direct evidence of inlining — but that the resulting byte counts grow non-monotonically, not linearly, and this book reports that honestly rather than forcing a clean story
- That recursion with a depth fixed at trace time compiles successfully, because the tracer simply follows the concrete Python call chain to its base case like any other Python recursion
- That recursion enforcement has two genuinely different ceilings stacked on top of each other — Python's own interpreter call-stack limit, and a separate, dedicated `TileRecursionError` the compiler raises at exactly 1,000 inlined frames — and that which one fires first is not the one intuition would predict

**What you need to know first:**

- Chapter 4's `ct.function(func=None, *, host=False, tile=True)` signature and its finding that `host=True` fully unwraps a function while `tile=False` shows no observable effect on a function called directly.
- Chapter 1's `TileError` hierarchy and its warning that compiled byte counts are sensitive to the exact surrounding source text — this chapter's comparisons are all made within a single script for exactly that reason.
- Chapter 10's naming-confound finding: the compiled cubin embeds function names as symbols, so every kernel and helper compared against another in this chapter shares the same name, `kernel_fn` and `helper_fn` respectively.

## 11.1 Composing Without Annotation `[FOUNDATIONAL]`

### Intuition

`ct.function`'s documentation states plainly: "When an unannotated function is called by a tile function, tile shall be added to the unannotated function's execution space... No explicit annotation is required." That is a specific, falsifiable claim — either an undecorated helper compiles identically to a decorated one, or it does not.

### Background

Every prior chapter's kernels called only cuTile Python's own built-in operations (`ct.load`, `ct.add`, `ct.sum`, and so on) directly inside the kernel body. This section is the first to call a genuinely user-defined helper function from inside a kernel at all.

### Worked Example 11.1.1 — a plain function versus one wrapped in `@ct.function`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ct.function's documentation states that an unannotated function called by a
# tile function automatically has tile added to its execution space, with no
# explicit annotation required. build_kernel(decorate) builds the identical
# kernel body and helper name either way, so a byte-count difference could
# only come from the decoration itself, never from a naming difference.
def build_kernel(decorate):
    def helper_fn(x):
        return ct.mul(x, ct.full(x.shape, 2.0, ct.float32))
    if decorate:
        helper_fn = ct.function(helper_fn)

    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = helper_fn(x)
        ct.store(c, (pid,), y)
    return kernel_fn

for name, decorate in [("plain, undecorated", False), ("@ct.function-decorated", True)]:
    size = compile_bytes(build_kernel(decorate))
    print(f"helper is {name}: {size} cubin bytes")
```

The complete file is `01_undecorated_vs_decorated_helper.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_undecorated_vs_decorated_helper.py
```

Genuinely run:

```
helper is plain, undecorated: 22112 cubin bytes
helper is @ct.function-decorated: 22112 cubin bytes
```

Byte-for-byte identical, confirmed with both the kernel and the helper sharing exactly the same names in both branches — the documentation's claim holds exactly as stated. Decorating a helper with `@ct.function` before calling it from a kernel changes nothing about what gets compiled; it exists purely for the cases Chapter 4 already covered, such as explicitly restricting or widening which execution spaces a function is callable from, not for enabling tile-callability in the first place.

## 11.2 `host`/`tile` Flags Revisited, From Inside a Kernel `[FOUNDATIONAL]`

### Intuition

Chapter 4 found that `ct.function(tile=False)` had no observable effect on a function called directly. That result left one honest gap: it never tested the one situation `tile`'s own documented meaning most directly addresses — "whether the function can be called from tile code" — by actually calling that function from inside a kernel. This section closes that gap.

### Background

`ct.function(func=None, *, host=False, tile=True)` documents `host` and `tile` as independent flags controlling which execution spaces may call the decorated function. If `tile=False` genuinely restricted tile-callability, calling such a function from inside a `@ct.kernel` body ought to be exactly where that restriction would surface.

### Worked Example 11.2.1 — every `host`/`tile` combination, called from a kernel

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Chapter 4 found that ct.function(tile=False) has no observable effect on a
# function called directly. This checks the same flag again, specifically
# for a function called from inside a kernel body -- the exact situation
# tile=False's own documented meaning ("whether the function can be called
# from tile code") should most directly govern.
def build_kernel(**function_kwargs):
    @ct.function(**function_kwargs)
    def helper_fn(x):
        return ct.mul(x, ct.full(x.shape, 2.0, ct.float32))

    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = helper_fn(x)
        ct.store(c, (pid,), y)
    return kernel_fn

for name, kwargs in [
    ("default (host=False, tile=True)", {}),
    ("host=True, tile=True", {"host": True, "tile": True}),
    ("host=True, tile=False", {"host": True, "tile": False}),
    ("host=False, tile=False", {"host": False, "tile": False}),
]:
    try:
        size = compile_bytes(build_kernel(**kwargs))
        print(f"helper decorated with {name}, called from a kernel: {size} cubin bytes")
    except Exception as e:
        print(f"helper decorated with {name}, called from a kernel: {type(e).__name__}: {e}")
```

The complete file is `02_function_host_tile_flags_from_a_kernel.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_function_host_tile_flags_from_a_kernel.py
```

Genuinely run:

```
helper decorated with default (host=False, tile=True), called from a kernel: 22240 cubin bytes
helper decorated with host=True, tile=True, called from a kernel: 22240 cubin bytes
helper decorated with host=True, tile=False, called from a kernel: 22240 cubin bytes
helper decorated with host=False, tile=False, called from a kernel: 22240 cubin bytes
```

All four combinations compile to the identical 22,240 bytes, including `tile=False` on its own and paired with `host=True`. Chapter 4's finding is confirmed a second time, in the one context where `tile=False`'s documented meaning had the clearest chance to matter and did not: in this installed `cuda-tile` 1.5.0, `tile` genuinely has no observable enforcement effect at all, whether the function is called directly or from inside a kernel body. (This section's 22,240-byte baseline is not meant to be compared against Section 11.1's 22,112-byte baseline — Chapter 1 already established that compiled byte counts are sensitive to the exact surrounding source text, and these two sections live in different files with different surrounding code. Every comparison this chapter makes stays within a single script.)

## 11.3 Repeated Calls Are Genuinely Inlined — But Not Linearly `[FOUNDATIONAL]`

### Intuition

If calling a helper function generates real code at the call site rather than a shared, reusable subroutine, calling it twice ought to produce more compiled bytes than calling it once. Whether that growth is linear, and whether it is even always positive, is a separate, genuinely open question this section checks directly.

### Background

GPU kernel compilers typically inline small helper functions rather than emitting real call/return instructions, both because tile code has no general call stack in the CPU sense and because inlining opens the door to further optimization across the call boundary. This section's helper is deliberately tiny, exactly the kind of function most likely to be inlined.

### Worked Example 11.3.1 — the same helper, called zero through three times

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

def helper_fn(x):
    return ct.mul(x, ct.full(x.shape, 2.0, ct.float32))

# build_kernel(n) traces a plain Python for loop over range(n) -- since n is
# an ordinary Python int, this unrolls at trace time into n literal calls to
# helper_fn, each one inlined separately. Every kernel below is named
# "kernel_fn" and every helper "helper_fn", so a byte-count difference can
# only come from how many times the helper was actually called.
def build_kernel(n):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        y = ct.load(a, (pid,), (tile_size,))
        for _ in range(n):
            y = helper_fn(y)
        ct.store(c, (pid,), y)
    return kernel_fn

for n in [0, 1, 2, 3]:
    size = compile_bytes(build_kernel(n))
    print(f"helper called {n} time(s): {size} cubin bytes")
```

The complete file is `03_repeated_calls_are_inlined.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_repeated_calls_are_inlined.py
```

Genuinely run:

```
helper called 0 time(s): 21664 cubin bytes
helper called 1 time(s): 22112 cubin bytes
helper called 2 time(s): 21984 cubin bytes
helper called 3 time(s): 22112 cubin bytes
```

> `[COMMON TRAP]` Going from zero calls to one call adds 448 bytes, confirming real, generated code appears at the call site — direct evidence of inlining. But going from one call to two *shrinks* the compiled output by 128 bytes, and three calls compiles to exactly the same size as one. This is a genuinely non-monotonic, non-linear result, verified deterministic across repeated fresh runs of this exact script. This book cannot see the compiler's internal register allocation or instruction scheduling decisions, so it cannot say why two calls compiles smaller than one — only that it measured this, twice, and is reporting it exactly as measured rather than forcing it into a "more calls, more code" story its own numbers do not support.

## 11.4 Bounded Recursion and Its Cost `[FOUNDATIONAL]`

### Intuition

Section 11.3 established that repeated calls are inlined. Recursion is a special case of a function calling itself — if the recursion depth is fixed and known before compilation even starts, there is no fundamental reason the tracer could not simply follow the concrete call chain down to its base case, the same way Python already does for any other recursive call with a known argument.

### Background

Nothing about `recursive_halve` below is data-dependent: `remaining` is a plain Python `int`, decremented by ordinary Python arithmetic, and the `if remaining <= 0` check is evaluated by the Python interpreter during tracing, not compiled into the kernel as a runtime branch. This is fundamentally different from data-dependent recursion, where the depth would only be known once the kernel actually ran.

### Worked Example 11.4.1 — recursion depth zero through three

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# recursive_halve's own recursion depth is a plain Python int, known
# completely at trace time -- nothing here is a runtime, data-dependent
# branch. The tracer simply follows the real Python call chain down to its
# base case, the same way it would for any other ordinary Python recursion
# with a concrete argument.
def recursive_halve(x, remaining: int):
    if remaining <= 0:
        return x
    return recursive_halve(ct.mul(x, ct.full(x.shape, 0.5, ct.float32)), remaining - 1)

def build_kernel(depth):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        y = ct.load(a, (pid,), (tile_size,))
        y = recursive_halve(y, depth)
        ct.store(c, (pid,), y)
    return kernel_fn

for depth in [0, 1, 2, 3]:
    size = compile_bytes(build_kernel(depth))
    print(f"recursion depth={depth}: {size} cubin bytes")
```

The complete file is `04_bounded_recursion_and_its_cost.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_bounded_recursion_and_its_cost.py
```

Genuinely run:

```
recursion depth=0: 21664 cubin bytes
recursion depth=1: 22240 cubin bytes
recursion depth=2: 22496 cubin bytes
recursion depth=3: 22752 cubin bytes
```

Trace-time-bounded recursion genuinely compiles, confirming the intuition: with a concrete depth, this is no different from any other Python function call chain the tracer can simply follow to completion. Depth `0` matches Section 11.3's zero-calls result exactly (21,664 bytes), a real consistency check — both scripts trace the identical trivial kernel body, load then store with no helper invoked at all. Past that shared starting point, however, recursion's growth is not the same shape as the loop's: depths `1` through `3` grow by 576, then 256, then 256 bytes — mostly steady, but not identical to Section 11.3's own non-monotonic pattern for the same nominal "how many times was the helper effectively applied" question. Composing the same primitive two different syntactic ways, a loop versus recursion, does not guarantee the same compiled-size behavior, even when both are fully resolved at trace time.

## 11.5 The Recursion Limit: Two Different Ceilings `[FOUNDATIONAL]`

### Intuition

Section 11.4 confirmed that a small, fixed recursion depth compiles fine. Nothing about that guarantees an *arbitrary* fixed depth compiles — at some point, either the compiler itself imposes a limit, or the tracing process running inside ordinary Python hits Python's own limits first. This section finds out which, and exactly where.

### Background

Python's interpreter enforces its own recursion limit (`sys.getrecursionlimit()`, 1,000 by default) on any Python call chain, including the tracer's own recursive descent through source code during compilation. Separately, cuTile Python's compiler could, in principle, impose its own limit on how many function-call frames it is willing to inline. Both are real, independent ceilings, and there is no guarantee in advance about which one a given recursion depth reaches first.

### Worked Example 11.5.1 — sweeping depths from 990 to 1,001

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def recursive_halve(x, remaining: int):
    if remaining <= 0:
        return x
    return recursive_halve(ct.mul(x, ct.full(x.shape, 0.5, ct.float32)), remaining - 1)

def build_kernel(depth):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        y = ct.load(a, (pid,), (tile_size,))
        y = recursive_halve(y, depth)
        ct.store(c, (pid,), y)
    return kernel_fn

# The compiler documents its own inlining depth limit at 1000 (see
# 06_unbounded_recursion.py). Sweeping depths right around that number
# checks which limit is actually reached first: the tile compiler's own
# documented 1000-frame cap, or Python's own interpreter call-stack limit,
# exhausted first by the tracer's own recursive descent through the source.
for depth in [990, 995, 996, 997, 998, 999, 1000, 1001]:
    kernel = build_kernel(depth)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"depth={depth}: {len(buf.getvalue())} cubin bytes")
    except RecursionError:
        print(f"depth={depth}: RecursionError: maximum recursion depth exceeded")
    except ct.TileError as e:
        print(f"depth={depth}: {type(e).__name__}: {str(e).splitlines()[0]}")
```

The complete file is `05_recursion_limit_two_ceilings.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_recursion_limit_two_ceilings.py
```

Genuinely run:

```
depth=990: RecursionError: maximum recursion depth exceeded
depth=995: RecursionError: maximum recursion depth exceeded
depth=996: RecursionError: maximum recursion depth exceeded
depth=997: RecursionError: maximum recursion depth exceeded
depth=998: RecursionError: maximum recursion depth exceeded
depth=999: TileRecursionError: Maximum recursion depth (1000) reached while inlining a function call
depth=1000: TileRecursionError: Maximum recursion depth (1000) reached while inlining a function call
depth=1001: TileRecursionError: Maximum recursion depth (1000) reached while inlining a function call
```

> `[COMMON TRAP]` This crossover runs backwards from what intuition suggests. Depths `990` through `998` — the *smaller* depths in this sweep — hit Python's own bare `RecursionError` first, before the tile compiler's own dedicated error ever has a chance to fire. Depths `999` and above instead reach the compiler's own clean, dedicated `TileRecursionError`, naming its exact 1,000-frame limit. A smaller requested recursion depth failing with a *less* informative, more generic error than a larger one is not the direction this book's intuition would have predicted going in. Confirmed deterministic across repeated fresh runs of this exact script, and confirmed identical whether these depths are swept in one script (as here) or run one at a time in entirely separate, isolated processes. This book cannot see far enough into the tracer's own internal call structure to explain why the crossover sits exactly where it does — only that it measured this exact boundary, twice, and reports it as measured.

### Worked Example 11.5.2 — a genuinely unbounded recursion

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# No base case at all -- a genuinely unbounded recursion during tracing.
def unbounded_recursion(x):
    return unbounded_recursion(ct.mul(x, ct.full(x.shape, 0.5, ct.float32)))

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = unbounded_recursion(x)
    ct.store(c, (pid,), y)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(kernel_fn, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"unbounded recursion: {len(buf.getvalue())} cubin bytes")
except ct.TileError as e:
    full_text = str(e)
    lines = full_text.splitlines()
    print(f"exception type: {type(e).__name__}")
    print(f"exception MRO: {[c.__name__ for c in type(e).__mro__]}")
    print(f"first line: {lines[0]}")
    print(f"total lines in message: {len(lines)}")
    print(f"total bytes in message: {len(full_text)}")
```

The complete file is `06_unbounded_recursion.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_unbounded_recursion.py
```

Genuinely run:

```
exception type: TileRecursionError
exception MRO: ['TileRecursionError', 'TileError', 'Exception', 'BaseException', 'object']
first line: Maximum recursion depth (1000) reached while inlining a function call
total lines in message: 3001
total bytes in message: 246967
```

`TileRecursionError` is a real, dedicated exception class this book has not encountered before, sitting directly under `TileError` alongside `TileTypeError` and `TileSyntaxError` from earlier chapters. This script deliberately does not print the raw exception text in full: the genuine message is 3,001 lines and 246,967 bytes, almost entirely the same repeated frame (`unbounded_recursion` calling itself) printed 999 times, plus the initial `kernel_fn` frame — 1,000 frames in total, matching the error's own stated limit exactly. Printing the summary above rather than the full text is a deliberate choice about what belongs on a page, not a substitution for genuine output — running `06_unbounded_recursion.py` yourself reproduces the complete, real 3,001-line message, unedited.

## Complete Runnable Code

### File: `01_undecorated_vs_decorated_helper.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# ct.function's documentation states that an unannotated function called by a
# tile function automatically has tile added to its execution space, with no
# explicit annotation required. build_kernel(decorate) builds the identical
# kernel body and helper name either way, so a byte-count difference could
# only come from the decoration itself, never from a naming difference.
def build_kernel(decorate):
    def helper_fn(x):
        return ct.mul(x, ct.full(x.shape, 2.0, ct.float32))
    if decorate:
        helper_fn = ct.function(helper_fn)

    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = helper_fn(x)
        ct.store(c, (pid,), y)
    return kernel_fn

for name, decorate in [("plain, undecorated", False), ("@ct.function-decorated", True)]:
    size = compile_bytes(build_kernel(decorate))
    print(f"helper is {name}: {size} cubin bytes")
```

```bash
python3 01_undecorated_vs_decorated_helper.py
```

### File: `02_function_host_tile_flags_from_a_kernel.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Chapter 4 found that ct.function(tile=False) has no observable effect on a
# function called directly. This checks the same flag again, specifically
# for a function called from inside a kernel body -- the exact situation
# tile=False's own documented meaning ("whether the function can be called
# from tile code") should most directly govern.
def build_kernel(**function_kwargs):
    @ct.function(**function_kwargs)
    def helper_fn(x):
        return ct.mul(x, ct.full(x.shape, 2.0, ct.float32))

    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = helper_fn(x)
        ct.store(c, (pid,), y)
    return kernel_fn

for name, kwargs in [
    ("default (host=False, tile=True)", {}),
    ("host=True, tile=True", {"host": True, "tile": True}),
    ("host=True, tile=False", {"host": True, "tile": False}),
    ("host=False, tile=False", {"host": False, "tile": False}),
]:
    try:
        size = compile_bytes(build_kernel(**kwargs))
        print(f"helper decorated with {name}, called from a kernel: {size} cubin bytes")
    except Exception as e:
        print(f"helper decorated with {name}, called from a kernel: {type(e).__name__}: {e}")
```

```bash
python3 02_function_host_tile_flags_from_a_kernel.py
```

### File: `03_repeated_calls_are_inlined.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

def helper_fn(x):
    return ct.mul(x, ct.full(x.shape, 2.0, ct.float32))

# build_kernel(n) traces a plain Python for loop over range(n) -- since n is
# an ordinary Python int, this unrolls at trace time into n literal calls to
# helper_fn, each one inlined separately. Every kernel below is named
# "kernel_fn" and every helper "helper_fn", so a byte-count difference can
# only come from how many times the helper was actually called.
def build_kernel(n):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        y = ct.load(a, (pid,), (tile_size,))
        for _ in range(n):
            y = helper_fn(y)
        ct.store(c, (pid,), y)
    return kernel_fn

for n in [0, 1, 2, 3]:
    size = compile_bytes(build_kernel(n))
    print(f"helper called {n} time(s): {size} cubin bytes")
```

```bash
python3 03_repeated_calls_are_inlined.py
```

### File: `04_bounded_recursion_and_its_cost.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# recursive_halve's own recursion depth is a plain Python int, known
# completely at trace time -- nothing here is a runtime, data-dependent
# branch. The tracer simply follows the real Python call chain down to its
# base case, the same way it would for any other ordinary Python recursion
# with a concrete argument.
def recursive_halve(x, remaining: int):
    if remaining <= 0:
        return x
    return recursive_halve(ct.mul(x, ct.full(x.shape, 0.5, ct.float32)), remaining - 1)

def build_kernel(depth):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        y = ct.load(a, (pid,), (tile_size,))
        y = recursive_halve(y, depth)
        ct.store(c, (pid,), y)
    return kernel_fn

for depth in [0, 1, 2, 3]:
    size = compile_bytes(build_kernel(depth))
    print(f"recursion depth={depth}: {size} cubin bytes")
```

```bash
python3 04_bounded_recursion_and_its_cost.py
```

### File: `05_recursion_limit_two_ceilings.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def recursive_halve(x, remaining: int):
    if remaining <= 0:
        return x
    return recursive_halve(ct.mul(x, ct.full(x.shape, 0.5, ct.float32)), remaining - 1)

def build_kernel(depth):
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        y = ct.load(a, (pid,), (tile_size,))
        y = recursive_halve(y, depth)
        ct.store(c, (pid,), y)
    return kernel_fn

# The compiler documents its own inlining depth limit at 1000 (see
# 06_unbounded_recursion.py). Sweeping depths right around that number
# checks which limit is actually reached first: the tile compiler's own
# documented 1000-frame cap, or Python's own interpreter call-stack limit,
# exhausted first by the tracer's own recursive descent through the source.
for depth in [990, 995, 996, 997, 998, 999, 1000, 1001]:
    kernel = build_kernel(depth)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"depth={depth}: {len(buf.getvalue())} cubin bytes")
    except RecursionError:
        print(f"depth={depth}: RecursionError: maximum recursion depth exceeded")
    except ct.TileError as e:
        print(f"depth={depth}: {type(e).__name__}: {str(e).splitlines()[0]}")
```

```bash
python3 05_recursion_limit_two_ceilings.py
```

### File: `06_unbounded_recursion.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# No base case at all -- a genuinely unbounded recursion during tracing.
def unbounded_recursion(x):
    return unbounded_recursion(ct.mul(x, ct.full(x.shape, 0.5, ct.float32)))

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    y = unbounded_recursion(x)
    ct.store(c, (pid,), y)

buf = io.BytesIO()
try:
    ct.compilation.export_kernel(kernel_fn, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"unbounded recursion: {len(buf.getvalue())} cubin bytes")
except ct.TileError as e:
    full_text = str(e)
    lines = full_text.splitlines()
    print(f"exception type: {type(e).__name__}")
    print(f"exception MRO: {[c.__name__ for c in type(e).__mro__]}")
    print(f"first line: {lines[0]}")
    print(f"total lines in message: {len(lines)}")
    print(f"total bytes in message: {len(full_text)}")
```

```bash
python3 06_unbounded_recursion.py
```

## Chapter Summary

An undecorated Python helper called from inside a `@ct.kernel` body compiles byte-for-byte identically to the same helper wrapped in `@ct.function`, confirming the documentation's claim that explicit annotation is genuinely unnecessary. Chapter 4's earlier finding that `ct.function(tile=False)` has no observable effect extended cleanly into the one context where it might have mattered most: even a helper explicitly marked `tile=False` compiles and is callable from a kernel without restriction, in every `host`/`tile` combination tested. Calling the same small helper repeatedly genuinely duplicates real compiled code at each call site — clear evidence of inlining — but the growth measured across zero, one, two, and three calls was non-monotonic rather than linear, and this book reported that honestly rather than smoothing it into a cleaner-sounding pattern its own numbers do not support. Recursion with a depth fixed entirely at trace time compiles successfully, the tracer simply following the concrete Python call chain to its base case, though its own byte-count growth did not match the loop's pattern either. Finally, recursion enforcement turned out to have two genuinely separate ceilings: Python's own interpreter recursion limit, exhausted first at the smaller depths tested in this chapter's sweep, and a distinct, dedicated `TileRecursionError` the compiler itself raises at exactly 1,000 inlined frames, reached instead at the larger depths — a crossover running backwards from what intuition alone would have predicted, confirmed deterministically rather than assumed.

## Self-Check Questions

1. Section 11.1 found that decorating a helper with `@ct.function` made no measurable difference when called from a kernel. Given that finding, what is `@ct.function` actually for, based on what Chapter 4 and this chapter have together established?
2. Section 11.3's byte counts for zero through three calls were 21,664 / 22,112 / 21,984 / 22,112 — non-monotonic. What specific claim would be unsupported by this data, even though "more calls means more code" sounds intuitively obvious?
3. Section 11.4's recursion-depth-zero result matched Section 11.3's zero-calls result exactly. Why is that agreement a meaningful consistency check, rather than a coincidence this book should ignore?
4. Section 11.5.1 found that smaller recursion depths (990-998) failed with Python's own generic `RecursionError`, while larger depths (999-1001) reached the tile compiler's own `TileRecursionError` first. Why can this book not explain *why* this crossover runs in this direction, and what would it need to do so?
5. Section 11.5.2 deliberately printed a summary of `TileRecursionError`'s message (line count, byte count, first line) rather than the full 3,001-line text. Does this violate this book's standing rule against fabricating output? Why or why not?

## Where We Go Next

This chapter established that composing kernels out of ordinary Python functions works exactly as documented, that composition is implemented by real inlining rather than shared subroutines, and that inlining has a real, enforced limit distinct from Python's own. Chapter 12 turns to composing kernels a different way: not by calling helper functions inside one kernel, but by exporting several distinct, related kernels into a single compiled artifact with `ct.compilation.export_kernel`'s support for multiple kernels at once — the same driver-free, `export_kernel`-only discipline this book has followed since Chapter 1, now applied to a program built from more than one kernel entry point.

## Worked Solutions

**1.** Based on Chapter 4 and this chapter together, `@ct.function` exists to let a function be called from execution spaces it would not otherwise be callable from by default — most notably `host=True`, which Chapter 4 found fully unwraps a function for host-side use. It is not needed merely to make an ordinary helper callable from tile code, since that already happens automatically for any unannotated function; `tile=False`, meanwhile, was found in both chapters to have no enforced restricting effect at all in this installed version.

**2.** "More calls always means more code" would be unsupported — two calls measurably compiled to *less* code than one call in this exact, repeated measurement. The narrower, supportable claim is that at least one call produces measurably more code than zero calls (proving inlining happens at all), not that the relationship between call count and size is monotonic.

**3.** Because both scripts, despite testing conceptually different things (a loop trip count of zero, and a recursion depth of zero), trace down to the literal same kernel body: load a tile, then immediately store it, with no helper invoked at all. If a helper genuinely contributes nothing when its call count is zero in both formulations, the two independently-written scripts landing on the exact same byte count is a real, meaningful check that neither script has some other, unrelated difference secretly affecting its baseline.

**4.** This book can only observe results at the `export_kernel` boundary — a `RecursionError`, a `TileRecursionError`, or a successful compile — and cannot see the Python call stack depth or the tracer's own internal bookkeeping at the moment either limit is reached. Explaining the direction of the crossover would require instrumenting or reading the tracer's own internal implementation, which is exactly the kind of access this book's `export_kernel`-only, driver-free discipline does not have, separate from the "don't reproduce vendor source" restriction this book has followed since its introduction.

**5.** No. The full, real, unedited 3,001-line, 246,967-byte message was genuinely produced by genuinely running `06_unbounded_recursion.py` — nothing about its content was invented. What appears in this chapter's text is an honest, explicitly-labeled *summary* of that genuine output (its type, its class hierarchy, its first line, its exact line and byte counts), not a fabricated replacement for it, and the canonical script itself reproduces the complete real text for anyone who runs it. Fabrication would mean inventing content that was never actually produced; this is instead a disclosed editorial choice about how much of a genuinely-produced, very large output belongs reprinted on a page.
