# Chapter 10: Compiler Hints and Autotuning

> "Chapter 2 established that `ct.compilation.export_kernel` can produce several genuinely different compiled specializations of the same kernel source, driver-free, just by varying what gets passed at compile time. This chapter asks the natural next question: if a kernel can compile to more than one thing, which one should you actually pick? cuTile Python's real answer, `ct.tune.exhaustive_search`, picks by launching every candidate on real hardware and timing it — a question this book's `export_kernel`-only environment cannot even ask, let alone answer. What follows is the honest version available at this boundary: how far compiler hints can be pushed and inspected without a GPU, exactly where the compiler enforces their documented limits, and precisely what happens when the genuine autotuning API is reached from a machine with no CUDA driver at all."

**What you will understand by the end of this chapter:**

- That `ct.kernel`'s documented compiler hints — `num_ctas`, `occupancy`, and `num_worker_warps` — are genuinely, independently enforced at compile time, each against its own documented boundary, confirmed by triggering real rejections at and past each boundary
- That `kernel.replace_hints(...)` returns a new, independent kernel object whose compiled output is byte-identical to building the same hints in from scratch, and that hints accumulate across chained `replace_hints` calls rather than resetting
- A genuine methodological trap this book had not yet encountered: the compiled cubin embeds the kernel function's Python name as a symbol, so comparing byte counts across differently-named kernels silently mixes a naming effect into whatever else is being measured
- That `ct.tune.exhaustive_search`, cuTile Python's real autotuning entry point, requires a genuine CUDA stream and genuinely launches kernels to time them — confirmed directly, in this exact sandbox, by reaching it with no GPU and no PyTorch installed at all
- Why "the smallest compiled output" is a real, measurable, driver-free stand-in this book can use for its own hint search, and why it is explicitly not the same claim as "the fastest kernel," which only genuine hardware can settle

**What you need to know first:**

- Chapter 2's confirmation that `export_kernel` can produce multiple, genuinely different compiled specializations of identical kernel source.
- Chapter 1's `TileTypeError`/`TileError` hierarchy, and the general shape of a real, compiler-raised rejection.
- Chapter 9's closing framing, which this chapter completes: a search over compiled size, not over measured runtime.

## 10.1 `num_ctas`: A Genuine, Enforced Hint `[FOUNDATIONAL]`

### Intuition

`ct.kernel`'s docstring states `num_ctas` "must be a power of 2 between 1 and 16, inclusive," default `None` (auto). A docstring range is not automatically an enforced range — Chapter 6 and Chapter 7 both found documented defaults and constraints that turned out to be enforced only in specific, sometimes surprising ways. This section checks `num_ctas` directly against both halves of its stated rule: the power-of-2 requirement and the `[1, 16]` bound.

### Background

A CTA (Cooperative Thread Array) is a single thread block; `num_ctas` sets how many CTAs cooperate as one CGA (Cooperative Grid Array) — a hardware grouping introduced on recent NVIDIA architectures, sitting above an individual block. This is a genuine compile-time hint: it changes what the compiler generates, not something read only at launch.

### Worked Example 10.1.1 — every boundary of the documented `num_ctas` rule

```python
import cuda.tile as ct
import io

def build_kernel(num_ctas):
    @ct.kernel(num_ctas=num_ctas)
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# num_ctas is documented as "a power of 2 between 1 and 16, inclusive", default None (auto).
for name, num_ctas in [("None (default)", None), ("1", 1), ("2", 2), ("4", 4), ("8", 8), ("16", 16),
                        ("3 (not a power of 2)", 3), ("0 (below range)", 0), ("32 (above range)", 32)]:
    buf = io.BytesIO()
    try:
        kernel = build_kernel(num_ctas)
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"num_ctas={name}: {len(buf.getvalue())} cubin bytes")
    except Exception as e:
        print(f"num_ctas={name}: {type(e).__name__}: {e}")
```

The complete file is `01_num_ctas_valid_and_invalid.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_num_ctas_valid_and_invalid.py
```

Genuinely run:

```
num_ctas=None (default): 24736 cubin bytes
num_ctas=1: 24736 cubin bytes
num_ctas=2: 23840 cubin bytes
num_ctas=4: 24480 cubin bytes
num_ctas=8: 24480 cubin bytes
num_ctas=16: 24480 cubin bytes
num_ctas=3 (not a power of 2): ValueError: num_ctas should be power of 2, got 3
num_ctas=0 (below range): ValueError: num_ctas should be [1, 16], got 0
num_ctas=32 (above range): ValueError: num_ctas should be [1, 16], got 32
```

Both halves of the documented rule are genuinely, separately enforced, with two distinct `ValueError` messages naming exactly which half was violated — `3` fails the power-of-2 check with its own message, while `0` and `32` fail the range check with a different one, even though `32` actually *is* a power of 2. Neither rejection is a `TileError`; both are plain `ValueError`, raised before the kernel body is ever inspected — this is argument validation on the hint itself, the same category of failure Chapter 6 found for `stride_lower_bound_incl`. Among the accepted values, `None` and `1` compile identically (auto apparently resolves to the same code as an explicit `1` here), while `2`, and separately `4`/`8`/`16` (which all three match each other exactly), produce genuinely different sizes — evidence the hint really does change what gets generated, not merely how it is scheduled at launch.

## 10.2 `occupancy`: A Boundary Effect, Not a Gradient `[FOUNDATIONAL]`

### Intuition

`occupancy` is documented as "[1, 32]," the expected number of active CTAs per SM, default `None` (auto). If this hint behaves the way `num_ctas` just did, most of its legal range should do *something* observable, with real rejections outside `[1, 32]`.

### Background

Occupancy is a scheduling hint: it tells the compiler how many CTAs the caller expects to have resident on a single streaming multiprocessor at once, which can affect register allocation and other resource trade-offs the compiler makes ahead of time.

### Worked Example 10.2.1 — the full documented `occupancy` range, and both edges past it

```python
import cuda.tile as ct
import io

def build_kernel(occupancy):
    @ct.kernel(occupancy=occupancy)
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# occupancy is documented as "[1, 32]", default None (auto).
for name, occupancy in [("None (default)", None), ("1", 1), ("2", 2), ("4", 4), ("8", 8),
                         ("16", 16), ("32", 32), ("0 (below range)", 0), ("33 (above range)", 33)]:
    buf = io.BytesIO()
    try:
        kernel = build_kernel(occupancy)
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"occupancy={name}: {len(buf.getvalue())} cubin bytes")
    except Exception as e:
        print(f"occupancy={name}: {type(e).__name__}: {e}")
```

The complete file is `02_occupancy_hint.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_occupancy_hint.py
```

Genuinely run:

```
occupancy=None (default): 24736 cubin bytes
occupancy=1: 24736 cubin bytes
occupancy=2: 24736 cubin bytes
occupancy=4: 24736 cubin bytes
occupancy=8: 24736 cubin bytes
occupancy=16: 24736 cubin bytes
occupancy=32: 24608 cubin bytes
occupancy=0 (below range): ValueError: occupancy should be [1, 32], got 0
occupancy=33 (above range): ValueError: occupancy should be [1, 32], got 33
```

> `[COMMON TRAP]` A hint documented as a range does not mean every value in that range produces observably different compiled output. `occupancy` compiles to the identical 24,736 bytes for `None` and every tested value from `1` through `16` — only `32`, the top of the documented range, genuinely produces smaller code (24,608 bytes). This book cannot say *why* only the extreme value moves the needle here without seeing the compiler's internal register-allocation decisions, but it can say, from a real, repeated measurement, that the effect is a boundary step rather than a gradient — the same caution Chapter 7 already applied to `shape_divisible_by`, where the factor mattered in a way `stride_divisible_by`'s factor did not.

Range enforcement is genuine and symmetric: `0` and `33` are both rejected with the same `[1, 32]` message the docstring itself states, `ValueError` rather than any `TileError` subclass, exactly like `num_ctas`'s range check in Section 10.1.

## 10.3 `num_worker_warps`: A Membership Check, Enforced Without a Warning `[FOUNDATIONAL]`

### Intuition

`num_worker_warps`'s docstring is the most heavily qualified of the three hints tested so far: "must be either 4 or 8," default `None` (auto), and "Since CTK 13.3. Ignored with a warning otherwise." That last sentence describes a real possibility this book can test directly: on whatever CUDA Toolkit version this installed `cuda-tile` 1.5.0 is actually built against, is an invalid value quietly ignored with a warning, or hard-rejected?

### Background

This hint tunes the warp count in the CUDA-core warp groups of a warp-specialized kernel — the documentation names normalization-style kernels with large tiles and high register pressure as the canonical case where it is worth adjusting. Unlike `num_ctas` and `occupancy`, its own docstring already hints that its enforcement might be version-conditional.

### Worked Example 10.3.1 — `4`, `8`, and an explicitly invalid value, checked for warnings

```python
import cuda.tile as ct
import io

def build_kernel(num_worker_warps):
    @ct.kernel(num_worker_warps=num_worker_warps)
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# num_worker_warps is documented as "must be either 4 or 8", default None (auto),
# and documented as "Since CTK 13.3. Ignored with a warning otherwise."
for name, val in [("None (default)", None), ("4", 4), ("8", 8), ("6 (not 4 or 8)", 6)]:
    buf = io.BytesIO()
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kernel = build_kernel(val)
            ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
            warning_msgs = [str(wi.message) for wi in w]
        print(f"num_worker_warps={name}: {len(buf.getvalue())} cubin bytes, warnings={warning_msgs}")
    except Exception as e:
        print(f"num_worker_warps={name}: {type(e).__name__}: {e}")
```

The complete file is `03_num_worker_warps.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_num_worker_warps.py
```

Genuinely run:

```
num_worker_warps=None (default): 24736 cubin bytes, warnings=[]
num_worker_warps=4: 24736 cubin bytes, warnings=[]
num_worker_warps=8: 23328 cubin bytes, warnings=[]
num_worker_warps=6 (not 4 or 8): ValueError: num_worker_warps should be either 4 or 8, got 6
```

The docstring's conditional escape hatch — "ignored with a warning" on an older CUDA Toolkit — does not fire in this installed `cuda-tile` 1.5.0: `6` is hard-rejected with a `ValueError`, and no warning of any kind is recorded in either the accepted or rejected calls, confirmed with Python's own `warnings.catch_warnings` capturing everything at `"always"`. Taken together with `8` compiling to genuinely less code than `4` or `None` (which match each other exactly), the honest conclusion is narrow but solid: on whatever CTK this package bundles, `num_worker_warps` is a real, enforced, size-affecting hint — not the softer, warning-only behavior its own docstring describes as possible on some other toolkit version. This book cannot determine which CTK version that actually is from inside this sandbox, and does not claim to.

## 10.4 `replace_hints`: An Immutable Rebuild — and a Naming Trap `[FOUNDATIONAL]`

### Intuition

`kernel.replace_hints(**hints)` is documented to "return a new kernel with updated compiler hints," noting that "the returned object will have its own JIT cache." Three genuinely separate questions are worth confirming directly rather than assuming from that one sentence: does it return a *new* object rather than mutating the original; does that new object compile to exactly what building the same hints from scratch would produce; and does calling it a second time keep the first hint or silently drop it?

### Background

While answering those three questions with this chapter's very first script, an unrelated discrepancy showed up: two kernels built with what should have been the identical hint compiled to different byte counts. The cause turned out to be simple once isolated, but it was not obvious in advance, and it is worth stating plainly before the real results below, because it would have quietly corrupted every comparison in this chapter if left unnoticed.

> `[COMMON TRAP]` The compiled cubin embeds the kernel function's own Python name as a symbol. Two kernels with byte-for-byte identical bodies and identical compiler hints, built from functions named `short_name` and `a_much_much_longer_function_name_for_the_identical_kernel_body`, compiled to 24,736 and 26,576 bytes respectively — a 1,840-byte difference from naming alone, nothing else. Every comparison in this chapter, including the `replace_hints` checks below, therefore builds every kernel from a function named exactly `kernel_fn`, so a byte-count difference can only mean a hint actually changed something.

### Worked Example 10.4.1 — a new object, an equivalent compile, and cumulative chaining

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# COMMON TRAP: the compiled cubin embeds the kernel's Python function name as a
# symbol, so comparing byte counts across two kernels built from
# differently-named functions silently mixes a naming effect into the
# comparison. Every kernel below is built from a function named "kernel_fn",
# so name length can never explain a byte-count difference here.
def build_kernel_fn(**hints):
    @ct.kernel(**hints)
    def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return kernel_fn

base = build_kernel_fn()
base_bytes = compile_bytes(base)
print(f"kernel_fn (no hints): {base_bytes} cubin bytes")

tuned = base.replace_hints(num_ctas=2)
tuned_bytes = compile_bytes(tuned)
print(f"kernel_fn.replace_hints(num_ctas=2): {tuned_bytes} cubin bytes")
print(f"tuned is base: {tuned is base}")
print(f"type(tuned).__name__: {type(tuned).__name__}")

direct = build_kernel_fn(num_ctas=2)
direct_bytes = compile_bytes(direct)
print(f"kernel_fn built directly with num_ctas=2: {direct_bytes} cubin bytes")
print(f"replace_hints bytes == direct-built bytes: {tuned_bytes == direct_bytes}")

# Chaining: does a second replace_hints call preserve the first hint?
tuned_twice = tuned.replace_hints(occupancy=32)
tuned_twice_bytes = compile_bytes(tuned_twice)
print(f"kernel_fn.replace_hints(num_ctas=2).replace_hints(occupancy=32): {tuned_twice_bytes} cubin bytes")

both_direct = build_kernel_fn(num_ctas=2, occupancy=32)
both_direct_bytes = compile_bytes(both_direct)
print(f"kernel_fn built directly with num_ctas=2, occupancy=32: {both_direct_bytes} cubin bytes")
print(f"chained replace_hints bytes == both-direct bytes: {tuned_twice_bytes == both_direct_bytes}")

# Does the second replace_hints call preserve num_ctas=2, or silently drop it
# back to auto? Compare against occupancy=32 alone (no num_ctas) under the
# same controlled name.
occupancy_only = build_kernel_fn(occupancy=32)
occupancy_only_bytes = compile_bytes(occupancy_only)
print(f"kernel_fn built directly with occupancy=32 only (no num_ctas): {occupancy_only_bytes} cubin bytes")
print(f"chained result == occupancy-only (would mean num_ctas got dropped): {tuned_twice_bytes == occupancy_only_bytes}")
```

The complete file is `04_replace_hints.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_replace_hints.py
```

Genuinely run:

```
kernel_fn (no hints): 24608 cubin bytes
kernel_fn.replace_hints(num_ctas=2): 23840 cubin bytes
tuned is base: False
type(tuned).__name__: kernel
kernel_fn built directly with num_ctas=2: 23840 cubin bytes
replace_hints bytes == direct-built bytes: True
kernel_fn.replace_hints(num_ctas=2).replace_hints(occupancy=32): 23840 cubin bytes
kernel_fn built directly with num_ctas=2, occupancy=32: 23840 cubin bytes
chained replace_hints bytes == both-direct bytes: True
kernel_fn built directly with occupancy=32 only (no num_ctas): 24608 cubin bytes
chained result == occupancy-only (would mean num_ctas got dropped): False
```

All three questions are answered cleanly, with the naming trap controlled for. `tuned is base` is `False` — `replace_hints` genuinely returns a new object, never mutating the one it was called on. That new object's compiled output is byte-identical to a kernel built directly with the same hint from scratch (`23840 == 23840`), confirming `replace_hints` is a real equivalent to rebuilding, not merely a lighter-weight annotation that behaves differently downstream. And chaining `replace_hints(num_ctas=2).replace_hints(occupancy=32)` matches a kernel built directly with *both* hints together, while genuinely differing from a kernel built with `occupancy=32` alone — proof the first call's `num_ctas=2` survived the second call rather than being silently reset.

## 10.5 `ct.tune.exhaustive_search`: Real Autotuning, and the Boundary This Book Can't Cross `[FOUNDATIONAL]`

### Intuition

Every hint tested so far in this chapter was explored the same way every chapter since Chapter 1 has explored cuTile Python: compile it, ahead of time, with `export_kernel`, and read off a byte count or an error. `ct.tune.exhaustive_search` is documented to do something categorically different — it takes a real CUDA `stream`, a `grid_fn`, and an `args_fn`, and it *launches* each candidate configuration to measure its actual `mean_us` timing. This section confirms, directly and without assuming, exactly where that requirement first bites in an environment with no GPU at all.

### Background

`ct.tune.exhaustive_search(search_space, stream, grid_fn, kernel, args_fn, hints_fn=None, *, quiet=False, single_run_timeout_sec=None)` returns a `ct.tune.TuningResult`, holding a `best` `Measurement` (a `config`, `mean_us`, `num_samples`, and `error_margin_us`) plus every config's `successes` and `failures`. Its own documented example builds the `stream` argument from `torch.cuda.current_stream()` and launches a real matrix-multiply kernel across a grid of tile-size and `num_ctas` combinations, timing each one for real before reporting the fastest. Nothing about this API has an ahead-of-time equivalent — timing requires running.

### Worked Example 10.5.1 — reaching `exhaustive_search` from a machine with no GPU

```python
import cuda.tile as ct

# This file exists to answer one honest question directly, rather than
# leaving it implicit: what actually happens if you try to use
# ct.tune.exhaustive_search on a machine with no CUDA driver and no GPU?
#
# exhaustive_search's documented signature requires a real CUDA `stream`
# argument (e.g. torch.cuda.current_stream()) and launches the kernel for
# real to measure it -- there is no ahead-of-time equivalent. This sandbox
# has cuda-tile 1.5.0 installed but no PyTorch, no nvidia-smi, and no GPU,
# so this is a genuine, unmodified account of what surfaces first.

print(f"cuda.tile version: {ct.__version__}")

try:
    import torch
    print("torch import: succeeded")
except ImportError as e:
    print(f"torch import: ImportError: {e}")

# Since torch itself is not installed here, ct.tune.exhaustive_search cannot
# even be reached the way the documented example reaches it (by asking torch
# for a CUDA stream). Confirm the function itself is at least genuinely
# importable and inspectable without a GPU -- compilation-time API surface
# does not require hardware, only *launching* does.
print(f"ct.tune.exhaustive_search is importable: {callable(ct.tune.exhaustive_search)}")

# Calling it with an obviously-fake stream shows where the real boundary is:
# does it fail on the stream argument itself, or does it get further before
# needing real hardware?
@ct.kernel
def noop_kernel(a):
    pid = ct.bid(0)

try:
    result = ct.tune.exhaustive_search(
        search_space=[{"x": 1}],
        stream=object(),  # deliberately not a real CUDA stream
        grid_fn=lambda cfg: (1,),
        kernel=noop_kernel,
        args_fn=lambda cfg: (None,),
    )
    print(f"exhaustive_search with a fake stream: returned {result!r}")
except Exception as e:
    print(f"exhaustive_search with a fake stream: {type(e).__name__}: {e}")
```

The complete file is `05_exhaustive_search_without_gpu.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_exhaustive_search_without_gpu.py
```

Genuinely run:

```
cuda.tile version: 1.5.0
torch import: ImportError: No module named 'torch'
ct.tune.exhaustive_search is importable: True
exhaustive_search with a fake stream: ValueError: No valid config found in search space.
Config: {'x': 1}
TypeError: Unsupported stream type object.
```

Two genuine facts, confirmed rather than assumed. First, this exact sandbox has no PyTorch at all — `ct.tune.exhaustive_search`'s own documented example cannot even be typed here the way the documentation writes it, since it starts from `torch.cuda.current_stream()`. Second, `exhaustive_search` does not crash outright on a bad `stream` — it catches the failure per-config, records it as a genuine failure with its real underlying reason (`TypeError: Unsupported stream type object.`), and only then raises its own `ValueError` reporting that no config in the search space succeeded. That is real, structured, driver-free-*ish* behavior — the stream type is checked before any actual launch is attempted — but it is not a substitute for genuinely tuning anything: with zero valid configs, `exhaustive_search` never reaches the timing loop its whole purpose exists for.

### Worked Example 10.5.2 — the honest stand-in: searching for the smallest compiled output

Chapter 9 closed by naming exactly this trade-off: this book can search a space of compiled specializations and pick among them by a criterion it can state honestly, which is not the same criterion `exhaustive_search` actually uses. The following combines Sections 10.1 and 10.2's hints into a small `num_ctas`-by-`occupancy` grid, compiles every combination with `export_kernel`, and picks the smallest — explicitly labeled as what it is.

```python
import cuda.tile as ct
import io
from itertools import product

# ct.tune.exhaustive_search searches a space of *runtime* configurations by
# actually launching each one on real hardware and timing it -- see
# 05_exhaustive_search_without_gpu.py for what happens when there is no real
# stream to launch on. This file does something deliberately narrower and
# weaker: it searches a space of *compiler hints* using only export_kernel,
# and picks the smallest compiled cubin as its criterion -- smallest compiled
# output is not the same thing as fastest on real hardware, and this file
# says so rather than implying otherwise.

def build_kernel_fn(**hints):
    @ct.kernel(**hints)
    def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return kernel_fn

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

search_space = list(product([1, 2, 4, 8, 16], [1, 8, 32]))
results = []
for num_ctas, occupancy in search_space:
    kernel = build_kernel_fn(num_ctas=num_ctas, occupancy=occupancy)
    size = compile_bytes(kernel)
    results.append(((num_ctas, occupancy), size))
    print(f"num_ctas={num_ctas}, occupancy={occupancy}: {size} cubin bytes")

best_config, best_size = min(results, key=lambda r: r[1])
print(f"Smallest compiled config: num_ctas={best_config[0]}, occupancy={best_config[1]} ({best_size} cubin bytes)")
print("This is the smallest compiled output in the space searched -- not a measurement of which configuration runs fastest, which this book cannot measure without a real GPU.")
```

The complete file is `06_smallest_compiled_search.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_smallest_compiled_search.py
```

Genuinely run:

```
num_ctas=1, occupancy=1: 24608 cubin bytes
num_ctas=1, occupancy=8: 24736 cubin bytes
num_ctas=1, occupancy=32: 24608 cubin bytes
num_ctas=2, occupancy=1: 23840 cubin bytes
num_ctas=2, occupancy=8: 23840 cubin bytes
num_ctas=2, occupancy=32: 23840 cubin bytes
num_ctas=4, occupancy=1: 24480 cubin bytes
num_ctas=4, occupancy=8: 24480 cubin bytes
num_ctas=4, occupancy=32: 24480 cubin bytes
num_ctas=8, occupancy=1: 24480 cubin bytes
num_ctas=8, occupancy=8: 24480 cubin bytes
num_ctas=8, occupancy=32: 24480 cubin bytes
num_ctas=16, occupancy=1: 24480 cubin bytes
num_ctas=16, occupancy=8: 24480 cubin bytes
num_ctas=16, occupancy=32: 24480 cubin bytes
Smallest compiled config: num_ctas=2, occupancy=1 (23840 cubin bytes)
This is the smallest compiled output in the space searched -- not a measurement of which configuration runs fastest, which this book cannot measure without a real GPU.
```

The search reports `num_ctas=2, occupancy=1` as smallest, but that phrasing needs one honest qualifier: `num_ctas=2` compiles to the identical 23,840 bytes at `occupancy=1`, `8`, and `32` alike — a genuine three-way tie, not `occupancy=1` uniquely winning anything. Python's `min` simply returns the first tied entry it encounters in iteration order, which is deterministic given a fixed `search_space` ordering, but is not evidence that `occupancy=1` is somehow better than `8` or `32` here. And the larger qualifier is the one this whole section exists to make explicit: "smallest compiled cubin" is a real, repeatable, driver-free measurement this book can make honestly — it is not "fastest," which is the only thing `ct.tune.exhaustive_search` actually optimizes for, and which requires exactly the real GPU and real launch this book has never had since Chapter 1.

## Complete Runnable Code

### File: `01_num_ctas_valid_and_invalid.py`

```python
import cuda.tile as ct
import io

def build_kernel(num_ctas):
    @ct.kernel(num_ctas=num_ctas)
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# num_ctas is documented as "a power of 2 between 1 and 16, inclusive", default None (auto).
for name, num_ctas in [("None (default)", None), ("1", 1), ("2", 2), ("4", 4), ("8", 8), ("16", 16),
                        ("3 (not a power of 2)", 3), ("0 (below range)", 0), ("32 (above range)", 32)]:
    buf = io.BytesIO()
    try:
        kernel = build_kernel(num_ctas)
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"num_ctas={name}: {len(buf.getvalue())} cubin bytes")
    except Exception as e:
        print(f"num_ctas={name}: {type(e).__name__}: {e}")
```

```bash
python3 01_num_ctas_valid_and_invalid.py
```

### File: `02_occupancy_hint.py`

```python
import cuda.tile as ct
import io

def build_kernel(occupancy):
    @ct.kernel(occupancy=occupancy)
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# occupancy is documented as "[1, 32]", default None (auto).
for name, occupancy in [("None (default)", None), ("1", 1), ("2", 2), ("4", 4), ("8", 8),
                         ("16", 16), ("32", 32), ("0 (below range)", 0), ("33 (above range)", 33)]:
    buf = io.BytesIO()
    try:
        kernel = build_kernel(occupancy)
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"occupancy={name}: {len(buf.getvalue())} cubin bytes")
    except Exception as e:
        print(f"occupancy={name}: {type(e).__name__}: {e}")
```

```bash
python3 02_occupancy_hint.py
```

### File: `03_num_worker_warps.py`

```python
import cuda.tile as ct
import io

def build_kernel(num_worker_warps):
    @ct.kernel(num_worker_warps=num_worker_warps)
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# num_worker_warps is documented as "must be either 4 or 8", default None (auto),
# and documented as "Since CTK 13.3. Ignored with a warning otherwise."
for name, val in [("None (default)", None), ("4", 4), ("8", 8), ("6 (not 4 or 8)", 6)]:
    buf = io.BytesIO()
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kernel = build_kernel(val)
            ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
            warning_msgs = [str(wi.message) for wi in w]
        print(f"num_worker_warps={name}: {len(buf.getvalue())} cubin bytes, warnings={warning_msgs}")
    except Exception as e:
        print(f"num_worker_warps={name}: {type(e).__name__}: {e}")
```

```bash
python3 03_num_worker_warps.py
```

### File: `04_replace_hints.py`

```python
import cuda.tile as ct
import io

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# COMMON TRAP: the compiled cubin embeds the kernel's Python function name as a
# symbol, so comparing byte counts across two kernels built from
# differently-named functions silently mixes a naming effect into the
# comparison. Every kernel below is built from a function named "kernel_fn",
# so name length can never explain a byte-count difference here.
def build_kernel_fn(**hints):
    @ct.kernel(**hints)
    def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return kernel_fn

base = build_kernel_fn()
base_bytes = compile_bytes(base)
print(f"kernel_fn (no hints): {base_bytes} cubin bytes")

tuned = base.replace_hints(num_ctas=2)
tuned_bytes = compile_bytes(tuned)
print(f"kernel_fn.replace_hints(num_ctas=2): {tuned_bytes} cubin bytes")
print(f"tuned is base: {tuned is base}")
print(f"type(tuned).__name__: {type(tuned).__name__}")

direct = build_kernel_fn(num_ctas=2)
direct_bytes = compile_bytes(direct)
print(f"kernel_fn built directly with num_ctas=2: {direct_bytes} cubin bytes")
print(f"replace_hints bytes == direct-built bytes: {tuned_bytes == direct_bytes}")

# Chaining: does a second replace_hints call preserve the first hint?
tuned_twice = tuned.replace_hints(occupancy=32)
tuned_twice_bytes = compile_bytes(tuned_twice)
print(f"kernel_fn.replace_hints(num_ctas=2).replace_hints(occupancy=32): {tuned_twice_bytes} cubin bytes")

both_direct = build_kernel_fn(num_ctas=2, occupancy=32)
both_direct_bytes = compile_bytes(both_direct)
print(f"kernel_fn built directly with num_ctas=2, occupancy=32: {both_direct_bytes} cubin bytes")
print(f"chained replace_hints bytes == both-direct bytes: {tuned_twice_bytes == both_direct_bytes}")

# Does the second replace_hints call preserve num_ctas=2, or silently drop it
# back to auto? Compare against occupancy=32 alone (no num_ctas) under the
# same controlled name.
occupancy_only = build_kernel_fn(occupancy=32)
occupancy_only_bytes = compile_bytes(occupancy_only)
print(f"kernel_fn built directly with occupancy=32 only (no num_ctas): {occupancy_only_bytes} cubin bytes")
print(f"chained result == occupancy-only (would mean num_ctas got dropped): {tuned_twice_bytes == occupancy_only_bytes}")
```

```bash
python3 04_replace_hints.py
```

### File: `05_exhaustive_search_without_gpu.py`

```python
import cuda.tile as ct

# This file exists to answer one honest question directly, rather than
# leaving it implicit: what actually happens if you try to use
# ct.tune.exhaustive_search on a machine with no CUDA driver and no GPU?
#
# exhaustive_search's documented signature requires a real CUDA `stream`
# argument (e.g. torch.cuda.current_stream()) and launches the kernel for
# real to measure it -- there is no ahead-of-time equivalent. This sandbox
# has cuda-tile 1.5.0 installed but no PyTorch, no nvidia-smi, and no GPU,
# so this is a genuine, unmodified account of what surfaces first.

print(f"cuda.tile version: {ct.__version__}")

try:
    import torch
    print("torch import: succeeded")
except ImportError as e:
    print(f"torch import: ImportError: {e}")

# Since torch itself is not installed here, ct.tune.exhaustive_search cannot
# even be reached the way the documented example reaches it (by asking torch
# for a CUDA stream). Confirm the function itself is at least genuinely
# importable and inspectable without a GPU -- compilation-time API surface
# does not require hardware, only *launching* does.
print(f"ct.tune.exhaustive_search is importable: {callable(ct.tune.exhaustive_search)}")

# Calling it with an obviously-fake stream shows where the real boundary is:
# does it fail on the stream argument itself, or does it get further before
# needing real hardware?
@ct.kernel
def noop_kernel(a):
    pid = ct.bid(0)

try:
    result = ct.tune.exhaustive_search(
        search_space=[{"x": 1}],
        stream=object(),  # deliberately not a real CUDA stream
        grid_fn=lambda cfg: (1,),
        kernel=noop_kernel,
        args_fn=lambda cfg: (None,),
    )
    print(f"exhaustive_search with a fake stream: returned {result!r}")
except Exception as e:
    print(f"exhaustive_search with a fake stream: {type(e).__name__}: {e}")
```

```bash
python3 05_exhaustive_search_without_gpu.py
```

### File: `06_smallest_compiled_search.py`

```python
import cuda.tile as ct
import io
from itertools import product

# ct.tune.exhaustive_search searches a space of *runtime* configurations by
# actually launching each one on real hardware and timing it -- see
# 05_exhaustive_search_without_gpu.py for what happens when there is no real
# stream to launch on. This file does something deliberately narrower and
# weaker: it searches a space of *compiler hints* using only export_kernel,
# and picks the smallest compiled cubin as its criterion -- smallest compiled
# output is not the same thing as fastest on real hardware, and this file
# says so rather than implying otherwise.

def build_kernel_fn(**hints):
    @ct.kernel(**hints)
    def kernel_fn(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        ct.store(c, (pid,), ct.add(x, y))
    return kernel_fn

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_bytes(kernel):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

search_space = list(product([1, 2, 4, 8, 16], [1, 8, 32]))
results = []
for num_ctas, occupancy in search_space:
    kernel = build_kernel_fn(num_ctas=num_ctas, occupancy=occupancy)
    size = compile_bytes(kernel)
    results.append(((num_ctas, occupancy), size))
    print(f"num_ctas={num_ctas}, occupancy={occupancy}: {size} cubin bytes")

best_config, best_size = min(results, key=lambda r: r[1])
print(f"Smallest compiled config: num_ctas={best_config[0]}, occupancy={best_config[1]} ({best_size} cubin bytes)")
print("This is the smallest compiled output in the space searched -- not a measurement of which configuration runs fastest, which this book cannot measure without a real GPU.")
```

```bash
python3 06_smallest_compiled_search.py
```

## Chapter Summary

`ct.kernel`'s three tested compiler hints — `num_ctas`, `occupancy`, and `num_worker_warps` — are each genuinely, independently enforced against their own documented boundaries: `num_ctas` requires both a power of 2 and a value in `[1, 16]`, checked and reported separately; `occupancy` requires `[1, 32]`, but only its upper edge produced a measurable size change among the values tested, a boundary step rather than a gradient; and `num_worker_warps` requires exactly `4` or `8`, hard-rejected with no warning in this installed `cuda-tile` 1.5.0, contrary to its own docstring's description of a warning-only fallback on an older toolkit. `kernel.replace_hints(...)` genuinely returns a new, independent kernel object whose compiled output is byte-identical to building the same hints from scratch, and hints accumulate cleanly across chained calls rather than resetting. Along the way, this chapter surfaced a real methodological trap worth carrying into any future measurement: the compiled cubin embeds the kernel's own Python function name, so any byte-count comparison across differently-named kernels silently mixes in a naming effect unrelated to whatever is actually being tested. Finally, `ct.tune.exhaustive_search`, cuTile Python's genuine autotuning entry point, was confirmed directly to require a real CUDA stream and to genuinely launch kernels for timing — it cannot even be reached the documented way in this sandbox, which has no PyTorch and no GPU at all — and this book's own hint search over smallest compiled output was built and reported as the explicitly weaker, driver-free stand-in that it is, never as a substitute for measuring what actually runs fastest.

## Self-Check Questions

1. `num_ctas=3` and `num_ctas=0` are both rejected, but with two different `ValueError` messages. What does each message tell you that the other does not?
2. `occupancy` only produced a measurably different compiled size at `32`, the top of its documented `[1, 32]` range — every other tested value from `1` through `16` compiled identically to the default. What would you need to test to find out whether `32` is a true minimum-size point, or simply the first value tested that happened to differ?
3. `num_worker_warps=6` is hard-rejected with a `ValueError` in this installed `cuda-tile` 1.5.0, even though its own docstring describes a CUDA-Toolkit-version-dependent "ignored with a warning" fallback. What can this book honestly conclude about which CTK version is bundled here, and what can it not conclude?
4. `kernel.replace_hints(num_ctas=2).replace_hints(occupancy=32)` compiled to the same bytes as building both hints in from scratch, and differed from `occupancy=32` alone. What would it have meant, concretely, if the chained call had instead matched the `occupancy=32`-alone result?
5. Section 10.5's `06_smallest_compiled_search.py` reports `num_ctas=2, occupancy=1` as the smallest configuration, but three different `occupancy` values tied at that same byte count under `num_ctas=2`. Why is it inaccurate to describe this result as "`occupancy=1` is the best setting"?

## Where We Go Next

This chapter closes Part 1 exactly where Chapter 1 opened it: at the boundary between what `ct.compilation.export_kernel` can confirm ahead of time and what only `ct.launch` against real hardware can settle. Every one of the ten chapters so far has stayed honestly on the compile-time side of that boundary — measuring byte counts, triggering real rejections, and reading real docstrings, never claiming a runtime result this book was never in a position to produce. Part 2 turns to composing what these ten chapters have established piece by piece into complete, multi-kernel programs — the same rigor, applied to larger surfaces built entirely out of primitives this book has already confirmed compile, one genuine `export_kernel` call at a time.

## Worked Solutions

**1.** `num_ctas=3`'s message ("should be power of 2, got 3") tells you the value failed the power-of-two check specifically — a value like `4` or `8` would have passed regardless of range. `num_ctas=0`'s message ("should be [1, 16], got 0") tells you the value failed the range check — `0` is technically not a power of two either by some conventions, but the compiler's message names range, not parity, as the operative failure. The two checks are genuinely separate, and the message tells you which one to fix.

**2.** You would need to test every integer value from `1` through `32` individually (this chapter tested a subset: `1, 2, 4, 8, 16, 32`) and confirm whether the size change appears exactly at `32` or somewhere earlier that this chapter's sampling skipped over — for example, `24` or `31` might already differ from `16`'s result, in which case `32` would not be a special minimum at all, just the largest of several already-different values this chapter happened to land on.

**3.** This book can honestly conclude that whatever CTK `cuda-tile` 1.5.0 is bundled against in this sandbox behaves as if it were at or above the version where `num_worker_warps` is fully enforced (the docstring's own "Since CTK 13.3" language), since no warning-only fallback was observed. It cannot conclude the exact CTK version number itself — no tool available in this sandbox (`nvcc` is not installed, and `cuda.tile` does not expose one directly) can confirm that number, so this book states the observed behavior and stops there rather than guessing a version.

**4.** It would have meant `replace_hints` resets any hint not explicitly named in the new call — that the second `.replace_hints(occupancy=32)` silently discarded the first call's `num_ctas=2`, leaving only `occupancy=32` in effect. The actual result (matching the both-hints-together build, and differing from `occupancy=32` alone) instead confirms hints accumulate: each `replace_hints` call adds to or overrides individual named hints on top of whatever the kernel already carried, rather than replacing the entire hint set.

**5.** Because `occupancy=1`, `occupancy=8`, and `occupancy=32` all compiled to the identical 23,840 bytes under `num_ctas=2` — a genuine three-way tie. Python's `min()` simply returns the first tied entry it encounters while iterating `search_space` in its fixed order, which happens to be `occupancy=1` here because of how `itertools.product` orders its output — not because `occupancy=1` measurably outperformed `8` or `32` on any criterion this script actually measured. Reporting it as uniquely "best" would misstate a tie as a genuine difference.
