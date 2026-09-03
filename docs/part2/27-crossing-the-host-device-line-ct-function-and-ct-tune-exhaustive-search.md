# Chapter 27: Crossing the Host/Device Line with ct.function and ct.tune.exhaustive_search

> "Every kernel this book has written so far has lived entirely on one side of a line it never had to name. `ct.function(host=True)` names that line — and `ct.tune.exhaustive_search` immediately walks right up to the one wall on the other side this book has always had."

**What you will understand by the end of this chapter:**

- `ct.function(func=None, /, *, host=False, tile=True)`: the decorator that names a function's *execution spaces* — where it may legally be called from — rather than changing anything about tile code the compiler emits
- That the default `@ct.function()` rejects a direct host-code call with a plain Python `RuntimeError`, not any `Tile*` exception this book has catalogued, because the rejection happens entirely outside compilation
- A self-caught naming-confound repair in the style of Chapter 10: an initial, uncontrolled byte-count comparison across decorator forms is corrected by a second, fully naming-controlled one, which finds all three forms compile to *identical* bytes
- `host=True`'s actual, sole observable effect: it lets one function definition be called directly from host-side Python *and* from inside a kernel body, with zero compiled-byte difference from leaving the function undecorated
- That the documented `tile` parameter — "whether the function can be called from tile code," default `True` — has no enforced effect this book's testing surface can detect: `tile=False` never once blocks a tile-code call, in either combination tested
- `ct.tune.exhaustive_search`'s own pre-launch argument validation: an empty search space is rejected immediately, and a non-empty one whose every candidate fails is reported through a single summarizing exception naming the *first* config attempted, confirmed by reversing the list
- That `args_fn` failures are caught by the exact same per-candidate mechanism as a missing CUDA stream, with no distinction between "ordinary Python bug" and "hardware-related failure" at the reporting level
- A capstone wiring a `host=True` function into both a `grid_fn` callback and the kernel body it drives, fed into a real `exhaustive_search` call that honestly reaches this book's now-familiar no-stream wall rather than fabricating a result

**What you need to know first:**

- Chapter 10's naming-confound discipline (kernel and helper names must be held constant across any byte-count comparison), used to correct this chapter's own first, uncontrolled observation.
- This book's `export_kernel`-only, driver-free verification method, and the earlier confirmation that this sandbox has no GPU, no `torch`, and no `nvidia-smi` — directly relevant here, since `ct.tune.exhaustive_search`'s own documented example requires a real CUDA stream and a real `ct.launch`.
- The `Tile*` exception hierarchy this book has been cataloguing since Chapter 1, useful for noticing when an error is instead a plain Python exception raised outside compilation entirely.

## 27.1 ct.function and the Default Host/Device Boundary

### Intuition

Every kernel body written so far in this book has lived entirely inside `@ct.kernel` — tile code. A plain, undecorated Python helper function called from inside a kernel body has always worked without any special declaration, because cuTile Python treats an unannotated function as usable wherever it's called from: if a tile function calls it, it becomes tile code too, automatically and recursively, with no annotation required. `ct.function` exists to make that boundary explicit, and to open a second one this book has never used: functions callable directly from ordinary host-side Python as well.

### Background

`ct.function(func=None, /, *, host=False, tile=True)` — usable bare (`@ct.function`) or with keyword arguments (`@ct.function(host=True)`). `host` controls whether the decorated function can be called from host code; its default, `False`, matches how an undecorated helper already behaves for tile-code purposes, but a bare `@ct.function()` additionally installs a dispatch check that actively rejects a direct host-side call, rather than merely never having been designed for one.

### Worked Example 27.1.1 — an undecorated helper vs. the default @ct.function() decorator

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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# An UNDECORATED helper function: usable directly from host-side Python
# code (an ordinary function call) AND, per ct.function's own
# docstring, implicitly treated as tile code when called from within a
# kernel -- no explicit annotation required.
def plain_helper(x):
    return x + 1

print("calling plain undecorated helper from host code:", plain_helper(5))

@ct.kernel
def kernel_plain_helper(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = plain_helper(x)
    ct.store(c, (0,), y)
print(f"plain helper called from tile code: {compile_bytes(kernel_plain_helper, sig1)} cubin bytes")

# The DEFAULT @ct.function() decorator (host=False, tile=True) wraps
# the function so that calling it directly from host code is rejected
# -- a plain Python RuntimeError, not a Tile* exception, since this
# happens entirely outside compilation, before the compiler is ever
# invoked.
@ct.function
def default_decorated_helper(x):
    return x + 1

print("calling @ct.function()-decorated helper from host code:")
try:
    result = default_decorated_helper(5)
    print(f"result: {result} (unexpected)")
except Exception as e:
    print(f"{type(e).__name__}: {e}")

@ct.kernel
def kernel_default_decorated_helper(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = default_decorated_helper(x)
    ct.store(c, (0,), y)
print(f"@ct.function()-decorated helper called from tile code: {compile_bytes(kernel_default_decorated_helper, sig1)} cubin bytes")
```

Genuinely run:

```
calling plain undecorated helper from host code: 6
plain helper called from tile code: 20832 cubin bytes
calling @ct.function()-decorated helper from host code:
RuntimeError: Tile functions can only be called from tile code.
@ct.function()-decorated helper called from tile code: 21232 cubin bytes
```

### Discussion

The plain helper works both ways with no fuss at all: called from host code it's just a Python function (`6`), and called from inside `kernel_plain_helper` it's inlined into tile code and compiles to 20832 bytes. The `@ct.function()`-decorated helper, called directly from host code, is rejected with `RuntimeError: Tile functions can only be called from tile code.` — a plain Python `RuntimeError`, not any `Tile*` exception this book has been cataloguing since Chapter 1, because the rejection is ordinary Python dispatch logic that runs before the compiler is ever invoked. Called correctly, from inside `kernel_default_decorated_helper`, it compiles to 21232 bytes.

That gap — 20832 versus 21232 — invites an immediate conclusion: decorating a tile-called helper with `@ct.function()` costs bytes. But the two kernels have different names (`kernel_plain_helper` vs. `kernel_default_decorated_helper`) and the two helpers have different names (`plain_helper` vs. `default_decorated_helper`), which means Chapter 10's naming confound is sitting directly inside this comparison, uncontrolled. Before this book trusts that conclusion, Section 27.2 repeats the identical comparison with every name held constant.

## 27.2 A Naming-Controlled Comparison Across All Three Decorator Forms

### Intuition

Section 27.1's byte-count gap might be entirely real, or it might be nothing but Chapter 10's naming confound reappearing in a new guise. The only way to know is to rerun the same three decorator choices — undecorated, default `@ct.function()`, and `@ct.function(host=True)` — with the kernel and the helper both named identically every time, changing only the decorator under test.

### Background

All three variants below use `kernel_fn` as the kernel's name and `helper` as the helper's name, throughout. `host=True` is added as the third variant specifically because it's the form this chapter's later capstone needs: a function callable from both sides of the host/device line at once.

### Worked Example 27.2.1 — undecorated, @ct.function(), and @ct.function(host=True), naming held constant

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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Section 27.1's two byte counts (20832 vs 21232) differed -- but that
# comparison used two DIFFERENT kernel names and two DIFFERENT helper
# names, so Chapter 10's naming confound was never controlled for. This
# section repeats the same three decorator choices with the kernel AND
# the helper both named identically ("kernel_fn" and "helper") every
# time, changing only the decorator.
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n_plain = compile_bytes(kernel_fn, sig1)
print(f"kernel calling plain undecorated helper: {n_plain} cubin bytes")

@ct.function
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n_default = compile_bytes(kernel_fn, sig1)
print(f"kernel calling @ct.function()-decorated helper: {n_default} cubin bytes")
print(f"identical bytes: {n_plain == n_default}")

@ct.function(host=True)
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n_host_true = compile_bytes(kernel_fn, sig1)
print(f"kernel calling @ct.function(host=True)-decorated helper: {n_host_true} cubin bytes")
print(f"plain == host_true: {n_plain == n_host_true}")

# And host=True's whole point: the identical function, called directly
# from host code, with no dispatch rejection at all.
print("calling @ct.function(host=True)-decorated helper from host code:", helper(5))
```

Genuinely run:

```
kernel calling plain undecorated helper: 20576 cubin bytes
kernel calling @ct.function()-decorated helper: 20576 cubin bytes
identical bytes: True
kernel calling @ct.function(host=True)-decorated helper: 20576 cubin bytes
plain == host_true: True
calling @ct.function(host=True)-decorated helper from host code: 6
```

### Discussion

Under full naming control, all three decorator forms compile to the identical 20576 bytes. Section 27.1's gap was Chapter 10's naming confound after all: decorating a tile-called helper with `@ct.function()`, in any of its forms tested here, has no effect whatsoever on the compiled tile code. What differs is only what happens when the *same* function is called from host-side Python: the default form (whether left undecorated or wrapped in a bare `@ct.function()`) rejects that call, while `@ct.function(host=True)` accepts it, returning `6` exactly as a plain Python call would. `host=True`'s entire distinguishing effect, then, is on the host-code dispatch path — on *who may call the function* — not on anything the compiler emits for tile code at all, which is exactly what a "host/device boundary" ought to mean: a gate, not a code-generation knob.

## 27.3 The tile Parameter: Documented but Unenforced

### Intuition

`ct.function`'s docstring documents `tile` as controlling "whether the function can be called from tile code," defaulting to `True`. Setting it to `False` alongside `host=True` should, by that description, make a function host-only: callable directly from Python, but rejected the moment a kernel tries to call it during compilation. And `host=False, tile=False` together should, by the same logic, make a function unusable from *either* side of the line at all.

### Background

Both combinations are tested below, holding the kernel and helper names constant at `kernel_fn`/`helper` throughout, as in Section 27.2.

### Worked Example 27.3.1 — tile=False in two combinations

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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.function's docstring documents "tile" as controlling whether a
# function "can be called from tile code" (default True). Setting it
# to False, alongside host=True, should therefore -- per the
# documented meaning -- make a function host-only and reject a call
# from within a kernel being compiled.
@ct.function(host=True, tile=False)
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_fn, sig1)
    print(f"kernel calling @ct.function(host=True, tile=False)-decorated helper: {n} cubin bytes")
except Exception as e:
    print(f"kernel calling @ct.function(host=True, tile=False)-decorated helper: {type(e).__name__}: {e}")

print("calling the same helper directly from host code:", helper(5))

# For completeness: host=False, tile=False. If "tile" genuinely gated
# tile-code usage, this combination would be rejected from BOTH host
# code (host=False) and tile code (tile=False) -- unusable anywhere at
# all. It isn't: it behaves identically to the plain @ct.function()
# default in every observable way.
@ct.function(host=False, tile=False)
def helper(x):
    return x + 1

print("calling @ct.function(host=False, tile=False)-decorated helper from host code:")
try:
    result = helper(5)
    print(f"result: {result} (unexpected)")
except Exception as e:
    print(f"{type(e).__name__}: {e}")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n = compile_bytes(kernel_fn, sig1)
print(f"kernel calling @ct.function(host=False, tile=False)-decorated helper: {n} cubin bytes")
```

Genuinely run:

```
kernel calling @ct.function(host=True, tile=False)-decorated helper: 20576 cubin bytes
calling the same helper directly from host code: 6
calling @ct.function(host=False, tile=False)-decorated helper from host code:
RuntimeError: Tile functions can only be called from tile code.
kernel calling @ct.function(host=False, tile=False)-decorated helper: 20576 cubin bytes
```

### Discussion

Neither prediction holds. `host=True, tile=False` compiles cleanly from tile code (20576 bytes — the same figure Section 27.2 found for every naming-controlled variant), even though the docstring says `tile=False` should mean the function can't be called from tile code at all; the same function, called directly from host code, returns `6` exactly as `host=True` alone would produce. `host=False, tile=False` fares no differently from the plain, undecorated default: calling it from host code still raises the identical `RuntimeError: Tile functions can only be called from tile code.`, and calling it from inside `kernel_fn` still compiles to 20576 bytes — the same as if `tile=False` had never been written at all. In both combinations tested, `tile=False` changed nothing this book's `export_kernel`-only method could detect. This sits in the same family as Chapter 22's `cat` docstring decomposition and Chapter 19's untested `bitcast` recommendation: a documented parameter, accepted without complaint, that this book's genuine testing surface simply cannot find doing anything in the installed 1.5.0 release. That gap is reported plainly, as a gap — not corrected, and not treated as proof the parameter does nothing in every version or every code path this book hasn't tested.

## 27.4 ct.tune.exhaustive_search: Search-Space Validation and First-Error Reporting

### Intuition

cuTile Python ships an autotuning entry point, `ct.tune.exhaustive_search`, meant to try every candidate configuration in a search space, time each one on a real CUDA stream, and report the fastest. Every ingredient that would let this book actually *measure* anything — a real stream, a real `ct.launch` — has been unreachable since this book confirmed, early on, that this sandbox has no GPU, no `torch`, and no `nvidia-smi` at all. But `exhaustive_search` validates its own arguments in ordinary Python before it ever needs a stream, and that validation is fully genuine, testable ground this book's method can stand on.

### Background

`exhaustive_search(search_space, stream, grid_fn, kernel, args_fn, hints_fn=None, *, quiet=False, single_run_timeout_sec=None) -> TuningResult` searches the given configs and returns a `TuningResult` holding the best `Measurement` found, alongside its full lists of `successes` and `failures`. Every genuine call this section makes, however, ends by *raising* rather than returning — because with `stream=None`, every single candidate fails, and `exhaustive_search` apparently raises a summarizing exception rather than returning an all-failure `TuningResult`, once nothing at all has succeeded.

### Worked Example 27.4.1 — an empty search space, then three configs with no real stream

```python
import cuda.tile as ct
import cuda.tile.tune as tune

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

# ct.tune.exhaustive_search's own docstring example requires a real
# CUDA stream (torch.cuda.current_stream()) and a real ct.launch --
# both unreachable in this book's export_kernel-only, driver-free
# environment. But exhaustive_search validates its own arguments before
# ever touching a stream: an empty search space is rejected immediately
# with a plain Python ValueError, not a Tile* exception, since this
# check happens entirely in ordinary Python before compilation.
try:
    result = tune.exhaustive_search([], None, lambda cfg: (1,), kernel_fn, lambda cfg: ())
    print(f"exhaustive_search([]): {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search([]): {type(e).__name__}: {e}")

# A non-empty search space with stream=None reaches further: it
# attempts each config in order, catching whatever exception each one
# raises (here, every config fails identically, since there is no real
# stream to run on) and only raises once EVERY config has failed --
# reporting the FIRST config's own error specifically, not a generic
# summary. Three distinct configs confirm the reported config really is
# the first one in search_space order, not an arbitrary or last one.
three_configs = [{"tag": "first"}, {"tag": "second"}, {"tag": "third"}]
try:
    result = tune.exhaustive_search(three_configs, None, lambda cfg: (1,), kernel_fn, lambda cfg: (), quiet=True)
    print(f"exhaustive_search(three configs, stream=None): {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search(three configs, stream=None): {type(e).__name__}: {e}")

# Reversing the same three configs' order confirms the reported failure
# really does track search_space order, rather than always naming
# "first" regardless of position.
reversed_configs = [{"tag": "third"}, {"tag": "second"}, {"tag": "first"}]
try:
    result = tune.exhaustive_search(reversed_configs, None, lambda cfg: (1,), kernel_fn, lambda cfg: (), quiet=True)
    print(f"exhaustive_search(reversed configs, stream=None): {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search(reversed configs, stream=None): {type(e).__name__}: {e}")
```

Genuinely run:

```
exhaustive_search([]): ValueError: Search space is empty.
exhaustive_search(three configs, stream=None): ValueError: No valid config found in search space.
Config: {'tag': 'first'}
TypeError: Stream is required, got None
exhaustive_search(reversed configs, stream=None): ValueError: No valid config found in search space.
Config: {'tag': 'third'}
TypeError: Stream is required, got None
```

### Discussion

The empty-list case is rejected immediately with `ValueError: Search space is empty.` — no stream, no kernel compilation, nothing beyond checking the length of one list. Once the search space is non-empty but every candidate is doomed, `exhaustive_search` doesn't propagate the first exception it hits and stop; it catches each config's own failure, moves on to the next, and only after every candidate has failed does it raise one summarizing `ValueError: No valid config found in search space.`, with the specific failing config and that config's own exception type and message appended beneath it. The two three-config tests confirm exactly which failure gets reported: the config tagged `"first"` when it's first in the list, and `"third"` when the same three configs are reversed and `"third"` is now first — the reported failure tracks list order, not any property of the tags themselves. Whether `exhaustive_search` would stop early at the first *success* it encountered (returning a partial `TuningResult` rather than trying every remaining candidate) isn't something this test can distinguish, since all three configs here are equally doomed by the missing stream — but the identity of the reported failure is unambiguous, and it is exactly "the first one attempted."

## 27.5 args_fn Failures Are Caught the Same Way

### Intuition

`args_fn` is presumably invoked lazily, once `exhaustive_search` actually tries to prepare a given config for timing. If that's right, an `args_fn` that raises for its own config should look, from the caller's perspective, exactly like Section 27.4's missing-stream failure: one more way a single candidate can fail, not a different kind of error requiring different handling.

### Background

A single-config search space is paired with an `args_fn` that unconditionally raises `ValueError`.

### Worked Example 27.5.1 — a broken args_fn, caught per-candidate

```python
import cuda.tile as ct
import cuda.tile.tune as tune

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

# args_fn is only invoked lazily, once exhaustive_search actually tries
# to warm up a given config -- so an args_fn that raises for its own
# config is caught by the exact same per-config try/except as a stream
# error, and reported the same way: as that config's own failure, not
# a crash of the whole search.
def bad_args_fn(cfg):
    raise ValueError(f"cannot build args for config {cfg}")

try:
    result = tune.exhaustive_search([{"tag": "only"}], None, lambda cfg: (1,), kernel_fn, bad_args_fn, quiet=True)
    print(f"exhaustive_search with failing args_fn: {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search with failing args_fn: {type(e).__name__}: {e}")
```

Genuinely run:

```
exhaustive_search with failing args_fn: ValueError: No valid config found in search space.
Config: {'tag': 'only'}
ValueError: cannot build args for config {'tag': 'only'}
```

### Discussion

`bad_args_fn`'s own `ValueError: cannot build args for config {'tag': 'only'}` is caught by the identical mechanism Section 27.4 already showed handling a missing-stream `TypeError`: reported as that one config's own failure, wrapped in the same `ValueError: No valid config found in search space.` summary. Nothing distinguishes "the stream doesn't exist" from "`args_fn` itself is broken" at the reporting level — `exhaustive_search`'s per-candidate handling appears to wrap every step of preparing and running one config behind one uniform boundary, rather than treating stream errors, argument-building errors, and launch errors as separately-reported stages. That uniformity is exactly what lets this driver-free sandbox exercise `exhaustive_search`'s error-reporting machinery honestly: a failure with nothing to do with CUDA at all — an ordinary bug in user-supplied Python — still surfaces through precisely the path a real, hardware-related failure would use.

## 27.6 Capstone: A Host Function Shared Between tune's Driver Code and the Kernel It Tunes

### Intuition

Section 27.2 found `host=True`'s entire purpose is letting one function definition serve both host-side Python and tile code without duplicating it. The most natural place that shows up in practice is exactly the kind of driver code `exhaustive_search` expects: a `grid_fn` callback that computes launch dimensions from a config, sitting right alongside the kernel body those dimensions describe.

### Background

`scale_factor`, decorated `@ct.function(host=True)`, is used directly inside a `grid_fn` lambda (ordinary host-side Python) and, separately, from inside `kernel_scaled`'s tile-code body — one definition, both call sites. The resulting kernel and `grid_fn` are then wired into one real `exhaustive_search` call.

### Worked Example 27.6.1 — one host=True function, two call sites, one real exhaustive_search call

```python
import cuda.tile as ct
import cuda.tile.tune as tune
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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: a single @ct.function(host=True)-decorated helper, used
# BOTH by ordinary host-side driver code (here, a grid_fn callback of
# the kind ct.tune.exhaustive_search expects) AND from inside the
# kernel body it tunes -- one definition, shared across the host/device
# line, which is the entire point of host=True existing.
@ct.function(host=True)
def scale_factor(n):
    return n * 2

grid_fn = lambda cfg: (scale_factor(cfg["blocks"]),)
print(f"scale_factor used directly in host-side grid_fn: grid_fn({{'blocks': 3}}) = {grid_fn({'blocks': 3})}")

@ct.kernel
def kernel_scaled(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = x * scale_factor(2)
    ct.store(c, (0,), y)
print(f"the SAME scale_factor called from inside the kernel body: {compile_bytes(kernel_scaled, sig1)} cubin bytes")

# Wiring this kernel into an actual exhaustive_search call, to show the
# intended integration end to end -- honestly reaching the same wall
# this chapter has already documented: no real CUDA stream, so no
# config can ever converge, and the search reports that plainly rather
# than silently pretending to tune anything.
def args_fn(cfg):
    return ()

try:
    result = tune.exhaustive_search([{"blocks": 3}], None, grid_fn, kernel_scaled, args_fn, quiet=True)
    print(f"exhaustive_search wired to kernel_scaled: {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search wired to kernel_scaled: {type(e).__name__}: {e}")
```

Genuinely run:

```
scale_factor used directly in host-side grid_fn: grid_fn({'blocks': 3}) = (6,)
the SAME scale_factor called from inside the kernel body: 20384 cubin bytes
exhaustive_search wired to kernel_scaled: ValueError: No valid config found in search space.
Config: {'blocks': 3}
TypeError: Stream is required, got None
```

### Discussion

`scale_factor` works both ways from one definition: called directly as ordinary Python inside `grid_fn`, `grid_fn({'blocks': 3})` returns `(6,)`, and called from inside `kernel_scaled`'s tile code, the same function compiles cleanly to 20384 bytes. Wiring the same kernel and the same `grid_fn` into `exhaustive_search` reaches exactly the wall Section 27.4 already named: with `stream=None`, the one candidate config fails for the identical reason every earlier one did (`TypeError: Stream is required, got None`), and the search reports it through the same `ValueError: No valid config found in search space.` path. This chapter cannot show `exhaustive_search` actually selecting a winning configuration — that would require a real CUDA stream and a real launch, permanently outside this book's `export_kernel`-only, driver-free method. What it can show, honestly and completely, is everything up to that wall: a host function shared correctly across both sides of the line `ct.function` draws, wired into the real tuning entry point, failing for precisely the reason this book has been explicit about since it first confirmed there is no GPU or `torch` installation here to launch anything on at all.

## Chapter Summary

This chapter crossed a line every earlier kernel in this book had stayed on one side of. `ct.function(host=False|True, tile=True|False)` names a function's execution spaces rather than changing anything about the tile code the compiler emits: an initial, uncontrolled comparison across decorator forms suggested otherwise, but Chapter 10's naming confound was sitting inside that comparison, and a properly naming-controlled rerun found all three forms — undecorated, default `@ct.function()`, and `@ct.function(host=True)` — compile to the identical byte count. `host=True`'s entire observable effect is on the host-code dispatch path: it lets a function be called directly from Python in addition to tile code, with zero change to what gets compiled. The documented `tile` parameter, by contrast, showed no enforced effect at all in either combination tested — a genuine documentation-vs-behavior gap, reported as a gap rather than smoothed over. `ct.tune.exhaustive_search` then supplied this chapter's second half: an autotuning entry point whose real purpose — measuring real kernel launches across a search space — is permanently unreachable in this GPU-less, torch-less sandbox, but whose own pre-launch argument validation is fully genuine and testable. An empty search space is rejected immediately; a non-empty one whose every candidate fails is reported through one summarizing exception naming the first config attempted, confirmed by reversing the list; and an `args_fn` failure is caught by the exact same per-candidate mechanism as a missing stream, with no special-casing between an ordinary Python bug and a hardware-related one. The capstone combined both halves of the chapter: a `host=True` function shared identically between a `grid_fn` callback and the kernel body it drives, wired into a real `exhaustive_search` call that reaches the same no-stream wall honestly, rather than fabricating a result this book's method has no way to produce.

## Self-Check Questions

1. Section 27.1's uncontrolled comparison (20832 vs. 21232 bytes) and Section 27.2's naming-controlled rerun (20576 bytes for all three forms) reached opposite conclusions about whether `@ct.function()` costs bytes. What specifically had to be held constant between the two kernels for Section 27.2's comparison to be trustworthy, and why was Section 27.1's comparison not simply "wrong," but rather uninterpretable on its own?
2. Section 27.3 found `tile=False` had no detectable effect in either combination tested (`host=True, tile=False` and `host=False, tile=False`). Suppose a future cuTile Python release genuinely started enforcing `tile=False`. What specific, currently-passing line from Section 27.3's worked example would you expect to start failing, and with what kind of exception, given the pattern this book has seen for every other enforced restriction?
3. Section 27.4 confirmed that `exhaustive_search`'s failure message names the *first* config attempted, using a forward list and a reversed list of the same three configs. Why was testing with the SAME three configs in reversed order a stronger confirmation than testing with three entirely different configs in a different order would have been?
4. Section 27.5 found that `args_fn`'s own exception is caught by the identical mechanism as a missing-stream `TypeError`. What would it mean for cuTile Python's design if, instead, `args_fn` errors were allowed to propagate directly out of `exhaustive_search` uncaught, rather than being wrapped in the same per-candidate reporting as every other kind of per-config failure?
5. The capstone reaches `TypeError: Stream is required, got None` — the exact same wall this book met in Section 27.4. Given everything this chapter was able to demonstrate about `exhaustive_search` without a real stream, what is the smallest additional capability (beyond simply "a real GPU") this book's method would need in order to show `exhaustive_search` actually returning a `TuningResult` rather than raising?

## Where We Go Next

This chapter closed the host/device question Chapter 26's own closing section raised — `ct.function(host=True)` turned out to be a dispatch gate on who may call a function, not a lever on what tile code gets compiled — and pushed `ct.tune.exhaustive_search` as far as a GPU-less, torch-less sandbox honestly allows: past every argument-validation and error-reporting path, right up to the wall a real stream would be needed to cross. A different, entirely untouched family sits in `dir(ct)`, confirmed by direct introspection rather than assumption: `atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_or`, `atomic_xchg`, and `atomic_xor`. Every kernel this book has written treats each tile's memory access as if nothing else were touching the same memory at the same time — a concept this book has never once needed, and atomic operations exist specifically for the case where that assumption stops holding.

## Worked Solutions

**1.** For Section 27.2's comparison to be trustworthy, the kernel's own name and the helper's own name both had to be held constant (`kernel_fn` and `helper`, respectively) across all three decorator variants, following exactly the discipline Chapter 10 established when it discovered that a kernel's function name embeds into its compiled cubin and can shift the byte count on its own, independent of anything else changing. Section 27.1's comparison wasn't simply "wrong" in the sense of getting a false number — both byte counts it reported were genuine, correctly-measured compilation results. It was uninterpretable because it changed two things at once (the decorator AND both names), so there was no way to attribute the observed gap to either cause specifically. The gap could have come entirely from the naming confound, entirely from the decorator, or some mix of both — and Section 27.1's own data contained no way to tell those apart. Only by holding everything else fixed and changing one variable at a time could Section 27.2 isolate the decorator's actual effect.

**2.** The line most likely to start failing is the `compile_bytes(kernel_fn, sig1)` call inside the `try` block for `@ct.function(host=True, tile=False)` — the one that currently compiles successfully to 20576 bytes despite `tile=False` supposedly forbidding exactly that call. If `tile=False` began being enforced, this book's own pattern for every other enforced restriction points toward a `TileTypeError` or a new, more specific `Tile*` exception raised during compilation (analogous to how `TileUnsupportedFeatureError` or `TileTypeError` have appeared throughout this book whenever tile code attempts something the compiler explicitly disallows) rather than a plain Python `RuntimeError` — because a plain `RuntimeError` in this chapter has consistently marked a *host-side* dispatch rejection that happens before compilation ever starts, while a tile-code-side rejection would need to happen during compilation, which is exactly where this book's `Tile*` hierarchy has always originated.

**3.** Testing with the SAME three configs, merely reordered, isolates search_space order as the only variable that changed between the two calls — the tags `"first"`, `"second"`, and `"third"` are identical objects doing identical (failing) work in both runs, so the only thing that could explain the reported failure changing from `"first"` to `"third"` is its position in the list. If three entirely different configs had been used in a different order instead, two variables would have changed at once (which configs, and what order they're in), leaving open the possibility that some property of the configs themselves — rather than their position — determined which one got reported. Reusing the identical three tags under reversal is this chapter's own version of Chapter 10's naming-confound discipline: change exactly one thing, and only one thing, between two runs being compared.

**4.** Letting `args_fn` errors propagate uncaught would mean `exhaustive_search` treats "the user's own argument-construction code is buggy" as categorically different from "this particular config isn't runnable" — a design that trusts `args_fn` more than it trusts the stream, the kernel, or the launch itself. That would be a defensible design (a bug in caller-supplied code arguably deserves to surface immediately and loudly, rather than being buried inside a "no valid config" summary alongside every other kind of failure), but it isn't the design this book found: every genuine test in Section 27.5 showed `args_fn`'s exception wrapped in the identical per-candidate handling as a stream error, meaning cuTile Python's actual design treats ALL per-config failures uniformly, whatever their origin, and only reports which config failed and why after every candidate has been given an equal chance to fail on its own terms.

**5.** The smallest additional capability would be a real CUDA stream object accepted in place of `None` — nothing else in this chapter's tests failed before reaching the `stream`-dependent step; the search-space validation, the config iteration, the `grid_fn` and `args_fn` calls, and even the compilation of `kernel_scaled` itself all completed successfully every time. A genuine stream doesn't strictly require a full GPU in the abstract sense the question raises as a baseline — it specifically requires whatever a real GPU and driver combination in this environment would provide (which is what "a real GPU" already implies here, since this sandbox's `torch`-less, `nvidia-smi`-less state is precisely what stands between `stream=None` and a genuine stream) — but the chapter's own tests narrow the wall down to exactly one missing ingredient, rather than "everything about running a kernel," which is worth stating plainly: this book's method got further into `exhaustive_search`'s real machinery than it might have looked capable of, and the one thing left is the one thing this book has never had.

## Complete Runnable Code

### File: `01_function_host_default_and_runtime_error.py`

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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# An UNDECORATED helper function: usable directly from host-side Python
# code (an ordinary function call) AND, per ct.function's own
# docstring, implicitly treated as tile code when called from within a
# kernel -- no explicit annotation required.
def plain_helper(x):
    return x + 1

print("calling plain undecorated helper from host code:", plain_helper(5))

@ct.kernel
def kernel_plain_helper(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = plain_helper(x)
    ct.store(c, (0,), y)
print(f"plain helper called from tile code: {compile_bytes(kernel_plain_helper, sig1)} cubin bytes")

# The DEFAULT @ct.function() decorator (host=False, tile=True) wraps
# the function so that calling it directly from host code is rejected
# -- a plain Python RuntimeError, not a Tile* exception, since this
# happens entirely outside compilation, before the compiler is ever
# invoked.
@ct.function
def default_decorated_helper(x):
    return x + 1

print("calling @ct.function()-decorated helper from host code:")
try:
    result = default_decorated_helper(5)
    print(f"result: {result} (unexpected)")
except Exception as e:
    print(f"{type(e).__name__}: {e}")

@ct.kernel
def kernel_default_decorated_helper(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = default_decorated_helper(x)
    ct.store(c, (0,), y)
print(f"@ct.function()-decorated helper called from tile code: {compile_bytes(kernel_default_decorated_helper, sig1)} cubin bytes")
```

### File: `02_naming_controlled_decorator_comparison.py`

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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Section 27.1's two byte counts (20832 vs 21232) differed -- but that
# comparison used two DIFFERENT kernel names and two DIFFERENT helper
# names, so Chapter 10's naming confound was never controlled for. This
# section repeats the same three decorator choices with the kernel AND
# the helper both named identically ("kernel_fn" and "helper") every
# time, changing only the decorator.
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n_plain = compile_bytes(kernel_fn, sig1)
print(f"kernel calling plain undecorated helper: {n_plain} cubin bytes")

@ct.function
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n_default = compile_bytes(kernel_fn, sig1)
print(f"kernel calling @ct.function()-decorated helper: {n_default} cubin bytes")
print(f"identical bytes: {n_plain == n_default}")

@ct.function(host=True)
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n_host_true = compile_bytes(kernel_fn, sig1)
print(f"kernel calling @ct.function(host=True)-decorated helper: {n_host_true} cubin bytes")
print(f"plain == host_true: {n_plain == n_host_true}")

# And host=True's whole point: the identical function, called directly
# from host code, with no dispatch rejection at all.
print("calling @ct.function(host=True)-decorated helper from host code:", helper(5))
```

### File: `03_tile_parameter_documented_but_unenforced.py`

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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.function's docstring documents "tile" as controlling whether a
# function "can be called from tile code" (default True). Setting it
# to False, alongside host=True, should therefore -- per the
# documented meaning -- make a function host-only and reject a call
# from within a kernel being compiled.
@ct.function(host=True, tile=False)
def helper(x):
    return x + 1

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_fn, sig1)
    print(f"kernel calling @ct.function(host=True, tile=False)-decorated helper: {n} cubin bytes")
except Exception as e:
    print(f"kernel calling @ct.function(host=True, tile=False)-decorated helper: {type(e).__name__}: {e}")

print("calling the same helper directly from host code:", helper(5))

# For completeness: host=False, tile=False. If "tile" genuinely gated
# tile-code usage, this combination would be rejected from BOTH host
# code (host=False) and tile code (tile=False) -- unusable anywhere at
# all. It isn't: it behaves identically to the plain @ct.function()
# default in every observable way.
@ct.function(host=False, tile=False)
def helper(x):
    return x + 1

print("calling @ct.function(host=False, tile=False)-decorated helper from host code:")
try:
    result = helper(5)
    print(f"result: {result} (unexpected)")
except Exception as e:
    print(f"{type(e).__name__}: {e}")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = helper(x)
    ct.store(c, (0,), y)
n = compile_bytes(kernel_fn, sig1)
print(f"kernel calling @ct.function(host=False, tile=False)-decorated helper: {n} cubin bytes")
```

### File: `04_tune_empty_search_space_and_first_error_ordering.py`

```python
import cuda.tile as ct
import cuda.tile.tune as tune

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

# ct.tune.exhaustive_search's own docstring example requires a real
# CUDA stream (torch.cuda.current_stream()) and a real ct.launch --
# both unreachable in this book's export_kernel-only, driver-free
# environment. But exhaustive_search validates its own arguments before
# ever touching a stream: an empty search space is rejected immediately
# with a plain Python ValueError, not a Tile* exception, since this
# check happens entirely in ordinary Python before compilation.
try:
    result = tune.exhaustive_search([], None, lambda cfg: (1,), kernel_fn, lambda cfg: ())
    print(f"exhaustive_search([]): {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search([]): {type(e).__name__}: {e}")

# A non-empty search space with stream=None reaches further: it
# attempts each config in order, catching whatever exception each one
# raises (here, every config fails identically, since there is no real
# stream to run on) and only raises once EVERY config has failed --
# reporting the FIRST config's own error specifically, not a generic
# summary. Three distinct configs confirm the reported config really is
# the first one in search_space order, not an arbitrary or last one.
three_configs = [{"tag": "first"}, {"tag": "second"}, {"tag": "third"}]
try:
    result = tune.exhaustive_search(three_configs, None, lambda cfg: (1,), kernel_fn, lambda cfg: (), quiet=True)
    print(f"exhaustive_search(three configs, stream=None): {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search(three configs, stream=None): {type(e).__name__}: {e}")

# Reversing the same three configs' order confirms the reported failure
# really does track search_space order, rather than always naming
# "first" regardless of position.
reversed_configs = [{"tag": "third"}, {"tag": "second"}, {"tag": "first"}]
try:
    result = tune.exhaustive_search(reversed_configs, None, lambda cfg: (1,), kernel_fn, lambda cfg: (), quiet=True)
    print(f"exhaustive_search(reversed configs, stream=None): {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search(reversed configs, stream=None): {type(e).__name__}: {e}")
```

### File: `05_tune_args_fn_errors_caught_the_same_way.py`

```python
import cuda.tile as ct
import cuda.tile.tune as tune

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

# args_fn is only invoked lazily, once exhaustive_search actually tries
# to warm up a given config -- so an args_fn that raises for its own
# config is caught by the exact same per-config try/except as a stream
# error, and reported the same way: as that config's own failure, not
# a crash of the whole search.
def bad_args_fn(cfg):
    raise ValueError(f"cannot build args for config {cfg}")

try:
    result = tune.exhaustive_search([{"tag": "only"}], None, lambda cfg: (1,), kernel_fn, bad_args_fn, quiet=True)
    print(f"exhaustive_search with failing args_fn: {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search with failing args_fn: {type(e).__name__}: {e}")
```

### File: `06_capstone_host_function_shared_across_tune_and_kernel.py`

```python
import cuda.tile as ct
import cuda.tile.tune as tune
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

sig1 = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: a single @ct.function(host=True)-decorated helper, used
# BOTH by ordinary host-side driver code (here, a grid_fn callback of
# the kind ct.tune.exhaustive_search expects) AND from inside the
# kernel body it tunes -- one definition, shared across the host/device
# line, which is the entire point of host=True existing.
@ct.function(host=True)
def scale_factor(n):
    return n * 2

grid_fn = lambda cfg: (scale_factor(cfg["blocks"]),)
print(f"scale_factor used directly in host-side grid_fn: grid_fn({{'blocks': 3}}) = {grid_fn({'blocks': 3})}")

@ct.kernel
def kernel_scaled(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = x * scale_factor(2)
    ct.store(c, (0,), y)
print(f"the SAME scale_factor called from inside the kernel body: {compile_bytes(kernel_scaled, sig1)} cubin bytes")

# Wiring this kernel into an actual exhaustive_search call, to show the
# intended integration end to end -- honestly reaching the same wall
# this chapter has already documented: no real CUDA stream, so no
# config can ever converge, and the search reports that plainly rather
# than silently pretending to tune anything.
def args_fn(cfg):
    return ()

try:
    result = tune.exhaustive_search([{"blocks": 3}], None, grid_fn, kernel_scaled, args_fn, quiet=True)
    print(f"exhaustive_search wired to kernel_scaled: {result} (unexpected)")
except Exception as e:
    print(f"exhaustive_search wired to kernel_scaled: {type(e).__name__}: {e}")
```
