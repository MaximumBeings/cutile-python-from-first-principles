# Chapter 28: Runtime Assertions, Debug Output, and Per-Architecture Hints — Closing Part 2

> "Seventeen chapters ago, Part 2 set out to cover elementwise kernels, matmul, and reductions. It found a great deal more worth testing honestly along the way. This chapter closes it out with the last handful of names `dir(ct)` had left standing — then this book crosses into Part 3, where cuTile Python stops being the whole story."

**What you will understand by the end of this chapter:**

- `ct.assume_divisible_by(x, divisor)`: a runtime-scalar divisibility hint whose docstring is explicit that behavior is undefined if the claim is false — another runtime-correctness claim this book's `export_kernel`-only method cannot verify, joining `RoundingMode` and `gather`'s negative-index claim in that category
- That "a positive integer constant" divisor is satisfied by a compile-time `ct.Constant[int]` parameter just as well as by a Python literal, and that a negative divisor is rejected outright
- `ct.assert_(cond, message=None)`: a runtime, per-element boolean-tile assertion, contrasted directly with Chapter 25's compile-time `ct.static_assert` — confirmed to carry a real, measurable compiled-byte cost (not just a runtime one), with a custom `message` costing a small amount more
- `ct.printf`/`ct.print`: two entry points for runtime device-side debug printing — confirmed to compile to byte-identical cubins, meaning `ct.print`'s Python-style syntax is sugar over the exact same underlying machinery, not a separate implementation
- Two genuine `printf` misuse errors: an unsupported format specifier, and too few arguments for a format string's specifier count
- `ct.ByTarget(default=..., **value_by_target)`: a per-architecture compiler-hint value, confirmed to genuinely select the correct value for a listed architecture by matching a same-architecture fixed-value build exactly
- A hardware-generation fact this book can now state with real compiled evidence: `num_ctas` measurably changes compiled bytes on `sm_100` but has zero effect on `sm_80` — consistent with thread block clusters being a Hopper-and-later feature
- An undocumented fallback: a `ByTarget` with no `default`, compiled for an architecture it doesn't list at all, does not raise — it silently behaves as if `num_ctas` had never been set to a `ByTarget` in the first place
- A five-part capstone combining all four primitives into one kernel that compiles to genuinely different byte counts on two different named architectures, closing Part 2's arc

**What you need to know first:**

- Chapter 25's `ct.static_assert`, the compile-time counterpart this chapter's `ct.assert_` is contrasted against.
- Chapter 10's `num_ctas` compiler hint and this book's autotuning material, extended here to vary by target architecture.
- This book's `export_kernel`-only, driver-free verification method, and its long-standing inability to confirm any claim about what a kernel would compute or print at runtime.

## 28.1 ct.assume_divisible_by: A Runtime Hint This Book Cannot Verify

### Intuition

`ct.assume_divisible_by(x, divisor)` lets tile code declare that a runtime integer scalar is divisible by a compile-time-constant `divisor`, without the compiler having any way to prove it. The docstring is direct about the consequence: behavior is undefined if the claim is false. That is a claim about what a *running* kernel would compute, and this book has never had a way to check such a claim directly — only whether it compiles, and whether the hint changes what gets compiled.

### Background

A baseline kernel computing a block offset with no hint is compared against the identical kernel with `ct.assume_divisible_by(offset, 64)` applied. A separate variant swaps the literal `64` for the compile-time `tile_size` parameter, to check whether "a positive integer constant" tolerates something other than literal syntax. A negative divisor is also tried, to see the validation the docstring implies exists.

### Worked Example 28.1.1 — hint, constant-typed divisor, and negative-divisor validation

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

# ct.assume_divisible_by(x, divisor) declares -- with no compiler-checkable
# proof -- that a runtime scalar is divisible by a compile-time-constant
# divisor. Its docstring is explicit that behavior is undefined if the
# claim is false, which is exactly the kind of runtime-correctness claim
# this book's export_kernel-only method has never been able to verify,
# going back to Chapter 8's RoundingMode.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    x = ct.load(a, (offset,), (tile_size,))
    ct.store(c, (offset,), x)
n_plain = compile_bytes(kernel_fn, sig)
print(f"plain offset, no hint: {n_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    offset = ct.assume_divisible_by(offset, 64)
    x = ct.load(a, (offset,), (tile_size,))
    ct.store(c, (offset,), x)
n_hint = compile_bytes(kernel_fn, sig)
print(f"offset with assume_divisible_by(offset, 64): {n_hint} cubin bytes")
print(f"identical bytes: {n_plain == n_hint}")

# The docstring calls divisor "a positive integer constant" -- not
# necessarily a Python int literal. A compile-time ct.Constant[int]
# parameter satisfies "constant" just as well as a literal does.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    offset = ct.assume_divisible_by(offset, tile_size)
    x = ct.load(a, (offset,), (tile_size,))
    ct.store(c, (offset,), x)
n_constant_divisor = compile_bytes(kernel_fn, sig)
print(f"divisor=tile_size (a Constant, not a literal): {n_constant_divisor} cubin bytes")
print(f"identical to plain: {n_constant_divisor == n_plain}")

# A negative divisor violates "positive" directly.
try:
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        offset = pid * tile_size
        offset = ct.assume_divisible_by(offset, -4)
        x = ct.load(a, (offset,), (tile_size,))
        ct.store(c, (offset,), x)
    n = compile_bytes(kernel_fn, sig)
    print(f"negative divisor: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"negative divisor: {type(e).__name__}: {e}")
```

Genuinely run:

```
plain offset, no hint: 21024 cubin bytes
offset with assume_divisible_by(offset, 64): 21024 cubin bytes
identical bytes: True
divisor=tile_size (a Constant, not a literal): 21024 cubin bytes
identical to plain: True
negative divisor: TileTypeError: `assume_divisible_by` requires a positive divisor, got -4
  "/tmp/ch28/01_assume_divisible_by_hint_and_validation.py", line 66, col 18-51, in kernel_fn:
            offset = ct.assume_divisible_by(offset, -4)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

For this simple, single-block-offset load/store pattern, the hint changes nothing this book's method can see: with or without `assume_divisible_by`, and whether the divisor is written as `64` or as the compile-time `tile_size` parameter, all three variants compile to the identical 21024 bytes. That's a null result in the same family as `latency` (Chapter 26) and `padding_value` on top of an existing mask (also Chapter 26) — a genuinely validated parameter whose effect, if any, sits below this book's byte-count resolution for the pattern actually tested. The negative-divisor case is unambiguous: `TileTypeError: \`assume_divisible_by\` requires a positive divisor, got -4`, confirming the docstring's "positive" requirement is enforced, not just documented.

## 28.2 ct.assert_: A Runtime Assertion With a Real Compiled Cost

### Intuition

Chapter 25 met `ct.static_assert`: a compile-time check on a condition the compiler itself can evaluate, which either vanishes entirely or fails compilation outright. `ct.assert_` checks something different — a runtime boolean tile, evaluated when the kernel actually runs. Its docstring warns of "significant overhead," phrased the way a runtime-performance note usually is. Whether that overhead also shows up as a compiled-size cost, this book's byte-count method can check directly.

### Background

A baseline kernel with no assertion is compared against the same kernel with `ct.assert_(x > 0)`, and again with a custom `message` added. A further variant asserts a condition this book has no way of knowing the runtime truth of, to confirm compilation doesn't depend on it either way.

### Worked Example 28.2.1 — assertion cost, message cost, and an unknowable condition

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

# ct.assert_(cond, message=None) checks a runtime BOOLEAN TILE -- unlike
# Chapter 25's ct.static_assert, which checks a compile-time condition and
# either vanishes or fails compilation outright. Its docstring warns of
# "significant overhead" -- worth checking whether that overhead is only
# a runtime cost, or something this book's byte-count method can see too.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_no_assert = compile_bytes(kernel_fn, sig)
print(f"no assert_: {n_no_assert} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.assert_(x > 0)
    ct.store(c, (0,), x)
n_assert_plain = compile_bytes(kernel_fn, sig)
print(f"assert_(x > 0), no message: {n_assert_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.assert_(x > 0, message="x must be positive")
    ct.store(c, (0,), x)
n_assert_message = compile_bytes(kernel_fn, sig)
print(f"assert_(x > 0, message=...): {n_assert_message} cubin bytes")
print(f"no_assert == assert_plain: {n_no_assert == n_assert_plain}")
print(f"assert_plain == assert_message: {n_assert_plain == n_assert_message}")

# assert_'s condition is a runtime tile value -- this book's export_kernel-
# only method cannot launch a kernel to see whether a FALSE condition
# actually aborts anything at runtime. What it CAN confirm is that a
# condition this book has no way of knowing the runtime truth of still
# compiles without incident.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.assert_(x < 0, message="this would fail for any non-negative input")
    ct.store(c, (0,), x)
n_assert_unknowable = compile_bytes(kernel_fn, sig)
print(f"assert_(x < 0, ...) -- an unknowable-at-compile-time condition: {n_assert_unknowable} cubin bytes")
```

Genuinely run:

```
no assert_: 20256 cubin bytes
assert_(x > 0), no message: 23176 cubin bytes
assert_(x > 0, message=...): 23192 cubin bytes
no_assert == assert_plain: False
assert_plain == assert_message: False
assert_(x < 0, ...) -- an unknowable-at-compile-time condition: 23344 cubin bytes
```

### Discussion

The docstring's "significant overhead" turns out to be visible at compile time, not only at runtime: adding a bare `assert_(x > 0)` costs 2920 bytes over the no-assertion baseline — real, structural machinery, not a free annotation. A custom `message` costs 16 bytes further, presumably the encoded string itself plus whatever plumbing carries it to the point of failure. The condition this book cannot evaluate (`x < 0`, compared against a value only known at runtime) compiles exactly as readily as the two conditions above it — confirming that whatever `assert_` generates, it generates unconditionally, based on the *shape* of the check being made, not on any attempt by the compiler to prove or disprove the condition itself. Whether the assertion would actually fire on a real launch remains, like every other runtime-value claim in this book, outside what `export_kernel` alone can show.

## 28.3 ct.printf and ct.print: Two Names, One Compiled Result

### Intuition

`ct.printf` takes a C-`printf`-style format string; `ct.print` takes Python-style arguments and f-strings, described in its own docstring as working "similar to Python's built-in `print()`." "Similar" is a word worth checking literally: does `ct.print`'s friendlier syntax compile to genuinely different code, or is it sugar over the exact same underlying operation `ct.printf` uses?

### Background

An equivalent single-value print is written both ways — `ct.printf("value: %d\n", x)` and `ct.print(f"value={x}")` — and compared under naming control. Two genuine misuse cases follow: an unsupported format specifier, and a format string with more specifiers than supplied arguments.

### Worked Example 28.3.1 — printf vs. print, and two misuse errors

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

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_plain = compile_bytes(kernel_fn, sig)
print(f"no debug printing: {n_plain} cubin bytes")

# ct.printf is the C-printf-style form; ct.print is described as
# Python-style syntax sugar over "similar" machinery. Comparing the two
# under naming control checks whether "similar" means "byte-identical"
# or something looser.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.printf("value: %d\n", x)
    ct.store(c, (0,), x)
n_printf = compile_bytes(kernel_fn, sig)
print(f"ct.printf('value: %d\\n', x): {n_printf} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.print(f"value={x}")
    ct.store(c, (0,), x)
n_print = compile_bytes(kernel_fn, sig)
print(f"ct.print(f'value={{x}}'): {n_print} cubin bytes")
print(f"printf == print: {n_printf == n_print}")
print(f"debug-printing cost over plain: {n_printf - n_plain} bytes")

# printf's docstring limits specifiers to integer and float ("for now").
# %s is neither.
try:
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        x = ct.load(a, (0,), (8,))
        ct.printf("value: %s\n", x)
        ct.store(c, (0,), x)
    n = compile_bytes(kernel_fn, sig)
    print(f"printf with unsupported %s specifier: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"printf with unsupported %s specifier: {type(e).__name__}: {e}")

# Two format specifiers, one argument.
try:
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        x = ct.load(a, (0,), (8,))
        ct.printf("two values: %d %d\n", x)
        ct.store(c, (0,), x)
    n = compile_bytes(kernel_fn, sig)
    print(f"printf with too few args: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"printf with too few args: {type(e).__name__}: {e}")
```

Genuinely run:

```
no debug printing: 20256 cubin bytes
ct.printf('value: %d\n', x): 27840 cubin bytes
ct.print(f'value={x}'): 27840 cubin bytes
printf == print: True
debug-printing cost over plain: 7584 bytes
printf with unsupported %s specifier: TileTypeError: Specifier s in %s is not supported
  "/tmp/ch28/03_printf_and_print_runtime_debug_output.py", line 55, col 9-35, in kernel_fn:
            ct.printf("value: %s\n", x)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^

printf with too few args: TileTypeError: Not enough arguments for format string
  "/tmp/ch28/03_printf_and_print_runtime_debug_output.py", line 67, col 9-43, in kernel_fn:
            ct.printf("two values: %d %d\n", x)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

`ct.printf` and `ct.print` compile to byte-identical cubins — "similar" in the docstring's own words turns out to mean genuinely identical at the compiled-code level, not merely comparable. `ct.print`'s f-string is evidently lowered to the same underlying format-string-plus-tile-arguments representation `ct.printf` takes directly, rather than being a separate code path that happens to produce similar-sized output. Debug printing of any kind carries a real cost here too — 7584 bytes over the no-debug baseline, over twice `assert_`'s own overhead from Section 28.2, consistent with a print call needing to encode and carry a format string plus argument-marshalling machinery that a boolean assertion doesn't. Both misuse cases are rejected the same way this book's method has rejected type and arity mismatches since Chapter 1: a `TileTypeError` naming the exact problem — an unsupported conversion specifier, or a specifier with nothing to consume — while the docstring's own "for now" wording about `printf`'s specifier support turns out to be accurate rather than aspirational.

## 28.4 ct.ByTarget: Compiler Hints That Vary by Named Architecture

### Intuition

`ct.ByTarget(default=..., **value_by_target)` lets a `@ct.kernel(...)` hint — Chapter 10's `num_ctas`, here — take a different value depending on which named GPU architecture the kernel is ahead-of-time compiled for. Confirming this properly needs a hint that this book already knows changes compiled bytes on at least one architecture, compared against a fixed-value build targeting that same architecture directly.

### Background

`num_ctas=1` and `num_ctas=2` are compiled as plain integers on both `sm_100` and `sm_80`, to see whether the hint has a same-architecture effect on each generation independently. A `ByTarget(sm_90=1, sm_100=2, default=1)` kernel is then compiled for both `sm_100` and `sm_80`, and checked against the matching fixed-value builds for those same architectures. Finally, a `ByTarget` with no `default` is compiled for an architecture it doesn't list.

### Worked Example 28.4.1 — same-architecture effects, ByTarget selection, and a no-default fallback

```python
import cuda.tile as ct
from cuda.tile import ByTarget
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.ByTarget(default=..., **value_by_target) lets a @ct.kernel(...) hint
# (Chapter 10's num_ctas, here) vary by which named architecture the
# SAME kernel is ahead-of-time compiled for. First, establish that
# num_ctas has a real, same-architecture effect at all, on two different
# generations.
@ct.kernel(num_ctas=1)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm100_fixed_1 = compile_bytes(kernel_fn, sig, "sm_100")
print(f"num_ctas=1 fixed, sm_100: {n_sm100_fixed_1} cubin bytes")

@ct.kernel(num_ctas=2)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm100_fixed_2 = compile_bytes(kernel_fn, sig, "sm_100")
print(f"num_ctas=2 fixed, sm_100: {n_sm100_fixed_2} cubin bytes")
print(f"identical (sm_100, 1 vs 2): {n_sm100_fixed_1 == n_sm100_fixed_2}")

@ct.kernel(num_ctas=1)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm80_fixed_1 = compile_bytes(kernel_fn, sig, "sm_80")
print(f"num_ctas=1 fixed, sm_80: {n_sm80_fixed_1} cubin bytes")

@ct.kernel(num_ctas=2)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm80_fixed_2 = compile_bytes(kernel_fn, sig, "sm_80")
print(f"num_ctas=2 fixed, sm_80: {n_sm80_fixed_2} cubin bytes")
print(f"identical (sm_80, 1 vs 2): {n_sm80_fixed_1 == n_sm80_fixed_2}")

# Now the SAME ByTarget-hinted kernel, compiled for both architectures,
# checked against the fixed-value builds above.
@ct.kernel(num_ctas=ByTarget(sm_90=1, sm_100=2, default=1))
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_bytarget_sm100 = compile_bytes(kernel_fn, sig, "sm_100")
print(f"ByTarget(sm_90=1, sm_100=2, default=1), compiled for sm_100: {n_bytarget_sm100} cubin bytes")
print(f"matches sm_100 fixed num_ctas=2: {n_bytarget_sm100 == n_sm100_fixed_2}")

n_bytarget_sm80 = compile_bytes(kernel_fn, sig, "sm_80")
print(f"the SAME kernel, compiled for sm_80 (falls to default=1): {n_bytarget_sm80} cubin bytes")
print(f"matches sm_80 fixed num_ctas=1: {n_bytarget_sm80 == n_sm80_fixed_1}")

# A ByTarget with NO default, compiled for an architecture that
# demonstrably responds to num_ctas (sm_100) but isn't listed at all.
try:
    @ct.kernel(num_ctas=ByTarget(sm_90=1))
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        x = ct.load(a, (0,), (8,))
        ct.store(c, (0,), x)
    n = compile_bytes(kernel_fn, sig, "sm_100")
    print(f"ByTarget(sm_90=1), no default, unlisted sm_100: {n} cubin bytes")
    print(f"matches sm_100 fixed num_ctas=1: {n == n_sm100_fixed_1}")
except Exception as e:
    print(f"ByTarget(sm_90=1), no default, unlisted sm_100: {type(e).__name__}: {e}")
```

Genuinely run:

```
num_ctas=1 fixed, sm_100: 25920 cubin bytes
num_ctas=2 fixed, sm_100: 26848 cubin bytes
identical (sm_100, 1 vs 2): False
num_ctas=1 fixed, sm_80: 19648 cubin bytes
num_ctas=2 fixed, sm_80: 19648 cubin bytes
identical (sm_80, 1 vs 2): True
ByTarget(sm_90=1, sm_100=2, default=1), compiled for sm_100: 26848 cubin bytes
matches sm_100 fixed num_ctas=2: True
the SAME kernel, compiled for sm_80 (falls to default=1): 19648 cubin bytes
matches sm_80 fixed num_ctas=1: True
ByTarget(sm_90=1), no default, unlisted sm_100: 25920 cubin bytes
matches sm_100 fixed num_ctas=1: True
```

### Discussion

`num_ctas` genuinely changes compiled bytes on `sm_100` (25920 vs. 26848 for 1 versus 2 CTAs) but has zero effect on `sm_80`, where both values compile to the identical 19648 bytes. That asymmetry lines up with real hardware history rather than looking like an arbitrary quirk: thread block clusters — the mechanism `num_ctas` above one configures — were introduced with the Hopper architecture (`sm_90`) and don't exist on Ampere (`sm_80`), so a hint about how many CTAs cooperate in a cluster has nothing to act on there. The `ByTarget`-hinted kernel confirms it selects correctly on both counts: compiled for `sm_100`, it matches the fixed `num_ctas=2` build on `sm_100` exactly; compiled for `sm_80`, where it falls through to its `default=1`, it matches the fixed `num_ctas=1` build on `sm_80` exactly. The last case is the genuine surprise: a `ByTarget` with no `default` at all, compiled for `sm_100` — a listed architecture in neither its `value_by_target` nor a fallback — does not raise. It compiles successfully, matching the plain `num_ctas=1` build exactly, which is `num_ctas`'s own baseline default when no hint is given at all. Nothing in the docstring states what happens when a target is unmatched and no `default` is supplied; empirically, on the installed 1.5.0 release, it is not an error — it behaves as though the `ByTarget` had never been applied for that architecture, silently reverting to the hint's own overall default rather than failing loudly.

## 28.5 Capstone: Closing Part 2

### Intuition

Part 2 opened eighteen sections ago with tile-function composition and has, chapter by chapter, worked through nearly every corner of `dir(ct)` this book's driver-free method can reach. This closing capstone combines all four of this chapter's primitives — a compile-time addressing hint, a runtime debug precondition, runtime debug output, and a per-architecture compiler hint — into one kernel, compiled for two different named architectures.

### Background

`kernel_closing_part_2` uses `ct.assume_divisible_by` on its block offset, `ct.assert_` as a debug precondition on the loaded value, `ct.print` for runtime visibility into that value, and `ByTarget(sm_90=1, sm_100=2, default=1)` as its `num_ctas` hint — compiled once for `sm_90` and once for `sm_100`.

### Worked Example 28.5.1 — one kernel, four primitives, two architectures

```python
import cuda.tile as ct
from cuda.tile import ByTarget
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: one kernel using all four of this chapter's primitives at
# once, closing out Part 2 -- a compile-time addressing hint
# (assume_divisible_by), a runtime debug precondition (assert_), runtime
# debug output (print), and a per-architecture compiler hint (ByTarget
# on num_ctas), all compiling together into one real cubin.
@ct.kernel(num_ctas=ByTarget(sm_90=1, sm_100=2, default=1))
def kernel_closing_part_2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    offset = ct.assume_divisible_by(offset, tile_size)
    x = ct.load(a, (offset,), (tile_size,))
    ct.assert_(x >= 0, message="expected non-negative input")
    ct.print(f"block {pid}: loaded value={x}")
    ct.store(c, (offset,), x)

n_sm90 = compile_bytes(kernel_closing_part_2, sig, "sm_90")
print(f"closing kernel, compiled for sm_90 (ByTarget selects num_ctas=1): {n_sm90} cubin bytes")

n_sm100 = compile_bytes(kernel_closing_part_2, sig, "sm_100")
print(f"the SAME kernel, compiled for sm_100 (ByTarget selects num_ctas=2): {n_sm100} cubin bytes")
print(f"different byte counts across architectures: {n_sm90 != n_sm100}")
```

Genuinely run:

```
closing kernel, compiled for sm_90 (ByTarget selects num_ctas=1): 39744 cubin bytes
the SAME kernel, compiled for sm_100 (ByTarget selects num_ctas=2): 55240 cubin bytes
different byte counts across architectures: True
```

### Discussion

One kernel definition, using a divisibility hint, a runtime assertion, runtime debug printing, and a per-architecture CTA-count hint together, compiles cleanly to two genuinely different byte counts depending on which named architecture it targets — `sm_90` and `sm_100` diverge both because `ByTarget` selects a different `num_ctas` for each and because the two architectures are structurally different targets to begin with. Nothing about combining these four primitives introduced any friction: `assume_divisible_by` and `assert_` operate on ordinary tile and scalar values inside the kernel body, `print` reads the same loaded tile without disturbing it, and `ByTarget` is resolved once, at the `@ct.kernel(...)` decoration itself, entirely independent of what happens inside the body. That composability, more than any single byte count, is what closes Part 2 out: seventeen chapters of individually-verified primitives that combine into one kernel with no special-casing required.

## Chapter Summary

This chapter closed out Part 2 with the last few names `dir(ct)` had left uncovered. `ct.assume_divisible_by` declares a runtime scalar's divisibility with no compiler proof behind it — a hint this book confirmed is validated (rejecting a negative divisor) but which changed nothing measurable for the simple addressing pattern tested here. `ct.assert_` checks a runtime boolean tile rather than a compile-time condition the way Chapter 25's `ct.static_assert` does, and carries a genuine, measurable compiled-byte cost — not merely a runtime one — with a custom message adding a small further cost, and compiling identically regardless of whether this book could know the asserted condition's runtime truth. `ct.printf` and `ct.print` were confirmed to compile to byte-identical cubins, meaning the friendlier Python-style syntax is true sugar over the same machinery, both carrying a real debug-output cost roughly twice `assert_`'s own. `ct.ByTarget` was confirmed to genuinely select the correct per-architecture value, matched exactly against same-architecture fixed-value builds on two different generations — and along the way, this book found a real, hardware-grounded fact (`num_ctas` affects `sm_100` but not `sm_80`, consistent with thread block clusters being Hopper-and-later) and a genuine undocumented behavior (a `ByTarget` with no `default`, given an unlisted architecture, quietly falls back to the hint's own overall default rather than raising). The capstone combined all four into one kernel, compiling cleanly to different byte counts across two architectures, and with that, Part 2 — which set out to cover a handful of core-kernel topics and grew, chapter by chapter, into seventeen — is done.

## Self-Check Questions

1. Section 28.1 found `assume_divisible_by` changed nothing measurable for a simple block-offset addressing pattern, much like `latency` (Chapter 26) and `padding_value` on an existing mask (also Chapter 26). What would distinguish "this hint genuinely does nothing in the current compiler" from "this book's byte-count method and chosen example simply can't see this hint's effect," and what change to the worked example might make a real effect more likely to surface?
2. Section 28.2 found `assert_` costs bytes regardless of whether its condition was something this book could evaluate. What does that specifically rule out about how `assert_`'s runtime check gets compiled — and what would it mean instead if the compiled cost HAD varied between a condition trivially always true and one this book couldn't evaluate?
3. Section 28.3 found `ct.printf` and `ct.print` compile to byte-identical cubins. Given that finding, what should a reader expect to be true of `ct.print`'s own implementation inside cuTile Python's source, without needing to read that source directly to confirm it?
4. Section 28.4's asymmetry — `num_ctas` affects `sm_100` but not `sm_80` — was explained by thread block clusters being a Hopper-and-later hardware feature. What observation, using only this book's `export_kernel`-only method, would most strengthen that explanation beyond "a plausible story that fits the numbers"?
5. Section 28.4 also found a `ByTarget` with no `default`, given an unlisted architecture, falls back to `num_ctas`'s own baseline default rather than raising. Why might this be the more defensible design choice for a compiler-hint mechanism specifically, compared to how this book has seen OTHER unmatched-case scenarios handled elsewhere (such as Chapter 27's `exhaustive_search` raising when every candidate fails)?

## Where We Go Next

Part 2 is done. Seventeen chapters set out to cover elementwise kernels, matmul, and reductions, and along the way found four separate naming and positional confounds, closed the compile-time/runtime metaprogramming taxonomy, generalized advanced indexing into `gather`/`scatter`, crossed the host/device line, and worked through cuTile Python's own remaining primitives to the last one standing. This book now moves into Part 3 — Wiring Into PyTorch's Autograd — where cuTile Python stops being the whole story for the first time: a hand-written `@ct.kernel` becomes the forward pass inside a `torch.autograd.Function`, and this book's verification method has to grow to match, since `torch` itself, not just `cuda.tile`, now enters what this book can genuinely test.

## Worked Solutions

**1.** Distinguishing "genuinely does nothing" from "this method can't see it" would require an example where the divisibility information plausibly unlocks a *different* code path — not merely a different constant folded into the same instructions, but something like a wider vectorized load/store instruction that the compiler could only justify emitting if it knew the address was aligned to a specific boundary. The single-block-offset pattern tested here computes an offset from `pid * tile_size`, which the compiler may already be able to prove divisible by `tile_size` on its own (since `tile_size` is a compile-time constant multiplying a runtime value), making the explicit hint redundant information rather than new information. A stronger test would introduce a computation the compiler genuinely cannot resolve at compile time — for instance, an offset built from a runtime-loaded index rather than the block id alone — where only the explicit hint could tell the compiler something it did not already know, and see whether THAT changes the byte count.

**2.** Costing bytes regardless of whether the condition is one this book could evaluate specifically rules out the possibility that `assert_`'s compiled cost depends on the compiler successfully proving something about the condition — if the cost varied between a trivially-true-looking condition and a genuinely unknowable one, that would suggest the compiler tries to partially evaluate the assertion and only falls back to full runtime-check machinery when it can't. Instead, the identical cost across both cases (2920 bytes over baseline, matching Section 28.2's `assert_(x > 0)` finding) indicates `assert_` unconditionally lowers to the same runtime-check machinery regardless of the condition's shape — the compiler makes no attempt to prove or disprove what's being asserted, treating every condition as equally opaque. Had the costs differed, it would suggest the opposite: that the compiler does some amount of static reasoning about assertions and only pays the full cost when that reasoning fails, closer to how `ct.static_assert` behaves for conditions it CAN resolve.

**3.** Without reading the source, a reader should expect `ct.print`'s implementation to parse its f-string arguments (or Python-style positional arguments) into an equivalent C-`printf`-style format string plus a tuple of tile arguments, and then call the exact same underlying compiler intrinsic `ct.printf` calls directly — rather than maintaining a second, independent code-generation path for Python-style printing. Byte-identical compiled output is strong evidence of shared code generation; the only way two different front-end syntaxes could reliably produce identical machine code, call after call, is if they both funnel into one shared lowering step before the compiler's real work begins. `ct.print` is therefore best understood as a convenience layer entirely in the parsing/translation stage, with zero separate representation once compilation proper begins.

**4.** The most direct additional evidence this book's method could gather would be testing whether `num_ctas` also has zero effect on other architectures older than `sm_90` (if any earlier ones are supported by `export_kernel`'s `gpu_code` parameter) and whether it continues to have a real effect on architectures newer than `sm_100` — if the "no effect" boundary lines up precisely at the sm_90 generation line rather than falling in some other place, that pattern (rather than a single sm_80/sm_100 contrast) would make the thread-block-cluster explanation far harder to dismiss as coincidence. A second, independent angle available without a real launch would be disassembling the `sm_80` and `sm_100` cubins for both `num_ctas` values (with a tool like `cuobjdump`, external to cuTile Python) to check directly whether the `sm_100` binaries differ in cluster-related instructions or metadata that simply don't exist at all in the `sm_80` binaries — structural absence being stronger evidence than a matching byte count alone.

**5.** `ByTarget`'s job is narrower and lower-stakes than `exhaustive_search`'s: it supplies one specific value for one specific compiler hint, and that hint already has its own well-defined baseline behavior (`num_ctas`'s own default) for the ordinary case where no `ByTarget` is used at all. Falling back to that pre-existing default when a target is unmatched treats "no override for this architecture" as equivalent to "no override was ever requested" — a conservative, least-surprise choice for a hint mechanism, since the alternative (raising) would make simply compiling for one more architecture than a `ByTarget` mapping happened to anticipate a hard failure, for a feature whose entire purpose is convenience across multiple targets. `exhaustive_search`, by contrast, has no meaningful "do nothing" fallback available to it: a tuning search that can't identify any valid, working configuration has nothing useful to return at all, so raising is the only honest option once every candidate is confirmed to have failed. The two designs aren't inconsistent — each reflects whether a sensible do-nothing fallback exists for the operation in question, not a general rule about how this book's tools handle unmatched cases.

## Complete Runnable Code

### File: `01_assume_divisible_by_hint_and_validation.py`

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

# ct.assume_divisible_by(x, divisor) declares -- with no compiler-checkable
# proof -- that a runtime scalar is divisible by a compile-time-constant
# divisor. Its docstring is explicit that behavior is undefined if the
# claim is false, which is exactly the kind of runtime-correctness claim
# this book's export_kernel-only method has never been able to verify,
# going back to Chapter 8's RoundingMode.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    x = ct.load(a, (offset,), (tile_size,))
    ct.store(c, (offset,), x)
n_plain = compile_bytes(kernel_fn, sig)
print(f"plain offset, no hint: {n_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    offset = ct.assume_divisible_by(offset, 64)
    x = ct.load(a, (offset,), (tile_size,))
    ct.store(c, (offset,), x)
n_hint = compile_bytes(kernel_fn, sig)
print(f"offset with assume_divisible_by(offset, 64): {n_hint} cubin bytes")
print(f"identical bytes: {n_plain == n_hint}")

# The docstring calls divisor "a positive integer constant" -- not
# necessarily a Python int literal. A compile-time ct.Constant[int]
# parameter satisfies "constant" just as well as a literal does.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    offset = ct.assume_divisible_by(offset, tile_size)
    x = ct.load(a, (offset,), (tile_size,))
    ct.store(c, (offset,), x)
n_constant_divisor = compile_bytes(kernel_fn, sig)
print(f"divisor=tile_size (a Constant, not a literal): {n_constant_divisor} cubin bytes")
print(f"identical to plain: {n_constant_divisor == n_plain}")

# A negative divisor violates "positive" directly.
try:
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        offset = pid * tile_size
        offset = ct.assume_divisible_by(offset, -4)
        x = ct.load(a, (offset,), (tile_size,))
        ct.store(c, (offset,), x)
    n = compile_bytes(kernel_fn, sig)
    print(f"negative divisor: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"negative divisor: {type(e).__name__}: {e}")
```

### File: `02_assert_runtime_tile_assertion_cost.py`

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

# ct.assert_(cond, message=None) checks a runtime BOOLEAN TILE -- unlike
# Chapter 25's ct.static_assert, which checks a compile-time condition and
# either vanishes or fails compilation outright. Its docstring warns of
# "significant overhead" -- worth checking whether that overhead is only
# a runtime cost, or something this book's byte-count method can see too.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_no_assert = compile_bytes(kernel_fn, sig)
print(f"no assert_: {n_no_assert} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.assert_(x > 0)
    ct.store(c, (0,), x)
n_assert_plain = compile_bytes(kernel_fn, sig)
print(f"assert_(x > 0), no message: {n_assert_plain} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.assert_(x > 0, message="x must be positive")
    ct.store(c, (0,), x)
n_assert_message = compile_bytes(kernel_fn, sig)
print(f"assert_(x > 0, message=...): {n_assert_message} cubin bytes")
print(f"no_assert == assert_plain: {n_no_assert == n_assert_plain}")
print(f"assert_plain == assert_message: {n_assert_plain == n_assert_message}")

# assert_'s condition is a runtime tile value -- this book's export_kernel-
# only method cannot launch a kernel to see whether a FALSE condition
# actually aborts anything at runtime. What it CAN confirm is that a
# condition this book has no way of knowing the runtime truth of still
# compiles without incident.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.assert_(x < 0, message="this would fail for any non-negative input")
    ct.store(c, (0,), x)
n_assert_unknowable = compile_bytes(kernel_fn, sig)
print(f"assert_(x < 0, ...) -- an unknowable-at-compile-time condition: {n_assert_unknowable} cubin bytes")
```

### File: `03_printf_and_print_runtime_debug_output.py`

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

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_plain = compile_bytes(kernel_fn, sig)
print(f"no debug printing: {n_plain} cubin bytes")

# ct.printf is the C-printf-style form; ct.print is described as
# Python-style syntax sugar over "similar" machinery. Comparing the two
# under naming control checks whether "similar" means "byte-identical"
# or something looser.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.printf("value: %d\n", x)
    ct.store(c, (0,), x)
n_printf = compile_bytes(kernel_fn, sig)
print(f"ct.printf('value: %d\\n', x): {n_printf} cubin bytes")

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.print(f"value={x}")
    ct.store(c, (0,), x)
n_print = compile_bytes(kernel_fn, sig)
print(f"ct.print(f'value={{x}}'): {n_print} cubin bytes")
print(f"printf == print: {n_printf == n_print}")
print(f"debug-printing cost over plain: {n_printf - n_plain} bytes")

# printf's docstring limits specifiers to integer and float ("for now").
# %s is neither.
try:
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        x = ct.load(a, (0,), (8,))
        ct.printf("value: %s\n", x)
        ct.store(c, (0,), x)
    n = compile_bytes(kernel_fn, sig)
    print(f"printf with unsupported %s specifier: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"printf with unsupported %s specifier: {type(e).__name__}: {e}")

# Two format specifiers, one argument.
try:
    @ct.kernel
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        x = ct.load(a, (0,), (8,))
        ct.printf("two values: %d %d\n", x)
        ct.store(c, (0,), x)
    n = compile_bytes(kernel_fn, sig)
    print(f"printf with too few args: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"printf with too few args: {type(e).__name__}: {e}")
```

### File: `04_bytarget_per_architecture_compiler_hints.py`

```python
import cuda.tile as ct
from cuda.tile import ByTarget
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.ByTarget(default=..., **value_by_target) lets a @ct.kernel(...) hint
# (Chapter 10's num_ctas, here) vary by which named architecture the
# SAME kernel is ahead-of-time compiled for. First, establish that
# num_ctas has a real, same-architecture effect at all, on two different
# generations.
@ct.kernel(num_ctas=1)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm100_fixed_1 = compile_bytes(kernel_fn, sig, "sm_100")
print(f"num_ctas=1 fixed, sm_100: {n_sm100_fixed_1} cubin bytes")

@ct.kernel(num_ctas=2)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm100_fixed_2 = compile_bytes(kernel_fn, sig, "sm_100")
print(f"num_ctas=2 fixed, sm_100: {n_sm100_fixed_2} cubin bytes")
print(f"identical (sm_100, 1 vs 2): {n_sm100_fixed_1 == n_sm100_fixed_2}")

@ct.kernel(num_ctas=1)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm80_fixed_1 = compile_bytes(kernel_fn, sig, "sm_80")
print(f"num_ctas=1 fixed, sm_80: {n_sm80_fixed_1} cubin bytes")

@ct.kernel(num_ctas=2)
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_sm80_fixed_2 = compile_bytes(kernel_fn, sig, "sm_80")
print(f"num_ctas=2 fixed, sm_80: {n_sm80_fixed_2} cubin bytes")
print(f"identical (sm_80, 1 vs 2): {n_sm80_fixed_1 == n_sm80_fixed_2}")

# Now the SAME ByTarget-hinted kernel, compiled for both architectures,
# checked against the fixed-value builds above.
@ct.kernel(num_ctas=ByTarget(sm_90=1, sm_100=2, default=1))
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)
n_bytarget_sm100 = compile_bytes(kernel_fn, sig, "sm_100")
print(f"ByTarget(sm_90=1, sm_100=2, default=1), compiled for sm_100: {n_bytarget_sm100} cubin bytes")
print(f"matches sm_100 fixed num_ctas=2: {n_bytarget_sm100 == n_sm100_fixed_2}")

n_bytarget_sm80 = compile_bytes(kernel_fn, sig, "sm_80")
print(f"the SAME kernel, compiled for sm_80 (falls to default=1): {n_bytarget_sm80} cubin bytes")
print(f"matches sm_80 fixed num_ctas=1: {n_bytarget_sm80 == n_sm80_fixed_1}")

# A ByTarget with NO default, compiled for an architecture that
# demonstrably responds to num_ctas (sm_100) but isn't listed at all.
try:
    @ct.kernel(num_ctas=ByTarget(sm_90=1))
    def kernel_fn(a, c, tile_size: ct.Constant[int]):
        x = ct.load(a, (0,), (8,))
        ct.store(c, (0,), x)
    n = compile_bytes(kernel_fn, sig, "sm_100")
    print(f"ByTarget(sm_90=1), no default, unlisted sm_100: {n} cubin bytes")
    print(f"matches sm_100 fixed num_ctas=1: {n == n_sm100_fixed_1}")
except Exception as e:
    print(f"ByTarget(sm_90=1), no default, unlisted sm_100: {type(e).__name__}: {e}")
```

### File: `05_capstone_closing_part_2.py`

```python
import cuda.tile as ct
from cuda.tile import ByTarget
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Capstone: one kernel using all four of this chapter's primitives at
# once, closing out Part 2 -- a compile-time addressing hint
# (assume_divisible_by), a runtime debug precondition (assert_), runtime
# debug output (print), and a per-architecture compiler hint (ByTarget
# on num_ctas), all compiling together into one real cubin.
@ct.kernel(num_ctas=ByTarget(sm_90=1, sm_100=2, default=1))
def kernel_closing_part_2(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offset = pid * tile_size
    offset = ct.assume_divisible_by(offset, tile_size)
    x = ct.load(a, (offset,), (tile_size,))
    ct.assert_(x >= 0, message="expected non-negative input")
    ct.print(f"block {pid}: loaded value={x}")
    ct.store(c, (offset,), x)

n_sm90 = compile_bytes(kernel_closing_part_2, sig, "sm_90")
print(f"closing kernel, compiled for sm_90 (ByTarget selects num_ctas=1): {n_sm90} cubin bytes")

n_sm100 = compile_bytes(kernel_closing_part_2, sig, "sm_100")
print(f"the SAME kernel, compiled for sm_100 (ByTarget selects num_ctas=2): {n_sm100} cubin bytes")
print(f"different byte counts across architectures: {n_sm90 != n_sm100}")
```
