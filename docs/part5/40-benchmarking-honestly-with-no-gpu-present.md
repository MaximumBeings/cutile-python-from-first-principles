# Chapter 40: Benchmarking Honestly With No GPU Present

> "The number you cannot honestly measure is not a gap in the report — leaving it out, clearly labeled, is the report telling the truth."

**What you will understand by the end of this chapter:**

- Exactly which numbers this book's driver-free sandbox can never honestly produce (kernel latency, throughput, occupancy, memory bandwidth — anything that requires `ct.launch` against a live CUDA stream) and exactly which numbers it can: the wall-clock cost of `export_kernel` itself.
- A genuine, measured cold-start effect: the first `export_kernel` call in a fresh process is reliably at least ten times slower than every call that follows it in the same process — and why that makes warm-up runs mandatory rather than optional for any compile-time measurement this book makes.
- That a compiled kernel's byte count and its compile time are not the same signal: across a fourfold growth in tile shape that grew cubin size by more than eleven times, compile time barely moved at all.
- Why this chapter's own "Genuinely run" numbers are reported differently from every prior chapter's: as illustrative wall-clock measurements alongside robust, threshold-based boolean claims, rather than exact figures a rerun is expected to reproduce byte-for-byte.

**What you need to know first:**

- Chapter 1's very first confirmation that `ct.launch` raises `RuntimeError: Found no NVIDIA driver on your system...` in this book's build environment, and every chapter since that has relied on `ct.compilation.export_kernel` instead.
- Chapter 39's `export_kernel` findings: `output_format="cubin"` versus `"tileir_bytecode"`, and how ahead-of-time re-specialization under `ct.Constant[int]` parameters compiles a distinct artifact per constant value.
- This book's standing verification discipline of never asserting a number it did not genuinely measure — extended in this chapter to also mean never asserting more *precision* about a number than a genuine, repeated measurement actually supports.

## 40.1 The Cold-Start Trap

### Intuition

This book has quoted hundreds of `export_kernel` byte counts since Part 0, and every one of them was exactly reproducible: compile the same kernel and signature twice, get the same bytes twice. Wall-clock time is a different kind of number. It cannot be exactly reproducible even in principle — the same call, run twice on the same machine, can take a different number of milliseconds depending on what else the machine is doing. That alone does not make timing dishonest to report; it means the reporting has to change shape, from "this exact number" to "this measured relationship, confirmed to hold up over repeated measurement." Before trusting any compile-time number this chapter produces, the first thing worth checking is whether every call to `export_kernel` behaves the same way, or whether the very first one in a fresh process is doing something different from the rest.

### Background

A compiler is itself a program, and a Python compiler in particular typically has its own one-time costs on first use — importing heavy submodules, initializing internal caches, doing setup work that later calls can skip. If `cuda-tile`'s compiler has any such cost, it should show up as a first call that is measurably slower than every call after it, in a way that repeats across independent runs. That is a testable claim about this book's own installed package, not an assumption to take on faith from how compilers in general tend to behave.

### Worked Example 40.1.1: timing eight consecutive calls in a fresh process

```python
import cuda.tile as ct
import io
import statistics
import time

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_once():
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return buf.getvalue()

# --- No live GPU means ct.launch's wall-clock time can never be this
# book's benchmark. But export_kernel itself has a wall-clock cost, and
# nothing in this book's driver-free method forbids measuring that
# honestly. The first thing worth checking, before trusting any of
# those measurements: whether the FIRST call in a fresh process behaves
# like every call after it. ---
NUM_CALLS = 8
call_times = []
for i in range(NUM_CALLS):
    t0 = time.perf_counter()
    compile_once()
    call_times.append(time.perf_counter() - t0)

first_call = call_times[0]
warm_calls = call_times[1:]
warm_median = statistics.median(warm_calls)

print(f"call 0 (cold): {first_call * 1000:.2f} ms")
for i, t in enumerate(warm_calls, start=1):
    print(f"call {i} (warm): {t * 1000:.2f} ms")
print()
print(f"warm-call median (calls 1-{NUM_CALLS - 1}): {warm_median * 1000:.3f} ms")
print(f"cold call is at least 10x the warm median: {first_call >= 10 * warm_median}")
```

### Genuinely run

```text
call 0 (cold): 128.99 ms
call 1 (warm): 6.98 ms
call 2 (warm): 5.61 ms
call 3 (warm): 6.74 ms
call 4 (warm): 6.05 ms
call 5 (warm): 7.34 ms
call 6 (warm): 7.38 ms
call 7 (warm): 7.37 ms

warm-call median (calls 1-7): 6.978 ms
cold call is at least 10x the warm median: True
```

### Discussion

This run's cold call took about 129 milliseconds against a warm median under 7 — roughly an eighteenfold gap. Re-running this exact file independently, several times, produced cold-call times ranging from just over 200 milliseconds up to more than ten full seconds on one occasion, while the warm-call median stayed consistently in the single-digit milliseconds every time. The raw cold-start number is not a fact worth memorizing — it swings by two orders of magnitude between runs, most likely reflecting how much of the compiler's own machinery (and its underlying dependencies) had already been paged in and initialized before this process started. What is stable, checked directly against every one of those repeated runs, is the *relationship*: the first call is dramatically slower than the calls that follow it, by at least a factor of ten every single time, and usually by much more.

That is the reason this chapter reports a boolean threshold — `first_call >= 10 * warm_median` — rather than a bare millisecond figure as its actual claim. A number that varies by 50x between otherwise-identical runs is not a number this book can respectably ask a reader to reproduce; a relationship that holds every time it was checked is. Practically, it also settles a methodology question the rest of this chapter depends on: any compile-time measurement in this book from here on discards its first call as a warm-up and measures only what comes after, exactly as Worked Example 40.2.1 does next.

## 40.2 Capstone: Compile Time Does Not Track Output Size

### Intuition

Chapter 39 showed that a kernel compiled at a larger `ct.Constant` shape produces a bigger cubin. It would be easy to assume that a bigger cubin also takes longer to produce — more machine code, more compiler work. That assumption is worth testing rather than accepting, using the same warmed-up, repeated-trial, median-based methodology Section 40.1 just established, on a kernel whose output size is already known to grow substantially with shape: Chapter 35's shape-generic `kernel_matmul`.

### Background

`kernel_matmul` takes `m`, `k`, and `n` as `ct.Constant[int]` parameters, so each distinct `(m, k, n)` triple ahead-of-time compiles to its own artifact — this book confirmed as much back in Chapter 35, where `(8, 16, 8)` and `(8, 8, 8)` produced different cubin byte counts from identical source. Sweeping a range of square shapes from `(8, 8, 8)` up to `(64, 64, 64)` should make the byte-count growth substantial and easy to see; whether the *compile time* grows anywhere near proportionally is the open question, and it also bears directly on a real engineering decision this book has not yet had occasion to raise: a production deployment ahead-of-time compiling several specializations of the same kernel needs to know whether that cost scales with the size of what comes out, or with something closer to a fixed per-call overhead.

### Worked Example 40.2.1: compile time and cubin size across a shape sweep

```python
import cuda.tile as ct
import io
import statistics
import time

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def kernel_matmul(a, b, c, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    x = ct.load(a, (0, 0), (m, k))
    y = ct.load(b, (0, 0), (k, n))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

def sig_for(m, k, n):
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2), m, k, n],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

def compile_once(m, k, n):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_matmul, [sig_for(m, k, n)], buf, gpu_code="sm_90", output_format="cubin")
    return buf.getvalue()

# --- Pay the one-time cold-start cost Section 40.1 measured, once,
# before timing anything -- exactly the discipline that finding
# demands. Every measurement below is a warmed-up call. ---
compile_once(8, 8, 8)

TRIALS = 10
shapes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]
medians = {}
byte_counts = {}
for shape in shapes:
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        out = compile_once(*shape)
        times.append(time.perf_counter() - t0)
    medians[shape] = statistics.median(times)
    byte_counts[shape] = len(out)
    print(f"shape={shape}: median compile time {medians[shape] * 1000:.3f} ms, cubin size {byte_counts[shape]} bytes")

print()
time_ratio = medians[(64, 64, 64)] / medians[(8, 8, 8)]
byte_ratio = byte_counts[(64, 64, 64)] / byte_counts[(8, 8, 8)]
print(f"(64,64,64) vs (8,8,8): compile-time ratio {time_ratio:.2f}x, cubin-size ratio {byte_ratio:.2f}x")
print(f"output size grew far more than compile time did: {byte_ratio >= 5 * time_ratio}")

# --- The total wall-clock cost of ahead-of-time compiling all four
# specializations, back to back -- the real number a build pipeline
# choosing how many shapes to pre-compile would actually pay. ---
t0 = time.perf_counter()
for shape in shapes:
    compile_once(*shape)
total_all_four = time.perf_counter() - t0
sum_of_medians = sum(medians.values())
print()
print(f"total time to export all {len(shapes)} specializations once each: {total_all_four * 1000:.2f} ms")
print(f"sum of per-shape median times: {sum_of_medians * 1000:.2f} ms")
print(f"total cost is in the same ballpark as the sum of medians (within 2x): "
      f"{0.5 * sum_of_medians <= total_all_four <= 2 * sum_of_medians}")
```

### Genuinely run

```text
shape=(8, 8, 8): median compile time 6.731 ms, cubin size 30888 bytes
shape=(16, 16, 16): median compile time 5.819 ms, cubin size 35752 bytes
shape=(32, 32, 32): median compile time 5.938 ms, cubin size 73000 bytes
shape=(64, 64, 64): median compile time 6.293 ms, cubin size 360360 bytes

(64,64,64) vs (8,8,8): compile-time ratio 0.93x, cubin-size ratio 11.67x
output size grew far more than compile time did: True

total time to export all 4 specializations once each: 26.04 ms
sum of per-shape median times: 24.78 ms
total cost is in the same ballpark as the sum of medians (within 2x): True
```

### Discussion

The cubin grew from 30,888 bytes at `(8, 8, 8)` to 360,360 bytes at `(64, 64, 64)` — nearly twelvefold. Compile time did not follow it: the largest shape's median compile time was, in this run, if anything slightly *faster* than the smallest shape's, and every shape's median stayed within about a millisecond of every other's. Re-running this comparison independently produced the same qualitative picture each time — the compile-time ratio between the largest and smallest shape hovered close to 1.0, while the byte-size ratio stayed pinned at 11.67x every time, since that ratio depends only on the deterministic cubin sizes Chapter 35's own methodology already established. Sizeable, real code-size growth produced no correspondingly sizeable compile-time growth.

The likely explanation is that `export_kernel`'s wall-clock cost, at these shapes, is dominated by a largely fixed per-call overhead — parsing, the compiler's own internal setup, serialization — rather than by work that scales with how much machine code ultimately comes out. That has a genuinely practical consequence for exactly the question Section 40.1's Discussion raised: a build pipeline ahead-of-time compiling several specializations of one kernel should expect a cost close to (number of specializations) times (one call's fixed overhead), not a cost that grows with the total size of the resulting binaries. This run's own numbers bear that out directly — the total time to export all four specializations back to back landed within a couple of milliseconds of the sum of their individually-measured medians, exactly what additive, roughly-fixed-per-call overhead predicts. None of these absolute millisecond figures are this chapter's real claim, for the same reason Section 40.1 gave: they will differ, sometimes substantially, on a different machine or a different run. The relationships — size decoupled from time, total cost tracking the sum of its parts — are what a reader can expect to reproduce.

## Chapter Summary

This chapter drew an explicit line this book has walked up to but never stated outright: `ct.launch`'s wall-clock cost — latency, throughput, anything about how a kernel actually runs — is permanently off-limits to this book's driver-free sandbox, confirmed unreachable since Chapter 1. `export_kernel`'s own wall-clock cost is not off-limits, but it demanded a different honesty discipline than this book's byte-count claims ever have: timing is not exactly reproducible, so this chapter reported robust, threshold-based relationships checked across repeated runs rather than bare figures. Two such relationships came out of that discipline as genuine findings. First, a fresh process's first `export_kernel` call is reliably at least ten times slower than the calls that follow it — a real cold-start effect, of wildly varying absolute size from run to run, that makes discarding a first call as warm-up mandatory rather than a nicety. Second, across a shape sweep that grew a kernel's compiled cubin by nearly twelvefold, its compile time barely moved — output size and compile time are not the same signal, and a build pipeline's ahead-of-time compilation cost tracks the number of specializations compiled far more closely than it tracks how large those specializations turn out to be.

## Self-Check Questions

1. Every prior chapter's "Genuinely run" block was expected to match a fresh rerun exactly. This chapter's blocks instead report medians and boolean thresholds. Explain why insisting on exact reproducibility here, rather than adapting the standard, would have forced a worse choice.
2. Section 40.1 discarded the first `export_kernel` call as a warm-up before measuring anything in Section 40.2. What would have gone wrong with Section 40.2's shape-sweep comparison if that discipline had been skipped and the very first call of the whole chapter had landed inside the `(8, 8, 8)` trial group?
3. Section 40.2 found compile time roughly flat across a nearly twelvefold growth in cubin size. Propose one concrete follow-up test that would help distinguish "compile time is dominated by fixed per-call overhead" from "compile time happens to be flat over this particular shape range but would grow at much larger shapes."
4. If this book had access to a real GPU and driver, which specific numbers could this chapter have reported that it could not report here, and which of this chapter's actual findings — the cold-start effect, the size/time decoupling — would you expect to still hold true on real hardware, and which would you expect a driver to change?
5. This chapter measured `export_kernel`'s wall-clock cost using Python's `time.perf_counter()` around each call, in the same process as the code being measured. Name one source of measurement noise this method cannot rule out, and describe how you would rule it out if this book's build environment supported it.

## Where We Go Next

Part 5 closes here having drawn the sharpest line this book has drawn yet between what a genuinely run, driver-free environment can and cannot honestly claim — and having found real, repeatable substance on the side of that line it can reach. Part 6 turns to fused kernels for real neural-network layers and interoperability with SIMT code and JAX's foreign-function interface, building on every backward-kernel pattern Part 4 established and every compilation fact Part 5 confirmed. From there, this book's closing arc in Part 7 puts all of it to work on the problems it was always heading toward: Black-Scholes, rolling volatility, and Monte Carlo option pricing, each cross-checked by a second independent method exactly as rigorously as every chapter before it.

## Worked Solutions

**1.** Insisting on exact reproducibility for wall-clock numbers would have forced one of two bad choices: either silently picking numbers that happened to reproduce on one particular run and presenting them as if they always would (a form of the fabrication this book has refused since its first chapter), or refusing to report any timing information at all, which would have meant declaring benchmarking entirely off-limits rather than honestly separating the part of it this environment can support (repeatable relationships) from the part it cannot (exact figures). Adapting what "genuinely run" means for this specific kind of measurement — while still requiring every reported relationship to be checked against real, repeated runs — was the only option that stayed honest about both what was measured and how much confidence that measurement actually supports.

**2.** If the very first `export_kernel` call anywhere in the chapter had landed inside the `(8, 8, 8)` trial group instead of being paid off separately in Section 40.1, that group's ten trials would have included one dramatically slower cold-start call mixed in with nine ordinary warm ones. Using the median would have mostly protected the result from that single outlier, but using a mean instead would not have, and either way the `(8, 8, 8)` shape would have carried a cold-start artifact the other three shapes never paid — making it look, falsely, as though the smallest shape was the expensive one, for a reason that had nothing to do with shape at all.

**3.** A concrete follow-up would extend the same shape sweep methodology to substantially larger square shapes — for instance continuing past `(64, 64, 64)` to `(128, 128, 128)` and `(256, 256, 256)` — using the identical warmed-up, median-of-many-trials measurement, and checking whether the compile-time ratio between the largest and smallest shape stays close to 1.0 or begins climbing. A flat ratio across a much wider range would strengthen the fixed-overhead explanation considerably; a ratio that stays flat at first and then climbs at some larger shape would suggest the fixed overhead genuinely dominates only up to a point, after which the compiler's own per-instruction work starts to show through.

**4.** With a real GPU and driver, this chapter could have reported `ct.launch`'s own wall-clock latency, achieved throughput against a known problem size, and how both scale with tile shape and block count — none of which this sandboxed environment can produce at all. The cold-start effect measured here is specific to `export_kernel`'s own compiler machinery and has no obvious reason to disappear on real hardware — a compiler's first-use initialization cost is a property of the compiler, not of whether a GPU is attached, so it would likely still show up. The size/time decoupling finding is murkier to extrapolate: it describes `export_kernel`'s ahead-of-time compilation cost, not kernel execution cost, so it says nothing at all about whether a bigger compiled kernel runs proportionally slower on real hardware — that is a completely different question this chapter never claimed to answer.

**5.** Measuring wall-clock time with `time.perf_counter()` in the same Python process being measured cannot rule out interference from that process's own garbage collector running an unpredictable collection cycle in the middle of a timed call, adding noise unrelated to the compiler's actual work. If this book's build environment supported it, the way to rule that out would be to run each timed call in a freshly spawned subprocess (or with the garbage collector explicitly disabled for the duration of the measurement, via Python's `gc.disable()`), removing that specific source of interference so any remaining noise could be attributed with more confidence to the compiler itself or the underlying system.

## Complete Runnable Code

`01_the_cold_start_trap.py`:

```python
import cuda.tile as ct
import io
import statistics
import time

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def compile_once():
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_add, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return buf.getvalue()

# --- No live GPU means ct.launch's wall-clock time can never be this
# book's benchmark. But export_kernel itself has a wall-clock cost, and
# nothing in this book's driver-free method forbids measuring that
# honestly. The first thing worth checking, before trusting any of
# those measurements: whether the FIRST call in a fresh process behaves
# like every call after it. ---
NUM_CALLS = 8
call_times = []
for i in range(NUM_CALLS):
    t0 = time.perf_counter()
    compile_once()
    call_times.append(time.perf_counter() - t0)

first_call = call_times[0]
warm_calls = call_times[1:]
warm_median = statistics.median(warm_calls)

print(f"call 0 (cold): {first_call * 1000:.2f} ms")
for i, t in enumerate(warm_calls, start=1):
    print(f"call {i} (warm): {t * 1000:.2f} ms")
print()
print(f"warm-call median (calls 1-{NUM_CALLS - 1}): {warm_median * 1000:.3f} ms")
print(f"cold call is at least 10x the warm median: {first_call >= 10 * warm_median}")
```

`02_capstone_compile_time_does_not_track_output_size.py`:

```python
import cuda.tile as ct
import io
import statistics
import time

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def kernel_matmul(a, b, c, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    x = ct.load(a, (0, 0), (m, k))
    y = ct.load(b, (0, 0), (k, n))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

def sig_for(m, k, n):
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2), m, k, n],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

def compile_once(m, k, n):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_matmul, [sig_for(m, k, n)], buf, gpu_code="sm_90", output_format="cubin")
    return buf.getvalue()

# --- Pay the one-time cold-start cost Section 40.1 measured, once,
# before timing anything -- exactly the discipline that finding
# demands. Every measurement below is a warmed-up call. ---
compile_once(8, 8, 8)

TRIALS = 10
shapes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]
medians = {}
byte_counts = {}
for shape in shapes:
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        out = compile_once(*shape)
        times.append(time.perf_counter() - t0)
    medians[shape] = statistics.median(times)
    byte_counts[shape] = len(out)
    print(f"shape={shape}: median compile time {medians[shape] * 1000:.3f} ms, cubin size {byte_counts[shape]} bytes")

print()
time_ratio = medians[(64, 64, 64)] / medians[(8, 8, 8)]
byte_ratio = byte_counts[(64, 64, 64)] / byte_counts[(8, 8, 8)]
print(f"(64,64,64) vs (8,8,8): compile-time ratio {time_ratio:.2f}x, cubin-size ratio {byte_ratio:.2f}x")
print(f"output size grew far more than compile time did: {byte_ratio >= 5 * time_ratio}")

# --- The total wall-clock cost of ahead-of-time compiling all four
# specializations, back to back -- the real number a build pipeline
# choosing how many shapes to pre-compile would actually pay. ---
t0 = time.perf_counter()
for shape in shapes:
    compile_once(*shape)
total_all_four = time.perf_counter() - t0
sum_of_medians = sum(medians.values())
print()
print(f"total time to export all {len(shapes)} specializations once each: {total_all_four * 1000:.2f} ms")
print(f"sum of per-shape median times: {sum_of_medians * 1000:.2f} ms")
print(f"total cost is in the same ballpark as the sum of medians (within 2x): "
      f"{0.5 * sum_of_medians <= total_all_four <= 2 * sum_of_medians}")
```
