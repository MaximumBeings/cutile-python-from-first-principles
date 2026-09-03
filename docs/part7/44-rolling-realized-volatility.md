# Chapter 44: Rolling Realized Volatility

> "Volatility is not a fact about the world in the way a stock's closing price is a fact — it is an estimate, and every estimate depends on the window of history chosen to compute it." — a plain restatement of why "the" volatility of an asset is always shorthand for "volatility, measured over this particular stretch of time."

**What you will understand by the end of this chapter:**

- How a running total (a prefix sum) turns a sliding-window computation that would otherwise re-scan every element of the window at every step into a single subtraction per step, using `ct.cumsum` — present in `cuda.tile`'s public surface since this book's first API inspection, but unused by any chapter until now.
- How to compute a rolling sum, and then a rolling mean, variance, and standard deviation, from two prefix sums (of a series, and of its squares) addressed at two fixed offsets in global memory.
- How the resulting rolling standard deviation is annualized into the "realized volatility" figure real pricing systems compute from historical returns, using a compile-time constant precomputed outside the kernel because `math.sqrt` cannot be called on a runtime tile value from inside one.
- How this chapter's rolling volatility estimate closes the loop this book left open in Chapter 43: the `sigma` that chapter treated as a given input to Black-Scholes becomes, here, a number computed from a simulated return series and fed into that exact same, unchanged formula.

**What you need to know first:**

- The `@ct.kernel` / `ct.load` / `ct.store` pattern, and this book's `ct.load`-requires-a-power-of-two-shape constraint, which shapes several of this chapter's specific numeric choices.
- `ct.maximum`, `ct.sqrt`, and ordinary tile arithmetic (Parts 0-6).
- Chapter 43's Black-Scholes formula and its `normal_cdf` helper, reused unchanged in this chapter's capstone.
- The cross-check discipline this book has used since Part 3: an independent, structurally different computation — not a second copy of the same formula — is what a genuine correctness claim rests on.

Chapter 43 built Black-Scholes pricing from five given inputs, one of which — `sigma`, the underlying's volatility — a real trading system does not receive from anywhere; it has to compute it from historical prices. This chapter builds that computation: rolling realized volatility from a series of returns, using a genuinely different technique (prefix sums) from anything this book has built before, and closes with feeding its own output directly into Chapter 43's pricing formula.

## 44.1 Rolling Sums via Prefix Sums

### Intuition

A rolling, or sliding-window, statistic recomputes some summary (a sum, a mean, a standard deviation) over the most recent `W` elements of a series, once per step, as the window slides forward one element at a time. The direct way to compute this — summing all `W` elements inside the window, at every one of the series' positions — does `W` units of work per output, for a total cost that grows with both the series length and the window size. A prefix sum (`cumsum`) sidesteps this entirely for anything expressible as a running total: once `cumsum[i]` holds the sum of every element up through position `i`, the sum of any window of `W` consecutive elements ending at position `i` is just `cumsum[i] - cumsum[i-W]` — one subtraction, regardless of how large `W` is.

### Background

This chapter implements that idea as two kernels forming a small pipeline, addressing the same array in global memory from two different offsets rather than trying to "shift" a tile's contents within a single kernel body. Stage 1, `kernel_prefix_sum`, loads the whole series as one `(1, N)` tile and calls `ct.cumsum(xt, axis=1)` — this book's first use of that function, despite it having been visible in `cuda.tile`'s public surface since Chapter 39's original API survey — storing the running-total array back to global memory. Stage 2, `kernel_rolling_sum`, then loads that *same* stored array twice: once starting at offset `0`, and once starting at offset `W`. Both loads read from the identical global-memory array; what differs is only the starting offset passed to `ct.load`, the same kind of fixed, compile-time-constant addressing Chapter 42's Mixture-of-Experts kernel used to slice each expert's weights out of a stacked array by a constant leading index. Subtracting the two loaded tiles elementwise computes every window's sum in one operation, with no per-window loop at all.

This chapter's series length `N = 32` and window size `W = 16` are chosen so that the *output* array's length, `M = N - W = 16`, is itself a power of two — a direct consequence of this book's long-standing `ct.load`-requires-a-power-of-two-shape constraint, which applies to `kernel_rolling_sum`'s loads of the `M`-length slices just as much as to any other tile load in this book.

### Worked Example 44.1.1: rolling sums from a prefix sum, cross-checked against direct windowing

```python
import cuda.tile as ct
import io
import torch

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

N = 32
W = 16
M = N - W  # number of rolling windows this pipeline produces (16, a power of two)

# --- Stage 1: cumsum turns a running total into something each output
# position can read directly, rather than re-summing a window's worth
# of elements from scratch every time the window slides forward one
# step. cumsum[i] holds the sum of every element from the start of the
# series up through position i (an inclusive prefix sum), stored back
# to global memory so Stage 2 can address it at two different offsets. ---
@ct.kernel
def kernel_prefix_sum(x, cumsum_out):
    xt = ct.load(x, (0, 0), (1, N))
    ct.store(cumsum_out, (0, 0), ct.cumsum(xt, axis=1))

# --- Stage 2: the sum of any W consecutive elements ending at position
# i is just cumsum[i] - cumsum[i-W] -- one subtraction instead of
# summing W elements again. Loading the SAME cumsum array at two fixed
# offsets, W apart, computes every window's sum in this one
# elementwise subtraction, with no window-by-window looping at all. ---
@ct.kernel
def kernel_rolling_sum(cumsum, out):
    later = ct.load(cumsum, (0, W), (1, M))
    earlier = ct.load(cumsum, (0, 0), (1, M))
    ct.store(out, (0, 0), later - earlier)


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_prefix_sum:", compile_bytes(kernel_prefix_sum, sig), "cubin bytes")
print("kernel_rolling_sum:", compile_bytes(kernel_rolling_sum, sig), "cubin bytes")

torch.manual_seed(0)
x = torch.randn(1, N)

# host-side simulation, mirroring the two kernels' own formula exactly
cumsum = torch.cumsum(x, dim=1)
later = cumsum[:, W:]
earlier = cumsum[:, :N - W]
rolling_sum_prefix = later - earlier

# --- The independent check: a completely different algorithm, not a
# second copy of the same subtraction. torch's unfold carves the series
# into overlapping windows directly and sums each one from scratch,
# exactly the "recompute every window" approach the prefix-sum
# technique exists to avoid. If the two disagree, the prefix-sum
# indexing (which window lines up with which output position) is
# wrong, not just a numerical rounding difference. ---
windows = x.unfold(dimension=1, size=W, step=1)  # shape (1, N - W + 1, W)
windows = windows[:, 1:, :]  # drop the window starting at index 0 to match the prefix-sum indexing above
rolling_sum_direct = windows.sum(dim=2)

print("rolling sum shape:", rolling_sum_prefix.shape)
print("prefix-sum rolling sum matches direct-unfold rolling sum:",
      torch.allclose(rolling_sum_prefix, rolling_sum_direct, atol=1e-5))
```

### Genuinely run

```text
kernel_prefix_sum: 22832 cubin bytes
kernel_rolling_sum: 22704 cubin bytes
rolling sum shape: torch.Size([1, 16])
prefix-sum rolling sum matches direct-unfold rolling sum: True
```

### Discussion

`kernel_prefix_sum` and `kernel_rolling_sum` compile to nearly identical sizes (`22832` and `22704` bytes), which is unsurprising once their bodies are compared: one loads a `(1, 32)` tile and calls `cumsum`, the other loads two `(1, 16)` tiles and subtracts them — comparable amounts of work, just arranged differently.

The independent check here is worth being precise about, because it is easy to accidentally check a formula against a differently-shaped copy of itself rather than against genuinely different reasoning. `windows.sum(dim=2)` after `unfold` performs the "obvious," un-optimized computation this chapter's prefix-sum trick was built specifically to avoid — summing every element of every window directly, with no running total anywhere. That the two agree exactly (within floating-point tolerance) is evidence that the prefix-sum *indexing* is correct — that output position `j` really does correspond to the window `cumsum[W:N] - cumsum[0:N-W]` implies — and not merely that two similar-looking formulas produce similar-looking numbers. Note, too, what this section's indexing does *not* cover: a 32-element series with a 16-element window admits `N - W + 1 = 17` valid windows in total (starting at every index from `0` through `16`), but this pipeline produces only `M = 16` outputs. Section 44.1's Self-Check Questions return to exactly which window goes missing, and why.

## 44.2 Capstone: Annualized Realized Volatility, Feeding Chapter 43's Formula

### Intuition

Realized volatility, the number Chapter 43's `sigma` stood in for, is the annualized standard deviation of an asset's returns over some trailing window — a rolling standard deviation is exactly what Section 44.1's rolling-sum technique needs one small extension to produce. Variance can be written as `E[X^2] - E[X]^2` (the mean of the squares, minus the square of the mean), and both of those moments are themselves rolling sums — so the same offset-subtraction trick, applied twice (once to the returns, once to the returns squared), yields a rolling mean and a rolling variance together.

### Background

`kernel_prefix_sums` computes and stores two running totals from one loaded tile of returns: `ct.cumsum(r, axis=1)` and `ct.cumsum(r * r, axis=1)`. `kernel_rolling_volatility` then loads both prefix-sum arrays at the same two offsets Section 44.1 used, turning them into a rolling sum and a rolling sum-of-squares, dividing by `W` to get a rolling mean and (via `E[X^2] - E[X]^2`) a rolling variance. Two details need care here that Section 44.1's plain rolling sum did not. First, subtracting two nearly-equal floating-point quantities (`window_sumsq / W` and `mean * mean`) can, for a window with genuinely tiny variance, produce a small *negative* number purely from rounding, even though a true variance can never be negative — `ct.maximum(var, 0.0)` guards against this before the result reaches `ct.sqrt`, which would otherwise be asked to take the square root of a negative number. Second, annualizing the resulting daily standard deviation means multiplying by the square root of the number of trading days being annualized against (`TRADING_DAYS = 252`, the standard convention) — but `math.sqrt` cannot be called on a runtime tile value from inside a `@ct.kernel` body, so `ANNUALIZATION = math.sqrt(TRADING_DAYS)` is computed once in plain Python, outside the kernel, and the kernel body multiplies by that already-known constant, the same way every other compile-time constant in this book's kernels has been supplied.

The capstone closes by doing exactly what Chapter 43's own "Where We Go Next" promised: taking this section's own computed output — the most recent window's annualized realized volatility — and passing it as `sigma` into Chapter 43's Black-Scholes formula, reused here without a single change, to price a hypothetical six-month option.

### Worked Example 44.2.1: rolling volatility, cross-checked, then priced through Chapter 43's formula

```python
import cuda.tile as ct
import io
import math
import torch

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

N = 32
W = 16
M = N - W
TRADING_DAYS = 252.0
# --- Tile kernels can't call math.sqrt on a runtime value, but
# TRADING_DAYS never changes at trace time, so its square root is a
# plain Python float computed once, outside the kernel, the same way
# every other compile-time constant in this book's kernels has been. ---
ANNUALIZATION = math.sqrt(TRADING_DAYS)

# --- Stage 1: two prefix sums computed from the same loaded tile --
# the running sum of returns, and the running sum of squared returns.
# Both are needed because variance is E[X^2] - E[X]^2, and a rolling
# variance needs a rolling version of both moments. ---
@ct.kernel
def kernel_prefix_sums(returns, sum_out, sumsq_out):
    r = ct.load(returns, (0, 0), (1, N))
    ct.store(sum_out, (0, 0), ct.cumsum(r, axis=1))
    ct.store(sumsq_out, (0, 0), ct.cumsum(r * r, axis=1))

# --- Stage 2: Section 44.1's offset-subtraction trick, applied to both
# prefix sums at once, turns into a rolling mean and a rolling
# variance; ct.maximum guards against the rare case where floating-
# point rounding in the subtraction leaves a variance that should be
# exactly zero as a tiny negative number instead, which ct.sqrt cannot
# handle. The result is annualized the standard way realized
# volatility is reported: multiplied by the square root of the number
# of trading days being annualized against. ---
@ct.kernel
def kernel_rolling_volatility(cumsum, cumsum_sq, vol_out):
    sum_later = ct.load(cumsum, (0, W), (1, M))
    sum_earlier = ct.load(cumsum, (0, 0), (1, M))
    sq_later = ct.load(cumsum_sq, (0, W), (1, M))
    sq_earlier = ct.load(cumsum_sq, (0, 0), (1, M))

    window_sum = sum_later - sum_earlier
    window_sumsq = sq_later - sq_earlier
    mean = window_sum / W
    var = window_sumsq / W - mean * mean
    var = ct.maximum(var, 0.0)
    daily_vol = ct.sqrt(var)
    annualized_vol = daily_vol * ANNUALIZATION
    ct.store(vol_out, (0, 0), annualized_vol)


sig2 = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_prefix_sums:", compile_bytes(kernel_prefix_sums, sig2), "cubin bytes")
print("kernel_rolling_volatility:", compile_bytes(kernel_rolling_volatility, sig2), "cubin bytes")

torch.manual_seed(0)
# a simulated daily log-return series (roughly 1% typical daily move)
returns = torch.randn(1, N) * 0.01

# host-side simulation, mirroring the two kernels' own formula exactly
cumsum = torch.cumsum(returns, dim=1)
cumsum_sq = torch.cumsum(returns * returns, dim=1)
sum_later = cumsum[:, W:]
sum_earlier = cumsum[:, :N - W]
sq_later = cumsum_sq[:, W:]
sq_earlier = cumsum_sq[:, :N - W]
window_sum = sum_later - sum_earlier
window_sumsq = sq_later - sq_earlier
mean = window_sum / W
var = (window_sumsq / W - mean * mean).clamp(min=0.0)
daily_vol = var.sqrt()
annualized_vol_prefix = daily_vol * ANNUALIZATION

# --- The independent check: direct per-window standard deviation via
# unfold, using none of the prefix-sum machinery -- the same kind of
# structurally different comparison Section 44.1 used for a plain
# rolling sum, now applied to a rolling standard deviation. ---
windows = returns.unfold(dimension=1, size=W, step=1)  # (1, N - W + 1, W)
windows = windows[:, 1:, :]  # drop the window starting at index 0 to match the prefix-sum indexing above
direct_std = windows.std(dim=2, unbiased=False)
annualized_vol_direct = direct_std * ANNUALIZATION

max_err = (annualized_vol_prefix - annualized_vol_direct).abs().max().item()
print("annualized realized volatility shape:", annualized_vol_prefix.shape)
print("max abs error, prefix-sum rolling vol vs direct-unfold rolling vol:", max_err)

# --- Chapter 43's Black-Scholes formula, reused unchanged: the sigma
# that chapter treated as a given input is now this section's own
# computed output, closing the loop between "what volatility does a
# pricing formula need" and "where a real system would get it." ---
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
A1, A2, A3, A4, A5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
P = 0.2316419

def host_cdf_approx(x):
    ax = x.abs()
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return torch.where(x >= 0.0, upper, 1.0 - upper)

sigma_realized = annualized_vol_prefix[0, -1].item()
S = torch.tensor(100.0)
K = torch.tensor(105.0)
T = torch.tensor(0.5)
r = torch.tensor(0.03)
sigma = torch.tensor(sigma_realized)

sqrt_t = T.sqrt()
d1 = (torch.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
d2 = d1 - sigma * sqrt_t
discount = torch.exp(-r * T)
call = S * host_cdf_approx(d1) - K * discount * host_cdf_approx(d2)
put = K * discount * host_cdf_approx(-d2) - S * host_cdf_approx(-d1)

print()
print(f"pricing a 6-month option (S=100, K=105, r=3%) using this window's realized vol ({sigma_realized:.6f}) as sigma:")
print("call price:", call.item())
print("put price:", put.item())
```

### Genuinely run

```text
kernel_prefix_sums: 27608 cubin bytes
kernel_rolling_volatility: 31384 cubin bytes
annualized realized volatility shape: torch.Size([1, 16])
max abs error, prefix-sum rolling vol vs direct-unfold rolling vol: 1.4901161193847656e-08

pricing a 6-month option (S=100, K=105, r=3%) using this window's realized vol (0.171737) as sigma:
call price: 3.3956336975097656
put price: 6.832389831542969
```

### Discussion

The independent check's error, `1.49e-8`, is smaller even than Chapter 43's tightest measured error — expected, since this comparison involves only subtraction, division, and a single square root on each side, with no rational-polynomial CDF approximation anywhere in the computation. `kernel_rolling_volatility`'s `31384` bytes exceed `kernel_prefix_sums`'s `27608`, reflecting the extra arithmetic (four loads instead of two, a division, a squared mean, the variance clamp, and a square root) needed to turn two prefix sums into an annualized standard deviation.

The final numbers — a realized volatility of about `17.2%`, a $3.40 call, and a $6.83 put on a $100 underlying with a $105 strike — are not the point of this section so much as the fact that they exist at all: this chapter took a simulated series of daily returns, computed a genuine rolling statistic from it using a technique (prefix sums) this book had not used before, and fed that statistic directly into a formula from two chapters ago without changing a single line of that formula. Chapter 43 could not have priced this option on its own, because it never had a `sigma` to price it with; this chapter supplies one, honestly, from data.

## Chapter Summary

`cuda.tile` has `ct.cumsum`, and this chapter used it for the first time to turn a sliding-window computation into a running total: `cumsum[i] - cumsum[i-W]` gives the sum of any `W`-element window in one subtraction, implemented as two kernels that address the same global-memory array at two fixed offsets rather than shifting a tile's contents within one kernel body. Section 44.1 established this for a plain rolling sum, cross-checked against `torch.unfold`'s direct, un-optimized windowing — a structurally different computation, not a second copy of the same subtraction. Section 44.2 extended the same trick to two prefix sums at once (of returns, and of returns squared) to compute a rolling mean and variance, annualized into realized volatility using a compile-time constant (`math.sqrt` cannot run on a tile value inside a kernel) and guarded against small negative variances from floating-point rounding with `ct.maximum`. The capstone closed the loop Chapter 43 left open: its most recent rolling volatility estimate was fed, unchanged, into Chapter 43's own Black-Scholes formula to price a real option from simulated market data.

## Self-Check Questions

1. Why does `kernel_rolling_sum` load the *same* `cumsum` array twice, at two different fixed offsets, rather than computing a shifted copy of the array some other way — and where else in this book has addressing the same global-memory array at a different fixed offset appeared?
2. A 32-element series with a 16-element window admits `N - W + 1 = 17` valid windows in total (starting at every index from `0` through `16`), but Section 44.1's pipeline produces only `M = N - W = 16` outputs. Which one of the 17 valid windows is missing, and why does the `cumsum[i] - cumsum[i-W]` subtraction, as written, not produce it directly?
3. Why is `ANNUALIZATION` computed as `math.sqrt(TRADING_DAYS)` in plain Python outside `kernel_rolling_volatility`, rather than calling `math.sqrt` on a runtime tile value from inside the kernel body?
4. `window_sumsq / W - mean * mean` can produce a small negative number even though a true variance can never be negative. Why can this happen, and what specifically would go wrong if `ct.maximum(var, 0.0)` were removed?
5. The capstone feeds the *most recent* window's annualized realized volatility into Chapter 43's Black-Scholes formula as `sigma`, rather than, say, averaging all 16 windows' volatility estimates together. What assumption about volatility does choosing the most recent window make, and when might that assumption be a poor one?

## Where We Go Next

Chapter 45 closes Part 7, and this book's from-first-principles arc, with a Monte Carlo option pricer: simulating many random price paths forward under geometric Brownian motion — using this chapter's realized volatility as one of its own inputs — and pricing an option as the discounted average of those simulated paths' payoffs. That answer is then checked against Chapter 43's closed-form Black-Scholes price for the very same option, two pricing methods built independently across three chapters, converging on the same number by two entirely different routes.

## Worked Solutions

**1.** Loading the same array at two fixed offsets lets `ct.load` do the work of "shifting" for free, as a read from a different starting address, rather than requiring some in-kernel operation to rearrange a tile's already-loaded contents by an arbitrary amount. Chapter 42's Mixture-of-Experts kernel used the same idea: `w1[e]`, `b1[e]`, `w2[e]`, and `b2[e]` were all sliced out of arrays with a leading `NUM_EXPERTS` dimension using `ct.load`'s leading index `e`, addressing the same global-memory array at a different fixed offset for each expert in the unrolled loop, rather than loading the whole stacked array and slicing a tile afterward.

**2.** The missing window is the very first one: the window starting at index `0`, i.e., `x[0:16]`. Writing this window's sum using the prefix-sum identity would require `cumsum[15] - cumsum[-1]`, but `cumsum[-1]` — the sum of the empty prefix before the series starts — is never stored anywhere; `cumsum_out` only ever holds `cumsum[0]` through `cumsum[N-1]`. `kernel_rolling_sum`'s two loads, starting at offsets `0` and `W`, only ever produce differences of the form `cumsum[W+j] - cumsum[j]` for `j >= 0`, which by construction excludes the one window that would need a "prefix sum before the start of the array." Including it would require either storing an explicit leading zero (so `cumsum_out` has length `N+1` with `cumsum_out[0] = 0`), or special-casing the first window separately.

**3.** `math.sqrt` is an ordinary Python function operating on Python numbers; it has no defined behavior for a `cuda.tile` `Tile` object representing many values computed at kernel-launch time, and `TRADING_DAYS` is a fixed Python `float` known completely at the time the kernel is being defined and compiled, never something that varies per element or per launch. Since its square root can be computed once and reused as a plain constant, computing it in ordinary Python outside the kernel — the same way every other compile-time constant (`INV_SQRT_2PI`, the polynomial coefficients in Chapter 43, `SCALE` in Chapter 41's attention kernel) has been supplied throughout this book — avoids ever needing `math.sqrt` to operate on a runtime tile value in the first place.

**4.** `window_sumsq / W` and `mean * mean` are two independently-rounded floating-point quantities that are mathematically equal to each other exactly when the window's true variance is zero (every element in the window identical) — and floating-point subtraction of two nearly-equal numbers can produce a tiny result of either sign purely from each operand's own rounding error, even when the true mathematical difference is exactly zero or slightly positive. Without `ct.maximum(var, 0.0)`, a window that happens to trigger this case would pass a small negative number into `ct.sqrt`, which has no real-valued result for a negative input — producing `NaN` in that window's output rather than the correctly near-zero volatility the window actually has.

**5.** Feeding the most recent window's volatility into the pricing formula assumes that recent volatility is the best available estimate of *future* volatility over the option's life — that whatever regime produced returns over the last `W` days is a reasonable proxy for the regime the next `T` years of the option's life will see. This is a reasonable default when volatility is relatively stable, but it is a poor assumption immediately after a volatility regime change — for instance, right after an unusually calm or unusually turbulent stretch of `W` days that is not representative of what is likely to follow, where an average across many windows, or a model that explicitly accounts for volatility's tendency to revert toward a longer-run average, would be a more defensible input to a real pricing system.

## Complete Runnable Code

`01_rolling_sums_via_prefix_sums.py`:

```python
import cuda.tile as ct
import io
import torch

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

N = 32
W = 16
M = N - W  # number of rolling windows this pipeline produces (16, a power of two)

# --- Stage 1: cumsum turns a running total into something each output
# position can read directly, rather than re-summing a window's worth
# of elements from scratch every time the window slides forward one
# step. cumsum[i] holds the sum of every element from the start of the
# series up through position i (an inclusive prefix sum), stored back
# to global memory so Stage 2 can address it at two different offsets. ---
@ct.kernel
def kernel_prefix_sum(x, cumsum_out):
    xt = ct.load(x, (0, 0), (1, N))
    ct.store(cumsum_out, (0, 0), ct.cumsum(xt, axis=1))

# --- Stage 2: the sum of any W consecutive elements ending at position
# i is just cumsum[i] - cumsum[i-W] -- one subtraction instead of
# summing W elements again. Loading the SAME cumsum array at two fixed
# offsets, W apart, computes every window's sum in this one
# elementwise subtraction, with no window-by-window looping at all. ---
@ct.kernel
def kernel_rolling_sum(cumsum, out):
    later = ct.load(cumsum, (0, W), (1, M))
    earlier = ct.load(cumsum, (0, 0), (1, M))
    ct.store(out, (0, 0), later - earlier)


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_prefix_sum:", compile_bytes(kernel_prefix_sum, sig), "cubin bytes")
print("kernel_rolling_sum:", compile_bytes(kernel_rolling_sum, sig), "cubin bytes")

torch.manual_seed(0)
x = torch.randn(1, N)

# host-side simulation, mirroring the two kernels' own formula exactly
cumsum = torch.cumsum(x, dim=1)
later = cumsum[:, W:]
earlier = cumsum[:, :N - W]
rolling_sum_prefix = later - earlier

# --- The independent check: a completely different algorithm, not a
# second copy of the same subtraction. torch's unfold carves the series
# into overlapping windows directly and sums each one from scratch,
# exactly the "recompute every window" approach the prefix-sum
# technique exists to avoid. If the two disagree, the prefix-sum
# indexing (which window lines up with which output position) is
# wrong, not just a numerical rounding difference. ---
windows = x.unfold(dimension=1, size=W, step=1)  # shape (1, N - W + 1, W)
windows = windows[:, 1:, :]  # drop the window starting at index 0 to match the prefix-sum indexing above
rolling_sum_direct = windows.sum(dim=2)

print("rolling sum shape:", rolling_sum_prefix.shape)
print("prefix-sum rolling sum matches direct-unfold rolling sum:",
      torch.allclose(rolling_sum_prefix, rolling_sum_direct, atol=1e-5))
```

`02_capstone_annualized_realized_volatility.py`:

```python
import cuda.tile as ct
import io
import math
import torch

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

N = 32
W = 16
M = N - W
TRADING_DAYS = 252.0
# --- Tile kernels can't call math.sqrt on a runtime value, but
# TRADING_DAYS never changes at trace time, so its square root is a
# plain Python float computed once, outside the kernel, the same way
# every other compile-time constant in this book's kernels has been. ---
ANNUALIZATION = math.sqrt(TRADING_DAYS)

# --- Stage 1: two prefix sums computed from the same loaded tile --
# the running sum of returns, and the running sum of squared returns.
# Both are needed because variance is E[X^2] - E[X]^2, and a rolling
# variance needs a rolling version of both moments. ---
@ct.kernel
def kernel_prefix_sums(returns, sum_out, sumsq_out):
    r = ct.load(returns, (0, 0), (1, N))
    ct.store(sum_out, (0, 0), ct.cumsum(r, axis=1))
    ct.store(sumsq_out, (0, 0), ct.cumsum(r * r, axis=1))

# --- Stage 2: Section 44.1's offset-subtraction trick, applied to both
# prefix sums at once, turns into a rolling mean and a rolling
# variance; ct.maximum guards against the rare case where floating-
# point rounding in the subtraction leaves a variance that should be
# exactly zero as a tiny negative number instead, which ct.sqrt cannot
# handle. The result is annualized the standard way realized
# volatility is reported: multiplied by the square root of the number
# of trading days being annualized against. ---
@ct.kernel
def kernel_rolling_volatility(cumsum, cumsum_sq, vol_out):
    sum_later = ct.load(cumsum, (0, W), (1, M))
    sum_earlier = ct.load(cumsum, (0, 0), (1, M))
    sq_later = ct.load(cumsum_sq, (0, W), (1, M))
    sq_earlier = ct.load(cumsum_sq, (0, 0), (1, M))

    window_sum = sum_later - sum_earlier
    window_sumsq = sq_later - sq_earlier
    mean = window_sum / W
    var = window_sumsq / W - mean * mean
    var = ct.maximum(var, 0.0)
    daily_vol = ct.sqrt(var)
    annualized_vol = daily_vol * ANNUALIZATION
    ct.store(vol_out, (0, 0), annualized_vol)


sig2 = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_prefix_sums:", compile_bytes(kernel_prefix_sums, sig2), "cubin bytes")
print("kernel_rolling_volatility:", compile_bytes(kernel_rolling_volatility, sig2), "cubin bytes")

torch.manual_seed(0)
# a simulated daily log-return series (roughly 1% typical daily move)
returns = torch.randn(1, N) * 0.01

# host-side simulation, mirroring the two kernels' own formula exactly
cumsum = torch.cumsum(returns, dim=1)
cumsum_sq = torch.cumsum(returns * returns, dim=1)
sum_later = cumsum[:, W:]
sum_earlier = cumsum[:, :N - W]
sq_later = cumsum_sq[:, W:]
sq_earlier = cumsum_sq[:, :N - W]
window_sum = sum_later - sum_earlier
window_sumsq = sq_later - sq_earlier
mean = window_sum / W
var = (window_sumsq / W - mean * mean).clamp(min=0.0)
daily_vol = var.sqrt()
annualized_vol_prefix = daily_vol * ANNUALIZATION

# --- The independent check: direct per-window standard deviation via
# unfold, using none of the prefix-sum machinery -- the same kind of
# structurally different comparison Section 44.1 used for a plain
# rolling sum, now applied to a rolling standard deviation. ---
windows = returns.unfold(dimension=1, size=W, step=1)  # (1, N - W + 1, W)
windows = windows[:, 1:, :]  # drop the window starting at index 0 to match the prefix-sum indexing above
direct_std = windows.std(dim=2, unbiased=False)
annualized_vol_direct = direct_std * ANNUALIZATION

max_err = (annualized_vol_prefix - annualized_vol_direct).abs().max().item()
print("annualized realized volatility shape:", annualized_vol_prefix.shape)
print("max abs error, prefix-sum rolling vol vs direct-unfold rolling vol:", max_err)

# --- Chapter 43's Black-Scholes formula, reused unchanged: the sigma
# that chapter treated as a given input is now this section's own
# computed output, closing the loop between "what volatility does a
# pricing formula need" and "where a real system would get it." ---
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
A1, A2, A3, A4, A5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
P = 0.2316419

def host_cdf_approx(x):
    ax = x.abs()
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return torch.where(x >= 0.0, upper, 1.0 - upper)

sigma_realized = annualized_vol_prefix[0, -1].item()
S = torch.tensor(100.0)
K = torch.tensor(105.0)
T = torch.tensor(0.5)
r = torch.tensor(0.03)
sigma = torch.tensor(sigma_realized)

sqrt_t = T.sqrt()
d1 = (torch.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
d2 = d1 - sigma * sqrt_t
discount = torch.exp(-r * T)
call = S * host_cdf_approx(d1) - K * discount * host_cdf_approx(d2)
put = K * discount * host_cdf_approx(-d2) - S * host_cdf_approx(-d1)

print()
print(f"pricing a 6-month option (S=100, K=105, r=3%) using this window's realized vol ({sigma_realized:.6f}) as sigma:")
print("call price:", call.item())
print("put price:", put.item())
```
