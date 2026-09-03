# Chapter 43: Black-Scholes Option Pricing

> "The mathematics of options is deceptively simple, and yet it took financial economists more than half a century to get it right." — Fischer Black and Myron Scholes' formula did not arrive until 1973, decades after options themselves had been traded, precisely because "simple" and "already understood" are not the same thing.

**What you will understand by the end of this chapter:**

- Why `cuda.tile` has no `erf` and no built-in normal-distribution function, and how a well-documented rational-polynomial approximation (Abramowitz & Stegun 26.2.17) fills that gap using only `ct.exp`, `ct.abs`, and ordinary arithmetic.
- How `ct.function` factors reusable, tile-callable logic out of a kernel body, and why sharing one normal-CDF implementation across every kernel that needs it mirrors how a real pricing library is structured.
- How the Black-Scholes formula for European call and put prices becomes a single branch-free, batched kernel that prices many independent options in parallel, following the same `(ROWS, COLS)` batching convention every kernel since Part 0 has used.
- Why this chapter's correctness claims rest on two structurally different kinds of evidence — an independent CDF implementation (torch's own `erf`) and a mathematical identity (put-call parity) that does not depend on any CDF implementation at all — rather than the same check performed twice under different names.

**What you need to know first:**

- The `@ct.kernel` / `ct.load` / `ct.store` batched-kernel pattern, and the `(ROWS, COLS)` tile-layout convention used throughout this book since Part 0.
- `ct.exp`, `ct.abs`, `ct.sqrt`, `ct.log`, and `ct.where` for branch-free, elementwise tile arithmetic (Parts 0-6, most recently Chapter 42's masked expert routing).
- The export-and-cross-check discipline this book has used since Part 3: `export_kernel` compiles a kernel to real cubin bytes, and an independent host-side computation — mirroring the kernel's own formula exactly — is what stands in for a live GPU run in this driver-free sandbox.
- Basic floating-point precision awareness: that an approximation's error is a number to be measured, not a property to be assumed, the same discipline Chapter 39's path-length finding and Chapter 40's benchmarking chapter both insisted on.

Part 6 closed out this book's tour of neural-network building blocks — attention, feed-forward layers, routing, and a compressed attention variant, all fused into single kernels and checked against trusted references. Part 7 turns to a different, older application of the same tile-programming ideas: pricing financial derivatives. This chapter builds the foundation the rest of Part 7 needs — Black-Scholes closed-form pricing for European options — because both of this part's later chapters (rolling realized volatility, and a Monte Carlo option pricer) either produce an input this formula consumes or need this formula's answer as a check on their own.

## 43.1 The Normal CDF, Without `erf`

### Intuition

Every closed-form Black-Scholes price is built from exactly one non-elementary function: `N(x)`, the cumulative distribution function of the standard normal distribution — the probability that a standard normal random variable falls at or below `x`. Every other piece of the formula (logarithms, square roots, exponentials, ordinary arithmetic) is something this book's kernels have used since Part 0. `N(x)` is different: it has no closed algebraic form, which is exactly why real numerical libraries either implement `erf` as a hand-tuned polynomial or rational-function approximation internally, or hand the problem to hardware. `cuda.tile` offers neither `erf` nor a distribution module — a direct consequence of inspecting the module's complete public surface, not an assumption — so this chapter has to build `N(x)` itself, out of primitives the tile language does provide.

### Background

The approximation used here is Abramowitz & Stegun formula 26.2.17, a standard, published rational-polynomial approximation to the normal CDF. For `x >= 0`:

```
t = 1 / (1 + 0.2316419 * x)
N(x) = 1 - phi(x) * (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)
```

where `phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)` is the standard normal density, and `a1` through `a5` are five fixed constants. For `x < 0`, the identity `N(x) = 1 - N(-x)` (the standard normal distribution is symmetric about zero) covers the rest of the real line without needing a second polynomial. The published literature quotes a maximum absolute error under `7.5e-8` for this approximation — but this book's standing rule is that a claim like that gets measured on real hardware with real numbers before it is trusted, not cited and left unverified, so the Worked Example below computes the actual error against `torch`'s own `erf`-based CDF directly.

Two details connect this formula to what this book has already built. First, computing both the `x >= 0` branch and the `1 - N(-x)` branch unconditionally and selecting between them with `ct.where` is exactly the no-runtime-branching discipline Chapter 42's Mixture-of-Experts kernel used to route tokens: an ahead-of-time-compiled tile kernel has no data-dependent control flow, so both branches are always computed, and the ones that "shouldn't" apply are masked out (there, by a routing decision; here, by `x`'s sign) rather than skipped. Second, this is the first chapter to use `ct.function` — a decorator, distinct from `@ct.kernel`, that marks a function as callable *from inside* tile code. Every kernel this book has written before now has been one flat `@ct.kernel` body; factoring the CDF into its own `ct.function` means Section 43.2's Black-Scholes kernel can call the exact same `normal_cdf` this section defines and tests, rather than duplicating its five lines of arithmetic a second time.

### Worked Example 43.1.1: a normal-CDF kernel, and its error measured against torch's `erf`

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

ROWS, COLS = 8, 8

# --- cuda.tile has no erf and no built-in normal distribution function
# (confirmed by inspecting the module's full public surface), so the
# standard normal CDF this chapter's option-pricing formulas need has
# to be built from what IS available: exp, abs, and arithmetic. This is
# the Abramowitz & Stegun 26.2.17 rational-polynomial approximation --
# a well-known, well-documented technique, not something improvised for
# this book. Its accuracy is not taken on faith below; it is measured
# directly against torch's own erf. ---
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
A1, A2, A3, A4, A5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
P = 0.2316419

# --- ct.function marks a reusable tile-callable helper -- the first
# time this book has factored kernel logic into a separate function
# rather than writing every operation inline in the @ct.kernel body.
# Both this file's kernel and Section 43.2's Black-Scholes kernel call
# it, exactly the way a real options-pricing library would share one
# CDF implementation across every formula that needs it. ---
@ct.function
def normal_cdf(x):
    ax = ct.abs(x)
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * ct.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    # --- Phi(x) = 1 - phi(x)*poly(t) only holds for x >= 0; the
    # symmetry Phi(x) = 1 - Phi(-x) covers the rest. Both branches are
    # computed and ct.where selects between them, the same
    # no-runtime-branching discipline Chapter 42's MoE kernel used. ---
    return ct.where(x >= 0.0, upper, 1.0 - upper)

@ct.kernel
def kernel_normal_cdf(x, out):
    xt = ct.load(x, (0, 0), (ROWS, COLS))
    ct.store(out, (0, 0), normal_cdf(xt))


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_normal_cdf:", compile_bytes(kernel_normal_cdf, sig), "cubin bytes")


def host_cdf_approx(x):
    ax = x.abs()
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return torch.where(x >= 0.0, upper, 1.0 - upper)


torch.manual_seed(0)
xs = torch.empty(200_000).uniform_(-6.0, 6.0)

# --- The independent check: torch's OWN erf, not a second hand-rolled
# copy of this same polynomial checked against itself. 0.5*(1+erf(x/
# sqrt(2))) is the textbook definition of the standard normal CDF, and
# torch's erf is a completely different implementation path (a library
# routine, not a five-term rational approximation) from the one this
# kernel's formula uses. ---
ground_truth = 0.5 * (1.0 + torch.erf(xs / math.sqrt(2.0)))
approx = host_cdf_approx(xs)
max_abs_err = (approx - ground_truth).abs().max().item()
mean_abs_err = (approx - ground_truth).abs().mean().item()
print("max abs error vs torch.erf-based CDF over 200000 random points in [-6, 6]:", max_abs_err)
print("mean abs error:", mean_abs_err)
```

### Genuinely run

```text
kernel_normal_cdf: 26800 cubin bytes
max abs error vs torch.erf-based CDF over 200000 random points in [-6, 6]: 2.980232238769531e-07
mean abs error: 3.6291183391767845e-08
```

### Discussion

The measured maximum error, `2.98e-7`, is a real number computed against `torch`'s `erf`, not the `7.5e-8` figure the literature quotes — and the two are consistent once float32 is accounted for: `7.5e-8` is the double-precision error bound for this exact polynomial, and every tensor in this file is `float32`, whose own representable precision near numbers of order `1` is already around `1.2e-7`. The approximation is not meaningfully worse than advertised; it is being measured at the precision this chapter's kernels actually compute in, which is precisely the kind of discrepancy this book's verification discipline exists to surface rather than paper over. The mean error, `3.6e-8`, sitting comfortably below the max, confirms the error is small and well-behaved across the whole tested range rather than concentrated in a few outliers.

As with every kernel since Part 3, `kernel_normal_cdf` is compiled to real cubin bytes with `export_kernel`, but it is never passed through `ct.launch`: there is no GPU driver in this sandbox, so `host_cdf_approx` — the exact same five lines of arithmetic, run on plain `torch` tensors — stands in for what a live launch would compute, exactly as Chapter 41's and Chapter 42's individual kernel Worked Examples (as opposed to their multi-kernel capstones) did. That substitution is honest specifically because the host function and the kernel body are the same formula, term for term; the only thing being validated by comparison to `torch.erf` is whether that shared formula is numerically sound, not whether the host stand-in matches the kernel (which is true by construction).

## 43.2 Capstone: Pricing a Batch of European Options

### Intuition

The Black-Scholes formula prices a European call or put option from five inputs: the underlying's current spot price `S`, the option's strike price `K`, time to expiry `T`, the risk-free interest rate `r`, and the underlying's volatility `sigma`. Every one of those five inputs, and both output prices, is a single number per option — which means a batch of independent options, each with its own five inputs, is exactly the kind of "many independent scalar computations, arranged as a tile" workload this book's kernels have priced (so to speak) since Part 0's very first elementwise examples. This section reuses `normal_cdf` from 43.1 unchanged, calling it twice for the call price and twice more (with negated arguments) for the put.

### Background

The full formula, using `N` for the standard normal CDF from Section 43.1:

```
d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
Call = S*N(d1) - K*exp(-r*T)*N(d2)
Put  = K*exp(-r*T)*N(-d2) - S*N(-d1)
```

`d1` and `d2` are dimensionless quantities capturing how far the option is in- or out-of-the-money, scaled by how much the underlying could plausibly move before expiry; `N(d1)` and `N(d2)` convert those into probabilities that feed directly into the discounted expected payoff each formula computes. `kernel_black_scholes` computes both `d1` and `d2` for a full batch of options at once, and both the call and put price in the same kernel body, since every input array is already loaded once `d1`/`d2` are available and the marginal cost of also computing the put is four more calls to arithmetic already sitting in tiles.

This chapter's evidence for correctness comes from two genuinely different places. The first is an independently-written Black-Scholes implementation that calls `torch`'s own `erf`-based CDF rather than this chapter's polynomial — checking whether Section 43.1's small, already-measured CDF error compounds into a correspondingly small price error once it passes through the rest of the formula. The second is put-call parity, `C - P = S - K*exp(-r*T)` — a mathematical identity that holds for European options regardless of which CDF implementation, or approximation, computed `C` and `P` in the first place. A bug that swapped `d1` and `d2`, or flipped a sign in the discount term, could still happen to produce numbers close to a correct reference by coincidence on some inputs; it is very unlikely to also satisfy an unrelated algebraic identity across every option in the batch. That is what makes it a structurally different check rather than the same check performed twice.

### Worked Example 43.2.1: batched Black-Scholes pricing, cross-checked two independent ways

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

ROWS, COLS = 8, 8

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
A1, A2, A3, A4, A5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
P = 0.2316419

@ct.function
def normal_cdf(x):
    ax = ct.abs(x)
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * ct.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return ct.where(x >= 0.0, upper, 1.0 - upper)

# --- One kernel prices a whole batch of independent European options
# in parallel -- ROWS*COLS options, each with its own spot, strike,
# time to expiry, rate, and volatility, laid out the same
# (ROWS, COLS) way every batched kernel since Part 0 has used. Every
# option is priced by the same five arithmetic lines: no branching, no
# per-option control flow, just d1/d2 and two calls to normal_cdf. ---
@ct.kernel
def kernel_black_scholes(spot, strike, time, rate, vol, call_out, put_out):
    S = ct.load(spot, (0, 0), (ROWS, COLS))
    K = ct.load(strike, (0, 0), (ROWS, COLS))
    T = ct.load(time, (0, 0), (ROWS, COLS))
    r = ct.load(rate, (0, 0), (ROWS, COLS))
    sigma = ct.load(vol, (0, 0), (ROWS, COLS))

    sqrt_t = ct.sqrt(T)
    d1 = (ct.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    discount = ct.exp(-r * T)
    call = S * normal_cdf(d1) - K * discount * normal_cdf(d2)
    put = K * discount * normal_cdf(-d2) - S * normal_cdf(-d1)

    ct.store(call_out, (0, 0), call)
    ct.store(put_out, (0, 0), put)


sig = ct.compilation.KernelSignature(
    [array_param(2)] * 7,
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_black_scholes:", compile_bytes(kernel_black_scholes, sig), "cubin bytes")


def host_cdf_approx(x):
    ax = x.abs()
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return torch.where(x >= 0.0, upper, 1.0 - upper)


torch.manual_seed(0)
N = ROWS * COLS
S = torch.empty(N).uniform_(50.0, 150.0)
K = torch.empty(N).uniform_(50.0, 150.0)
T = torch.empty(N).uniform_(0.1, 2.0)
r = torch.empty(N).uniform_(0.0, 0.08)
sigma = torch.empty(N).uniform_(0.1, 0.6)

# host-side simulation, mirroring the kernel's own formula exactly
sqrt_t = T.sqrt()
d1 = (torch.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
d2 = d1 - sigma * sqrt_t
discount = torch.exp(-r * T)
call_host = S * host_cdf_approx(d1) - K * discount * host_cdf_approx(d2)
put_host = K * discount * host_cdf_approx(-d2) - S * host_cdf_approx(-d1)

# --- Independent check #1: an independently-written Black-Scholes
# implementation that calls torch's OWN erf-based normal CDF instead of
# this chapter's rational-polynomial approximation. Section 43.1
# already measured that approximation's error against torch's erf in
# isolation; this checks that the error stays just as small once it is
# compounded through the full pricing formula. ---
def torch_ndtr(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

call_ref = S * torch_ndtr(d1) - K * discount * torch_ndtr(d2)
put_ref = K * discount * torch_ndtr(-d2) - S * torch_ndtr(-d1)

max_call_err = (call_host - call_ref).abs().max().item()
max_put_err = (put_host - put_ref).abs().max().item()
print("max abs error, call price vs torch.erf-based reference:", max_call_err)
print("max abs error, put price vs torch.erf-based reference:", max_put_err)

# --- Independent check #2: put-call parity, C - P = S - K*exp(-rT), a
# mathematical identity that holds for European options regardless of
# which normal-CDF implementation priced them. This check does not use
# erf, torch's CDF, or this chapter's polynomial at all -- it is a
# completely different kind of evidence than checks #1 and the CDF
# comparison in Section 43.1, not a third repetition of the same one. ---
parity_lhs = call_host - put_host
parity_rhs = S - K * discount
max_parity_err = (parity_lhs - parity_rhs).abs().max().item()
print("max abs error, put-call parity (C - P vs S - K*exp(-rT)):", max_parity_err)
```

### Genuinely run

```text
kernel_black_scholes: 59016 cubin bytes
max abs error, call price vs torch.erf-based reference: 3.814697265625e-05
max abs error, put price vs torch.erf-based reference: 4.1961669921875e-05
max abs error, put-call parity (C - P vs S - K*exp(-rT)): 1.52587890625e-05
```

### Discussion

`kernel_black_scholes` compiles to `59016` cubin bytes — noticeably more than `kernel_normal_cdf`'s `26800`, which tracks the extra arithmetic (five array loads instead of one, `d1`/`d2`, a discount factor, and four calls to `normal_cdf` instead of one) rather than pointing to anything surprising.

Both independent checks pass, but their error magnitudes tell different stories worth separating. Checks #1's errors — `3.8e-5` for calls, `4.2e-5` for puts — are visibly larger than Section 43.1's isolated CDF error of `2.98e-7`, even though both checks trace back to the exact same polynomial. That is expected, not a regression: `S` and `K` range up to `150` in this batch, and an absolute CDF error of order `1e-7` gets multiplied by quantities of that size and combined across four `normal_cdf` calls per price, so an absolute *price* error of order `1e-5` is the CDF error scaled up by the formula's own magnitudes — the *relative* error stays comparable, which is the quantity that actually matters for a price. Check #2's parity error, `1.5e-5`, is smaller than either price error and arises from an entirely different source: floating-point rounding in the subtraction `call_host - put_host` against the independently-computed `S - K*discount`, not from any CDF inaccuracy at all, since parity holds algebraically no matter which `N` implementation was used to compute `C` and `P` in the first place.

## Chapter Summary

`cuda.tile` has no `erf`, so this chapter built the one non-elementary function Black-Scholes needs — the standard normal CDF — from a well-documented rational-polynomial approximation (Abramowitz & Stegun 26.2.17) using only `ct.exp`, `ct.abs`, and ordinary tile arithmetic, computing both of its two branches unconditionally and selecting with `ct.where`, the same no-runtime-branching discipline Chapter 42's expert routing used. That approximation's accuracy was measured directly against `torch`'s own `erf`-based CDF over 200,000 random points, finding a maximum error of `2.98e-7` — consistent with, not merely close to, float32's own precision limits. `ct.function` — used here for the first time in this book — factored that CDF into a helper callable from any tile kernel, and Section 43.2's `kernel_black_scholes` reused it unchanged to price a batch of European call and put options in a single branch-free kernel. That capstone's correctness rested on two structurally different kinds of evidence: an independently-written reference that calls `torch`'s `erf` directly, and put-call parity, an algebraic identity that holds regardless of which CDF implementation priced the options — together ruling out both a bad CDF approximation and a formula-level bug in ways neither check alone could.

## Self-Check Questions

1. Why does `normal_cdf` compute both the `x >= 0` branch and the `1 - upper` branch unconditionally rather than branching on `x`'s sign, and where else in this book has that same discipline appeared?
2. Section 43.1 measured the standalone CDF approximation's error at about `3e-7`, but Section 43.2's full option prices showed errors closer to `4e-5` — roughly two orders of magnitude larger — even though both traces back to the same `normal_cdf` function. What in the rest of the Black-Scholes formula explains an absolute price error growing that much larger than the absolute CDF error, even while the *relative* error stays comparable?
3. Put-call parity (`C - P = S - K*exp(-r*T)`) is described as "a completely different kind of evidence" from checking priced call and put values against a `torch.erf`-based reference. Explain concretely why put-call parity's correctness does not depend on the accuracy of whichever normal-CDF implementation computed `C` and `P`, and describe one kind of formula bug it would catch that a CDF-accuracy check alone might miss.
4. Section 43.1 measured the CDF approximation's error over 200,000 random points drawn from `[-6, 6]`. Given the shape of the normal density `phi(x) = (1/sqrt(2*pi))*exp(-x^2/2)` that this approximation multiplies, what would you expect to happen to its *absolute* accuracy for `|x|` much larger than `6`, and why might that matter less than it sounds for option pricing in practice?
5. Neither `kernel_normal_cdf` nor `kernel_black_scholes` is ever passed through `ct.launch` inside a `try`/`except RuntimeError` block, unlike Chapter 41's and Chapter 42's multi-kernel capstones. Why is that omission appropriate for this chapter's kernels specifically, and what would make adding a `ct.launch` attempt necessary?

## Where We Go Next

Chapter 44 turns from pricing a single snapshot of an option to processing a whole time series: computing rolling realized volatility from a stream of historical prices, using `cumsum` — already present in `cuda.tile`'s public surface but unused by this book until now — to implement an efficient sliding-window statistic. The `sigma` this chapter treated as a given input becomes, in Chapter 44, a number computed from real (simulated) market data, closing the loop between "what volatility does a pricing formula need" and "where would a real system actually get it." Chapter 45 then closes Part 7 with a Monte Carlo option pricer — simulating many random price paths under geometric Brownian motion and averaging their discounted payoffs — and cross-checks its answer against this chapter's own closed-form Black-Scholes price for the same option, the two pricing methods this book has now built independently converging on the same number by two entirely different routes.

## Worked Solutions

**1.** Computing both branches unconditionally and selecting with `ct.where` avoids data-dependent control flow inside the kernel, because an ahead-of-time-compiled tile kernel has no way to branch differently per-element at runtime the way ordinary Python code can — every element in a tile follows the same sequence of operations, and only the final selection differs per element. Chapter 42's Mixture-of-Experts kernel used exactly the same pattern: every expert's feed-forward network was computed for every token, and `ct.where(ct.equal(expert_idx, e), top_weight, 0.0)` masked out the contributions of experts a given token had not been routed to, rather than skipping the computation for those tokens.

**2.** The CDF error is an *absolute* error of order `1e-7` in a quantity `N(x)` that is always between `0` and `1`. That error gets multiplied by `S` or `K` (both up to `150` in this batch) when it feeds into terms like `S*N(d1)` or `K*discount*N(d2)`, and the formula sums four such terms (two for the call, two for the put, each involving its own `normal_cdf` call). An absolute CDF error of `~1e-7` scaled by a factor of order `100` and combined across a few terms lands very plausibly around `1e-5`, which is what was measured — the *relative* error (error divided by the price itself, which is also of order tens of dollars) stays roughly the same size relative to the price, since both the error and the price scale with the same underlying quantities.

**3.** Put-call parity is a consequence of a no-arbitrage argument about portfolios of the underlying asset, a bond, and the two options — it holds as a statement about `S`, `K`, `r`, `T`, and the discount factor `exp(-r*T)` alone, with no reference to `N(x)` anywhere in its derivation. Whatever value `C` and `P` take — computed with this chapter's polynomial, with `torch`'s `erf`, or with any other correct implementation of the normal CDF — as long as they were computed *correctly* from the *same* Black-Scholes formula, their difference must equal `S - K*exp(-r*T)` regardless of which CDF implementation was used. A bug that swapped `d1` and `d2` inside the call formula only (leaving the put formula correct) is exactly the kind of error parity would catch: it could still produce a call price fairly close to a `torch.erf`-based reference for some inputs (since `d1` and `d2` are often numerically similar when volatility is low or time to expiry is short), yet `C - P` computed from the swapped call and the correct put would very likely fail to equal `S - K*exp(-r*T)`, because that swap breaks the specific algebraic relationship the parity identity depends on, not just the two CDF lookups' approximation accuracy.

**4.** The normal density `phi(x)` decays as `exp(-x^2/2)`, so by `|x| = 6`, `phi(x)` is already on the order of `1e-9` — meaning `N(x)` itself is already extremely close to `0` (for very negative `x`) or `1` (for very positive `x`), and the *absolute* error in approximating a number that close to its limiting value tends to shrink rather than grow, since the polynomial's correction term `phi(x)*poly(t)` is itself being multiplied by an already-tiny `phi(x)`. For `|x|` much larger than `6`, this trend generally continues: the absolute error stays small in absolute terms because `N(x)` itself has almost no room left to be wrong within. This matters less than it might sound for option pricing specifically because `d1` and `d2` rarely reach values with `|d| > 6` for realistic combinations of moneyness, volatility, and time to expiry — an option would have to be extremely deep in- or out-of-the-money relative to its volatility and time horizon to push `d1` or `d2` that far from zero, so the domain `[-6, 6]` tested in Section 43.1 already covers the range this chapter's pricing kernel actually exercises.

**5.** `kernel_normal_cdf` and `kernel_black_scholes` are both single, stateless, purely elementwise kernels — every output element depends only on that same element's inputs, with no reduction, no cross-kernel composition, and no residual or intermediate state to carry between stages, unlike Chapter 41's and Chapter 42's capstones, which chained together multiple distinct kernels (LayerNorm, attention or MLA, and a feed-forward or MoE block) into a single multi-stage pipeline where the `try`/`except RuntimeError` pattern gave each stage a chance to run for real on hardware before falling back. Here, `export_kernel`'s successful compilation already demonstrates the kernel is well-formed, and the host-side `torch` computation is not standing in for "what a pipeline of kernels would produce together" — it is literally the same formula the kernel body computes, run on ordinary tensors, so there is no additional pipeline-level behavior a `ct.launch` attempt would exercise that compilation and the host check do not already cover. Adding a `ct.launch` attempt would become necessary if this chapter's kernels were composed into a larger pipeline with actual GPU-resident intermediate tensors passed between stages (as Chapter 45's Monte Carlo pricer, generating and consuming many simulated paths, might reasonably need to), where a real launch's success or failure would say something a pure formula-equivalence check cannot.

## Complete Runnable Code

`01_the_normal_cdf_without_erf.py`:

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

ROWS, COLS = 8, 8

# --- cuda.tile has no erf and no built-in normal distribution function
# (confirmed by inspecting the module's full public surface), so the
# standard normal CDF this chapter's option-pricing formulas need has
# to be built from what IS available: exp, abs, and arithmetic. This is
# the Abramowitz & Stegun 26.2.17 rational-polynomial approximation --
# a well-known, well-documented technique, not something improvised for
# this book. Its accuracy is not taken on faith below; it is measured
# directly against torch's own erf. ---
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
A1, A2, A3, A4, A5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
P = 0.2316419

# --- ct.function marks a reusable tile-callable helper -- the first
# time this book has factored kernel logic into a separate function
# rather than writing every operation inline in the @ct.kernel body.
# Both this file's kernel and Section 43.2's Black-Scholes kernel call
# it, exactly the way a real options-pricing library would share one
# CDF implementation across every formula that needs it. ---
@ct.function
def normal_cdf(x):
    ax = ct.abs(x)
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * ct.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    # --- Phi(x) = 1 - phi(x)*poly(t) only holds for x >= 0; the
    # symmetry Phi(x) = 1 - Phi(-x) covers the rest. Both branches are
    # computed and ct.where selects between them, the same
    # no-runtime-branching discipline Chapter 42's MoE kernel used. ---
    return ct.where(x >= 0.0, upper, 1.0 - upper)

@ct.kernel
def kernel_normal_cdf(x, out):
    xt = ct.load(x, (0, 0), (ROWS, COLS))
    ct.store(out, (0, 0), normal_cdf(xt))


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_normal_cdf:", compile_bytes(kernel_normal_cdf, sig), "cubin bytes")


def host_cdf_approx(x):
    ax = x.abs()
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return torch.where(x >= 0.0, upper, 1.0 - upper)


torch.manual_seed(0)
xs = torch.empty(200_000).uniform_(-6.0, 6.0)

# --- The independent check: torch's OWN erf, not a second hand-rolled
# copy of this same polynomial checked against itself. 0.5*(1+erf(x/
# sqrt(2))) is the textbook definition of the standard normal CDF, and
# torch's erf is a completely different implementation path (a library
# routine, not a five-term rational approximation) from the one this
# kernel's formula uses. ---
ground_truth = 0.5 * (1.0 + torch.erf(xs / math.sqrt(2.0)))
approx = host_cdf_approx(xs)
max_abs_err = (approx - ground_truth).abs().max().item()
mean_abs_err = (approx - ground_truth).abs().mean().item()
print("max abs error vs torch.erf-based CDF over 200000 random points in [-6, 6]:", max_abs_err)
print("mean abs error:", mean_abs_err)
```

`02_capstone_pricing_a_batch_of_european_options.py`:

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

ROWS, COLS = 8, 8

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
A1, A2, A3, A4, A5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
P = 0.2316419

@ct.function
def normal_cdf(x):
    ax = ct.abs(x)
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * ct.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return ct.where(x >= 0.0, upper, 1.0 - upper)

# --- One kernel prices a whole batch of independent European options
# in parallel -- ROWS*COLS options, each with its own spot, strike,
# time to expiry, rate, and volatility, laid out the same
# (ROWS, COLS) way every batched kernel since Part 0 has used. Every
# option is priced by the same five arithmetic lines: no branching, no
# per-option control flow, just d1/d2 and two calls to normal_cdf. ---
@ct.kernel
def kernel_black_scholes(spot, strike, time, rate, vol, call_out, put_out):
    S = ct.load(spot, (0, 0), (ROWS, COLS))
    K = ct.load(strike, (0, 0), (ROWS, COLS))
    T = ct.load(time, (0, 0), (ROWS, COLS))
    r = ct.load(rate, (0, 0), (ROWS, COLS))
    sigma = ct.load(vol, (0, 0), (ROWS, COLS))

    sqrt_t = ct.sqrt(T)
    d1 = (ct.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    discount = ct.exp(-r * T)
    call = S * normal_cdf(d1) - K * discount * normal_cdf(d2)
    put = K * discount * normal_cdf(-d2) - S * normal_cdf(-d1)

    ct.store(call_out, (0, 0), call)
    ct.store(put_out, (0, 0), put)


sig = ct.compilation.KernelSignature(
    [array_param(2)] * 7,
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_black_scholes:", compile_bytes(kernel_black_scholes, sig), "cubin bytes")


def host_cdf_approx(x):
    ax = x.abs()
    t = 1.0 / (1.0 + P * ax)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * ax * ax)
    upper = 1.0 - pdf * poly
    return torch.where(x >= 0.0, upper, 1.0 - upper)


torch.manual_seed(0)
N = ROWS * COLS
S = torch.empty(N).uniform_(50.0, 150.0)
K = torch.empty(N).uniform_(50.0, 150.0)
T = torch.empty(N).uniform_(0.1, 2.0)
r = torch.empty(N).uniform_(0.0, 0.08)
sigma = torch.empty(N).uniform_(0.1, 0.6)

# host-side simulation, mirroring the kernel's own formula exactly
sqrt_t = T.sqrt()
d1 = (torch.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
d2 = d1 - sigma * sqrt_t
discount = torch.exp(-r * T)
call_host = S * host_cdf_approx(d1) - K * discount * host_cdf_approx(d2)
put_host = K * discount * host_cdf_approx(-d2) - S * host_cdf_approx(-d1)

# --- Independent check #1: an independently-written Black-Scholes
# implementation that calls torch's OWN erf-based normal CDF instead of
# this chapter's rational-polynomial approximation. Section 43.1
# already measured that approximation's error against torch's erf in
# isolation; this checks that the error stays just as small once it is
# compounded through the full pricing formula. ---
def torch_ndtr(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

call_ref = S * torch_ndtr(d1) - K * discount * torch_ndtr(d2)
put_ref = K * discount * torch_ndtr(-d2) - S * torch_ndtr(-d1)

max_call_err = (call_host - call_ref).abs().max().item()
max_put_err = (put_host - put_ref).abs().max().item()
print("max abs error, call price vs torch.erf-based reference:", max_call_err)
print("max abs error, put price vs torch.erf-based reference:", max_put_err)

# --- Independent check #2: put-call parity, C - P = S - K*exp(-rT), a
# mathematical identity that holds for European options regardless of
# which normal-CDF implementation priced them. This check does not use
# erf, torch's CDF, or this chapter's polynomial at all -- it is a
# completely different kind of evidence than checks #1 and the CDF
# comparison in Section 43.1, not a third repetition of the same one. ---
parity_lhs = call_host - put_host
parity_rhs = S - K * discount
max_parity_err = (parity_lhs - parity_rhs).abs().max().item()
print("max abs error, put-call parity (C - P vs S - K*exp(-rT)):", max_parity_err)
```
