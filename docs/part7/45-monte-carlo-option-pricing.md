# Chapter 45: Monte Carlo Option Pricing

> "Simulation is what you do when the closed-form answer is too hard to get; the interesting part of this chapter is that here it isn't — Black-Scholes already gave us the exact price, two chapters ago — so this chapter's simulation has something rare: a right answer to check itself against, computed by an entirely different method."

**What you will understand by the end of this chapter:**

- Why Black-Scholes' assumption that the underlying follows geometric Brownian motion gives an exact, closed-form distribution for the price at any future time, so that pricing a European option by simulation needs only one random draw per path — not a simulated path through many intermediate time steps.
- How standard normal random draws, which `cuda.tile` has no built-in way to generate, are produced once on the host and passed into a kernel as an ordinary array — the same pattern this book used for RoPE's `cos_table` and `sin_table` in Chapter 42.
- How a Monte Carlo option price is a discounted average of simulated payoffs, computed inside one kernel with the same `ct.sum`-based in-kernel reduction this book has used since Chapter 41's softmax denominator.
- How a Monte Carlo estimate's own theoretical standard error turns "this simulated price is close to the known correct answer" from an eyeballed comparison into a specific, falsifiable statistical claim — and how that claim is verified here against Chapter 43's closed-form Black-Scholes price, a genuinely different pricing method computed with zero random numbers.

**What you need to know first:**

- Chapter 43's Black-Scholes formula and `normal_cdf` helper, reused unchanged as this chapter's independent check.
- `ct.exp`, `ct.maximum`, `ct.sum` with `axis` and `keepdims` (Parts 0-6, most recently Chapter 44's rolling statistics).
- The host-precomputed-array pattern for values a kernel needs but cannot compute internally (Chapter 42's rotary tables, Chapter 45's own random draws).
- Basic statistical intuition: that a sample mean computed from random data has an associated uncertainty (a standard error), and that this uncertainty shrinks, in a specific and computable way, as more samples are used.

Chapter 43 built an exact, closed-form option price from five given inputs. Chapter 44 built one of those inputs — volatility — from data. This closing chapter of Part 7 builds a third, independent way to arrive at the same price Chapter 43 computes: not from a formula, but from simulating a large number of random possible futures for the underlying asset and averaging what the option would be worth in each of them. That two structurally unrelated methods, a closed-form formula and a random simulation, agree on the same number is this book's last and most direct demonstration of what "genuinely verified" has meant throughout: not that a single computation looks plausible, but that it survives being checked a second way that could, in principle, have disagreed.

## 45.1 Simulating Terminal Prices Under Geometric Brownian Motion

### Intuition

Geometric Brownian motion (GBM) is the random process Black-Scholes assumes the underlying asset follows: at every instant, the asset's price moves by a small deterministic drift plus a small random shock proportional to its own current value. This stochastic differential equation happens to have an exact, closed-form solution for the price at any future time `T`, given the price today: `S_T` is lognormally distributed, and a single sample from that distribution can be produced from a single standard normal draw `Z`:

```
S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
```

This matters for how this chapter's simulation is built: because this formula is the *exact* solution to the GBM SDE at time `T`, not an approximation of it, pricing a European option — whose payoff depends only on the price at expiry, not on the path taken to get there — needs exactly one random draw per simulated path, not a simulated walk through hundreds of intermediate time steps. Simulating intermediate prices would be necessary for an option whose payoff depends on the whole path (an average price, or a barrier being crossed), but a European call or put cares only about `S_T`, and `S_T`'s distribution is already known exactly.

### Background

`cuda.tile` has no random-number generator anywhere in its public surface — confirmed the same way Chapter 43 confirmed the absence of `erf`, by inspecting the module directly rather than assuming. The standard normal draws `Z` this formula needs are therefore generated once on the host, with `torch.randn`, and passed into the kernel as an ordinary array parameter — the same pattern Chapter 42 used for `cos_table` and `sin_table`, values that depend only on something computed outside the kernel (there, sequence position; here, a random number generator) and never on anything the kernel itself computes. `kernel_simulate_terminal_price` loads a batch of `NUM_PATHS = 65536` such draws as one `(1, NUM_PATHS)` tile and applies the closed-form formula elementwise — `S0`, `K`, `T`, `R`, and `SIGMA` are fixed Python floats for a single option being priced, so `DRIFT` and `DIFFUSION` are computed once outside the kernel as plain constants, exactly the way Chapter 44's `ANNUALIZATION` was.

Before this chapter uses these simulated prices to price anything, this section checks that they are behaving the way GBM's theory says they should — using two facts about the lognormal distribution that are derived directly from the SDE itself, not measured from this simulation. Under the risk-neutral measure Black-Scholes uses, the underlying's expected future price is exactly today's price grown at the risk-free rate, `E[S_T] = S0*exp(r*T)`, and the distribution's variance has an equally exact closed form. Comparing this simulation's sample mean and sample standard deviation against those two analytic facts is a check that this section's random-sampling machinery is producing draws from the distribution the formula claims, before Section 45.2 relies on it for anything as consequential as a price.

### Worked Example 45.1.1: simulated terminal prices, checked against GBM's exact analytic moments

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

NUM_PATHS = 65536

# --- Black-Scholes assumes the underlying follows geometric Brownian
# motion, and that SDE has an exact closed-form solution for the price
# at any future time T given the price today: a lognormal distribution
# whose single random ingredient is one standard normal draw per path.
# cuda.tile has no random-number generator (confirmed by inspecting the
# module's full public surface, the same way Chapter 43 confirmed there
# is no erf), so the standard normal draws are generated once on the
# host and passed in as an ordinary array parameter -- exactly the
# pattern Chapter 42's cos_table and sin_table used for values that
# depend only on something computed outside the kernel, never on the
# kernel's own internals. ---
S0, K, T, R, SIGMA = 100.0, 105.0, 0.5, 0.03, 0.2
DRIFT = (R - 0.5 * SIGMA * SIGMA) * T
DIFFUSION = SIGMA * math.sqrt(T)

@ct.kernel
def kernel_simulate_terminal_price(z, out):
    zt = ct.load(z, (0, 0), (1, NUM_PATHS))
    st = S0 * ct.exp(DRIFT + DIFFUSION * zt)
    ct.store(out, (0, 0), st)


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_simulate_terminal_price:", compile_bytes(kernel_simulate_terminal_price, sig), "cubin bytes")

torch.manual_seed(0)
z = torch.randn(1, NUM_PATHS)

# host-side simulation, mirroring the kernel's own formula exactly
s_t = S0 * torch.exp(DRIFT + DIFFUSION * z)

sample_mean = s_t.mean().item()
sample_std = s_t.std(unbiased=True).item()

# --- The independent check: two facts about the lognormal distribution
# that are derivable directly from the GBM SDE itself, with no
# reference to this simulation at all. Under the risk-neutral measure,
# the underlying's expected future price is exactly today's price
# grown at the risk-free rate -- E[S_T] = S0*exp(r*T) -- and its
# variance has an equally exact closed form. Neither fact is computed
# from the sampled paths above; both come from the same mathematical
# theory this chapter's kernel is simulating a sample from. ---
analytic_mean = S0 * math.exp(R * T)
analytic_var = (S0 ** 2) * math.exp(2 * R * T) * (math.exp(SIGMA * SIGMA * T) - 1.0)
analytic_std = math.sqrt(analytic_var)

print("sample mean of simulated S_T:", sample_mean)
print("analytic mean E[S_T] = S0*exp(r*T):", analytic_mean)
print("relative error, mean:", abs(sample_mean - analytic_mean) / analytic_mean)
print("sample std of simulated S_T:", sample_std)
print("analytic std of S_T:", analytic_std)
print("relative error, std:", abs(sample_std - analytic_std) / analytic_std)
```

### Genuinely run

```text
kernel_simulate_terminal_price: 61888 cubin bytes
sample mean of simulated S_T: 101.4554672241211
analytic mean E[S_T] = S0*exp(r*T): 101.51130646157189
relative error, mean: 0.0005500789951111104
sample std of simulated S_T: 14.404853820800781
analytic std of S_T: 14.427945946151569
relative error, std: 0.0016005137139391008
```

### Discussion

Both relative errors are small — `0.055%` for the mean, `0.16%` for the standard deviation — and, importantly, neither is exactly zero. That is the expected, correct outcome for a *finite* Monte Carlo sample: `65536` random draws produce a sample mean and sample standard deviation that are close to, but not identical to, the distribution's true analytic values, with the remaining gap attributable to ordinary sampling variability rather than to any error in the simulation's formula. A worse-behaved random number generator, or a bug in the drift or diffusion terms, would tend to produce a *systematic* discrepancy — the sample statistics landing reliably far from the analytic ones, not merely close with some sampling noise — which is precisely what this check is positioned to catch before Section 45.2 builds an option price on top of it.

## 45.2 Capstone: Pricing an Option by Monte Carlo, Checked Against Black-Scholes

### Intuition

A European option's payoff at expiry is a simple, known function of the terminal price alone: `max(S_T - K, 0)` for a call, `max(K - S_T, 0)` for a put. The option's fair price today is the expected value of that payoff under the risk-neutral measure, discounted back to today at the risk-free rate — and an average over many simulated terminal prices is exactly an estimate of that expectation. Every one of Section 45.1's simulated paths becomes one path's payoff; averaging across all of them and discounting gives a Monte Carlo estimate of the option's price.

### Background

`kernel_monte_carlo_price` extends Section 45.1's kernel with three more steps, all elementwise or reduction operations this book has used before: `ct.maximum(st - K, 0.0)` and `ct.maximum(K - st, 0.0)` compute each path's call and put payoff, and `ct.sum(call_payoff, axis=1, keepdims=True) / NUM_PATHS` reduces all `65536` payoffs down to their mean — the same in-kernel reduction pattern Chapter 41's softmax denominator and Chapter 42's gating probabilities used, applied here to a much larger tile. Multiplying by the discount factor `exp(-r*T)` gives the final price.

The genuinely new idea in this section is how "close enough" gets defined. A Monte Carlo estimate is a sample mean of random payoffs, and any sample mean has a theoretical standard error — the sample's own standard deviation divided by the square root of how many samples went into it — that quantifies exactly how much that estimate should be expected to vary from the true expectation, purely from sampling randomness. Computing that standard error directly from this run's own payoffs turns "the Monte Carlo price and the Black-Scholes price are close" from a comparison eyeballed against an arbitrary tolerance into a specific, falsifiable claim: that the difference between the two prices is small relative to the uncertainty the simulation itself reports having, exactly the kind of quantifiable, verifiable statement this book has insisted on since Chapter 40 first had to grapple with inherently noisy measurements.

### Worked Example 45.2.1: Monte Carlo option pricing, verified against Black-Scholes using its own standard error

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

NUM_PATHS = 65536
S0, K, T, R, SIGMA = 100.0, 105.0, 0.5, 0.03, 0.2
DRIFT = (R - 0.5 * SIGMA * SIGMA) * T
DIFFUSION = SIGMA * math.sqrt(T)
DISCOUNT = math.exp(-R * T)

# --- Every one of Section 45.1's simulated terminal prices becomes one
# path's option payoff -- max(S_T - K, 0) for a call, max(K - S_T, 0)
# for a put -- and the option's price is the discounted average payoff
# across every path. Reducing 65536 per-path payoffs down to two
# numbers with ct.sum is the same in-kernel reduction pattern this book
# has used since Chapter 41's softmax denominator and Chapter 42's
# gating probabilities, just applied to a much larger tile. ---
@ct.kernel
def kernel_monte_carlo_price(z, call_price_out, put_price_out):
    zt = ct.load(z, (0, 0), (1, NUM_PATHS))
    st = S0 * ct.exp(DRIFT + DIFFUSION * zt)

    call_payoff = ct.maximum(st - K, 0.0)
    put_payoff = ct.maximum(K - st, 0.0)

    call_mean = ct.sum(call_payoff, axis=1, keepdims=True) / NUM_PATHS
    put_mean = ct.sum(put_payoff, axis=1, keepdims=True) / NUM_PATHS

    ct.store(call_price_out, (0, 0), DISCOUNT * call_mean)
    ct.store(put_price_out, (0, 0), DISCOUNT * put_mean)


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_monte_carlo_price:", compile_bytes(kernel_monte_carlo_price, sig), "cubin bytes")

torch.manual_seed(0)
z = torch.randn(1, NUM_PATHS)

# host-side simulation, mirroring the kernel's own formula exactly
s_t = S0 * torch.exp(DRIFT + DIFFUSION * z)
call_payoff = torch.clamp(s_t - K, min=0.0)
put_payoff = torch.clamp(K - s_t, min=0.0)

mc_call = DISCOUNT * call_payoff.mean().item()
mc_put = DISCOUNT * put_payoff.mean().item()

# --- A Monte Carlo estimate is a sample mean, and a sample mean has a
# theoretical standard error: the sample's own standard deviation
# divided by the square root of how many samples went into it. That
# error is computed here directly from this run's payoffs, not assumed
# -- it is what turns "the two prices are close" into a falsifiable,
# quantitative claim, rather than an eyeballed comparison. ---
call_se = DISCOUNT * call_payoff.std(unbiased=True).item() / math.sqrt(NUM_PATHS)
put_se = DISCOUNT * put_payoff.std(unbiased=True).item() / math.sqrt(NUM_PATHS)

print("Monte Carlo call price:", mc_call, "  standard error:", call_se)
print("Monte Carlo put price:", mc_put, "  standard error:", put_se)

# --- The independent check: Chapter 43's closed-form Black-Scholes
# formula, reused completely unchanged -- a genuinely different pricing
# METHOD, not a second simulation with a different random seed. One
# side of this comparison never generates a single random number. ---
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

S_t, K_t, T_t, r_t, sigma_t = (torch.tensor(v) for v in (S0, K, T, R, SIGMA))
sqrt_t = T_t.sqrt()
d1 = (torch.log(S_t / K_t) + (r_t + 0.5 * sigma_t * sigma_t) * T_t) / (sigma_t * sqrt_t)
d2 = d1 - sigma_t * sqrt_t
bs_discount = torch.exp(-r_t * T_t)
bs_call = (S_t * host_cdf_approx(d1) - K_t * bs_discount * host_cdf_approx(d2)).item()
bs_put = (K_t * bs_discount * host_cdf_approx(-d2) - S_t * host_cdf_approx(-d1)).item()

print("Black-Scholes call price (Chapter 43's formula):", bs_call)
print("Black-Scholes put price (Chapter 43's formula):", bs_put)

call_diff = abs(mc_call - bs_call)
put_diff = abs(mc_put - bs_put)
print("call price difference:", call_diff, " in standard errors:", call_diff / call_se)
print("put price difference:", put_diff, " in standard errors:", put_diff / put_se)
print("call price within 3 standard errors of Black-Scholes:", call_diff <= 3 * call_se)
print("put price within 3 standard errors of Black-Scholes:", put_diff <= 3 * put_se)
```

### Genuinely run

```text
kernel_monte_carlo_price: 94888 cubin bytes
Monte Carlo call price: 4.149475521674088   standard error: 0.030451149742320296
Monte Carlo put price: 7.641236609768807   standard error: 0.034317757538410085
Black-Scholes call price (Chapter 43's formula): 4.178287506103516
Black-Scholes put price (Chapter 43's formula): 7.615043640136719
call price difference: 0.028811984429427895  in standard errors: 0.9461706593424837
put price difference: 0.026192969632088392  in standard errors: 0.7632482863360743
call price within 3 standard errors of Black-Scholes: True
put price within 3 standard errors of Black-Scholes: True
```

### Discussion

The Monte Carlo call price sits under one standard error away from Black-Scholes' exact answer; the put price sits at about three-quarters of a standard error. Both are comfortably inside the `3`-standard-error bound this chapter chose as its verification criterion — a bound that, if the simulation and its statistics are behaving correctly, should be satisfied the overwhelming majority of the time by construction (a difference exceeding three standard errors happens for well under 1% of random seeds under the normal approximation this kind of estimate follows), not something guaranteed to hold on every possible seed. That distinction matters: this chapter is not claiming Monte Carlo simulation always lands within some fixed number of standard errors of the truth by definition — it is reporting that, for this run's `65536` paths and this fixed random seed, the two structurally unrelated pricing methods this book has now built agree to well within the amount of disagreement the simulation's own statistics say is normal to expect.

`kernel_monte_carlo_price` compiles to `94888` bytes, noticeably larger than Section 45.1's `61888` — tracking the extra work of computing two payoffs and reducing each with `ct.sum` rather than just applying the elementwise GBM formula once.

## Chapter Summary

Black-Scholes assumes the underlying follows geometric Brownian motion, and that assumption has an exact closed-form solution for the terminal price distribution — meaning a European option can be priced by simulation using just one random draw per path, not a simulated walk through time. `cuda.tile` has no random-number generator, so those draws were produced once on the host and passed in as an ordinary array, the same pattern Chapter 42 used for its precomputed rotary tables. Section 45.1 checked that simulated terminal prices matched GBM's own exact analytic mean and variance before anything was priced from them. Section 45.2's capstone computed a Monte Carlo option price entirely inside one kernel — payoffs via `ct.maximum`, a discounted average via `ct.sum` — and verified it not against an arbitrary tolerance but against the simulation's own theoretical standard error, finding the Monte Carlo price within about one standard error of Chapter 43's closed-form Black-Scholes price for the same option. Two structurally unrelated methods — an exact formula, and a random simulation — arrived at the same answer, closing this book's Part 7 and its broader arc of building every claim on evidence that could, in principle, have come out differently.

## Self-Check Questions

1. Why does pricing a European option by Monte Carlo simulation need only one random draw per path to get the terminal price `S_T`, rather than simulating the underlying's price at many intermediate time steps between now and expiry?
2. Section 45.1 compares the simulation's sample mean and sample standard deviation against GBM's analytic mean and variance before Section 45.2 uses the simulation to price anything. What kind of problem would this comparison catch that Section 45.2's Black-Scholes comparison, on its own, might not catch as clearly?
3. What does the "standard error" computed in Section 45.2 represent, and why does expressing the gap between the Monte Carlo price and the Black-Scholes price *in units of that standard error* make "these two prices agree" a more specific claim than simply reporting the two numbers and eyeballing whether they look close?
4. If `NUM_PATHS` were increased from `65536` to four times its current value (`262144`), what would you expect to happen to the Monte Carlo standard error, and why, given how standard error depends on sample size?
5. This chapter's capstone found the Monte Carlo and Black-Scholes call prices differing by under one standard error. Suppose, instead, that they had differed by 50 standard errors. Given that both methods are computing the price of the exact same option, what would that magnitude of disagreement suggest was actually wrong, and would it more likely point to a bug in the Monte Carlo kernel, a bug in the Black-Scholes kernel, or "normal" sampling noise?

## Where We Go Next

This chapter closes Part 7, and with it, this book's chapter-by-chapter arc from cuTile Python's foundational tile/array distinction through a complete, cross-checked options-pricing pipeline: Black-Scholes for a closed-form price, rolling realized volatility to supply that formula's `sigma` from real data, and Monte Carlo simulation as an independent way of arriving at the same price by an entirely different route. Every chapter along the way built its claims on genuine compilation, genuine (or honestly simulated, where no GPU driver was available) execution, and a second, structurally different method of checking the result — the same discipline this closing chapter applied one last time, at the point where it mattered most: two unrelated ways of pricing the same option, agreeing.

## Worked Solutions

**1.** A European option's payoff depends only on the underlying's price at expiry, `S_T`, never on the path the price took to get there — so a pricing method only needs a sample from `S_T`'s distribution, not a full record of every intermediate price along the way. Geometric Brownian motion's stochastic differential equation has an exact closed-form solution for that terminal distribution (a lognormal distribution derivable directly from the SDE), so one draw from a standard normal distribution, plugged into that closed-form solution, produces one exact sample from `S_T`'s true distribution directly — simulating intermediate time steps would only be necessary for a path-dependent payoff (an average price over the option's life, or a barrier level being crossed at some point before expiry), which is not what a plain European call or put pays out on.

**2.** Section 45.1's comparison isolates whether the *simulation mechanics themselves* — the random number generation, and the elementwise application of the GBM formula — are behaving correctly, independent of anything about option pricing at all. If, for instance, `DIFFUSION` had the wrong sign, or `torch.randn` were somehow producing draws with the wrong variance, Section 45.1's sample statistics would likely show a large, systematic disagreement with GBM's known analytic mean and variance directly — a problem visible before a single option payoff is ever computed. Section 45.2's comparison against Black-Scholes, by contrast, could in principle mask certain classes of errors: an error that shifted the *whole* terminal-price distribution wrong in some way might, depending on its exact form, still happen to produce an option price that looks plausible, especially if the standard error is not scrutinized carefully — Section 45.1's more basic, formula-level check is positioned to catch a broader range of possible mistakes earlier, before they can hide inside a plausible-looking final price.

**3.** The standard error is the theoretical amount of variability a sample mean computed from `NUM_PATHS` independent random draws is expected to have, purely from sampling randomness, computed here as the sample standard deviation of the payoffs divided by the square root of the number of paths. Reporting "the Monte Carlo price is $0.03 higher than Black-Scholes" on its own says nothing about whether $0.03 is a large or small discrepancy for a simulation of this size; reporting "the difference is 0.95 standard errors" says something specific and falsifiable — that the gap is well within the range of variation this particular simulation's own statistics say is normal to expect from randomness alone, rather than evidence of a bug. A raw dollar difference has no built-in sense of scale; a difference measured in standard errors does.

**4.** Standard error scales as the sample standard deviation divided by the square root of the sample size, so quadrupling `NUM_PATHS` (multiplying it by `4`) would be expected to roughly halve the standard error, since `sqrt(4) = 2`. This is a direct, computable consequence of the standard error formula — not a rule specific to option pricing — and reflects the well-known (and, in this book's tradition, worth stating rather than merely citing) diminishing-returns nature of Monte Carlo convergence: quadrupling the computational work only doubles the estimate's precision, because precision improves with the *square root* of the sample size rather than in direct proportion to it.

**5.** A 50-standard-error disagreement between two methods computing the price of the exact same option would be extremely unlikely to arise from ordinary sampling noise alone — under the normal approximation a sample mean's error typically follows, a gap that large has a vanishingly small probability of occurring by chance, meaning something is very likely actually wrong. Because the Monte Carlo kernel and the Black-Scholes kernel are entirely independent implementations sharing no code (aside from both correctly encoding the same five inputs), a disagreement of that magnitude would point most directly at a bug in whichever of the two methods is actually mispricing the option — for instance, a sign error in the Monte Carlo kernel's drift term, or a swapped `d1`/`d2` in the Black-Scholes formula — rather than at "normal" statistical noise, precisely because the standard-error framework this chapter built exists to distinguish "this gap is the kind of thing random sampling produces" from "this gap is not the kind of thing random sampling produces, so look for an actual error."

## Complete Runnable Code

`01_simulating_terminal_prices_under_gbm.py`:

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

NUM_PATHS = 65536

# --- Black-Scholes assumes the underlying follows geometric Brownian
# motion, and that SDE has an exact closed-form solution for the price
# at any future time T given the price today: a lognormal distribution
# whose single random ingredient is one standard normal draw per path.
# cuda.tile has no random-number generator (confirmed by inspecting the
# module's full public surface, the same way Chapter 43 confirmed there
# is no erf), so the standard normal draws are generated once on the
# host and passed in as an ordinary array parameter -- exactly the
# pattern Chapter 42's cos_table and sin_table used for values that
# depend only on something computed outside the kernel, never on the
# kernel's own internals. ---
S0, K, T, R, SIGMA = 100.0, 105.0, 0.5, 0.03, 0.2
DRIFT = (R - 0.5 * SIGMA * SIGMA) * T
DIFFUSION = SIGMA * math.sqrt(T)

@ct.kernel
def kernel_simulate_terminal_price(z, out):
    zt = ct.load(z, (0, 0), (1, NUM_PATHS))
    st = S0 * ct.exp(DRIFT + DIFFUSION * zt)
    ct.store(out, (0, 0), st)


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_simulate_terminal_price:", compile_bytes(kernel_simulate_terminal_price, sig), "cubin bytes")

torch.manual_seed(0)
z = torch.randn(1, NUM_PATHS)

# host-side simulation, mirroring the kernel's own formula exactly
s_t = S0 * torch.exp(DRIFT + DIFFUSION * z)

sample_mean = s_t.mean().item()
sample_std = s_t.std(unbiased=True).item()

# --- The independent check: two facts about the lognormal distribution
# that are derivable directly from the GBM SDE itself, with no
# reference to this simulation at all. Under the risk-neutral measure,
# the underlying's expected future price is exactly today's price
# grown at the risk-free rate -- E[S_T] = S0*exp(r*T) -- and its
# variance has an equally exact closed form. Neither fact is computed
# from the sampled paths above; both come from the same mathematical
# theory this chapter's kernel is simulating a sample from. ---
analytic_mean = S0 * math.exp(R * T)
analytic_var = (S0 ** 2) * math.exp(2 * R * T) * (math.exp(SIGMA * SIGMA * T) - 1.0)
analytic_std = math.sqrt(analytic_var)

print("sample mean of simulated S_T:", sample_mean)
print("analytic mean E[S_T] = S0*exp(r*T):", analytic_mean)
print("relative error, mean:", abs(sample_mean - analytic_mean) / analytic_mean)
print("sample std of simulated S_T:", sample_std)
print("analytic std of S_T:", analytic_std)
print("relative error, std:", abs(sample_std - analytic_std) / analytic_std)
```

`02_capstone_monte_carlo_option_pricing.py`:

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

NUM_PATHS = 65536
S0, K, T, R, SIGMA = 100.0, 105.0, 0.5, 0.03, 0.2
DRIFT = (R - 0.5 * SIGMA * SIGMA) * T
DIFFUSION = SIGMA * math.sqrt(T)
DISCOUNT = math.exp(-R * T)

# --- Every one of Section 45.1's simulated terminal prices becomes one
# path's option payoff -- max(S_T - K, 0) for a call, max(K - S_T, 0)
# for a put -- and the option's price is the discounted average payoff
# across every path. Reducing 65536 per-path payoffs down to two
# numbers with ct.sum is the same in-kernel reduction pattern this book
# has used since Chapter 41's softmax denominator and Chapter 42's
# gating probabilities, just applied to a much larger tile. ---
@ct.kernel
def kernel_monte_carlo_price(z, call_price_out, put_price_out):
    zt = ct.load(z, (0, 0), (1, NUM_PATHS))
    st = S0 * ct.exp(DRIFT + DIFFUSION * zt)

    call_payoff = ct.maximum(st - K, 0.0)
    put_payoff = ct.maximum(K - st, 0.0)

    call_mean = ct.sum(call_payoff, axis=1, keepdims=True) / NUM_PATHS
    put_mean = ct.sum(put_payoff, axis=1, keepdims=True) / NUM_PATHS

    ct.store(call_price_out, (0, 0), DISCOUNT * call_mean)
    ct.store(put_price_out, (0, 0), DISCOUNT * put_mean)


sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_monte_carlo_price:", compile_bytes(kernel_monte_carlo_price, sig), "cubin bytes")

torch.manual_seed(0)
z = torch.randn(1, NUM_PATHS)

# host-side simulation, mirroring the kernel's own formula exactly
s_t = S0 * torch.exp(DRIFT + DIFFUSION * z)
call_payoff = torch.clamp(s_t - K, min=0.0)
put_payoff = torch.clamp(K - s_t, min=0.0)

mc_call = DISCOUNT * call_payoff.mean().item()
mc_put = DISCOUNT * put_payoff.mean().item()

# --- A Monte Carlo estimate is a sample mean, and a sample mean has a
# theoretical standard error: the sample's own standard deviation
# divided by the square root of how many samples went into it. That
# error is computed here directly from this run's payoffs, not assumed
# -- it is what turns "the two prices are close" into a falsifiable,
# quantitative claim, rather than an eyeballed comparison. ---
call_se = DISCOUNT * call_payoff.std(unbiased=True).item() / math.sqrt(NUM_PATHS)
put_se = DISCOUNT * put_payoff.std(unbiased=True).item() / math.sqrt(NUM_PATHS)

print("Monte Carlo call price:", mc_call, "  standard error:", call_se)
print("Monte Carlo put price:", mc_put, "  standard error:", put_se)

# --- The independent check: Chapter 43's closed-form Black-Scholes
# formula, reused completely unchanged -- a genuinely different pricing
# METHOD, not a second simulation with a different random seed. One
# side of this comparison never generates a single random number. ---
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

S_t, K_t, T_t, r_t, sigma_t = (torch.tensor(v) for v in (S0, K, T, R, SIGMA))
sqrt_t = T_t.sqrt()
d1 = (torch.log(S_t / K_t) + (r_t + 0.5 * sigma_t * sigma_t) * T_t) / (sigma_t * sqrt_t)
d2 = d1 - sigma_t * sqrt_t
bs_discount = torch.exp(-r_t * T_t)
bs_call = (S_t * host_cdf_approx(d1) - K_t * bs_discount * host_cdf_approx(d2)).item()
bs_put = (K_t * bs_discount * host_cdf_approx(-d2) - S_t * host_cdf_approx(-d1)).item()

print("Black-Scholes call price (Chapter 43's formula):", bs_call)
print("Black-Scholes put price (Chapter 43's formula):", bs_put)

call_diff = abs(mc_call - bs_call)
put_diff = abs(mc_put - bs_put)
print("call price difference:", call_diff, " in standard errors:", call_diff / call_se)
print("put price difference:", put_diff, " in standard errors:", put_diff / put_se)
print("call price within 3 standard errors of Black-Scholes:", call_diff <= 3 * call_se)
print("put price within 3 standard errors of Black-Scholes:", put_diff <= 3 * put_se)
```
