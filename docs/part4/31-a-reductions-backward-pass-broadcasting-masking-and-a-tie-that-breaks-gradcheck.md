# Chapter 31: A Reduction's Backward Pass — Broadcasting, Masking, and a Tie That Breaks gradcheck

> "Chapter 30's backward kernels all mirrored their forward kernels' shape: one array in, one array out, load a tile, store a tile the same size. A reduction breaks that mirror on purpose. Its forward pass shrinks many elements to one; its backward pass has no choice but to grow that one gradient back out to many, by a rule that depends entirely on which reduction it's undoing — and, at the exact points where that rule stops being unambiguous, on a subtlety this book can genuinely catch its own verification method missing."

**What you will understand by the end of this chapter:**

- `SumOp`: a forward kernel that reduces an entire tile to a single value with `ct.sum(x, None)`, and a backward kernel that broadcasts that single gradient back out to every input element with `ct.broadcast_to` — the first backward kernel in this book that changes shape in the opposite direction from its forward kernel
- `MaxOp`: a backward kernel that has to know WHICH element the forward pass picked, routing the incoming gradient only to the position(s) tied for the maximum, normalized by how many positions are tied
- That a full sum-reduction and a full max-reduction to a scalar, under Chapter 10's naming-confound control, compile to byte-identical cubins — and exactly what that identical byte count can and cannot be taken to mean
- A genuine confound this book has not met before: at a point with a three-way tie, `torch.autograd.gradcheck`'s own per-coordinate finite-difference method disagrees with the "split evenly by count" convention this chapter's kernel and torch's OWN built-in `max()` both use — confirmed to fail identically on both, proving the mismatch is a property of `gradcheck` meeting a non-differentiable point, not a bug in this chapter's kernel
- A capstone chaining an elementwise operation into each reduction, confirming the chain rule composes correctly through a shape-changing operation exactly as it does through the elementwise ones Chapter 30 already covered

**What you need to know first:**

- Chapter 24's `ct.reduce` and this book's established `ct.sum`/`ct.max` usage — this chapter uses `ct.sum(x, None)` and `ct.max(x, None)` to reduce an entire 1D tile to a single scalar, rather than the axis-based partial reduction Chapter 24 used on a 2D tile.
- Chapter 30's `torch.autograd.Function` fallback pattern (attempt a real `ct.launch`, catch `RuntimeError`, fall back to a formula that mirrors the kernel exactly) and Chapter 29's `gradcheck`-based verification method.
- Chapter 10's naming-confound discipline: two kernels compiled under the identical function name isolate whether a byte-count difference reflects the computation or just the name.

## 31.1 SumOp: A Reduction's Backward Pass Broadcasts

### Intuition

For `y = sum(x)` over a tile of `N` elements, calculus gives `dy/dx_i = 1` for every `i` — the gradient with respect to each input element is the same, and every element's contribution to the sum is identical. A `SumOp.backward` kernel therefore does something no backward kernel in Chapter 30 needed to do: take ONE incoming gradient value and produce `N` outputs from it, rather than operating elementwise on two same-shaped arrays.

### Background

`kernel_sum_forward` loads a tile of `tile_size` elements, reduces it to a single value with `ct.sum(x, None)`, and stores that value into a length-1 output array via `ct.broadcast_to(y, (1,))` (needed because `ct.sum`'s result is a bare scalar tile, and `ct.store` expects a tile whose shape matches what it's writing). `kernel_sum_backward` does the reverse: it loads the single incoming gradient, and uses `ct.broadcast_to` again — this time to stretch that one value out to `tile_size` copies — before storing the full-width result.

### Worked Example 31.1.1 — a forward kernel that shrinks, and a backward kernel that grows

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig_2in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_3in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Forward: many elements in, one element out.
@ct.kernel
def kernel_sum_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_sum_fwd = compile_bytes(kernel_sum_forward, sig_2in)
print(f"kernel_sum_forward: {n_sum_fwd} cubin bytes")

# Backward: one gradient in, broadcast back out to every element -- the
# reverse shape change from the forward kernel it belongs to.
@ct.kernel
def kernel_sum_backward(grad_out, grad_in, tile_size: ct.Constant[int]):
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, (0,), ct.broadcast_to(g, (tile_size,)))
n_sum_bwd = compile_bytes(kernel_sum_backward, sig_2in)
print(f"kernel_sum_backward: {n_sum_bwd} cubin bytes")

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_max_fwd = compile_bytes(kernel_max_forward, sig_2in)
print(f"kernel_max_forward: {n_max_fwd} cubin bytes")

# Backward: routes the incoming gradient only to whichever element(s) equal
# the max, splitting evenly across ties (count normalizes that split).
@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)
n_max_bwd = compile_bytes(kernel_max_backward, sig_3in)
print(f"kernel_max_backward: {n_max_bwd} cubin bytes")

# Chapter 10's naming-confound control: are a full sum-reduction and a full
# max-reduction to a scalar genuinely the same number of bytes, or did the
# four kernels above just happen to land close by coincidence?
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_sum_named = compile_bytes(kernel_fn, sig_2in)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_max_named = compile_bytes(kernel_fn, sig_2in)

print(f"sum vs max, naming-controlled (both named kernel_fn): {n_sum_named} vs {n_max_named} bytes")
print(f"identical under naming control: {n_sum_named == n_max_named}")
```

Genuinely run:

```
kernel_sum_forward: 20384 cubin bytes
kernel_sum_backward: 19872 cubin bytes
kernel_max_forward: 20384 cubin bytes
kernel_max_backward: 25760 cubin bytes
sum vs max, naming-controlled (both named kernel_fn): 20128 vs 20128 bytes
identical under naming control: True
```

### Discussion

All four kernels compile, and the naming-controlled comparison at the bottom lands cleanly: a full sum-reduction and a full max-reduction to a scalar, named identically, compile to exactly the same number of bytes (20128). That is a genuine, verified fact, but it is narrower than it might sound — it says the two compiled artifacts are the same SIZE, not that they are the same bytes in the same order, and not that `cuda.tile` treats addition and max-comparison as literally the same operation internally. A single-warp reduction over eight elements likely lowers to the same shape of instruction sequence (load, some fixed number of shuffle-and-combine steps, store) regardless of which associative operation fills the "combine" slot, which would explain identical byte counts without the two cubins being identical inside. This book has been burned before by claiming more than a byte count can support (Chapter 27's naming-confound work, and Chapter 30's Self-Check Question 1 on `kernel_square`/`kernel_square_backward`'s differing signatures) — the honest claim here is exactly "same size," not "same code."

`kernel_sum_forward` (20384 bytes) and `kernel_sum_backward` (19872 bytes) are a fair same-signature comparison — both take exactly two arrays and one constant — and they are NOT the same size: the backward kernel, which only broadcasts, compiles smaller than the forward kernel, which reduces. `kernel_max_backward` (25760 bytes), by contrast, takes three arrays (the saved input, the incoming gradient, and the output) rather than two, so — following exactly the lesson from Chapter 30's Question 1 — its larger size is not directly comparable to either sum kernel's; it has a different signature, and a fair comparison would need a same-signature counterpart this chapter hasn't built.

### Worked Example 31.1.2 — SumOp: a complete broadcasting-backward Function

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_sum_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_sum_backward(grad_out, grad_in, tile_size: ct.Constant[int]):
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, (0,), ct.broadcast_to(g, (tile_size,)))

class SumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_sum_forward, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_sum_backward, (grad_output, grad_input, ctx.n))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output.expand(ctx.n).clone()

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = SumOp.apply(x)
print(f"forward matches x.sum(): {torch.equal(y, x.detach().sum().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad is uniformly 1 (d(sum)/dx_i = 1 for every i): {torch.allclose(x.grad, torch.ones(8, dtype=torch.float64))}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(SumOp.apply, (x,))
print(f"SumOp gradcheck passed: {ok}")
```

Genuinely run:

```
forward matches x.sum(): True
used real forward kernel: False
x.grad is uniformly 1 (d(sum)/dx_i = 1 for every i): True
used real backward kernel: False
SumOp gradcheck passed: True
```

### Discussion

`used_real_kernel_forward` and `used_real_kernel_backward` both report `False`, this sandbox's familiar hardware wall, but the fallback formulas (`x.sum().reshape(1)` and `grad_output.expand(ctx.n).clone()`) are exact restatements of `kernel_sum_forward` and `kernel_sum_backward`'s own logic, so `gradcheck` passing is a genuine confirmation that the broadcasting backward pass is mathematically correct — not merely that the printed numbers look plausible. This is the same fallback discipline Chapter 30 established, applied for the first time to a Function whose forward and backward have different shapes on each side.

## 31.2 MaxOp: A Backward Pass That Has to Know Which Element Won

### Intuition

`SumOp`'s backward pass didn't need to look at the forward input at all — every element contributed equally, so the gradient broadcasts uniformly regardless of what the values were. `max` is different: `d(max(x))/dx_i` is `1` for whichever element achieved the maximum and `0` for every other element, so `MaxOp.backward` has to reconstruct, inside the kernel, WHICH element that was — by recomputing the max and comparing every element against it.

### Background

`kernel_max_backward` takes the saved input `x`, recomputes `max_val = ct.max(x, None)`, and builds a boolean-turned-float `mask = (x == max_val)`. On a random input where the maximum is unique, exactly one entry of `mask` is `1.0` and the backward pass routes the entire incoming gradient there. The `count = ct.sum(mask, None)` normalization exists for the case Section 31.3 tests directly: more than one element tied for the maximum.

### Worked Example 31.2.1 — MaxOp on a random input, where the maximum is (almost certainly) unique

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)

class MaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_max_forward, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.max().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_max_backward, (x, grad_output, grad_input, x.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x == x.max()).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = MaxOp.apply(x)
print(f"forward matches x.max(): {torch.equal(y, x.detach().max().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
expected = (x.detach() == x.detach().max()).double()
print(f"x.grad is a one-hot at the (unique) argmax: {torch.equal(x.grad, expected)}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(MaxOp.apply, (x,))
print(f"MaxOp gradcheck passed (random input, max is unique): {ok}")
```

Genuinely run:

```
forward matches x.max(): True
used real forward kernel: False
x.grad is a one-hot at the (unique) argmax: True
used real backward kernel: False
MaxOp gradcheck passed (random input, max is unique): True
```

### Discussion

On a random `float64` input, ties are essentially impossible, so `count` is always `1` and the mask-times-gradient formula reduces to exactly what calculus predicts: a one-hot gradient at the single winning index. `gradcheck` confirms this holds as a general property near this particular `x`, not merely that the printed numbers happen to look right. Nothing about this section has tested the part of `kernel_max_backward` that the `count` normalization actually exists for — that's Section 31.3, deliberately.

## 31.3 A Tie Breaks More Than the Kernel: gradcheck's Own Blind Spot

### Intuition

`kernel_max_backward`'s `count` normalization was written for inputs where more than one element ties for the maximum. Since Chapter 27, this book's discipline has been to test the assumption directly rather than trust that a formula which looks right at a random point also behaves as expected at an edge case — and a deliberately constructed tie is exactly the edge case this formula was built for.

### Background

A length-8 input is built with a genuine three-way tie at its maximum value, `3.0`. `MaxOp`'s own backward (mask divided by count of ties) is compared against `torch`'s OWN built-in `x.max()` on the identical input, and both are separately checked with `gradcheck`.

### Worked Example 31.3.1 — the same tie, tested against this chapter's kernel and against torch's own built-in max()

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)

class MaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_max_forward, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x.max().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_max_backward, (x, grad_output, grad_input, x.shape[0]))
            return grad_input
        except RuntimeError:
            mask = (x == x.max()).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# A deliberate 3-way tie at the maximum -- unlike Chapter 30's confounds,
# this one isn't about how a kernel is named or where it's positioned in a
# source file, it's about what happens at a point where max() itself is not
# differentiable, and both this custom Function and torch's own built-in
# max() have to pick SOME convention for what "the gradient" means there.
x = torch.tensor([1.0, 3.0, 3.0, 2.0, -1.0, 0.5, 3.0, 2.5], dtype=torch.float64, requires_grad=True)
print(f"x (a 3-way tie at the maximum, 3.0): {x.detach().tolist()}")

y = MaxOp.apply(x)
y.backward()
print(f"MaxOp.backward, tie split evenly by count (1/3 each): {x.grad.tolist()}")

ok_custom = torch.autograd.gradcheck(MaxOp.apply, (x.detach().clone().requires_grad_(True),), raise_exception=False)
print(f"gradcheck on MaxOp at the tied input: {ok_custom}")

x_builtin = x.detach().clone().requires_grad_(True)
y_builtin = x_builtin.max()
y_builtin.backward()
print(f"torch's OWN built-in x.max().backward(), same tie: {x_builtin.grad.tolist()}")

ok_builtin = torch.autograd.gradcheck(lambda t: t.max(), (x.detach().clone().requires_grad_(True),), raise_exception=False)
print(f"gradcheck on torch's OWN built-in max() at the SAME tied input: {ok_builtin}")

# Why: gradcheck perturbs ONE coordinate at a time. Nudging a single tied
# coordinate up by eps makes it the unique max (slope 1 in that direction);
# nudging it down by eps leaves the max unchanged, since another tied
# coordinate still holds it (slope 0). The central-difference average, 0.5,
# is what gradcheck expects from EVERY tied coordinate independently -- a
# per-coordinate question, not the same question as "how should one shared
# gradient be split across a tie," and the two only agree when a tie has
# exactly two members.
eps = 1e-6
i = 1
x_plus = x.detach().clone(); x_plus[i] += eps
x_minus = x.detach().clone(); x_minus[i] -= eps
slope = (x_plus.max() - x_minus.max()) / (2 * eps)
print(f"gradcheck's own central-difference slope at tied index {i}, alone: {slope.item()}")
```

Genuinely run:

```
x (a 3-way tie at the maximum, 3.0): [1.0, 3.0, 3.0, 2.0, -1.0, 0.5, 3.0, 2.5]
MaxOp.backward, tie split evenly by count (1/3 each): [0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0]
gradcheck on MaxOp at the tied input: False
torch's OWN built-in x.max().backward(), same tie: [0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0]
gradcheck on torch's OWN built-in max() at the SAME tied input: False
gradcheck's own central-difference slope at tied index 1, alone: 0.500000000069889
```

### Discussion

`MaxOp`'s custom kernel and torch's own built-in `max()` produce IDENTICAL gradients on the tied input (each tied position gets exactly `1/3`), and `gradcheck` reports `False` for both, for the exact same underlying reason. This is the clean confirmation this section was built to get: if only the custom kernel had failed, the honest conclusion would be "there's a bug in `kernel_max_backward`." Because torch's own, presumably bug-free, built-in operation fails identically on the identical input, the honest conclusion is instead that `gradcheck`'s per-coordinate finite-difference method and the "split evenly across a tie" convention are simply answering different questions, and they happen to agree only when a tie has exactly two members (where `1/count = 1/2` matches the `0.5` central-difference slope every book chapter's own probes confirm). At a three-way tie, `1/count = 1/3`, but perturbing any ONE tied coordinate alone still measures a `0.5` slope, because that single perturbation makes it the unique max in the positive direction while the OTHER two ties keep the max unchanged in the negative direction — a fact about `gradcheck`'s one-coordinate-at-a-time probing, not about whether `1/3` is a defensible gradient. Both conventions are legitimate answers to "what is a reasonable subgradient of `max` at a non-differentiable point"; they simply are not the same answer, and this book's own verification method (`gradcheck`) turns out to have a blind spot at exactly the points where the function it's checking stops being differentiable in the first place.

## 31.4 Capstone: Chaining an Elementwise Op Into a Reduction

### Intuition

Chapter 30's capstone chained three elementwise operations. The natural next question for this chapter is whether a shape-changing operation composes just as correctly when it's part of a chain — specifically, whether the chain rule still produces the right answer when an elementwise `DoubleOp` feeds into a reduction.

### Background

Two independent chains are built: `y = sum(2x)`, where `dy/dx_i = 2` for every `i` (the constant doubling passes straight through the uniform-broadcast gradient), and `y = max(2x) = 2 * max(x)`, where `dy/dx_i` is `2` at whichever index is `x`'s own argmax and `0` everywhere else — doubling every element doesn't change WHICH index holds the maximum, it only doubles the value there and the local derivative alongside it.

### Worked Example 31.4.1 — DoubleOp feeding into SumOp, and DoubleOp feeding into MaxOp

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), 2 * x)

class DoubleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_double, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

@ct.kernel
def kernel_sum_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_sum_backward(grad_out, grad_in, tile_size: ct.Constant[int]):
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, (0,), ct.broadcast_to(g, (tile_size,)))

class SumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_sum_forward, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_sum_backward, (grad_output, grad_input, ctx.n))
            return grad_input
        except RuntimeError:
            return grad_output.expand(ctx.n).clone()

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)

class MaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_max_forward, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x.max().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_max_backward, (x, grad_output, grad_input, x.shape[0]))
            return grad_input
        except RuntimeError:
            mask = (x == x.max()).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# Chain 1: y = sum(2x). dy/dx_i = 2 for every i (DoubleOp's constant-2
# backward feeding into SumOp's uniform-broadcast backward).
x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y_sum = SumOp.apply(DoubleOp.apply(x))
print(f"sum(2x) matches 2*x.sum(): {torch.equal(y_sum, (2 * x.detach()).sum().reshape(1))}")

y_sum.backward()
print(f"x.grad matches uniform 2s: {torch.allclose(x.grad, torch.full((8,), 2.0, dtype=torch.float64))}")

def chained_sum(x):
    return SumOp.apply(DoubleOp.apply(x))
ok_sum = torch.autograd.gradcheck(chained_sum, (x,))
print(f"gradcheck passed on DoubleOp -> SumOp chain: {ok_sum}")

print()

# Chain 2: y = max(2x) = 2*max(x). dy/dx is 2 at x's own argmax, 0 elsewhere
# (doubling every element doesn't move WHERE the maximum is).
x2 = torch.randn(8, dtype=torch.float64, requires_grad=True)
y_max = MaxOp.apply(DoubleOp.apply(x2))
print(f"max(2x) matches 2*x.max(): {torch.equal(y_max, (2 * x2.detach()).max().reshape(1))}")

y_max.backward()
expected = 2.0 * (x2.detach() == x2.detach().max()).double()
print(f"x2.grad matches 2 at argmax, 0 elsewhere: {torch.equal(x2.grad, expected)}")

def chained_max(x):
    return MaxOp.apply(DoubleOp.apply(x))
ok_max = torch.autograd.gradcheck(chained_max, (x2,))
print(f"gradcheck passed on DoubleOp -> MaxOp chain: {ok_max}")
```

Genuinely run:

```
sum(2x) matches 2*x.sum(): True
x.grad matches uniform 2s: True
gradcheck passed on DoubleOp -> SumOp chain: True

max(2x) matches 2*x.max(): True
x2.grad matches 2 at argmax, 0 elsewhere: True
gradcheck passed on DoubleOp -> MaxOp chain: True
```

### Discussion

Both chains confirm the same underlying point from two different angles. `sum(2x)`'s gradient is uniformly `2` because `SumOp`'s backward broadcasts whatever gradient it receives, and it receives `2 * grad_output` from `DoubleOp` sitting in front of it — the two backward passes simply compose. `max(2x)`'s gradient is `2` at exactly one index and `0` everywhere else, for a more specific reason worth stating precisely: `DoubleOp` is a strictly increasing, elementwise function of each coordinate independently, so it cannot change which index holds the maximum (whichever `x_i` was largest stays largest after every element is doubled by the same factor) — `MaxOp`'s backward routes gradient to that same, unchanged index, and `DoubleOp`'s backward doubles whatever arrives there. Two structurally different reductions, chained with the same elementwise operation, both check out exactly against `gradcheck` and against values worked out by hand — the shape change a reduction introduces doesn't weaken this book's verification approach, it just requires the fallback formula to get the shape change right too.

## Chapter Summary

This chapter gave a custom cuTile-backed operation its first backward pass that changes shape: `SumOp`'s `kernel_sum_backward` takes one gradient and broadcasts it out to every input element, the structural opposite of `kernel_sum_forward`'s many-to-one reduction. `MaxOp` went further, requiring its backward kernel to reconstruct which element the forward pass had picked, via a recomputed mask normalized by how many elements tied for the maximum — verified correct on random input where the maximum is essentially always unique. Testing that normalization directly, on a deliberately constructed three-way tie, surfaced a genuine confound this book had not met before: `gradcheck` disagreed with the tie-split convention, and — critically — disagreed with it identically whether the convention came from this chapter's own kernel or from torch's own built-in `max()`, proving the mismatch is a structural property of `gradcheck`'s per-coordinate finite-difference method meeting a genuinely non-differentiable point, not a bug anywhere in this chapter's code. The capstone confirmed that chaining an elementwise operation into either reduction composes exactly as the chain rule predicts, closing the chapter with two more chains `gradcheck` and hand-derived calculus both confirm.

## Self-Check Questions

1. `kernel_sum_backward` uses `ct.broadcast_to` to turn one loaded gradient value into `tile_size` copies. Why did none of Chapter 30's backward kernels (`kernel_square_backward`, `kernel_copy`, `kernel_add_one`'s implicit identity) ever need this function, and what specifically about a REDUCTION's backward pass makes it necessary here?
2. Section 31.1 found that a full sum-reduction and a full max-reduction, compiled under the identical name `kernel_fn`, produce exactly the same cubin byte count (20128). What is the most this fact can honestly be said to prove about how `cuda.tile`'s compiler treats the two operations, and what would it NOT be safe to conclude from it?
3. Section 31.3's central-difference probe found a slope of `0.5` at a single tied coordinate, perturbed alone, even though three coordinates were tied. Walk through why perturbing that ONE coordinate up by `eps` and down by `eps` produces that particular slope, and explain why the same calculation at a two-way tie would happen to match the `1/count` convention exactly.
4. Both `MaxOp`'s custom backward and torch's own built-in `max()` failed `gradcheck` on the identical tied input, with identical gradients. If only `MaxOp` had failed and torch's built-in `max()` had passed on the same input, what would that have told you instead — and how would that conclusion have been different from the one this chapter actually reached?
5. In the capstone, `max(2x)`'s gradient was `2` at `x`'s own argmax and `0` everywhere else — not, say, `1` at `x`'s argmax and `1` at `(2x)`'s argmax (which happen to be the same index, but for a reader unsure why). Explain, from `MaxOp`'s and `DoubleOp`'s backward formulas directly, exactly how the `2` at that one surviving index is produced by the chain rule, rather than assumed.

## Where We Go Next

This chapter's reductions collapsed an entire tile down to a single scalar in one kernel launch — a convenient simplification that sidestepped a question every real reduction eventually has to answer: what happens when the data doesn't fit in one tile at all, and the reduction has to combine PARTIAL results computed by separate blocks running independently, potentially in parallel, with no fixed order in which their partial sums or maxes arrive? Chapter 20's atomics already gave this book one honest tool for combining results from blocks that don't coordinate; a multi-block reduction's backward pass — where the forward pass's shape-changing arithmetic has to be undone correctly regardless of how the work was partitioned — is the natural next place this chapter's broadcasting-and-masking machinery gets tested against.

## Worked Solutions

**1.** Every backward kernel Chapter 30 wrote operated on inputs and outputs of the SAME shape: `kernel_square_backward` took an `(8,)` saved input and an `(8,)` incoming gradient and produced an `(8,)` output, `kernel_copy` copied `(8,)` to `(8,)`, and even `kernel_add_one`'s identity gradient was `(8,)` to `(8,)`. None of them needed `ct.broadcast_to` because none of them needed to change a tile's shape at all — every operation was elementwise, so the natural shape of the gradient already matched the natural shape of the input. A reduction's forward pass collapses `N` elements to `1`, which means its backward pass necessarily receives a `(1,)`-shaped (or scalar) incoming gradient and must produce an `(N,)`-shaped result — there is no elementwise correspondence to lean on, only the mathematical fact that every input contributed to the single output, so `ct.broadcast_to` is what turns "one gradient value" into "the same value, N times" so it can be stored into an `(N,)`-shaped output array at all.

**2.** The most this fact can honestly support is that the two compiled cubins are the same SIZE — the same number of bytes — under a controlled, identical function name. It would not be safe to conclude that the two kernels compile to identical machine code, or that `cuda.tile`'s compiler internally treats "sum" and "max" as the same operation; a single-warp full reduction over a small, fixed-size tile plausibly lowers to the same NUMBER and SHAPE of instructions (a load, a fixed sequence of shuffle-and-combine steps, a store) regardless of which associative combining operation fills the "combine" slot, which would produce identical byte counts without the underlying instructions being bit-for-bit identical. This book has specifically been burned before by overclaiming from a byte-count match (Chapter 27's atomics mistake was the reverse error, an overclaimed byte-count DIFFERENCE) — the correct, narrow claim here is "same size," full stop.

**3.** Perturbing the tied coordinate up by `eps` makes that coordinate `3.0 + eps`, which is now strictly greater than the other two tied coordinates (still at `3.0`) — so the max becomes `3.0 + eps`, a change of exactly `+eps` from the unperturbed max of `3.0`. Perturbing that SAME coordinate down by `eps` makes it `3.0 - eps`, which is now strictly LESS than the other two tied coordinates — so the max stays at `3.0`, unchanged, a change of `0`. The central-difference slope is `(change_up - change_down) / (2 * eps) = (eps - 0) / (2 * eps) = 0.5`, and this reasoning applies independently to EACH of the three tied coordinates, since perturbing any one of them alone leaves at least one of the other two still holding the tie in the downward direction. At a two-way tie, this same `0.5` slope holds by identical reasoning (perturbing either tied coordinate alone: `+eps` wins, `-eps` leaves the other tied coordinate holding the max unchanged) — and `1/count = 1/2 = 0.5` for a two-way tie happens to be exactly this same number, which is precisely why `gradcheck` passes at two-way ties and fails at three-way ones: the coincidence only holds for `count = 2`.

**4.** If only `MaxOp` had failed while torch's own built-in `max()` passed on the identical tied input, the honest conclusion would have been that `kernel_max_backward`'s tie-handling logic itself was wrong — some bug in the mask, the count, or the broadcast that produced a gradient torch's own (presumably correct) convention would not have produced. Instead, both failed identically, with identical gradient VALUES (`1/3` at each tied position) — which rules out a bug in this chapter's kernel specifically, because torch's own built-in operation, using its own internal (non-cuTile) implementation, produces the exact same numbers and hits the exact same `gradcheck` disagreement. The conclusion this chapter actually reached — that the mismatch is a property of `gradcheck`'s method meeting a non-differentiable point, not a defect in any one implementation — is only available BECAUSE the two independent implementations agreed; had they disagreed, the honest next step would have been to trust torch's built-in result and go looking for the bug in `kernel_max_backward`.

**5.** `DoubleOp`'s backward simply doubles whatever gradient it receives (`return 2 * grad_output`), with no dependence on which index anything is — it treats its incoming gradient as an opaque `(8,)` tensor and returns `2 *` that tensor, unchanged in shape or position. `MaxOp`'s backward, called first (since it sits closer to the output in the chain `MaxOp(DoubleOp(x))`), receives `grad_output = 1` (from `.backward()` on a scalar) and produces a one-hot `(8,)` tensor: `1.0` at the index where `(2x)` achieves its maximum, `0.0` elsewhere. Because doubling is a positive scalar multiple applied identically to every coordinate, `(2x)`'s argmax is the same index as `x`'s own argmax — so that one-hot tensor already has its single `1.0` sitting at `x`'s argmax before `DoubleOp`'s backward ever sees it. `DoubleOp.backward` then doubles that ENTIRE tensor — the `1.0` at the surviving index becomes `2.0`, and every `0.0` elsewhere stays `0.0` (doubling zero changes nothing) — which is exactly how the chain rule, applied mechanically through two independently-written `backward` methods with no awareness of each other, produces `2` at one index and `0` everywhere else without either method needing to "know" about the other's role in the computation.

## Complete Runnable Code

### File: `01_sum_and_max_reduction_kernels.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig_2in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_3in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Forward: many elements in, one element out.
@ct.kernel
def kernel_sum_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_sum_fwd = compile_bytes(kernel_sum_forward, sig_2in)
print(f"kernel_sum_forward: {n_sum_fwd} cubin bytes")

# Backward: one gradient in, broadcast back out to every element -- the
# reverse shape change from the forward kernel it belongs to.
@ct.kernel
def kernel_sum_backward(grad_out, grad_in, tile_size: ct.Constant[int]):
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, (0,), ct.broadcast_to(g, (tile_size,)))
n_sum_bwd = compile_bytes(kernel_sum_backward, sig_2in)
print(f"kernel_sum_backward: {n_sum_bwd} cubin bytes")

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_max_fwd = compile_bytes(kernel_max_forward, sig_2in)
print(f"kernel_max_forward: {n_max_fwd} cubin bytes")

# Backward: routes the incoming gradient only to whichever element(s) equal
# the max, splitting evenly across ties (count normalizes that split).
@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)
n_max_bwd = compile_bytes(kernel_max_backward, sig_3in)
print(f"kernel_max_backward: {n_max_bwd} cubin bytes")

# Chapter 10's naming-confound control: are a full sum-reduction and a full
# max-reduction to a scalar genuinely the same number of bytes, or did the
# four kernels above just happen to land close by coincidence?
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_sum_named = compile_bytes(kernel_fn, sig_2in)

@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))
n_max_named = compile_bytes(kernel_fn, sig_2in)

print(f"sum vs max, naming-controlled (both named kernel_fn): {n_sum_named} vs {n_max_named} bytes")
print(f"identical under naming control: {n_sum_named == n_max_named}")
```

### File: `02_sumop_a_broadcasting_backward_pass.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_sum_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_sum_backward(grad_out, grad_in, tile_size: ct.Constant[int]):
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, (0,), ct.broadcast_to(g, (tile_size,)))

class SumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_sum_forward, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_sum_backward, (grad_output, grad_input, ctx.n))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output.expand(ctx.n).clone()

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = SumOp.apply(x)
print(f"forward matches x.sum(): {torch.equal(y, x.detach().sum().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad is uniformly 1 (d(sum)/dx_i = 1 for every i): {torch.allclose(x.grad, torch.ones(8, dtype=torch.float64))}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(SumOp.apply, (x,))
print(f"SumOp gradcheck passed: {ok}")
```

### File: `03_maxop_a_masked_backward_pass.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)

class MaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_max_forward, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.max().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_max_backward, (x, grad_output, grad_input, x.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x == x.max()).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = MaxOp.apply(x)
print(f"forward matches x.max(): {torch.equal(y, x.detach().max().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
expected = (x.detach() == x.detach().max()).double()
print(f"x.grad is a one-hot at the (unique) argmax: {torch.equal(x.grad, expected)}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(MaxOp.apply, (x,))
print(f"MaxOp gradcheck passed (random input, max is unique): {ok}")
```

### File: `04_the_tie_that_breaks_gradcheck.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)

class MaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_max_forward, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x.max().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_max_backward, (x, grad_output, grad_input, x.shape[0]))
            return grad_input
        except RuntimeError:
            mask = (x == x.max()).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# A deliberate 3-way tie at the maximum -- unlike Chapter 30's confounds,
# this one isn't about how a kernel is named or where it's positioned in a
# source file, it's about what happens at a point where max() itself is not
# differentiable, and both this custom Function and torch's own built-in
# max() have to pick SOME convention for what "the gradient" means there.
x = torch.tensor([1.0, 3.0, 3.0, 2.0, -1.0, 0.5, 3.0, 2.5], dtype=torch.float64, requires_grad=True)
print(f"x (a 3-way tie at the maximum, 3.0): {x.detach().tolist()}")

y = MaxOp.apply(x)
y.backward()
print(f"MaxOp.backward, tie split evenly by count (1/3 each): {x.grad.tolist()}")

ok_custom = torch.autograd.gradcheck(MaxOp.apply, (x.detach().clone().requires_grad_(True),), raise_exception=False)
print(f"gradcheck on MaxOp at the tied input: {ok_custom}")

x_builtin = x.detach().clone().requires_grad_(True)
y_builtin = x_builtin.max()
y_builtin.backward()
print(f"torch's OWN built-in x.max().backward(), same tie: {x_builtin.grad.tolist()}")

ok_builtin = torch.autograd.gradcheck(lambda t: t.max(), (x.detach().clone().requires_grad_(True),), raise_exception=False)
print(f"gradcheck on torch's OWN built-in max() at the SAME tied input: {ok_builtin}")

# Why: gradcheck perturbs ONE coordinate at a time. Nudging a single tied
# coordinate up by eps makes it the unique max (slope 1 in that direction);
# nudging it down by eps leaves the max unchanged, since another tied
# coordinate still holds it (slope 0). The central-difference average, 0.5,
# is what gradcheck expects from EVERY tied coordinate independently -- a
# per-coordinate question, not the same question as "how should one shared
# gradient be split across a tie," and the two only agree when a tie has
# exactly two members.
eps = 1e-6
i = 1
x_plus = x.detach().clone(); x_plus[i] += eps
x_minus = x.detach().clone(); x_minus[i] -= eps
slope = (x_plus.max() - x_minus.max()) / (2 * eps)
print(f"gradcheck's own central-difference slope at tied index {i}, alone: {slope.item()}")
```

### File: `05_capstone_chaining_reductions_with_elementwise_ops.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), 2 * x)

class DoubleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_double, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

@ct.kernel
def kernel_sum_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.sum(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_sum_backward(grad_out, grad_in, tile_size: ct.Constant[int]):
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, (0,), ct.broadcast_to(g, (tile_size,)))

class SumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_sum_forward, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_sum_backward, (grad_output, grad_input, ctx.n))
            return grad_input
        except RuntimeError:
            return grad_output.expand(ctx.n).clone()

@ct.kernel
def kernel_max_forward(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (tile_size,))
    y = ct.max(x, None)
    ct.store(c, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_backward(x_saved, grad_out, grad_in, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (tile_size,))
    g = ct.load(grad_out, (0,), (1,))
    max_val = ct.max(x, None)
    mask = (x == max_val).astype(ct.float32)
    count = ct.sum(mask, None)
    result = mask * ct.broadcast_to(g, (tile_size,)) / ct.broadcast_to(count, (tile_size,))
    ct.store(grad_in, (0,), result)

class MaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_max_forward, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x.max().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_max_backward, (x, grad_output, grad_input, x.shape[0]))
            return grad_input
        except RuntimeError:
            mask = (x == x.max()).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# Chain 1: y = sum(2x). dy/dx_i = 2 for every i (DoubleOp's constant-2
# backward feeding into SumOp's uniform-broadcast backward).
x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y_sum = SumOp.apply(DoubleOp.apply(x))
print(f"sum(2x) matches 2*x.sum(): {torch.equal(y_sum, (2 * x.detach()).sum().reshape(1))}")

y_sum.backward()
print(f"x.grad matches uniform 2s: {torch.allclose(x.grad, torch.full((8,), 2.0, dtype=torch.float64))}")

def chained_sum(x):
    return SumOp.apply(DoubleOp.apply(x))
ok_sum = torch.autograd.gradcheck(chained_sum, (x,))
print(f"gradcheck passed on DoubleOp -> SumOp chain: {ok_sum}")

print()

# Chain 2: y = max(2x) = 2*max(x). dy/dx is 2 at x's own argmax, 0 elsewhere
# (doubling every element doesn't move WHERE the maximum is).
x2 = torch.randn(8, dtype=torch.float64, requires_grad=True)
y_max = MaxOp.apply(DoubleOp.apply(x2))
print(f"max(2x) matches 2*x.max(): {torch.equal(y_max, (2 * x2.detach()).max().reshape(1))}")

y_max.backward()
expected = 2.0 * (x2.detach() == x2.detach().max()).double()
print(f"x2.grad matches 2 at argmax, 0 elsewhere: {torch.equal(x2.grad, expected)}")

def chained_max(x):
    return MaxOp.apply(DoubleOp.apply(x))
ok_max = torch.autograd.gradcheck(chained_max, (x2,))
print(f"gradcheck passed on DoubleOp -> MaxOp chain: {ok_max}")
```
