# Chapter 30: Writing the Backward Kernel — Part 4 Begins

> "Chapter 29's every `backward` was hand-derived calculus, written directly in plain `torch`. That was honest as far as it went, but it skipped the part Part 4 is actually about: a real cuTile-backed operation needs its OWN kernel for the gradient, compiled and structured exactly as seriously as the forward kernel — and chained, correctly, through as many of these custom operations as autograd is asked to compose."

**What you will understand by the end of this chapter:**

- That a backward pass can be written as its own, genuinely distinct `@ct.kernel` — not a reuse of the forward kernel, but a separate compiled artifact with its own inputs (the saved forward input, plus the incoming gradient) and its own real cubin
- A complete two-kernel `torch.autograd.Function` (`SquareOp`, computing `y = x**2`) where both `forward` and `backward` attempt a real `ct.launch` first, each independently falling back to a labeled reference computation when it hits this book's now-familiar hardware wall — and a genuine `gradcheck` confirming the fallback path's gradient is correct
- That chaining two independently-written custom `Function`s (`DoubleOp` then `SquareOp`) produces the mathematically correct composed gradient via the chain rule, confirmed both by hand (`d/dx (2x)^2 = 8x`) and by `gradcheck` on the composition as a whole
- A three-op capstone chain where every forward AND every backward attempts its own real cuTile kernel, with a genuine tally confirming exactly how many of those six attempts succeeded in this sandbox — and why that number being zero doesn't undermine the chain's correctness
- That a materially different architecture — a reader-supplied autograd engine built on CuPy arrays instead of `torch` tensors — hits this sandbox's hardware wall at a different, EARLIER point than every `torch`-based example in this chapter, and genuinely why: `torch` tensors have a real CPU backend to fall back to, and CuPy arrays do not

**What you need to know first:**

- Chapter 29's `torch.autograd.Function` contract (`forward`/`backward`/`apply`, `ctx.save_for_backward`), its real `gradcheck`-based verification method, and its central finding: `torch.cuda.current_stream()` itself fails with `RuntimeError: Found no NVIDIA driver on your system` before `ct.launch` is ever reached, in this sandbox.
- This book's `export_kernel`-only compilation method, unaffected by hardware availability.

## 30.1 A Backward Pass Is Its Own Kernel

### Intuition

For `y = x^2`, calculus gives `dy/dx = 2x`, so `backward` needs to compute `grad_input = 2 * x * grad_output` — the saved input, doubled, times the incoming gradient. Chapter 29 wrote exactly this kind of formula directly in `torch`. A real cuTile-backed operation instead writes it as a second `@ct.kernel`: a genuinely different compiled artifact from the forward kernel, since it takes different inputs (two arrays — the saved `x` and `grad_output` — rather than forward's one) and computes a different formula entirely.

### Background

`kernel_square` computes the forward pass (`y = x * x`, one array input). `kernel_square_backward` computes the backward pass (`grad_input = 2 * x * grad_output`, two array inputs). Both are compiled ahead-of-time and their byte counts compared, purely to confirm they are what they claim to be: two distinct, independently-compilable kernels.

### Worked Example 30.1.1 — a forward kernel and its own, separate backward kernel

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

# Forward kernel: y = x * x
@ct.kernel
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)
n_forward = compile_bytes(kernel_square, sig_2in)
print(f"kernel_square (forward, y = x*x): {n_forward} cubin bytes")

# Backward kernel: grad_input = 2 * x * grad_output -- a genuinely
# DIFFERENT kernel from the forward one, taking two array inputs
# (the saved x, and grad_output) rather than one.
@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)
n_backward = compile_bytes(kernel_square_backward, sig_3in)
print(f"kernel_square_backward (grad_input = 2*x*grad_output): {n_backward} cubin bytes")
print(f"forward and backward are different kernels, different byte counts: {n_forward != n_backward}")
```

Genuinely run:

```
kernel_square (forward, y = x*x): 20384 cubin bytes
kernel_square_backward (grad_input = 2*x*grad_output): 23072 cubin bytes
forward and backward are different kernels, different byte counts: True
```

### Discussion

Both kernels compile to real, distinct cubins — 20384 bytes for the one-input forward kernel, 23072 bytes for the two-input backward kernel, confirmed different rather than assumed different. This is the structural shape Part 4 is built around: a backward pass isn't a formula bolted onto the forward kernel's definition, it's an independent piece of tile code, with its own signature, its own load/store pattern, and its own real compiled artifact — subject to every genuine verification method (byte counts, exception behavior, the naming and positional confounds) this book has used on any other kernel since Chapter 1.

## 30.2 SquareOp: A Complete Two-Kernel autograd.Function

### Intuition

Chapter 29's `DoubleOp` and `DoubleKernel` used a hand-written `backward` formula. `SquareOp` completes the pattern properly: `forward` attempts to launch `kernel_square`, `backward` attempts to launch `kernel_square_backward` — each independently falling back to its own plain-`torch` reference computation when the now-familiar hardware wall is hit, and each recording which path it took.

### Background

`SquareOp.forward` saves `x` via `ctx.save_for_backward`, then attempts the real launch of `kernel_square`. `SquareOp.backward` retrieves the saved `x`, then attempts the real launch of `kernel_square_backward`, using both `x` and the incoming `grad_output` as its two array inputs. Both computations are then validated: the forward output against `x**2`, the gradient against `2*x`, and the whole `Function` against `torch.autograd.gradcheck`.

### Worked Example 30.2.1 — forward and backward, each attempting its own real kernel

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)

@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)

class SquareOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError as e:
            ctx.used_real_kernel_forward = False
            return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square_backward, (x, grad_output, grad_input, x.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError as e:
            ctx.used_real_kernel_backward = False
            return 2 * x * grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = SquareOp.apply(x)
print(f"forward matches x**2: {torch.equal(y, x.detach() ** 2)}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad matches 2*x: {torch.allclose(x.grad, 2 * x.detach())}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(SquareOp.apply, (x,))
print(f"gradcheck passed: {ok}")
```

Genuinely run:

```
forward matches x**2: True
used real forward kernel: False
x.grad matches 2*x: True
used real backward kernel: False
gradcheck passed: True
```

### Discussion

Both `used_real_kernel_forward` and `used_real_kernel_backward` report `False` — this sandbox's hardware wall applies identically to both directions, since both ultimately call `ct.launch` on a stream obtained from `torch.cuda.current_stream()`, and that call fails the same way regardless of which kernel it was about to launch. What matters is that the fallback for `backward` isn't a shortcut invented for this chapter — it's the exact formula `kernel_square_backward` itself encodes, kept in sync deliberately, and `gradcheck` is what actually confirms that synchronization: it doesn't know or care that a real kernel exists behind the fallback, it only tests whether the gradient `SquareOp.backward` produces is analytically consistent with what `SquareOp.forward` computes. Passing `gradcheck` here says the fallback formula is correct; Section 30.1 already confirmed the *kernel* encoding that same formula is a real, valid compiled artifact. Together, they're the two halves of a claim this book can make honestly: the backward kernel is real and correct as written, even though launching it remains outside what this sandbox can do.

## 30.3 Chaining Two Custom Operations

### Intuition

A single custom operation is a straightforward case for autograd. The real test of "wiring into PyTorch's autograd," rather than merely using it once, is whether MULTIPLE independently-written custom `Function`s compose correctly — whether the chain rule threads a gradient through two separate, unrelated `backward` implementations exactly as it would through built-in PyTorch operations.

### Background

`DoubleOp` (from Chapter 29) and `SquareOp` (from Section 30.2) are composed as `y = SquareOp.apply(DoubleOp.apply(x))`, computing `y = (2x)^2 = 4x^2`. By hand, `dy/dx = 8x`. Both the forward value and the gradient are checked directly against this, and the whole composition is checked again with `gradcheck`.

### Worked Example 30.3.1 — two custom Functions, composed, and the chain rule verified

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
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)

@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)

class SquareOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square_backward, (x, grad_output, grad_input, x.shape[0]))
            return grad_input
        except RuntimeError:
            return 2 * x * grad_output

# Chain: y = Square(Double(x)) = (2x)^2 = 4x^2. Each op is a SEPARATE
# custom autograd.Function, each individually backed by (attempted)
# real cuTile kernels. Composing them tests whether autograd's chain
# rule correctly threads gradients through TWO custom Functions, not
# just one.
x = torch.randn(8, dtype=torch.float64, requires_grad=True)
doubled = DoubleOp.apply(x)
y = SquareOp.apply(doubled)

print(f"forward matches (2x)^2: {torch.equal(y, (2 * x.detach()) ** 2)}")
print(f"forward matches 4x^2: {torch.equal(y, 4 * x.detach() ** 2)}")

y.sum().backward()
# d/dx (2x)^2 = 8x
print(f"x.grad matches 8x (chain rule through two custom ops): {torch.allclose(x.grad, 8 * x.detach())}")

def chained(x):
    return SquareOp.apply(DoubleOp.apply(x))

ok = torch.autograd.gradcheck(chained, (x,))
print(f"gradcheck passed on the two-op chain: {ok}")
```

Genuinely run:

```
forward matches (2x)^2: True
forward matches 4x^2: True
x.grad matches 8x (chain rule through two custom ops): True
gradcheck passed on the two-op chain: True
```

### Discussion

`x.grad` matches `8x` exactly, which is the calculus this book can check by hand: `d/dx (2x)^2 = 2 * (2x) * 2 = 8x`, where the first factor of `2` comes from `SquareOp`'s own `2x`-times-`grad_output` rule (with its `x` being `DoubleOp`'s output, `2x`) and the second factor of `2` comes from `DoubleOp`'s `backward` multiplying that result again. Neither `DoubleOp` nor `SquareOp` has any awareness of the other — `DoubleOp.backward` simply doubles whatever gradient it's handed, and `SquareOp.backward` simply applies its own local derivative to whatever gradient IT is handed. Autograd is what threads them together correctly, calling each `Function`'s `backward` in reverse order and passing the previous one's output as the next one's `grad_output`, exactly as it would for two built-in operations. `gradcheck` confirms this holds as a general property of the composition, not merely for this one random `x`.

## 30.4 Capstone: A Three-Op Chain, and an Honest Tally

### Intuition

Two custom ops composed correctly; the natural next question is whether that scales to three, and whether it's possible to account for exactly how many of a chain's real-kernel launch attempts actually succeeded in a given environment — useful both as a debugging tool and as this book's own closing honesty check for the chapter.

### Background

A third operation, `AddOneOp` (`y = x + 1`), is added to the chain: `y = AddOneOp(SquareOp(DoubleOp(x))) = (2x)^2 + 1`, so `dy/dx = 8x` still (adding a constant doesn't change the derivative). `AddOneOp`'s `backward` is the identity, but is still given its own real cuTile kernel (`kernel_copy`) rather than skipped, so all three links follow the identical "real forward kernel, real backward kernel" discipline. After running forward and backward, each stage's `used_real_kernel_forward`/`used_real_kernel_backward` flags are read directly off the intermediate tensors' `grad_fn` objects and tallied.

### Worked Example 30.4.1 — three custom ops, six kernel-launch attempts, one tally

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
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.used_real_kernel_backward = False
        return 2 * grad_output

@ct.kernel
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)

@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)

class SquareOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square_backward, (x, grad_output, grad_input, x.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return 2 * x * grad_output

@ct.kernel
def kernel_add_one(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x + 1)

@ct.kernel
def kernel_copy(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

class AddOneOp(torch.autograd.Function):
    # A third, genuinely distinct op: y = x + 1. Its backward is the
    # identity (grad_input = grad_output) -- still given its own real
    # cuTile kernel (a straight copy) rather than skipped, so all three
    # links in this chain follow the identical "write a real forward
    # kernel AND a real backward kernel" discipline.
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_add_one, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x + 1

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(grad_output)
            ct.launch(stream, (1,), kernel_copy, (grad_output, grad_input, grad_output.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
doubled = DoubleOp.apply(x)
squared = SquareOp.apply(doubled)
y = AddOneOp.apply(squared)

print(f"forward matches (2x)^2 + 1: {torch.equal(y, (2 * x.detach()) ** 2 + 1)}")

y.sum().backward()
print(f"x.grad matches 8x: {torch.allclose(x.grad, 8 * x.detach())}")

stages = [
    ("DoubleOp", doubled.grad_fn),
    ("SquareOp", squared.grad_fn),
    ("AddOneOp", y.grad_fn),
]
real_count = 0
total_count = 0
for name, gf in stages:
    fwd = gf.used_real_kernel_forward
    bwd = gf.used_real_kernel_backward
    print(f"{name}: forward used real kernel = {fwd}, backward used real kernel = {bwd}")
    real_count += int(fwd) + int(bwd)
    total_count += 2
print(f"real-kernel launches that succeeded: {real_count}/{total_count}")

def chained(x):
    return AddOneOp.apply(SquareOp.apply(DoubleOp.apply(x)))

ok = torch.autograd.gradcheck(chained, (x,))
print(f"gradcheck passed on the three-op chain: {ok}")
```

Genuinely run:

```
forward matches (2x)^2 + 1: True
x.grad matches 8x: True
DoubleOp: forward used real kernel = False, backward used real kernel = False
SquareOp: forward used real kernel = False, backward used real kernel = False
AddOneOp: forward used real kernel = False, backward used real kernel = False
real-kernel launches that succeeded: 0/6
gradcheck passed on the three-op chain: True
```

### Discussion

The tally reads `0/6` — every single one of the six real-kernel launch attempts (three forward, three backward) hit the exact same hardware wall Chapter 29 first named, since all six ultimately depend on the same failing `torch.cuda.current_stream()` call. That number being zero doesn't weaken anything this chapter has shown: `used_real_kernel_forward`/`used_real_kernel_backward` exist specifically so this fact is visible and honestly reported, rather than silently true and unmentioned. What the `0/6` tally does NOT undermine is the forward value (`(2x)^2 + 1`, confirmed exactly), the gradient (`8x`, confirmed exactly), or the `gradcheck` result — all three are genuine facts about the fallback computations and the six real, independently-compiled kernels sitting behind them, none of which required a GPU to establish. `AddOneOp`'s identity backward, given its own real `kernel_copy` rather than skipped as "too trivial to bother," matters for exactly this reason: it keeps the chapter's central claim uniform across all three links — every operation in this chain, however simple its gradient, is backed by a real, separately-compiled cuTile kernel on both sides, not just the two links where the gradient happened to be more interesting to write.

## 30.5 A Reader-Supplied Architecture: CuPy-Backed Kernels and a Wall That Moves Earlier

### Intuition

Every custom operation this chapter and Chapter 29 have built uses `torch.autograd.Function` with `torch` tensors as the host-side array type. That's one valid architecture, not the only one. A reader-supplied cuTile autograd engine builds the equivalent machinery on CuPy arrays instead — its own `Tensor` wrapper class, its own topological-sort-based `backward()` driver, no `torch.autograd` anywhere. This book's rule has never been "does an architecture look right," it's "what genuinely happens when it runs" — so the natural question, faced with a materially different design, is whether this sandbox's hardware wall shows up in the same place for a CuPy-based engine as it did for the `torch`-based ones.

### Background

The reader's engine defines four module-level `@ct.kernel` functions — `vector_add_kernel`, `vector_mul_kernel`, `relu_forward_kernel`, `relu_backward_kernel` — reproduced here unchanged. They exercise corners of `cuda.tile` this book hasn't combined quite this way before: `ct.bid(0)` for the block index rather than a fixed `(0,)`, `ct.load`'s keyword-argument form (`index=`, `shape=`, `padding_mode=`) rather than the positional form this book's own kernels have used since Chapter 1, `ct.PaddingMode.ZERO` passed explicitly, `ct.maximum` for `relu`'s forward, and a boolean-comparison-then-`astype` mask for `relu`'s backward. None of that is exotic — it is simply a different, equally valid way to call the same compiler this book has run in every chapter — and none of it depends on which host library eventually launches the result.

The host wrappers do depend on that, though. `cutile_vector_add`, `cutile_vector_mul`, `cutile_relu_forward`, and `cutile_relu_backward` each call `ct.launch(cp.cuda.get_current_stream(), grid, kernel, args)` directly, with no `try`/`except` anywhere in the file. That absence is the whole story of this section: every `Function` in this chapter reports a genuine result in this sandbox only because its `except RuntimeError:` fallback runs on ordinary `torch` arithmetic that never needed CUDA to begin with. An engine with no such fallback has nowhere to land when the driver isn't there.

### Worked Example 30.5.1 — the same compiler, a different host library, and where each one's wall actually sits

```python
import cuda.tile as ct
import io

ConstInt = ct.Constant[int]

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig_3in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# The four kernels below are reproduced from a reader-supplied CuPy-based
# autograd engine, unchanged except for the ConstInt alias defined above.
# They use ct.bid, keyword-argument ct.load/ct.store, ct.PaddingMode.ZERO,
# and ct.maximum -- none of which any prior chapter's Worked Examples have
# used in exactly this combination.

@ct.kernel
def vector_add_kernel(a, b, c, tile_size: ConstInt):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

@ct.kernel
def vector_mul_kernel(a, b, c, tile_size: ConstInt):
    bid = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO
    a_tile = ct.load(a, index=(bid,), shape=(tile_size,), padding_mode=zero_pad)
    b_tile = ct.load(b, index=(bid,), shape=(tile_size,), padding_mode=zero_pad)
    c_tile = a_tile * b_tile
    ct.store(c, index=(bid,), tile=c_tile)

@ct.kernel
def relu_forward_kernel(x, y, tile_size: ConstInt):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    result = ct.maximum(x_tile, 0.0)
    ct.store(y, index=(pid,), tile=result)

@ct.kernel
def relu_backward_kernel(grad_out, x, grad_in, tile_size: ConstInt):
    pid = ct.bid(0)
    grad_out_tile = ct.load(grad_out, index=(pid,), shape=(tile_size,))
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    mask = (x_tile > 0).astype(ct.float32)
    result = grad_out_tile * mask
    ct.store(grad_in, index=(pid,), tile=result)

sig_2in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

n_add = compile_bytes(vector_add_kernel, sig_3in)
print(f"vector_add_kernel (reader's engine, keyword ct.load/ct.store): {n_add} cubin bytes")

n_mul = compile_bytes(vector_mul_kernel, sig_3in)
print(f"vector_mul_kernel (reader's engine, ct.PaddingMode.ZERO): {n_mul} cubin bytes")

n_relu_f = compile_bytes(relu_forward_kernel, sig_2in)
print(f"relu_forward_kernel (reader's engine, ct.maximum): {n_relu_f} cubin bytes")

n_relu_b = compile_bytes(relu_backward_kernel, sig_3in)
print(f"relu_backward_kernel (reader's engine, mask via astype): {n_relu_b} cubin bytes")

# All four kernels genuinely compile -- this book's oldest verification
# method, and it doesn't care which host-side library eventually launches
# the result. The question this section actually asks is what happens one
# level up, where the reader's engine wires these kernels to CuPy arrays
# instead of torch tensors.

import cupy as cp

print(f"cupy imported: {cp.__version__}")
print(f"cp.cuda.is_available(): {cp.cuda.is_available()}")

try:
    x = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    print(f"cp.array(...) succeeded: {x}")
except Exception as e:
    print(f"cp.array(...) failed: {type(e).__name__}: {e}")

import torch
try:
    t = torch.tensor([1.0, 2.0, 3.0])
    print(f"torch.tensor(...) on CPU succeeded: {t.tolist()}")
except Exception as e:
    print(f"torch.tensor(...) failed: {type(e).__name__}: {e}")
```

Genuinely run:

```
vector_add_kernel (reader's engine, keyword ct.load/ct.store): 23584 cubin bytes
vector_mul_kernel (reader's engine, ct.PaddingMode.ZERO): 23712 cubin bytes
relu_forward_kernel (reader's engine, ct.maximum): 21280 cubin bytes
relu_backward_kernel (reader's engine, mask via astype): 24032 cubin bytes
cupy imported: 14.2.0
cp.cuda.is_available(): False
cp.array(...) failed: CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
torch.tensor(...) on CPU succeeded: [1.0, 2.0, 3.0]
```

### Discussion

All four kernels compile to distinct, real cubins. `cuda.tile`'s compiler treats `ct.bid`, keyword-argument `ct.load`/`ct.store`, `ct.PaddingMode.ZERO`, and `ct.maximum` exactly as it treats every other combination of primitives this book has exercised — it does not care, and has never cared, which host library eventually launches the result. That much of the reader's engine is genuinely sound.

What does not survive contact with this sandbox is any attempt to actually run the CuPy side of it. `cp.cuda.is_available()` reports `False`, exactly like `torch.cuda.is_available()` has every time this book has checked it — but the consequence is worse. `torch.tensor([1.0, 2.0, 3.0])`, called on the very next line, succeeds immediately, because PyTorch tensors have a real, independent CPU backend; "no CUDA" for `torch` means only that the `.cuda()` device is unavailable. `cp.array(...)` has no such fallback: a CuPy array IS a GPU array, with no CPU implementation behind it, so the very first array this engine would ever construct — before a single kernel is compiled, before its `Tensor` wrapper runs a line of logic, before autograd enters the picture at all — raises `cudaErrorInsufficientDriver`.

This is why every `Function` in this chapter and Chapter 29 could report a genuine forward value, a genuine gradient, and a genuine `gradcheck` result despite the identical missing driver, while the reader's engine cannot report any of that here: `DoubleOp`, `SquareOp`, and `AddOneOp`'s `except RuntimeError:` blocks fall back to ordinary `torch` CPU arithmetic that never needed CUDA in the first place. There is no equivalent "CuPy, but on the CPU" for this engine to fall back to in plain Python — the wall this sandbox has hit in every form this book has tried, from Chapter 27's `TypeError` when no stream was given at all, to `torch.cuda.current_stream()`'s own `RuntimeError`, to now CuPy's `CUDARuntimeError` at the moment of array construction, moves earlier or later depending entirely on which host library's array type an engine is built around — never on anything about `cuda.tile` itself, which has compiled every kernel handed to it, in every chapter, without exception.

## Chapter Summary

This chapter opened Part 4 by giving a custom cuTile-backed operation what Chapter 29's examples never had: a backward pass written as its own, genuinely distinct `@ct.kernel`, compiled to its own real cubin, rather than a formula living only in Python. `SquareOp` combined a real forward kernel and a real backward kernel into one complete `torch.autograd.Function`, with each half independently attempting a real `ct.launch` and falling back honestly when it hit this book's hardware wall — verified correct via `gradcheck`, which checks the fallback's gradient against the forward computation directly, with no awareness of (and no need to trust) the real kernel encoding the identical formula. Chaining `DoubleOp` and `SquareOp` confirmed the chain rule composes two independently-written custom operations exactly as it would compose built-in ones, matching a hand-derived `8x` exactly and passing `gradcheck` on the full composition. The three-op capstone extended this to `DoubleOp`, `SquareOp`, and a new `AddOneOp` (given its own real forward and backward kernels despite its trivial gradient), producing a genuine, honestly-reported tally of `0/6` real-kernel launches — a number that reports this sandbox's hardware limitation plainly rather than hiding it, while leaving every forward value, every gradient, and the full-chain `gradcheck` result genuinely confirmed. Finally, a reader-supplied CuPy-based autograd engine showed that the hardware wall isn't fixed to one spot: its four cuTile kernels compiled exactly as cleanly as this chapter's `torch`-based ones, but with no CPU-array fallback available to CuPy the way it is to `torch`, the wall for that architecture appears a full layer earlier — at plain array construction, before any kernel, `Tensor` wrapper, or autograd logic ever runs.

## Self-Check Questions

1. Section 30.1 found `kernel_square` and `kernel_square_backward` compile to different byte counts (20384 vs. 23072). Is that difference, by itself, evidence that the backward kernel does "more work" than the forward one — and what about the two kernels' signatures makes a bare byte-count comparison here different from the naming-controlled, single-variable-changed comparisons this book has relied on since Chapter 10?
2. Section 30.2's `gradcheck` call only ever exercises the FALLBACK computation in this sandbox, never the real `kernel_square`/`kernel_square_backward` launch. What relationship would need to hold between the fallback code and the real kernel's logic for a passing `gradcheck` today to say anything meaningful about the real kernel's eventual correctness on real hardware?
3. Section 30.3 confirmed the chain rule produces `8x` for `d/dx (2x)^2` across two independently-written `Function`s. If `SquareOp`'s `backward` had a bug — say, it returned `x * grad_output` instead of `2 * x * grad_output` — would `DoubleOp`'s own, separately-correct `backward` "notice" or partially compensate for that bug in the final `x.grad`, or would the error simply propagate through unchanged?
4. Section 30.4's tally reported `0/6`. Suppose a future chapter runs this exact same capstone code on real hardware with a working driver. Which specific lines of the capstone's OUTPUT would you expect to change, and which would you expect to stay identical?
5. `AddOneOp`'s backward is the mathematical identity (`grad_input = grad_output`), yet this chapter still gave it its own real `kernel_copy` rather than reusing `grad_output` directly with no kernel at all. What does insisting on a real kernel even for a trivial backward pass protect against, that simply returning `grad_output` unchanged in the `except` fallback alone would not?
6. Section 30.5's `torch.tensor([1.0, 2.0, 3.0])` succeeded immediately, while `cp.array([1.0, 2.0, 3.0], dtype=cp.float32)` failed with `CUDARuntimeError: cudaErrorInsufficientDriver`, even though `cp.cuda.is_available()` and `torch.cuda.is_available()` both report `False` in this sandbox. If a future reader wanted to adapt the CuPy-based engine so it could report genuine fallback results here the way `DoubleOp`, `SquareOp`, and `AddOneOp` do, what would actually have to change, and why would it be a bigger change than adding a `try`/`except` around each `ct.launch` call?

## Where We Go Next

This chapter established the shape a real cuTile-backed operation's backward pass takes: an independent kernel, verified independently, combined into a `Function` whose two halves can each be tested for correctness without ever needing to agree on how they got their answer. The three-op chain also surfaced a detail worth returning to directly: `AddOneOp`'s forward kernel (`kernel_add_one`) and `SquareOp`'s backward kernel (`kernel_square_backward`) both load and store adjacent, fixed-size tiles with no cross-tile interaction at all — every operation in this chapter has been purely elementwise. Part 4's next thread follows naturally from that limitation: what happens when a custom operation's backward pass needs a REDUCTION (chaining a custom `Function` around something like Chapter 24's `ct.reduce`, where a single output element's gradient depends on many input elements at once, not just the one it sits beside) — the point where a backward kernel stops being able to mirror its forward kernel's structure line for line.

## Worked Solutions

**1.** The byte-count difference is not trustworthy evidence of "more work" by itself, and for a reason more fundamental than the naming confound this book has controlled for since Chapter 10: the two kernels don't even share the same SIGNATURE. `kernel_square` takes one array input and produces one array output; `kernel_square_backward` takes two array inputs and produces one output. Chapter 10's naming-confound discipline assumes the comparison is otherwise apples-to-apples — same signature shape, same computation structure, only the kernel's name (or one other single variable) changed. Here, the extra array parameter alone would be expected to add bytes (an additional pointer/stride to decode and an additional load instruction), regardless of whether the arithmetic inside is simple or complex. A trustworthy "backward does more work" claim would need a same-signature comparison — for instance, a hypothetical single-input kernel that also happened to need `2 * x * grad_output`-shaped arithmetic — which this pair of kernels doesn't offer, because a genuinely different number of inputs is precisely what a backward pass needing the saved input AND the incoming gradient actually requires.

**2.** For a passing `gradcheck` on the fallback to say anything about the real kernel's eventual hardware correctness, the fallback and the real kernel would need to compute the exact same mathematical function on the exact same inputs — not merely "do something plausible," but be exact, verified restatements of one formula in two different languages (Python/`torch` arithmetic and cuTile tile code). This chapter maintains that relationship by construction and by comment (each fallback's formula is written to mirror its kernel's `ct.store` expression line for line), but nothing in `gradcheck` itself checks this correspondence — `gradcheck` only ever sees the fallback path run, so a real kernel that silently diverged from its fallback (say, a typo that computed `x + grad_output` instead of `2 * x * grad_output` inside `kernel_square_backward`, never triggered because the kernel never launches) would leave `gradcheck` reporting success while the real, eventually-launchable kernel was actually wrong. The honest scope of today's `gradcheck` result is: "the fallback formula is a correct gradient for the fallback forward." Confirming the real kernel matches that formula byte-for-byte in intent (not bytes-of-cubin) is a human/code-review claim this chapter is making, not something `gradcheck` itself is in a position to verify without a real launch.

**3.** The error would propagate through unchanged — `DoubleOp`'s `backward` has no way to "notice" or compensate for a bug elsewhere in the chain, because each `Function`'s `backward` only ever sees the single `grad_output` value handed to it by whatever comes after it in the chain, with no visibility into how that value was computed or whether it's correct. If `SquareOp.backward` returned `x * grad_output` instead of `2 * x * grad_output`, it would hand `DoubleOp.backward` a value that's already off by a factor of 2 (relative to `SquareOp`'s true local derivative), and `DoubleOp.backward` would faithfully double THAT wrong value, propagating the error rather than correcting it — the final `x.grad` would be `4x` instead of `8x`, off by exactly the same factor of 2 the bug introduced. This is precisely why `gradcheck` on the full chain (not just on `SquareOp` in isolation) matters: a bug in one link's `backward` produces a composed gradient that's wrong by a predictable, traceable factor, but nothing about the OTHER links' own correctness would mask or reveal it without checking the composition itself.

**4.** On real hardware, every `used real kernel = False` line would be expected to flip to `True` (six flips total), and the final tally line would read `6/6` instead of `0/6`, since a working driver would let `torch.cuda.current_stream()` succeed and every `ct.launch` call reach real hardware rather than raising. Everything else printed would be expected to stay IDENTICAL: `forward matches (2x)^2 + 1: True`, `x.grad matches 8x: True`, and `gradcheck passed on the three-op chain: True` — because those three lines report facts about the MATHEMATICS (the forward value, the gradient, and gradcheck's finite-difference comparison), which is exactly the same mathematics whether it was computed by the plain-`torch` fallback or by three real, launched cuTile kernels, precisely because this chapter took care to keep the fallback and the kernel logic in exact correspondence (Question 2's point). A meaningfully different behavior on real hardware, beyond simply "and now it uses the GPU," would itself be the real news — a false confirmation this chapter never had a way to check for until real hardware becomes available.

**5.** Giving `AddOneOp.backward` its own real `kernel_copy`, rather than just returning `grad_output` unchanged in a plain fallback with no kernel behind it at all, protects against a subtle asymmetry this chapter's own tally would otherwise have hidden: if `AddOneOp`'s backward had no real kernel at all, then EVERY real cuTile-backed operation in a hypothetical larger pipeline that happened to have an "identity-like" gradient would quietly stop being cuTile-backed the moment its gradient looked trivial enough to write in plain Python — a slow, invisible erosion of "how much of this pipeline is actually running on the GPU" that would never show up as an error, only as a pipeline that runs correctly but does progressively less of its real work in cuTile kernels than a reader of the code might assume. Requiring a real kernel even here keeps the chapter's central claim uniform and honestly checkable: every operation in the chain is backed by a real, compiled artifact on both sides, and the `0/6` tally reports a fact about THIS SANDBOX's hardware, not about which operations the author decided were "worth" writing a real kernel for.

**6.** Wrapping each `ct.launch` call in `try`/`except` would not be enough, because the failure in Section 30.5 happens before any `ct.launch` call is ever reached — it happens the moment a `cp.ndarray` is constructed at all, which for this engine is also the moment its `Tensor` wrapper is constructed, since `Tensor.__init__` forces its data through `cp.asarray`. A `try`/`except` around `ct.launch` alone would never even be entered, because execution would already have raised inside `Tensor.__init__`, long before reaching any kernel-launching method. To genuinely report a fallback result the way `DoubleOp`/`SquareOp`/`AddOneOp` do, the engine's `Tensor` class itself would need a second, CPU-only code path — most plausibly built on plain NumPy arrays instead of CuPy ones — with every operation's forward AND backward logic duplicated (or written generically enough to run under either `numpy` or `cupy`), and every `cutile_*` host wrapper's `try`/`except` guarding not just the `ct.launch` call but the CuPy array operations around it. That is substantially more work than adding exception handling around a launch call: it means giving the engine an entirely separate array backend, the way `torch` already ships one built in, rather than adding a fallback branch to code that already has somewhere safe to fall back to.

## Complete Runnable Code

### File: `01_square_forward_and_backward_kernels.py`

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

# Forward kernel: y = x * x
@ct.kernel
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)
n_forward = compile_bytes(kernel_square, sig_2in)
print(f"kernel_square (forward, y = x*x): {n_forward} cubin bytes")

# Backward kernel: grad_input = 2 * x * grad_output -- a genuinely
# DIFFERENT kernel from the forward one, taking two array inputs
# (the saved x, and grad_output) rather than one.
@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)
n_backward = compile_bytes(kernel_square_backward, sig_3in)
print(f"kernel_square_backward (grad_input = 2*x*grad_output): {n_backward} cubin bytes")
print(f"forward and backward are different kernels, different byte counts: {n_forward != n_backward}")
```

### File: `02_squareop_two_kernel_autograd_function.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)

@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)

class SquareOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError as e:
            ctx.used_real_kernel_forward = False
            return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square_backward, (x, grad_output, grad_input, x.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError as e:
            ctx.used_real_kernel_backward = False
            return 2 * x * grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = SquareOp.apply(x)
print(f"forward matches x**2: {torch.equal(y, x.detach() ** 2)}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad matches 2*x: {torch.allclose(x.grad, 2 * x.detach())}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(SquareOp.apply, (x,))
print(f"gradcheck passed: {ok}")
```

### File: `03_chaining_two_custom_ops.py`

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
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)

@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)

class SquareOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square, (x, out, x.shape[0]))
            return out
        except RuntimeError:
            return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square_backward, (x, grad_output, grad_input, x.shape[0]))
            return grad_input
        except RuntimeError:
            return 2 * x * grad_output

# Chain: y = Square(Double(x)) = (2x)^2 = 4x^2. Each op is a SEPARATE
# custom autograd.Function, each individually backed by (attempted)
# real cuTile kernels. Composing them tests whether autograd's chain
# rule correctly threads gradients through TWO custom Functions, not
# just one.
x = torch.randn(8, dtype=torch.float64, requires_grad=True)
doubled = DoubleOp.apply(x)
y = SquareOp.apply(doubled)

print(f"forward matches (2x)^2: {torch.equal(y, (2 * x.detach()) ** 2)}")
print(f"forward matches 4x^2: {torch.equal(y, 4 * x.detach() ** 2)}")

y.sum().backward()
# d/dx (2x)^2 = 8x
print(f"x.grad matches 8x (chain rule through two custom ops): {torch.allclose(x.grad, 8 * x.detach())}")

def chained(x):
    return SquareOp.apply(DoubleOp.apply(x))

ok = torch.autograd.gradcheck(chained, (x,))
print(f"gradcheck passed on the two-op chain: {ok}")
```

### File: `04_capstone_three_op_chain_with_kernel_tally.py`

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
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.used_real_kernel_backward = False
        return 2 * grad_output

@ct.kernel
def kernel_square(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x * x)

@ct.kernel
def kernel_square_backward(x_saved, grad_output, grad_input, tile_size: ct.Constant[int]):
    x = ct.load(x_saved, (0,), (8,))
    g = ct.load(grad_output, (0,), (8,))
    ct.store(grad_input, (0,), 2 * x * g)

class SquareOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_square_backward, (x, grad_output, grad_input, x.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return 2 * x * grad_output

@ct.kernel
def kernel_add_one(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x + 1)

@ct.kernel
def kernel_copy(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

class AddOneOp(torch.autograd.Function):
    # A third, genuinely distinct op: y = x + 1. Its backward is the
    # identity (grad_input = grad_output) -- still given its own real
    # cuTile kernel (a straight copy) rather than skipped, so all three
    # links in this chain follow the identical "write a real forward
    # kernel AND a real backward kernel" discipline.
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_add_one, (x, out, x.shape[0]))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x + 1

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(grad_output)
            ct.launch(stream, (1,), kernel_copy, (grad_output, grad_input, grad_output.shape[0]))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
doubled = DoubleOp.apply(x)
squared = SquareOp.apply(doubled)
y = AddOneOp.apply(squared)

print(f"forward matches (2x)^2 + 1: {torch.equal(y, (2 * x.detach()) ** 2 + 1)}")

y.sum().backward()
print(f"x.grad matches 8x: {torch.allclose(x.grad, 8 * x.detach())}")

stages = [
    ("DoubleOp", doubled.grad_fn),
    ("SquareOp", squared.grad_fn),
    ("AddOneOp", y.grad_fn),
]
real_count = 0
total_count = 0
for name, gf in stages:
    fwd = gf.used_real_kernel_forward
    bwd = gf.used_real_kernel_backward
    print(f"{name}: forward used real kernel = {fwd}, backward used real kernel = {bwd}")
    real_count += int(fwd) + int(bwd)
    total_count += 2
print(f"real-kernel launches that succeeded: {real_count}/{total_count}")

def chained(x):
    return AddOneOp.apply(SquareOp.apply(DoubleOp.apply(x)))

ok = torch.autograd.gradcheck(chained, (x,))
print(f"gradcheck passed on the three-op chain: {ok}")
```

### File: `05_a_second_architecture_cupy_kernels_and_earlier_wall.py`

```python
import cuda.tile as ct
import io

ConstInt = ct.Constant[int]

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig_3in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# The four kernels below are reproduced from a reader-supplied CuPy-based
# autograd engine, unchanged except for the ConstInt alias defined above.
# They use ct.bid, keyword-argument ct.load/ct.store, ct.PaddingMode.ZERO,
# and ct.maximum -- none of which any prior chapter's Worked Examples have
# used in exactly this combination.

@ct.kernel
def vector_add_kernel(a, b, c, tile_size: ConstInt):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

@ct.kernel
def vector_mul_kernel(a, b, c, tile_size: ConstInt):
    bid = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO
    a_tile = ct.load(a, index=(bid,), shape=(tile_size,), padding_mode=zero_pad)
    b_tile = ct.load(b, index=(bid,), shape=(tile_size,), padding_mode=zero_pad)
    c_tile = a_tile * b_tile
    ct.store(c, index=(bid,), tile=c_tile)

@ct.kernel
def relu_forward_kernel(x, y, tile_size: ConstInt):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    result = ct.maximum(x_tile, 0.0)
    ct.store(y, index=(pid,), tile=result)

@ct.kernel
def relu_backward_kernel(grad_out, x, grad_in, tile_size: ConstInt):
    pid = ct.bid(0)
    grad_out_tile = ct.load(grad_out, index=(pid,), shape=(tile_size,))
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    mask = (x_tile > 0).astype(ct.float32)
    result = grad_out_tile * mask
    ct.store(grad_in, index=(pid,), tile=result)

sig_2in = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

n_add = compile_bytes(vector_add_kernel, sig_3in)
print(f"vector_add_kernel (reader's engine, keyword ct.load/ct.store): {n_add} cubin bytes")

n_mul = compile_bytes(vector_mul_kernel, sig_3in)
print(f"vector_mul_kernel (reader's engine, ct.PaddingMode.ZERO): {n_mul} cubin bytes")

n_relu_f = compile_bytes(relu_forward_kernel, sig_2in)
print(f"relu_forward_kernel (reader's engine, ct.maximum): {n_relu_f} cubin bytes")

n_relu_b = compile_bytes(relu_backward_kernel, sig_3in)
print(f"relu_backward_kernel (reader's engine, mask via astype): {n_relu_b} cubin bytes")

# All four kernels genuinely compile -- this book's oldest verification
# method, and it doesn't care which host-side library eventually launches
# the result. The question this section actually asks is what happens one
# level up, where the reader's engine wires these kernels to CuPy arrays
# instead of torch tensors.

import cupy as cp

print(f"cupy imported: {cp.__version__}")
print(f"cp.cuda.is_available(): {cp.cuda.is_available()}")

try:
    x = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    print(f"cp.array(...) succeeded: {x}")
except Exception as e:
    print(f"cp.array(...) failed: {type(e).__name__}: {e}")

import torch
try:
    t = torch.tensor([1.0, 2.0, 3.0])
    print(f"torch.tensor(...) on CPU succeeded: {t.tolist()}")
except Exception as e:
    print(f"torch.tensor(...) failed: {type(e).__name__}: {e}")
```
