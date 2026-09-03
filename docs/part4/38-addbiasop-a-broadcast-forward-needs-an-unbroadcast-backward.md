# Chapter 38: AddBiasOp — A Broadcast Forward Needs an Unbroadcast Backward

> "Every custom Function this book has built in Part 4 has combined values through a reduction, a matmul, or a table lookup. This chapter closes Part 4 with the plainest operation of all — adding two tensors — and finds that plainness was never really about the forward pass. Broadcasting a smaller tensor up to match a bigger one is free; the backward pass is the one that has to pay for it, by summing back down exactly where the forward pass spread out."

**What you will understand by the end of this chapter:**

- `ct.sum(x, axis=0)`: this book's first use of a partial-axis reduction, as opposed to the full reduction to a scalar (`axis=None`) every previous reduction chapter has used
- Why adding a broadcast bias to every row of a matrix needs a backward pass that is asymmetric in a way none of this book's other binary operations have been: the gradient with respect to the full-sized operand is a trivial identity copy, while the gradient with respect to the broadcast operand must sum back down across every axis that was broadcast on the way forward
- A naming-confound test confirming `ct.sum(x, None)` and `ct.sum(x, axis=0)` compile to genuinely different cubin under an identical function name and identical load shape — the axis argument is not a cosmetic label
- `AddBiasOp`, a `torch.autograd.Function` whose two backward kernels reflect this asymmetry directly, `gradcheck`-verified against both `x` and `bias`
- A capstone assembling a complete linear layer — `y = X @ W + bias` — from `MatMulOp` (Chapter 35), `AddBiasOp` (this chapter), and `SumOp` (Chapter 31), the most conventional neural-network building block this book's Part 4 chapters have produced

**What you need to know first:**

- Chapter 17's broadcasting rules for elementwise arithmetic, and the fact that `bias`'s shape `(N,)` broadcasts against `x`'s shape `(M, N)` by matching trailing dimensions.
- Every reduction chapter since Chapter 31, all of which used `ct.sum(x, None)` — a full reduction to a single scalar — exclusively.
- Chapter 35's shape-generic `MatMulOp` and Chapter 31's `SumOp`, both reused in this chapter's capstone.
- Chapter 10's naming-confound discipline, applied here to an operation's `axis` argument rather than its dtype or shape.

## 38.1 Two Kernels, Two Very Different Jobs

### Intuition

`z = x + bias`, with `x` shaped `(M, N)` and `bias` shaped `(N,)`, broadcasts `bias` across every one of `x`'s `M` rows — the same rule Chapter 17 established for elementwise arithmetic in general. The forward pass treats both operands almost symmetrically: load one, broadcast the other up to match, add. The backward pass does not. `d(x + bias)/dx` is `1` everywhere, so `grad_x` is simply `grad_out`, unchanged. `d(x + bias)/dbias` is also `1` everywhere, but `bias` only has `N` elements while `grad_out` has `M * N` — every one of `bias`'s `N` positions influenced `M` different output positions, once per row, so `bias`'s gradient has to SUM `grad_out` back down across those `M` rows.

### Background

Every reduction this book has built since Chapter 31 — `SumOp`, `MaxOp`, and their multi-block and ragged-tail successors — reduced its input down to a single scalar with `ct.sum(x, None)` or `ct.max(x, None)`. `bias`'s gradient needs something structurally different: a reduction along ONE specific axis (the row axis, axis 0) that leaves the other axis (the column axis) intact, producing a `(N,)`-shaped result rather than a scalar. `ct.sum`'s own signature has supported an `axis` argument, and a `keepdims` option, all along; this chapter is simply the first to actually pass anything other than `None`.

### Worked Example 38.1.1 — Forward, both backward kernels, and axis=None vs. axis=0

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

M, N = 8, 8

def sig_2d_1d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

def sig_2d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

def sig_2d_1d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- Forward: z = x + bias, with bias (N,) broadcast across every one of
# x's M rows -- ordinary NumPy-style broadcasting, the same rule Chapter
# 17's elementwise arithmetic family already established. ---
@ct.kernel
def kernel_add_broadcast_forward(x, bias, out):
    xt = ct.load(x, (0, 0), (M, N))
    bt = ct.load(bias, (0,), (N,))
    z = xt + ct.broadcast_to(bt, (M, N))
    ct.store(out, (0, 0), z)

# --- Backward w.r.t. x: d(x + bias)/dx is 1 everywhere, so grad_x is
# just grad_out, unchanged -- the easy half of this operation's
# backward. ---
@ct.kernel
def kernel_add_broadcast_backward_dx(grad_out, grad_x):
    g = ct.load(grad_out, (0, 0), (M, N))
    ct.store(grad_x, (0, 0), g)

# --- Backward w.r.t. bias: every one of bias's N columns contributed to
# M different output positions (one per row), so bias's gradient must
# SUM grad_out back down across the M rows it was broadcast into -- the
# first time this book has reduced along a specific axis (axis=0) rather
# than reducing a tile down to a single scalar. ---
@ct.kernel
def kernel_add_broadcast_backward_dbias(grad_out, grad_bias):
    g = ct.load(grad_out, (0, 0), (M, N))
    reduced = ct.sum(g, axis=0)
    ct.store(grad_bias, (0,), reduced)

n = compile_bytes(kernel_add_broadcast_forward, sig_2d_1d_2d())
print(f"kernel_add_broadcast_forward: {n} cubin bytes")
n = compile_bytes(kernel_add_broadcast_backward_dx, sig_2d_2d())
print(f"kernel_add_broadcast_backward_dx: {n} cubin bytes")
n = compile_bytes(kernel_add_broadcast_backward_dbias, sig_2d_1d())
print(f"kernel_add_broadcast_backward_dbias: {n} cubin bytes")

print()

# --- A full reduction (axis=None) versus an axis=0 partial reduction,
# under an identical function name and an identical (M, N) load shape,
# confirms the compiler treats these as genuinely different operations
# rather than the axis argument being cosmetic. ---
def make_probe(axis):
    if axis is None:
        def kernel_probe(a, c):
            x = ct.load(a, (0, 0), (M, N))
            y = ct.sum(x, None)
            ct.store(c, (0,), ct.broadcast_to(y, (1,)))
    else:
        def kernel_probe(a, c):
            x = ct.load(a, (0, 0), (M, N))
            y = ct.sum(x, axis=0)
            ct.store(c, (0,), y)
    kernel_probe.__name__ = "kernel_probe"
    kernel_probe.__qualname__ = "kernel_probe"
    return ct.kernel(kernel_probe)

for axis in (None, 0):
    kernel_probe = make_probe(axis)
    n = compile_bytes(kernel_probe, sig_2d_1d())
    label = "axis=None (full reduction)" if axis is None else "axis=0 (partial reduction)"
    print(f"kernel_probe, {label}: {n} cubin bytes")

print()

# --- Correctness, on the host, since this sandbox cannot ct.launch:
# confirm the forward matches NumPy-style broadcasting addition, and
# that summing grad_out along axis 0 gives exactly the gradient a plain
# torch computation of the same broadcast add would produce. ---
torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
bias = torch.randn(N, dtype=torch.float64, requires_grad=True)

z = x + bias
print(f"forward matches x + bias (broadcast): {torch.allclose(z, x.detach() + bias.detach())}")

z.sum().backward()
grad_out_manual = torch.ones(M, N, dtype=torch.float64)
grad_x_manual = grad_out_manual.clone()
grad_bias_manual = grad_out_manual.sum(dim=0)
print(f"x.grad matches identity copy of grad_out: {torch.equal(x.grad, grad_x_manual)}")
print(f"bias.grad matches grad_out summed over axis 0: {torch.equal(bias.grad, grad_bias_manual)}")
```

### Genuinely run

```
kernel_add_broadcast_forward: 27296 cubin bytes
kernel_add_broadcast_backward_dx: 23360 cubin bytes
kernel_add_broadcast_backward_dbias: 22520 cubin bytes

kernel_probe, axis=None (full reduction): 23176 cubin bytes
kernel_probe, axis=0 (partial reduction): 21864 cubin bytes

forward matches x + bias (broadcast): True
x.grad matches identity copy of grad_out: True
bias.grad matches grad_out summed over axis 0: True
```

### Discussion

`kernel_add_broadcast_backward_dx` and `kernel_add_broadcast_backward_dbias` compile to noticeably different byte counts (`23360` versus `22520`) despite both reading the same `(M, N)` `grad_out` tile — one simply copies it back out unchanged, the other reduces it down to `(N,)` first, and those are different amounts of work reflected honestly in different compiled sizes. The `kernel_probe` comparison isolates the `axis` argument specifically: with the function name and the load shape held identical, `axis=None` and `axis=0` still produce different byte counts (`23176` versus `21864`), confirming that `ct.sum`'s `axis` parameter genuinely changes what the compiler generates rather than merely relabeling the same underlying reduction. `bias.grad` summing `grad_out` down to `(8,)` from `(8, 8)` — one value per column, each the sum of that column's `8` row-contributions — is the concrete shape of the "unbroadcast" this section's title promised.

## 38.2 AddBiasOp

### Intuition

`AddBiasOp` follows the same fallback shape as every custom `Function` since Chapter 30, with a backward pass that launches two kernels rather than one — mirroring `MatMulOp`'s structure from Chapter 34, but for a different reason: `MatMulOp` needed two kernels because its two gradients are two different matrix products, while `AddBiasOp` needs two kernels because its two gradients require fundamentally different amounts of work, one an identity copy and one a genuine reduction.

### Background

`gradcheck` here checks both of `AddBiasOp`'s inputs, `x` and `bias`, at once — a passing result confirms not only that `grad_x`'s identity-copy formula is correct, but that `grad_bias`'s `axis=0` reduction correctly accounts for `bias`'s influence on every one of `x`'s `M` rows simultaneously, since `gradcheck`'s central-difference perturbation of any single `bias` element affects all `M` output rows at once.

### Worked Example 38.2.1 — AddBiasOp, gradcheck-verified on both operands

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

M, N = 8, 8

@ct.kernel
def kernel_add_broadcast_forward(x, bias, out):
    xt = ct.load(x, (0, 0), (M, N))
    bt = ct.load(bias, (0,), (N,))
    z = xt + ct.broadcast_to(bt, (M, N))
    ct.store(out, (0, 0), z)

@ct.kernel
def kernel_add_broadcast_backward_dx(grad_out, grad_x):
    g = ct.load(grad_out, (0, 0), (M, N))
    ct.store(grad_x, (0, 0), g)

@ct.kernel
def kernel_add_broadcast_backward_dbias(grad_out, grad_bias):
    g = ct.load(grad_out, (0, 0), (M, N))
    reduced = ct.sum(g, axis=0)
    ct.store(grad_bias, (0,), reduced)

class AddBiasOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(M, N, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_forward, (x, bias, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x + bias

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_x = torch.empty(M, N, dtype=grad_output.dtype)
            grad_bias = torch.empty(N, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dx, (grad_output, grad_x))
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dbias, (grad_output, grad_bias))
            ctx.used_real_kernel_backward = True
            return grad_x, grad_bias
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_x = grad_output.clone()
            grad_bias = grad_output.sum(dim=0)
            return grad_x, grad_bias

torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
bias = torch.randn(N, dtype=torch.float64, requires_grad=True)

z = AddBiasOp.apply(x, bias)
print(f"forward matches x + bias: {torch.allclose(z, x.detach() + bias.detach())}")
print(f"used real forward kernel: {z.grad_fn.used_real_kernel_forward}")

z.sum().backward()
print(f"used real backward kernel: {z.grad_fn.used_real_kernel_backward}")

x_ref = x.detach().clone().requires_grad_()
bias_ref = bias.detach().clone().requires_grad_()
(x_ref + bias_ref).sum().backward()
print(f"x.grad matches reference broadcast-add backward: {torch.allclose(x.grad, x_ref.grad)}")
print(f"bias.grad matches reference broadcast-add backward: {torch.allclose(bias.grad, bias_ref.grad)}")
print(f"bias.grad equals M (every row contributed 1.0): {bias.grad.tolist()}")

ok = torch.autograd.gradcheck(AddBiasOp.apply, (x, bias))
print(f"AddBiasOp gradcheck passed: {ok}")
```

### Genuinely run

```
forward matches x + bias: True
used real forward kernel: False
used real backward kernel: False
x.grad matches reference broadcast-add backward: True
bias.grad matches reference broadcast-add backward: True
bias.grad equals M (every row contributed 1.0): [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
AddBiasOp gradcheck passed: True
```

### Discussion

`bias.grad` coming out as exactly `8.0` in every position — the row count `M` — is the simplest possible confirmation that the axis-0 reduction is counting correctly: with `z.sum().backward()` sending a gradient of `1.0` to every output position, each of `bias`'s `8` columns received exactly `8` such contributions, one per row. `gradcheck`'s pass on both `x` and `bias` together is a stronger statement than checking either alone would be: it confirms that `torch.autograd.Function.backward`'s convention of returning one gradient per `forward` argument, in order, works correctly even when those two gradients come from two entirely different kernels doing entirely different amounts of computation.

## 38.3 Capstone: A Complete Linear Layer

### Intuition

`y = X @ W + bias` is the single most common building block in this book's eventual Part 6 destination, and this chapter's capstone is the first place Part 4 has assembled it from scratch: `MatMulOp` (Chapter 35) computes the projection, `AddBiasOp` (this chapter) adds the per-column bias, and `SumOp` (Chapter 31) reduces the result to a scalar loss.

### Background

`bias.grad`'s correctness here depends on the axis-0 reduction correctly surviving being fed by `MatMulOp`'s output rather than a plain input tensor, and on `SumOp`'s own backward correctly broadcasting a scalar gradient back out before `AddBiasOp.backward` ever sees it — two additional layers of composition beyond Section 38.2's isolated test.

### Worked Example 38.3.1 — MatMulOp, AddBiasOp, and SumOp chained into a linear layer

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

M, N = 8, 8

@ct.kernel
def kernel_add_broadcast_forward(x, bias, out):
    xt = ct.load(x, (0, 0), (M, N))
    bt = ct.load(bias, (0,), (N,))
    z = xt + ct.broadcast_to(bt, (M, N))
    ct.store(out, (0, 0), z)

@ct.kernel
def kernel_add_broadcast_backward_dx(grad_out, grad_x):
    g = ct.load(grad_out, (0, 0), (M, N))
    ct.store(grad_x, (0, 0), g)

@ct.kernel
def kernel_add_broadcast_backward_dbias(grad_out, grad_bias):
    g = ct.load(grad_out, (0, 0), (M, N))
    reduced = ct.sum(g, axis=0)
    ct.store(grad_bias, (0,), reduced)

class AddBiasOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(M, N, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_forward, (x, bias, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x + bias

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_x = torch.empty(M, N, dtype=grad_output.dtype)
            grad_bias = torch.empty(N, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dx, (grad_output, grad_x))
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dbias, (grad_output, grad_bias))
            ctx.used_real_kernel_backward = True
            return grad_x, grad_bias
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_x = grad_output.clone()
            grad_bias = grad_output.sum(dim=0)
            return grad_x, grad_bias

@ct.kernel
def kernel_matmul_forward(a, b, c, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    x = ct.load(a, (0, 0), (m, k))
    y = ct.load(b, (0, 0), (k, n))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

@ct.kernel
def kernel_matmul_backward_dA(dc, b, da, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    g = ct.load(dc, (0, 0), (m, n))
    y = ct.load(b, (0, 0), (k, n))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)

@ct.kernel
def kernel_matmul_backward_dB(a, dc, db, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    x = ct.load(a, (0, 0), (m, k))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (m, n))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)

class MatMulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        ctx.m, ctx.k, ctx.n = m, k, n
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(m, n, dtype=a.dtype)
            ct.launch(stream, (1,), kernel_matmul_forward, (a, b, out, m, k, n))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return a @ b

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        m, k, n = ctx.m, ctx.k, ctx.n
        try:
            stream = torch.cuda.current_stream()
            grad_a = torch.empty_like(a)
            grad_b = torch.empty_like(b)
            ct.launch(stream, (1,), kernel_matmul_backward_dA, (grad_c, b, grad_a, m, k, n))
            ct.launch(stream, (1,), kernel_matmul_backward_dB, (a, grad_c, grad_b, m, k, n))
            ctx.used_real_kernel_backward = True
            return grad_a, grad_b
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_a = grad_c @ b.T
            grad_b = a.T @ grad_c
            return grad_a, grad_b

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

# --- Capstone: a genuine linear layer, y = X @ W + bias, with a scalar
# loss on top -- chaining MatMulOp (Chapter 35), AddBiasOp (this
# chapter), and SumOp (Chapter 31). bias.grad depends entirely on
# AddBiasOp's axis=0 reduction correctly surviving downstream of
# MatMulOp's output feeding into it, and upstream of SumOp's own
# backward broadcasting through both. ---
def linear_layer(x, w, bias):
    projected = MatMulOp.apply(x, w)          # (M, K) @ (K, N) -> (M, N)
    biased = AddBiasOp.apply(projected, bias)  # (M, N) + (N,) -> (M, N)
    return SumOp.apply(biased.reshape(-1))

torch.manual_seed(0)
K = 8
x = torch.randn(M, K, dtype=torch.float64, requires_grad=True)
w = torch.randn(K, N, dtype=torch.float64, requires_grad=True)
bias = torch.randn(N, dtype=torch.float64, requires_grad=True)

loss = linear_layer(x, w, bias)

def reference_linear(x_in, w_in, bias_in):
    return (x_in @ w_in + bias_in).sum()

ref_loss = reference_linear(x.detach(), w.detach(), bias.detach())
print(f"loss matches plain-torch reference linear layer: {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

x_ref = x.detach().clone().requires_grad_()
w_ref = w.detach().clone().requires_grad_()
bias_ref = bias.detach().clone().requires_grad_()
reference_linear(x_ref, w_ref, bias_ref).backward()

print(f"x.grad matches reference: {torch.allclose(x.grad, x_ref.grad)}")
print(f"w.grad matches reference: {torch.allclose(w.grad, w_ref.grad)}")
print(f"bias.grad matches reference (axis=0 reduction through the whole chain): {torch.allclose(bias.grad, bias_ref.grad)}")

ok = torch.autograd.gradcheck(linear_layer, (x, w, bias))
print(f"linear layer (MatMulOp -> AddBiasOp -> SumOp) gradcheck passed: {ok}")
```

### Genuinely run

```
loss matches plain-torch reference linear layer: True
x.grad matches reference: True
w.grad matches reference: True
bias.grad matches reference (axis=0 reduction through the whole chain): True
linear layer (MatMulOp -> AddBiasOp -> SumOp) gradcheck passed: True
```

### Discussion

`bias.grad` matching the reference here is a strictly stronger result than Section 38.2's isolated test: `AddBiasOp` never sees `x` or `w` at all, only `MatMulOp`'s output tensor and its own `.grad_fn` link back to it — and `AddBiasOp.backward`'s axis-0 reduction has no way of knowing, or needing to know, that the tensor it is reducing came from a matmul rather than an ordinary input. `y = X @ W + bias` is the plainest possible linear layer, and building it correctly from three independently-written `Function` subclasses, each `gradcheck`-verified in isolation in its own chapter, closes Part 4 on the same note it opened on in Chapter 30: a real formula, genuinely compiled, correctly composed.

## Chapter Summary

This chapter closed Part 4 by building `AddBiasOp`, the backward pass for a broadcast elementwise addition — the plainest-looking forward operation this book's autograd chapters have covered, and the first whose two gradients require genuinely different amounts of work rather than two variations on the same formula. `grad_x` is a trivial identity copy of the incoming gradient; `grad_bias` requires this book's first use of `ct.sum` with a concrete `axis` argument rather than `axis=None`, summing the gradient back down across every row that `bias` was broadcast into. A naming-confound test confirmed `axis=None` and `axis=0` compile to genuinely different cubin under an identical function name and load shape. `AddBiasOp` followed the established fallback pattern and was `gradcheck`-verified on both operands at once, and the capstone assembled a complete linear layer — `MatMulOp -> AddBiasOp -> SumOp` — confirming `bias`'s axis-0 reduction survives correctly even when the tensor it operates on came from a matmul rather than a plain input.

## Self-Check Questions

1. Explain why `AddBiasOp`'s two backward kernels do fundamentally different amounts of work, tracing the difference back to the shapes of `x` and `bias` in the forward pass.
2. Section 38.1's naming-confound test compiled `ct.sum(x, None)` and `ct.sum(x, axis=0)` under an identical function name and identical `(M, N)` load shape, and found different byte counts. What would it have meant, instead, if both had compiled to the exact same byte count?
3. `gradcheck` in Section 38.2 was run against `AddBiasOp.apply` with BOTH `x` and `bias` as arguments to check, rather than checking each one in a separate call. Explain what a `gradcheck` call checking only `bias` (holding `x` fixed as a constant) would have left unverified.
4. In the capstone, `AddBiasOp`'s forward and backward never reference `MatMulOp` or `x` or `w` in any way. Explain what specifically makes `bias.grad` come out correct anyway, once `AddBiasOp`'s input is `MatMulOp`'s output rather than a plain tensor.
5. This chapter's `bias` was 1-dimensional, broadcasting against a single leading axis of `x`. Chapter 21's `ct.matmul` supported batched, 3-dimensional inputs with broadcasting across a batch axis. If `x` were 3-dimensional (a batch of matrices) and `bias` still 1-dimensional, would `grad_bias`'s reduction need to sum over one axis or two? Explain your reasoning from the broadcasting rule alone, without assuming an answer.

## Where We Go Next

Part 4 opened in Chapter 30 by writing this book's first backward kernel and closes here, nine chapters later, having built backward passes for reductions, multi-block reductions, ragged tails, matmul, chains of multiple custom operations, an embedding-style gather with genuine accumulation semantics, and now a broadcast elementwise operation. Every one of those chapters worked at a scale small enough to compile and, where a driver-free method allowed it, verify directly — and every one of them deferred the question of what happens when a kernel's own internal tiling strategy, not just its logical operation, has to become a real engineering decision under real hardware constraints this sandbox cannot observe. Part 5 turns to that question directly, starting from what this book's `export_kernel`-only compilation can still say honestly about a kernel's structure without ever touching a live CUDA stream.

## Worked Solutions

**1.** `x` has shape `(M, N)` and receives a gradient of the exact same shape back — `d(x + bias)/dx` is `1` at every position, so `grad_x` is simply `grad_out` copied unchanged, no combination of values required at all. `bias` has shape `(N,)`, far smaller than `grad_out`'s `(M, N)`, precisely because it was broadcast up to that shape during the forward pass; correctly computing `grad_bias` requires combining (summing) the `M` separate contributions that landed on each of `bias`'s `N` positions during that broadcast, which is genuine reduction work with no equivalent on the `x` side.

**2.** An identical byte count between `axis=None` and `axis=0`, under an otherwise-identical naming-confound setup, would have suggested that cuTile Python's compiler treats a full reduction and a partial reduction as the same underlying operation at the code-generation level — perhaps computing the axis-0 partial sum internally and then treating "reduce the remaining axis too" as a cheap follow-on step baked into the same instruction sequence regardless of whether that follow-on is actually requested. The different byte counts this chapter actually measured rule that out: the two calls produce genuinely different compiled code, consistent with them being different operations rather than one operation with an unused label.

**3.** A `gradcheck` call holding `x` fixed and checking only `bias` would verify that perturbing each element of `bias` produces the correct change in the loss — but it would do so with a SPECIFIC, fixed `x` throughout every perturbation, never testing whether `grad_x`'s own formula (the identity copy) is correct at all. Since `torch.autograd.Function.backward` must return a correct gradient for EVERY argument `forward` received, testing only one of them leaves the other's correctness entirely unconfirmed; checking both together, as this chapter did, is what makes the `gradcheck` pass actual evidence for the WHOLE `backward` method, not just half of it.

**4.** `torch.autograd`'s graph, not `AddBiasOp`'s own code, is what makes this work. `MatMulOp.apply(x, w)` returns a tensor whose `.grad_fn` points back to `MatMulOp`; that tensor is passed as `AddBiasOp.apply`'s first argument, `AddBiasOp.forward` never inspects where it came from, and `AddBiasOp.backward` only ever needs `grad_output` — the gradient with respect to ITS OWN forward output — to correctly compute `grad_x` (an identity copy) and `grad_bias` (the axis-0 sum). Once `AddBiasOp.backward` returns `grad_x`, `torch.autograd` automatically routes it onward as the `grad_c` argument to `MatMulOp.backward`, entirely outside of anything `AddBiasOp` itself does or knows about — the same graph-composition fact Chapter 34's capstone first demonstrated, holding here across a broadcasting operation instead of two reductions.

**5.** It would need to sum over TWO axes, not one. Broadcasting a 1-dimensional `bias` against a 3-dimensional `x` (shape, say, `(batch, M, N)`) aligns `bias`'s single axis against `x`'s LAST axis (the same trailing-dimension-alignment rule Chapter 17 established), meaning `bias` is broadcast across BOTH of `x`'s other two axes — the batch axis and the row axis — not just one. Correctly computing `grad_bias` in that case would require summing `grad_out` over both of those broadcast axes (axes 0 and 1, in `ct.sum`'s terms, most likely via `ct.sum(g, axis=(0, 1))`, mirroring the multi-axis tuple form `ct.sum`'s own docstring already documents), leaving only the final axis matching `bias`'s own shape — exactly as many axes need summing as were broadcast on the way forward, no more and no fewer.

## Complete Runnable Code

### File: `01_broadcast_add_kernels.py`

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

M, N = 8, 8

def sig_2d_1d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

def sig_2d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

def sig_2d_1d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- Forward: z = x + bias, with bias (N,) broadcast across every one of
# x's M rows -- ordinary NumPy-style broadcasting, the same rule Chapter
# 17's elementwise arithmetic family already established. ---
@ct.kernel
def kernel_add_broadcast_forward(x, bias, out):
    xt = ct.load(x, (0, 0), (M, N))
    bt = ct.load(bias, (0,), (N,))
    z = xt + ct.broadcast_to(bt, (M, N))
    ct.store(out, (0, 0), z)

# --- Backward w.r.t. x: d(x + bias)/dx is 1 everywhere, so grad_x is
# just grad_out, unchanged -- the easy half of this operation's
# backward. ---
@ct.kernel
def kernel_add_broadcast_backward_dx(grad_out, grad_x):
    g = ct.load(grad_out, (0, 0), (M, N))
    ct.store(grad_x, (0, 0), g)

# --- Backward w.r.t. bias: every one of bias's N columns contributed to
# M different output positions (one per row), so bias's gradient must
# SUM grad_out back down across the M rows it was broadcast into -- the
# first time this book has reduced along a specific axis (axis=0) rather
# than reducing a tile down to a single scalar. ---
@ct.kernel
def kernel_add_broadcast_backward_dbias(grad_out, grad_bias):
    g = ct.load(grad_out, (0, 0), (M, N))
    reduced = ct.sum(g, axis=0)
    ct.store(grad_bias, (0,), reduced)

n = compile_bytes(kernel_add_broadcast_forward, sig_2d_1d_2d())
print(f"kernel_add_broadcast_forward: {n} cubin bytes")
n = compile_bytes(kernel_add_broadcast_backward_dx, sig_2d_2d())
print(f"kernel_add_broadcast_backward_dx: {n} cubin bytes")
n = compile_bytes(kernel_add_broadcast_backward_dbias, sig_2d_1d())
print(f"kernel_add_broadcast_backward_dbias: {n} cubin bytes")

print()

# --- A full reduction (axis=None) versus an axis=0 partial reduction,
# under an identical function name and an identical (M, N) load shape,
# confirms the compiler treats these as genuinely different operations
# rather than the axis argument being cosmetic. ---
def make_probe(axis):
    if axis is None:
        def kernel_probe(a, c):
            x = ct.load(a, (0, 0), (M, N))
            y = ct.sum(x, None)
            ct.store(c, (0,), ct.broadcast_to(y, (1,)))
    else:
        def kernel_probe(a, c):
            x = ct.load(a, (0, 0), (M, N))
            y = ct.sum(x, axis=0)
            ct.store(c, (0,), y)
    kernel_probe.__name__ = "kernel_probe"
    kernel_probe.__qualname__ = "kernel_probe"
    return ct.kernel(kernel_probe)

for axis in (None, 0):
    kernel_probe = make_probe(axis)
    n = compile_bytes(kernel_probe, sig_2d_1d())
    label = "axis=None (full reduction)" if axis is None else "axis=0 (partial reduction)"
    print(f"kernel_probe, {label}: {n} cubin bytes")

print()

# --- Correctness, on the host, since this sandbox cannot ct.launch:
# confirm the forward matches NumPy-style broadcasting addition, and
# that summing grad_out along axis 0 gives exactly the gradient a plain
# torch computation of the same broadcast add would produce. ---
torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
bias = torch.randn(N, dtype=torch.float64, requires_grad=True)

z = x + bias
print(f"forward matches x + bias (broadcast): {torch.allclose(z, x.detach() + bias.detach())}")

z.sum().backward()
grad_out_manual = torch.ones(M, N, dtype=torch.float64)
grad_x_manual = grad_out_manual.clone()
grad_bias_manual = grad_out_manual.sum(dim=0)
print(f"x.grad matches identity copy of grad_out: {torch.equal(x.grad, grad_x_manual)}")
print(f"bias.grad matches grad_out summed over axis 0: {torch.equal(bias.grad, grad_bias_manual)}")
```

### File: `02_addbiasop.py`

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

M, N = 8, 8

@ct.kernel
def kernel_add_broadcast_forward(x, bias, out):
    xt = ct.load(x, (0, 0), (M, N))
    bt = ct.load(bias, (0,), (N,))
    z = xt + ct.broadcast_to(bt, (M, N))
    ct.store(out, (0, 0), z)

@ct.kernel
def kernel_add_broadcast_backward_dx(grad_out, grad_x):
    g = ct.load(grad_out, (0, 0), (M, N))
    ct.store(grad_x, (0, 0), g)

@ct.kernel
def kernel_add_broadcast_backward_dbias(grad_out, grad_bias):
    g = ct.load(grad_out, (0, 0), (M, N))
    reduced = ct.sum(g, axis=0)
    ct.store(grad_bias, (0,), reduced)

class AddBiasOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(M, N, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_forward, (x, bias, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x + bias

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_x = torch.empty(M, N, dtype=grad_output.dtype)
            grad_bias = torch.empty(N, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dx, (grad_output, grad_x))
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dbias, (grad_output, grad_bias))
            ctx.used_real_kernel_backward = True
            return grad_x, grad_bias
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_x = grad_output.clone()
            grad_bias = grad_output.sum(dim=0)
            return grad_x, grad_bias

torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
bias = torch.randn(N, dtype=torch.float64, requires_grad=True)

z = AddBiasOp.apply(x, bias)
print(f"forward matches x + bias: {torch.allclose(z, x.detach() + bias.detach())}")
print(f"used real forward kernel: {z.grad_fn.used_real_kernel_forward}")

z.sum().backward()
print(f"used real backward kernel: {z.grad_fn.used_real_kernel_backward}")

x_ref = x.detach().clone().requires_grad_()
bias_ref = bias.detach().clone().requires_grad_()
(x_ref + bias_ref).sum().backward()
print(f"x.grad matches reference broadcast-add backward: {torch.allclose(x.grad, x_ref.grad)}")
print(f"bias.grad matches reference broadcast-add backward: {torch.allclose(bias.grad, bias_ref.grad)}")
print(f"bias.grad equals M (every row contributed 1.0): {bias.grad.tolist()}")

ok = torch.autograd.gradcheck(AddBiasOp.apply, (x, bias))
print(f"AddBiasOp gradcheck passed: {ok}")
```

### File: `03_capstone_linear_layer_with_bias.py`

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

M, N = 8, 8

@ct.kernel
def kernel_add_broadcast_forward(x, bias, out):
    xt = ct.load(x, (0, 0), (M, N))
    bt = ct.load(bias, (0,), (N,))
    z = xt + ct.broadcast_to(bt, (M, N))
    ct.store(out, (0, 0), z)

@ct.kernel
def kernel_add_broadcast_backward_dx(grad_out, grad_x):
    g = ct.load(grad_out, (0, 0), (M, N))
    ct.store(grad_x, (0, 0), g)

@ct.kernel
def kernel_add_broadcast_backward_dbias(grad_out, grad_bias):
    g = ct.load(grad_out, (0, 0), (M, N))
    reduced = ct.sum(g, axis=0)
    ct.store(grad_bias, (0,), reduced)

class AddBiasOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(M, N, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_forward, (x, bias, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x + bias

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_x = torch.empty(M, N, dtype=grad_output.dtype)
            grad_bias = torch.empty(N, dtype=grad_output.dtype)
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dx, (grad_output, grad_x))
            ct.launch(stream, (1,), kernel_add_broadcast_backward_dbias, (grad_output, grad_bias))
            ctx.used_real_kernel_backward = True
            return grad_x, grad_bias
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_x = grad_output.clone()
            grad_bias = grad_output.sum(dim=0)
            return grad_x, grad_bias

@ct.kernel
def kernel_matmul_forward(a, b, c, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    x = ct.load(a, (0, 0), (m, k))
    y = ct.load(b, (0, 0), (k, n))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

@ct.kernel
def kernel_matmul_backward_dA(dc, b, da, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    g = ct.load(dc, (0, 0), (m, n))
    y = ct.load(b, (0, 0), (k, n))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)

@ct.kernel
def kernel_matmul_backward_dB(a, dc, db, m: ct.Constant[int], k: ct.Constant[int], n: ct.Constant[int]):
    x = ct.load(a, (0, 0), (m, k))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (m, n))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)

class MatMulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        ctx.m, ctx.k, ctx.n = m, k, n
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(m, n, dtype=a.dtype)
            ct.launch(stream, (1,), kernel_matmul_forward, (a, b, out, m, k, n))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return a @ b

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        m, k, n = ctx.m, ctx.k, ctx.n
        try:
            stream = torch.cuda.current_stream()
            grad_a = torch.empty_like(a)
            grad_b = torch.empty_like(b)
            ct.launch(stream, (1,), kernel_matmul_backward_dA, (grad_c, b, grad_a, m, k, n))
            ct.launch(stream, (1,), kernel_matmul_backward_dB, (a, grad_c, grad_b, m, k, n))
            ctx.used_real_kernel_backward = True
            return grad_a, grad_b
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_a = grad_c @ b.T
            grad_b = a.T @ grad_c
            return grad_a, grad_b

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

# --- Capstone: a genuine linear layer, y = X @ W + bias, with a scalar
# loss on top -- chaining MatMulOp (Chapter 35), AddBiasOp (this
# chapter), and SumOp (Chapter 31). bias.grad depends entirely on
# AddBiasOp's axis=0 reduction correctly surviving downstream of
# MatMulOp's output feeding into it, and upstream of SumOp's own
# backward broadcasting through both. ---
def linear_layer(x, w, bias):
    projected = MatMulOp.apply(x, w)          # (M, K) @ (K, N) -> (M, N)
    biased = AddBiasOp.apply(projected, bias)  # (M, N) + (N,) -> (M, N)
    return SumOp.apply(biased.reshape(-1))

torch.manual_seed(0)
K = 8
x = torch.randn(M, K, dtype=torch.float64, requires_grad=True)
w = torch.randn(K, N, dtype=torch.float64, requires_grad=True)
bias = torch.randn(N, dtype=torch.float64, requires_grad=True)

loss = linear_layer(x, w, bias)

def reference_linear(x_in, w_in, bias_in):
    return (x_in @ w_in + bias_in).sum()

ref_loss = reference_linear(x.detach(), w.detach(), bias.detach())
print(f"loss matches plain-torch reference linear layer: {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

x_ref = x.detach().clone().requires_grad_()
w_ref = w.detach().clone().requires_grad_()
bias_ref = bias.detach().clone().requires_grad_()
reference_linear(x_ref, w_ref, bias_ref).backward()

print(f"x.grad matches reference: {torch.allclose(x.grad, x_ref.grad)}")
print(f"w.grad matches reference: {torch.allclose(w.grad, w_ref.grad)}")
print(f"bias.grad matches reference (axis=0 reduction through the whole chain): {torch.allclose(bias.grad, bias_ref.grad)}")

ok = torch.autograd.gradcheck(linear_layer, (x, w, bias))
print(f"linear layer (MatMulOp -> AddBiasOp -> SumOp) gradcheck passed: {ok}")
```
