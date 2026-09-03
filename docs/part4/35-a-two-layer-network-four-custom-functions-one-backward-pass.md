# Chapter 35: A Two-Layer Network — Four Custom Functions, One Backward Pass

> "Chapter 34 closed by chaining two custom operations into a scalar loss and asking `torch.autograd` to compose their backward passes with no help from either one's own code. This chapter asks that same question with twice as many links in the chain, and adds a wrinkle: one of matmul's two appearances now needs a different shape than the other, which means `MatMulOp` itself has to stop assuming it already knows what shape it's going to see."

**What you will understand by the end of this chapter:**

- Why `MatMulOp`, as built in Chapter 34, cannot serve a network with two differently shaped matmuls without change — and how making `m`, `k`, `n` compile-time `ct.Constant` parameters, read from the tensors' own `.shape` at call time, lets one kernel definition and one `Function` class serve both
- That ahead-of-time compilation specializes the exact same kernel FUNCTION separately for each distinct `(m, k, n)` triple, producing different cubin byte counts from identical source code — the same specialization behavior this book has confirmed for `ct.Constant` since it was first introduced, now shown for a three-constant signature
- `ReluOp`, a new `torch.autograd.Function` built from Chapter 30's `relu_forward_kernel` and `relu_backward_kernel` — kernels that chapter compiled but never wired into `autograd` — following the same real-launch-then-fallback pattern as every other custom `Function` in this book
- That `gradcheck`'s central-difference method requires inputs kept safely away from ReLU's non-differentiable point at zero, and how to construct a test input that guarantees this deterministically rather than relying on chance
- A full two-layer network — `MatMulOp -> ReluOp -> MatMulOp -> SumOp` — with `torch.autograd` correctly composing four independently written custom `Function` subclasses' backward passes into gradients matching a plain-`torch` reference network's own native backward, confirmed by both direct comparison and `gradcheck`

**What you need to know first:**

- Chapter 34's `MatMulOp` and its two-kernel backward (`dA = dC @ B^T`, `dB = A^T @ dC`), and Chapter 31's `SumOp`, both reused and extended in this chapter.
- Chapter 30's `relu_forward_kernel` and `relu_backward_kernel`, compiled in that chapter but not yet wrapped in a `Function` — this chapter does that for the first time.
- Chapter 30's `torch.autograd.Function` fallback pattern and Chapter 29's `gradcheck`-based verification method.
- How `ct.Constant` parameters cause ahead-of-time re-specialization, established the first time this book parameterized a kernel with a compile-time constant.

## 35.1 MatMulOp Becomes Shape-Generic

### Intuition

Chapter 34's `MatMulOp` hard-coded `A`'s shape as `(8, 16)` and `B`'s as `(16, 8)` directly into its three kernels' `ct.load` calls. A two-layer network needs two matmuls at two different shapes — the first layer's `(8, 16) @ (16, 8)` and the second layer's `(8, 8) @ (8, 8)` — and Chapter 34's kernels have no way to serve the second one. The fix is to stop hard-coding the shape at all: make `m`, `k`, `n` compile-time `ct.Constant[int]` parameters, the same mechanism `tile_size` has used throughout this book, and read their values from `a.shape` and `b.shape` inside `MatMulOp.forward` rather than assuming them in advance.

### Background

Making a kernel's tile shape a `ct.Constant` parameter is not new to this book — `tile_size` has done exactly this since Part 0. What's new here is doing it with three constants at once (`m`, `k`, `n`) across all three of matmul's kernels (forward, `backward_dA`, `backward_dB`) simultaneously, and confirming that compiling the identical kernel source at two different `(m, k, n)` triples produces two genuinely different cubins — the ahead-of-time specialization this book has relied on from the start, now exercised on a shape triple rather than a single block size.

### Worked Example 35.1.1 — The same three kernel functions, compiled at two different shapes

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

def sig_mkn(m, k, n):
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2),
         ct.compilation.ConstantConstraint(m), ct.compilation.ConstantConstraint(k), ct.compilation.ConstantConstraint(n)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- Chapter 34's MatMulOp hard-coded (8,16)x(16,8) into its kernels. A
# two-layer network needs a second matmul at a different shape -- (8,8)x
# (8,8) -- so this chapter makes the shape a compile-time ct.Constant
# triple (m, k, n) instead, letting one kernel definition serve both
# matmuls in the network. ---
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

# --- The network's first layer: X (8,16) @ W1 (16,8) -> H (8,8). ---
for name, kernel in [("kernel_matmul_forward", kernel_matmul_forward),
                      ("kernel_matmul_backward_dA", kernel_matmul_backward_dA),
                      ("kernel_matmul_backward_dB", kernel_matmul_backward_dB)]:
    n = compile_bytes(kernel, sig_mkn(8, 16, 8))
    print(f"{name}, shape (m=8, k=16, n=8): {n} cubin bytes")

print()

# --- The network's second layer: H (8,8) @ W2 (8,8) -> O (8,8). Same
# three kernel FUNCTIONS, a different ct.Constant triple -- ahead-of-time
# compilation specializes each shape separately, exactly as this book has
# confirmed every time it has varied a ct.Constant parameter. ---
for name, kernel in [("kernel_matmul_forward", kernel_matmul_forward),
                      ("kernel_matmul_backward_dA", kernel_matmul_backward_dA),
                      ("kernel_matmul_backward_dB", kernel_matmul_backward_dB)]:
    n = compile_bytes(kernel, sig_mkn(8, 8, 8))
    print(f"{name}, shape (m=8, k=8, n=8): {n} cubin bytes")
```

### Genuinely run

```
kernel_matmul_forward, shape (m=8, k=16, n=8): 32552 cubin bytes
kernel_matmul_backward_dA, shape (m=8, k=16, n=8): 31528 cubin bytes
kernel_matmul_backward_dB, shape (m=8, k=16, n=8): 31784 cubin bytes

kernel_matmul_forward, shape (m=8, k=8, n=8): 31272 cubin bytes
kernel_matmul_backward_dA, shape (m=8, k=8, n=8): 31528 cubin bytes
kernel_matmul_backward_dB, shape (m=8, k=8, n=8): 31528 cubin bytes
```

### Discussion

Each of the three kernel functions above has exactly one definition in this file, yet each produces two different byte counts depending only on which `(m, k, n)` triple its `KernelSignature` carries — `kernel_matmul_forward` compiles to `32552` bytes for `(8, 16, 8)` and `31272` bytes for `(8, 8, 8)`, a direct consequence of `export_kernel` specializing the kernel ahead of time for the exact shape it was asked to target, the same behavior this book relied on the first time it made a block size a `ct.Constant` rather than a runtime value. Two of the six byte counts happen to coincide (`kernel_matmul_backward_dA` at `(8, 16, 8)` and both backward kernels at `(8, 8, 8)` all print `31528`) — a coincidence of these particular shapes rather than a meaningful pattern, worth noting only so it isn't mistaken for a bug.

## 35.2 ReluOp: Wiring Chapter 30's Kernels Into autograd for the First Time

### Intuition

Chapter 30 compiled `relu_forward_kernel` and `relu_backward_kernel` — real, working cubin, byte-counted — but that chapter's point was the CuPy engine architecture surrounding them, not a `torch.autograd.Function`. Those two kernels have sat compiled but unwired ever since. `ReluOp` wires them, following the identical fallback pattern this book has used for every custom `Function` since Chapter 30 itself introduced it: attempt a real `ct.launch`, catch the `RuntimeError` this sandbox always raises, fall back to a plain-`torch` formula that mirrors the kernel exactly.

### Background

`gradcheck`'s central-difference method perturbs each input by a small `eps` (`1e-6` by default) in both directions and compares the resulting slope to the analytic gradient. ReLU's gradient is discontinuous at exactly zero, so an input with any element close enough to zero risks the perturbation landing on the wrong side of that discontinuity, which would make `gradcheck` fail for a reason that has nothing to do with whether `ReluOp`'s backward is correct. The input constructed for `ReluOp`'s own `gradcheck` call is built to rule this out deterministically — every element pushed at least `0.5` away from zero — rather than trusting that a random draw happens not to land too close.

### Worked Example 35.2.1 — MatMulOp, now shape-generic, and ReluOp arrives

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

# --- Chapter 30's reader-supplied relu kernels, reused verbatim -- but
# Chapter 30 only compiled them, it never wired them into a
# torch.autograd.Function. ReluOp does that for the first time here. ---
ConstInt = ct.Constant[int]

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

class ReluOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), relu_forward_kernel, (x, out, ctx.n))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), relu_backward_kernel, (grad_output, x, grad_input, ctx.n))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x > 0).to(x.dtype)
            return grad_output * mask

torch.manual_seed(0)

# --- MatMulOp is now shape-generic: the same class serves an (8,16)x
# (16,8) matmul and an (8,8)x(8,8) matmul without modification. ---
x1 = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
w1 = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
h = MatMulOp.apply(x1, w1)
print(f"MatMulOp (8,16)x(16,8) forward matches x1 @ w1: {torch.allclose(h, x1.detach() @ w1.detach())}")

x2 = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
w2 = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
o = MatMulOp.apply(x2, w2)
print(f"MatMulOp (8,8)x(8,8) forward matches x2 @ w2: {torch.allclose(o, x2.detach() @ w2.detach())}")

ok_matmul = torch.autograd.gradcheck(MatMulOp.apply, (x1, w1)) and torch.autograd.gradcheck(MatMulOp.apply, (x2, w2))
print(f"MatMulOp gradcheck passed at both shapes: {ok_matmul}")

# --- ReluOp on its own, before it joins the network. ---
z = torch.randn(64, dtype=torch.float64, requires_grad=True)
r = ReluOp.apply(z)
print(f"ReluOp forward matches torch.relu: {torch.allclose(r, torch.relu(z.detach()))}")
print(f"used real forward kernel: {r.grad_fn.used_real_kernel_forward}")

r.sum().backward()
print(f"used real backward kernel: {r.grad_fn.used_real_kernel_backward}")
print(f"z.grad matches (z > 0): {torch.equal(z.grad, (z.detach() > 0).to(z.dtype))}")

# gradcheck's central-difference method needs every input safely away
# from ReLU's non-differentiable point at zero, so this input is built to
# keep every element at least 0.5 away from zero in either direction.
z_gc = torch.rand(64, dtype=torch.float64) * 2 - 1
z_gc = (z_gc + torch.sign(z_gc) * 0.5).requires_grad_()
ok_relu = torch.autograd.gradcheck(ReluOp.apply, (z_gc,))
print(f"ReluOp gradcheck passed: {ok_relu}")
```

### Genuinely run

```
MatMulOp (8,16)x(16,8) forward matches x1 @ w1: True
MatMulOp (8,8)x(8,8) forward matches x2 @ w2: True
MatMulOp gradcheck passed at both shapes: True
ReluOp forward matches torch.relu: True
used real forward kernel: False
used real backward kernel: False
z.grad matches (z > 0): True
ReluOp gradcheck passed: True
```

### Discussion

`MatMulOp`'s two `gradcheck` calls, one at each shape, both pass without a single line of the class itself changing between them — `m`, `k`, `n` are read from `a.shape` and `b.shape` inside `forward` and threaded through `ctx` to `backward`, so the class adapts to whatever shape it's called with rather than assuming one in advance. `ReluOp` follows the exact fallback shape every custom `Function` in this book has followed since Chapter 30, and both its forward and backward fall into the `except RuntimeError` branch here, consistent with every other kernel this driver-less sandbox has tried to launch. The deliberately-constructed `z_gc` input is the more interesting line: it isn't there to make the test easier, it's there to make the test's own precondition (inputs away from ReLU's kink) hold by construction instead of by luck, which matters because a `gradcheck` failure caused by an unlucky near-zero draw would look identical to a real bug in `ReluOp.backward` without careful inspection.

## 35.3 Capstone: A Two-Layer Network, Four Functions, One Backward Pass

### Intuition

`MatMulOp -> ReluOp -> MatMulOp -> SumOp` is a minimal but genuine feedforward network: a linear layer, a nonlinearity, a second linear layer, and a scalar loss. Four custom `Function` subclasses, two of them (`MatMulOp`) called twice at two different shapes, need their backward passes composed correctly by `torch.autograd` with no code anywhere in this chapter that manually chains one `Function`'s gradient into the next.

### Background

The network is defined as an ordinary Python function, `two_layer_network`, calling `MatMulOp.apply`, `ReluOp.apply`, and `SumOp.apply` in sequence with plain `torch.Tensor.reshape` calls (not cuTile operations) bridging the 2D matmul outputs and the 1D `ReluOp`/`SumOp` kernels. `gradcheck` is called directly on `two_layer_network`, exercising the full four-`Function` graph with respect to all three leaf tensors — `x`, `w1`, and `w2` — at once.

### Worked Example 35.3.1 — A two-layer network, forward and backward, end to end

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

ConstInt = ct.Constant[int]

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

class ReluOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), relu_forward_kernel, (x, out, ctx.n))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), relu_backward_kernel, (grad_output, x, grad_input, ctx.n))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x > 0).to(x.dtype)
            return grad_output * mask

# Chapter 31's SumOp kernels, reused verbatim.
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

# --- Capstone: a two-layer feedforward network, forward and backward,
# built entirely from four independently written custom Functions --
# MatMulOp (twice, at two different shapes), ReluOp, and SumOp -- with no
# code in any one of them referencing any other. ---
def two_layer_network(x, w1, w2):
    h = MatMulOp.apply(x, w1)          # (8,16) @ (16,8) -> (8,8)
    h_relu = ReluOp.apply(h.reshape(64)).reshape(8, 8)
    o = MatMulOp.apply(h_relu, w2)     # (8,8) @ (8,8) -> (8,8)
    return SumOp.apply(o.reshape(64))

torch.manual_seed(0)
x = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
w1 = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
w2 = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

loss = two_layer_network(x, w1, w2)

def reference_network(x_in, w1_in, w2_in):
    h_ref = x_in @ w1_in
    h_relu_ref = torch.relu(h_ref)
    o_ref = h_relu_ref @ w2_in
    return o_ref.sum()

ref_loss = reference_network(x.detach(), w1.detach(), w2.detach())
print(f"loss matches plain-torch reference network: {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

x_ref = x.detach().clone().requires_grad_()
w1_ref = w1.detach().clone().requires_grad_()
w2_ref = w2.detach().clone().requires_grad_()
reference_network(x_ref, w1_ref, w2_ref).backward()

print(f"x.grad matches reference network's own backward: {torch.allclose(x.grad, x_ref.grad)}")
print(f"w1.grad matches reference network's own backward: {torch.allclose(w1.grad, w1_ref.grad)}")
print(f"w2.grad matches reference network's own backward: {torch.allclose(w2.grad, w2_ref.grad)}")

ok = torch.autograd.gradcheck(two_layer_network, (x, w1, w2))
print(f"two-layer network (MatMulOp -> ReluOp -> MatMulOp -> SumOp) gradcheck passed: {ok}")
```

### Genuinely run

```
loss matches plain-torch reference network: True
x.grad matches reference network's own backward: True
w1.grad matches reference network's own backward: True
w2.grad matches reference network's own backward: True
two-layer network (MatMulOp -> ReluOp -> MatMulOp -> SumOp) gradcheck passed: True
```

### Discussion

`reference_network` calls `torch`'s own built-in `@` and `torch.relu` directly, with no custom `Function` anywhere in it, and its `.backward()` uses `torch.autograd`'s native, battle-tested gradients as the standard the custom four-`Function` chain is measured against. Every one of `x.grad`, `w1.grad`, and `w2.grad` matches that reference exactly, and `gradcheck`'s independent central-difference method agrees. Nothing about `MatMulOp`, `ReluOp`, or `SumOp` was written with this specific network in mind — `MatMulOp` doesn't know it will be called twice at different shapes, `ReluOp` doesn't know its input came from a matmul, and `SumOp` doesn't know its input passed through a nonlinearity first. `torch.autograd`'s graph, built purely from the `.grad_fn` links `Function.apply` leaves behind at each step, is what makes a network assembled from independently authored pieces produce correct gradients without any of those pieces being aware of each other — the same structural fact Chapter 34 demonstrated with two `Function` subclasses, now shown holding across four.

## Chapter Summary

This chapter generalized `MatMulOp` from Chapter 34 into a shape-generic operation, replacing its hard-coded `(8, 16)`/`(16, 8)` shapes with compile-time `ct.Constant` parameters `m`, `k`, `n` read from the input tensors' own shapes — the same ahead-of-time specialization this book has relied on since `ct.Constant` was first introduced, now applied to a three-value shape triple rather than a single block size, and confirmed to produce genuinely different cubin byte counts per shape from identical kernel source. It introduced `ReluOp`, the first `torch.autograd.Function` built from Chapter 30's `relu_forward_kernel` and `relu_backward_kernel`, which that chapter compiled but never wired into `autograd`, following the established real-launch-then-fallback pattern and using a deliberately constructed input to keep `gradcheck` safely away from ReLU's non-differentiable point at zero. The capstone assembled a full two-layer feedforward network — `MatMulOp -> ReluOp -> MatMulOp -> SumOp` — from four independently written custom `Function` subclasses, none aware of the others' existence, and confirmed both by direct comparison against a plain-`torch` reference network's native backward and by `gradcheck` that `torch.autograd`'s own chain-rule bookkeeping composes all four correctly.

## Self-Check Questions

1. Chapter 34's `MatMulOp` hard-coded its shapes directly into `ct.load` calls inside its kernels. Explain specifically what about a two-layer network's structure makes that design insufficient, and why making `m`, `k`, `n` into `ct.Constant` parameters read from `.shape` fixes it.
2. Section 35.1's naming-adjacent test compiled the same three kernel functions at two different `(m, k, n)` triples and got different byte counts each time. Is this the same kind of naming-confound test Chapter 10 established, or a different kind of test? Explain the distinction.
3. Why does `gradcheck`, run on `ReluOp` alone, need its input to be constructed so that every element sits at least some fixed distance from zero, when `gradcheck` calls on `MatMulOp` and `SumOp` earlier in this book carried no equivalent requirement?
4. In the capstone, `torch.autograd.gradcheck(two_layer_network, (x, w1, w2))` checks gradients with respect to three separate leaf tensors at once, not one. Explain what would have to be true about `MatMulOp.backward`'s return value for `w1`'s gradient specifically to reach `w1.grad` correctly, tracing the path from `SumOp`'s single output backward through both matmuls.
5. Suppose `two_layer_network` had been written with `h_relu = torch.relu(h.reshape(64)).reshape(8, 8)` — the built-in `torch.relu` — instead of `ReluOp.apply(...)`, leaving `MatMulOp` and `SumOp` unchanged. Would `gradcheck` on the resulting function still pass? Would it still be testing this chapter's `ReluOp` backward kernel? Explain the difference between these two questions.

## Where We Go Next

This chapter's two-layer network still ran everything as a single tile per kernel launch — no grid of blocks, no tiling of the matmul across multiple thread blocks the way a production linear layer would need. Chapter 34 already named the open question of whether a matmul tiled across a grid needs multi-pass backward structure analogous to Chapter 32's multi-block max; this chapter's shape-generic `MatMulOp` makes that question slightly more concrete, since the same `ct.Constant`-driven specialization used here for shape would also need to specialize per-block index math if the matmul ever grew past one tile. Whether that question belongs to a later Part 4 chapter or waits for Part 5's performance work, as Chapter 34 suggested, remains open.

## Worked Solutions

**1.** A two-layer network needs two matmuls, `X @ W1` at `(8, 16) @ (16, 8)` and `H_relu @ W2` at `(8, 8) @ (8, 8)` — two different shapes. Chapter 34's `MatMulOp` baked its one shape directly into `ct.load(a, (0, 0), (8, 16))` and similar calls inside its kernels, so it could only ever serve one specific shape; calling it on the second layer's `(8, 8)` tensors would either fail outright or silently load the wrong number of elements. Making `m`, `k`, `n` into `ct.Constant[int]` parameters, and reading their actual values from `a.shape` and `b.shape` inside `forward`, removes the assumption entirely — the kernel functions become generic over shape, and `MatMulOp` can serve both layers of the network from one unmodified class.

**2.** It is a different kind of test, even though it looks similar on the surface. Chapter 10's naming confound holds the function name and the computation's INPUT identical, varying only the name, specifically to prove that the compiler is (or isn't) sensitive to superficial naming rather than genuine computation. Section 35.1's test holds the function name AND the source code identical, varying only which `ct.Constant` values the `KernelSignature` supplies — that isn't testing for a naming artifact at all, it's directly confirming ahead-of-time specialization: that `export_kernel` genuinely re-compiles a kernel differently for each distinct constant value, which is the expected, documented behavior of `ct.Constant`, not a confound to rule out.

**3.** `MatMulOp` and `SumOp` are both linear operations — matrix multiplication and summation — with gradients that are smooth (in fact locally exact, matching the analytic formula) everywhere, so a `gradcheck` perturbation of any size, in any direction, from any starting point, always sees the same local behavior. `ReluOp`'s gradient is a step function: `1` where the input is positive, `0` where it's negative, and undefined exactly at zero. If `gradcheck`'s perturbation straddles that boundary — the input sits within `eps` of zero — the finite-difference slope it computes reflects a mix of both sides, a value that neither branch of `ReluOp.backward`'s mask would produce, and the check would fail for a reason having nothing to do with whether the backward kernel is implemented correctly. Keeping every element at least `0.5` away from zero guarantees the tiny perturbation `gradcheck` uses can never cross that boundary.

**4.** `SumOp.backward` returns a gradient with respect to its own input, the flattened `O` tensor. That gradient must flow through `MatMulOp.apply(h_relu, w2)`'s backward as `grad_c`, which `MatMulOp.backward` uses to compute `grad_b = a.T @ grad_c` (here, `h_relu.T @ grad_c`) as the gradient with respect to `w2`, and `grad_a = grad_c @ b.T` as the gradient flowing further back into `h_relu`. That second gradient must then pass through `ReluOp.backward`'s mask, and the result becomes `grad_c` for the FIRST `MatMulOp.apply(x, w1)` call, whose `backward` computes `grad_b = x.T @ grad_c` as `w1`'s gradient. For `w1.grad` to come out correct, every one of those handoffs — `SumOp` to the second `MatMulOp`, that `MatMulOp` to `ReluOp`, `ReluOp` to the first `MatMulOp` — has to return exactly the gradient tensor the next `Function` in the chain expects as its own `grad_output` argument, with shapes lining up at each step; `torch.autograd`'s graph-walking machinery is what invokes each `backward` in the right order and feeds each one's return value into the next.

**5.** `gradcheck` would still pass — `torch.relu` has a correct, well-tested native backward, and neither `MatMulOp` nor `SumOp` changed, so the composed gradient would still be correct. But it would no longer be testing this chapter's `ReluOp` at all: `ReluOp.forward` and `ReluOp.backward`, along with the `relu_forward_kernel`/`relu_backward_kernel` pair they wrap, simply wouldn't be part of the computation graph `gradcheck` walks. The first question — does `gradcheck` pass — is about whether the OVERALL function computes correct gradients, regardless of which pieces compute them. The second question — is `ReluOp`'s backward being tested — is about whether THIS SPECIFIC code path was exercised at all. A passing `gradcheck` can never, by itself, answer which of several possible implementations actually ran; only reading the code (or, as in every custom `Function` in this book, checking `used_real_kernel_backward`, which would simply not exist on a plain `torch.relu` call) can answer that.

## Complete Runnable Code

### File: `01_shape_generic_matmul_kernels.py`

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

def sig_mkn(m, k, n):
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2),
         ct.compilation.ConstantConstraint(m), ct.compilation.ConstantConstraint(k), ct.compilation.ConstantConstraint(n)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- Chapter 34's MatMulOp hard-coded (8,16)x(16,8) into its kernels. A
# two-layer network needs a second matmul at a different shape -- (8,8)x
# (8,8) -- so this chapter makes the shape a compile-time ct.Constant
# triple (m, k, n) instead, letting one kernel definition serve both
# matmuls in the network. ---
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

# --- The network's first layer: X (8,16) @ W1 (16,8) -> H (8,8). ---
for name, kernel in [("kernel_matmul_forward", kernel_matmul_forward),
                      ("kernel_matmul_backward_dA", kernel_matmul_backward_dA),
                      ("kernel_matmul_backward_dB", kernel_matmul_backward_dB)]:
    n = compile_bytes(kernel, sig_mkn(8, 16, 8))
    print(f"{name}, shape (m=8, k=16, n=8): {n} cubin bytes")

print()

# --- The network's second layer: H (8,8) @ W2 (8,8) -> O (8,8). Same
# three kernel FUNCTIONS, a different ct.Constant triple -- ahead-of-time
# compilation specializes each shape separately, exactly as this book has
# confirmed every time it has varied a ct.Constant parameter. ---
for name, kernel in [("kernel_matmul_forward", kernel_matmul_forward),
                      ("kernel_matmul_backward_dA", kernel_matmul_backward_dA),
                      ("kernel_matmul_backward_dB", kernel_matmul_backward_dB)]:
    n = compile_bytes(kernel, sig_mkn(8, 8, 8))
    print(f"{name}, shape (m=8, k=8, n=8): {n} cubin bytes")
```

### File: `02_matmulop_becomes_shape_generic_and_reluop_arrives.py`

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

# --- Chapter 30's reader-supplied relu kernels, reused verbatim -- but
# Chapter 30 only compiled them, it never wired them into a
# torch.autograd.Function. ReluOp does that for the first time here. ---
ConstInt = ct.Constant[int]

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

class ReluOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), relu_forward_kernel, (x, out, ctx.n))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), relu_backward_kernel, (grad_output, x, grad_input, ctx.n))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x > 0).to(x.dtype)
            return grad_output * mask

torch.manual_seed(0)

# --- MatMulOp is now shape-generic: the same class serves an (8,16)x
# (16,8) matmul and an (8,8)x(8,8) matmul without modification. ---
x1 = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
w1 = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
h = MatMulOp.apply(x1, w1)
print(f"MatMulOp (8,16)x(16,8) forward matches x1 @ w1: {torch.allclose(h, x1.detach() @ w1.detach())}")

x2 = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
w2 = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
o = MatMulOp.apply(x2, w2)
print(f"MatMulOp (8,8)x(8,8) forward matches x2 @ w2: {torch.allclose(o, x2.detach() @ w2.detach())}")

ok_matmul = torch.autograd.gradcheck(MatMulOp.apply, (x1, w1)) and torch.autograd.gradcheck(MatMulOp.apply, (x2, w2))
print(f"MatMulOp gradcheck passed at both shapes: {ok_matmul}")

# --- ReluOp on its own, before it joins the network. ---
z = torch.randn(64, dtype=torch.float64, requires_grad=True)
r = ReluOp.apply(z)
print(f"ReluOp forward matches torch.relu: {torch.allclose(r, torch.relu(z.detach()))}")
print(f"used real forward kernel: {r.grad_fn.used_real_kernel_forward}")

r.sum().backward()
print(f"used real backward kernel: {r.grad_fn.used_real_kernel_backward}")
print(f"z.grad matches (z > 0): {torch.equal(z.grad, (z.detach() > 0).to(z.dtype))}")

# gradcheck's central-difference method needs every input safely away
# from ReLU's non-differentiable point at zero, so this input is built to
# keep every element at least 0.5 away from zero in either direction.
z_gc = torch.rand(64, dtype=torch.float64) * 2 - 1
z_gc = (z_gc + torch.sign(z_gc) * 0.5).requires_grad_()
ok_relu = torch.autograd.gradcheck(ReluOp.apply, (z_gc,))
print(f"ReluOp gradcheck passed: {ok_relu}")
```

### File: `03_capstone_a_two_layer_network_end_to_end.py`

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

ConstInt = ct.Constant[int]

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

class ReluOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), relu_forward_kernel, (x, out, ctx.n))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty_like(x)
            ct.launch(stream, (1,), relu_backward_kernel, (grad_output, x, grad_input, ctx.n))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x > 0).to(x.dtype)
            return grad_output * mask

# Chapter 31's SumOp kernels, reused verbatim.
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

# --- Capstone: a two-layer feedforward network, forward and backward,
# built entirely from four independently written custom Functions --
# MatMulOp (twice, at two different shapes), ReluOp, and SumOp -- with no
# code in any one of them referencing any other. ---
def two_layer_network(x, w1, w2):
    h = MatMulOp.apply(x, w1)          # (8,16) @ (16,8) -> (8,8)
    h_relu = ReluOp.apply(h.reshape(64)).reshape(8, 8)
    o = MatMulOp.apply(h_relu, w2)     # (8,8) @ (8,8) -> (8,8)
    return SumOp.apply(o.reshape(64))

torch.manual_seed(0)
x = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
w1 = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
w2 = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

loss = two_layer_network(x, w1, w2)

def reference_network(x_in, w1_in, w2_in):
    h_ref = x_in @ w1_in
    h_relu_ref = torch.relu(h_ref)
    o_ref = h_relu_ref @ w2_in
    return o_ref.sum()

ref_loss = reference_network(x.detach(), w1.detach(), w2.detach())
print(f"loss matches plain-torch reference network: {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

x_ref = x.detach().clone().requires_grad_()
w1_ref = w1.detach().clone().requires_grad_()
w2_ref = w2.detach().clone().requires_grad_()
reference_network(x_ref, w1_ref, w2_ref).backward()

print(f"x.grad matches reference network's own backward: {torch.allclose(x.grad, x_ref.grad)}")
print(f"w1.grad matches reference network's own backward: {torch.allclose(w1.grad, w1_ref.grad)}")
print(f"w2.grad matches reference network's own backward: {torch.allclose(w2.grad, w2_ref.grad)}")

ok = torch.autograd.gradcheck(two_layer_network, (x, w1, w2))
print(f"two-layer network (MatMulOp -> ReluOp -> MatMulOp -> SumOp) gradcheck passed: {ok}")
```
