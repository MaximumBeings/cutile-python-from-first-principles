# Chapter 34: MatMulOp — A Backward Pass Built From Two More Matmuls

> "Every custom backward kernel this book has written so far has been, in some sense, a mirror of its forward: a reduction's backward broadcasts, a mask's backward re-applies the same mask. This chapter's forward is `ct.matmul`, and its backward is also built entirely from `ct.matmul` — just called twice, on two different pairs of operands, with a transpose worked into each. The mirror still holds. It just needs a second reflection to see it."

**What you will understand by the end of this chapter:**

- Why a matmul forward, unlike every reduction this book has built a backward for, needs two genuinely different backward kernels rather than one kernel reused with reordered arguments — because `dA = dC @ B^T` and `dB = A^T @ dC` are two different matrix products with two different result shapes
- That `ct.transpose`, introduced in Chapter 22 as an operation that "only changes how a tile's own elements are arranged, addressed, and named," composes directly with `ct.matmul` inside a single kernel body, with no intermediate store or second kernel launch required
- A naming-confound test, in the tradition of Chapter 10, showing that an identically named, identically shaped kernel compiles to the exact same cubin byte count whether or not a `ct.transpose` call sits between the load and the `ct.matmul` — a compiled-level confirmation of what Chapter 22 already said in words
- How `MatMulOp`, a `torch.autograd.Function` whose forward and backward both attempt a real `ct.launch` before falling back to plain `torch`, reproduces `a @ b` forward and both of matmul's gradients backward, confirmed against `gradcheck`
- That `MatMulOp` composes with Chapter 31's `SumOp`, reused verbatim, into a scalar loss `sum(A @ B)` — and that `torch.autograd`'s own chain-rule bookkeeping links the two independently written `Function` subclasses with no code in either one that knows the other exists

**What you need to know first:**

- Chapter 21's `ct.matmul` (shape rules, dtype promotion) and Chapter 22's `ct.transpose` (a relabeling of a tile's own elements, not a memory operation).
- Chapter 30's `torch.autograd.Function` fallback pattern (attempt a real `ct.launch`, catch `RuntimeError`, fall back to a formula that mirrors the kernel exactly) and Chapter 29's `gradcheck`-based verification method.
- Chapter 31's `SumOp` and its kernels, which this chapter's capstone reuses without modification.
- Chapter 10's naming-confound discipline: two kernels compiled under the identical function name, with identical signatures and load shapes, isolate whether a byte-count difference reflects the computation or just the name.

## 34.1 Two Kernels for Two Gradients: dA = dC @ B^T, dB = A^T @ dC

### Intuition

For `C = A @ B`, ordinary calculus gives two gradient formulas, not one: `dA = dC @ B^T` and `dB = A^T @ dC`. Every backward kernel this book has written before this chapter — a reduction's broadcast, a mask's re-application — has been small enough, and symmetric enough, that one kernel (or one kernel per direction, differing only in which array plays which role) could carry the whole backward pass. Matmul's two gradients don't share that symmetry: `dA` has the same shape as `A`, `dB` has the same shape as `B`, and the two formulas transpose different operands. There is no way to write one kernel that computes both, or to call the same kernel twice with arguments merely reordered — `dC @ B^T` and `A^T @ dC` are genuinely different matrix products.

### Background

Both gradient formulas can be computed by kernels that do nothing this book hasn't already shown working separately: `ct.load`, `ct.transpose` (Chapter 22), and `ct.matmul` (Chapter 21), composed inside one kernel body. The forward kernel and both backward kernels stay deliberately small and single-tile — `A` is `(8, 16)`, `B` is `(16, 8)`, `C` is `(8, 8)` — the same scale Chapter 21 used for its own first `ct.matmul` example. Tiling a matmul backward pass across a grid of blocks the way a production GEMM would is a performance concern belonging to Part 5, not to this chapter's question of whether `torch.autograd` can be wired to a multi-kernel backward at all.

### Worked Example 34.1.1 — Compiling matmul's forward and both backward kernels

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

def sig_n(n_arrays):
    return ct.compilation.KernelSignature(
        [array_param(2) for _ in range(n_arrays)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- forward: C = A @ B, with A (8,16), B (16,8), C (8,8). ---
@ct.kernel
def kernel_matmul_forward(a, b, c):
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
n = compile_bytes(kernel_matmul_forward, sig_n(3))
print(f"kernel_matmul_forward: {n} cubin bytes")

# --- backward: dA = dC @ B^T, using ct.transpose on the loaded B tile
# rather than any array-level transpose -- the first time this book
# composes ct.matmul with an operation from its own permute/transpose
# chapter. ---
@ct.kernel
def kernel_matmul_backward_dA(dc, b, da):
    g = ct.load(dc, (0, 0), (8, 8))
    y = ct.load(b, (0, 0), (16, 8))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)
n = compile_bytes(kernel_matmul_backward_dA, sig_n(3))
print(f"kernel_matmul_backward_dA: {n} cubin bytes")

# --- backward: dB = A^T @ dC. ---
@ct.kernel
def kernel_matmul_backward_dB(a, dc, db):
    x = ct.load(a, (0, 0), (8, 16))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (8, 8))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)
n = compile_bytes(kernel_matmul_backward_dB, sig_n(3))
print(f"kernel_matmul_backward_dB: {n} cubin bytes")

print()

# --- naming-confound test: does compiling matmul(x, y) versus
# matmul(transpose(x), y) under the IDENTICAL function name, IDENTICAL
# square load shapes, and an IDENTICAL signature produce the same
# cubin, or does the transpose show up in the byte count? ---
def make_probe(use_transpose):
    if use_transpose:
        def kernel_probe(x_arr, y_arr, out):
            x = ct.load(x_arr, (0, 0), (8, 8))
            y = ct.load(y_arr, (0, 0), (8, 8))
            xt = ct.transpose(x)
            z = ct.matmul(xt, y)
            ct.store(out, (0, 0), z)
    else:
        def kernel_probe(x_arr, y_arr, out):
            x = ct.load(x_arr, (0, 0), (8, 8))
            y = ct.load(y_arr, (0, 0), (8, 8))
            z = ct.matmul(x, y)
            ct.store(out, (0, 0), z)
    kernel_probe.__name__ = "kernel_probe"
    kernel_probe.__qualname__ = "kernel_probe"
    return ct.kernel(kernel_probe)

for use_transpose in (False, True):
    kernel_probe = make_probe(use_transpose)
    n = compile_bytes(kernel_probe, sig_n(3))
    label = "with transpose(x)" if use_transpose else "plain matmul(x, y)"
    print(f"kernel_probe, {label}: {n} cubin bytes")
```

### Genuinely run

```
kernel_matmul_forward: 31912 cubin bytes
kernel_matmul_backward_dA: 30888 cubin bytes
kernel_matmul_backward_dB: 31272 cubin bytes

kernel_probe, plain matmul(x, y): 30376 cubin bytes
kernel_probe, with transpose(x): 30376 cubin bytes
```

### Discussion

Three genuinely different kernels compile to three genuinely different byte counts — unsurprising, since they load different shapes and store different results. The more interesting result is the last two lines. `kernel_probe`, compiled once with a plain `ct.matmul(x, y)` body and once with `ct.matmul(ct.transpose(x), y)`, under the identical function name, identical `(8, 8)` load shapes on both operands, and the identical three-array signature, produces the exact same byte count: `30376` both times. Chapter 22 described `ct.transpose` in words as an operation that changes only "how a tile's own elements are arranged, addressed, and named" rather than moving anything through memory. This is the first time this book has tested that description at the level of a compiled artifact rather than a docstring, and by Chapter 10's naming-confound standard — same name, same shapes, same signature — the result is a real confirmation: whatever `ct.transpose` costs, it costs nothing that shows up in this kernel's cubin size once it's fused into a following `ct.matmul`. That is one data point at one shape, not a universal claim about every shape or every operation composed with `transpose` — but it is a genuine, reproducible one.

## 34.2 MatMulOp: Wiring the Two-Kernel Backward Into autograd.Function

### Intuition

Every custom `Function` this book has built since Chapter 30 follows the same shape: `forward` attempts a real `ct.launch`, catches the `RuntimeError` this driver-less sandbox always raises, and falls back to a plain-`torch` formula that mirrors the kernel exactly; `backward` does the same. `MatMulOp` is the first `Function` whose `backward` needs to launch two kernels instead of one — `kernel_matmul_backward_dA` and `kernel_matmul_backward_dB` — because, as Section 34.1 established, matmul's two gradients cannot share a kernel.

### Background

`ctx.save_for_backward(a, b)` saves both original operands, exactly as `gradcheck`-verified `Function` subclasses have throughout Part 3 and Part 4; `backward` reads them back via `ctx.saved_tensors` and feeds them, along with the incoming `grad_c`, into the same two-kernel structure Worked Example 34.1.1 just compiled.

### Worked Example 34.2.1 — MatMulOp: a complete two-kernel backward Function

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

def sig_n(n_arrays):
    return ct.compilation.KernelSignature(
        [array_param(2) for _ in range(n_arrays)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_matmul_forward(a, b, c):
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

@ct.kernel
def kernel_matmul_backward_dA(dc, b, da):
    g = ct.load(dc, (0, 0), (8, 8))
    y = ct.load(b, (0, 0), (16, 8))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)

@ct.kernel
def kernel_matmul_backward_dB(a, dc, db):
    x = ct.load(a, (0, 0), (8, 16))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (8, 8))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)

class MatMulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(8, 8, dtype=a.dtype)
            ct.launch(stream, (1,), kernel_matmul_forward, (a, b, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return a @ b

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_a = torch.empty_like(a)
            grad_b = torch.empty_like(b)
            ct.launch(stream, (1,), kernel_matmul_backward_dA, (grad_c, b, grad_a))
            ct.launch(stream, (1,), kernel_matmul_backward_dB, (a, grad_c, grad_b))
            ctx.used_real_kernel_backward = True
            return grad_a, grad_b
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_a = grad_c @ b.T
            grad_b = a.T @ grad_c
            return grad_a, grad_b

torch.manual_seed(0)
a = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
b = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)

c = MatMulOp.apply(a, b)
print(f"forward matches a @ b: {torch.allclose(c, a.detach() @ b.detach())}")
print(f"used real forward kernel: {c.grad_fn.used_real_kernel_forward}")

c.sum().backward()
print(f"used real backward kernel: {c.grad_fn.used_real_kernel_backward}")
print(f"a.grad matches dC @ b.T: {torch.allclose(a.grad, torch.ones(8, 8, dtype=torch.float64) @ b.detach().T)}")
print(f"b.grad matches a.T @ dC: {torch.allclose(b.grad, a.detach().T @ torch.ones(8, 8, dtype=torch.float64))}")

ok = torch.autograd.gradcheck(MatMulOp.apply, (a, b))
print(f"MatMulOp gradcheck passed: {ok}")
```

### Genuinely run

```
forward matches a @ b: True
used real forward kernel: False
used real backward kernel: False
a.grad matches dC @ b.T: True
b.grad matches a.T @ dC: True
MatMulOp gradcheck passed: True
```

### Discussion

`used real forward kernel` and `used real backward kernel` both print `False`, exactly as every custom `Function` in this driver-less sandbox has since Chapter 30 — `torch.cuda.current_stream()` fails before `ct.launch` is ever reached, so both branches fall back to the plain-`torch` formulas. What's new here is that `backward`'s fallback branch has two lines instead of one, `grad_a = grad_c @ b.T` and `grad_b = a.T @ grad_c`, mirroring the two-kernel structure the real, hardware path would take. `gradcheck` verifies that this fallback pair, not the uncompiled kernel pair, computes the mathematically correct gradient — the same limitation this book has named plainly since Chapter 31: `gradcheck` in this sandbox never touches `ct.launch`, so it verifies the formula, and Section 34.1's `export_kernel` compilation is what verifies the kernels exist and compile, and the two kinds of evidence stay explicitly separate rather than blurred into one.

## 34.3 Capstone: Chaining MatMulOp Into a Scalar Loss

### Intuition

A matmul rarely stands alone in a real training loop — it is typically one step feeding a scalar loss that `.backward()` is called on. `sum(A @ B)` is a minimal version of that pattern: `MatMulOp`'s forward output feeds directly into Chapter 31's `SumOp`, and a single `.backward()` call must correctly compose both custom `Function` subclasses' backward methods with no code in either one that references the other.

### Background

`SumOp`'s kernels operate on a 1D tile, so `MatMulOp`'s `(8, 8)` output is reshaped to `(64,)` — an ordinary `torch.Tensor.reshape`, done outside any kernel, not a cuTile operation — before being handed to `SumOp.apply`. `torch.autograd` tracks both `Function` calls on its own graph; `chained_loss`'s `gradcheck` call exercises that composed graph end to end.

### Worked Example 34.3.1 — Chaining MatMulOp into a scalar loss

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

def sig_n(n_arrays, ndim=2):
    return ct.compilation.KernelSignature(
        [array_param(ndim) for _ in range(n_arrays)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_matmul_forward(a, b, c):
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

@ct.kernel
def kernel_matmul_backward_dA(dc, b, da):
    g = ct.load(dc, (0, 0), (8, 8))
    y = ct.load(b, (0, 0), (16, 8))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)

@ct.kernel
def kernel_matmul_backward_dB(a, dc, db):
    x = ct.load(a, (0, 0), (8, 16))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (8, 8))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)

class MatMulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(8, 8, dtype=a.dtype)
            ct.launch(stream, (1,), kernel_matmul_forward, (a, b, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return a @ b

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_a = torch.empty_like(a)
            grad_b = torch.empty_like(b)
            ct.launch(stream, (1,), kernel_matmul_backward_dA, (grad_c, b, grad_a))
            ct.launch(stream, (1,), kernel_matmul_backward_dB, (a, grad_c, grad_b))
            ctx.used_real_kernel_backward = True
            return grad_a, grad_b
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_a = grad_c @ b.T
            grad_b = a.T @ grad_c
            return grad_a, grad_b

# Chapter 31's SumOp kernels, reused verbatim -- the same full-reduction
# kernel that closed out the previous reduction chapters now sits
# downstream of a matmul instead of an elementwise op.
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

# --- Capstone: loss = sum(A @ B), a genuinely common pattern -- a matmul
# feeding a scalar loss -- chaining Chapter 34's new MatMulOp into
# Chapter 31's SumOp. autograd.backward() must compose MatMulOp's
# two-matmul backward with SumOp's broadcast backward with no help from
# either Function about the other's existence. ---
torch.manual_seed(0)
a = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
b = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)

c = MatMulOp.apply(a, b)
loss = SumOp.apply(c.reshape(64))

ref_loss = (a.detach() @ b.detach()).sum()
print(f"loss matches (a @ b).sum(): {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

ref_a = a.detach().clone().requires_grad_()
ref_b = b.detach().clone().requires_grad_()
(ref_a @ ref_b).sum().backward()
print(f"a.grad matches autograd's own (a @ b).sum() backward: {torch.allclose(a.grad, ref_a.grad)}")
print(f"b.grad matches autograd's own (a @ b).sum() backward: {torch.allclose(b.grad, ref_b.grad)}")

def chained_loss(a_in, b_in):
    c_in = MatMulOp.apply(a_in, b_in)
    return SumOp.apply(c_in.reshape(64))

ok = torch.autograd.gradcheck(chained_loss, (a, b))
print(f"chained MatMulOp -> SumOp gradcheck passed: {ok}")
```

### Genuinely run

```
loss matches (a @ b).sum(): True
a.grad matches autograd's own (a @ b).sum() backward: True
b.grad matches autograd's own (a @ b).sum() backward: True
chained MatMulOp -> SumOp gradcheck passed: True
```

### Discussion

Nothing in `MatMulOp` mentions `SumOp`, and nothing in `SumOp` mentions `MatMulOp` — the only thing connecting them is that one's forward output, reshaped, becomes the other's forward input, on an ordinary `torch.Tensor` that carries a `.grad_fn` pointer back to whichever `Function` produced it. That is enough: `loss.backward()` walks the graph `torch.autograd` built from those `.grad_fn` links, calling `SumOp.backward` first and feeding its returned gradient straight into `MatMulOp.backward` as that call's `grad_c` argument, with `torch.autograd` itself doing every bit of that wiring. The cross-check against `ref_a.grad` and `ref_b.grad` — computed by calling built-in `torch` matmul and sum directly and letting `torch`'s own native backward run — confirms the composed custom path lands on the exact values a fully native computation would, and `gradcheck`'s central-difference test on the whole `chained_loss` function confirms the same thing by an entirely independent numerical method.

## Chapter Summary

This chapter built `MatMulOp`, the first custom `torch.autograd.Function` in this book whose backward pass requires two structurally different kernels rather than one. `dA = dC @ B^T` and `dB = A^T @ dC` are two different matrix products, each composing `ct.matmul` (Chapter 21) with `ct.transpose` (Chapter 22) inside a single kernel body — the first time this book has combined operations from two different earlier chapters inside one backward kernel. A naming-confound test, following Chapter 10's discipline, found that an identically named, identically shaped kernel compiles to the exact same cubin byte count whether or not a `ct.transpose` call appears before the `ct.matmul` — a compiled-level confirmation of Chapter 22's own description of transpose as a pure relabeling operation. `MatMulOp` itself followed the established fallback pattern from Chapter 30 onward, its forward and backward both attempting real `ct.launch` calls before falling back to plain-`torch` formulas that `gradcheck` confirmed correct. The capstone chained `MatMulOp` into Chapter 31's `SumOp`, reused verbatim, to build a scalar loss `sum(A @ B)`, and confirmed that `torch.autograd`'s own chain-rule bookkeeping — not any code shared between the two `Function` subclasses — is what correctly composes their two independently written backward passes.

## Self-Check Questions

1. Why does `MatMulOp` need two separate backward kernels rather than one kernel reused with reordered arguments, in contrast to `SumOp`'s single backward kernel from Chapter 31?
2. The naming-confound test in Section 34.1 found identical byte counts for `matmul(x, y)` and `matmul(transpose(x), y)` under an identical kernel name and identical shapes. What does this result confirm about Chapter 22's own description of what `ct.transpose` does to a tile?
3. `kernel_matmul_backward_dA` transposes `B`, not `dC`. Trace through the shapes involved in `dA = dC @ B^T` and explain why the transpose has to land on `B` specifically, not on either of the other two arrays in that kernel.
4. In the capstone, `gradcheck` passed on the chained `MatMulOp -> SumOp` computation without either `Function`'s code referencing the other by name. Explain, structurally, what actually links the two `Function` subclasses together at runtime, and why that link is enough for `torch.autograd` to compose their backward passes correctly.
5. Suppose `kernel_matmul_backward_dA` had a bug and computed `dc @ b` instead of `dc @ transpose(b)`. Would `gradcheck`, run in this driver-less sandbox, catch that bug? Explain why or why not, tying your answer to which code path `gradcheck` actually exercises here, as this book has established since Chapter 31.

## Where We Go Next

This chapter's matmul stayed deliberately small and single-tile — one block, fixed shapes, no grid of blocks to coordinate. Chapter 32 already showed that a reduction spanning multiple blocks needs real inter-block coordination (atomics, or a second reduction pass) that a single-tile reduction never has to think about. A tiled matmul spanning a grid of blocks, the way Part 5's performance chapters will eventually need, raises the same question for `ct.matmul`'s backward: does `dA = dC @ B^T` tiled across a grid need its own multi-pass structure analogous to Chapter 32's multi-block max, or does matmul's accumulate-then-store structure sidestep that problem the way sum's `atomic_add` did? That question, along with whatever Part 4 chapter comes after it, is still open.

## Worked Solutions

**1.** `SumOp`'s backward is a single broadcast: the incoming scalar gradient gets copied out to every position of the input, and that same operation works regardless of which "direction" you think of it as running — one kernel suffices. `MatMulOp`'s two gradients, `dA = dC @ B^T` (shape matching `A`) and `dB = A^T @ dC` (shape matching `B`), are two different matrix products with two different operand pairings and two different result shapes; there is no reordering of one kernel's arguments that turns it into the other, so each gradient needs its own kernel.

**2.** Chapter 22 described `ct.transpose` as changing only "how a tile's own elements are arranged, addressed, and named" — not moving anything through memory. The naming-confound test in Section 34.1 tested that description at the level of a compiled artifact for the first time: `kernel_probe` compiled to the identical `30376` byte count whether its body called `ct.matmul(x, y)` or `ct.matmul(ct.transpose(x), y)`, under the same function name and the same load shapes. That identical byte count is a genuine, reproducible confirmation that, at least for this shape and this composition with `ct.matmul`, the transpose costs nothing that shows up in the compiled cubin — consistent with Chapter 22's claim, now backed by a compile-time measurement rather than only a docstring.

**3.** `dC` has shape `(8, 8)`, matching `C`; `B` has shape `(16, 8)`; the target `dA` must have shape `(8, 16)`, matching `A`. `dC @ B^T` needs `B^T` to have shape `(8, 16)` so that `dC`'s `(8, 8)` can matmul against it to produce `(8, 16)` — and `B^T` only has that shape if `B`, shape `(16, 8)`, is the one being transposed. Transposing `dC` instead would produce a `(8, 8)` tile (no shape change at all, since it's square) that doesn't fix the underlying mismatch, and transposing the result after the fact would require an entirely different, unbroadcastable arrangement of the multiply. The transpose has to land on `B` because `B` is the only one of the two operands whose transposed shape actually lines up with what `dC`'s matmul needs to reach `A`'s shape.

**4.** The link is a `torch.Tensor`'s `.grad_fn` attribute. When `MatMulOp.apply(a, b)` returns `c`, `c.grad_fn` points back to `MatMulOp`'s backward machinery; when `c.reshape(64)` is passed into `SumOp.apply`, the reshaped tensor still carries a connection back through `c` to that same `grad_fn`. `torch.autograd` builds its computation graph purely from these `grad_fn` links as `Function.apply` calls happen, with no participation from either `Function`'s own source code — `MatMulOp` never imports or calls `SumOp`, and vice versa. When `.backward()` is called on the final scalar, `torch.autograd` walks that graph in reverse, calling each node's `backward` method and feeding its returned gradient tensor(s) as the next node's incoming gradient argument. That graph-walking is entirely `torch.autograd`'s responsibility, which is exactly why two independently authored `Function` subclasses compose correctly without either one being written with the other in mind.

**5.** No — `gradcheck`, run in this driver-less sandbox, would not catch that bug. As established since Chapter 31, `gradcheck` in this environment never reaches a real `ct.launch` call at all, because `torch.cuda.current_stream()` raises `RuntimeError` before any kernel launch is attempted; every `Function` this book has built always falls into its `except RuntimeError` fallback branch here, and `gradcheck` only ever exercises that fallback formula (`grad_c @ b.T`, written directly in Python), never the kernel source `kernel_matmul_backward_dA` actually contains. A bug placed specifically inside the kernel body — `ct.matmul(g, y)` instead of `ct.matmul(g, ct.transpose(y))` — lives entirely in code `gradcheck` never runs on this machine. Only the compile-time evidence from Section 34.1 (confirming the kernel compiles as written) and, eventually, a real-hardware test of `ct.launch` itself could catch a bug of that specific kind — exactly the same verification gap this book named plainly for `padding_mode` in Chapter 33.

## Complete Runnable Code

### File: `01_matmul_forward_and_backward_kernels.py`

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

def sig_n(n_arrays):
    return ct.compilation.KernelSignature(
        [array_param(2) for _ in range(n_arrays)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- forward: C = A @ B, with A (8,16), B (16,8), C (8,8). ---
@ct.kernel
def kernel_matmul_forward(a, b, c):
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)
n = compile_bytes(kernel_matmul_forward, sig_n(3))
print(f"kernel_matmul_forward: {n} cubin bytes")

# --- backward: dA = dC @ B^T, using ct.transpose on the loaded B tile
# rather than any array-level transpose -- the first time this book
# composes ct.matmul with an operation from its own permute/transpose
# chapter. ---
@ct.kernel
def kernel_matmul_backward_dA(dc, b, da):
    g = ct.load(dc, (0, 0), (8, 8))
    y = ct.load(b, (0, 0), (16, 8))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)
n = compile_bytes(kernel_matmul_backward_dA, sig_n(3))
print(f"kernel_matmul_backward_dA: {n} cubin bytes")

# --- backward: dB = A^T @ dC. ---
@ct.kernel
def kernel_matmul_backward_dB(a, dc, db):
    x = ct.load(a, (0, 0), (8, 16))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (8, 8))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)
n = compile_bytes(kernel_matmul_backward_dB, sig_n(3))
print(f"kernel_matmul_backward_dB: {n} cubin bytes")

print()

# --- naming-confound test: does compiling matmul(x, y) versus
# matmul(transpose(x), y) under the IDENTICAL function name, IDENTICAL
# square load shapes, and an IDENTICAL signature produce the same
# cubin, or does the transpose show up in the byte count? ---
def make_probe(use_transpose):
    if use_transpose:
        def kernel_probe(x_arr, y_arr, out):
            x = ct.load(x_arr, (0, 0), (8, 8))
            y = ct.load(y_arr, (0, 0), (8, 8))
            xt = ct.transpose(x)
            z = ct.matmul(xt, y)
            ct.store(out, (0, 0), z)
    else:
        def kernel_probe(x_arr, y_arr, out):
            x = ct.load(x_arr, (0, 0), (8, 8))
            y = ct.load(y_arr, (0, 0), (8, 8))
            z = ct.matmul(x, y)
            ct.store(out, (0, 0), z)
    kernel_probe.__name__ = "kernel_probe"
    kernel_probe.__qualname__ = "kernel_probe"
    return ct.kernel(kernel_probe)

for use_transpose in (False, True):
    kernel_probe = make_probe(use_transpose)
    n = compile_bytes(kernel_probe, sig_n(3))
    label = "with transpose(x)" if use_transpose else "plain matmul(x, y)"
    print(f"kernel_probe, {label}: {n} cubin bytes")
```

### File: `02_matmulop_a_backward_built_from_two_more_matmuls.py`

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

def sig_n(n_arrays):
    return ct.compilation.KernelSignature(
        [array_param(2) for _ in range(n_arrays)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_matmul_forward(a, b, c):
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

@ct.kernel
def kernel_matmul_backward_dA(dc, b, da):
    g = ct.load(dc, (0, 0), (8, 8))
    y = ct.load(b, (0, 0), (16, 8))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)

@ct.kernel
def kernel_matmul_backward_dB(a, dc, db):
    x = ct.load(a, (0, 0), (8, 16))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (8, 8))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)

class MatMulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(8, 8, dtype=a.dtype)
            ct.launch(stream, (1,), kernel_matmul_forward, (a, b, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return a @ b

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_a = torch.empty_like(a)
            grad_b = torch.empty_like(b)
            ct.launch(stream, (1,), kernel_matmul_backward_dA, (grad_c, b, grad_a))
            ct.launch(stream, (1,), kernel_matmul_backward_dB, (a, grad_c, grad_b))
            ctx.used_real_kernel_backward = True
            return grad_a, grad_b
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_a = grad_c @ b.T
            grad_b = a.T @ grad_c
            return grad_a, grad_b

torch.manual_seed(0)
a = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
b = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)

c = MatMulOp.apply(a, b)
print(f"forward matches a @ b: {torch.allclose(c, a.detach() @ b.detach())}")
print(f"used real forward kernel: {c.grad_fn.used_real_kernel_forward}")

c.sum().backward()
print(f"used real backward kernel: {c.grad_fn.used_real_kernel_backward}")
print(f"a.grad matches dC @ b.T: {torch.allclose(a.grad, torch.ones(8, 8, dtype=torch.float64) @ b.detach().T)}")
print(f"b.grad matches a.T @ dC: {torch.allclose(b.grad, a.detach().T @ torch.ones(8, 8, dtype=torch.float64))}")

ok = torch.autograd.gradcheck(MatMulOp.apply, (a, b))
print(f"MatMulOp gradcheck passed: {ok}")
```

### File: `03_capstone_chaining_matmulop_into_a_scalar_loss.py`

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

def sig_n(n_arrays, ndim=2):
    return ct.compilation.KernelSignature(
        [array_param(ndim) for _ in range(n_arrays)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_matmul_forward(a, b, c):
    x = ct.load(a, (0, 0), (8, 16))
    y = ct.load(b, (0, 0), (16, 8))
    z = ct.matmul(x, y)
    ct.store(c, (0, 0), z)

@ct.kernel
def kernel_matmul_backward_dA(dc, b, da):
    g = ct.load(dc, (0, 0), (8, 8))
    y = ct.load(b, (0, 0), (16, 8))
    yt = ct.transpose(y)
    result = ct.matmul(g, yt)
    ct.store(da, (0, 0), result)

@ct.kernel
def kernel_matmul_backward_dB(a, dc, db):
    x = ct.load(a, (0, 0), (8, 16))
    xt = ct.transpose(x)
    g = ct.load(dc, (0, 0), (8, 8))
    result = ct.matmul(xt, g)
    ct.store(db, (0, 0), result)

class MatMulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty(8, 8, dtype=a.dtype)
            ct.launch(stream, (1,), kernel_matmul_forward, (a, b, out))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return a @ b

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            grad_a = torch.empty_like(a)
            grad_b = torch.empty_like(b)
            ct.launch(stream, (1,), kernel_matmul_backward_dA, (grad_c, b, grad_a))
            ct.launch(stream, (1,), kernel_matmul_backward_dB, (a, grad_c, grad_b))
            ctx.used_real_kernel_backward = True
            return grad_a, grad_b
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_a = grad_c @ b.T
            grad_b = a.T @ grad_c
            return grad_a, grad_b

# Chapter 31's SumOp kernels, reused verbatim -- the same full-reduction
# kernel that closed out the previous reduction chapters now sits
# downstream of a matmul instead of an elementwise op.
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

# --- Capstone: loss = sum(A @ B), a genuinely common pattern -- a matmul
# feeding a scalar loss -- chaining Chapter 34's new MatMulOp into
# Chapter 31's SumOp. autograd.backward() must compose MatMulOp's
# two-matmul backward with SumOp's broadcast backward with no help from
# either Function about the other's existence. ---
torch.manual_seed(0)
a = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
b = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)

c = MatMulOp.apply(a, b)
loss = SumOp.apply(c.reshape(64))

ref_loss = (a.detach() @ b.detach()).sum()
print(f"loss matches (a @ b).sum(): {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

ref_a = a.detach().clone().requires_grad_()
ref_b = b.detach().clone().requires_grad_()
(ref_a @ ref_b).sum().backward()
print(f"a.grad matches autograd's own (a @ b).sum() backward: {torch.allclose(a.grad, ref_a.grad)}")
print(f"b.grad matches autograd's own (a @ b).sum() backward: {torch.allclose(b.grad, ref_b.grad)}")

def chained_loss(a_in, b_in):
    c_in = MatMulOp.apply(a_in, b_in)
    return SumOp.apply(c_in.reshape(64))

ok = torch.autograd.gradcheck(chained_loss, (a, b))
print(f"chained MatMulOp -> SumOp gradcheck passed: {ok}")
```
