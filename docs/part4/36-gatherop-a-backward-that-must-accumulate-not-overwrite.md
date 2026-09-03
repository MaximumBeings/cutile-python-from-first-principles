# Chapter 36: GatherOp — A Backward That Must Accumulate, Not Overwrite

> "Chapter 32 discovered that combining values across multiple blocks sometimes needs atomics because the alternative is a lost update. This chapter discovers the same fact one level up: a gradient can need atomics not because of how many blocks compute it, but because of how many times the SAME element was read on the way forward."

**What you will understand by the end of this chapter:**

- `ct.gather(array, indices)`, called directly on a kernel's array parameter, reading an embedding-style lookup table at runtime-computed row indices — the first time this book's autograd chapters have used gather rather than a fixed-shape `ct.load`
- Why a backward pass built from a plain `ct.scatter` is genuinely WRONG whenever the forward lookup's indices repeat: `ct.scatter` overwrites, so whichever repeated index happens to write last silently discards every earlier contribution to that row's gradient
- Why `ct.atomic_add`, in the same bulk indexed form `ct.gather` and `ct.scatter` both use, fixes this by accumulating instead of overwriting — confirmed correct against `torch`'s own `index_add_`, in a host-side simulation following the same method Chapter 33 used for `padding_mode`, since this specific correctness question is invisible to this book's `export_kernel`-only compile-time method
- `GatherOp`, a `torch.autograd.Function` whose backward launches the accumulating kernel and whose fallback mirrors it with `index_add_`, `gradcheck`-verified specifically on an index list containing repeats
- A capstone chaining `GatherOp` into Chapter 35's shape-generic `MatMulOp` and Chapter 31's `SumOp` — an embedding lookup feeding a linear projection feeding a scalar loss — with the repeated-index accumulation correctly surviving three additional layers of backward composition

**What you need to know first:**

- Chapter 26's `ct.gather` and `ct.scatter` (bulk, per-axis runtime indexing, called directly on a kernel's array parameter rather than a loaded tile).
- Chapter 32's confirmation that `ct.atomic_add` accepts `float32` and follows the same indices convention as `gather`/`scatter`.
- Chapter 33's host-side simulation method, used here again for a correctness question this book's compile-only verification cannot answer on its own.
- Chapter 35's shape-generic `MatMulOp` and Chapter 31's `SumOp`, both reused in this chapter's capstone.
- This book's established rule that a tile shape loaded via `ct.load` must be a power of two — the reason this chapter's lookup count and embedding dimension are both fixed at 8 rather than some more natural-looking number like 6.

## 36.1 ct.gather Reads From a Table; Two Ways to Write Its Gradient Back

### Intuition

An embedding lookup table is a small, ordinary `(vocab_size, dim)` array, and looking several rows up at once — the forward pass of an embedding layer — is exactly what `ct.gather` was built for: one row index per lookup, broadcast against a full range of column indices, reading `dim` values per row in one call. The backward pass has to do the reverse: given a gradient for each looked-up row, write each one back into the table at the row it came from. If the same row was looked up more than once, as it usually is in any real batch of token indices, the backward pass has to add those contributions together, not just place the most recent one.

### Background

`ct.scatter(array, indices, value)` is a bulk, per-axis indexed WRITE, and its docstring says plainly that a masked-out or out-of-bounds write "simply doesn't happen" — it says nothing about what happens when two in-bounds writes target the same location, because ordinary stores don't need to: whichever one executes last wins, and cuTile Python's own launch model does not guarantee an order. `ct.atomic_add(array, indices, update)`, confirmed since Chapter 32 to follow "the same convention as `gather()` and `scatter()`," reads, adds, and writes back atomically per element — the read-modify-write cycle that turns "last write wins" into "every write contributes."

### Worked Example 36.1.1 — Gather forward, and two backward strategies

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

NUM_IDX = 8
DIM = 8
VOCAB = 8

def sig_3arr_2const():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1, dtype=ct.int32), array_param(2),
         ct.compilation.ConstantConstraint(NUM_IDX), ct.compilation.ConstantConstraint(DIM)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_gather_forward(table, idx, out, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, (0, 0), gathered)

# --- WRONG: plain ct.scatter overwrites. When idx has repeated values,
# whichever lookup happens to write last wins, and every earlier
# contribution to that row's gradient is silently lost. ---
@ct.kernel
def kernel_scatter_backward_wrong(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.scatter(grad_table, (row, col), g)

# --- CORRECT: ct.atomic_add accumulates instead of overwriting, so
# repeated indices correctly sum every lookup's contribution. ---
@ct.kernel
def kernel_scatter_backward_atomic(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.atomic_add(grad_table, (row, col), g)

n = compile_bytes(kernel_gather_forward, sig_3arr_2const())
print(f"kernel_gather_forward: {n} cubin bytes")
n = compile_bytes(kernel_scatter_backward_wrong, sig_3arr_2const())
print(f"kernel_scatter_backward_wrong (ct.scatter, overwrites): {n} cubin bytes")
n = compile_bytes(kernel_scatter_backward_atomic, sig_3arr_2const())
print(f"kernel_scatter_backward_atomic (ct.atomic_add, accumulates): {n} cubin bytes")

print()

# --- Now confirm, on the CPU with plain torch (no ct.launch, this
# sandbox has no driver), that the two backward strategies actually
# disagree once idx contains a repeat, and that only the accumulating
# one matches the correct reference gradient. This is the same
# host-side-simulation method Chapter 33 used for padding_mode: a
# correctness question this book's export_kernel-only method cannot
# answer by itself, tested against cuda.tile's own documented semantics
# instead. ---
torch.manual_seed(0)
idx = torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64)
print(f"idx: {idx.tolist()} (2 and 5 each appear twice)")

grad_out = torch.arange(1, NUM_IDX * DIM + 1, dtype=torch.float64).reshape(NUM_IDX, DIM)

# Correct reference: torch's own index_add_, which accumulates.
grad_table_ref = torch.zeros(VOCAB, DIM, dtype=torch.float64)
grad_table_ref.index_add_(0, idx, grad_out)

# Simulated "overwrite" behavior of a plain ct.scatter: later rows in
# idx simply replace earlier writes to the same table row.
grad_table_overwrite = torch.zeros(VOCAB, DIM, dtype=torch.float64)
for i in range(NUM_IDX):
    grad_table_overwrite[idx[i]] = grad_out[i]

print(f"row 2 via accumulation (correct): {grad_table_ref[2].tolist()}")
print(f"row 2 via overwrite (what ct.scatter would give): {grad_table_overwrite[2].tolist()}")
print(f"row 5 via accumulation (correct): {grad_table_ref[5].tolist()}")
print(f"row 5 via overwrite (what ct.scatter would give): {grad_table_overwrite[5].tolist()}")
print(f"overwrite matches correct accumulation: {torch.equal(grad_table_overwrite, grad_table_ref)}")
```

### Genuinely run

```
kernel_gather_forward: 25600 cubin bytes
kernel_scatter_backward_wrong (ct.scatter, overwrites): 26128 cubin bytes
kernel_scatter_backward_atomic (ct.atomic_add, accumulates): 27552 cubin bytes

idx: [2, 5, 2, 0, 7, 5, 1, 3] (2 and 5 each appear twice)
row 2 via accumulation (correct): [18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
row 2 via overwrite (what ct.scatter would give): [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
row 5 via accumulation (correct): [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0]
row 5 via overwrite (what ct.scatter would give): [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0]
overwrite matches correct accumulation: False
```

### Discussion

All three kernels compile cleanly — `export_kernel` has no way to know, and no reason to care, whether a kernel's numerical strategy is the right one for a given use, only whether it is well-formed. Row 2 makes the gap between the two strategies concrete: `idx` looks it up at positions 0 and 2, whose gradient contributions are `[1..8]` and `[17..24]`. Accumulation correctly sums them to `[18, 20, 22, ..., 32]`; the overwrite strategy simply keeps whichever one happened to come last in the loop (`[17..24]`), silently discarding the first lookup's entire contribution. `overwrite matches correct accumulation: False` confirms these are not two equally valid implementations with a minor numerical difference — one of them is wrong whenever `idx` repeats, and a lookup table used more than once per batch is the ordinary case, not an edge case.

## 36.2 GatherOp: Wiring the Accumulating Backward Into autograd

### Intuition

`GatherOp` follows the same shape every custom `Function` in this book has followed since Chapter 30: `forward` attempts a real `ct.launch`, falls back to a plain-`torch` equivalent (`table.index_select(0, idx)`) on the `RuntimeError` this sandbox always raises; `backward` does the same, using `kernel_scatter_backward_atomic` on the real path and `index_add_` — the exact function Section 36.1 used as ground truth — on the fallback path.

### Background

`idx` carries no gradient of its own — it is an integer tensor of positions, not a differentiable quantity — so `GatherOp.backward` returns `None` for it, following `torch.autograd.Function`'s convention that `backward` returns exactly one value (or `None`) per `forward` argument. `backward`'s real path must zero-initialize `grad_table` with `torch.zeros_like` before the atomic-add kernel accumulates into it, since `ct.atomic_add` reads the existing value before adding — starting from anything other than zero would corrupt every row's gradient.

### Worked Example 36.2.1 — GatherOp, gradcheck-verified against repeated indices

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
def kernel_gather_forward(table, idx, out, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, (0, 0), gathered)

@ct.kernel
def kernel_scatter_backward_atomic(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.atomic_add(grad_table, (row, col), g)

class GatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx):
        ctx.save_for_backward(table, idx)
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (1,), kernel_gather_forward, (table, idx32, out, num_idx, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (1,), kernel_scatter_backward_atomic, (grad_output, idx32, grad_table, num_idx, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None

torch.manual_seed(0)
VOCAB, DIM, NUM_IDX = 8, 8, 8
table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64)

out = GatherOp.apply(table, idx)
print(f"forward matches table.index_select(0, idx): {torch.allclose(out, table.detach().index_select(0, idx))}")
print(f"used real forward kernel: {out.grad_fn.used_real_kernel_forward}")

out.sum().backward()
print(f"used real backward kernel: {out.grad_fn.used_real_kernel_backward}")

ref_table = table.detach().clone().requires_grad_()
ref_table.index_select(0, idx).sum().backward()
print(f"table.grad matches reference index_select backward: {torch.allclose(table.grad, ref_table.grad)}")
print(f"table.grad[2] (row looked up twice) is the sum of both contributions: {table.grad[2].tolist()}")

def gather_apply(t):
    return GatherOp.apply(t, idx)

ok = torch.autograd.gradcheck(gather_apply, (table,))
print(f"GatherOp gradcheck passed (with repeated indices in idx): {ok}")
```

### Genuinely run

```
forward matches table.index_select(0, idx): True
used real forward kernel: False
used real backward kernel: False
table.grad matches reference index_select backward: True
table.grad[2] (row looked up twice) is the sum of both contributions: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
GatherOp gradcheck passed (with repeated indices in idx): True
```

### Discussion

`table.grad[2]` is `2.0` in every position, not `1.0` — a direct, numerical confirmation that row 2's gradient correctly reflects both of its two lookups rather than just one. `gradcheck` is deliberately run with the SAME repeated-index `idx` used throughout this chapter, not a fresh index list without repeats — a `gradcheck` call that happened to avoid repeated indices entirely would never exercise the accumulation behavior this chapter is actually about, and would pass just as easily for the wrong, overwriting kernel from Section 36.1. Testing the accumulating property specifically, rather than testing `GatherOp` in some configuration that never triggers it, is what makes this `gradcheck` result meaningful evidence for the claim this chapter makes.

## 36.3 Capstone: An Embedding Lookup Feeding a Linear Projection

### Intuition

A minimal but genuine use of an embedding lookup is a classifier: look up a batch of token embeddings, project them through a learned weight matrix, and sum to a scalar loss. `GatherOp -> MatMulOp -> SumOp` is exactly that pattern, and it puts the repeated-index accumulation Section 36.2 confirmed in isolation under three additional layers of composed backward — through `MatMulOp`'s two-kernel backward and `SumOp`'s broadcast — to see whether it survives.

### Background

`MatMulOp` here is Chapter 35's shape-generic version, called with the gathered `(8, 8)` lookup result as its first operand and a learned `(8, 8)` projection matrix as its second; `SumOp` is Chapter 31's, applied to the flattened `(64,)` projection output. `gradcheck` is called with respect to both `table` and `w` at once, with `idx` held fixed as the non-differentiable input it is.

### Worked Example 36.3.1 — GatherOp, MatMulOp, and SumOp, chained into a classifier

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
def kernel_gather_forward(table, idx, out, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, (0, 0), gathered)

@ct.kernel
def kernel_scatter_backward_atomic(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.atomic_add(grad_table, (row, col), g)

class GatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx):
        ctx.save_for_backward(table, idx)
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (1,), kernel_gather_forward, (table, idx32, out, num_idx, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (1,), kernel_scatter_backward_atomic, (grad_output, idx32, grad_table, num_idx, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None

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

# --- Capstone: an embedding lookup feeding a linear projection feeding a
# scalar loss -- sum((table[idx]) @ W) -- chaining GatherOp (this
# chapter), MatMulOp (Chapter 35's shape-generic version), and SumOp
# (Chapter 31), with idx deliberately containing repeated indices so the
# gradient this chain produces for table depends on GatherOp's
# accumulate-not-overwrite backward being correct. ---
def embedding_classifier(table, w, idx):
    looked_up = GatherOp.apply(table, idx)
    projected = MatMulOp.apply(looked_up, w)
    return SumOp.apply(projected.reshape(-1))

torch.manual_seed(0)
VOCAB, DIM, NUM_IDX = 8, 8, 8
table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
w = torch.randn(DIM, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64)

loss = embedding_classifier(table, w, idx)

def reference_classifier(table_in, w_in, idx_in):
    return (table_in.index_select(0, idx_in) @ w_in).sum()

ref_loss = reference_classifier(table.detach(), w.detach(), idx)
print(f"loss matches plain-torch reference: {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

table_ref = table.detach().clone().requires_grad_()
w_ref = w.detach().clone().requires_grad_()
reference_classifier(table_ref, w_ref, idx).backward()

print(f"table.grad matches reference (repeated-index accumulation included): {torch.allclose(table.grad, table_ref.grad)}")
print(f"w.grad matches reference: {torch.allclose(w.grad, w_ref.grad)}")

def gc_fn(table_in, w_in):
    return embedding_classifier(table_in, w_in, idx)

ok = torch.autograd.gradcheck(gc_fn, (table, w))
print(f"embedding_classifier (GatherOp -> MatMulOp -> SumOp) gradcheck passed: {ok}")
```

### Genuinely run

```
loss matches plain-torch reference: True
table.grad matches reference (repeated-index accumulation included): True
w.grad matches reference: True
embedding_classifier (GatherOp -> MatMulOp -> SumOp) gradcheck passed: True
```

### Discussion

`table.grad` matching `table_ref.grad` exactly is the more demanding confirmation than Section 36.2's isolated test: here, the gradient flowing back to `table` has already passed through `SumOp`'s broadcast and `MatMulOp`'s two-kernel backward before it ever reaches `GatherOp.backward`, and the accumulation for rows 2 and 5 still comes out correct on the other side of that composition. Nothing about `MatMulOp` or `SumOp` treats `table`'s repeated rows specially — they simply pass whatever gradient tensor arrives at their input backward, and it is `GatherOp.backward`'s own `ct.atomic_add`-based (or, here, `index_add_`-based) accumulation that is entirely responsible for those repeats resolving correctly once the gradient finally reaches it.

## Chapter Summary

This chapter built `GatherOp`, wrapping `ct.gather` (called directly on a kernel's array parameter, reading an embedding-style lookup table by runtime row index) and confronting a genuine correctness question its backward pass raises: a plain `ct.scatter`-based backward overwrites, silently discarding earlier contributions whenever the forward lookup's indices repeat, while `ct.atomic_add` — used here in the same bulk indexed form as `gather` and `scatter` — correctly accumulates them instead. A host-side simulation, following the same method Chapter 33 used for `padding_mode`, confirmed the two strategies produce different results on repeated indices and that only the accumulating one matches `torch`'s own `index_add_`. `GatherOp` followed this book's established fallback pattern and was `gradcheck`-verified specifically against an index list containing repeats, since a test that avoided repeats would never have exercised the property the chapter is about. The capstone chained `GatherOp` into Chapter 35's shape-generic `MatMulOp` and Chapter 31's `SumOp` to build a small embedding classifier, confirming the repeated-index accumulation survives correctly through two further layers of composed backward passes.

## Self-Check Questions

1. `ct.scatter`'s own docstring says a masked-out or out-of-bounds write "simply doesn't happen," but says nothing about what happens when two in-bounds writes target the same location. Explain why the docstring doesn't need to address that case for `scatter`'s own intended use, and why `GatherOp`'s backward pass turns it into a real correctness problem.
2. Section 36.1 confirmed the difference between `ct.scatter` and `ct.atomic_add` using a host-side `torch` simulation rather than this book's usual `export_kernel`-only compile-time method. Explain specifically why compilation alone cannot distinguish a correct backward kernel from an incorrect one here, tying your answer to what an `ArrayConstraint` does and doesn't encode.
3. `GatherOp.backward` returns `grad_table, None` rather than two gradient tensors. Explain what the `None` represents and why returning a real gradient tensor for `idx` instead would not make sense.
4. Section 36.2's `gradcheck` call deliberately used the same repeated-index `idx` as every other example in this chapter, rather than a fresh index list with no repeats. Explain why a `gradcheck` pass on a non-repeating index list would have been much weaker evidence for this chapter's central claim.
5. In the capstone, `table.grad` must pass correctly through `SumOp`'s backward and `MatMulOp`'s two-kernel backward before ever reaching `GatherOp.backward`. Explain why `MatMulOp` and `SumOp` do not need to know or care that `table.grad`'s eventual destination has repeated-index rows, tracing the argument back to what each `Function`'s `backward` method is actually responsible for.

## Where We Go Next

This chapter's embedding table was small enough that gathering all `NUM_IDX` rows happened in a single kernel launch, with no multi-block coordination at all — the same simplification Chapter 32 removed for reductions and Chapter 34 explicitly deferred for matmul. A production embedding layer looks up rows from a table far too large to fit any single tile, and a batch of lookups far larger than one block can hold at once — raising the same multi-block question this book has now deferred three times in a row, but this time for an operation whose backward already needs atomics even in the single-block case Section 36.1 covered. Whether multi-block gather needs anything beyond what this chapter's atomics already provide, or introduces its own new wrinkle the way Chapter 32's multi-block max did for reductions, remains open.

## Worked Solutions

**1.** `ct.scatter`'s intended use is an ordinary, single write per location — code that scatters into an array normally arranges for each index to be written at most once, so what happens on a collision is simply outside what the function is designed to handle, and its docstring correctly limits itself to describing masking and bounds behavior instead. `GatherOp`'s backward pass is different: the SAME table row can legitimately be looked up multiple times in one forward pass (an embedding table is looked up by a batch of token indices, and tokens repeat), which means the corresponding backward write is not a single, one-time store at all — it is a genuine multi-write accumulation, a case `scatter`'s documented behavior was never meant to cover and does not handle correctly.

**2.** An `ArrayConstraint` encodes rank and dtype, never a concrete array length or the actual index VALUES a kernel will be launched with — `export_kernel` compiles a kernel once, ahead of time, for a given signature shape, with no visibility into what data will later flow through it. Whether `idx` contains a repeated value is a fact about the DATA at launch time, not about the kernel's structure, so no amount of inspecting the compiled kernel in isolation can reveal whether its scatter-based backward would go wrong on some future input — the same category of limitation Chapter 33 named for `padding_mode`, now recurring for a different reason (data values instead of array length).

**3.** `torch.autograd.Function.backward` must return exactly one gradient (or `None`) for each argument `forward` received, in order; `forward(ctx, table, idx)` took two arguments, so `backward` must return two values. `idx` is an integer tensor of positions into `table`, not a differentiable quantity — there is no meaningful "rate of change of the loss with respect to which row you looked up," since row indices are discrete choices, not continuous values a gradient could describe. Returning `None` for `idx` tells `torch.autograd` correctly that no gradient should (or can) flow to it, and any real tensor returned there would have no valid mathematical meaning to represent.

**4.** `gradcheck` verifies that `GatherOp.apply`'s backward formula produces the correct numerical gradient for whatever specific input it is given — it says nothing at all about inputs it wasn't given. Chapter 32's WRONG overwrite-based kernel would pass a `gradcheck` call made against an index list with no repeats just as easily as the correct accumulating kernel would, because on such an input the two strategies never disagree in the first place — overwriting and accumulating produce identical results when every index is unique. Only testing `gradcheck` against an `idx` that actually contains a repeat exercises the code path where the two strategies diverge, making a pass on THAT specific input real evidence that the accumulating behavior, not just some behavior that happens to look correct on easy inputs, is implemented correctly.

**5.** Each `Function`'s `backward` method is responsible only for computing the correct gradient with respect to ITS OWN inputs, given whatever gradient tensor arrives from downstream — it has no visibility into, and no need for any, what happens further back up the chain. `SumOp.backward` broadcasts its incoming scalar gradient out to match its own input's shape; `MatMulOp.backward` computes `dA = dC @ B^T` and `dB = A^T @ dC` from its own saved operands and incoming gradient. Neither one inspects what `table` looks like, whether `idx` exists, or whether any repeated indices are anywhere in the picture — they simply pass along a correctly-shaped gradient tensor to whatever comes next in the graph. The repeated-index accumulation is entirely `GatherOp.backward`'s own responsibility, and the reason the whole chain works is that each `Function` correctly discharges only its own local piece of the chain rule, trusting `torch.autograd` to assemble the pieces in order.

## Complete Runnable Code

### File: `01_gather_and_two_scatter_backward_strategies.py`

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

NUM_IDX = 8
DIM = 8
VOCAB = 8

def sig_3arr_2const():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1, dtype=ct.int32), array_param(2),
         ct.compilation.ConstantConstraint(NUM_IDX), ct.compilation.ConstantConstraint(DIM)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_gather_forward(table, idx, out, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, (0, 0), gathered)

# --- WRONG: plain ct.scatter overwrites. When idx has repeated values,
# whichever lookup happens to write last wins, and every earlier
# contribution to that row's gradient is silently lost. ---
@ct.kernel
def kernel_scatter_backward_wrong(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.scatter(grad_table, (row, col), g)

# --- CORRECT: ct.atomic_add accumulates instead of overwriting, so
# repeated indices correctly sum every lookup's contribution. ---
@ct.kernel
def kernel_scatter_backward_atomic(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.atomic_add(grad_table, (row, col), g)

n = compile_bytes(kernel_gather_forward, sig_3arr_2const())
print(f"kernel_gather_forward: {n} cubin bytes")
n = compile_bytes(kernel_scatter_backward_wrong, sig_3arr_2const())
print(f"kernel_scatter_backward_wrong (ct.scatter, overwrites): {n} cubin bytes")
n = compile_bytes(kernel_scatter_backward_atomic, sig_3arr_2const())
print(f"kernel_scatter_backward_atomic (ct.atomic_add, accumulates): {n} cubin bytes")

print()

# --- Now confirm, on the CPU with plain torch (no ct.launch, this
# sandbox has no driver), that the two backward strategies actually
# disagree once idx contains a repeat, and that only the accumulating
# one matches the correct reference gradient. This is the same
# host-side-simulation method Chapter 33 used for padding_mode: a
# correctness question this book's export_kernel-only method cannot
# answer by itself, tested against cuda.tile's own documented semantics
# instead. ---
torch.manual_seed(0)
idx = torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64)
print(f"idx: {idx.tolist()} (2 and 5 each appear twice)")

grad_out = torch.arange(1, NUM_IDX * DIM + 1, dtype=torch.float64).reshape(NUM_IDX, DIM)

# Correct reference: torch's own index_add_, which accumulates.
grad_table_ref = torch.zeros(VOCAB, DIM, dtype=torch.float64)
grad_table_ref.index_add_(0, idx, grad_out)

# Simulated "overwrite" behavior of a plain ct.scatter: later rows in
# idx simply replace earlier writes to the same table row.
grad_table_overwrite = torch.zeros(VOCAB, DIM, dtype=torch.float64)
for i in range(NUM_IDX):
    grad_table_overwrite[idx[i]] = grad_out[i]

print(f"row 2 via accumulation (correct): {grad_table_ref[2].tolist()}")
print(f"row 2 via overwrite (what ct.scatter would give): {grad_table_overwrite[2].tolist()}")
print(f"row 5 via accumulation (correct): {grad_table_ref[5].tolist()}")
print(f"row 5 via overwrite (what ct.scatter would give): {grad_table_overwrite[5].tolist()}")
print(f"overwrite matches correct accumulation: {torch.equal(grad_table_overwrite, grad_table_ref)}")
```

### File: `02_gatherop_accumulates_not_overwrites.py`

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
def kernel_gather_forward(table, idx, out, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, (0, 0), gathered)

@ct.kernel
def kernel_scatter_backward_atomic(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.atomic_add(grad_table, (row, col), g)

class GatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx):
        ctx.save_for_backward(table, idx)
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (1,), kernel_gather_forward, (table, idx32, out, num_idx, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (1,), kernel_scatter_backward_atomic, (grad_output, idx32, grad_table, num_idx, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None

torch.manual_seed(0)
VOCAB, DIM, NUM_IDX = 8, 8, 8
table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64)

out = GatherOp.apply(table, idx)
print(f"forward matches table.index_select(0, idx): {torch.allclose(out, table.detach().index_select(0, idx))}")
print(f"used real forward kernel: {out.grad_fn.used_real_kernel_forward}")

out.sum().backward()
print(f"used real backward kernel: {out.grad_fn.used_real_kernel_backward}")

ref_table = table.detach().clone().requires_grad_()
ref_table.index_select(0, idx).sum().backward()
print(f"table.grad matches reference index_select backward: {torch.allclose(table.grad, ref_table.grad)}")
print(f"table.grad[2] (row looked up twice) is the sum of both contributions: {table.grad[2].tolist()}")

def gather_apply(t):
    return GatherOp.apply(t, idx)

ok = torch.autograd.gradcheck(gather_apply, (table,))
print(f"GatherOp gradcheck passed (with repeated indices in idx): {ok}")
```

### File: `03_capstone_embedding_classifier.py`

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
def kernel_gather_forward(table, idx, out, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, (0, 0), gathered)

@ct.kernel
def kernel_scatter_backward_atomic(grad_out, idx, grad_table, num_idx: ct.Constant[int], dim: ct.Constant[int]):
    idx_tile = ct.load(idx, (0,), (num_idx,))
    row = ct.reshape(idx_tile, (num_idx, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, (0, 0), (num_idx, dim))
    ct.atomic_add(grad_table, (row, col), g)

class GatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx):
        ctx.save_for_backward(table, idx)
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (1,), kernel_gather_forward, (table, idx32, out, num_idx, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        num_idx = idx.shape[0]
        dim = table.shape[1]
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (1,), kernel_scatter_backward_atomic, (grad_output, idx32, grad_table, num_idx, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None

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

# --- Capstone: an embedding lookup feeding a linear projection feeding a
# scalar loss -- sum((table[idx]) @ W) -- chaining GatherOp (this
# chapter), MatMulOp (Chapter 35's shape-generic version), and SumOp
# (Chapter 31), with idx deliberately containing repeated indices so the
# gradient this chain produces for table depends on GatherOp's
# accumulate-not-overwrite backward being correct. ---
def embedding_classifier(table, w, idx):
    looked_up = GatherOp.apply(table, idx)
    projected = MatMulOp.apply(looked_up, w)
    return SumOp.apply(projected.reshape(-1))

torch.manual_seed(0)
VOCAB, DIM, NUM_IDX = 8, 8, 8
table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
w = torch.randn(DIM, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64)

loss = embedding_classifier(table, w, idx)

def reference_classifier(table_in, w_in, idx_in):
    return (table_in.index_select(0, idx_in) @ w_in).sum()

ref_loss = reference_classifier(table.detach(), w.detach(), idx)
print(f"loss matches plain-torch reference: {torch.allclose(loss, ref_loss.reshape(1))}")

loss.backward()

table_ref = table.detach().clone().requires_grad_()
w_ref = w.detach().clone().requires_grad_()
reference_classifier(table_ref, w_ref, idx).backward()

print(f"table.grad matches reference (repeated-index accumulation included): {torch.allclose(table.grad, table_ref.grad)}")
print(f"w.grad matches reference: {torch.allclose(w.grad, w_ref.grad)}")

def gc_fn(table_in, w_in):
    return embedding_classifier(table_in, w_in, idx)

ok = torch.autograd.gradcheck(gc_fn, (table, w))
print(f"embedding_classifier (GatherOp -> MatMulOp -> SumOp) gradcheck passed: {ok}")
```
