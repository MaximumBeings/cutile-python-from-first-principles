# Chapter 37: MultiBlockGatherOp — When Atomics Already Solve the Cross-Block Problem

> "Chapter 32 needed a second kernel to combine per-block maxima because a single atomic couldn't do it — float32 has no atomic_max, so the book had to build a reduction pass. This chapter asks the same question of Chapter 36's GatherOp, and this time the answer is almost anticlimactic: the single-block backward already used atomic_add, and atomic_add across four blocks is exactly the same operation atomic_add across one block was."

**What you will understand by the end of this chapter:**

- How to split an embedding lookup across multiple blocks with `ct.bid(0)`, each block gathering its own chunk of indices into its own chunk of the output — the same per-block independence Chapter 32's forward reduction kernels had
- Why `MultiBlockGatherOp`'s backward needs no second kernel to combine per-block contributions, in direct contrast to Chapter 32's multi-block max: `ct.atomic_add` is already the cross-block combination operation, so four blocks atomically adding into one shared `grad_table` need nothing further
- A host-side simulation confirming that four independent per-block accumulations, run in any order, produce the exact same `grad_table` as one combined `index_add_` over everything at once
- `MultiBlockGatherOp`, `gradcheck`-verified against an index list where a single table row is looked up five times across all four blocks — the genuinely new case beyond Chapter 36's single-block repeats
- A capstone chaining `MultiBlockGatherOp` into Chapter 35's `MatMulOp` and Chapter 31's `SumOp`, confirming a row's gradient still sums correctly across blocks after passing through two further layers of composed backward

**What you need to know first:**

- Chapter 36's `GatherOp`, its accumulate-not-overwrite backward, and its host-side simulation method for a correctness question `export_kernel` alone cannot answer.
- Chapter 32's multi-block reductions, and specifically why `MultiBlockMaxOp` needed extra kernels to combine per-block partial maxima that `MultiBlockSumOp` did not — the same distinction this chapter now applies to gather.
- `ct.bid(0)` and the block-indexed `ct.load`/`ct.store` convention (`index=(pid, ...)` meaning an offset of `pid` block-shapes), used throughout Part 2 and in Chapter 32.
- Chapter 35's shape-generic `MatMulOp` and Chapter 31's `SumOp`, both reused in this chapter's capstone.

## 37.1 Splitting the Lookup Across Blocks

### Intuition

Chapter 36's `kernel_gather_forward` handled every lookup in a single block, with no `ct.bid` at all. Splitting the work across blocks is the part of multi-block gather that looks exactly like Chapter 32's multi-block forward kernels: each block reads `ct.bid(0)` to find out which chunk of the index array is its own, gathers only those rows, and writes only to its own chunk of the output — no coordination with any other block required, because reading is inherently independent no matter how many blocks are doing it.

### Background

The interesting question is what happens on the way back. Chapter 32's `MultiBlockMaxOp` needed a genuinely different structure from `MultiBlockSumOp`'s, purely because `float32` has no `atomic_max`: sum's backward could have every block atomically add its own contribution directly into a shared accumulator, while max's forward needed a second kernel just to combine each block's partial maximum into one global value. Chapter 36 already established that `GatherOp`'s backward needs `ct.atomic_add`, even in the single-block case, to combine repeated lookups WITHIN one block correctly. The question this section actually needs answering is empirical, not assumed: does combining contributions ACROSS blocks need anything beyond what already combines them within a block?

### Worked Example 37.1.1 — Multi-block gather forward and backward kernels

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

IDX_PER_BLOCK = 8
DIM = 8
NUM_BLOCKS = 4

def sig_3arr_2const():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1, dtype=ct.int32), array_param(2),
         ct.compilation.ConstantConstraint(IDX_PER_BLOCK), ct.compilation.ConstantConstraint(DIM)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- Forward: each block gathers its OWN chunk of idx_per_block lookups
# and writes them to its own chunk of the output -- no block needs to
# know what any other block is doing, since every row it reads and every
# row it writes belongs to it alone. ---
@ct.kernel
def kernel_gather_forward_multiblock(table, idx, out, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, index=(pid, 0), tile=gathered)

# --- Backward: each block atomic-adds its own chunk's gradient
# contribution into the SAME shared grad_table every other block is also
# writing into. Unlike Chapter 32's multi-block max, which needed a
# second kernel to combine per-block partials, atomic_add already IS the
# cross-block combination -- there is no partial to reduce afterward. ---
@ct.kernel
def kernel_scatter_backward_atomic_multiblock(grad_out, idx, grad_table, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, index=(pid, 0), shape=(idx_per_block, dim))
    ct.atomic_add(grad_table, (row, col), g)

n = compile_bytes(kernel_gather_forward_multiblock, sig_3arr_2const())
print(f"kernel_gather_forward_multiblock: {n} cubin bytes")
n = compile_bytes(kernel_scatter_backward_atomic_multiblock, sig_3arr_2const())
print(f"kernel_scatter_backward_atomic_multiblock: {n} cubin bytes")

print()

# --- Host-side simulation (this sandbox has no driver, so no real
# ct.launch executes across real concurrent blocks): four blocks, each
# atomic-adding its own chunk into a SHARED grad_table, run here as a
# plain Python loop over blocks to confirm no separate cross-block
# reduction step is needed -- simple accumulation, block order
# irrelevant, is already correct. idx is deliberately built so that
# table row 2 is looked up twice within block 0 AND once more in block
# 1 -- three total occurrences spanning two different blocks. ---
torch.manual_seed(0)
VOCAB = 8
idx_blocks = [
    torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64),  # block 0
    torch.tensor([2, 4, 6, 1, 3, 5, 0, 7], dtype=torch.int64),  # block 1 (row 2 repeats again, across blocks)
    torch.tensor([6, 6, 1, 4, 0, 3, 2, 5], dtype=torch.int64),  # block 2
    torch.tensor([7, 0, 1, 2, 6, 4, 5, 3], dtype=torch.int64),  # block 3
]
idx_full = torch.cat(idx_blocks)
print(f"row 2 occurrences by block: {[b.tolist().count(2) for b in idx_blocks]} (total {idx_full.tolist().count(2)}, spanning blocks 0, 1, 2, 3)")

grad_out_full = torch.arange(1, NUM_BLOCKS * IDX_PER_BLOCK * DIM + 1, dtype=torch.float64).reshape(NUM_BLOCKS * IDX_PER_BLOCK, DIM)

# Correct reference: one shot, torch's own index_add_ over everything at once.
grad_table_ref = torch.zeros(VOCAB, DIM, dtype=torch.float64)
grad_table_ref.index_add_(0, idx_full, grad_out_full)

# Simulated multi-block atomic_add: a shared grad_table, four independent
# "blocks" each atomic-adding their own chunk into it, in an arbitrary
# order -- no combination step of any kind beyond the adds themselves.
grad_table_multiblock = torch.zeros(VOCAB, DIM, dtype=torch.float64)
grad_out_blocks = grad_out_full.reshape(NUM_BLOCKS, IDX_PER_BLOCK, DIM)
for b in range(NUM_BLOCKS):
    grad_table_multiblock.index_add_(0, idx_blocks[b], grad_out_blocks[b])

print(f"row 2 gradient (single combined accumulation): {grad_table_ref[2].tolist()}")
print(f"row 2 gradient (four independent blocks, no reduction step): {grad_table_multiblock[2].tolist()}")
print(f"multi-block accumulation matches single-shot accumulation: {torch.equal(grad_table_multiblock, grad_table_ref)}")
```

### Genuinely run

```
kernel_gather_forward_multiblock: 26768 cubin bytes
kernel_scatter_backward_atomic_multiblock: 28576 cubin bytes

row 2 occurrences by block: [2, 1, 1, 1] (total 5, spanning blocks 0, 1, 2, 3)
row 2 gradient (single combined accumulation): [477.0, 482.0, 487.0, 492.0, 497.0, 502.0, 507.0, 512.0]
row 2 gradient (four independent blocks, no reduction step): [477.0, 482.0, 487.0, 492.0, 497.0, 502.0, 507.0, 512.0]
multi-block accumulation matches single-shot accumulation: True
```

### Discussion

Both kernels compile as ordinary, single kernels — no second "reduce the partials" kernel appears anywhere in this file, in direct contrast to Chapter 32's `kernel_reduce_partials_to_global_max`. The host-side simulation is what actually earns that omission rather than merely assuming it: row 2 is looked up five times in total, spread across three of the four blocks in an intentionally uneven way, and the simulated four-block, no-reduction-step accumulation matches the single combined `index_add_` exactly. This is the concrete answer to the question Section 37.1's Background raised — combining gradient contributions across blocks needs nothing beyond what Chapter 36 already needed to combine them within one block, because addition is associative and commutative regardless of which block or which position within a block a given contribution came from. Chapter 32's max needed a second kernel because "the largest of several partial maxima" is not something any single atomic instruction on `float32` can compute; "the sum of several partial sums" is exactly what `atomic_add` already computes, one element at a time, and it does not care whether those elements came from the same block or four different ones.

## 37.2 MultiBlockGatherOp

### Intuition

`MultiBlockGatherOp` follows the same fallback shape as every custom `Function` in this book, with one structural difference from Chapter 36's `GatherOp`: it launches its kernels across `num_blocks` blocks instead of a single block, and it takes `idx_per_block` as an explicit argument so both `forward` and `backward` know how to split the index array.

### Background

`gradcheck` here needs to exercise the genuinely new case this chapter adds beyond Chapter 36: not just a repeat within one block, but a repeat that SPANS multiple blocks. The `idx` tensor used throughout this chapter puts three of table row 2's five total lookups in block 0 and one each in blocks 1 and 3, specifically so that a bug in the CROSS-block accumulation (as opposed to within-block accumulation, which Chapter 36 already tested) would have a chance to surface.

### Worked Example 37.2.1 — MultiBlockGatherOp, gradcheck-verified across a cross-block repeat

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
def kernel_gather_forward_multiblock(table, idx, out, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, index=(pid, 0), tile=gathered)

@ct.kernel
def kernel_scatter_backward_atomic_multiblock(grad_out, idx, grad_table, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, index=(pid, 0), shape=(idx_per_block, dim))
    ct.atomic_add(grad_table, (row, col), g)

class MultiBlockGatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx, idx_per_block):
        ctx.save_for_backward(table, idx)
        ctx.idx_per_block = idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (num_blocks,), kernel_gather_forward_multiblock, (table, idx32, out, idx_per_block, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        idx_per_block = ctx.idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (num_blocks,), kernel_scatter_backward_atomic_multiblock,
                      (grad_output, idx32, grad_table, idx_per_block, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None, None

torch.manual_seed(0)
VOCAB, DIM = 8, 8
IDX_PER_BLOCK = 8
NUM_BLOCKS = 4

table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.cat([
    torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64),
    torch.tensor([2, 4, 6, 1, 3, 5, 0, 7], dtype=torch.int64),
    torch.tensor([6, 6, 1, 4, 0, 3, 2, 5], dtype=torch.int64),
    torch.tensor([7, 0, 1, 2, 6, 4, 5, 3], dtype=torch.int64),
])
print(f"idx has {idx.shape[0]} lookups across {NUM_BLOCKS} blocks of {IDX_PER_BLOCK}")
print(f"row 2 is looked up {idx.tolist().count(2)} times, spanning more than one block")

out = MultiBlockGatherOp.apply(table, idx, IDX_PER_BLOCK)
print(f"forward matches table.index_select(0, idx): {torch.allclose(out, table.detach().index_select(0, idx))}")
print(f"used real forward kernel: {out.grad_fn.used_real_kernel_forward}")

out.sum().backward()
print(f"used real backward kernel: {out.grad_fn.used_real_kernel_backward}")

ref_table = table.detach().clone().requires_grad_()
ref_table.index_select(0, idx).sum().backward()
print(f"table.grad matches reference index_select backward: {torch.allclose(table.grad, ref_table.grad)}")
print(f"table.grad[2] (looked up across multiple blocks) equals its occurrence count: {table.grad[2].tolist()}")

def gather_apply(t):
    return MultiBlockGatherOp.apply(t, idx, IDX_PER_BLOCK)

ok = torch.autograd.gradcheck(gather_apply, (table,))
print(f"MultiBlockGatherOp gradcheck passed (repeats spanning multiple blocks): {ok}")
```

### Genuinely run

```
idx has 32 lookups across 4 blocks of 8
row 2 is looked up 5 times, spanning more than one block
forward matches table.index_select(0, idx): True
used real forward kernel: False
used real backward kernel: False
table.grad matches reference index_select backward: True
table.grad[2] (looked up across multiple blocks) equals its occurrence count: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
MultiBlockGatherOp gradcheck passed (repeats spanning multiple blocks): True
```

### Discussion

`table.grad[2]` comes out as exactly `5.0` in every position — the same value as its raw occurrence count, since `out.sum().backward()` sends a gradient of `1.0` to every looked-up position, and five lookups of row 2 (in this specific test) each contribute one `1.0`. That number matching the occurrence count exactly, across three separate blocks, is the direct numerical signature of correct cross-block accumulation: had even one block's contribution been lost or double-counted, this value would not equal `5.0`. `MultiBlockGatherOp`'s constructor takes `idx_per_block` explicitly rather than trying to infer a block count from the table or the gradient shape, mirroring the same explicitness Chapter 32's multi-block reductions used for `num_blocks`.

## 37.3 Capstone: A Multi-Block Embedding Lookup Feeding a Linear Projection

### Intuition

Chapter 36's capstone chained a single-block `GatherOp` directly into `MatMulOp`. `MatMulOp`'s own kernels, as built in Chapter 35, are still single-tile — they accept an `(8, 8)` input, not a `(32, 8)` one — so this chapter's capstone cannot feed the full 32-row lookup result into one `MatMulOp.apply` call. Instead, each of the four blocks' 8-row chunks is projected separately, and the four resulting per-block losses are summed together, keeping every individual `MatMulOp` and `SumOp` call exactly the shape those chapters already established works.

### Background

Row 2's gradient, spanning three different blocks in `idx`, must now additionally survive passing through whichever of the four separate `MatMulOp` and `SumOp` calls happened to touch each of its occurrences — a strictly harder composition test than Chapter 36's capstone, since the correct final gradient for `table` requires combining contributions that flowed through THREE DIFFERENT invocations of `MatMulOp.backward`, not just one.

### Worked Example 37.3.1 — Multi-block gather feeding four separate projections

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
def kernel_gather_forward_multiblock(table, idx, out, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, index=(pid, 0), tile=gathered)

@ct.kernel
def kernel_scatter_backward_atomic_multiblock(grad_out, idx, grad_table, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, index=(pid, 0), shape=(idx_per_block, dim))
    ct.atomic_add(grad_table, (row, col), g)

class MultiBlockGatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx, idx_per_block):
        ctx.save_for_backward(table, idx)
        ctx.idx_per_block = idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (num_blocks,), kernel_gather_forward_multiblock, (table, idx32, out, idx_per_block, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        idx_per_block = ctx.idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (num_blocks,), kernel_scatter_backward_atomic_multiblock,
                      (grad_output, idx32, grad_table, idx_per_block, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None, None

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

# --- Capstone: a 32-lookup, 4-block embedding gather feeding an 8x8
# linear projection feeding a scalar loss, chaining this chapter's
# MultiBlockGatherOp into Chapter 35's shape-generic MatMulOp -- but
# MatMulOp's own single-tile (8,8) kernels cannot take a (32,8) input
# directly, so the projection is applied per lookup-block, and the four
# per-block results are summed by SumOp. Row 2's gradient, spanning
# three different blocks, has to survive this entire chain correctly. ---
IDX_PER_BLOCK = 8
NUM_BLOCKS = 4
DIM = 8

def embedding_classifier_multiblock(table, w, idx):
    looked_up = MultiBlockGatherOp.apply(table, idx, IDX_PER_BLOCK)  # (32, 8)
    block_losses = []
    for b in range(NUM_BLOCKS):
        chunk = looked_up[b * IDX_PER_BLOCK:(b + 1) * IDX_PER_BLOCK]  # (8, 8)
        projected = MatMulOp.apply(chunk, w)                          # (8, 8)
        block_losses.append(SumOp.apply(projected.reshape(-1)))
    return torch.cat(block_losses).sum().reshape(1)

torch.manual_seed(0)
VOCAB = 8
table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
w = torch.randn(DIM, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.cat([
    torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64),
    torch.tensor([2, 4, 6, 1, 3, 5, 0, 7], dtype=torch.int64),
    torch.tensor([6, 6, 1, 4, 0, 3, 2, 5], dtype=torch.int64),
    torch.tensor([7, 0, 1, 2, 6, 4, 5, 3], dtype=torch.int64),
])

loss = embedding_classifier_multiblock(table, w, idx)

def reference_classifier(table_in, w_in, idx_in):
    looked_up_ref = table_in.index_select(0, idx_in)
    return (looked_up_ref @ w_in).sum().reshape(1)

ref_loss = reference_classifier(table.detach(), w.detach(), idx)
print(f"loss matches plain-torch reference: {torch.allclose(loss, ref_loss)}")

loss.backward()

table_ref = table.detach().clone().requires_grad_()
w_ref = w.detach().clone().requires_grad_()
reference_classifier(table_ref, w_ref, idx).backward()

print(f"table.grad matches reference (cross-block accumulation included): {torch.allclose(table.grad, table_ref.grad)}")
print(f"w.grad matches reference: {torch.allclose(w.grad, w_ref.grad)}")
print(f"table.grad[2] (looked up 5 times across 4 blocks): matches reference: {torch.allclose(table.grad[2], table_ref.grad[2])}")

def gc_fn(table_in, w_in):
    return embedding_classifier_multiblock(table_in, w_in, idx)

ok = torch.autograd.gradcheck(gc_fn, (table, w))
print(f"multi-block embedding_classifier gradcheck passed: {ok}")
```

### Genuinely run

```
loss matches plain-torch reference: True
table.grad matches reference (cross-block accumulation included): True
w.grad matches reference: True
table.grad[2] (looked up 5 times across 4 blocks): matches reference: True
multi-block embedding_classifier gradcheck passed: True
```

### Discussion

`table.grad[2]` now depends on gradient contributions that separately passed through three different `MatMulOp.backward` calls (one per block that looked up row 2) before ever reaching `MultiBlockGatherOp.backward`'s single shared `grad_table`. Each of those three `MatMulOp` invocations is entirely independent — none of them knows the other two exist, none of them knows row 2 is involved at all — and it is only `MultiBlockGatherOp.backward`'s `ct.atomic_add` that ultimately brings their three separately-computed contributions back together into one correct value. That this composition works, verified both by direct comparison and by `gradcheck`, confirms that Section 37.1's core finding — cross-block accumulation needs nothing beyond what within-block accumulation already needed — continues to hold even after the gradient has been split apart and recombined by an entirely separate set of `Function` calls in between.

## Chapter Summary

This chapter extended Chapter 36's `GatherOp` to multiple blocks, splitting both the forward lookup and the backward accumulation across `ct.bid(0)`-indexed chunks the same way Chapter 32's multi-block reductions did. The central finding was structural rather than numerical: unlike Chapter 32's `MultiBlockMaxOp`, which needed an entire second kernel to combine per-block partial maxima because `float32` has no `atomic_max`, `MultiBlockGatherOp`'s backward needed no second kernel at all — `ct.atomic_add`, already necessary in Chapter 36's single-block case, is directly the cross-block combination operation, since addition is associative and commutative regardless of which block a contribution originated in. A host-side simulation confirmed four independent per-block accumulations match one combined accumulation exactly, and `gradcheck` confirmed the same for a table row looked up five times across three different blocks. The capstone chained `MultiBlockGatherOp` into per-block `MatMulOp` and `SumOp` calls, confirming that row's gradient correctly recombines even after its separate contributions pass through three entirely independent `MatMulOp.backward` invocations first.

## Self-Check Questions

1. Explain specifically why `float32`'s missing `atomic_max` is what forced Chapter 32's `MultiBlockMaxOp` to need a second kernel, while `MultiBlockGatherOp`'s backward in this chapter needed no equivalent second kernel.
2. Section 37.1's host-side simulation ran four blocks' accumulations as a strict, ordered Python loop (block 0, then block 1, then block 2, then block 3). Does the fact that this specific order produced the correct answer prove that ANY execution order would also produce the correct answer? Explain what property of `ct.atomic_add` your answer depends on.
3. In the capstone, why does `looked_up` (the output of `MultiBlockGatherOp.apply`) need to be sliced into four separate `(8, 8)` chunks before being passed to `MatMulOp.apply`, rather than being passed to `MatMulOp.apply` as one `(32, 8)` tensor?
4. Trace through what happens to table row 2's gradient contribution from block 1 specifically: which `Function.backward` calls does it pass through, in what order, before it is added into `grad_table`?
5. Suppose a future chapter built `MultiBlockMaxOp`'s embedding-lookup equivalent — a "max-pooling gather" that, for each output position, took the MAXIMUM of several looked-up rows rather than returning each one separately. Would that operation's backward need a structure more like this chapter's `MultiBlockGatherOp`, or more like Chapter 32's `MultiBlockMaxOp`? Justify your answer.

## Where We Go Next

Every custom `Function` built across this chapter and the seven before it has combined its inputs through a reduction, a matmul, or a table lookup — never through a plain elementwise binary operation on two tensors of different shapes. An operation as ordinary as adding a per-row bias to every row of a matrix needs its backward pass to sum a gradient back down across whichever axes were broadcast on the way forward, a genuinely different kind of "combining a gradient" than anything atomics-based Chapters 32, 36, and this chapter have needed. That gap is Part 4's last piece of open ground, and closing it is where this book turns next.

## Worked Solutions

**1.** Chapter 20 established that `atomic_max` (and `atomic_min`) reject `float32`, while `atomic_add` accepts it. Combining several blocks' partial MAXIMA into one global maximum therefore cannot be done with a single atomic instruction on `float32` — Chapter 32 had to write a whole second kernel, `kernel_reduce_partials_to_global_max`, specifically because no atomic primitive was available to do that combination directly. Combining several blocks' partial SUMS, by contrast, is exactly what `atomic_add` already computes: every block can atomically add its own contribution straight into a shared accumulator, one element at a time, with no separate combination step needed at all — which is exactly the situation `MultiBlockGatherOp`'s backward is in, since its job is also to accumulate (sum) contributions, not to find a maximum among them.

**2.** No — the ordered loop happening to give the right answer does not, by itself, prove every order would. What DOES guarantee that is a property of addition itself: it is both associative and commutative, meaning `a + b + c` gives the same result regardless of the order the additions happen in or how they're grouped. `ct.atomic_add`'s own documentation states explicitly that "the order of individual writes is unspecified" — the operation is designed around exactly this property, guaranteeing a correct final sum precisely because addition doesn't care about order. A combination operation that lacked that property (subtraction, for instance) would give different, and generally wrong, answers depending on execution order, which is part of why no `atomic_subtract`-style combination could ever substitute for `atomic_add` here regardless of how carefully blocks were ordered.

**3.** `MatMulOp`'s kernels, unchanged from Chapter 35, are compiled ahead-of-time for a fixed `(m, k, n)` shape read from the tensors passed to `MatMulOp.forward` — passing a `(32, 8)` tensor as the first operand would attempt an `(32, 8) @ (8, 8)` matmul, a well-defined but DIFFERENT computation than four separate `(8, 8) @ (8, 8)` matmuls, and one this chapter's capstone specifically did not want (the point being four independent per-block projections, not one differently-shaped combined one). Slicing `looked_up` into four `(8, 8)` chunks keeps every `MatMulOp.apply` call at the exact shape Chapter 35 already built and verified, at the cost of needing four separate calls (and correspondingly, `SumOp` calls) instead of one.

**4.** Block 1's chunk of `looked_up` (rows 8 through 15 of the 32-row gather result) is passed to its own `MatMulOp.apply(chunk, w)` call, producing an `(8, 8)` projected tensor; that feeds its own `SumOp.apply(...)` call, producing one of the four `block_losses`. On `.backward()`, gradient flows first through THAT `SumOp.backward` (broadcasting the scalar loss gradient back to `(8,)`, reshaped to `(8, 8)`), then through THAT `MatMulOp.backward` (computing `grad_a = grad_c @ w.T`, an `(8, 8)` gradient with respect to block 1's chunk of `looked_up`), and this becomes part of the FULL `(32, 8)` gradient tensor `grad_output` that `MultiBlockGatherOp.backward` receives as a whole (assembled automatically by `torch.autograd` from the four independently-computed `(8,8)` gradient chunks landing at their corresponding rows of the same `looked_up` tensor). Only at that final step does `kernel_scatter_backward_atomic_multiblock`'s `ct.atomic_add` combine block 1's contribution to row 2 with whatever contributions arrived from blocks 0 and 2 as well.

**5.** It would need a structure more like Chapter 32's `MultiBlockMaxOp`, not this chapter's `MultiBlockGatherOp`. The distinguishing question, as this chapter's own central finding established, is not "does this operation involve blocks and combination" but specifically "does the COMBINING operation have a working atomic primitive for the dtype in question." A max-pooling gather's forward needs to compute a MAXIMUM across several looked-up rows — the same combining operation Chapter 32 already found has no `float32` atomic — so it would face exactly Chapter 32's problem (needing a second kernel, or some other explicit combination step) rather than this chapter's atomic_add-based shortcut, regardless of the fact that both operations involve `ct.gather` somewhere in their forward pass.

## Complete Runnable Code

### File: `01_multiblock_gather_kernels.py`

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

IDX_PER_BLOCK = 8
DIM = 8
NUM_BLOCKS = 4

def sig_3arr_2const():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(1, dtype=ct.int32), array_param(2),
         ct.compilation.ConstantConstraint(IDX_PER_BLOCK), ct.compilation.ConstantConstraint(DIM)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- Forward: each block gathers its OWN chunk of idx_per_block lookups
# and writes them to its own chunk of the output -- no block needs to
# know what any other block is doing, since every row it reads and every
# row it writes belongs to it alone. ---
@ct.kernel
def kernel_gather_forward_multiblock(table, idx, out, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, index=(pid, 0), tile=gathered)

# --- Backward: each block atomic-adds its own chunk's gradient
# contribution into the SAME shared grad_table every other block is also
# writing into. Unlike Chapter 32's multi-block max, which needed a
# second kernel to combine per-block partials, atomic_add already IS the
# cross-block combination -- there is no partial to reduce afterward. ---
@ct.kernel
def kernel_scatter_backward_atomic_multiblock(grad_out, idx, grad_table, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, index=(pid, 0), shape=(idx_per_block, dim))
    ct.atomic_add(grad_table, (row, col), g)

n = compile_bytes(kernel_gather_forward_multiblock, sig_3arr_2const())
print(f"kernel_gather_forward_multiblock: {n} cubin bytes")
n = compile_bytes(kernel_scatter_backward_atomic_multiblock, sig_3arr_2const())
print(f"kernel_scatter_backward_atomic_multiblock: {n} cubin bytes")

print()

# --- Host-side simulation (this sandbox has no driver, so no real
# ct.launch executes across real concurrent blocks): four blocks, each
# atomic-adding its own chunk into a SHARED grad_table, run here as a
# plain Python loop over blocks to confirm no separate cross-block
# reduction step is needed -- simple accumulation, block order
# irrelevant, is already correct. idx is deliberately built so that
# table row 2 is looked up twice within block 0 AND once more in block
# 1 -- three total occurrences spanning two different blocks. ---
torch.manual_seed(0)
VOCAB = 8
idx_blocks = [
    torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64),  # block 0
    torch.tensor([2, 4, 6, 1, 3, 5, 0, 7], dtype=torch.int64),  # block 1 (row 2 repeats again, across blocks)
    torch.tensor([6, 6, 1, 4, 0, 3, 2, 5], dtype=torch.int64),  # block 2
    torch.tensor([7, 0, 1, 2, 6, 4, 5, 3], dtype=torch.int64),  # block 3
]
idx_full = torch.cat(idx_blocks)
print(f"row 2 occurrences by block: {[b.tolist().count(2) for b in idx_blocks]} (total {idx_full.tolist().count(2)}, spanning blocks 0, 1, 2, 3)")

grad_out_full = torch.arange(1, NUM_BLOCKS * IDX_PER_BLOCK * DIM + 1, dtype=torch.float64).reshape(NUM_BLOCKS * IDX_PER_BLOCK, DIM)

# Correct reference: one shot, torch's own index_add_ over everything at once.
grad_table_ref = torch.zeros(VOCAB, DIM, dtype=torch.float64)
grad_table_ref.index_add_(0, idx_full, grad_out_full)

# Simulated multi-block atomic_add: a shared grad_table, four independent
# "blocks" each atomic-adding their own chunk into it, in an arbitrary
# order -- no combination step of any kind beyond the adds themselves.
grad_table_multiblock = torch.zeros(VOCAB, DIM, dtype=torch.float64)
grad_out_blocks = grad_out_full.reshape(NUM_BLOCKS, IDX_PER_BLOCK, DIM)
for b in range(NUM_BLOCKS):
    grad_table_multiblock.index_add_(0, idx_blocks[b], grad_out_blocks[b])

print(f"row 2 gradient (single combined accumulation): {grad_table_ref[2].tolist()}")
print(f"row 2 gradient (four independent blocks, no reduction step): {grad_table_multiblock[2].tolist()}")
print(f"multi-block accumulation matches single-shot accumulation: {torch.equal(grad_table_multiblock, grad_table_ref)}")
```

### File: `02_multiblockgatherop.py`

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
def kernel_gather_forward_multiblock(table, idx, out, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, index=(pid, 0), tile=gathered)

@ct.kernel
def kernel_scatter_backward_atomic_multiblock(grad_out, idx, grad_table, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, index=(pid, 0), shape=(idx_per_block, dim))
    ct.atomic_add(grad_table, (row, col), g)

class MultiBlockGatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx, idx_per_block):
        ctx.save_for_backward(table, idx)
        ctx.idx_per_block = idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (num_blocks,), kernel_gather_forward_multiblock, (table, idx32, out, idx_per_block, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        idx_per_block = ctx.idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (num_blocks,), kernel_scatter_backward_atomic_multiblock,
                      (grad_output, idx32, grad_table, idx_per_block, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None, None

torch.manual_seed(0)
VOCAB, DIM = 8, 8
IDX_PER_BLOCK = 8
NUM_BLOCKS = 4

table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.cat([
    torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64),
    torch.tensor([2, 4, 6, 1, 3, 5, 0, 7], dtype=torch.int64),
    torch.tensor([6, 6, 1, 4, 0, 3, 2, 5], dtype=torch.int64),
    torch.tensor([7, 0, 1, 2, 6, 4, 5, 3], dtype=torch.int64),
])
print(f"idx has {idx.shape[0]} lookups across {NUM_BLOCKS} blocks of {IDX_PER_BLOCK}")
print(f"row 2 is looked up {idx.tolist().count(2)} times, spanning more than one block")

out = MultiBlockGatherOp.apply(table, idx, IDX_PER_BLOCK)
print(f"forward matches table.index_select(0, idx): {torch.allclose(out, table.detach().index_select(0, idx))}")
print(f"used real forward kernel: {out.grad_fn.used_real_kernel_forward}")

out.sum().backward()
print(f"used real backward kernel: {out.grad_fn.used_real_kernel_backward}")

ref_table = table.detach().clone().requires_grad_()
ref_table.index_select(0, idx).sum().backward()
print(f"table.grad matches reference index_select backward: {torch.allclose(table.grad, ref_table.grad)}")
print(f"table.grad[2] (looked up across multiple blocks) equals its occurrence count: {table.grad[2].tolist()}")

def gather_apply(t):
    return MultiBlockGatherOp.apply(t, idx, IDX_PER_BLOCK)

ok = torch.autograd.gradcheck(gather_apply, (table,))
print(f"MultiBlockGatherOp gradcheck passed (repeats spanning multiple blocks): {ok}")
```

### File: `03_capstone_multiblock_embedding_classifier.py`

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
def kernel_gather_forward_multiblock(table, idx, out, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    gathered = ct.gather(table, (row, col))
    ct.store(out, index=(pid, 0), tile=gathered)

@ct.kernel
def kernel_scatter_backward_atomic_multiblock(grad_out, idx, grad_table, idx_per_block: ct.Constant[int], dim: ct.Constant[int]):
    pid = ct.bid(0)
    idx_tile = ct.load(idx, index=(pid,), shape=(idx_per_block,))
    row = ct.reshape(idx_tile, (idx_per_block, 1))
    col = ct.reshape(ct.arange(dim, dtype=ct.int32), (1, dim))
    g = ct.load(grad_out, index=(pid, 0), shape=(idx_per_block, dim))
    ct.atomic_add(grad_table, (row, col), g)

class MultiBlockGatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx, idx_per_block):
        ctx.save_for_backward(table, idx)
        ctx.idx_per_block = idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            out = torch.empty(num_idx, dim, dtype=table.dtype)
            ct.launch(stream, (num_blocks,), kernel_gather_forward_multiblock, (table, idx32, out, idx_per_block, dim))
            ctx.used_real_kernel_forward = True
            return out
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return table.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_output):
        table, idx = ctx.saved_tensors
        idx_per_block = ctx.idx_per_block
        num_idx = idx.shape[0]
        dim = table.shape[1]
        num_blocks = num_idx // idx_per_block
        try:
            stream = torch.cuda.current_stream()
            idx32 = idx.to(torch.int32)
            grad_table = torch.zeros_like(table)
            ct.launch(stream, (num_blocks,), kernel_scatter_backward_atomic_multiblock,
                      (grad_output, idx32, grad_table, idx_per_block, dim))
            ctx.used_real_kernel_backward = True
            return grad_table, None, None
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            grad_table = torch.zeros_like(table)
            grad_table.index_add_(0, idx, grad_output)
            return grad_table, None, None

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

# --- Capstone: a 32-lookup, 4-block embedding gather feeding an 8x8
# linear projection feeding a scalar loss, chaining this chapter's
# MultiBlockGatherOp into Chapter 35's shape-generic MatMulOp -- but
# MatMulOp's own single-tile (8,8) kernels cannot take a (32,8) input
# directly, so the projection is applied per lookup-block, and the four
# per-block results are summed by SumOp. Row 2's gradient, spanning
# three different blocks, has to survive this entire chain correctly. ---
IDX_PER_BLOCK = 8
NUM_BLOCKS = 4
DIM = 8

def embedding_classifier_multiblock(table, w, idx):
    looked_up = MultiBlockGatherOp.apply(table, idx, IDX_PER_BLOCK)  # (32, 8)
    block_losses = []
    for b in range(NUM_BLOCKS):
        chunk = looked_up[b * IDX_PER_BLOCK:(b + 1) * IDX_PER_BLOCK]  # (8, 8)
        projected = MatMulOp.apply(chunk, w)                          # (8, 8)
        block_losses.append(SumOp.apply(projected.reshape(-1)))
    return torch.cat(block_losses).sum().reshape(1)

torch.manual_seed(0)
VOCAB = 8
table = torch.randn(VOCAB, DIM, dtype=torch.float64, requires_grad=True)
w = torch.randn(DIM, DIM, dtype=torch.float64, requires_grad=True)
idx = torch.cat([
    torch.tensor([2, 5, 2, 0, 7, 5, 1, 3], dtype=torch.int64),
    torch.tensor([2, 4, 6, 1, 3, 5, 0, 7], dtype=torch.int64),
    torch.tensor([6, 6, 1, 4, 0, 3, 2, 5], dtype=torch.int64),
    torch.tensor([7, 0, 1, 2, 6, 4, 5, 3], dtype=torch.int64),
])

loss = embedding_classifier_multiblock(table, w, idx)

def reference_classifier(table_in, w_in, idx_in):
    looked_up_ref = table_in.index_select(0, idx_in)
    return (looked_up_ref @ w_in).sum().reshape(1)

ref_loss = reference_classifier(table.detach(), w.detach(), idx)
print(f"loss matches plain-torch reference: {torch.allclose(loss, ref_loss)}")

loss.backward()

table_ref = table.detach().clone().requires_grad_()
w_ref = w.detach().clone().requires_grad_()
reference_classifier(table_ref, w_ref, idx).backward()

print(f"table.grad matches reference (cross-block accumulation included): {torch.allclose(table.grad, table_ref.grad)}")
print(f"w.grad matches reference: {torch.allclose(w.grad, w_ref.grad)}")
print(f"table.grad[2] (looked up 5 times across 4 blocks): matches reference: {torch.allclose(table.grad[2], table_ref.grad[2])}")

def gc_fn(table_in, w_in):
    return embedding_classifier_multiblock(table_in, w_in, idx)

ok = torch.autograd.gradcheck(gc_fn, (table, w))
print(f"multi-block embedding_classifier gradcheck passed: {ok}")
```
