# Chapter 32: Multi-Block Reductions — When Sum Has Atomics and Max Doesn't

> "Chapter 31's reductions collapsed an entire tile down to a scalar in one kernel launch, which quietly assumed the data fit in one block's worth of tile. A real reduction usually doesn't: the work is split across blocks that run independently, with no fixed order and no shared memory between them, and 'combine the partial results' turns out to mean something completely different depending on whether the combining operation is one this hardware has an atomic instruction for."

**What you will understand by the end of this chapter:**

- That `ct.atomic_add` supports `float32` while `ct.atomic_max` does not — a restriction Chapter 20 already found and this chapter meets again, in a new context, as the exact reason a multi-block sum and a multi-block max cannot be built the same way
- `MultiBlockSumOp`: a forward kernel where every block atomically accumulates its own partial sum into one shared scalar, and a backward kernel where every block independently broadcasts the same incoming gradient to its own slice — two kernel launches total, made possible entirely by `atomic_add`'s `float32` support
- `MultiBlockMaxOp`: without a `float32` atomic max, a forward pass that needs TWO kernel launches (each block writes its own local max to its own slot, then a small second kernel reduces those slots to one global value) and a backward pass that needs THREE (recompute which elements tie for that global max, per block; reduce those per-block tie counts to one global count; only then distribute the gradient) — five kernel launches for one operation, an asymmetry with sum that traces to a single, specific, already-known hardware restriction
- That each block writing to its OWN slot in an array needs no atomic operation at all, even though multiple blocks run "concurrently" — a distinction from Chapter 20's atomics, which exist specifically for when multiple blocks might touch the SAME location
- A capstone chaining an elementwise `DoubleOp` into both multi-block reductions over a 32-element input split across 4 blocks, confirmed correct with `gradcheck` exactly as the single-block versions were in Chapter 31

**What you need to know first:**

- Chapter 31's `SumOp`/`MaxOp`: the broadcasting-backward and masked-backward formulas this chapter reuses unchanged — multi-block partitioning is an implementation-organization question, not a change to what the mathematically correct forward or backward VALUE is.
- Chapter 20's atomic operations (`ct.atomic_add`, `ct.atomic_max`, and the rest) and its finding that `float32` support splits them into two groups.
- Chapter 30's `torch.autograd.Function` fallback pattern, extended here to a `Function` whose forward or backward attempts MULTIPLE sequential `ct.launch` calls rather than one.

## 32.1 Compiling the Multi-Block Kernels: Atomics for Sum, Two Phases for Max

### Intuition

A single-block reduction (Chapter 31) loads the whole tile, reduces it, and stores one value — there is only ever one block, so there is nothing to combine. The moment a reduction spans multiple blocks, each block can only see its own slice, and "the final answer" requires combining N independent partial results computed by N blocks that share no memory and run in an unspecified order. `ct.atomic_add` can do that combination directly, in place, because addition is genuinely commutative and associative at the hardware level, and this book already knows (Chapter 20) that `atomic_add` supports `float32`. `ct.atomic_max` exists too, and does the equivalent job for max — but Chapter 20 also already found that it does NOT support `float32`, and this chapter is the first place that restriction actually blocks something this book wants to build.

### Background

Six kernels are compiled here, plus one deliberately kept for what fails: `kernel_bad_atomic_max` demonstrates the `float32` rejection directly, in the exact shape this chapter needs it (accumulating per-block maxima into a shared scalar). `kernel_sum_multiblock_forward`/`kernel_sum_multiblock_backward` are Chapter 31's sum kernels, adapted with `ct.bid(0)`-based indexing so each block reads and writes its own slice rather than a single fixed `(0,)` offset. The four `max` kernels split the job `atomic_max` would have done into two explicit phases: `kernel_max_partial` (each block stores its own local max, no synchronization needed since every block writes a different slot) and `kernel_reduce_partials_to_global_max` (a small single-block reduction — literally Chapter 31's own `kernel_max_forward` pattern, reused to combine a handful of partial maxima instead of the original data) — with the same two-phase shape repeated for tie-counting in `kernel_max_count_partial`/`kernel_reduce_partial_counts`, and a final `kernel_max_multiblock_backward` that only runs once both the global max and the global tie count are known.

### Worked Example 32.1.1 — the float32 atomic_max wall, and the seven kernels built around it

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

def sig_n(n_arrays, const_val):
    return ct.compilation.KernelSignature(
        [array_param(1) for _ in range(n_arrays)] + [ct.compilation.ConstantConstraint(const_val)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- confirm the float32 atomic_max restriction Chapter 20 already found,
# now hit in a genuinely new context: trying to combine per-block maxima. ---
@ct.kernel
def kernel_bad_atomic_max(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.atomic_max(acc, (0,), partial_max)

try:
    n = compile_bytes(kernel_bad_atomic_max, sig_n(2, 8))
    print(f"atomic_max on float32 accumulator: {n} bytes (unexpected)")
except Exception as e:
    print(f"atomic_max on float32 accumulator: {type(e).__name__}: {e}")

print()

# --- Sum: atomic_add DOES support float32, so one kernel per direction. ---
@ct.kernel
def kernel_sum_multiblock_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)
n = compile_bytes(kernel_sum_multiblock_forward, sig_n(2, 8))
print(f"kernel_sum_multiblock_forward: {n} cubin bytes")

@ct.kernel
def kernel_sum_multiblock_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))
n = compile_bytes(kernel_sum_multiblock_backward, sig_n(2, 8))
print(f"kernel_sum_multiblock_backward: {n} cubin bytes")

print()

# --- Max: no float32 atomic_max, so two phases -- each block writes to its
# OWN slot (no synchronization needed), then a small second kernel reduces
# those per-block partials down to one value. ---
@ct.kernel
def kernel_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))
n = compile_bytes(kernel_max_partial, sig_n(2, 8))
print(f"kernel_max_partial: {n} cubin bytes")

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))
n = compile_bytes(kernel_reduce_partials_to_global_max, sig_n(2, 4))
print(f"kernel_reduce_partials_to_global_max: {n} cubin bytes")

@ct.kernel
def kernel_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    local_count = ct.sum(mask, None)
    ct.store(partial_counts, index=(pid,), tile=ct.broadcast_to(local_count, (1,)))
n = compile_bytes(kernel_max_count_partial, sig_n(3, 8))
print(f"kernel_max_count_partial: {n} cubin bytes")

@ct.kernel
def kernel_reduce_partial_counts(partial_counts, out, num_blocks: ct.Constant[int]):
    x = ct.load(partial_counts, (0,), (num_blocks,))
    y = ct.sum(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))
n = compile_bytes(kernel_reduce_partial_counts, sig_n(2, 4))
print(f"kernel_reduce_partial_counts: {n} cubin bytes")

@ct.kernel
def kernel_max_multiblock_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)
n = compile_bytes(kernel_max_multiblock_backward, sig_n(5, 8))
print(f"kernel_max_multiblock_backward: {n} cubin bytes")
```

Genuinely run:

```
atomic_max on float32 accumulator: TileTypeError: Unsupported array dtype: float32
  "/tmp/ch32_final/01_multiblock_sum_and_max_kernels.py", line 28, col 5-41, in kernel_bad_atomic_max:
        ct.atomic_max(acc, (0,), partial_max)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


kernel_sum_multiblock_forward: 22144 cubin bytes
kernel_sum_multiblock_backward: 20656 cubin bytes

kernel_max_partial: 21024 cubin bytes
kernel_reduce_partials_to_global_max: 20656 cubin bytes
kernel_max_count_partial: 22816 cubin bytes
kernel_reduce_partial_counts: 20400 cubin bytes
kernel_max_multiblock_backward: 30336 cubin bytes
```

### Discussion

The `TileTypeError` is not a new discovery — Chapter 20 already established that `atomic_max`, `atomic_min`, and the three bitwise atomics all reject `float32` while `atomic_add`, `atomic_xchg`, and `atomic_cas` accept it. What is new here is meeting that restriction as a genuine, concrete obstacle rather than an abstract fact recorded in a table: it is the entire reason `kernel_sum_multiblock_forward` gets to be one simple kernel while the equivalent max operation needs `kernel_max_partial` and `kernel_reduce_partials_to_global_max` working together. Both `kernel_max_partial` and `kernel_reduce_partials_to_global_max` compile without incident — nothing about writing each block's result to its OWN array slot, or reducing a handful of partials in a second, tiny kernel, requires an atomic operation at all, because there is no location any two blocks (or the second kernel) ever touch at the same time.

## 32.2 MultiBlockSumOp: Atomics Make This Easy

### Intuition

With `atomic_add` supporting `float32`, a multi-block sum needs exactly the same TWO kernels a single-block sum needed in Chapter 31 — `kernel_sum_multiblock_forward` and `kernel_sum_multiblock_backward` — just launched over a `grid` of multiple blocks instead of one, with each block's `ct.bid(0)` selecting which slice of the array it owns.

### Background

`MultiBlockSumOp` launches `kernel_sum_multiblock_forward` over `NUM_BLOCKS = 4` blocks of `BLOCK_SIZE = 8` (32 elements total), each block adding its own partial sum into a single pre-zeroed accumulator via `ct.atomic_add`. `backward` launches `kernel_sum_multiblock_backward` over the same 4 blocks, each one independently broadcasting the SAME incoming gradient value out to its own 8-element slice — no atomics needed here either, since every block writes to a disjoint region of `grad_input`.

### Worked Example 32.2.1 — one atomic-accumulated forward, one broadcast-per-block backward

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
N = NUM_BLOCKS * BLOCK_SIZE

@ct.kernel
def kernel_sum_multiblock_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)

@ct.kernel
def kernel_sum_multiblock_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))

class MultiBlockSumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            acc = torch.zeros(1, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_forward, (x, acc, BLOCK_SIZE))
            ctx.used_real_kernel_forward = True
            return acc
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_backward, (grad_output, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output.expand(ctx.n).clone()

x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y = MultiBlockSumOp.apply(x)
print(f"forward matches x.sum() across {NUM_BLOCKS} blocks of {BLOCK_SIZE}: {torch.equal(y, x.detach().sum().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad is uniformly 1 across all {N} elements: {torch.allclose(x.grad, torch.ones(N, dtype=torch.float64))}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(MultiBlockSumOp.apply, (x,))
print(f"MultiBlockSumOp gradcheck passed: {ok}")
```

Genuinely run:

```
forward matches x.sum() across 4 blocks of 8: True
used real forward kernel: False
x.grad is uniformly 1 across all 32 elements: True
used real backward kernel: False
MultiBlockSumOp gradcheck passed: True
```

### Discussion

Splitting the sum across four blocks changes nothing about the mathematically correct forward value or gradient — `gradcheck` passes on the 32-element input exactly as it did on Chapter 31's 8-element one, because `atomic_add`'s job is entirely about HOW the hardware would combine four independent partial sums, never about WHAT the correct total is. The fallback formulas (`x.sum()`, `grad_output.expand(ctx.n)`) don't know or care how many blocks the real kernel would have used either — they compute the same answer a single block, four blocks, or four hundred blocks would all be required to agree on.

## 32.3 MultiBlockMaxOp: Three Passes for One Gradient

### Intuition

`SumOp`'s multi-block story needed no new idea beyond `ct.bid`-based indexing, because `atomic_add` handled the actual cross-block combination. `MaxOp` has no such atomic available for `float32`, so both its forward AND its backward have to be organized as explicit, ordered passes: nothing about tie-counting can even begin until the TRUE global max — the result of combining every block's local max — is already known.

### Background

`MultiBlockMaxOp.forward` launches two kernels in sequence: `kernel_max_partial` (each block's local max into its own slot) and `kernel_reduce_partials_to_global_max` (one small kernel combining those `NUM_BLOCKS` values into the true global max). The result — both the input `x` and the computed global max — is saved via `ctx.save_for_backward(x, out)`, so `backward` doesn't need to recompute the forward reduction from scratch. `backward` then launches three kernels: `kernel_max_count_partial` (each block's local count of elements tied with the ALREADY-KNOWN global max), `kernel_reduce_partial_counts` (summing those per-block counts into one global tie count), and finally `kernel_max_multiblock_backward` (distributing the gradient using both the global max and the global count, block by block).

### Worked Example 32.3.1 — two passes to find the max, three passes to distribute its gradient

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
N = NUM_BLOCKS * BLOCK_SIZE

@ct.kernel
def kernel_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    local_count = ct.sum(mask, None)
    ct.store(partial_counts, index=(pid,), tile=ct.broadcast_to(local_count, (1,)))

@ct.kernel
def kernel_reduce_partial_counts(partial_counts, out, num_blocks: ct.Constant[int]):
    x = ct.load(partial_counts, (0,), (num_blocks,))
    y = ct.sum(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_multiblock_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)

class MultiBlockMaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            partials = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_partial, (x, partials, BLOCK_SIZE))
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partials_to_global_max, (partials, out, NUM_BLOCKS))
            ctx.used_real_kernel_forward = True
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            out = x.max().reshape(1)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, global_max = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            partial_counts = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_count_partial, (x, global_max, partial_counts, BLOCK_SIZE))
            global_count = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partial_counts, (partial_counts, global_count, NUM_BLOCKS))
            grad_input = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_multiblock_backward, (x, global_max, grad_output, global_count, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x == global_max).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y = MultiBlockMaxOp.apply(x)
print(f"forward matches x.max() across {NUM_BLOCKS} blocks of {BLOCK_SIZE}: {torch.equal(y, x.detach().max().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
expected = (x.detach() == x.detach().max()).double()
print(f"x.grad is a one-hot at the (unique) global argmax: {torch.equal(x.grad, expected)}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(MultiBlockMaxOp.apply, (x,))
print(f"MultiBlockMaxOp gradcheck passed (random input, max is unique): {ok}")
```

Genuinely run:

```
forward matches x.max() across 4 blocks of 8: True
used real forward kernel: False
x.grad is a one-hot at the (unique) global argmax: True
used real backward kernel: False
MultiBlockMaxOp gradcheck passed (random input, max is unique): True
```

### Discussion

The fallback branches make the same point Section 32.2's did — `x.max()` and the mask-divided-by-count formula don't know how many blocks a real launch would have used — but the REAL-launch code path this time is structurally richer, and worth reading even though this sandbox never reaches it: it is exactly the sequence a working GPU would need, no more and no less. `ctx.save_for_backward(x, out)` matters specifically because it lets `backward` skip re-deriving the global max from scratch; without it, `backward` would need its OWN copy of `kernel_max_partial` plus `kernel_reduce_partials_to_global_max` just to recover a value `forward` already computed once.

## 32.4 Capstone: Chaining an Elementwise Op into a Multi-Block Reduction

### Intuition

Chapter 31's capstone confirmed the chain rule composes correctly through a single-block reduction. The natural extension is whether the same composition still holds once the reduction itself is spread across multiple blocks and, for `max`, multiple sequential kernel passes — a chain that now has real internal structure on both sides of the elementwise operation.

### Background

`DoubleOp` is adapted with `ct.bid(0)`-based indexing so it too operates block by block over the 32-element input, then feeds into `MultiBlockSumOp` and, separately, `MultiBlockMaxOp`. Both chains are checked against hand-derived formulas and `gradcheck`, exactly as Chapter 31's single-block versions were.

### Worked Example 32.4.1 — DoubleOp feeding into a 4-block sum, and into a 4-block max

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
N = NUM_BLOCKS * BLOCK_SIZE

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=2 * x)

class DoubleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_double, (x, out, BLOCK_SIZE))
            return out
        except RuntimeError:
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

@ct.kernel
def kernel_sum_multiblock_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)

@ct.kernel
def kernel_sum_multiblock_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))

class MultiBlockSumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            acc = torch.zeros(1, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_forward, (x, acc, BLOCK_SIZE))
            return acc
        except RuntimeError:
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_backward, (grad_output, grad_input, BLOCK_SIZE))
            return grad_input
        except RuntimeError:
            return grad_output.expand(ctx.n).clone()

@ct.kernel
def kernel_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    local_count = ct.sum(mask, None)
    ct.store(partial_counts, index=(pid,), tile=ct.broadcast_to(local_count, (1,)))

@ct.kernel
def kernel_reduce_partial_counts(partial_counts, out, num_blocks: ct.Constant[int]):
    x = ct.load(partial_counts, (0,), (num_blocks,))
    y = ct.sum(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_multiblock_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)

class MultiBlockMaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            partials = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_partial, (x, partials, BLOCK_SIZE))
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partials_to_global_max, (partials, out, NUM_BLOCKS))
        except RuntimeError:
            out = x.max().reshape(1)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, global_max = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            partial_counts = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_count_partial, (x, global_max, partial_counts, BLOCK_SIZE))
            global_count = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partial_counts, (partial_counts, global_count, NUM_BLOCKS))
            grad_input = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_multiblock_backward, (x, global_max, grad_output, global_count, grad_input, BLOCK_SIZE))
            return grad_input
        except RuntimeError:
            mask = (x == global_max).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# Chain 1: y = sum(2x) across 32 elements, 4 blocks. dy/dx_i = 2 for every i.
x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y_sum = MultiBlockSumOp.apply(DoubleOp.apply(x))
print(f"sum(2x) matches 2*x.sum(): {torch.equal(y_sum, (2 * x.detach()).sum().reshape(1))}")

y_sum.backward()
print(f"x.grad matches uniform 2s across all {N}: {torch.allclose(x.grad, torch.full((N,), 2.0, dtype=torch.float64))}")

def chained_sum(x):
    return MultiBlockSumOp.apply(DoubleOp.apply(x))
ok_sum = torch.autograd.gradcheck(chained_sum, (x,))
print(f"gradcheck passed on DoubleOp -> MultiBlockSumOp chain: {ok_sum}")

print()

# Chain 2: y = max(2x) = 2*max(x) across 32 elements, 4 blocks.
x2 = torch.randn(N, dtype=torch.float64, requires_grad=True)
y_max = MultiBlockMaxOp.apply(DoubleOp.apply(x2))
print(f"max(2x) matches 2*x.max(): {torch.equal(y_max, (2 * x2.detach()).max().reshape(1))}")

y_max.backward()
expected = 2.0 * (x2.detach() == x2.detach().max()).double()
print(f"x2.grad matches 2 at global argmax, 0 elsewhere: {torch.equal(x2.grad, expected)}")

def chained_max(x):
    return MultiBlockMaxOp.apply(DoubleOp.apply(x))
ok_max = torch.autograd.gradcheck(chained_max, (x2,))
print(f"gradcheck passed on DoubleOp -> MultiBlockMaxOp chain: {ok_max}")
```

Genuinely run:

```
sum(2x) matches 2*x.sum(): True
x.grad matches uniform 2s across all 32: True
gradcheck passed on DoubleOp -> MultiBlockSumOp chain: True

max(2x) matches 2*x.max(): True
x2.grad matches 2 at global argmax, 0 elsewhere: True
gradcheck passed on DoubleOp -> MultiBlockMaxOp chain: True
```

### Discussion

Both chains pass exactly as Chapter 31's single-block versions did, for the same underlying reason stated there: block partitioning is an organization detail of HOW a reduction is computed on real hardware, never a change to WHAT the correct answer is. The `max` chain in particular is now backed by five real, independently-compiled kernels (two in `DoubleOp`'s and `MultiBlockMaxOp`'s forwards combined, three in `MultiBlockMaxOp`'s backward) working in a specific, necessary order — and `gradcheck` still can't see any of that structure, because it only ever tests the fallback path in this sandbox. What it CAN confirm, as it has every chapter since 29, is that the fallback path's formula is correct — and this chapter took the additional step of writing every one of the real kernels that formula would need to match, compiled and verified individually in Section 32.1.

## Chapter Summary

This chapter took Chapter 31's single-block reductions and asked what changes when the data doesn't fit in one block. For sum, the answer was almost nothing: `ct.atomic_add` supports `float32`, so a multi-block sum needs exactly the same two kernels a single-block sum needed, just launched over more blocks with `ct.bid(0)`-based indexing. For max, the answer was a great deal: `ct.atomic_max` does not support `float32` — a restriction Chapter 20 had already found, met here for the first time as an actual obstacle — so combining per-block maxima needs its own small second kernel rather than one atomic instruction, and correctly normalizing a tied gradient needs the SAME two-phase pattern applied a second time, to counts instead of maxima, before a final kernel can distribute the gradient at all. `MultiBlockMaxOp` ended up needing five real kernel launches to `MultiBlockSumOp`'s two, an asymmetry this chapter traced to one specific, already-known cause rather than treating it as an unexplained fact about "max being harder than sum" in general. A capstone confirmed that chaining an elementwise operation into either multi-block reduction still produces the mathematically correct result, exactly as `gradcheck` already confirmed for the single-block versions.

## Self-Check Questions

1. Chapter 20 found that `atomic_add` supports `float32` while `atomic_max` does not. State the practical, structural consequence this chapter drew from that fact: specifically, why does a multi-block SUM need only one kernel launch per direction, while a multi-block MAX needs more than one, purely because of this dtype restriction?
2. `kernel_max_partial` has every block write its own local max to `partials[pid]` using a plain `ct.store`, with no atomic operation anywhere. Multiple blocks are running "concurrently" here — so why does this NOT need the same kind of atomic protection Chapter 20's atomics exist for?
3. `MultiBlockMaxOp.forward` saves `(x, out)` via `ctx.save_for_backward`, where `out` is the already-computed global max. Suppose `backward` had instead re-run `kernel_max_partial` and `kernel_reduce_partials_to_global_max` itself, using only the saved `x`. Would this produce a WRONG answer, or merely a wasteful one — and what's the argument either way?
4. `kernel_max_multiblock_backward` divides by a GLOBAL tie count (`kernel_reduce_partial_counts`'s output, summed across every block), not by each block's own local count. Construct a concrete input where at least one block has zero elements equal to the global max, and explain what would go wrong in that block's own `grad_in` slice if it divided by ITS OWN local count instead.
5. Trace the exact kernel-launch sequence for `MultiBlockMaxOp` applied to a single input: how many total `ct.launch` calls happen across one full forward-then-backward cycle, and which specific dtype restriction is responsible for that number being larger than `MultiBlockSumOp`'s corresponding total?

## Where We Go Next

This chapter's multi-block kernels all assumed the input divides evenly into `NUM_BLOCKS` blocks of exactly `BLOCK_SIZE` elements each — 32 elements into 4 blocks of 8, with no leftover. Real data is rarely that convenient, and every `ct.load` this book has written since Chapter 1 has quietly relied on a block's full `tile_size` worth of data actually being there to read. What happens at the EDGE — the last block, when the array's length isn't a clean multiple of the block size, and some threads in that final block have no real element to load or store at all — is a question this book has deferred since its very first kernel, and it is the natural next place a backward-kernel chapter's discipline gets tested against.

## Worked Solutions

**1.** `atomic_add`'s `float32` support means a single kernel, launched once, can have every block accumulate its own partial sum directly into one shared scalar in a single read-modify-write step — no second kernel is needed because the hardware instruction itself does the combining. `atomic_max` rejecting `float32` means there is no equivalent single instruction available for combining per-block maxima on this dtype at all: the combination has to happen some OTHER way, and the way this chapter chose (each block writes its own slot, then a second kernel reduces those slots) requires an actual second kernel launch, because the first kernel's blocks have no shared memory or synchronization with each other to fall back on within a single launch. The dtype restriction is the entire reason: an `int32` accumulator could have used `atomic_max` directly and needed only one kernel, exactly like `atomic_add`'s `float32` case.

**2.** Chapter 20's atomics exist specifically for the case where multiple blocks might read-modify-write the SAME memory location, where a plain (non-atomic) read-then-write from two blocks could interleave and lose one block's update. `kernel_max_partial` never has that problem: `ct.store(partials, index=(pid,), ...)` writes to `partials[pid]`, and since `pid = ct.bid(0)` is a DIFFERENT value for every block, every block is writing to a DIFFERENT, non-overlapping memory location. Two blocks running concurrently and each writing to their own, distinct slot is exactly as safe as two people each writing in their own notebook at the same table — no coordination is needed because nothing is shared.

**3.** It would produce the CORRECT answer, not a wrong one — `kernel_max_partial` and `kernel_reduce_partials_to_global_max` are pure functions of `x` alone, so re-running them in `backward` would recompute the exact same global max `forward` already found. The argument against doing so is purely about waste: `forward` already paid the cost of two kernel launches to compute this value once, and `ctx.save_for_backward` exists specifically so that value can be handed to `backward` directly rather than recomputed from scratch — the same principle Chapter 30's `SquareOp` used when it saved `x` itself (rather than recomputing anything derived from it) so `backward` wouldn't need `forward`'s intermediate work a second time.

**4.** Construct `x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ..., up to some block containing only values less than 8.0]` split into blocks such that the global max, `8.0`, appears in only ONE block (say block 0), and some OTHER block (say block 2) contains no element equal to `8.0` at all. If block 2 divided by its OWN local count of ties with the global max, that count would be `0` (no element in block 2 equals `8.0`), and `mask * grad_output / count` would divide by zero for every element in block 2's slice of `grad_in` — producing `NaN` (or `inf`) values in a region of the gradient that should correctly be all zeros, since none of block 2's elements contributed to the max at all. Dividing by the GLOBAL count instead (computed once, correctly, across every block, and guaranteed at least `1` since the max itself must equal something) avoids this entirely: block 2's mask is all zeros, and `0 * grad_output / global_count` is safely `0` everywhere, regardless of what any other block's tie count happens to be.

**5.** One full forward-then-backward cycle for `MultiBlockMaxOp` launches five kernels total: `kernel_max_partial` and `kernel_reduce_partials_to_global_max` in `forward` (two), then `kernel_max_count_partial`, `kernel_reduce_partial_counts`, and `kernel_max_multiblock_backward` in `backward` (three). `MultiBlockSumOp`'s corresponding total is two: `kernel_sum_multiblock_forward` (one, using `atomic_add` to combine across blocks in the same launch) and `kernel_sum_multiblock_backward` (one, needing no combination at all since broadcasting requires no cross-block information). The dtype restriction responsible for the gap is exactly the one from Question 1: `atomic_max` does not support `float32`, so every place `MultiBlockSumOp` could resolve a cross-block combination inside a single kernel via `atomic_add`, `MultiBlockMaxOp` needed a separate, explicit reduction kernel instead — twice over, once for the max itself and once for the tie count that max makes necessary.

## Complete Runnable Code

### File: `01_multiblock_sum_and_max_kernels.py`

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

def sig_n(n_arrays, const_val):
    return ct.compilation.KernelSignature(
        [array_param(1) for _ in range(n_arrays)] + [ct.compilation.ConstantConstraint(const_val)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- confirm the float32 atomic_max restriction Chapter 20 already found,
# now hit in a genuinely new context: trying to combine per-block maxima. ---
@ct.kernel
def kernel_bad_atomic_max(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.atomic_max(acc, (0,), partial_max)

try:
    n = compile_bytes(kernel_bad_atomic_max, sig_n(2, 8))
    print(f"atomic_max on float32 accumulator: {n} bytes (unexpected)")
except Exception as e:
    print(f"atomic_max on float32 accumulator: {type(e).__name__}: {e}")

print()

# --- Sum: atomic_add DOES support float32, so one kernel per direction. ---
@ct.kernel
def kernel_sum_multiblock_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)
n = compile_bytes(kernel_sum_multiblock_forward, sig_n(2, 8))
print(f"kernel_sum_multiblock_forward: {n} cubin bytes")

@ct.kernel
def kernel_sum_multiblock_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))
n = compile_bytes(kernel_sum_multiblock_backward, sig_n(2, 8))
print(f"kernel_sum_multiblock_backward: {n} cubin bytes")

print()

# --- Max: no float32 atomic_max, so two phases -- each block writes to its
# OWN slot (no synchronization needed), then a small second kernel reduces
# those per-block partials down to one value. ---
@ct.kernel
def kernel_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))
n = compile_bytes(kernel_max_partial, sig_n(2, 8))
print(f"kernel_max_partial: {n} cubin bytes")

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))
n = compile_bytes(kernel_reduce_partials_to_global_max, sig_n(2, 4))
print(f"kernel_reduce_partials_to_global_max: {n} cubin bytes")

@ct.kernel
def kernel_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    local_count = ct.sum(mask, None)
    ct.store(partial_counts, index=(pid,), tile=ct.broadcast_to(local_count, (1,)))
n = compile_bytes(kernel_max_count_partial, sig_n(3, 8))
print(f"kernel_max_count_partial: {n} cubin bytes")

@ct.kernel
def kernel_reduce_partial_counts(partial_counts, out, num_blocks: ct.Constant[int]):
    x = ct.load(partial_counts, (0,), (num_blocks,))
    y = ct.sum(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))
n = compile_bytes(kernel_reduce_partial_counts, sig_n(2, 4))
print(f"kernel_reduce_partial_counts: {n} cubin bytes")

@ct.kernel
def kernel_max_multiblock_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)
n = compile_bytes(kernel_max_multiblock_backward, sig_n(5, 8))
print(f"kernel_max_multiblock_backward: {n} cubin bytes")
```

### File: `02_multiblocksumop_atomics_make_this_easy.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
N = NUM_BLOCKS * BLOCK_SIZE

@ct.kernel
def kernel_sum_multiblock_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)

@ct.kernel
def kernel_sum_multiblock_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))

class MultiBlockSumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            acc = torch.zeros(1, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_forward, (x, acc, BLOCK_SIZE))
            ctx.used_real_kernel_forward = True
            return acc
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_backward, (grad_output, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output.expand(ctx.n).clone()

x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y = MultiBlockSumOp.apply(x)
print(f"forward matches x.sum() across {NUM_BLOCKS} blocks of {BLOCK_SIZE}: {torch.equal(y, x.detach().sum().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad is uniformly 1 across all {N} elements: {torch.allclose(x.grad, torch.ones(N, dtype=torch.float64))}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(MultiBlockSumOp.apply, (x,))
print(f"MultiBlockSumOp gradcheck passed: {ok}")
```

### File: `03_multiblockmaxop_three_passes_for_one_gradient.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
N = NUM_BLOCKS * BLOCK_SIZE

@ct.kernel
def kernel_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    local_count = ct.sum(mask, None)
    ct.store(partial_counts, index=(pid,), tile=ct.broadcast_to(local_count, (1,)))

@ct.kernel
def kernel_reduce_partial_counts(partial_counts, out, num_blocks: ct.Constant[int]):
    x = ct.load(partial_counts, (0,), (num_blocks,))
    y = ct.sum(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_multiblock_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)

class MultiBlockMaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            partials = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_partial, (x, partials, BLOCK_SIZE))
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partials_to_global_max, (partials, out, NUM_BLOCKS))
            ctx.used_real_kernel_forward = True
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            out = x.max().reshape(1)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, global_max = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            partial_counts = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_count_partial, (x, global_max, partial_counts, BLOCK_SIZE))
            global_count = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partial_counts, (partial_counts, global_count, NUM_BLOCKS))
            grad_input = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_multiblock_backward, (x, global_max, grad_output, global_count, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x == global_max).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y = MultiBlockMaxOp.apply(x)
print(f"forward matches x.max() across {NUM_BLOCKS} blocks of {BLOCK_SIZE}: {torch.equal(y, x.detach().max().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
expected = (x.detach() == x.detach().max()).double()
print(f"x.grad is a one-hot at the (unique) global argmax: {torch.equal(x.grad, expected)}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(MultiBlockMaxOp.apply, (x,))
print(f"MultiBlockMaxOp gradcheck passed (random input, max is unique): {ok}")
```

### File: `04_capstone_chaining_into_multiblock_reductions.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
N = NUM_BLOCKS * BLOCK_SIZE

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=2 * x)

class DoubleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_double, (x, out, BLOCK_SIZE))
            return out
        except RuntimeError:
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

@ct.kernel
def kernel_sum_multiblock_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)

@ct.kernel
def kernel_sum_multiblock_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))

class MultiBlockSumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            acc = torch.zeros(1, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_forward, (x, acc, BLOCK_SIZE))
            return acc
        except RuntimeError:
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_sum_multiblock_backward, (grad_output, grad_input, BLOCK_SIZE))
            return grad_input
        except RuntimeError:
            return grad_output.expand(ctx.n).clone()

@ct.kernel
def kernel_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    local_count = ct.sum(mask, None)
    ct.store(partial_counts, index=(pid,), tile=ct.broadcast_to(local_count, (1,)))

@ct.kernel
def kernel_reduce_partial_counts(partial_counts, out, num_blocks: ct.Constant[int]):
    x = ct.load(partial_counts, (0,), (num_blocks,))
    y = ct.sum(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_max_multiblock_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)

class MultiBlockMaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            partials = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_partial, (x, partials, BLOCK_SIZE))
            out = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partials_to_global_max, (partials, out, NUM_BLOCKS))
        except RuntimeError:
            out = x.max().reshape(1)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, global_max = ctx.saved_tensors
        try:
            stream = torch.cuda.current_stream()
            partial_counts = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_count_partial, (x, global_max, partial_counts, BLOCK_SIZE))
            global_count = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partial_counts, (partial_counts, global_count, NUM_BLOCKS))
            grad_input = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_max_multiblock_backward, (x, global_max, grad_output, global_count, grad_input, BLOCK_SIZE))
            return grad_input
        except RuntimeError:
            mask = (x == global_max).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# Chain 1: y = sum(2x) across 32 elements, 4 blocks. dy/dx_i = 2 for every i.
x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y_sum = MultiBlockSumOp.apply(DoubleOp.apply(x))
print(f"sum(2x) matches 2*x.sum(): {torch.equal(y_sum, (2 * x.detach()).sum().reshape(1))}")

y_sum.backward()
print(f"x.grad matches uniform 2s across all {N}: {torch.allclose(x.grad, torch.full((N,), 2.0, dtype=torch.float64))}")

def chained_sum(x):
    return MultiBlockSumOp.apply(DoubleOp.apply(x))
ok_sum = torch.autograd.gradcheck(chained_sum, (x,))
print(f"gradcheck passed on DoubleOp -> MultiBlockSumOp chain: {ok_sum}")

print()

# Chain 2: y = max(2x) = 2*max(x) across 32 elements, 4 blocks.
x2 = torch.randn(N, dtype=torch.float64, requires_grad=True)
y_max = MultiBlockMaxOp.apply(DoubleOp.apply(x2))
print(f"max(2x) matches 2*x.max(): {torch.equal(y_max, (2 * x2.detach()).max().reshape(1))}")

y_max.backward()
expected = 2.0 * (x2.detach() == x2.detach().max()).double()
print(f"x2.grad matches 2 at global argmax, 0 elsewhere: {torch.equal(x2.grad, expected)}")

def chained_max(x):
    return MultiBlockMaxOp.apply(DoubleOp.apply(x))
ok_max = torch.autograd.gradcheck(chained_max, (x2,))
print(f"gradcheck passed on DoubleOp -> MultiBlockMaxOp chain: {ok_max}")
```
