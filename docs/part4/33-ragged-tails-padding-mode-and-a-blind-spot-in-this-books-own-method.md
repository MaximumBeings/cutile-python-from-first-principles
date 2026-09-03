# Chapter 33: Ragged Tails, padding_mode, and a Blind Spot in This Book's Own Method

> "Chapter 32's multi-block kernels all assumed 32 elements split evenly into 4 blocks of 8, with nothing left over. Real data rarely cooperates. The moment an array's length isn't a clean multiple of the block size, the last block's `ct.load` reaches past the array's real end — and for the first time in this book, the compiler has no opinion at all about whether the choice made there is correct."

**What you will understand by the end of this chapter:**

- That `ct.load` fills a partially out-of-bounds tile according to `padding_mode`, defaulting to `PaddingMode.UNDETERMINED` — and that `ct.store` handles the identical situation with no such decision at all, simply ignoring out-of-bounds elements
- A genuine confirmation, via a faithful host-side simulation of `cuda.tile`'s own documented padding semantics, that `PaddingMode.ZERO` is always safe for `ct.sum` but can silently produce the WRONG answer for `ct.max` when the real data is negative — and that `PaddingMode.NEG_INF` is what `ct.max` actually needs
- That every one of the six `PaddingMode` values compiles without complaint, under Chapter 10's naming-confound control, to the identical byte count — the compiler cannot and does not distinguish a correct choice from an incorrect one, because that distinction is a fact about runtime DATA, not about the kernel's source code
- The first genuinely-acknowledged limit of this book's own driver-free, `export_kernel`-only verification method: a class of bug that compiles perfectly cleanly and can only be caught by actually running the kernel against real, ragged data
- A capstone building complete `RaggedSumOp` and `RaggedMaxOp` operations over a genuinely non-evenly-divisible 13-element input split across 2 blocks of 8, with every padding-sensitive kernel threaded through with the correct choice, verified end to end with `gradcheck`

**What you need to know first:**

- Chapter 32's multi-block `MultiBlockSumOp`/`MultiBlockMaxOp`, including the two-phase partial-then-reduce pattern this chapter reuses for a ragged `MaxOp`.
- Chapter 10's naming-confound discipline, applied here to a full sweep of `PaddingMode` values rather than a single pair of kernels.
- This book's `export_kernel`-only compilation workflow, and the honest recognition that it has, until now, always been ABLE to catch the bugs this book went looking for.

## 33.1 The Tail: When N Isn't a Clean Multiple of block_size

### Intuition

Every kernel this book has written assumes `ct.load(a, index=(pid,), shape=(block_size,))` reads a full, real tile. When an array's length isn't a multiple of `block_size`, the LAST block's load reaches past the array's actual end — `cuda.tile`'s own documentation is explicit that these out-of-bounds elements are filled according to `padding_mode`, and that the default, `PaddingMode.UNDETERMINED`, means exactly what it says: unspecified. Six named alternatives exist (`ZERO`, `NEG_ZERO`, `NAN`, `POS_INF`, `NEG_INF`, plus `UNDETERMINED` itself), and — unlike almost everything else this book has tested — the compiler accepts every single one without comment, on a kernel that has no idea whether the real array it will eventually run against is ragged at all.

### Background

An `ArrayConstraint` fixes only an array's rank and dtype, never its concrete length — so a kernel written assuming a clean multiple of `block_size` and one written for a genuinely ragged array are, as far as `export_kernel` is concerned, indistinguishable. This section compiles a small family of kernels that make this concrete: a sum kernel using the default (`UNDETERMINED`) padding, one using `ZERO`, two `max`-partial kernels differing only in `ZERO` versus `NEG_INF`, all six `PaddingMode` values compiled under Chapter 10's naming control, and a kernel demonstrating that `ct.store` needs no `padding_mode` parameter at all.

### Worked Example 33.1.1 — every padding_mode compiles; the compiler has no opinion on which one is correct

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

# The compiler has no idea the real array has 13 elements, not a multiple of
# block_size=8 -- an ArrayConstraint only fixes ndim and dtype, never a
# concrete length. So a kernel written for a ragged tail compiles IDENTICALLY
# to one that assumes a clean multiple, whichever padding_mode is chosen.

@ct.kernel
def kernel_sum_undetermined(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))  # default: UNDETERMINED
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)
n = compile_bytes(kernel_sum_undetermined, sig_2in)
print(f"kernel_sum_undetermined (default padding_mode): {n} cubin bytes")

@ct.kernel
def kernel_sum_zero_pad(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)
n = compile_bytes(kernel_sum_zero_pad, sig_2in)
print(f"kernel_sum_zero_pad: {n} cubin bytes")

@ct.kernel
def kernel_max_zero_pad(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))
n = compile_bytes(kernel_max_zero_pad, sig_2in)
print(f"kernel_max_zero_pad: {n} cubin bytes")

@ct.kernel
def kernel_max_neg_inf_pad(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))
n = compile_bytes(kernel_max_neg_inf_pad, sig_2in)
print(f"kernel_max_neg_inf_pad: {n} cubin bytes")

# All six PaddingMode values, confirmed accepted at compile time -- the
# compiler never distinguishes a "correct for this reduction" choice from
# an incorrect one; that distinction only exists at runtime, against real
# ragged data.
for mode in ct.PaddingMode:
    @ct.kernel
    def kernel_probe(a, c, block_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=mode)
        ct.store(c, index=(pid,), tile=x)
    n = compile_bytes(kernel_probe, sig_2in)
    print(f"padding_mode={mode.value}: compiles, {n} cubin bytes")

# Store's own documented asymmetry: an out-of-bounds STORE is simply
# ignored, with no padding_mode-equivalent decision to make at all.
@ct.kernel
def kernel_ragged_store(a, c, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(c, index=(pid,), tile=x)
n = compile_bytes(kernel_ragged_store, sig_2in)
print(f"kernel_ragged_store (out-of-bounds store, no padding_mode param exists for it): {n} cubin bytes")
```

Genuinely run:

```
kernel_sum_undetermined (default padding_mode): 22016 cubin bytes
kernel_sum_zero_pad: 21888 cubin bytes
kernel_max_zero_pad: 21024 cubin bytes
kernel_max_neg_inf_pad: 21168 cubin bytes
padding_mode=undetermined: compiles, 21024 cubin bytes
padding_mode=zero: compiles, 21024 cubin bytes
padding_mode=neg_zero: compiles, 21024 cubin bytes
padding_mode=nan: compiles, 21024 cubin bytes
padding_mode=pos_inf: compiles, 21024 cubin bytes
padding_mode=neg_inf: compiles, 21024 cubin bytes
kernel_ragged_store (out-of-bounds store, no padding_mode param exists for it): 21152 cubin bytes
```

### Discussion

All six `PaddingMode` values compile, under the identical function name `kernel_probe`, to the identical byte count (21024) — a genuine, naming-controlled confirmation that the compiler treats the choice of padding value as compile-time-neutral information, encoded the same way regardless of which of the six it is. That is unlike every earlier confound this book has met: Chapter 19's docstring-overpromise, Chapter 24's source-line-position confound, and every `TileTypeError` since Chapter 10 were all things the COMPILER caught and reported. A wrong `padding_mode` choice for a given reduction is not something `export_kernel` can catch, has any way to catch, or was ever going to catch — because whether `ZERO` or `NEG_INF` is "correct" depends entirely on what values the REAL array holds at runtime, which no `ArrayConstraint` this book has ever written encodes. `kernel_ragged_store`'s successful compilation makes the matching point from the other direction: `ct.store` has no `padding_mode` parameter at all to get right or wrong, because its documented behavior — silently ignore out-of-bounds elements — requires no choice in the first place.

## 33.2 Padding Value Matters: ZERO Is Right for Sum, Wrong for Max

### Intuition

`cuda.tile`'s own documentation states plainly what each `padding_mode` fills out-of-bounds elements with — `ZERO` with `0.0`, `NEG_INF` with negative infinity, and so on. Since this sandbox cannot launch a real kernel to observe padding in action, the next best thing this book's discipline allows is a host-side simulation that is faithful to those DOCUMENTED semantics exactly — padding a real array up to the next full multiple of `block_size` with a chosen value, then applying the same reduction a real kernel would, and comparing against the true answer computed over only the real elements.

### Background

A 13-element array (`N = 13`, `BLOCK_SIZE = 8`, `NUM_BLOCKS = 2`) is padded to 16 elements two ways: with `0.0` and with `-inf`. For `sum`, only the `ZERO`-padded version is tested, against arbitrary (mixed-sign) data — since adding `0.0` to a running total can never change the result, regardless of what the real data looks like. For `max`, BOTH paddings are tested against a deliberately ALL-NEGATIVE dataset — the case that exposes the difference, since a padding value of `0.0` is greater than every real value in the array, and would otherwise never appear as a candidate at all.

### Worked Example 33.2.1 — a faithful simulation of cuda.tile's own documented padding semantics

```python
import torch

torch.manual_seed(0)

N = 13
BLOCK_SIZE = 8
NUM_BLOCKS = 2  # ceil(13 / 8)
PADDED_LEN = NUM_BLOCKS * BLOCK_SIZE  # 16, 3 elements past the real array

# This simulates, on the host, exactly what ct.load(..., padding_mode=X)
# documents that it does at each out-of-bounds position: fill with X's
# value. No real kernel launch is possible in this sandbox, but the
# padding semantics themselves are fully specified in cuda.tile's own
# docstrings, so a faithful host-side model of them is a genuine test of
# the CHOICE of padding_mode, independent of whether the kernel can run.

def simulate_padded_reduction(real_data, pad_value, reduce_fn):
    padded = torch.full((PADDED_LEN,), pad_value, dtype=real_data.dtype)
    padded[:N] = real_data
    return reduce_fn(padded)

# --- Sum: any real dataset, ZERO padding is always safe. ---
data_sum = torch.randn(N, dtype=torch.float64)
true_sum = data_sum.sum()
sim_sum_zero = simulate_padded_reduction(data_sum, 0.0, torch.sum)
print(f"true sum of the {N} real elements: {true_sum.item()}")
print(f"simulated ZERO-padded sum over {PADDED_LEN}: {sim_sum_zero.item()}")
print(f"ZERO padding gives the correct sum: {torch.equal(true_sum, sim_sum_zero)}")

print()

# --- Max: an all-negative dataset exposes ZERO padding as WRONG. ---
data_max = -torch.rand(N, dtype=torch.float64) - 0.1  # every real value in [-1.1, -0.1]
true_max = data_max.max()
sim_max_zero = simulate_padded_reduction(data_max, 0.0, torch.max)
sim_max_neg_inf = simulate_padded_reduction(data_max, float("-inf"), torch.max)
print(f"true max of the {N} real (all-negative) elements: {true_max.item()}")
print(f"simulated ZERO-padded max: {sim_max_zero.item()}")
print(f"ZERO padding gives the correct max: {torch.equal(true_max, sim_max_zero)}")
print(f"simulated NEG_INF-padded max: {sim_max_neg_inf.item()}")
print(f"NEG_INF padding gives the correct max: {torch.equal(true_max, sim_max_neg_inf)}")
```

Genuinely run:

```
true sum of the 13 real elements: -3.998410033730461
simulated ZERO-padded sum over 16: -3.998410033730461
ZERO padding gives the correct sum: True

true max of the 13 real (all-negative) elements: -0.20973526023744607
simulated ZERO-padded max: 0.0
ZERO padding gives the correct max: False
simulated NEG_INF-padded max: -0.20973526023744607
NEG_INF padding gives the correct max: True
```

### Discussion

The `sum` case confirms the boring, correct default: padding with the additive identity never perturbs a sum, whatever the real data looks like. The `max` case is the one worth remembering: `ZERO`-padding the all-negative array reports `0.0` as the maximum — a value that never appeared anywhere in the real 13 elements, and is wrong by construction, since every real element is strictly less than `0.0`. `NEG_INF` padding, by contrast, contributes a value that can never win a max comparison against any finite real number, so it reproduces the true maximum exactly. Nothing about this bug would show up as a crash, an exception, or an obviously-wrong-looking number in the general case — it would show up as a silently incorrect answer that happens to look plausible, specifically in the case where every real value in a block is negative. This is precisely the kind of error this book's `export_kernel`-only method has no way to catch on its own: Section 33.1 already confirmed the compiler accepts `ZERO` padding on a `max` kernel exactly as readily as it accepts `NEG_INF`.

## 33.3 Capstone: A Ragged Multi-Block Sum and Max, End to End

### Intuition

Sections 33.1 and 33.2 established the pieces separately: which `PaddingMode` values compile, and which ones are mathematically correct for which reduction. The capstone wires both together into complete `torch.autograd.Function` operations, over a genuinely ragged 13-element input, extending Chapter 32's `MultiBlockSumOp`/`MultiBlockMaxOp` with the one change this chapter is about: every kernel that loads the real array now specifies the padding mode Section 33.2 confirmed is correct.

### Background

`RaggedSumOp` reuses Chapter 32's exact two-kernel shape, with `kernel_ragged_sum_forward` now loading with `padding_mode=ct.PaddingMode.ZERO`. Its backward kernel needs NO padding-related change at all: `grad_input` is allocated at the real length (13), and the tail block's store — writing 8 values starting at index 8 — has 3 of those values fall outside `grad_input` entirely; `ct.store`'s documented "out-of-bounds elements are ignored" behavior handles this automatically. `RaggedMaxOp` reuses Chapter 32's three-kernel backward shape, with `padding_mode=ct.PaddingMode.NEG_INF` threaded through EVERY kernel that loads the real array — the initial per-block max, and the per-block tie-count — so that padded, non-existent elements can never be mistaken for the true maximum in either pass. The test data for `RaggedMaxOp` is deliberately all-negative, the exact case Section 33.2 showed `ZERO` padding would have gotten wrong.

### Worked Example 33.3.1 — RaggedSumOp and RaggedMaxOp over a genuinely non-evenly-divisible input

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

N = 13
BLOCK_SIZE = 8
NUM_BLOCKS = 2  # ceil(13 / 8) -- the last block has only 5 real elements

@ct.kernel
def kernel_ragged_sum_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)

@ct.kernel
def kernel_ragged_sum_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))

class RaggedSumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            acc = torch.zeros(1, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_sum_forward, (x, acc, BLOCK_SIZE))
            ctx.used_real_kernel_forward = True
            return acc
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            # grad_in is allocated at the REAL length (13), not the padded
            # length (16) -- the tail block's store writes 8 values starting
            # at index 8, so elements [13, 16) fall outside grad_in entirely.
            # ct.store documents out-of-bounds elements as simply ignored,
            # so this needs no padding_mode-equivalent decision at all.
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_sum_backward, (grad_output, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output.expand(ctx.n).clone()

x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y = RaggedSumOp.apply(x)
print(f"forward matches x.sum() over the real {N} elements: {torch.equal(y, x.detach().sum().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad is uniformly 1 across all {N} real elements: {torch.allclose(x.grad, torch.ones(N, dtype=torch.float64))}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(RaggedSumOp.apply, (x,))
print(f"RaggedSumOp gradcheck passed on a ragged ({N}-element, {NUM_BLOCKS}-block) input: {ok}")

print()

@ct.kernel
def kernel_ragged_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_ragged_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
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
def kernel_ragged_max_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)

class RaggedMaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            partials = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_max_partial, (x, partials, BLOCK_SIZE))
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
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_max_count_partial, (x, global_max, partial_counts, BLOCK_SIZE))
            global_count = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partial_counts, (partial_counts, global_count, NUM_BLOCKS))
            grad_input = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_max_backward, (x, global_max, grad_output, global_count, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x == global_max).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# All-negative real data, deliberately -- this is exactly the case Section
# 33.2 showed ZERO padding gets wrong, now wired through a complete
# multi-block Function on the real, ragged 13-element input.
x2 = -torch.rand(N, dtype=torch.float64) - 0.1
x2.requires_grad_(True)
y2 = RaggedMaxOp.apply(x2)
print(f"forward matches x.max() over the real {N} (all-negative) elements: {torch.equal(y2, x2.detach().max().reshape(1))}")
print(f"used real forward kernel: {y2.grad_fn.used_real_kernel_forward}")

y2.sum().backward()
expected = (x2.detach() == x2.detach().max()).double()
print(f"x2.grad is a one-hot at the (unique) global argmax: {torch.equal(x2.grad, expected)}")
print(f"used real backward kernel: {y2.grad_fn.used_real_kernel_backward}")

ok2 = torch.autograd.gradcheck(RaggedMaxOp.apply, (x2,))
print(f"RaggedMaxOp gradcheck passed on ragged, all-negative input: {ok2}")
```

Genuinely run:

```
forward matches x.sum() over the real 13 elements: True
used real forward kernel: False
x.grad is uniformly 1 across all 13 real elements: True
used real backward kernel: False
RaggedSumOp gradcheck passed on a ragged (13-element, 2-block) input: True

forward matches x.max() over the real 13 (all-negative) elements: True
used real forward kernel: False
x2.grad is a one-hot at the (unique) global argmax: True
used real backward kernel: False
RaggedMaxOp gradcheck passed on ragged, all-negative input: True
```

### Discussion

Both `Function`s report the correct forward value and pass `gradcheck` on a 13-element input that does not divide evenly by the 8-element block size — but it is worth being precise about what this DOES and does NOT confirm. The fallback formulas (`x.sum()`, `x.max()`, and the mask/count backward) are plain `torch` operations on the real, 13-length tensor; they never see a padded, 16-length array at all, and have no `padding_mode`-equivalent decision to get right or wrong. `gradcheck` passing therefore confirms the fallback's own math is correct, exactly as it has since Chapter 29 — it says nothing about whether the REAL kernel path, with its `padding_mode=ct.PaddingMode.NEG_INF` calls, would produce the same answer on real hardware. That confirmation is the one this chapter built separately, and by a different method: Section 33.1 confirmed those exact kernels compile, and Section 33.2's host-side simulation confirmed `NEG_INF` (not `ZERO`) is the mathematically correct choice for a `max` reduction's padding, using `cuda.tile`'s own documented semantics rather than a live run. Together, the two halves are the most this book's method can honestly claim about a ragged tail: the fallback's answer is right, and the real kernel's padding CHOICE is right by the library's own documentation and a faithful simulation of it — with the actual, hardware-executed agreement between the two remaining, like every `ct.launch` claim since Chapter 29, outside what this sandbox can verify directly.

## Chapter Summary

This chapter met the tail Chapter 32 deferred: what happens when an array's length isn't a clean multiple of the block size, and the last block's `ct.load` reaches past the real data. `cuda.tile` documents the answer precisely — out-of-bounds elements are filled according to `padding_mode`, defaulting to `UNDETERMINED` — but for the first time, this book's own `export_kernel`-only method could not settle whether a given choice was CORRECT, only whether it compiled, and all six `PaddingMode` values compiled identically under naming control. A host-side simulation, faithful to `cuda.tile`'s own documented padding semantics, filled that gap the only way this sandbox allows: confirming `ZERO` padding is always safe for `sum` but silently wrong for `max` on negative data, where `NEG_INF` is what correctness actually requires. `ct.store`'s matching asymmetry — out-of-bounds elements simply ignored, no `padding_mode` parameter to get right at all — meant `RaggedSumOp`'s backward kernel needed no special handling whatsoever for the identical ragged tail its forward kernel had to reason about carefully. The capstone wired both reductions into complete, `gradcheck`-verified `Function`s over a genuinely non-evenly-divisible 13-element input, while being explicit about the boundary of that verification: `gradcheck` confirms the fallback's math, not the real kernel's padding choice, which this chapter could only confirm by a different, non-execution-based method — the first chapter to need one.

## Self-Check Questions

1. Section 33.1's loop compiled all six `PaddingMode` values under the identical name `kernel_probe`, landing on the identical byte count. Contrast this with Chapter 19's docstring-overpromise or Chapter 24's source-line confound: what made THOSE bugs catchable by this book's compile-only method, and what specifically about a padding-mode choice makes it uncatchable by the same method?
2. Explain, without appealing to the simulation's printed numbers, WHY `0.0` is always a safe padding value for `ct.sum` regardless of the sign of the real data, while the same reasoning does not extend to `ct.max`.
3. `RaggedSumOp.backward`'s kernel needed no `padding_mode`-equivalent change to handle the ragged tail correctly, while `RaggedSumOp.forward`'s kernel did. Trace exactly why the STORE side of this operation is unaffected by raggedness in a way the LOAD side is not.
4. Suppose `kernel_ragged_max_count_partial` had used `padding_mode=ct.PaddingMode.ZERO` instead of `NEG_INF`, on the SAME all-negative test data from the capstone. Would the tie count it computes come out too high, too low, or unaffected — and what specific value would the padded positions incorrectly be compared against the global max?
5. `gradcheck` passed for both `RaggedSumOp` and `RaggedMaxOp` in the capstone. Given this chapter's own Discussion, would a passing `gradcheck` result have caught a mistake where `kernel_ragged_max_partial`'s `padding_mode` had been left at `ZERO` instead of `NEG_INF`? Explain why or why not, tying your answer to which code path `gradcheck` actually exercises in this sandbox.

## Where We Go Next

This chapter's ragged tail was a 1D problem: one array, one dimension that didn't divide evenly. Several of this book's most useful kernels — Chapter 21's matrix multiplication, Chapter 22's transpose — operate on 2D tiles, where EACH of two dimensions can independently fail to divide evenly by its own tile size, and a single block can find itself ragged along one axis while perfectly aligned along the other. Whether `padding_mode`'s single scalar fill value is even sufficient once a tile's out-of-bounds region is a genuinely two-dimensional shape, rather than a simple trailing run of extra elements, is a question this chapter's one-dimensional treatment does not yet answer.

## Worked Solutions

**1.** Chapter 19's docstring-overpromise and Chapter 24's source-line confound were both catchable because the compiler had enough information, at compile time, to determine that something was WRONG — a format specifier the compiler could statically reject, or a source position the compiler could statically observe affecting its own output. A padding-mode choice is different in kind: whether `ZERO` or `NEG_INF` is correct for a given kernel depends entirely on what VALUES the real array holds when the kernel actually runs — information an `ArrayConstraint` (fixing only rank and dtype) never carries, and information no static analysis of the kernel's SOURCE CODE could ever recover, because the "bug" isn't in the code's structure, it's in a mismatch between a compile-time choice and a runtime fact the compiler has no access to.

**2.** `ct.sum` combines every tile element with addition, and `0.0` is addition's identity element — adding `0.0` anywhere in a sum, any number of times, leaves the result exactly unchanged, regardless of whether the real data is positive, negative, or a mix; there is no real dataset for which `ZERO` padding could ever perturb a sum. `ct.max` combines elements by taking the larger of two, and `0.0` is not an identity element for that operation at all — it is simply a fixed, finite candidate value that will win the comparison against any real data point smaller than it. Since a real, all-negative dataset consists ENTIRELY of values smaller than `0.0`, every one of them, the padding value doesn't just risk perturbing the answer, it is GUARANTEED to win, replacing the true (negative) maximum with `0.0` every time this configuration arises.

**3.** `ct.store`'s documented behavior for a partially out-of-bounds tile is to simply ignore whichever elements fall outside the array — there is no "wrong value that could win a comparison" the way `ZERO` padding creates for `ct.max`, because nothing is ever WRITTEN to those out-of-bounds locations in the first place; they're skipped, not filled with a placeholder. `RaggedSumOp`'s backward kernel broadcasts one gradient value across all 8 positions of a tile and stores it — for the tail block, 3 of those 8 stored positions fall past `grad_input`'s real length of 13, and those 3 writes are simply dropped, leaving `grad_input`'s real 13 elements correctly populated with no garbage or undefined values anywhere. The load side has no equivalent "just don't produce a value" option: a `ct.sum` or `ct.max` computed over a tile NEEDS every position in that tile to hold SOME value to combine, so `ct.load` must choose one — the entire reason `padding_mode` exists as a decision at all.

**4.** The tie count would come out TOO HIGH. `ZERO` padding would fill the tail block's out-of-bounds positions with `0.0`; since the test data is all-negative, `0.0` is strictly greater than the true global max (which is itself negative), so the padded `0.0` positions would never be compared against a phantom maximum of `0.0` in this specific scenario — however, the count kernel compares against `global_max_arr`'s ALREADY-CORRECT value (computed earlier with `NEG_INF` padding), and since `0.0 != global_max` (a negative number), those padded positions would correctly evaluate `False` in the mask here. The genuinely dangerous case is different: if the global max itself were exactly `0.0` (achievable with non-negative test data), `ZERO`-padded phantom positions would incorrectly equal it, inflating the tie count with positions that were never real data at all — undercounting how much of the gradient each REAL tied position should receive, since the gradient gets divided by an artificially large count.

**5.** No — a passing `gradcheck` would NOT have caught that mistake. `gradcheck`, in this sandbox, only ever exercises the `except RuntimeError:` fallback branch, since `torch.cuda.current_stream()` fails before any `ct.launch` call is ever reached; the fallback computes `x.max()` directly on the real, unpadded tensor and has no dependency whatsoever on what `padding_mode` the real kernel's source code specifies. A `padding_mode=ZERO` mistake in `kernel_ragged_max_partial` lives entirely in a code path `gradcheck` never runs here — exactly the gap this chapter's Section 33.3 Discussion named explicitly: `gradcheck` verifies the fallback's mathematics, and only the SEPARATE compile-plus-simulation method from Sections 33.1 and 33.2 (confirming the kernel compiles, and confirming `NEG_INF` is the value cuda.tile's own documentation says is needed) can speak to whether the real kernel's padding choice is correct.

## Complete Runnable Code

### File: `01_ragged_tail_kernels_compile.py`

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

# The compiler has no idea the real array has 13 elements, not a multiple of
# block_size=8 -- an ArrayConstraint only fixes ndim and dtype, never a
# concrete length. So a kernel written for a ragged tail compiles IDENTICALLY
# to one that assumes a clean multiple, whichever padding_mode is chosen.

@ct.kernel
def kernel_sum_undetermined(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,))  # default: UNDETERMINED
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)
n = compile_bytes(kernel_sum_undetermined, sig_2in)
print(f"kernel_sum_undetermined (default padding_mode): {n} cubin bytes")

@ct.kernel
def kernel_sum_zero_pad(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)
n = compile_bytes(kernel_sum_zero_pad, sig_2in)
print(f"kernel_sum_zero_pad: {n} cubin bytes")

@ct.kernel
def kernel_max_zero_pad(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))
n = compile_bytes(kernel_max_zero_pad, sig_2in)
print(f"kernel_max_zero_pad: {n} cubin bytes")

@ct.kernel
def kernel_max_neg_inf_pad(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))
n = compile_bytes(kernel_max_neg_inf_pad, sig_2in)
print(f"kernel_max_neg_inf_pad: {n} cubin bytes")

# All six PaddingMode values, confirmed accepted at compile time -- the
# compiler never distinguishes a "correct for this reduction" choice from
# an incorrect one; that distinction only exists at runtime, against real
# ragged data.
for mode in ct.PaddingMode:
    @ct.kernel
    def kernel_probe(a, c, block_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=mode)
        ct.store(c, index=(pid,), tile=x)
    n = compile_bytes(kernel_probe, sig_2in)
    print(f"padding_mode={mode.value}: compiles, {n} cubin bytes")

# Store's own documented asymmetry: an out-of-bounds STORE is simply
# ignored, with no padding_mode-equivalent decision to make at all.
@ct.kernel
def kernel_ragged_store(a, c, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(c, index=(pid,), tile=x)
n = compile_bytes(kernel_ragged_store, sig_2in)
print(f"kernel_ragged_store (out-of-bounds store, no padding_mode param exists for it): {n} cubin bytes")
```

### File: `02_padding_correctness_simulation.py`

```python
import torch

torch.manual_seed(0)

N = 13
BLOCK_SIZE = 8
NUM_BLOCKS = 2  # ceil(13 / 8)
PADDED_LEN = NUM_BLOCKS * BLOCK_SIZE  # 16, 3 elements past the real array

# This simulates, on the host, exactly what ct.load(..., padding_mode=X)
# documents that it does at each out-of-bounds position: fill with X's
# value. No real kernel launch is possible in this sandbox, but the
# padding semantics themselves are fully specified in cuda.tile's own
# docstrings, so a faithful host-side model of them is a genuine test of
# the CHOICE of padding_mode, independent of whether the kernel can run.

def simulate_padded_reduction(real_data, pad_value, reduce_fn):
    padded = torch.full((PADDED_LEN,), pad_value, dtype=real_data.dtype)
    padded[:N] = real_data
    return reduce_fn(padded)

# --- Sum: any real dataset, ZERO padding is always safe. ---
data_sum = torch.randn(N, dtype=torch.float64)
true_sum = data_sum.sum()
sim_sum_zero = simulate_padded_reduction(data_sum, 0.0, torch.sum)
print(f"true sum of the {N} real elements: {true_sum.item()}")
print(f"simulated ZERO-padded sum over {PADDED_LEN}: {sim_sum_zero.item()}")
print(f"ZERO padding gives the correct sum: {torch.equal(true_sum, sim_sum_zero)}")

print()

# --- Max: an all-negative dataset exposes ZERO padding as WRONG. ---
data_max = -torch.rand(N, dtype=torch.float64) - 0.1  # every real value in [-1.1, -0.1]
true_max = data_max.max()
sim_max_zero = simulate_padded_reduction(data_max, 0.0, torch.max)
sim_max_neg_inf = simulate_padded_reduction(data_max, float("-inf"), torch.max)
print(f"true max of the {N} real (all-negative) elements: {true_max.item()}")
print(f"simulated ZERO-padded max: {sim_max_zero.item()}")
print(f"ZERO padding gives the correct max: {torch.equal(true_max, sim_max_zero)}")
print(f"simulated NEG_INF-padded max: {sim_max_neg_inf.item()}")
print(f"NEG_INF padding gives the correct max: {torch.equal(true_max, sim_max_neg_inf)}")
```

### File: `03_capstone_raggedsumop_and_raggedmaxop.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

N = 13
BLOCK_SIZE = 8
NUM_BLOCKS = 2  # ceil(13 / 8) -- the last block has only 5 real elements

@ct.kernel
def kernel_ragged_sum_forward(a, acc, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.ZERO)
    partial = ct.sum(x, None)
    ct.atomic_add(acc, (0,), partial)

@ct.kernel
def kernel_ragged_sum_backward(grad_out, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    g = ct.load(grad_out, (0,), (1,))
    ct.store(grad_in, index=(pid,), tile=ct.broadcast_to(g, (block_size,)))

class RaggedSumOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n = x.shape[0]
        try:
            stream = torch.cuda.current_stream()
            acc = torch.zeros(1, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_sum_forward, (x, acc, BLOCK_SIZE))
            ctx.used_real_kernel_forward = True
            return acc
        except RuntimeError:
            ctx.used_real_kernel_forward = False
            return x.sum().reshape(1)

    @staticmethod
    def backward(ctx, grad_output):
        try:
            stream = torch.cuda.current_stream()
            # grad_in is allocated at the REAL length (13), not the padded
            # length (16) -- the tail block's store writes 8 values starting
            # at index 8, so elements [13, 16) fall outside grad_in entirely.
            # ct.store documents out-of-bounds elements as simply ignored,
            # so this needs no padding_mode-equivalent decision at all.
            grad_input = torch.empty(ctx.n, dtype=grad_output.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_sum_backward, (grad_output, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            return grad_output.expand(ctx.n).clone()

x = torch.randn(N, dtype=torch.float64, requires_grad=True)
y = RaggedSumOp.apply(x)
print(f"forward matches x.sum() over the real {N} elements: {torch.equal(y, x.detach().sum().reshape(1))}")
print(f"used real forward kernel: {y.grad_fn.used_real_kernel_forward}")

y.sum().backward()
print(f"x.grad is uniformly 1 across all {N} real elements: {torch.allclose(x.grad, torch.ones(N, dtype=torch.float64))}")
print(f"used real backward kernel: {y.grad_fn.used_real_kernel_backward}")

ok = torch.autograd.gradcheck(RaggedSumOp.apply, (x,))
print(f"RaggedSumOp gradcheck passed on a ragged ({N}-element, {NUM_BLOCKS}-block) input: {ok}")

print()

@ct.kernel
def kernel_ragged_max_partial(a, partials, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
    partial_max = ct.max(x, None)
    ct.store(partials, index=(pid,), tile=ct.broadcast_to(partial_max, (1,)))

@ct.kernel
def kernel_reduce_partials_to_global_max(partials, out, num_blocks: ct.Constant[int]):
    x = ct.load(partials, (0,), (num_blocks,))
    y = ct.max(x, None)
    ct.store(out, (0,), ct.broadcast_to(y, (1,)))

@ct.kernel
def kernel_ragged_max_count_partial(a, global_max_arr, partial_counts, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
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
def kernel_ragged_max_backward(a, global_max_arr, grad_out, count_arr, grad_in, block_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, index=(pid,), shape=(block_size,), padding_mode=ct.PaddingMode.NEG_INF)
    gmax = ct.load(global_max_arr, (0,), (1,))
    g = ct.load(grad_out, (0,), (1,))
    count = ct.load(count_arr, (0,), (1,))
    mask = (x == ct.broadcast_to(gmax, (block_size,))).astype(ct.float32)
    result = mask * ct.broadcast_to(g, (block_size,)) / ct.broadcast_to(count, (block_size,))
    ct.store(grad_in, index=(pid,), tile=result)

class RaggedMaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            partials = torch.empty(NUM_BLOCKS, dtype=x.dtype)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_max_partial, (x, partials, BLOCK_SIZE))
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
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_max_count_partial, (x, global_max, partial_counts, BLOCK_SIZE))
            global_count = torch.empty(1, dtype=x.dtype)
            ct.launch(stream, (1,), kernel_reduce_partial_counts, (partial_counts, global_count, NUM_BLOCKS))
            grad_input = torch.empty_like(x)
            ct.launch(stream, (NUM_BLOCKS,), kernel_ragged_max_backward, (x, global_max, grad_output, global_count, grad_input, BLOCK_SIZE))
            ctx.used_real_kernel_backward = True
            return grad_input
        except RuntimeError:
            ctx.used_real_kernel_backward = False
            mask = (x == global_max).to(x.dtype)
            count = mask.sum()
            return mask * grad_output.expand_as(x) / count

# All-negative real data, deliberately -- this is exactly the case Section
# 33.2 showed ZERO padding gets wrong, now wired through a complete
# multi-block Function on the real, ragged 13-element input.
x2 = -torch.rand(N, dtype=torch.float64) - 0.1
x2.requires_grad_(True)
y2 = RaggedMaxOp.apply(x2)
print(f"forward matches x.max() over the real {N} (all-negative) elements: {torch.equal(y2, x2.detach().max().reshape(1))}")
print(f"used real forward kernel: {y2.grad_fn.used_real_kernel_forward}")

y2.sum().backward()
expected = (x2.detach() == x2.detach().max()).double()
print(f"x2.grad is a one-hot at the (unique) global argmax: {torch.equal(x2.grad, expected)}")
print(f"used real backward kernel: {y2.grad_fn.used_real_kernel_backward}")

ok2 = torch.autograd.gradcheck(RaggedMaxOp.apply, (x2,))
print(f"RaggedMaxOp gradcheck passed on ragged, all-negative input: {ok2}")
```
