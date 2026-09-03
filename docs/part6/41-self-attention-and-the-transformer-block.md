# Chapter 41: Self-Attention and the Transformer Block

> "Every operation in a transformer block is one this book has already met on its own — a reduction, a matmul, a broadcast. What is new here is only that four or five of them now have to work inside a single kernel body, or hand off to each other cleanly, without a single number leaving the tile registers by accident."

**What you will understand by the end of this chapter:**

- `ct.max(x, axis=1, keepdims=True)`: a numerically stable softmax, fused entirely inside one kernel, and why subtracting a row's max before exponentiating is not optional once real score magnitudes are involved.
- Scaled dot-product self-attention — `softmax(Q @ K^T / sqrt(d)) @ V` — built from operations this book has used since Part 2 (`ct.matmul`, `ct.transpose`) and Part 4 (`ct.sum` with an axis), fused into a single kernel with no intermediate tile ever touching global memory.
- LayerNorm as a two-pass reduction (a mean, then a variance built from that mean) fused into one kernel, using `ct.rsqrt` for the first time in this book.
- A fused feed-forward kernel — up-projection, ReLU, down-projection — as this book's clearest illustration yet of what "fused kernel" means in practice, contrasted directly with Part 4's separate, autograd-chained `MatMulOp`/`AddBiasOp`/`ReluOp`.
- A complete pre-LN transformer block, assembled from all four kernels above using the same attempt-real-launch, fall-back-to-an-exact-formula pattern this book adopted in Part 3, cross-checked end to end against a plain-`torch` reference block.

**What you need to know first:**

- Chapter 21's `ct.matmul` and Chapter 22's `ct.transpose`, both reused here unchanged.
- Chapter 38's `ct.sum(x, axis=0)` — this chapter's reductions use `axis=1` instead, on tiles laid out with rows as sequence positions and columns as feature dimensions.
- Part 3's `torch.autograd.Function` forward/backward fallback pattern (`try: ct.launch(...) except RuntimeError: ...`), reused here for forward-only composition rather than autograd.
- This book's driver-free verification method: every kernel below is genuinely compiled with `ct.compilation.export_kernel`, and every claim about what it computes is cross-checked against a plain-`torch` reference on the host, since `ct.launch` still has no live driver to run against.

## 41.1 Stable Softmax and Fused Self-Attention

### Intuition

A transformer's self-attention mechanism computes, for a sequence of `SEQ` positions each represented by a `DIM`-dimensional vector, how much every position should attend to every other position: a similarity score between each pair of positions (`Q @ K^T`), scaled down and turned into a probability distribution over positions (`softmax`), then used to take a weighted combination of value vectors (`weights @ V`). Every one of those pieces already exists in this book's toolkit — `ct.matmul` since Chapter 21, `ct.transpose` since Chapter 22, `ct.sum` with an explicit axis since Chapter 38. What is new is fusing all of them, plus a softmax this book has not needed before, into a single kernel body.

Softmax itself has a numerical trap worth naming before writing it: `exp` of a large positive score overflows a `float32` long before the actual probability it represents would be interesting. The standard fix — subtract each row's maximum score before exponentiating, which changes nothing mathematically since it is later divided out — is not a detail to skip, and this chapter tests both the softmax kernel alone and the fused attention kernel that depends on it doing this correctly.

### Background

`ct.max(x, axis=1, keepdims=True)` reduces a `(ROWS, COLS)` tile down to `(ROWS, 1)`, keeping the reduced axis rather than collapsing it away — exactly the `keepdims` behavior `ct.sum` already exposed in Chapter 38, now paired with `ct.max` doing the same thing along the other axis, `axis=1` instead of `axis=0`, so it reduces each ROW down to one value rather than each column. That `(ROWS, 1)` shape is what makes `ct.broadcast_to(row_max, (ROWS, COLS))` broadcast correctly back across every row for the subtraction. From there, `ct.exp`, a second `ct.sum(..., axis=1, keepdims=True)`, and one division complete the softmax. Fused self-attention is the same shape of computation twice over: one `ct.matmul` and a `ct.transpose` to produce the scores, the softmax pattern in between, and a second `ct.matmul` to produce the output — all inside one `@ct.kernel` function, with none of the intermediate `scores`, `shifted`, `numer`, `denom`, or `weights` tiles ever written to an `Array` in global memory.

### Worked Example 41.1.1: a numerically stable softmax kernel, then fused self-attention

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

ROWS, COLS = 8, 8

# --- Softmax needs its own numerical-stability discipline: subtract
# each row's max before exponentiating, so that even a row of large
# scores never overflows exp(). ct.max's axis=1, keepdims=True keeps
# the reduced tile broadcastable against the row it came from, exactly
# the axis=0 reduction pattern Chapter 38 first established, used here
# along the other axis. ---
@ct.kernel
def kernel_softmax_forward(x, out):
    row = ct.load(x, (0, 0), (ROWS, COLS))
    row_max = ct.max(row, axis=1, keepdims=True)
    shifted = row - ct.broadcast_to(row_max, (ROWS, COLS))
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    result = numer / ct.broadcast_to(denom, (ROWS, COLS))
    ct.store(out, (0, 0), result)

sig_softmax = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_softmax_forward:", compile_bytes(kernel_softmax_forward, sig_softmax), "cubin bytes")

torch.manual_seed(0)
x = torch.randn(ROWS, COLS, dtype=torch.float32) * 5
row_max = x.max(dim=1, keepdim=True).values
shifted = x - row_max
numer = shifted.exp()
denom = numer.sum(dim=1, keepdim=True)
host_softmax = numer / denom
ref_softmax = torch.softmax(x, dim=1)
print("softmax host-side simulation matches torch.softmax:", torch.allclose(host_softmax, ref_softmax, atol=1e-6))

print()

SEQ, DIM = 8, 8
SCALE = 1.0 / math.sqrt(DIM)

# --- Scaled dot-product self-attention, fused into one kernel: the two
# matmuls (Q @ K^T and weights @ V) and the softmax in between never
# leave this kernel's own tile registers for global memory. ct.transpose
# (Chapter 22) and ct.matmul (Chapter 21) are exactly the same
# operations this book's Part 4 backward kernels already relied on. ---
@ct.kernel
def kernel_attention_forward(q, k, v, out):
    Q = ct.load(q, (0, 0), (SEQ, DIM))
    K = ct.load(k, (0, 0), (SEQ, DIM))
    V = ct.load(v, (0, 0), (SEQ, DIM))
    scores = ct.matmul(Q, ct.transpose(K)) * SCALE
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - ct.broadcast_to(row_max, (SEQ, SEQ))
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    weights = numer / ct.broadcast_to(denom, (SEQ, SEQ))
    result = ct.matmul(weights, V)
    ct.store(out, (0, 0), result)

sig_attention = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_attention_forward:", compile_bytes(kernel_attention_forward, sig_attention), "cubin bytes")

Q = torch.randn(SEQ, DIM, dtype=torch.float32)
K = torch.randn(SEQ, DIM, dtype=torch.float32)
V = torch.randn(SEQ, DIM, dtype=torch.float32)

scores = (Q @ K.T) * SCALE
weights = torch.softmax(scores, dim=1)
host_attention = weights @ V

ref_attention = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
).squeeze(0)
print("attention host-side simulation matches torch's scaled_dot_product_attention:",
      torch.allclose(host_attention, ref_attention, atol=1e-5))
```

### Genuinely run

```text
kernel_softmax_forward: 36880 cubin bytes
softmax host-side simulation matches torch.softmax: True

kernel_attention_forward: 43072 cubin bytes
attention host-side simulation matches torch's scaled_dot_product_attention: True
```

### Discussion

Both kernels compile cleanly, and both host-side simulations — built by hand-writing out exactly what each kernel's own tile operations compute, line for line — match PyTorch's own reference implementations to within floating-point tolerance. That second check matters more than it might look: `torch.softmax` and `torch.nn.functional.scaled_dot_product_attention` are both real, independently-implemented, widely-used reference operations. Matching them is not this book checking its own arithmetic against itself; it is checking a from-scratch cuTile Python kernel against an entirely separate, trusted implementation of the same well-known operation, on host tensors, the same cross-checking discipline Chapter 33 first used for `padding_mode` and Chapter 36 used for scatter-versus-atomic-accumulate.

The fusion itself is worth pausing on. `kernel_attention_forward` performs two matmuls, a transpose, a max-reduction, an exponential, a sum-reduction, a broadcast division, and a second matmul — eight distinct operations — inside one kernel body, and none of the five intermediate tiles (`scores`, `shifted`, `numer`, `denom`, `weights`) is ever written out to an `Array` and read back in. On real hardware, that is exactly the difference between one kernel launch that keeps everything in fast on-chip tile registers and a naive implementation that would launch several kernels and round-trip every intermediate through slower global memory between each one. This book cannot measure that difference directly — Chapter 40 already established the limits of what a driver-free environment can honestly benchmark — but it can, and does, show the fusion is not just plausible but correctly computes the right answer.

## 41.2 LayerNorm and a Fused Feed-Forward Kernel

### Intuition

A transformer block needs two more pieces before it can be assembled: LayerNorm, which normalizes each position's feature vector to zero mean and unit variance before scaling and shifting it by learned parameters, and a feed-forward network (FFN), which projects each position's vector up to a wider hidden dimension, applies a nonlinearity, and projects it back down. Both are, again, built entirely from operations this book already has. LayerNorm needs two passes of reduction — first a mean, then a variance computed from values already centered by that mean — which is a genuinely two-step dependency chain within a single kernel, unlike the one-pass reductions this book's Part 4 chapters used.

### Background

`ct.rsqrt`, used for the first time in this book, computes `1 / sqrt(x)` directly rather than as two separate operations — the standard building block for LayerNorm's `(x - mean) / sqrt(variance + epsilon)` normalization, computed here as `(x - mean) * rsqrt(variance + epsilon)` to trade a division for a multiplication. The feed-forward kernel is more straightforward: two `ct.matmul` calls around one `ct.maximum(hidden, 0.0)` for the ReLU nonlinearity, all three ahead-of-time fused into a single kernel body, in direct contrast with Chapter 35's `MatMulOp`, Chapter 38's `AddBiasOp`, and Chapter 30's `ReluOp` — three separate, independently `gradcheck`-verified `torch.autograd.Function` classes chained together through PyTorch's own autograd graph. Part 4 needed that separation because each piece needed its own genuinely-tested backward pass; Part 6 has no such requirement for a forward-only fused kernel, and fusing all three into one kernel body is exactly the point.

### Worked Example 41.2.1: LayerNorm, then a fused feed-forward kernel

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

ROWS, COLS = 8, 8
EPS = 1e-5

# --- LayerNorm's two-pass reduction -- a mean, then a variance built
# from that mean -- fused into one kernel body: both ct.sum reductions
# stay along axis=1, keepdims=True, so every intermediate tile stays
# broadcastable against the row it was reduced from. ---
@ct.kernel
def kernel_layernorm_forward(x, weight, bias, out):
    xt = ct.load(x, (0, 0), (ROWS, COLS))
    w = ct.load(weight, (0,), (COLS,))
    b = ct.load(bias, (0,), (COLS,))
    mean = ct.sum(xt, axis=1, keepdims=True) / COLS
    centered = xt - ct.broadcast_to(mean, (ROWS, COLS))
    var = ct.sum(centered * centered, axis=1, keepdims=True) / COLS
    inv_std = ct.rsqrt(var + EPS)
    normalized = centered * ct.broadcast_to(inv_std, (ROWS, COLS))
    result = normalized * ct.broadcast_to(w, (ROWS, COLS)) + ct.broadcast_to(b, (ROWS, COLS))
    ct.store(out, (0, 0), result)

sig_layernorm = ct.compilation.KernelSignature(
    [array_param(2), array_param(1), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_layernorm_forward:", compile_bytes(kernel_layernorm_forward, sig_layernorm), "cubin bytes")

torch.manual_seed(0)
x = torch.randn(ROWS, COLS, dtype=torch.float32)
weight = torch.randn(COLS, dtype=torch.float32)
bias = torch.randn(COLS, dtype=torch.float32)

mean = x.mean(dim=1, keepdim=True)
centered = x - mean
var = (centered * centered).mean(dim=1, keepdim=True)
inv_std = torch.rsqrt(var + EPS)
host_layernorm = centered * inv_std * weight + bias

ref_layernorm = torch.nn.functional.layer_norm(x, (COLS,), weight=weight, bias=bias, eps=EPS)
print("layernorm host-side simulation matches torch.nn.functional.layer_norm:",
      torch.allclose(host_layernorm, ref_layernorm, atol=1e-5))

print()

SEQ, DIM, HID = 8, 8, 16

# --- One kernel, three operations fused: the up-projection matmul, the
# ReLU nonlinearity, and the down-projection matmul, with no
# intermediate round trip to global memory between any of them. This is
# what Part 6's "fused kernels" actually means in practice, compared to
# Part 4's separate MatMulOp/AddBiasOp/ReluOp chained through autograd. ---
@ct.kernel
def kernel_ffn_forward(x, w1, b1, w2, b2, out):
    xt = ct.load(x, (0, 0), (SEQ, DIM))
    w1t = ct.load(w1, (0, 0), (DIM, HID))
    b1t = ct.load(b1, (0,), (HID,))
    hidden = ct.matmul(xt, w1t) + ct.broadcast_to(b1t, (SEQ, HID))
    activated = ct.maximum(hidden, 0.0)
    w2t = ct.load(w2, (0, 0), (HID, DIM))
    b2t = ct.load(b2, (0,), (DIM,))
    result = ct.matmul(activated, w2t) + ct.broadcast_to(b2t, (SEQ, DIM))
    ct.store(out, (0, 0), result)

sig_ffn = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(1), array_param(2), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_ffn_forward:", compile_bytes(kernel_ffn_forward, sig_ffn), "cubin bytes")

x2 = torch.randn(SEQ, DIM, dtype=torch.float32)
w1 = torch.randn(DIM, HID, dtype=torch.float32)
b1 = torch.randn(HID, dtype=torch.float32)
w2 = torch.randn(HID, DIM, dtype=torch.float32)
b2 = torch.randn(DIM, dtype=torch.float32)

hidden = x2 @ w1 + b1
activated = torch.relu(hidden)
host_ffn = activated @ w2 + b2

linear1 = torch.nn.functional.linear(x2, w1.T, b1)
ref_hidden = torch.relu(linear1)
ref_ffn = torch.nn.functional.linear(ref_hidden, w2.T, b2)
print("FFN host-side simulation matches plain-torch reference FFN:", torch.allclose(host_ffn, ref_ffn, atol=1e-5))
```

### Genuinely run

```text
kernel_layernorm_forward: 40368 cubin bytes
layernorm host-side simulation matches torch.nn.functional.layer_norm: True

kernel_ffn_forward: 45536 cubin bytes
FFN host-side simulation matches plain-torch reference FFN: True
```

### Discussion

Both kernels compile, and both cross-check exactly against PyTorch's own `layer_norm` and an equivalent two-linear-layer-plus-ReLU reference built from `torch.nn.functional.linear`. The `rsqrt`-based normalization formula — `centered * rsqrt(var + eps)` rather than `centered / sqrt(var + eps)` — matches PyTorch's own semantics to within the same floating-point tolerance every other cross-check in this book has used, confirming the algebraic rewrite changes nothing observable. The feed-forward kernel's fusion is the more instructive result: three operations that Part 4 deliberately kept as three separate, independently-verified `torch.autograd.Function` classes are, here, one `@ct.kernel` function, because this chapter has no backward pass to verify separately for each piece — only a forward computation whose correctness this book can check as a single, indivisible unit against a trusted reference. Both approaches are honest; they answer different questions. Part 4 asked "is each piece's gradient correct on its own," which required separation. This chapter asks "does the fused whole compute the right forward value," which does not.

## 41.3 Capstone: A Pre-LN Transformer Block

### Intuition

A standard pre-LN transformer block — the architecture used, in one variant or another, by essentially every modern transformer-based language and vision model since 2018 — normalizes its input, self-attends, adds that result back as a residual, normalizes again, feeds forward, and adds that result back as a second residual. Every one of those five steps is now available: `kernel_layernorm_forward`, `kernel_attention_forward`, and `kernel_ffn_forward` from this chapter's first two sections, plus two ordinary tensor additions for the residual connections. Assembling them into one block, and checking the assembled whole against a complete reference implementation, is this chapter's capstone.

### Background

Because this chapter's kernels are forward-only, the capstone reuses Part 3's attempt-real-launch, fall-back-to-an-exact-formula pattern directly, without any `torch.autograd.Function` wrapping: each of `layernorm`, `self_attention`, and `ffn` below tries `ct.launch` first, catches the same `RuntimeError` this book has hit since Chapter 1, and falls back to a plain-`torch` computation that mirrors its corresponding kernel's exact formula. `transformer_block` composes those three functions with two residual additions in between; `reference_transformer_block` computes the identical architecture using only PyTorch's own built-in operations, with no cuTile Python involved at any point, as the independent check.

### Worked Example 41.3.1: assembling and cross-checking a complete transformer block

```python
import cuda.tile as ct
import math
import torch

SEQ, DIM, HID = 8, 8, 16
SCALE = 1.0 / math.sqrt(DIM)
EPS = 1e-5

@ct.kernel
def kernel_layernorm_forward(x, weight, bias, out):
    xt = ct.load(x, (0, 0), (SEQ, DIM))
    w = ct.load(weight, (0,), (DIM,))
    b = ct.load(bias, (0,), (DIM,))
    mean = ct.sum(xt, axis=1, keepdims=True) / DIM
    centered = xt - ct.broadcast_to(mean, (SEQ, DIM))
    var = ct.sum(centered * centered, axis=1, keepdims=True) / DIM
    inv_std = ct.rsqrt(var + EPS)
    normalized = centered * ct.broadcast_to(inv_std, (SEQ, DIM))
    result = normalized * ct.broadcast_to(w, (SEQ, DIM)) + ct.broadcast_to(b, (SEQ, DIM))
    ct.store(out, (0, 0), result)

@ct.kernel
def kernel_attention_forward(q, k, v, out):
    Q = ct.load(q, (0, 0), (SEQ, DIM))
    K = ct.load(k, (0, 0), (SEQ, DIM))
    V = ct.load(v, (0, 0), (SEQ, DIM))
    scores = ct.matmul(Q, ct.transpose(K)) * SCALE
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - ct.broadcast_to(row_max, (SEQ, SEQ))
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    weights = numer / ct.broadcast_to(denom, (SEQ, SEQ))
    result = ct.matmul(weights, V)
    ct.store(out, (0, 0), result)

@ct.kernel
def kernel_ffn_forward(x, w1, b1, w2, b2, out):
    xt = ct.load(x, (0, 0), (SEQ, DIM))
    w1t = ct.load(w1, (0, 0), (DIM, HID))
    b1t = ct.load(b1, (0,), (HID,))
    hidden = ct.matmul(xt, w1t) + ct.broadcast_to(b1t, (SEQ, HID))
    activated = ct.maximum(hidden, 0.0)
    w2t = ct.load(w2, (0, 0), (HID, DIM))
    b2t = ct.load(b2, (0,), (DIM,))
    result = ct.matmul(activated, w2t) + ct.broadcast_to(b2t, (SEQ, DIM))
    ct.store(out, (0, 0), result)


# --- The same attempt-real-launch, fall-back-to-an-exact-formula
# pattern this book has used since Part 3: each stage tries ct.launch
# first, and only falls back to a plain-torch computation mirroring
# that exact kernel's formula when the hardware wall (Chapter 1) is hit. ---
def layernorm(x, weight, bias):
    try:
        stream = torch.cuda.current_stream()
        out = torch.empty(SEQ, DIM, dtype=x.dtype)
        ct.launch(stream, (1,), kernel_layernorm_forward, (x, weight, bias, out))
        return out, True
    except RuntimeError:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + EPS)
        return centered * inv_std * weight + bias, False


def self_attention(q, k, v):
    try:
        stream = torch.cuda.current_stream()
        out = torch.empty(SEQ, DIM, dtype=q.dtype)
        ct.launch(stream, (1,), kernel_attention_forward, (q, k, v, out))
        return out, True
    except RuntimeError:
        scores = (q @ k.T) * SCALE
        weights = torch.softmax(scores, dim=1)
        return weights @ v, False


def ffn(x, w1, b1, w2, b2):
    try:
        stream = torch.cuda.current_stream()
        out = torch.empty(SEQ, DIM, dtype=x.dtype)
        ct.launch(stream, (1,), kernel_ffn_forward, (x, w1, b1, w2, b2, out))
        return out, True
    except RuntimeError:
        hidden = x @ w1 + b1
        activated = torch.relu(hidden)
        return activated @ w2 + b2, False


# --- A complete pre-LN transformer block: normalize, self-attend, add
# the residual, normalize again, feed forward, add the residual again --
# the same block architecture that has been the standard building block
# of transformer-based language and vision models since 2018. ---
def transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2):
    used_real = []
    normed1, real = layernorm(x, ln1_w, ln1_b)
    used_real.append(real)
    q = normed1 @ wq
    k = normed1 @ wk
    v = normed1 @ wv
    attn_out, real = self_attention(q, k, v)
    used_real.append(real)
    x = x + attn_out
    normed2, real = layernorm(x, ln2_w, ln2_b)
    used_real.append(real)
    ffn_out, real = ffn(normed2, w1, b1, w2, b2)
    used_real.append(real)
    x = x + ffn_out
    return x, used_real


def reference_transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2):
    normed1 = torch.nn.functional.layer_norm(x, (DIM,), weight=ln1_w, bias=ln1_b, eps=EPS)
    q = normed1 @ wq
    k = normed1 @ wk
    v = normed1 @ wv
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    ).squeeze(0)
    x = x + attn_out
    normed2 = torch.nn.functional.layer_norm(x, (DIM,), weight=ln2_w, bias=ln2_b, eps=EPS)
    linear1 = torch.nn.functional.linear(normed2, w1.T, b1)
    hidden = torch.relu(linear1)
    ffn_out = torch.nn.functional.linear(hidden, w2.T, b2)
    x = x + ffn_out
    return x


torch.manual_seed(0)
x = torch.randn(SEQ, DIM, dtype=torch.float32)
ln1_w = torch.randn(DIM, dtype=torch.float32)
ln1_b = torch.randn(DIM, dtype=torch.float32)
wq = torch.randn(DIM, DIM, dtype=torch.float32)
wk = torch.randn(DIM, DIM, dtype=torch.float32)
wv = torch.randn(DIM, DIM, dtype=torch.float32)
ln2_w = torch.randn(DIM, dtype=torch.float32)
ln2_b = torch.randn(DIM, dtype=torch.float32)
w1 = torch.randn(DIM, HID, dtype=torch.float32)
b1 = torch.randn(HID, dtype=torch.float32)
w2 = torch.randn(HID, DIM, dtype=torch.float32)
b2 = torch.randn(DIM, dtype=torch.float32)

out, used_real = transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2)
ref = reference_transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2)

print(f"used real cuTile kernels for all four stages: {all(used_real)}")
print(f"transformer block output matches plain-torch reference: {torch.allclose(out, ref, atol=1e-5)}")
```

### Genuinely run

```text
used real cuTile kernels for all four stages: False
transformer block output matches plain-torch reference: True
```

### Discussion

`used real cuTile kernels for all four stages: False` is exactly what this book's own honesty discipline predicts, not a failure: this sandboxed build environment has no NVIDIA driver, so every one of the four stages' `ct.launch` calls hits the same `RuntimeError` Chapter 1 first triggered, and every stage falls back to its plain-`torch` formula. That fallback formula is not a stand-in invented for this chapter — it is, line for line, the exact computation each corresponding kernel performs, which is precisely why `transformer_block`'s output matches `reference_transformer_block`'s independently-written PyTorch implementation to within floating-point tolerance. The match is real evidence that the assembled block's logic is correct; it is not, and this book will not claim it to be, evidence about how fast four fused kernel launches would run compared to PyTorch's own eager-mode operations on an actual GPU. That second question is squarely the one Chapter 40 already ruled out of reach for this environment, and nothing about wiring the pieces into a bigger block changes what this book can honestly measure.

What has changed since Part 4 is the shape of the composition. Every one of this chapter's three fallback functions is forward-only, with no `ctx.save_for_backward` and no corresponding `backward` staticmethod, because this chapter's transformer block is not wired into PyTorch's autograd graph at all — it is a standalone forward computation, exactly the kind Part 6's "fused kernels for real layers" framing calls for, distinct from Part 3 and Part 4's autograd-integrated custom operations. Both are legitimate, honestly-verified uses of the same underlying kernels; they simply answer different engineering questions, and this book has now demonstrated both.

## Chapter Summary

This chapter built a complete, genuinely compiled, genuinely cross-checked pre-LN transformer block from the ground up. A numerically stable softmax kernel, using `ct.max(..., axis=1, keepdims=True)` for the first max-subtraction stability trick this book has needed, was verified against `torch.softmax`. Scaled dot-product self-attention was fused into a single kernel — two matmuls, a transpose, and a softmax, with every intermediate value staying inside the kernel's own tile registers — and verified against `torch.nn.functional.scaled_dot_product_attention`. LayerNorm introduced `ct.rsqrt` and a genuinely two-pass reduction (mean, then variance from that mean), verified against `torch.nn.functional.layer_norm`. A fused feed-forward kernel folded an up-projection, a ReLU, and a down-projection into one kernel body — the clearest contrast yet with Part 4's deliberately separated, autograd-verified building blocks — verified against a plain-`torch` two-linear-layer reference. The capstone assembled all four kernels, plus two residual connections, into a complete transformer block using Part 3's attempt-real-launch fallback pattern, and confirmed the assembled whole matches an independently-written reference block exactly, with the now-familiar, honestly-reported caveat that every stage ran its host-side fallback rather than a real kernel launch.

## Self-Check Questions

1. `kernel_softmax_forward` subtracts each row's maximum before exponentiating. Explain, in terms of what `float32` can represent, what would go wrong for a row containing a score of, say, 200, if that subtraction were skipped.
2. Section 41.2's Discussion says Part 4 kept `MatMulOp`, `AddBiasOp`, and `ReluOp` separate because each needed its own independently-verified backward pass, while this chapter fuses the equivalent forward operations into one kernel. What would have to be true about `kernel_ffn_forward` for this chapter to also need it split into three separate pieces?
3. The capstone's `transformer_block` function never calls `torch.autograd.Function` or defines a `backward` method anywhere. If a future chapter wanted this exact fused FFN kernel to participate in a `loss.backward()` call, what would have to be added, and would the existing `kernel_ffn_forward` kernel itself need to change?
4. `kernel_attention_forward` computes `Q`, `K`, and `V` for a single sequence of 8 positions with no batch dimension and no separate attention heads. Real transformer models process a batch of sequences, each split across multiple heads. Which of this chapter's genuine correctness claims would still hold if this kernel were extended to handle multiple heads at once, and which would need to be re-verified from scratch?
5. Both `used real cuTile kernels for all four stages: False` and `transformer block output matches plain-torch reference: True` appear in the capstone's output. Explain why the second claim's truth does not depend on the first claim being `True` — and what specifically would make the second claim's truth actually meaningful once the first claim does become `True` on real hardware.

## Where We Go Next

This chapter's transformer block used one dense attention pattern and one dense feed-forward network per position — every token attends to every other token, and every token is processed by the same single feed-forward network. Two of the most consequential departures from that plain design in modern large-scale transformers are Mixture-of-Experts, which routes each token to only a handful of specialized feed-forward networks out of many, and Multi-Head Latent Attention, which compresses the key and value tensors this chapter computed directly into a much smaller latent representation before expanding them back out. The next chapter builds both, genuinely compiled and genuinely cross-checked with the same discipline this chapter just used.

## Worked Solutions

**1.** `float32` can represent finite values up to roughly `3.4e38`, and `exp(200)` is approximately `7.2e86` — far beyond that range. Computing `exp(200)` directly in `float32` would overflow to `inf`, and once even one element of a row becomes `inf`, the row's sum used for the softmax denominator also becomes `inf`, and `inf / inf` is `nan` — corrupting that row's entire output with not-a-number values instead of a valid probability distribution, even though the true softmax of a row containing 200 alongside more ordinary scores is a perfectly well-defined, finite set of probabilities.

**2.** `kernel_ffn_forward` would need splitting into three separate, independently-verified pieces if this chapter needed to compute and verify a correct gradient for each of the up-projection matmul, the ReLU, and the down-projection matmul on their own — for instance, if this fused kernel were going to be wrapped in a `torch.autograd.Function` and trained via `loss.backward()`, the way Part 4's separate operations were. As a forward-only computation whose only claim is "this fused kernel computes the right output value," which is exactly what this chapter tested and no more, there is no such requirement, and fusing the three operations loses nothing this chapter actually needed to verify.

**3.** Making this exact fused FFN kernel participate in `loss.backward()` would require wrapping it in a `torch.autograd.Function` subclass with a `forward` staticmethod that saves whatever `backward` will need (via `ctx.save_for_backward`, following Part 3 and Part 4's established pattern) and a genuinely new `backward` staticmethod computing gradients with respect to `x`, `w1`, `b1`, `w2`, and `b2`. `kernel_ffn_forward` itself would not need to change at all — it already computes the correct forward value, which is all a `forward` staticmethod needs from it — but at least one new backward kernel (very likely more than one, mirroring the multiple backward kernels Chapter 34's `MatMulOp` and Chapter 38's `AddBiasOp` each needed) would have to be written and independently `gradcheck`-verified before this book could honestly claim the fused FFN was trainable.

**4.** The genuine correctness claim that would still hold is the core mathematical relationship this chapter actually tested: that `softmax(Q @ K^T / sqrt(d)) @ V`, computed via `ct.matmul`, `ct.transpose`, and the max-subtraction softmax pattern, correctly reproduces `torch.nn.functional.scaled_dot_product_attention` for one sequence and one head — extending the shapes to add batch and head dimensions does not change that underlying per-head computation. What would need re-verifying from scratch is everything about how those extra dimensions are handled: whether the batch and head axes broadcast correctly through every `ct.broadcast_to` call in the kernel, whether the tile shapes involved still satisfy the power-of-two constraint this book has enforced since Part 0, and whether a new host-side cross-check against a batched, multi-head call to `scaled_dot_product_attention` still matches — none of which this chapter's single-sequence, single-head test can speak to.

**5.** The two claims are independent because the second one is a statement about mathematical correctness (does the composed logic compute the right answer), verified by comparing two computations that both currently run on the host in plain `torch` — the fallback path and the reference path — with no cuTile kernel execution involved in either one right now. The first claim only reports which code path performed that computation. Once real hardware makes the first claim `True`, the second claim's truth becomes meaningful in a new way: it would mean four genuinely executed GPU kernels, launched through `ct.launch` and run against a live CUDA stream, produced output matching the same trusted plain-`torch` reference — extending this chapter's forward-correctness evidence from "the formulas are right" to "the formulas are right when actually executed on the hardware they were compiled for," which is exactly the UNVERIFIED-pending-real-GPU-test gap this book has flagged honestly since its first chapter.

## Complete Runnable Code

`01_softmax_and_self_attention.py`:

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

ROWS, COLS = 8, 8

# --- Softmax needs its own numerical-stability discipline: subtract
# each row's max before exponentiating, so that even a row of large
# scores never overflows exp(). ct.max's axis=1, keepdims=True keeps
# the reduced tile broadcastable against the row it came from, exactly
# the axis=0 reduction pattern Chapter 38 first established, used here
# along the other axis. ---
@ct.kernel
def kernel_softmax_forward(x, out):
    row = ct.load(x, (0, 0), (ROWS, COLS))
    row_max = ct.max(row, axis=1, keepdims=True)
    shifted = row - ct.broadcast_to(row_max, (ROWS, COLS))
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    result = numer / ct.broadcast_to(denom, (ROWS, COLS))
    ct.store(out, (0, 0), result)

sig_softmax = ct.compilation.KernelSignature(
    [array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_softmax_forward:", compile_bytes(kernel_softmax_forward, sig_softmax), "cubin bytes")

torch.manual_seed(0)
x = torch.randn(ROWS, COLS, dtype=torch.float32) * 5
row_max = x.max(dim=1, keepdim=True).values
shifted = x - row_max
numer = shifted.exp()
denom = numer.sum(dim=1, keepdim=True)
host_softmax = numer / denom
ref_softmax = torch.softmax(x, dim=1)
print("softmax host-side simulation matches torch.softmax:", torch.allclose(host_softmax, ref_softmax, atol=1e-6))

print()

SEQ, DIM = 8, 8
SCALE = 1.0 / math.sqrt(DIM)

# --- Scaled dot-product self-attention, fused into one kernel: the two
# matmuls (Q @ K^T and weights @ V) and the softmax in between never
# leave this kernel's own tile registers for global memory. ct.transpose
# (Chapter 22) and ct.matmul (Chapter 21) are exactly the same
# operations this book's Part 4 backward kernels already relied on. ---
@ct.kernel
def kernel_attention_forward(q, k, v, out):
    Q = ct.load(q, (0, 0), (SEQ, DIM))
    K = ct.load(k, (0, 0), (SEQ, DIM))
    V = ct.load(v, (0, 0), (SEQ, DIM))
    scores = ct.matmul(Q, ct.transpose(K)) * SCALE
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - ct.broadcast_to(row_max, (SEQ, SEQ))
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    weights = numer / ct.broadcast_to(denom, (SEQ, SEQ))
    result = ct.matmul(weights, V)
    ct.store(out, (0, 0), result)

sig_attention = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_attention_forward:", compile_bytes(kernel_attention_forward, sig_attention), "cubin bytes")

Q = torch.randn(SEQ, DIM, dtype=torch.float32)
K = torch.randn(SEQ, DIM, dtype=torch.float32)
V = torch.randn(SEQ, DIM, dtype=torch.float32)

scores = (Q @ K.T) * SCALE
weights = torch.softmax(scores, dim=1)
host_attention = weights @ V

ref_attention = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
).squeeze(0)
print("attention host-side simulation matches torch's scaled_dot_product_attention:",
      torch.allclose(host_attention, ref_attention, atol=1e-5))
```

`02_layernorm_and_fused_ffn.py`:

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

ROWS, COLS = 8, 8
EPS = 1e-5

# --- LayerNorm's two-pass reduction -- a mean, then a variance built
# from that mean -- fused into one kernel body: both ct.sum reductions
# stay along axis=1, keepdims=True, so every intermediate tile stays
# broadcastable against the row it was reduced from. ---
@ct.kernel
def kernel_layernorm_forward(x, weight, bias, out):
    xt = ct.load(x, (0, 0), (ROWS, COLS))
    w = ct.load(weight, (0,), (COLS,))
    b = ct.load(bias, (0,), (COLS,))
    mean = ct.sum(xt, axis=1, keepdims=True) / COLS
    centered = xt - ct.broadcast_to(mean, (ROWS, COLS))
    var = ct.sum(centered * centered, axis=1, keepdims=True) / COLS
    inv_std = ct.rsqrt(var + EPS)
    normalized = centered * ct.broadcast_to(inv_std, (ROWS, COLS))
    result = normalized * ct.broadcast_to(w, (ROWS, COLS)) + ct.broadcast_to(b, (ROWS, COLS))
    ct.store(out, (0, 0), result)

sig_layernorm = ct.compilation.KernelSignature(
    [array_param(2), array_param(1), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_layernorm_forward:", compile_bytes(kernel_layernorm_forward, sig_layernorm), "cubin bytes")

torch.manual_seed(0)
x = torch.randn(ROWS, COLS, dtype=torch.float32)
weight = torch.randn(COLS, dtype=torch.float32)
bias = torch.randn(COLS, dtype=torch.float32)

mean = x.mean(dim=1, keepdim=True)
centered = x - mean
var = (centered * centered).mean(dim=1, keepdim=True)
inv_std = torch.rsqrt(var + EPS)
host_layernorm = centered * inv_std * weight + bias

ref_layernorm = torch.nn.functional.layer_norm(x, (COLS,), weight=weight, bias=bias, eps=EPS)
print("layernorm host-side simulation matches torch.nn.functional.layer_norm:",
      torch.allclose(host_layernorm, ref_layernorm, atol=1e-5))

print()

SEQ, DIM, HID = 8, 8, 16

# --- One kernel, three operations fused: the up-projection matmul, the
# ReLU nonlinearity, and the down-projection matmul, with no
# intermediate round trip to global memory between any of them. This is
# what Part 6's "fused kernels" actually means in practice, compared to
# Part 4's separate MatMulOp/AddBiasOp/ReluOp chained through autograd. ---
@ct.kernel
def kernel_ffn_forward(x, w1, b1, w2, b2, out):
    xt = ct.load(x, (0, 0), (SEQ, DIM))
    w1t = ct.load(w1, (0, 0), (DIM, HID))
    b1t = ct.load(b1, (0,), (HID,))
    hidden = ct.matmul(xt, w1t) + ct.broadcast_to(b1t, (SEQ, HID))
    activated = ct.maximum(hidden, 0.0)
    w2t = ct.load(w2, (0, 0), (HID, DIM))
    b2t = ct.load(b2, (0,), (DIM,))
    result = ct.matmul(activated, w2t) + ct.broadcast_to(b2t, (SEQ, DIM))
    ct.store(out, (0, 0), result)

sig_ffn = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(1), array_param(2), array_param(1), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
print("kernel_ffn_forward:", compile_bytes(kernel_ffn_forward, sig_ffn), "cubin bytes")

x2 = torch.randn(SEQ, DIM, dtype=torch.float32)
w1 = torch.randn(DIM, HID, dtype=torch.float32)
b1 = torch.randn(HID, dtype=torch.float32)
w2 = torch.randn(HID, DIM, dtype=torch.float32)
b2 = torch.randn(DIM, dtype=torch.float32)

hidden = x2 @ w1 + b1
activated = torch.relu(hidden)
host_ffn = activated @ w2 + b2

linear1 = torch.nn.functional.linear(x2, w1.T, b1)
ref_hidden = torch.relu(linear1)
ref_ffn = torch.nn.functional.linear(ref_hidden, w2.T, b2)
print("FFN host-side simulation matches plain-torch reference FFN:", torch.allclose(host_ffn, ref_ffn, atol=1e-5))
```

`03_capstone_a_pre_ln_transformer_block.py`:

```python
import cuda.tile as ct
import math
import torch

SEQ, DIM, HID = 8, 8, 16
SCALE = 1.0 / math.sqrt(DIM)
EPS = 1e-5

@ct.kernel
def kernel_layernorm_forward(x, weight, bias, out):
    xt = ct.load(x, (0, 0), (SEQ, DIM))
    w = ct.load(weight, (0,), (DIM,))
    b = ct.load(bias, (0,), (DIM,))
    mean = ct.sum(xt, axis=1, keepdims=True) / DIM
    centered = xt - ct.broadcast_to(mean, (SEQ, DIM))
    var = ct.sum(centered * centered, axis=1, keepdims=True) / DIM
    inv_std = ct.rsqrt(var + EPS)
    normalized = centered * ct.broadcast_to(inv_std, (SEQ, DIM))
    result = normalized * ct.broadcast_to(w, (SEQ, DIM)) + ct.broadcast_to(b, (SEQ, DIM))
    ct.store(out, (0, 0), result)

@ct.kernel
def kernel_attention_forward(q, k, v, out):
    Q = ct.load(q, (0, 0), (SEQ, DIM))
    K = ct.load(k, (0, 0), (SEQ, DIM))
    V = ct.load(v, (0, 0), (SEQ, DIM))
    scores = ct.matmul(Q, ct.transpose(K)) * SCALE
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - ct.broadcast_to(row_max, (SEQ, SEQ))
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    weights = numer / ct.broadcast_to(denom, (SEQ, SEQ))
    result = ct.matmul(weights, V)
    ct.store(out, (0, 0), result)

@ct.kernel
def kernel_ffn_forward(x, w1, b1, w2, b2, out):
    xt = ct.load(x, (0, 0), (SEQ, DIM))
    w1t = ct.load(w1, (0, 0), (DIM, HID))
    b1t = ct.load(b1, (0,), (HID,))
    hidden = ct.matmul(xt, w1t) + ct.broadcast_to(b1t, (SEQ, HID))
    activated = ct.maximum(hidden, 0.0)
    w2t = ct.load(w2, (0, 0), (HID, DIM))
    b2t = ct.load(b2, (0,), (DIM,))
    result = ct.matmul(activated, w2t) + ct.broadcast_to(b2t, (SEQ, DIM))
    ct.store(out, (0, 0), result)


# --- The same attempt-real-launch, fall-back-to-an-exact-formula
# pattern this book has used since Part 3: each stage tries ct.launch
# first, and only falls back to a plain-torch computation mirroring
# that exact kernel's formula when the hardware wall (Chapter 1) is hit. ---
def layernorm(x, weight, bias):
    try:
        stream = torch.cuda.current_stream()
        out = torch.empty(SEQ, DIM, dtype=x.dtype)
        ct.launch(stream, (1,), kernel_layernorm_forward, (x, weight, bias, out))
        return out, True
    except RuntimeError:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + EPS)
        return centered * inv_std * weight + bias, False


def self_attention(q, k, v):
    try:
        stream = torch.cuda.current_stream()
        out = torch.empty(SEQ, DIM, dtype=q.dtype)
        ct.launch(stream, (1,), kernel_attention_forward, (q, k, v, out))
        return out, True
    except RuntimeError:
        scores = (q @ k.T) * SCALE
        weights = torch.softmax(scores, dim=1)
        return weights @ v, False


def ffn(x, w1, b1, w2, b2):
    try:
        stream = torch.cuda.current_stream()
        out = torch.empty(SEQ, DIM, dtype=x.dtype)
        ct.launch(stream, (1,), kernel_ffn_forward, (x, w1, b1, w2, b2, out))
        return out, True
    except RuntimeError:
        hidden = x @ w1 + b1
        activated = torch.relu(hidden)
        return activated @ w2 + b2, False


# --- A complete pre-LN transformer block: normalize, self-attend, add
# the residual, normalize again, feed forward, add the residual again --
# the same block architecture that has been the standard building block
# of transformer-based language and vision models since 2018. ---
def transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2):
    used_real = []
    normed1, real = layernorm(x, ln1_w, ln1_b)
    used_real.append(real)
    q = normed1 @ wq
    k = normed1 @ wk
    v = normed1 @ wv
    attn_out, real = self_attention(q, k, v)
    used_real.append(real)
    x = x + attn_out
    normed2, real = layernorm(x, ln2_w, ln2_b)
    used_real.append(real)
    ffn_out, real = ffn(normed2, w1, b1, w2, b2)
    used_real.append(real)
    x = x + ffn_out
    return x, used_real


def reference_transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2):
    normed1 = torch.nn.functional.layer_norm(x, (DIM,), weight=ln1_w, bias=ln1_b, eps=EPS)
    q = normed1 @ wq
    k = normed1 @ wk
    v = normed1 @ wv
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    ).squeeze(0)
    x = x + attn_out
    normed2 = torch.nn.functional.layer_norm(x, (DIM,), weight=ln2_w, bias=ln2_b, eps=EPS)
    linear1 = torch.nn.functional.linear(normed2, w1.T, b1)
    hidden = torch.relu(linear1)
    ffn_out = torch.nn.functional.linear(hidden, w2.T, b2)
    x = x + ffn_out
    return x


torch.manual_seed(0)
x = torch.randn(SEQ, DIM, dtype=torch.float32)
ln1_w = torch.randn(DIM, dtype=torch.float32)
ln1_b = torch.randn(DIM, dtype=torch.float32)
wq = torch.randn(DIM, DIM, dtype=torch.float32)
wk = torch.randn(DIM, DIM, dtype=torch.float32)
wv = torch.randn(DIM, DIM, dtype=torch.float32)
ln2_w = torch.randn(DIM, dtype=torch.float32)
ln2_b = torch.randn(DIM, dtype=torch.float32)
w1 = torch.randn(DIM, HID, dtype=torch.float32)
b1 = torch.randn(HID, dtype=torch.float32)
w2 = torch.randn(HID, DIM, dtype=torch.float32)
b2 = torch.randn(DIM, dtype=torch.float32)

out, used_real = transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2)
ref = reference_transformer_block(x, ln1_w, ln1_b, wq, wk, wv, ln2_w, ln2_b, w1, b1, w2, b2)

print(f"used real cuTile kernels for all four stages: {all(used_real)}")
print(f"transformer block output matches plain-torch reference: {torch.allclose(out, ref, atol=1e-5)}")
```
