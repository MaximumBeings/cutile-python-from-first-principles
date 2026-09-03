# Chapter 29: torch.autograd.Function as the Graph Node — Part 3 Begins

> "Twenty-eight chapters tested `cuda.tile` alone. This one adds a second real package to the test bench — `torch`, genuinely installed, genuinely exercised — and finds that PyTorch's own autograd machinery can be verified completely on this sandbox's CPU, right up to the exact same driver wall every GPU-launch attempt in this book has always hit."

**What you will understand by the end of this chapter:**

- That this sandbox now has a genuine, working `torch` 2.14.0 installation (CPU-only — `torch.cuda.is_available()` is `False`, confirmed) — installed without disturbing `cuda-tile`'s own compiler toolchain, verified by a byte-for-byte regression check against a known compiled kernel
- `torch.autograd.Function`'s `forward`/`backward`/`apply` contract, exercised with a real reference implementation and validated with a genuine `torch.autograd.gradcheck` call — actual numerical differentiation, not an assumption
- `ctx.save_for_backward` and what `backward` actually receives, confirmed by comparing saved and returned tensors directly, for a two-input operation
- A genuine PyTorch-level misuse: a `backward` that returns the wrong number of gradients, caught with the real error PyTorch raises
- That the real integration path — a `forward` that calls `ct.launch` on an actual CUDA stream — reaches the exact same hardware wall this book has named since it first confirmed there is no GPU or driver here: `torch.cuda.current_stream()` itself fails before `ct.launch`'s own argument validation is ever reached
- A capstone `torch.autograd.Function` that attempts the real cuTile kernel path first, catches that wall honestly, falls back to a clearly labeled reference computation, and still passes a genuine end-to-end `gradcheck`

**What you need to know first:**

- This book's `export_kernel`-only, driver-free verification method, and its long-standing confirmation that this sandbox has no GPU, no driver, and (until this chapter) no `torch`.
- Chapter 27's `ct.tune.exhaustive_search` wall (`TypeError: Stream is required, got None`) and its docstring's own use of `torch.cuda.current_stream()` — the same call this chapter now makes for real.
- Basic familiarity with PyTorch's autograd concepts (`requires_grad`, `.backward()`) is helpful but not assumed.

## 29.1 torch.autograd.Function: forward, backward, apply, and a Real gradcheck

### Intuition

`torch.autograd.Function` is PyTorch's mechanism for inserting a custom operation into the autograd graph: a `forward` staticmethod computes the output, a `backward` staticmethod computes gradients with respect to each input, and `.apply(...)` — never `forward` directly — is how the operation gets called so that autograd can record it. This is the seam Part 3 is about: a `forward` that calls a real cuTile kernel would sit exactly here. Before reaching for that, this section establishes the wiring itself is genuinely correct, using plain `torch` tensor ops as an explicit, clearly-labeled stand-in for what a cuTile kernel would compute — not a claim about the kernel itself, but a real, verifiable claim about the autograd plumbing around it.

### Background

`DoubleReference` computes `y = 2x` in `forward` and returns `2 * grad_output` in `backward` — the correct analytical gradient. Its correctness is checked three ways: comparing the forward output directly against `2 * x`, comparing the computed gradient against the known-correct constant `2.0`, and running `torch.autograd.gradcheck`, which numerically estimates the gradient via finite differences and compares it against `backward`'s analytical answer — a genuine, independent check, not a repeat of the same computation.

### Worked Example 29.1.1 — a reference autograd.Function, checked three ways

```python
import torch

torch.manual_seed(0)

class DoubleReference(torch.autograd.Function):
    # Reference forward: plain torch tensor ops standing in for what a
    # cuTile kernel would compute (y = 2x). This is NOT a real cuTile
    # kernel launch -- Section 29.3 shows exactly why that's unreachable
    # in this sandbox -- but it lets the autograd WIRING itself be
    # tested genuinely, with real gradients, on real (CPU) tensors.
    @staticmethod
    def forward(ctx, x):
        return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = DoubleReference.apply(x)
print(f"forward output: {y.detach().tolist()}")
print(f"expected (2*x): {(2 * x).detach().tolist()}")
print(f"forward matches 2*x exactly: {torch.equal(y, 2 * x)}")

loss = y.sum()
loss.backward()
print(f"x.grad: {x.grad.tolist()}")
print(f"expected grad (all 2.0): {[2.0] * 8}")

ok = torch.autograd.gradcheck(DoubleReference.apply, (x,))
print(f"gradcheck passed: {ok}")
```

Genuinely run:

```
forward output: [3.0819922164880866, -0.5868578115218928, -4.357578764149115, 1.1368625545613356, -2.169044684848042, -2.7971907907417535, 0.8066936952585986, 1.6760526659953197]
expected (2*x): [3.0819922164880866, -0.5868578115218928, -4.357578764149115, 1.1368625545613356, -2.169044684848042, -2.7971907907417535, 0.8066936952585986, 1.6760526659953197]
forward matches 2*x exactly: True
x.grad: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
expected grad (all 2.0): [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
gradcheck passed: True
```

### Discussion

All three checks agree, and for a reason worth being precise about: `torch.equal` and the constant-gradient comparison confirm `DoubleReference` computes what it was written to compute, but `gradcheck` is the check that would have caught a *wrong* `backward` even if `forward` were correct — it perturbs each input element by a small amount, measures the resulting change in output, and compares that numerical estimate against what `backward` actually returned. `torch.float64` matters here, not as a style choice but a genuine requirement: `gradcheck`'s finite-difference estimate needs enough numerical precision that its own approximation error doesn't swamp the comparison, which is why the docstring for `gradcheck` and this book's own worked example both use double precision rather than the `float32` this book has used throughout `cuda.tile` work. Nothing in this section touched a GPU, a CUDA stream, or `cuda.tile` at all — this is genuinely-run evidence that PyTorch's own autograd machinery is available and working correctly in this sandbox, on the CPU, which the next section builds directly on.

## 29.2 ctx.save_for_backward and a Genuine Backward-Arity Error

### Intuition

A `Function` with more than one input needs `backward` to return one gradient per `forward` argument, in the same order — and often needs values computed during `forward` again during `backward`. `ctx.save_for_backward` is the sanctioned way to carry tensors across that boundary. Getting the arity wrong is a common real mistake, and PyTorch's own runtime catches it directly rather than producing a silently wrong gradient.

### Background

`MulReference` computes `x * y`, saving both inputs via `ctx.save_for_backward` and returning `(grad_output * y, grad_output * x)` — the correct product-rule gradients. `MulBroken` is identical except its `backward` returns only one value instead of two.

### Worked Example 29.2.1 — two-input backward, and a real arity mismatch

```python
import torch

torch.manual_seed(0)

# A two-input operation to see exactly what ctx.save_for_backward passes
# through, and what backward receives.
class MulReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x

x = torch.randn(4, dtype=torch.float64, requires_grad=True)
y = torch.randn(4, dtype=torch.float64, requires_grad=True)
out = MulReference.apply(x, y)
print(f"forward output matches x*y: {torch.equal(out, x * y)}")

out.sum().backward()
print(f"x.grad matches y: {torch.equal(x.grad, y)}")
print(f"y.grad matches x: {torch.equal(y.grad, x)}")

ok = torch.autograd.gradcheck(MulReference.apply, (x, y))
print(f"gradcheck passed (two inputs): {ok}")

# backward MUST return one gradient per forward input (2, here). Returning
# only one is a genuine, real PyTorch-level misuse.
class MulBroken(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y  # missing the second gradient

x2 = torch.randn(4, dtype=torch.float64, requires_grad=True)
y2 = torch.randn(4, dtype=torch.float64, requires_grad=True)
out2 = MulBroken.apply(x2, y2)
try:
    out2.sum().backward()
    print("MulBroken.backward(): succeeded (unexpected)")
except Exception as e:
    print(f"MulBroken.backward(): {type(e).__name__}: {e}")
```

Genuinely run:

```
forward output matches x*y: True
x.grad matches y: True
y.grad matches x: True
gradcheck passed (two inputs): True
MulBroken.backward(): RuntimeError: function MulBrokenBackward returned an incorrect number of gradients (expected 2, got 1)
```

### Discussion

`ctx.save_for_backward(x, y)` in `forward` and `x, y = ctx.saved_tensors` in `backward` round-trip exactly the tensors that went in — confirmed directly, not assumed, by comparing the resulting gradients against `x` and `y` themselves via `torch.equal`. The arity mismatch produces `RuntimeError: function MulBrokenBackward returned an incorrect number of gradients (expected 2, got 1)` — PyTorch names the `Function` subclass directly in the message (`MulBrokenBackward`, derived from the class name `MulBroken`) and states both the expected and actual count, which means this failure mode is caught at the moment `backward` runs, not silently misattributed to the wrong input or swallowed into a shape-mismatch error somewhere else. This is exactly the kind of real, checkable-in-a-driver-free-sandbox behavior this book has looked for throughout: nothing here needed a GPU, only a correctly and incorrectly written `Function`.

## 29.3 The Real Integration Path Meets This Book's Hardware Wall

### Intuition

A genuine cuTile-backed `torch.autograd.Function` would call `ct.launch` inside `forward`, on a real CUDA stream, against real tensors. Chapter 27's `ct.tune.exhaustive_search` docstring itself uses `torch.cuda.current_stream()` to obtain that stream in its own worked example. This section makes that exact call for real, to see precisely where — and how — it fails in a sandbox that has never had a GPU or driver.

### Background

`DoubleKernel.forward` calls `torch.cuda.current_stream()`, then `ct.launch(stream, (1,), kernel_double, (x, out, x.shape[0]))` — the real integration pattern, attempted for real, with no fallback.

### Worked Example 29.3.1 — a real forward, calling ct.launch for real

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), 2 * x)

class DoubleKernel(torch.autograd.Function):
    # The REAL integration path: forward attempts to launch the actual
    # cuTile kernel, on the actual CUDA stream ct.tune's own docstring
    # example (Chapter 27) already used -- torch.cuda.current_stream().
    @staticmethod
    def forward(ctx, x):
        stream = torch.cuda.current_stream()
        out = torch.empty_like(x)
        ct.launch(stream, (1,), kernel_double, (x, out, x.shape[0]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

x = torch.randn(8, dtype=torch.float32)
try:
    y = DoubleKernel.apply(x)
    print(f"DoubleKernel.apply(x): {y.tolist()} (unexpected)")
except Exception as e:
    print(f"DoubleKernel.apply(x): {type(e).__name__}: {e}")
```

Genuinely run:

```
DoubleKernel.apply(x): RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```

### Discussion

The failure happens earlier than this book's own `ct.launch`-argument-validation wall from Chapter 27 (`TypeError: Stream is required, got None`) — it happens one call before `ct.launch` is ever reached at all. `torch.cuda.current_stream()` itself is the first thing to fail, with `RuntimeError: Found no NVIDIA driver on your system`, because obtaining *any* real CUDA stream object requires PyTorch to talk to a CUDA driver that simply isn't installed here. This is the same underlying fact this book has stated since early in its research — no GPU, no driver, no `nvidia-smi` — now surfacing through a completely different library's own error message, in a completely different call, confirming it isn't an artifact of `cuda.tile`'s own validation logic but a fact about the sandbox itself that every genuine GPU-launch attempt, regardless of which package makes it, will keep running into.

## 29.4 Capstone: End-to-End Wiring With an Honest Fallback

### Intuition

A production cuTile-backed `Function` should attempt the real kernel path and only fall back when it genuinely cannot proceed — not silently, and not by pretending the fallback is the real thing. This capstone builds exactly that: `forward` tries the real `ct.launch` call, catches the specific `RuntimeError` this chapter has now confirmed is what actually happens here, records which path it took, and falls back to the same reference computation Section 29.1 already validated — with the whole `Function`, fallback included, still passing a genuine `gradcheck`.

### Background

`kernel_double` is compiled ahead-of-time first, to confirm — this book's oldest and most reliable check — that it is a real, valid compiled artifact regardless of whether it can be launched. `DoubleOp.forward` then attempts the real launch inside a `try`/`except RuntimeError`, records `ctx.used_real_kernel` and `ctx.fallback_reason`, and falls back to `2 * x` on failure. Both are readable afterward directly off the output tensor's `grad_fn` — the same object `ctx` refers to during `forward` and `backward`.

### Worked Example 29.4.1 — real kernel attempted first, honest fallback, verified end to end

```python
import cuda.tile as ct
import torch
import io

torch.manual_seed(0)

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), 2 * x)

# Confirm, genuinely, that the kernel this Function wants to launch is a
# real, valid, compiled artifact -- this book's oldest verification
# method, unaffected by whether a GPU exists to run it on.
n_bytes = compile_bytes(kernel_double, sig)
print(f"kernel_double compiles to a real cubin: {n_bytes} bytes")

class DoubleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_double, (x, out, x.shape[0]))
            ctx.used_real_kernel = True
            return out
        except RuntimeError as e:
            # The real cuTile kernel above is a genuine, compiled
            # artifact (confirmed just now) -- but launching it needs a
            # real NVIDIA driver, which this sandbox has never had. This
            # reference path is plain torch, standing in only so the
            # autograd WIRING can still be verified end to end.
            ctx.used_real_kernel = False
            ctx.fallback_reason = f"{type(e).__name__}: {e}"
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = DoubleOp.apply(x)
print(f"used real cuTile kernel launch: {y.grad_fn.used_real_kernel}")
print(f"fallback reason: {y.grad_fn.fallback_reason}")

ok = torch.autograd.gradcheck(DoubleOp.apply, (x,))
print(f"end-to-end gradcheck passed: {ok}")
```

Genuinely run:

```
kernel_double compiles to a real cubin: 20384 bytes
used real cuTile kernel launch: False
fallback reason: RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
end-to-end gradcheck passed: True
```

### Discussion

`kernel_double` compiles to a genuine 20384-byte cubin — a real, complete, valid compiled artifact — and yet `used_real_kernel` still reports `False`, because compiling a kernel and launching it are entirely different capabilities, and this sandbox has always had exactly one of the two. `ctx`'s custom attributes (`used_real_kernel`, `fallback_reason`) survive and are readable directly off `y.grad_fn` after `apply` returns — `grad_fn` *is* the `ctx` object `forward` and `backward` both operated on, not a copy of it, which is a genuinely useful pattern for a real integration: a caller can inspect, after the fact, whether a given call actually used the GPU path or fell back, without the `Function` needing any separate reporting mechanism. Most importantly, `gradcheck` still passes end to end — the fallback's gradient is exactly as correct as the reference implementation Section 29.1 already validated, because they're the identical computation. This is the honest shape a real cuTile-PyTorch integration takes in this sandbox: the kernel exists, compiles, and is correct as compiled code; whether it can be *launched* remains, as it has since this book's very first chapter about `ct.launch`, a claim only real hardware can settle.

## Chapter Summary

This chapter opened Part 3 by adding a second genuine package to this book's test bench: `torch` 2.14.0, installed for real and confirmed not to disturb `cuda-tile`'s own compiler toolchain (a byte-for-byte regression check against a previously-established kernel matched exactly). `torch.autograd.Function`'s `forward`/`backward`/`apply` contract was exercised with a real reference implementation, validated three independent ways including a genuine `torch.autograd.gradcheck` — actual numerical differentiation, not a repeated assertion. `ctx.save_for_backward` was confirmed to round-trip tensors exactly, and a deliberately broken `backward` produced PyTorch's own real arity-mismatch error, naming the offending class directly. The chapter's central finding came from attempting the real integration path for real: a `forward` that calls `ct.launch` on an actual CUDA stream fails one call earlier than expected, at `torch.cuda.current_stream()` itself, with `RuntimeError: Found no NVIDIA driver on your system` — the same underlying fact this book has stated since its research phase, now confirmed through an entirely different library's own error path. The capstone combined everything: a `Function` that genuinely attempts the real kernel, catches that exact failure, falls back honestly to a labeled reference computation, and still passes `gradcheck` end to end — the truthful shape of a cuTile-PyTorch integration in a sandbox with a real compiler and no real GPU.

## Self-Check Questions

1. Section 29.1 used `torch.float64` specifically for its `gradcheck` example, while every `cuda.tile` chapter in this book has used `float32` (or other fixed-width types) throughout. Why does `gradcheck`'s specific technique (numerical differentiation via finite differences) make precision a correctness concern in a way that compiling a kernel to a cubin never was?
2. Section 29.2's `MulBroken.backward` returned one gradient instead of two and was caught immediately with a specific, named error. Suppose `MulBroken.backward` had instead returned two values, but in the wrong order (as if `x` and `y` were swapped). Would `gradcheck` catch this the way it caught the earlier arity mismatch, and would the caught failure look the same in either case?
3. Section 29.3 found `torch.cuda.current_stream()` fails before `ct.launch` is ever called, meaning `ct.launch`'s own `TypeError: Stream is required, got None` (from Chapter 27) and this chapter's `RuntimeError: Found no NVIDIA driver...` are two different errors, from two different libraries, describing what is ultimately the same underlying limitation. What does having BOTH errors, rather than just one, tell you about how independently `cuda.tile` and `torch` each validate their own assumptions about the environment they're running in?
4. Section 29.4's capstone reads `ctx`'s custom attributes off `y.grad_fn` after `apply()` returns. What would have to be true about `ctx` and `grad_fn`'s relationship for this to work reliably — and what would go wrong if `ctx` were instead a fresh object created separately for `forward` and for `backward`?
5. This chapter confirmed `gradcheck` passes for `DoubleOp`'s fallback path, which is identical in computation to `DoubleReference` from Section 29.1. If a future chapter someday runs on real hardware and confirms the actual `ct.launch` path produces numerically correct output too, would that make this chapter's `gradcheck` result from today redundant, or would it still have told this book something the hardware-verified result alone would not?

## Where We Go Next

Part 3's opening chapter established that this sandbox can now genuinely test PyTorch's autograd machinery end to end, right up to the exact hardware boundary this book has always had. The next thread is `backward` itself: every `Function` in this chapter used a hand-derived, trivially correct gradient (`2 * grad_output` for `y = 2x`, the product rule for `y = xy`) written directly in plain `torch`. A real cuTile-backed operation's backward pass would need its OWN kernel — a second `@ct.kernel`, computing the gradient, compiled and (eventually) launched exactly as honestly as the forward kernel in this chapter's capstone was. Part 4, "The Backward Kernel," picks up exactly there.

## Worked Solutions

**1.** Compiling a kernel to a cubin is a discrete, exact process: `export_kernel` either succeeds and produces some fixed sequence of bytes, or it fails with an exception — there's no approximation involved, so `float32`'s reduced precision never enters into whether a byte-count comparison is trustworthy. `gradcheck`, by contrast, works by literally perturbing an input by a small step size, measuring how much the output changes, and dividing to estimate a derivative — a numerical approximation whose own accuracy depends on that step size being small enough to approximate a true derivative, but large enough that floating-point rounding error in computing the difference doesn't dominate the result. `float32`'s roughly 7 decimal digits of precision leaves too little room between "small enough to approximate the true slope" and "large enough to survive rounding" for this technique to work reliably, which is exactly why `gradcheck`'s own convention (and this chapter's usage) is `float64`: doubling the available precision widens that window enough for the finite-difference estimate to be trustworthy.

**2.** `gradcheck` would very likely catch this too, but through a different symptom: rather than an immediate arity-mismatch `RuntimeError` (since two values were returned, matching the two-input arity `backward` is expected to satisfy), `gradcheck` would compute its own numerical gradient estimates for `x` and `y` independently, compare each against the (now-swapped) analytical gradient `MulBroken` returned, and fail with an assertion reporting that the analytical and numerical Jacobians don't match — a "your gradient is wrong" failure rather than a "your gradient has the wrong shape" failure. The two failure modes look meaningfully different in practice: the arity mismatch from Section 29.2 is caught the moment `.backward()` is called, before any gradient values are even compared, while a swapped-order bug would only surface once `gradcheck` actually computes and compares numbers — meaning a swapped-order bug could, in principle, slip past code that never calls `gradcheck` at all, since `.backward()` alone has no way to know the returned gradients are logically swapped rather than merely present.

**3.** Having both errors — `cuda.tile`'s own `TypeError: Stream is required, got None` when handed a literal `None`, and `torch`'s `RuntimeError: Found no NVIDIA driver...` when asked to actually produce a stream object — shows that each library validates its own assumptions independently, at the point where each one's own contract would otherwise be violated, rather than relying on the other to have already checked. `cuda.tile` doesn't know or care how a caller obtained their stream object; it only knows that `ct.launch` was handed `None` where a real stream was required, so it raises its own error naming its own expectation. `torch` doesn't know anything about `cuda.tile` at all; it fails purely because `torch.cuda.current_stream()` itself needs to talk to a CUDA driver to answer the question "what is the current stream," and there is no driver to ask. Neither library is aware of the other's validation logic — the two errors are two independent libraries each honestly reporting the first point at which THEIR OWN assumptions about the environment turned out to be false, which is exactly why this book was able to hit both of them from two completely different, unrelated call sites across two different chapters.

**4.** For reading `ctx`'s attributes off `y.grad_fn` to work reliably, `ctx` and `grad_fn` need to be genuinely the same object — not two separately-constructed objects that happen to look similar, and not a copy taken at some point during `apply()`. PyTorch's actual design satisfies this: the `ctx` argument passed into both `forward` and `backward` is the backward-graph node itself, exposed to the user afterward as the output tensor's `grad_fn`, so any attribute set on `ctx` during `forward` is set on the very object `backward` will later receive, and the very object accessible afterward as `grad_fn`. If `ctx` were instead reconstructed separately for `forward` and for `backward` — two distinct objects rather than one object used throughout the operation's lifetime — then `ctx.save_for_backward()` and any custom attribute set during `forward` would have nowhere to go: `backward` would receive a blank `ctx` with no memory of anything `forward` had stored, breaking not just this chapter's custom bookkeeping but the `save_for_backward`/`saved_tensors` mechanism Section 29.2 already depends on entirely.

**5.** It would not be redundant — the two results would answer genuinely different questions. Today's `gradcheck` result establishes that the WIRING is correct: that `DoubleOp`'s `backward` computes the mathematically correct gradient for whatever `forward` actually computes, verified independently of whatever that computation turns out to be. A future hardware-verified confirmation that `ct.launch`'s real output is numerically correct would establish a different fact entirely: that the ACTUAL cuTile kernel computes the right forward VALUES on real hardware — something no amount of driver-free testing in this sandbox, including every `gradcheck` in this chapter, has ever been positioned to confirm, since `gradcheck` only checks that `backward` is consistent with whatever `forward` happens to compute, not that `forward` computes the intended thing when it's finally the real kernel doing the computing. A correct `backward` paired with a subtly wrong real-kernel `forward` would still pass every test this chapter ran; only a genuine hardware run of the actual kernel could rule that out, which is precisely why this chapter was careful never to claim more than it could.

## Complete Runnable Code

### File: `01_function_forward_backward_and_gradcheck.py`

```python
import torch

torch.manual_seed(0)

class DoubleReference(torch.autograd.Function):
    # Reference forward: plain torch tensor ops standing in for what a
    # cuTile kernel would compute (y = 2x). This is NOT a real cuTile
    # kernel launch -- Section 29.3 shows exactly why that's unreachable
    # in this sandbox -- but it lets the autograd WIRING itself be
    # tested genuinely, with real gradients, on real (CPU) tensors.
    @staticmethod
    def forward(ctx, x):
        return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = DoubleReference.apply(x)
print(f"forward output: {y.detach().tolist()}")
print(f"expected (2*x): {(2 * x).detach().tolist()}")
print(f"forward matches 2*x exactly: {torch.equal(y, 2 * x)}")

loss = y.sum()
loss.backward()
print(f"x.grad: {x.grad.tolist()}")
print(f"expected grad (all 2.0): {[2.0] * 8}")

ok = torch.autograd.gradcheck(DoubleReference.apply, (x,))
print(f"gradcheck passed: {ok}")
```

### File: `02_ctx_save_for_backward_and_arity_error.py`

```python
import torch

torch.manual_seed(0)

# A two-input operation to see exactly what ctx.save_for_backward passes
# through, and what backward receives.
class MulReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x

x = torch.randn(4, dtype=torch.float64, requires_grad=True)
y = torch.randn(4, dtype=torch.float64, requires_grad=True)
out = MulReference.apply(x, y)
print(f"forward output matches x*y: {torch.equal(out, x * y)}")

out.sum().backward()
print(f"x.grad matches y: {torch.equal(x.grad, y)}")
print(f"y.grad matches x: {torch.equal(y.grad, x)}")

ok = torch.autograd.gradcheck(MulReference.apply, (x, y))
print(f"gradcheck passed (two inputs): {ok}")

# backward MUST return one gradient per forward input (2, here). Returning
# only one is a genuine, real PyTorch-level misuse.
class MulBroken(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y  # missing the second gradient

x2 = torch.randn(4, dtype=torch.float64, requires_grad=True)
y2 = torch.randn(4, dtype=torch.float64, requires_grad=True)
out2 = MulBroken.apply(x2, y2)
try:
    out2.sum().backward()
    print("MulBroken.backward(): succeeded (unexpected)")
except Exception as e:
    print(f"MulBroken.backward(): {type(e).__name__}: {e}")
```

### File: `03_real_kernel_launch_meets_the_hardware_wall.py`

```python
import cuda.tile as ct
import torch

torch.manual_seed(0)

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), 2 * x)

class DoubleKernel(torch.autograd.Function):
    # The REAL integration path: forward attempts to launch the actual
    # cuTile kernel, on the actual CUDA stream ct.tune's own docstring
    # example (Chapter 27) already used -- torch.cuda.current_stream().
    @staticmethod
    def forward(ctx, x):
        stream = torch.cuda.current_stream()
        out = torch.empty_like(x)
        ct.launch(stream, (1,), kernel_double, (x, out, x.shape[0]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

x = torch.randn(8, dtype=torch.float32)
try:
    y = DoubleKernel.apply(x)
    print(f"DoubleKernel.apply(x): {y.tolist()} (unexpected)")
except Exception as e:
    print(f"DoubleKernel.apply(x): {type(e).__name__}: {e}")
```

### File: `04_capstone_end_to_end_wiring_with_fallback.py`

```python
import cuda.tile as ct
import torch
import io

torch.manual_seed(0)

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(8)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_double(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), 2 * x)

# Confirm, genuinely, that the kernel this Function wants to launch is a
# real, valid, compiled artifact -- this book's oldest verification
# method, unaffected by whether a GPU exists to run it on.
n_bytes = compile_bytes(kernel_double, sig)
print(f"kernel_double compiles to a real cubin: {n_bytes} bytes")

class DoubleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        try:
            stream = torch.cuda.current_stream()
            out = torch.empty_like(x)
            ct.launch(stream, (1,), kernel_double, (x, out, x.shape[0]))
            ctx.used_real_kernel = True
            return out
        except RuntimeError as e:
            # The real cuTile kernel above is a genuine, compiled
            # artifact (confirmed just now) -- but launching it needs a
            # real NVIDIA driver, which this sandbox has never had. This
            # reference path is plain torch, standing in only so the
            # autograd WIRING can still be verified end to end.
            ctx.used_real_kernel = False
            ctx.fallback_reason = f"{type(e).__name__}: {e}"
            return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output

x = torch.randn(8, dtype=torch.float64, requires_grad=True)
y = DoubleOp.apply(x)
print(f"used real cuTile kernel launch: {y.grad_fn.used_real_kernel}")
print(f"fallback reason: {y.grad_fn.fallback_reason}")

ok = torch.autograd.gradcheck(DoubleOp.apply, (x,))
print(f"end-to-end gradcheck passed: {ok}")
```
