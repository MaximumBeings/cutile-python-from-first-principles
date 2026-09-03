# Chapter 20: Atomics and Memory Another Block Might Be Touching

> "Chapter 18 closed by describing every kernel this book had written so far as confined to one block reading and writing its own private slice of memory. This chapter is the first to break that confinement on purpose — not by accident, and not as a bug to fix, but as the entire reason the eight functions it covers exist at all."

**What you will understand by the end of this chapter:**

- `ct.atomic_add`, `ct.atomic_and`, `ct.atomic_or`, `ct.atomic_xor`, `ct.atomic_max`, `ct.atomic_min`, `ct.atomic_xchg`, and `ct.atomic_cas` — eight atomic read-modify-write operations that address an array directly through an explicit tile of indices, rather than the `(pid,)`-plus-`tile_size` block offset every `ct.load`/`ct.store` call in this book has used since Chapter 1 — and that `dir(ct)` lists one more of these (`atomic_min`) than Chapter 18's own forward reference named
- That disabling the default bounds check on `atomic_add` (`check_bounds=False`) measurably shrinks the compiled cubin under Chapter 10's naming control — the same kind of real, structural cost Chapter 16 first found bounds checking carrying
- `ct.MemoryOrder` and `ct.MemoryScope`, two enums this book has never needed before: that `atomic_add` accepts four of `MemoryOrder`'s five values but rejects `WEAK` outright, accepts four of `MemoryScope`'s five values but rejects `CLUSTER`, and — the sharper finding — that `MemoryScope.NONE` has no valid pairing with any memory ordering this family actually accepts, making it a documented enum value with no reachable use in this entire operation family
- That `float32` support splits these eight functions into two groups: `atomic_add`, `atomic_xchg`, and `atomic_cas` compile against `float32` operands, while `atomic_max`, `atomic_min`, and the three bitwise atomics (`atomic_and`/`atomic_or`/`atomic_xor`) all reject `float32` with a fourth distinct `TileTypeError` wording, extending Chapter 19's three-way error-message split to four
- That every atomic docstring's claim to "follow the same convention as `gather()` and `scatter()`" is genuinely true, confirmed by building the exact broadcasting multi-dimensional index pattern `atomic_cas`'s own docstring describes
- That a histogram kernel — the textbook reason atomic operations exist — compiles cleanly using `atomic_add` purely for its side effect, discarding the pre-update value the earlier sections always used

**What you need to know first:**

- Chapter 10's naming-confound rule and Chapter 16's finding that bounds-checking machinery has a real, measurable compiled cost.
- Chapter 17's broadcasting and dtype-promotion vocabulary, and Chapter 19's finding that a single operation family can reject the same input dtype with genuinely different error wordings depending on which specific function is called.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow as every chapter before it. This chapter's entire subject is concurrent access from multiple blocks, and this book has never been able to run more than one block's worth of anything on real hardware with `ct.launch` — every finding here is about what compiles, not about whether the resulting kernel resolves a race correctly.

## 20.1 Eight Functions, One New Addressing Model

### Intuition

Every kernel this book has written loads and stores tiles using a `(pid,)` starting offset and a fixed `tile_size` shape — Chapter 1's foundational addressing model, unchanged for nineteen chapters. `dir(ct)` lists eight atomic operations, and every one of their docstrings describes something structurally different: each takes the array directly, plus an explicit tile of `indices` built with something like `ct.arange`, addressing individual elements rather than a contiguous block-sized slice. `atomic_add()` "follows the same convention as `gather()` and `scatter()`" per its own docstring — machinery this book has never used before, introduced here for the first time because atomics need it.

### Background

Seven of the eight atomics — `atomic_add`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_max`, `atomic_min`, `atomic_xchg` — share an identical three-positional-argument shape: `(array, indices, update, /, *, check_bounds=True, memory_order=MemoryOrder.ACQ_REL, memory_scope=MemoryScope.DEVICE)`. Each reads the array element at the given index, combines it with `update` according to the function's name (add, bitwise AND, bitwise OR, and so on), writes the result back, and returns the value that was there immediately before the update — one call performing the read, the modify, and the write as a single atomic step, with the docstrings explicit that while each individual element's update is atomic, the operation as a whole across all indices is not, and the order of the individual writes is unspecified. `atomic_cas` breaks the pattern with four positional arguments instead of three: `(array, indices, expected, desired, ...)` — compare-and-swap, storing `desired` only where the current value equals `expected`, always returning the value that was actually there. `dir(ct)` also confirms this chapter's first small correction to Chapter 18's own words: Chapter 18's "Where We Go Next" named seven atomics — `atomic_add`, `atomic_max`, `atomic_cas`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_xchg` — and left `atomic_min` out entirely, even though it has sat in `dir(ct)` next to `atomic_max` the whole time.

### Worked Example 20.1.1 — all eight atomics against a simple int32 array

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Every atomic here addresses the array directly with an explicit tile of
# indices, built with ct.arange -- not the (pid,)-plus-tile_size block
# offset every ct.load/ct.store call in this book has used since Chapter 1.
# atomic_add reads array[idx], adds update, writes the result back, and
# returns the value that was there BEFORE the update -- one call performs
# the read, the modify, and the write as a single atomic step.
@ct.kernel
def kernel_add(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_add(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_add(a, idx, update): {compile_bytes(kernel_add, sig)} cubin bytes")

@ct.kernel
def kernel_and(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_and(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_and(a, idx, update): {compile_bytes(kernel_and, sig)} cubin bytes")

@ct.kernel
def kernel_or(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_or(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_or(a, idx, update): {compile_bytes(kernel_or, sig)} cubin bytes")

@ct.kernel
def kernel_xor(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_xor(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_xor(a, idx, update): {compile_bytes(kernel_xor, sig)} cubin bytes")

@ct.kernel
def kernel_max(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_max(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_max(a, idx, update): {compile_bytes(kernel_max, sig)} cubin bytes")

# ct.atomic_min: dir(ct) lists it alongside the other seven, even though
# Chapter 18's own forward reference to this chapter named only seven
# atomics (atomic_add, atomic_max, atomic_cas, atomic_and, atomic_or,
# atomic_xor, atomic_xchg) and left atomic_min out entirely.
@ct.kernel
def kernel_min(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_min(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_min(a, idx, update): {compile_bytes(kernel_min, sig)} cubin bytes")

@ct.kernel
def kernel_xchg(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_xchg(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_xchg(a, idx, update): {compile_bytes(kernel_xchg, sig)} cubin bytes")

# atomic_cas takes FOUR positional arguments -- array, indices, expected,
# desired -- rather than the (array, indices, update) shape every other
# atomic in this family shares.
@ct.kernel
def kernel_cas(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    expected = ct.full((tile_size,), 0, dtype=ct.int32)
    desired = ct.full((tile_size,), 42, dtype=ct.int32)
    old = ct.atomic_cas(a, idx, expected, desired)
    ct.store(out, (pid,), old)
print(f"atomic_cas(a, idx, expected, desired): {compile_bytes(kernel_cas, sig)} cubin bytes")
```

Genuinely run:

```
atomic_add(a, idx, update): 21760 cubin bytes
atomic_and(a, idx, update): 21760 cubin bytes
atomic_or(a, idx, update): 21760 cubin bytes
atomic_xor(a, idx, update): 21760 cubin bytes
atomic_max(a, idx, update): 21760 cubin bytes
atomic_min(a, idx, update): 21760 cubin bytes
atomic_xchg(a, idx, update): 21760 cubin bytes
atomic_cas(a, idx, expected, desired): 21760 cubin bytes
```

### Discussion

All eight compile without incident on the first attempt, in an environment that has never once run `ct.launch` — a reassuring confirmation that this book's AOT, driver-free compilation workflow extends cleanly to operations whose entire reason for existing is concurrent runtime behavior this workflow can never actually observe. All eight also land on the identical 21760-byte count, but `kernel_add`, `kernel_and`, `kernel_or`, `kernel_xor`, `kernel_max`, `kernel_min`, `kernel_xchg`, and `kernel_cas` are eight separate function names — Chapter 10's naming-confound rule means this is reported as raw data, not as evidence that all eight compile to structurally identical code. The genuinely new machinery here is `ct.arange(tile_size, dtype=ct.int32)` supplying `idx` directly to `atomic_add` as its `indices` argument — every prior chapter's `ct.load`/`ct.store` calls took a `(pid,)` starting position and a fixed shape, letting the compiler compute a contiguous range of addresses itself; this chapter's atomics instead take an already-materialized tile of specific index values, the `gather()`/`scatter()` convention this book is meeting for the first time. `dir(ct)`'s quiet correction of Chapter 18's own forward reference — eight atomics where only seven were named — is a small but genuine reminder that even this book's own chapter-to-chapter callbacks are not immune to the same kind of documentation/reality gap Chapters 18 and 19 found in cuTile Python's own docstrings.

## 20.2 The Cost of Checking Bounds

### Intuition

`atomic_add`'s docstring states that by default it checks whether each index falls within the array's bounds, silently skipping the update and returning an implementation-defined value for any index that doesn't — and that `check_bounds=False` disables this check entirely, shifting responsibility for staying in-bounds onto the caller. Chapter 16 already found that a compiler-inserted bounds check can carry a real, measurable byte cost. Does the same hold here?

### Background

Testing this only requires the same kernel body compiled twice, under Chapter 10's naming control, with the only difference being `check_bounds=True` (the default) versus `check_bounds=False`.

### Worked Example 20.2.1 — check_bounds=True vs check_bounds=False

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# The docstring says atomic_add checks indices are in bounds by default,
# skipping the update for any that aren't, and that check_bounds=False
# disables this at the caller's own risk. Under Chapter 10's naming
# control, does the bounds check actually cost anything measurable?
@ct.kernel
def kernel_fn(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_add(a, idx, update, check_bounds=True)
    ct.store(out, (pid,), old)
kernel_checked = kernel_fn
print(f"atomic_add check_bounds=True (default): {compile_bytes(kernel_checked, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_add(a, idx, update, check_bounds=False)
    ct.store(out, (pid,), old)
kernel_unchecked = kernel_fn
print(f"atomic_add check_bounds=False: {compile_bytes(kernel_unchecked, sig)} cubin bytes")
```

Genuinely run:

```
atomic_add check_bounds=True (default): 21760 cubin bytes
atomic_add check_bounds=False: 21504 cubin bytes
```

### Discussion

The prediction holds: disabling the default bounds check shrinks the cubin by exactly 256 bytes, under a genuine, naming-controlled comparison — `kernel_checked` and `kernel_unchecked` are both assigned from a `kernel_fn` defined identically apart from the one keyword argument. This confirms the docstring's description of what `check_bounds=True` does is not just a runtime behavioral claim but corresponds to real, compiled guard logic the compiler inserts and can just as cleanly omit. The 256-byte difference is the same order of magnitude Chapter 16 found for other compiler-inserted bounds machinery, though this book's `export_kernel`-only environment has no way to inspect what specific instructions that difference corresponds to — only that the difference is real, deterministic, and attributable to exactly one flipped keyword argument.

## 20.3 Memory Order, Memory Scope, and an Unreachable Enum Value

### Intuition

Every atomic in this family accepts two keyword-only arguments this book has never encountered before: `memory_order`, defaulting to `MemoryOrder.ACQ_REL`, and `memory_scope`, defaulting to `MemoryScope.DEVICE`. `ct.MemoryOrder` lists five values (`WEAK`, `RELAXED`, `ACQUIRE`, `RELEASE`, `ACQ_REL`) and `ct.MemoryScope` lists five values (`NONE`, `BLOCK`, `CLUSTER`, `DEVICE`, `SYS`) — twenty-five possible pairings in principle. Rather than exhaustively testing every pairing, this section tests each enum's five values independently against `atomic_add`, then follows up on whatever restrictions turn up.

### Background

Neither enum's values are explained in `atomic_add`'s own docstring beyond naming the default; understanding which values are even legal for an atomic operation requires testing each one directly and reading whatever the compiler's own rejection messages say about the ones that fail.

### Worked Example 20.3.1 — every MemoryOrder and MemoryScope value against atomic_add, then chasing the MemoryScope.NONE rejection

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.MemoryOrder has five values. atomic_add's default is ACQ_REL. Trying
# all five against atomic_add directly.
for mo in ct.MemoryOrder:
    def make_kernel(mo=mo):
        @ct.kernel
        def kernel_fn(a, out, tile_size: ct.Constant[int]):
            pid = ct.bid(0)
            idx = ct.arange(tile_size, dtype=ct.int32)
            update = ct.full((tile_size,), 10, dtype=ct.int32)
            old = ct.atomic_add(a, idx, update, memory_order=mo)
            ct.store(out, (pid,), old)
        return kernel_fn
    k = make_kernel()
    try:
        n = compile_bytes(k, sig)
        print(f"atomic_add memory_order={mo.name}: {n} cubin bytes")
    except Exception as e:
        print(f"atomic_add memory_order={mo.name}: {type(e).__name__}: {e}")

# ct.MemoryScope has five values. atomic_add's default is DEVICE. Trying
# all five, each paired with the default memory_order (ACQ_REL).
for ms in ct.MemoryScope:
    def make_kernel(ms=ms):
        @ct.kernel
        def kernel_fn(a, out, tile_size: ct.Constant[int]):
            pid = ct.bid(0)
            idx = ct.arange(tile_size, dtype=ct.int32)
            update = ct.full((tile_size,), 10, dtype=ct.int32)
            old = ct.atomic_add(a, idx, update, memory_scope=ms)
            ct.store(out, (pid,), old)
        return kernel_fn
    k = make_kernel()
    try:
        n = compile_bytes(k, sig)
        print(f"atomic_add memory_scope={ms.name}: {n} cubin bytes")
    except Exception as e:
        print(f"atomic_add memory_scope={ms.name}: {type(e).__name__}: {e}")

# MemoryScope.NONE failed above when paired with the default ACQ_REL
# ordering. Does pairing it with each of the OTHER three orderings this
# family actually accepts (RELAXED, ACQUIRE, RELEASE) fare any better?
for mo in [ct.MemoryOrder.RELAXED, ct.MemoryOrder.ACQUIRE, ct.MemoryOrder.RELEASE]:
    def make_kernel(mo=mo):
        @ct.kernel
        def kernel_fn(a, out, tile_size: ct.Constant[int]):
            pid = ct.bid(0)
            idx = ct.arange(tile_size, dtype=ct.int32)
            update = ct.full((tile_size,), 10, dtype=ct.int32)
            old = ct.atomic_add(a, idx, update, memory_order=mo, memory_scope=ct.MemoryScope.NONE)
            ct.store(out, (pid,), old)
        return kernel_fn
    k = make_kernel()
    try:
        n = compile_bytes(k, sig)
        print(f"atomic_add memory_order={mo.name}, memory_scope=NONE: {n} cubin bytes (unexpected)")
    except Exception as e:
        print(f"atomic_add memory_order={mo.name}, memory_scope=NONE: {type(e).__name__}: {e}")
```

Genuinely run:

```
atomic_add memory_order=WEAK: TileTypeError: Invalid memory order for tile_atomic_rmw. Got MemoryOrder.WEAK, expected one of MemoryOrder.RELAXED, MemoryOrder.ACQUIRE, MemoryOrder.RELEASE, MemoryOrder.ACQ_REL
  "/tmp/ch20/03_memory_order_and_scope.py", line 29, col 19-64, in kernel_fn:
                old = ct.atomic_add(a, idx, update, memory_order=mo)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_add memory_order=RELAXED: 21760 cubin bytes
atomic_add memory_order=ACQUIRE: 21760 cubin bytes
atomic_add memory_order=RELEASE: 21760 cubin bytes
atomic_add memory_order=ACQ_REL: 21760 cubin bytes
atomic_add memory_scope=NONE: TileTypeError: tile_atomic_rmw with ACQ_REL memory ordering requires a memory scope
  "/tmp/ch20/03_memory_order_and_scope.py", line 48, col 19-64, in kernel_fn:
                old = ct.atomic_add(a, idx, update, memory_scope=ms)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_add memory_scope=BLOCK: 21760 cubin bytes
atomic_add memory_scope=CLUSTER: TileTypeError: Invalid memory scope for tile_atomic_rmw. Got MemoryScope.CLUSTER, expected one of MemoryScope.NONE, MemoryScope.BLOCK, MemoryScope.DEVICE, MemoryScope.SYS
  "/tmp/ch20/03_memory_order_and_scope.py", line 48, col 19-64, in kernel_fn:
                old = ct.atomic_add(a, idx, update, memory_scope=ms)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_add memory_scope=DEVICE: 21760 cubin bytes
atomic_add memory_scope=SYS: 21760 cubin bytes
atomic_add memory_order=RELAXED, memory_scope=NONE: TileTypeError: tile_atomic_rmw with RELAXED memory ordering requires a memory scope
  "/tmp/ch20/03_memory_order_and_scope.py", line 68, col 19-98, in kernel_fn:
                old = ct.atomic_add(a, idx, update, memory_order=mo, memory_scope=ct.MemoryScope.NONE)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_add memory_order=ACQUIRE, memory_scope=NONE: TileTypeError: tile_atomic_rmw with ACQUIRE memory ordering requires a memory scope
  "/tmp/ch20/03_memory_order_and_scope.py", line 68, col 19-98, in kernel_fn:
                old = ct.atomic_add(a, idx, update, memory_order=mo, memory_scope=ct.MemoryScope.NONE)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_add memory_order=RELEASE, memory_scope=NONE: TileTypeError: tile_atomic_rmw with RELEASE memory ordering requires a memory scope
  "/tmp/ch20/03_memory_order_and_scope.py", line 68, col 19-98, in kernel_fn:
                old = ct.atomic_add(a, idx, update, memory_order=mo, memory_scope=ct.MemoryScope.NONE)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

`MemoryOrder.WEAK` is rejected outright for `atomic_add` — the error message itself names the four values that remain acceptable (`RELAXED`, `ACQUIRE`, `RELEASE`, `ACQ_REL`), all four of which compile to the identical 21760-byte count seen throughout this chapter under four different function names, so again reported only as raw data. `MemoryScope.CLUSTER` is similarly rejected outright, with an error naming the four scopes that remain (`NONE`, `BLOCK`, `DEVICE`, `SYS`) — `BLOCK`, `DEVICE`, and `SYS` all compile fine.

`MemoryScope.NONE` is the more interesting case. Paired with the default `ACQ_REL` ordering, it fails: "`tile_atomic_rmw with ACQ_REL memory ordering requires a memory scope`" — implying `NONE` might simply need pairing with a weaker ordering instead. But testing all three remaining orderings this family accepts (`RELAXED`, `ACQUIRE`, `RELEASE`) against `MemoryScope.NONE` produces the identical rejection, each just substituting its own ordering's name into the same sentence. Combined with `MemoryOrder.WEAK` being rejected outright in the first test — the one ordering whose name might suggest it's the natural partner for a "no scope" setting — this leaves `MemoryScope.NONE` in a genuinely unusual position: every memory ordering `atomic_add` accepts demands a real, non-`NONE` scope, and the one ordering that might have accepted `NONE` is rejected before the scope is ever considered. Within this specific function and the eight-function family it represents, `MemoryScope.NONE` is a documented, listed enum value with no test found here that successfully uses it — not because `NONE` itself is invalid syntax, but because nothing this book found pairs with it.

## 20.4 A Fourth Wording for Float Rejection

### Intuition

Chapter 19 found that a single bitwise-operation family could reject `float32` operands with three distinct `TileTypeError` wordings depending on exactly which function was called. Atomics reintroduce three of Chapter 19's exact bitwise operations — `atomic_and`, `atomic_or`, `atomic_xor` — alongside five others with no direct Chapter 19 counterpart. Real CUDA hardware has long supported a native `atomicAdd` instruction for `float32`, and `atomicExch` is a pure bit-swap with no arithmetic meaning to restrict by dtype, so this section tests whether float support splits along those hardware-shaped lines.

### Background

Testing all eight functions against `float32` operands directly, rather than assuming any hardware-based prediction holds, is the only way to find out.

### Worked Example 20.4.1 — float32 across all eight atomics

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Real CUDA hardware has long supported a native atomicAdd on float32,
# and atomicExch is a pure bit-swap with no arithmetic meaning to
# restrict by dtype -- so both might reasonably accept float32 where
# Chapter 19's pure bitwise family universally rejected it.
@ct.kernel
def kernel_add(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_add(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_add, sig)
    print(f"atomic_add(float32): {n} cubin bytes")
except Exception as e:
    print(f"atomic_add(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_xchg(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_xchg(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_xchg, sig)
    print(f"atomic_xchg(float32): {n} cubin bytes")
except Exception as e:
    print(f"atomic_xchg(float32): {type(e).__name__}: {e}")

# atomic_cas compares bit patterns for equality, which needs no
# arithmetic ordering over float32 at all -- a third plausible accept.
@ct.kernel
def kernel_cas(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    expected = ct.full((tile_size,), 0.0, dtype=ct.float32)
    desired = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_cas(a, idx, expected, desired)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_cas, sig)
    print(f"atomic_cas(float32): {n} cubin bytes")
except Exception as e:
    print(f"atomic_cas(float32): {type(e).__name__}: {e}")

# atomic_max and atomic_min: real CUDA hardware has never offered a
# native atomicMax/atomicMin instruction for float32 the way it has for
# atomicAdd -- do these two reject float32 while atomic_add just accepted it?
@ct.kernel
def kernel_max(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_max(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_max, sig)
    print(f"atomic_max(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_max(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_min(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_min(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_min, sig)
    print(f"atomic_min(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_min(float32): {type(e).__name__}: {e}")

# atomic_and, atomic_or, atomic_xor: the bitwise family. Chapter 19
# found the pure (non-atomic) bitwise ops universally reject float32.
@ct.kernel
def kernel_and(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_and(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_and, sig)
    print(f"atomic_and(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_and(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_or(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_or(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_or, sig)
    print(f"atomic_or(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_or(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_xor(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_xor(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_xor, sig)
    print(f"atomic_xor(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_xor(float32): {type(e).__name__}: {e}")
```

Genuinely run:

```
atomic_add(float32): 21760 cubin bytes
atomic_xchg(float32): 21760 cubin bytes
atomic_cas(float32): 21760 cubin bytes
atomic_max(float32): TileTypeError: Unsupported array dtype: float32
  "/tmp/ch20/04_float_dtype_split.py", line 74, col 11-39, in kernel_max:
        old = ct.atomic_max(a, idx, update)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_min(float32): TileTypeError: Unsupported array dtype: float32
  "/tmp/ch20/04_float_dtype_split.py", line 87, col 11-39, in kernel_min:
        old = ct.atomic_min(a, idx, update)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_and(float32): TileTypeError: Unsupported array dtype: float32
  "/tmp/ch20/04_float_dtype_split.py", line 102, col 11-39, in kernel_and:
        old = ct.atomic_and(a, idx, update)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_or(float32): TileTypeError: Unsupported array dtype: float32
  "/tmp/ch20/04_float_dtype_split.py", line 115, col 11-38, in kernel_or:
        old = ct.atomic_or(a, idx, update)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_xor(float32): TileTypeError: Unsupported array dtype: float32
  "/tmp/ch20/04_float_dtype_split.py", line 128, col 11-39, in kernel_xor:
        old = ct.atomic_xor(a, idx, update)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

The hardware-shaped prediction holds cleanly. `atomic_add`, `atomic_xchg`, and `atomic_cas` all compile against `float32` — the three atomics with either a real hardware float instruction (`atomicAdd`) or no arithmetic meaning that would need one (a bit-swap, a bit-pattern comparison). `atomic_max` and `atomic_min` both reject `float32`, joining `atomic_and`, `atomic_or`, and `atomic_xor` in an identical rejection: `TileTypeError: Unsupported array dtype: float32`. This wording is new — it matches none of the three float-rejection wordings Chapter 19 catalogued for the pure (non-atomic) bitwise family, making it a fourth distinct wording for what is, once again, fundamentally the same restriction: a dtype the operation's underlying hardware instruction, or its own defined semantics, has no meaningful way to handle. Five of the eight functions here reject `float32` with this one shared wording, while three accept it outright — a cleaner two-way split than Chapter 19's three-way one, but a reminder that "which functions in a family accept a given dtype" continues to be a question this book has found no shortcut around asking directly, docstring language notwithstanding.

## 20.5 The gather()/scatter() Convention, Confirmed Directly

### Intuition

Every atomic docstring states, in identical wording, that it "follows the same convention as `gather()` and `scatter()`." `atomic_cas`'s own docstring goes further, spelling out a concrete multi-dimensional example: a rank-2 array, indices passed as a tuple `(ind0, ind1)` where `ind0` has shape `(M, N, 1)` and `ind1` has shape `(M, 1, K)`, broadcasting together to a common `(M, N, K)` shape — the same broadcasting vocabulary Chapter 17 established for binary elementwise operations, now applied to index tiles instead of value tiles. Does this actually compile, or is it aspirational documentation the way Chapter 18 found `ct.where`'s docstring to be?

### Background

Building this exactly as documented, with `M = N = K = 8` for concreteness, and using `atomic_add` rather than `atomic_cas` to confirm the convention isn't somehow specific to compare-and-swap, is the direct test.

### Worked Example 20.5.1 — broadcasting 2D indices against a rank-2 array

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Every atomic docstring states it "follows the same convention as
# gather() and scatter()" -- machinery this book has never used directly.
# atomic_cas's own docstring spells out a 2D example: a rank-2 array,
# indices as a tuple (ind0, ind1) with ind0 shaped (M, N, 1) and ind1
# shaped (M, 1, K), broadcasting to a common (M, N, K) shape, exactly
# the way Chapter 17's binary elementwise family broadcasts two operand
# shapes together. This section builds that example for real, against
# atomic_add rather than atomic_cas, with M = N = K = 8.
sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_broadcast_atomic(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row = ct.reshape(ct.arange(8, dtype=ct.int32), (8, 1, 1))
    ind0 = ct.broadcast_to(row, (8, 8, 1))
    col = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 1, 8))
    ind1 = ct.broadcast_to(col, (8, 1, 8))
    update = ct.full((8, 8), 1, dtype=ct.int32)
    old = ct.atomic_add(a, (ind0, ind1), update)
    ct.store(out, (pid, 0, 0), old)
try:
    n = compile_bytes(kernel_broadcast_atomic, sig)
    print(f"atomic_add with broadcasting 2D indices (M=N=K=8): {n} cubin bytes")
except Exception as e:
    print(f"atomic_add with broadcasting 2D indices (M=N=K=8): {type(e).__name__}: {e}")
```

Genuinely run:

```
atomic_add with broadcasting 2D indices (M=N=K=8): 27592 cubin bytes
```

### Discussion

The kernel compiles cleanly, confirming the docstring's convention claim directly rather than taking it on faith: `ind0`, built from `ct.arange(8)` reshaped to `(8, 1, 1)` and broadcast to `(8, 8, 1)`, and `ind1`, built the same way but broadcast to `(8, 1, 8)`, combine into a tuple that `atomic_add` accepts as its `indices` argument for a rank-2 array, with `update` shaped `(8, 8)` broadcasting against the resulting `(8, 8, 8)` common index shape exactly as Chapter 17's elementwise family would broadcast two ordinary value tiles. This is a case where testing a docstring's claim against the real compiler simply confirms the claim rather than complicating it — a useful contrast to Chapters 18 and 19, where testing usually turned up either more permissiveness or less than documented. Here, "follows the same convention as `gather()` and `scatter()`" turns out to mean exactly what it says.

## 20.6 Capstone: A Histogram

### Intuition

Every finding in this chapter has been about compiled structure — what compiles, what doesn't, and how large the result is. None of it has touched the actual reason atomic operations exist: preventing two concurrent updates to the same memory location from silently losing one of them. A histogram is the textbook example. Given a tile of values, each interpreted as a bin index, incrementing the corresponding bin's counter with an ordinary (non-atomic) read-modify-write would be wrong the moment two elements — whether from the same block's own tile or from two different blocks entirely — land on the same bin: both could read the same old count, both add one, and both write back the same new count, silently losing an increment.

### Background

This capstone deliberately uses `atomic_add` in a way none of the earlier sections did: discarding the returned "old" value entirely. Every earlier example stored `old` somewhere, because a well-typed kernel body needs every value used or explicitly discarded, and this book's canonical style has generally stored it. A histogram has no use for the pre-update count — only the side effect (the bin got incremented) matters — so this capstone calls `ct.atomic_add` as a bare statement, its return value never bound to a name at all.

### Worked Example 20.6.1 — a histogram kernel

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Capstone: a histogram. Each block loads a tile of bin-index VALUES
# (not addresses) and atomically increments the corresponding counter in
# a shared bins array. This is the textbook reason atomics exist: two
# different blocks -- or even two elements within the same block's own
# tile -- can easily land on the same bin. Without atomic_add, two
# concurrent increments to the same bin could each read the same old
# count, each independently add one, and each write back the same new
# count, silently losing an increment that Chapter 1 through Chapter 19
# never had to worry about, because every kernel until now only ever
# touched memory no other block was touching at the same time.
#
# The "old" value atomic_add returns is discarded here -- a histogram
# only needs the side effect, not the pre-update count -- confirming
# atomic_add works as a fire-and-forget update, not only as a
# read-and-use-the-old-value operation the way Section 20.1 used it.
sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_histogram(values, bins, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(values, (pid,), (tile_size,))
    ones = ct.full((tile_size,), 1, dtype=ct.int32)
    ct.atomic_add(bins, x, ones, memory_order=ct.MemoryOrder.RELAXED)
try:
    n = compile_bytes(kernel_histogram, sig)
    print(f"histogram capstone (atomic_add per-bin increment): {n} cubin bytes")
except Exception as e:
    print(f"histogram capstone (atomic_add per-bin increment): {type(e).__name__}: {e}")
```

Genuinely run:

```
histogram capstone (atomic_add per-bin increment): 22016 cubin bytes
```

### Discussion

The kernel compiles cleanly, confirming `ct.atomic_add` can be called as a bare statement with its return value discarded entirely — a genuinely different usage pattern from every earlier example in this chapter, where `old` was always stored somewhere. `x`, loaded ordinarily with `ct.load` using Chapter 1's block-offset addressing, is then fed directly as `atomic_add`'s `indices` argument: the bin-index VALUES loaded from memory become the addresses the atomic update targets, a genuine composition of this book's original addressing model (loading `values`) with this chapter's new one (indexing `bins`). `memory_order=ct.MemoryOrder.RELAXED` is used deliberately here rather than the default `ACQ_REL`: a histogram's counter increments don't need to establish any ordering relationship with other memory operations the way a lock or a flag might, so the weakest ordering this family actually accepts (Section 20.3 found `WEAK` itself rejected) is the natural choice — though, as with every other claim in this chapter, this book's `export_kernel`-only environment can confirm only that the kernel compiles with this ordering, not that choosing `RELAXED` over `ACQ_REL` produces measurably different behavior on real hardware. As with every capstone since Chapter 15, a clean compile confirms the operations are well-typed and composed correctly — not that running this kernel against real concurrent blocks on real hardware would actually produce a correct histogram, which would require `ct.launch`, unavailable throughout this book.

## Chapter Summary

This chapter introduced cuTile Python's eight atomic operations — `atomic_add`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_max`, `atomic_min`, `atomic_xchg`, and `atomic_cas` — the first operations in this book defined entirely in terms of concurrent access from other blocks, addressed through an explicit tile of indices in the `gather()`/`scatter()` convention rather than the block-offset addressing every earlier chapter used. `dir(ct)` quietly corrected Chapter 18's own forward reference, which had named only seven of these eight, omitting `atomic_min`. Disabling `atomic_add`'s default bounds check measurably shrank the compiled cubin by 256 bytes under Chapter 10's naming control, extending Chapter 16's finding that bounds-checking machinery has a real structural cost. Two new enums, `MemoryOrder` and `MemoryScope`, were tested directly: `atomic_add` rejects `MemoryOrder.WEAK` and `MemoryScope.CLUSTER` outright, and — more sharply — rejects `MemoryScope.NONE` no matter which of the four accepted memory orderings it is paired with, leaving `NONE` a documented enum value with no combination this chapter found that actually compiles for this function family. `float32` support splits the eight functions cleanly in two: `atomic_add`, `atomic_xchg`, and `atomic_cas` accept it, while `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`, and `atomic_xor` reject it with a fourth distinct `TileTypeError` wording, extending Chapter 19's three-wording finding for the pure bitwise family. Every atomic docstring's claim to follow the `gather()`/`scatter()` convention was confirmed directly by building the exact broadcasting multi-dimensional index example `atomic_cas`'s own docstring describes. A histogram capstone combined this book's original block-offset `ct.load` addressing with this chapter's new index-tile addressing in one kernel, using `atomic_add` purely for its side effect with its returned value discarded — the textbook reason atomic operations exist, compiled cleanly in an environment that, as always, could confirm the structure but never the concurrent correctness.

## Self-Check Questions

1. Section 20.1 found `dir(ct)` lists `atomic_min` even though Chapter 18's own forward reference didn't name it. What does this suggest about relying on a chapter's own prose description of what comes next, versus checking `dir(ct)` directly, as a way of scoping what a chapter should actually cover?
2. Section 20.3 found `MemoryScope.NONE` rejected by every memory ordering `atomic_add` actually accepts, and found `MemoryOrder.WEAK` — the one ordering whose name might suggest pairing with "no scope" — rejected outright before scope is even considered. Is it accurate to say `MemoryScope.NONE` is "invalid" for `atomic_add`, or is there a more precise way to describe what this chapter's testing actually established?
3. Section 20.4 found `atomic_max` rejects `float32` while `atomic_add` accepts it, an asymmetry the discussion attributed to real CUDA hardware historically lacking a native `atomicMax` instruction for `float32`. This book's `export_kernel`-only environment produces no actual GPU instructions to inspect. What kind of evidence would you need to confirm the hardware-instruction explanation is the ACTUAL reason, rather than a plausible story that happens to fit the observed pattern?
4. Section 20.5 confirmed the `gather()`/`scatter()` broadcasting convention compiles as documented. Contrast this with Chapter 18's finding about `ct.where`'s docstring and Chapter 19's finding about `bitwise_and`'s docstring. What does having all three kinds of outcome — confirmed as documented, more permissive than documented, and less permissive than documented — in three consecutive chapters suggest about how much weight a docstring's prose should carry on its own, without testing?
5. Section 20.6's histogram discards `atomic_add`'s returned "old" value entirely, while every earlier section in this chapter stored it. What would you need, beyond what this book's `export_kernel`-only environment can check, to confirm that discarding the return value doesn't change anything about the atomic guarantee itself — that is, that the read-modify-write is still genuinely atomic even when the caller never looks at what was read?

## Where We Go Next

This chapter closed the loop Chapter 18 opened: every kernel this book had written before Chapter 20 was confined to memory no other block was touching at the same time, and this chapter finally introduced the eight operations built specifically for the case where that is no longer true. `dir(ct)` still lists operations this book has never touched at all, and the largest of them is impossible to miss: `ct.matmul`, `ct.mma`, and `ct.mma_scaled` — matrix multiplication and matrix multiply-accumulate, the operation a tile-based GPU programming model exists to accelerate in the first place. Every kernel this book has written, from Chapter 1's first tile load through this chapter's histogram, has worked with tiles as flat collections of independent elements, combined elementwise, reduced, scanned, compared, or atomically updated one position at a time — never multiplied as matrices, where every output element depends on an entire row and an entire column of input. Chapter 21 turns to `ct.matmul` and its relatives, the first operations in this book where a tile's two dimensions stop being independent of each other.

## Complete Runnable Code

### File: `01_atomics_basic_all_eight.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Every atomic here addresses the array directly with an explicit tile of
# indices, built with ct.arange -- not the (pid,)-plus-tile_size block
# offset every ct.load/ct.store call in this book has used since Chapter 1.
# atomic_add reads array[idx], adds update, writes the result back, and
# returns the value that was there BEFORE the update -- one call performs
# the read, the modify, and the write as a single atomic step.
@ct.kernel
def kernel_add(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_add(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_add(a, idx, update): {compile_bytes(kernel_add, sig)} cubin bytes")

@ct.kernel
def kernel_and(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_and(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_and(a, idx, update): {compile_bytes(kernel_and, sig)} cubin bytes")

@ct.kernel
def kernel_or(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_or(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_or(a, idx, update): {compile_bytes(kernel_or, sig)} cubin bytes")

@ct.kernel
def kernel_xor(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_xor(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_xor(a, idx, update): {compile_bytes(kernel_xor, sig)} cubin bytes")

@ct.kernel
def kernel_max(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_max(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_max(a, idx, update): {compile_bytes(kernel_max, sig)} cubin bytes")

# ct.atomic_min: dir(ct) lists it alongside the other seven, even though
# Chapter 18's own forward reference to this chapter named only seven
# atomics (atomic_add, atomic_max, atomic_cas, atomic_and, atomic_or,
# atomic_xor, atomic_xchg) and left atomic_min out entirely.
@ct.kernel
def kernel_min(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_min(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_min(a, idx, update): {compile_bytes(kernel_min, sig)} cubin bytes")

@ct.kernel
def kernel_xchg(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_xchg(a, idx, update)
    ct.store(out, (pid,), old)
print(f"atomic_xchg(a, idx, update): {compile_bytes(kernel_xchg, sig)} cubin bytes")

# atomic_cas takes FOUR positional arguments -- array, indices, expected,
# desired -- rather than the (array, indices, update) shape every other
# atomic in this family shares.
@ct.kernel
def kernel_cas(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    expected = ct.full((tile_size,), 0, dtype=ct.int32)
    desired = ct.full((tile_size,), 42, dtype=ct.int32)
    old = ct.atomic_cas(a, idx, expected, desired)
    ct.store(out, (pid,), old)
print(f"atomic_cas(a, idx, expected, desired): {compile_bytes(kernel_cas, sig)} cubin bytes")
```

### File: `02_check_bounds_cost.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# The docstring says atomic_add checks indices are in bounds by default,
# skipping the update for any that aren't, and that check_bounds=False
# disables this at the caller's own risk. Under Chapter 10's naming
# control, does the bounds check actually cost anything measurable?
@ct.kernel
def kernel_fn(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_add(a, idx, update, check_bounds=True)
    ct.store(out, (pid,), old)
kernel_checked = kernel_fn
print(f"atomic_add check_bounds=True (default): {compile_bytes(kernel_checked, sig)} cubin bytes")

@ct.kernel
def kernel_fn(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 10, dtype=ct.int32)
    old = ct.atomic_add(a, idx, update, check_bounds=False)
    ct.store(out, (pid,), old)
kernel_unchecked = kernel_fn
print(f"atomic_add check_bounds=False: {compile_bytes(kernel_unchecked, sig)} cubin bytes")
```

### File: `03_memory_order_and_scope.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.MemoryOrder has five values. atomic_add's default is ACQ_REL. Trying
# all five against atomic_add directly.
for mo in ct.MemoryOrder:
    def make_kernel(mo=mo):
        @ct.kernel
        def kernel_fn(a, out, tile_size: ct.Constant[int]):
            pid = ct.bid(0)
            idx = ct.arange(tile_size, dtype=ct.int32)
            update = ct.full((tile_size,), 10, dtype=ct.int32)
            old = ct.atomic_add(a, idx, update, memory_order=mo)
            ct.store(out, (pid,), old)
        return kernel_fn
    k = make_kernel()
    try:
        n = compile_bytes(k, sig)
        print(f"atomic_add memory_order={mo.name}: {n} cubin bytes")
    except Exception as e:
        print(f"atomic_add memory_order={mo.name}: {type(e).__name__}: {e}")

# ct.MemoryScope has five values. atomic_add's default is DEVICE. Trying
# all five, each paired with the default memory_order (ACQ_REL).
for ms in ct.MemoryScope:
    def make_kernel(ms=ms):
        @ct.kernel
        def kernel_fn(a, out, tile_size: ct.Constant[int]):
            pid = ct.bid(0)
            idx = ct.arange(tile_size, dtype=ct.int32)
            update = ct.full((tile_size,), 10, dtype=ct.int32)
            old = ct.atomic_add(a, idx, update, memory_scope=ms)
            ct.store(out, (pid,), old)
        return kernel_fn
    k = make_kernel()
    try:
        n = compile_bytes(k, sig)
        print(f"atomic_add memory_scope={ms.name}: {n} cubin bytes")
    except Exception as e:
        print(f"atomic_add memory_scope={ms.name}: {type(e).__name__}: {e}")

# MemoryScope.NONE failed above when paired with the default ACQ_REL
# ordering. Does pairing it with each of the OTHER three orderings this
# family actually accepts (RELAXED, ACQUIRE, RELEASE) fare any better?
for mo in [ct.MemoryOrder.RELAXED, ct.MemoryOrder.ACQUIRE, ct.MemoryOrder.RELEASE]:
    def make_kernel(mo=mo):
        @ct.kernel
        def kernel_fn(a, out, tile_size: ct.Constant[int]):
            pid = ct.bid(0)
            idx = ct.arange(tile_size, dtype=ct.int32)
            update = ct.full((tile_size,), 10, dtype=ct.int32)
            old = ct.atomic_add(a, idx, update, memory_order=mo, memory_scope=ct.MemoryScope.NONE)
            ct.store(out, (pid,), old)
        return kernel_fn
    k = make_kernel()
    try:
        n = compile_bytes(k, sig)
        print(f"atomic_add memory_order={mo.name}, memory_scope=NONE: {n} cubin bytes (unexpected)")
    except Exception as e:
        print(f"atomic_add memory_order={mo.name}, memory_scope=NONE: {type(e).__name__}: {e}")
```

### File: `04_float_dtype_split.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Real CUDA hardware has long supported a native atomicAdd on float32,
# and atomicExch is a pure bit-swap with no arithmetic meaning to
# restrict by dtype -- so both might reasonably accept float32 where
# Chapter 19's pure bitwise family universally rejected it.
@ct.kernel
def kernel_add(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_add(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_add, sig)
    print(f"atomic_add(float32): {n} cubin bytes")
except Exception as e:
    print(f"atomic_add(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_xchg(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_xchg(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_xchg, sig)
    print(f"atomic_xchg(float32): {n} cubin bytes")
except Exception as e:
    print(f"atomic_xchg(float32): {type(e).__name__}: {e}")

# atomic_cas compares bit patterns for equality, which needs no
# arithmetic ordering over float32 at all -- a third plausible accept.
@ct.kernel
def kernel_cas(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    expected = ct.full((tile_size,), 0.0, dtype=ct.float32)
    desired = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_cas(a, idx, expected, desired)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_cas, sig)
    print(f"atomic_cas(float32): {n} cubin bytes")
except Exception as e:
    print(f"atomic_cas(float32): {type(e).__name__}: {e}")

# atomic_max and atomic_min: real CUDA hardware has never offered a
# native atomicMax/atomicMin instruction for float32 the way it has for
# atomicAdd -- do these two reject float32 while atomic_add just accepted it?
@ct.kernel
def kernel_max(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_max(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_max, sig)
    print(f"atomic_max(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_max(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_min(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_min(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_min, sig)
    print(f"atomic_min(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_min(float32): {type(e).__name__}: {e}")

# atomic_and, atomic_or, atomic_xor: the bitwise family. Chapter 19
# found the pure (non-atomic) bitwise ops universally reject float32.
@ct.kernel
def kernel_and(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_and(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_and, sig)
    print(f"atomic_and(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_and(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_or(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_or(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_or, sig)
    print(f"atomic_or(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_or(float32): {type(e).__name__}: {e}")

@ct.kernel
def kernel_xor(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    idx = ct.arange(tile_size, dtype=ct.int32)
    update = ct.full((tile_size,), 1.0, dtype=ct.float32)
    old = ct.atomic_xor(a, idx, update)
    ct.store(out, (pid,), old)
try:
    n = compile_bytes(kernel_xor, sig)
    print(f"atomic_xor(float32): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"atomic_xor(float32): {type(e).__name__}: {e}")
```

### File: `05_broadcast_indices_gather_scatter_convention.py`

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Every atomic docstring states it "follows the same convention as
# gather() and scatter()" -- machinery this book has never used directly.
# atomic_cas's own docstring spells out a 2D example: a rank-2 array,
# indices as a tuple (ind0, ind1) with ind0 shaped (M, N, 1) and ind1
# shaped (M, 1, K), broadcasting to a common (M, N, K) shape, exactly
# the way Chapter 17's binary elementwise family broadcasts two operand
# shapes together. This section builds that example for real, against
# atomic_add rather than atomic_cas, with M = N = K = 8.
sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(3), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_broadcast_atomic(a, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    row = ct.reshape(ct.arange(8, dtype=ct.int32), (8, 1, 1))
    ind0 = ct.broadcast_to(row, (8, 8, 1))
    col = ct.reshape(ct.arange(8, dtype=ct.int32), (1, 1, 8))
    ind1 = ct.broadcast_to(col, (8, 1, 8))
    update = ct.full((8, 8), 1, dtype=ct.int32)
    old = ct.atomic_add(a, (ind0, ind1), update)
    ct.store(out, (pid, 0, 0), old)
try:
    n = compile_bytes(kernel_broadcast_atomic, sig)
    print(f"atomic_add with broadcasting 2D indices (M=N=K=8): {n} cubin bytes")
except Exception as e:
    print(f"atomic_add with broadcasting 2D indices (M=N=K=8): {type(e).__name__}: {e}")
```

### File: `06_capstone_histogram.py`

```python
import cuda.tile as ct
import io

def array_param(dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    return len(buf.getvalue())

# Capstone: a histogram. Each block loads a tile of bin-index VALUES
# (not addresses) and atomically increments the corresponding counter in
# a shared bins array. This is the textbook reason atomics exist: two
# different blocks -- or even two elements within the same block's own
# tile -- can easily land on the same bin. Without atomic_add, two
# concurrent increments to the same bin could each read the same old
# count, each independently add one, and each write back the same new
# count, silently losing an increment that Chapter 1 through Chapter 19
# never had to worry about, because every kernel until now only ever
# touched memory no other block was touching at the same time.
#
# The "old" value atomic_add returns is discarded here -- a histogram
# only needs the side effect, not the pre-update count -- confirming
# atomic_add works as a fire-and-forget update, not only as a
# read-and-use-the-old-value operation the way Section 20.1 used it.
sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_histogram(values, bins, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(values, (pid,), (tile_size,))
    ones = ct.full((tile_size,), 1, dtype=ct.int32)
    ct.atomic_add(bins, x, ones, memory_order=ct.MemoryOrder.RELAXED)
try:
    n = compile_bytes(kernel_histogram, sig)
    print(f"histogram capstone (atomic_add per-bin increment): {n} cubin bytes")
except Exception as e:
    print(f"histogram capstone (atomic_add per-bin increment): {type(e).__name__}: {e}")
```
