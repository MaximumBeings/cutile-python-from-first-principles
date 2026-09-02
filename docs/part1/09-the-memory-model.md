# Chapter 9: The Memory Model

> "Every load and store in this book through Chapter 8 happened inside one block's own private computation — nothing it read or wrote was ever visible to, or dependent on, what any other block was doing at the same moment. `MemoryOrder` and `MemoryScope` are the vocabulary for the question every prior chapter's kernels never had to ask: when this block's write becomes visible to some other block, and to which other blocks specifically. This book still can't launch two blocks and watch them race — but it can compile every legal pairing of order and scope, and let the compiler's own acceptance and rejection of each one say what's actually enforced, independent of what this book can observe running."

**What you will understand by the end of this chapter:**

- That `ct.load` and `ct.store` accept genuinely complementary, non-overlapping subsets of `MemoryOrder` — `load` takes `WEAK`/`RELAXED`/`ACQUIRE` and rejects `RELEASE`/`ACQ_REL`, while `store` takes `WEAK`/`RELAXED`/`RELEASE` and rejects `ACQUIRE`/`ACQ_REL` — confirmed directly from both operations' real rejection messages
- That `RELAXED` and `ACQUIRE`/`RELEASE` genuinely require an explicit `memory_scope` — not merely "meaningless" without one, as the documentation's phrasing might suggest, but a hard compile-time requirement
- That `MemoryScope.CLUSTER`, a real, listed enum value, is genuinely rejected for both `ct.load` and `ct.atomic_add` in this compiler — a restriction narrower than the enum's own membership
- That `ct.atomic_add`'s documented defaults (`ACQ_REL`, `DEVICE`) are genuinely stronger than `ct.load`/`ct.store`'s (`WEAK`, `NONE`), and that its `check_bounds` parameter trades real, measured compiled code for the safety it provides

**What you need to know first:**

- Chapter 1's `ct.load`/`ct.store` signatures and Chapter 6's `ArrayConstraint` aliasing fields, which this chapter's atomics build directly on top of.
- Chapter 1's `TileTypeError` as the exception class for a genuine, compiler-caught type or configuration violation.

## 9.1 `ct.load`'s Memory Order: A Genuine Requirement, Not a Suggestion `[FOUNDATIONAL]`

### Intuition

`ct.load`'s documentation lists `WEAK`, `RELAXED`, and `ACQUIRE` as valid `memory_order` values, and states that `memory_scope` is "only meaningful when `memory_order` is not `WEAK`." That phrasing could describe either a soft recommendation or a hard requirement — this section tests which one it actually is.

### Background

`MemoryOrder.WEAK` is `ct.load`'s own documented default, needing no scope. Every other memory order genuinely orders memory operations relative to some *scope* of other threads — the compiler has to know which threads that ordering applies to before it can generate anything for it.

### Worked Example 9.1.1 — every `MemoryOrder` value, against `ct.load`, with no scope supplied

```python
import cuda.tile as ct
import io

def build_kernel(order):
    @ct.kernel
    def load_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,), memory_order=order)
        ct.store(c, (pid,), x)
    return load_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# WEAK, ct.load's own documented default, needs no memory_scope at all.
# RELAXED and ACQUIRE are documented as valid for a load, but this file
# tests them with memory_scope left at its own default (NONE) --
# RELEASE and ACQ_REL are tested too, even though the documentation
# already restricts load to WEAK/RELAXED/ACQUIRE.
for name, order in [("WEAK (default)", ct.MemoryOrder.WEAK), ("RELAXED", ct.MemoryOrder.RELAXED),
                     ("ACQUIRE", ct.MemoryOrder.ACQUIRE), ("RELEASE", ct.MemoryOrder.RELEASE),
                     ("ACQ_REL", ct.MemoryOrder.ACQ_REL)]:
    kernel = build_kernel(order)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"load memory_order={name}: {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"load memory_order={name}: {type(e).__name__}: {e}")
```

The complete file is `01_load_memory_order_requirements.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_load_memory_order_requirements.py
```

Genuinely run:

```
load memory_order=WEAK (default): 21792 cubin bytes
load memory_order=RELAXED: TileTypeError: tile_load with RELAXED memory ordering requires a memory scope
  "/tmp/ch9/01_load_memory_order_requirements.py", line 8, col 13-64, in load_kernel:
            x = ct.load(a, (pid,), (tile_size,), memory_order=order)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

load memory_order=ACQUIRE: TileTypeError: tile_load with ACQUIRE memory ordering requires a memory scope
  "/tmp/ch9/01_load_memory_order_requirements.py", line 8, col 13-64, in load_kernel:
            x = ct.load(a, (pid,), (tile_size,), memory_order=order)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

load memory_order=RELEASE: TileTypeError: Invalid memory order for tile_load. Got MemoryOrder.RELEASE, expected one of MemoryOrder.RELAXED, MemoryOrder.ACQUIRE, MemoryOrder.WEAK
  "/tmp/ch9/01_load_memory_order_requirements.py", line 8, col 13-64, in load_kernel:
            x = ct.load(a, (pid,), (tile_size,), memory_order=order)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

load memory_order=ACQ_REL: TileTypeError: Invalid memory order for tile_load. Got MemoryOrder.ACQ_REL, expected one of MemoryOrder.RELAXED, MemoryOrder.ACQUIRE, MemoryOrder.WEAK
  "/tmp/ch9/01_load_memory_order_requirements.py", line 8, col 13-64, in load_kernel:
            x = ct.load(a, (pid,), (tile_size,), memory_order=order)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Two genuinely distinct rejection messages, for two genuinely distinct reasons. `RELAXED` and `ACQUIRE` are *legal* orders for a load — the error names the missing ingredient (`requires a memory scope`), not the order itself. `RELEASE` and `ACQ_REL` are rejected on entirely different grounds: the error states the exact, exhaustive set `ct.load` actually accepts, and neither of these two is in it, no scope would fix that.

## 9.2 `ct.load`'s Memory Scope: A Boundary Narrower Than the Enum `[FOUNDATIONAL]`

### Intuition

Section 9.1 established that `RELAXED` and `ACQUIRE` need *some* scope. `MemoryScope` itself lists five real values: `NONE`, `BLOCK`, `CLUSTER`, `DEVICE`, `SYS`. Whether `ct.load` accepts all five once a scope is actually supplied is a separate, genuinely open question.

### Background

`BLOCK` scopes an ordering guarantee to threads within the same block; `DEVICE` to the whole GPU; `SYS` to the whole system, including the host; `CLUSTER` to a thread-block cluster, a genuine hardware grouping above a single block. Nothing in the documentation states that `ct.load` restricts which of these it accepts — this section checks directly.

### Worked Example 9.2.1 — five scope values, paired with a legal order

```python
import cuda.tile as ct
import io

def build_kernel(order, scope):
    @ct.kernel
    def load_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,), memory_order=order, memory_scope=scope)
        ct.store(c, (pid,), x)
    return load_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Pairing RELAXED/ACQUIRE with an explicit memory_scope, as Section 9.1
# found is required, lets each ordering actually compile.
for name, order, scope in [
    ("WEAK, NONE (default)", ct.MemoryOrder.WEAK, ct.MemoryScope.NONE),
    ("RELAXED, BLOCK", ct.MemoryOrder.RELAXED, ct.MemoryScope.BLOCK),
    ("RELAXED, DEVICE", ct.MemoryOrder.RELAXED, ct.MemoryScope.DEVICE),
    ("ACQUIRE, DEVICE", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.DEVICE),
    ("ACQUIRE, SYS", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.SYS),
    ("ACQUIRE, CLUSTER", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.CLUSTER),
]:
    kernel = build_kernel(order, scope)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"load ({name}): {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"load ({name}): {type(e).__name__}: {e}")
```

The complete file is `02_load_memory_scope_values.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_load_memory_scope_values.py
```

Genuinely run:

```
load (WEAK, NONE (default)): 21792 cubin bytes
load (RELAXED, BLOCK): 21792 cubin bytes
load (RELAXED, DEVICE): 21792 cubin bytes
load (ACQUIRE, DEVICE): 21536 cubin bytes
load (ACQUIRE, SYS): 21536 cubin bytes
load (ACQUIRE, CLUSTER): TileTypeError: Invalid memory scope for tile_load. Got MemoryScope.CLUSTER, expected one of MemoryScope.NONE, MemoryScope.BLOCK, MemoryScope.DEVICE, MemoryScope.SYS
  "/tmp/ch9/02_load_memory_scope_values.py", line 8, col 13-84, in load_kernel:
            x = ct.load(a, (pid,), (tile_size,), memory_order=order, memory_scope=scope)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

> `[COMMON TRAP]` `CLUSTER` is a real, listed `MemoryScope` value — genuinely usable elsewhere in cuTile Python's memory model — but genuinely rejected here, for `ct.load` specifically, with the compiler's own error stating the exact four scopes it does accept. A value's membership in an enum is not, on its own, evidence that every function accepting that enum's type will honor every one of its values — the same lesson Chapter 8 already drew from `ct.add` rejecting `RoundingMode.APPROX`.

Two further genuine findings sit inside the accepted rows. `WEAK` and `RELAXED` (at either scope tested) compile to the identical 21,792 bytes — the weakest ordering and an explicitly-scoped-but-still-relaxed ordering costing the same here. And `ACQUIRE` compiles to *less* code than either, at both scopes it was tested against — a stronger ordering guarantee producing smaller output, reported exactly as measured rather than forced into an "stronger ordering costs more" assumption this book's own numbers don't support.

## 9.3 `ct.store`'s Memory Order: The Mirror Image `[FOUNDATIONAL]`

### Intuition

`ct.load` accepts `WEAK`, `RELAXED`, `ACQUIRE`. A store is the operation making a write visible to other threads, not observing one — if cuTile Python's memory model is internally consistent the way real hardware memory models generally are, `ct.store` ought to accept the complementary set instead: `WEAK`, `RELAXED`, `RELEASE`.

### Background

An "acquire" order on a read establishes that subsequent operations in program order can't be reordered before it — appropriate for a load, which is how a thread learns about another thread's prior writes. A "release" order on a write establishes that prior operations in program order can't be reordered after it — appropriate for a store, which is how a thread publishes its own writes for later visibility.

### Worked Example 9.3.1 — `ct.store` against every `MemoryOrder` value

```python
import cuda.tile as ct
import io

def build_kernel(order, scope):
    @ct.kernel
    def store_kernel(a, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.full((tile_size,), 1.0, ct.float32)
        ct.store(a, (pid,), x, memory_order=order, memory_scope=scope)
    return store_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.store's own documented valid values are WEAK, RELAXED, and RELEASE --
# the mirror image of ct.load's WEAK/RELAXED/ACQUIRE. RELAXED and RELEASE
# are tested both with and without an explicit scope, and ACQUIRE/ACQ_REL
# are tested even though load and store give them opposite outcomes.
for name, order, scope in [
    ("WEAK, no scope (default)", ct.MemoryOrder.WEAK, ct.MemoryScope.NONE),
    ("RELAXED, no scope", ct.MemoryOrder.RELAXED, ct.MemoryScope.NONE),
    ("RELAXED, DEVICE", ct.MemoryOrder.RELAXED, ct.MemoryScope.DEVICE),
    ("RELEASE, no scope", ct.MemoryOrder.RELEASE, ct.MemoryScope.NONE),
    ("RELEASE, DEVICE", ct.MemoryOrder.RELEASE, ct.MemoryScope.DEVICE),
    ("ACQUIRE, DEVICE", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.DEVICE),
    ("ACQ_REL, DEVICE", ct.MemoryOrder.ACQ_REL, ct.MemoryScope.DEVICE),
]:
    kernel = build_kernel(order, scope)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"store ({name}): {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"store ({name}): {type(e).__name__}: {e}")
```

The complete file is `03_store_memory_order_requirements.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_store_memory_order_requirements.py
```

Genuinely run:

```
store (WEAK, no scope (default)): 18976 cubin bytes
store (RELAXED, no scope): TileTypeError: tile_store with RELAXED memory ordering requires a memory scope
  "/tmp/ch9/03_store_memory_order_requirements.py", line 9, col 9-70, in store_kernel:
            ct.store(a, (pid,), x, memory_order=order, memory_scope=scope)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

store (RELAXED, DEVICE): 19104 cubin bytes
store (RELEASE, no scope): TileTypeError: tile_store with RELEASE memory ordering requires a memory scope
  "/tmp/ch9/03_store_memory_order_requirements.py", line 9, col 9-70, in store_kernel:
            ct.store(a, (pid,), x, memory_order=order, memory_scope=scope)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

store (RELEASE, DEVICE): 19104 cubin bytes
store (ACQUIRE, DEVICE): TileTypeError: Invalid memory order for tile_store. Got MemoryOrder.ACQUIRE, expected one of MemoryOrder.RELAXED, MemoryOrder.RELEASE, MemoryOrder.WEAK
  "/tmp/ch9/03_store_memory_order_requirements.py", line 9, col 9-70, in store_kernel:
            ct.store(a, (pid,), x, memory_order=order, memory_scope=scope)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

store (ACQ_REL, DEVICE): TileTypeError: Invalid memory order for tile_store. Got MemoryOrder.ACQ_REL, expected one of MemoryOrder.RELAXED, MemoryOrder.RELEASE, MemoryOrder.WEAK
  "/tmp/ch9/03_store_memory_order_requirements.py", line 9, col 9-70, in store_kernel:
            ct.store(a, (pid,), x, memory_order=order, memory_scope=scope)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

The mirror image is genuinely confirmed, exactly as the conceptual argument predicted: `ct.store` accepts `WEAK`, `RELAXED`, and `RELEASE`, and its own error message states that exact set when `ACQUIRE` or `ACQ_REL` is rejected — the complement of `ct.load`'s accepted set, confirmed from both operations' real, independently-raised error text rather than assumed from the acquire/release naming alone. `RELAXED` and `RELEASE` both genuinely require a scope too, the identical requirement Section 9.1 found for `ct.load`.

## 9.4 Atomics: Stronger Defaults, and a Real Cost for Safety `[FOUNDATIONAL]`

### Intuition

`ct.atomic_add`'s documented defaults are `memory_order=ACQ_REL` and `memory_scope=DEVICE` — both meaningfully stronger than `ct.load`/`ct.store`'s `WEAK`/`NONE` defaults. That's a real design choice worth confirming directly: an atomic read-modify-write inherently coordinates across blocks, so a weak, unscoped default would defeat its purpose the way it never would for an ordinary load.

### Background

`ct.atomic_add(array, indices, update, *, check_bounds=True, memory_order=ACQ_REL, memory_scope=DEVICE)` reads an array element, adds `update`, and writes the result back — each individual element update is atomic, though the bulk operation as a whole is not. `check_bounds`, on by default, causes out-of-bounds indices to be silently skipped rather than corrupting memory; disabling it hands that responsibility entirely to the caller.

### Worked Example 9.4.1 — scope restrictions, and the cost of bounds checking

```python
import cuda.tile as ct
import io

def build_kernel(scope, check_bounds):
    @ct.kernel
    def atomic_kernel(a, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        indices = pid * tile_size + ct.arange(tile_size, dtype=ct.int32)
        update = ct.full((tile_size,), 1, ct.int32)
        old = ct.atomic_add(a, indices, update, memory_scope=scope, check_bounds=check_bounds)
    return atomic_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.int32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.atomic_add defaults to memory_order=ACQ_REL, memory_scope=DEVICE --
# a stronger default than ct.load/ct.store's WEAK/NONE, consistent with
# an atomic read-modify-write needing real cross-block visibility.
for name, scope in [("DEVICE (default)", ct.MemoryScope.DEVICE), ("BLOCK", ct.MemoryScope.BLOCK),
                     ("CLUSTER", ct.MemoryScope.CLUSTER), ("SYS", ct.MemoryScope.SYS)]:
    kernel = build_kernel(scope, True)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"atomic_add memory_scope={name}: {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"atomic_add memory_scope={name}: {type(e).__name__}: {e}")

# check_bounds is documented to guard every index against the array's
# real bounds by default -- disabling it removes real, generated code.
for name, check in [("True (default)", True), ("False", False)]:
    kernel = build_kernel(ct.MemoryScope.DEVICE, check)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"atomic_add check_bounds={name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `04_atomic_add_scope_and_bounds.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_atomic_add_scope_and_bounds.py
```

Genuinely run:

```
atomic_add memory_scope=DEVICE (default): 18992 cubin bytes
atomic_add memory_scope=BLOCK: 18864 cubin bytes
atomic_add memory_scope=CLUSTER: TileTypeError: Invalid memory scope for tile_atomic_rmw. Got MemoryScope.CLUSTER, expected one of MemoryScope.NONE, MemoryScope.BLOCK, MemoryScope.DEVICE, MemoryScope.SYS
  "/tmp/ch9/04_atomic_add_scope_and_bounds.py", line 10, col 15-94, in atomic_kernel:
            old = ct.atomic_add(a, indices, update, memory_scope=scope, check_bounds=check_bounds)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

atomic_add memory_scope=SYS: 18992 cubin bytes
atomic_add check_bounds=True (default): 18992 cubin bytes
atomic_add check_bounds=False: 18480 cubin bytes
```

`CLUSTER` is rejected for `ct.atomic_add` too, with the identical accepted-set message Section 9.2 found for `ct.load` — the same real restriction, confirmed a second time on a genuinely different operation, not a one-off quirk specific to loads. `BLOCK`'s scope compiles to less code than `DEVICE`'s or `SYS`'s (which match each other exactly) — a narrower, block-local ordering guarantee costing less than a device- or system-wide one, the direction "stronger scope, more code" would actually predict, in contrast to Section 9.2's `ACQUIRE` result. And `check_bounds=False` genuinely removes 512 bytes of real, generated bounds-checking logic — the clearest, most intuitive result in this chapter: disabling a safety check that the documentation explicitly says shifts responsibility onto the caller measurably shrinks what the compiler has to generate.

## Complete Runnable Code

### File: `01_load_memory_order_requirements.py`

```python
import cuda.tile as ct
import io

def build_kernel(order):
    @ct.kernel
    def load_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,), memory_order=order)
        ct.store(c, (pid,), x)
    return load_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# WEAK, ct.load's own documented default, needs no memory_scope at all.
# RELAXED and ACQUIRE are documented as valid for a load, but this file
# tests them with memory_scope left at its own default (NONE) --
# RELEASE and ACQ_REL are tested too, even though the documentation
# already restricts load to WEAK/RELAXED/ACQUIRE.
for name, order in [("WEAK (default)", ct.MemoryOrder.WEAK), ("RELAXED", ct.MemoryOrder.RELAXED),
                     ("ACQUIRE", ct.MemoryOrder.ACQUIRE), ("RELEASE", ct.MemoryOrder.RELEASE),
                     ("ACQ_REL", ct.MemoryOrder.ACQ_REL)]:
    kernel = build_kernel(order)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"load memory_order={name}: {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"load memory_order={name}: {type(e).__name__}: {e}")
```

```bash
python3 01_load_memory_order_requirements.py
```

### File: `02_load_memory_scope_values.py`

```python
import cuda.tile as ct
import io

def build_kernel(order, scope):
    @ct.kernel
    def load_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,), memory_order=order, memory_scope=scope)
        ct.store(c, (pid,), x)
    return load_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Pairing RELAXED/ACQUIRE with an explicit memory_scope, as Section 9.1
# found is required, lets each ordering actually compile.
for name, order, scope in [
    ("WEAK, NONE (default)", ct.MemoryOrder.WEAK, ct.MemoryScope.NONE),
    ("RELAXED, BLOCK", ct.MemoryOrder.RELAXED, ct.MemoryScope.BLOCK),
    ("RELAXED, DEVICE", ct.MemoryOrder.RELAXED, ct.MemoryScope.DEVICE),
    ("ACQUIRE, DEVICE", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.DEVICE),
    ("ACQUIRE, SYS", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.SYS),
    ("ACQUIRE, CLUSTER", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.CLUSTER),
]:
    kernel = build_kernel(order, scope)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"load ({name}): {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"load ({name}): {type(e).__name__}: {e}")
```

```bash
python3 02_load_memory_scope_values.py
```

### File: `03_store_memory_order_requirements.py`

```python
import cuda.tile as ct
import io

def build_kernel(order, scope):
    @ct.kernel
    def store_kernel(a, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.full((tile_size,), 1.0, ct.float32)
        ct.store(a, (pid,), x, memory_order=order, memory_scope=scope)
    return store_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.store's own documented valid values are WEAK, RELAXED, and RELEASE --
# the mirror image of ct.load's WEAK/RELAXED/ACQUIRE. RELAXED and RELEASE
# are tested both with and without an explicit scope, and ACQUIRE/ACQ_REL
# are tested even though load and store give them opposite outcomes.
for name, order, scope in [
    ("WEAK, no scope (default)", ct.MemoryOrder.WEAK, ct.MemoryScope.NONE),
    ("RELAXED, no scope", ct.MemoryOrder.RELAXED, ct.MemoryScope.NONE),
    ("RELAXED, DEVICE", ct.MemoryOrder.RELAXED, ct.MemoryScope.DEVICE),
    ("RELEASE, no scope", ct.MemoryOrder.RELEASE, ct.MemoryScope.NONE),
    ("RELEASE, DEVICE", ct.MemoryOrder.RELEASE, ct.MemoryScope.DEVICE),
    ("ACQUIRE, DEVICE", ct.MemoryOrder.ACQUIRE, ct.MemoryScope.DEVICE),
    ("ACQ_REL, DEVICE", ct.MemoryOrder.ACQ_REL, ct.MemoryScope.DEVICE),
]:
    kernel = build_kernel(order, scope)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"store ({name}): {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"store ({name}): {type(e).__name__}: {e}")
```

```bash
python3 03_store_memory_order_requirements.py
```

### File: `04_atomic_add_scope_and_bounds.py`

```python
import cuda.tile as ct
import io

def build_kernel(scope, check_bounds):
    @ct.kernel
    def atomic_kernel(a, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        indices = pid * tile_size + ct.arange(tile_size, dtype=ct.int32)
        update = ct.full((tile_size,), 1, ct.int32)
        old = ct.atomic_add(a, indices, update, memory_scope=scope, check_bounds=check_bounds)
    return atomic_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.int32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.atomic_add defaults to memory_order=ACQ_REL, memory_scope=DEVICE --
# a stronger default than ct.load/ct.store's WEAK/NONE, consistent with
# an atomic read-modify-write needing real cross-block visibility.
for name, scope in [("DEVICE (default)", ct.MemoryScope.DEVICE), ("BLOCK", ct.MemoryScope.BLOCK),
                     ("CLUSTER", ct.MemoryScope.CLUSTER), ("SYS", ct.MemoryScope.SYS)]:
    kernel = build_kernel(scope, True)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"atomic_add memory_scope={name}: {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"atomic_add memory_scope={name}: {type(e).__name__}: {e}")

# check_bounds is documented to guard every index against the array's
# real bounds by default -- disabling it removes real, generated code.
for name, check in [("True (default)", True), ("False", False)]:
    kernel = build_kernel(ct.MemoryScope.DEVICE, check)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"atomic_add check_bounds={name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 04_atomic_add_scope_and_bounds.py
```

## Chapter Summary

`ct.load` and `ct.store` accept genuinely complementary, non-overlapping subsets of `MemoryOrder` — `WEAK`/`RELAXED`/`ACQUIRE` for a load, `WEAK`/`RELAXED`/`RELEASE` for a store — confirmed from each operation's own real rejection message, not merely inferred from the acquire/release naming convention. `RELAXED`, `ACQUIRE`, and `RELEASE` all genuinely require an explicit `memory_scope`, a hard compile-time requirement rather than a soft recommendation, with a `TileTypeError` naming exactly what's missing. `MemoryScope.CLUSTER`, a real, listed enum value, is genuinely rejected by both `ct.load` and `ct.atomic_add` in this compiler, each with an error stating the same narrower, four-value accepted set — evidence that enum membership alone never guarantees a specific operation accepts a specific value, an instance of the same lesson Chapter 8 drew from `RoundingMode`. `ct.atomic_add`'s defaults (`ACQ_REL`, `DEVICE`) are genuinely stronger than `ct.load`/`ct.store`'s (`WEAK`, `NONE`), consistent with an atomic read-modify-write needing real cross-block coordination by design. And `check_bounds=False` measurably shrinks compiled output by removing real, generated bounds-checking logic — the most straightforwardly intuitive result this chapter measured.

## Self-Check Questions

1. `ct.load` rejects `RELEASE` and `ACQ_REL` outright, with a different error message than the one it gives for `RELAXED`/`ACQUIRE` without a scope. What is the substantive difference between "this order is illegal for this operation" and "this order is legal but incomplete without more information"?
2. Both `ct.load` and `ct.atomic_add` reject `MemoryScope.CLUSTER` with the identical style of error, naming the same four accepted scopes. Does testing this on two different operations make it safe to conclude `CLUSTER` is rejected everywhere in cuTile Python that accepts a `MemoryScope`? Why or why not?
3. `ct.atomic_add`'s default `memory_order` is `ACQ_REL`, meaningfully stronger than `ct.load`/`ct.store`'s `WEAK` default. Why does an atomic read-modify-write need a stronger default ordering than an ordinary load or store?
4. Section 9.2 found `ACQUIRE` compiling to less code than `WEAK`/`RELAXED` for `ct.load`, while Section 9.4 found `BLOCK`'s scope compiling to less code than `DEVICE`'s/`SYS`'s for `ct.atomic_add` — the second result matching "narrower guarantee, less code" far more directly than the first. What should you conclude about assuming a single, universal relationship between ordering/scope strength and compiled size across different operations?
5. `check_bounds=False` on `ct.atomic_add` measurably shrinks compiled output, and the documentation states it shifts responsibility for valid indices entirely onto the caller. What genuine risk does that trade-off carry that this book's own environment cannot demonstrate directly?

## Where We Go Next

This chapter measured every memory-ordering and atomic choice purely by what it compiles to — real, confirmable evidence about which combinations the compiler accepts and how large the result is, and nothing at all about whether a given ordering actually prevents a real race between two real blocks, which needs `ct.launch` against real hardware this book still cannot reach. Chapter 10 turns to `ct.tune` and autotuning: letting the compiler explore several genuinely different compiled specializations of the same kernel — the same multi-signature portability Chapter 2 and Chapter 5 already established as real and driver-free — and picking among them by a criterion this book can also state honestly: which one is smallest, not which one would actually run fastest on hardware this book has never had.

## Worked Solutions

**1.** "This order is illegal for this operation" — `ct.load` rejecting `RELEASE`/`ACQ_REL` — means no amount of additional configuration, including a scope, would ever make that combination compile; the order itself doesn't belong to that operation's vocabulary at all. "This order is legal but incomplete" — `RELAXED`/`ACQUIRE` without a scope — means the order is a genuine, accepted choice for that operation, but the compiler needs one more piece of information (which threads the ordering applies to) before it can generate anything for it; supplying that missing scope, as Section 9.2 did, is enough to make it compile.

**2.** No — two confirmed rejections, even on two different operations, are still only two data points. Every other function in cuTile Python that accepts a `memory_scope` parameter (`ct.store`, and any of the other atomic operations this chapter didn't test, such as `ct.atomic_max` or `ct.atomic_cas`) would need to be tested directly before extending "load and atomic_add reject CLUSTER" into "everything rejects CLUSTER" — the same caution this book applied to `ct.add`/`ct.sqrt`'s `RoundingMode.FULL` rejection in Chapter 8.

**3.** An atomic read-modify-write is inherently about coordinating a shared value across multiple blocks — reading it, modifying it, and writing it back in a way other blocks' concurrent atomics must observe correctly. A `WEAK` default, appropriate for an ordinary load or store that's frequently just reading or writing a block's own private data, would defeat the entire purpose of an operation whose only reason to exist is safe, ordered coordination between otherwise-independent blocks.

**4.** That there's no single, universal rule connecting "stronger ordering or scope" to "more compiled code" across every function this book has tested — each operation has to be measured on its own terms. Chapter 2's `opt_level`, Chapter 6's `alias_groups`, and Chapter 7's `stride_constant`/`shape_constant` pair already established the same caution for other fields; this chapter's `ACQUIRE`-versus-`BLOCK` contrast is the identical lesson applied to memory ordering and scope specifically.

**5.** With `check_bounds=False`, an index that is genuinely out of the array's bounds produces undefined behavior instead of being safely skipped — a real correctness and memory-safety risk on actual hardware. This book's `export_kernel`-only environment can confirm that disabling the check compiles successfully and produces smaller code, but it cannot demonstrate what "undefined behavior" from a real out-of-bounds atomic access on real device memory actually looks like, because observing that requires a genuine `ct.launch` against a real GPU — precisely the boundary this book has been explicit about not being able to cross since Chapter 1.
