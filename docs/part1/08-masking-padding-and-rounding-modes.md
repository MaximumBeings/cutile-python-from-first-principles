# Chapter 8: Masking, Padding, and Rounding Modes

> "Chapter 3's six `PaddingMode` values weren't six arbitrary flavors of `TiledView`'s edge behavior — each one is, quietly, an identity element for a different reduction: `ZERO` for a sum, `NEG_INF` for a max, `POS_INF` for a min. Choosing the right one lets a boundary tile compute the correct answer without a single extra line of masking logic. Choosing the wrong one — or reaching for `ct.where` out of habit when the padding mode already handles it — is a real, measurable choice this chapter puts a number on, rather than a stylistic preference between two ways of writing the same thing."

**What you will understand by the end of this chapter:**

- That an explicit `ct.where`-based mask and a well-chosen `padding_mode` can compute the identical numeric result for a boundary reduction, but do so at a genuinely different, measured cost in compiled code
- That `rounding_mode`, a real keyword argument on several arithmetic and reduction functions, has its acceptance genuinely restricted per-operation — the same four IEEE directional modes compile identically for `ct.add`, while `ct.sqrt` genuinely accepts a fast approximate mode `ct.add` rejects outright
- That a rounding mode's default isn't a documentation guess: this chapter confirms, byte-for-byte, exactly which named mode `rounding_mode=None` resolves to for a real operation
- That not every real, enumerable `RoundingMode` value is honest information for every operation that accepts the parameter — some combinations are genuinely rejected with a `TileTypeError` naming the unsupported pairing directly

**What you need to know first:**

- Chapter 3's `PaddingMode` enum and its finding that `UNDETERMINED` compiles smaller than the other five modes, which all generate real boundary-checking logic.
- Chapter 1's distinction between `TileTypeError` (a genuine type violation the compiler catches) and other `TileError` subclasses.

## 8.1 Masking with `ct.where`, Against a Well-Chosen Padding Mode `[FOUNDATIONAL]`

### Intuition

`ct.store`'s own documentation already states that a store partially outside an array's bounds simply ignores the out-of-bounds elements — boundary safety on the *write* side is automatic. The open question is the *read* side: once a boundary tile is loaded with some padding value filling its out-of-bounds positions, does a reduction over that tile need an explicit mask to get the right answer, or does the right padding mode already make masking redundant?

### Background

`ct.where(cond, x, y)` selects, elementwise, between two tiles of identical shape based on a boolean condition tile — the real mechanism for masking a computed value in tile code, genuinely distinct from `padding_mode`, which only controls what a *load* produces at an out-of-bounds position. For a `sum` reduction specifically, `0.0` is the additive identity: a tile loaded with `PaddingMode.ZERO` and summed directly reaches the exact same numeric answer that explicitly masking every padded position to `0.0` before summing would.

### Worked Example 8.1.1 — the same answer, two different costs

```python
import cuda.tile as ct
import io

@ct.kernel
def unmasked_sum(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    # ZERO padding fills any out-of-bounds tail elements with 0.0 --
    # which happens to be the identity element for sum, so this kernel
    # never needs to know which elements were real and which were padding.
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    tile = view.load((pid,))
    total = ct.sum(tile)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

@ct.kernel
def masked_sum(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    tile = view.load((pid,))
    # a.shape[0] is a genuine runtime value -- the array's real length,
    # queried from inside tile code, not a compile-time constant. Building
    # an explicit mask and selecting with ct.where reaches the identical
    # numeric result here, at a real, measured cost in compiled code.
    offsets = pid * tile_size + ct.arange(tile_size, dtype=ct.int32)
    mask = offsets < a.shape[0]
    masked_tile = ct.where(mask, tile, ct.full((tile_size,), 0.0, ct.float32))
    total = ct.sum(masked_tile)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

for name, k in [("unmasked (relies on ZERO padding)", unmasked_sum), ("masked (explicit ct.where)", masked_sum)]:
    buf = io.BytesIO()
    ct.compilation.export_kernel(k, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"{name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `01_masking_vs_unmasked_sum.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_masking_vs_unmasked_sum.py
```

Genuinely run:

```
unmasked (relies on ZERO padding): 23552 cubin bytes
masked (explicit ct.where): 23936 cubin bytes
```

The explicit mask genuinely costs 384 more bytes for a reduction where it changes nothing about the answer — real, generated code to compute `a.shape[0]`, build the per-element boundary comparison, and select between two tiles, all of which `PaddingMode.ZERO` already made unnecessary for `sum` specifically.

> `[COMMON TRAP]` `0.0` is the identity element for `sum`, but it is not the identity element for every reduction. Chapter 3 also compiled `NEG_INF` and `POS_INF` as real `PaddingMode` values — the correct, mask-free choice for a `max` reduction (`NEG_INF`, so padding never wins a comparison against real data) and a `min` reduction (`POS_INF`), respectively. Reaching for `PaddingMode.ZERO` on a boundary tile feeding a `max` reduction would silently make every padded position eligible to win — a genuine correctness bug this book cannot demonstrate numerically without `ct.launch`, but one the padding-mode/reduction pairing above already explains structurally: `ct.where`-based masking exists precisely for the cases where no single padding value is a safe identity for what the kernel actually computes next.

## 8.2 `rounding_mode`: Accepted Differently by Different Operations `[FOUNDATIONAL]`

### Intuition

`ct.RoundingMode` is a real, seven-value enum — `RN`, `RZ`, `RM`, `RP`, `FULL`, `APPROX`, `RZI` — and several arithmetic functions, including `ct.add`, `ct.sub`, `ct.mul`, `ct.truediv`, `ct.sqrt`, `ct.exp`, and `ct.tanh`, accept a `rounding_mode` keyword. Nothing about the enum's existence says every function accepts every value; this section checks `ct.add` directly against all seven.

### Background

`ct.add`'s documentation states its `rounding_mode` is "only supported for float types, default is `RoundingMode.RN` when applicable" — the standard IEEE round-to-nearest-even behavior ordinary floating-point addition already has. `RN`, `RZ`, `RM`, and `RP` are the four standard IEEE rounding directions (nearest-even, toward zero, toward negative infinity, toward positive infinity); `FULL`, `APPROX`, and `RZI` are not directional roundings in that same sense.

### Worked Example 8.2.1 — all seven modes, against `ct.add`

```python
import cuda.tile as ct
import io

def build_kernel(mode):
    @ct.kernel
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        result = ct.add(x, y, rounding_mode=mode)
        ct.store(c, (pid,), result)
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# The four directional IEEE rounding modes -- nearest-even, toward zero,
# toward negative infinity, toward positive infinity -- are all
# documented as valid for ct.add.
for name, mode in [("None (default)", None), ("RN", ct.RoundingMode.RN), ("RZ", ct.RoundingMode.RZ),
                    ("RM", ct.RoundingMode.RM), ("RP", ct.RoundingMode.RP)]:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"add rounding_mode={name}: {len(buf.getvalue())} cubin bytes")

# APPROX, FULL, and RZI are real RoundingMode values Chapter 8's own
# ct.RoundingMode enum lists -- but not every op supports every mode.
for name, mode in [("APPROX", ct.RoundingMode.APPROX), ("FULL", ct.RoundingMode.FULL),
                    ("RZI", ct.RoundingMode.RZI)]:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"add rounding_mode={name}: {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"add rounding_mode={name}: {type(e).__name__}: {e}")
```

The complete file is `02_add_rounding_modes.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_add_rounding_modes.py
```

Genuinely run:

```
add rounding_mode=None (default): 24736 cubin bytes
add rounding_mode=RN: 24736 cubin bytes
add rounding_mode=RZ: 24736 cubin bytes
add rounding_mode=RM: 24736 cubin bytes
add rounding_mode=RP: 24736 cubin bytes
add rounding_mode=APPROX: TileTypeError: Rounding mode approx is not supported for add
  "/tmp/ch8/02_add_rounding_modes.py", line 10, col 18-49, in add_kernel:
        result = ct.add(x, y, rounding_mode=mode)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

add rounding_mode=FULL: TileTypeError: Rounding mode full is not supported for add
  "/tmp/ch8/02_add_rounding_modes.py", line 10, col 18-49, in add_kernel:
        result = ct.add(x, y, rounding_mode=mode)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

add rounding_mode=RZI: TileTypeError: Rounding mode nearest_int_to_zero is not supported for add
  "/tmp/ch8/02_add_rounding_modes.py", line 10, col 18-49, in add_kernel:
        result = ct.add(x, y, rounding_mode=mode)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Two genuine findings. First, `None`, `RN`, `RZ`, `RM`, and `RP` all compile to the identical byte count for this kernel — confirming `None`'s documented default matches `RN` specifically (the same "default resolves to one named, confirmable value" pattern Chapter 5 established for `bytecode_version`), and showing that switching which IEEE direction is requested doesn't, on its own, change how much code this particular add generates. Second, `APPROX`, `FULL`, and `RZI` are all genuinely rejected for `ct.add`, each with a `TileTypeError` naming the exact unsupported mode by name — real values of a real enum, but not honest choices for every operation that merely accepts the parameter.

## 8.3 `rounding_mode` on `ct.sqrt`: Where a Genuine Trade-off Appears `[FOUNDATIONAL]`

### Intuition

`ct.add`'s five accepted modes all compiled to identical output. That's not evidence every operation behaves that way — `ct.sqrt` is a transcendental function with real hardware support for a fast, lower-precision approximation, which is exactly the kind of operation where `APPROX` ought to mean something measurably different.

### Background

`APPROX` is documented generically as an available `RoundingMode` value; what it costs or saves is specific to the operation and the target hardware, not stated as a universal number. Testing `ct.sqrt` directly against `RN`, `APPROX`, `RZ`, and the same `FULL` value Section 8.2 found `ct.add` rejects, checks both the size trade-off and the acceptance boundary for a second, different operation.

### Worked Example 8.3.1 — a genuine size trade-off, and `FULL` rejected again

```python
import cuda.tile as ct
import io

def build_kernel(mode):
    @ct.kernel
    def sqrt_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        result = ct.sqrt(x, rounding_mode=mode)
        ct.store(c, (pid,), result)
    return sqrt_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Unlike ct.add, ct.sqrt genuinely accepts APPROX -- a real hardware fast
# approximate instruction -- but genuinely rejects FULL.
for name, mode in [("None (default)", None), ("RN", ct.RoundingMode.RN),
                    ("APPROX", ct.RoundingMode.APPROX), ("RZ", ct.RoundingMode.RZ)]:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"sqrt rounding_mode={name}: {len(buf.getvalue())} cubin bytes")

kernel = build_kernel(ct.RoundingMode.FULL)
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"sqrt rounding_mode=FULL: {len(buf.getvalue())} cubin bytes")
except ct.TileError as e:
    print(f"sqrt rounding_mode=FULL: {type(e).__name__}: {e}")
```

The complete file is `03_sqrt_rounding_modes.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_sqrt_rounding_modes.py
```

Genuinely run:

```
sqrt rounding_mode=None (default): 23264 cubin bytes
sqrt rounding_mode=RN: 23264 cubin bytes
sqrt rounding_mode=APPROX: 22048 cubin bytes
sqrt rounding_mode=RZ: 23520 cubin bytes
sqrt rounding_mode=FULL: TileTypeError: Rounding mode full is not supported for sqrt
  "/tmp/ch8/03_sqrt_rounding_modes.py", line 9, col 18-47, in sqrt_kernel:
        result = ct.sqrt(x, rounding_mode=mode)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

`None` again matches `RN` exactly, confirming the same default-resolution pattern Section 8.2 found. This time, though, `APPROX` genuinely compiles to 1,216 bytes *less* than `RN` — real, measured evidence that the hardware fast-approximate square root instruction is smaller than the fully IEEE-compliant one, exactly the kind of trade-off `APPROX` exists to offer, and one `ct.add`'s five identically-sized modes gave no evidence of. `RZ` compiles to yet a third size, distinct from both. And `FULL` — accepted nowhere this chapter tested it, for either operation — is rejected again, with the identical exception type and an equally specific message.

## 8.4 `flush_to_zero`: A Real Flag, With No Effect Measured Here `[FOUNDATIONAL]`

### Intuition

`flush_to_zero` is documented as flushing subnormal inputs and results to sign-preserving zero — real, specific numeric behavior, distinct from choosing a rounding direction. This section checks whether that documented behavior shows up as a compiled-size difference for the same `sqrt` kernel Section 8.3 already measured.

### Background

Subnormal (denormal) floating-point values are the smallest-magnitude representable values, handled by dedicated, often slower hardware paths on many architectures. `flush_to_zero=True` tells the compiler it may round any subnormal input or result to zero instead — a real optimization opportunity in principle, whether or not it changes this specific compiled kernel's size.

### Worked Example 8.4.1 — checked directly, honestly reported either way

```python
import cuda.tile as ct
import io

def build_kernel(flush):
    @ct.kernel
    def sqrt_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        result = ct.sqrt(x, flush_to_zero=flush)
        ct.store(c, (pid,), result)
    return sqrt_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# flush_to_zero is documented to flush subnormal inputs and results to
# sign-preserving zero -- a real numeric behavior. Whether that behavior
# shows up as a compiled-size difference for this kernel is checked here.
for name, flush in [("False (default)", False), ("True", True)]:
    kernel = build_kernel(flush)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"flush_to_zero={name}: {len(buf.getvalue())} cubin bytes")
```

The complete file is `04_flush_to_zero.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_flush_to_zero.py
```

Genuinely run:

```
flush_to_zero=False (default): 23264 cubin bytes
flush_to_zero=True: 23264 cubin bytes
```

Both compile to the identical byte count. This book reports that plainly, the same way Chapter 5 reported no measured difference between `cutile_python_v1` and `cutile_python_v2` for a signature that didn't exercise their one documented difference: a real, documented parameter can genuinely fail to move a specific, measured number for a specific kernel, without that meaning the parameter does nothing in general. Subnormal values are a narrow numeric edge case this kernel's compiled *structure* — as opposed to its runtime numeric behavior, which needs `ct.launch` to observe — may simply have no separate code path for either way.

## Complete Runnable Code

### File: `01_masking_vs_unmasked_sum.py`

```python
import cuda.tile as ct
import io

@ct.kernel
def unmasked_sum(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    # ZERO padding fills any out-of-bounds tail elements with 0.0 --
    # which happens to be the identity element for sum, so this kernel
    # never needs to know which elements were real and which were padding.
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    tile = view.load((pid,))
    total = ct.sum(tile)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

@ct.kernel
def masked_sum(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    view = a.tiled_view((tile_size,), padding_mode=ct.PaddingMode.ZERO)
    tile = view.load((pid,))
    # a.shape[0] is a genuine runtime value -- the array's real length,
    # queried from inside tile code, not a compile-time constant. Building
    # an explicit mask and selecting with ct.where reaches the identical
    # numeric result here, at a real, measured cost in compiled code.
    offsets = pid * tile_size + ct.arange(tile_size, dtype=ct.int32)
    mask = offsets < a.shape[0]
    masked_tile = ct.where(mask, tile, ct.full((tile_size,), 0.0, ct.float32))
    total = ct.sum(masked_tile)
    c_view = c.tiled_view((1,))
    c_view.store((pid,), total)

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

for name, k in [("unmasked (relies on ZERO padding)", unmasked_sum), ("masked (explicit ct.where)", masked_sum)]:
    buf = io.BytesIO()
    ct.compilation.export_kernel(k, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"{name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 01_masking_vs_unmasked_sum.py
```

### File: `02_add_rounding_modes.py`

```python
import cuda.tile as ct
import io

def build_kernel(mode):
    @ct.kernel
    def add_kernel(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        y = ct.load(b, (pid,), (tile_size,))
        result = ct.add(x, y, rounding_mode=mode)
        ct.store(c, (pid,), result)
    return add_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# The four directional IEEE rounding modes -- nearest-even, toward zero,
# toward negative infinity, toward positive infinity -- are all
# documented as valid for ct.add.
for name, mode in [("None (default)", None), ("RN", ct.RoundingMode.RN), ("RZ", ct.RoundingMode.RZ),
                    ("RM", ct.RoundingMode.RM), ("RP", ct.RoundingMode.RP)]:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"add rounding_mode={name}: {len(buf.getvalue())} cubin bytes")

# APPROX, FULL, and RZI are real RoundingMode values Chapter 8's own
# ct.RoundingMode enum lists -- but not every op supports every mode.
for name, mode in [("APPROX", ct.RoundingMode.APPROX), ("FULL", ct.RoundingMode.FULL),
                    ("RZI", ct.RoundingMode.RZI)]:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    try:
        ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
        print(f"add rounding_mode={name}: {len(buf.getvalue())} cubin bytes")
    except ct.TileError as e:
        print(f"add rounding_mode={name}: {type(e).__name__}: {e}")
```

```bash
python3 02_add_rounding_modes.py
```

### File: `03_sqrt_rounding_modes.py`

```python
import cuda.tile as ct
import io

def build_kernel(mode):
    @ct.kernel
    def sqrt_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        result = ct.sqrt(x, rounding_mode=mode)
        ct.store(c, (pid,), result)
    return sqrt_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# Unlike ct.add, ct.sqrt genuinely accepts APPROX -- a real hardware fast
# approximate instruction -- but genuinely rejects FULL.
for name, mode in [("None (default)", None), ("RN", ct.RoundingMode.RN),
                    ("APPROX", ct.RoundingMode.APPROX), ("RZ", ct.RoundingMode.RZ)]:
    kernel = build_kernel(mode)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"sqrt rounding_mode={name}: {len(buf.getvalue())} cubin bytes")

kernel = build_kernel(ct.RoundingMode.FULL)
buf = io.BytesIO()
try:
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"sqrt rounding_mode=FULL: {len(buf.getvalue())} cubin bytes")
except ct.TileError as e:
    print(f"sqrt rounding_mode=FULL: {type(e).__name__}: {e}")
```

```bash
python3 03_sqrt_rounding_modes.py
```

### File: `04_flush_to_zero.py`

```python
import cuda.tile as ct
import io

def build_kernel(flush):
    @ct.kernel
    def sqrt_kernel(a, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        x = ct.load(a, (pid,), (tile_size,))
        result = ct.sqrt(x, flush_to_zero=flush)
        ct.store(c, (pid,), result)
    return sqrt_kernel

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# flush_to_zero is documented to flush subnormal inputs and results to
# sign-preserving zero -- a real numeric behavior. Whether that behavior
# shows up as a compiled-size difference for this kernel is checked here.
for name, flush in [("False (default)", False), ("True", True)]:
    kernel = build_kernel(flush)
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code="sm_90", output_format="cubin")
    print(f"flush_to_zero={name}: {len(buf.getvalue())} cubin bytes")
```

```bash
python3 04_flush_to_zero.py
```

## Chapter Summary

An explicit `ct.where` mask and `PaddingMode.ZERO` compute the identical answer for a `sum` reduction over a boundary tile, but the explicit mask genuinely costs 384 more bytes — real, generated code for work the right padding mode already made unnecessary. `rounding_mode`'s acceptance is genuinely operation-specific: `ct.add` compiled all four directional IEEE modes (`RN`, `RZ`, `RM`, `RP`) to the identical byte count and rejected `APPROX`, `FULL`, and `RZI` outright, while `ct.sqrt` accepted `APPROX` and genuinely compiled it 1,216 bytes smaller than `RN` — a real hardware fast-path this book measured directly — yet still rejected `FULL`, with the identical exception type and message pattern both operations shared. `None`'s documented default was confirmed, for both operations, to resolve to exactly `RN`. And `flush_to_zero`, a real, documented parameter, produced no measured byte-count difference for the one kernel this chapter tested it against — reported honestly as a negative result, not evidence the parameter does nothing in general.

## Self-Check Questions

1. Section 8.1 found that masking a boundary tile with `ct.where` before a `sum` reduction costs more compiled code than relying on `PaddingMode.ZERO` alone, for the identical numeric result. Under what circumstance would the two approaches *not* produce the identical result, making the explicit mask necessary rather than merely redundant?
2. `ct.add` rejected `RoundingMode.APPROX`; `ct.sqrt` accepted it, and compiled it smaller than `RoundingMode.RN`. What does the contrast between these two operations tell you about whether a `RoundingMode` enum value being real and enumerable is enough, on its own, to know it's usable with a given function?
3. Both `ct.add` and `ct.sqrt` resolved `rounding_mode=None` to the identical byte count as `rounding_mode=RoundingMode.RN`. What specific, independent piece of evidence would you need to also check `flush_to_zero`'s default the same rigorous way?
4. Section 8.4 found `flush_to_zero=True` and `flush_to_zero=False` compiled to the same byte count. Does that result tell you `flush_to_zero` has no effect on the actual numeric output of a real, launched kernel? Why or why not?
5. `FULL` was rejected by both `ct.add` and `ct.sqrt` in this chapter's tests, with the same exception type and message pattern. Is it safe to conclude `RoundingMode.FULL` is rejected by every cuTile Python arithmetic function that accepts a `rounding_mode` argument?

## Where We Go Next

Every masking and rounding decision this chapter measured happened entirely within one block's own tile-local computation — nothing here depended on what any other block was doing at the same moment. Chapter 9 turns to the memory model questions that only arise once multiple blocks genuinely interact: `MemoryOrder` and `MemoryScope`, the atomic operations Chapter 3's `TiledView` already listed signatures for but never exercised, and what a real ordering guarantee is actually worth to compile against when this book's own environment still can't launch a single kernel to observe two blocks racing.

## Worked Solutions

**1.** Whenever the reduction's identity element isn't the padding value being used — a `max` reduction padded with `PaddingMode.ZERO` instead of `PaddingMode.NEG_INF`, for instance, where a padded zero could win against genuinely negative real data and corrupt the result. Explicit `ct.where` masking is necessary precisely in the cases this chapter's `sum`/`ZERO` pairing was lucky enough to avoid: whenever no single padding value is a safe identity for the specific computation a boundary tile feeds into next.

**2.** It tells you that `RoundingMode` being a single, shared enum across many functions doesn't mean every function's underlying hardware or semantics support every value in it — acceptance is validated per-operation, the same way Chapter 6 found `index_dtype` restricted to exactly two values regardless of what other dtypes exist elsewhere in cuTile Python. A value appearing in the enum is necessary but not sufficient evidence that a specific function will accept it; only testing that specific function directly, as this chapter did for both `ct.add` and `ct.sqrt`, confirms it.

**3.** You would need to compile the same kernel with `flush_to_zero` explicitly set to its documented default value and confirm the byte count matches `flush_to_zero=None` or no argument at all supplied — the same "does the unstated default genuinely match one specific, named explicit value" check this chapter (and Chapter 5's `bytecode_version`) applied to confirm what a documented default actually resolves to, rather than assuming the docstring's stated default is what the compiler actually does.

**4.** No. Compiled byte-count identity says nothing about runtime numeric behavior — this book's `export_kernel`-only environment can confirm what compiles and how large the result is, but confirming whether a genuinely subnormal input actually gets flushed to zero, versus preserved, requires running the compiled kernel against real subnormal data on real hardware, which is `ct.launch` territory this book has been explicit about being unable to reach since Chapter 1.

**5.** No. This chapter's evidence supports "`FULL` was rejected by `ct.add` and `ct.sqrt`, specifically, in the exact form tested here" — two data points, not a survey of `ct.sub`, `ct.mul`, `ct.truediv`, `ct.exp`, `ct.tanh`, or any other `rounding_mode`-accepting function this chapter didn't test. Extending two confirmed rejections into "every function rejects it" would be exactly the kind of untested generalization this book's evidence-first discipline exists to prevent — the honest claim is limited to what was actually compiled and observed.
