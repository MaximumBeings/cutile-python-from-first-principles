# Chapter 25: Compile-Time Metaprogramming and the Last Exceptions This Book Hadn't Triggered

> "Chapter 1 introduced `TileTypeError` as the compiler's way of refusing to guess. Chapter 21 added `TileUnsupportedFeatureError` for hardware the compiler understands but cannot yet target. Between them, this book has treated eight more names in `dir(ct)` as scenery — exception types it could see but had never once caused. This chapter causes seven of them in a single sitting, and finds that the eighth was hiding in plain sight the entire time."

**What you will understand by the end of this chapter:**

- `ct.static_assert(condition, message=None)`: a compile-time-only check whose failure raises `TileStaticAssertionError`, and whose `message` is itself evaluated lazily, only when the assertion actually fails
- `ct.static_eval(expr)`: a way to read compile-time state — shapes, constants, a choice between two structurally different dynamic tiles — using ordinary Python expression syntax, and the two independent ways it can be misused: performing a runtime operation inside it (`TileStaticEvalError`) or assigning to a local variable inside it (`TileSyntaxError`)
- `ct.static_iter(iterable)`: the mechanism that makes a `for` loop's induction variable a genuine compile-time constant at each unrolled copy — confirmed by a case a plain `for i in range(...)` loop cannot handle, and bounded by the identical 1000-iteration cap this book has now met twice, from two unrelated angles
- `ct.function` and `TileRecursionError`: that tile functions are inlined at every call site, that a compile-time-bounded recursive tile function inlines to a finite chain without incident, and that an unbounded one hits a hard, documented 1000-frame inlining limit
- `TileValueError`: raised by Python's own tuple-unpacking syntax when the number of names on the left does not match the length of a dynamic tuple on the right — a mismatch no dtype or shape rule could have caught
- `ct.compiler_timeout(timeout_sec)` and a genuine confound in reproducing it: a persistent, on-disk compiler cache that can silently make a deliberately-tiny timeout succeed instead of firing, and the environment variable that turns the cache off
- `TileCompilerExecutionError`: what happens when a well-formed `gpu_code` string this book's own front-end cannot validate in advance turns out to be one the installed compiler binary does not support — carrying that binary's own stderr text
- That `TileCompilerExecutionError` and `TileCompilerTimeoutError` both inherit from `TileInternalError`, confirmed directly from the exception classes' `__mro__` — meaning this chapter closes out every one of `dir(ct)`'s concrete Tile exception types this book has ever named, without this environment ever raising a *bare* `TileInternalError` of its own

**What you need to know first:**

- Chapter 24's `ct.reduce` tuple-of-tiles return value, which this chapter's `TileValueError` example unpacks incorrectly on purpose.
- The full roster of Tile exception types this book has tracked since Chapter 1 (`TileTypeError`) and Chapter 21 (`TileUnsupportedFeatureError`), and the eight it named as never-triggered at the end of Chapter 24.
- No new environment setup: the same `export_kernel`-only, driver-free compilation workflow this book has used throughout — though this chapter is the first to also need an environment variable set before `cuda.tile` is imported.

## 25.1 ct.static_assert: A Compile-Time Check With Its Own Exception

### Intuition

`ct.static_assert(condition, message=None)` is a compile-time-only sibling of Python's own `assert`: `condition` must evaluate to a constant boolean known at compile time, and if it is `False`, compilation stops with a `TileStaticAssertionError`. Its docstring makes an unusual promise for something that looks like a plain function call: when `condition` is `True`, `message` is never even evaluated. That is worth confirming directly, since a lazily-evaluated argument is not how Python function calls ordinarily behave.

### Background

Three kernels probe this in order: a passing assertion (to confirm compilation proceeds normally when the condition holds), a failing assertion with an f-string `message` that interpolates a compile-time tile attribute, and a failing assertion with no `message` at all — testing the docstring's claim that a missing message becomes an empty string rather than the literal text "None".

### Worked Example 25.1.1 — a passing assertion, then two ways to fail one

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.static_assert(condition, message=None) asserts a compile-time
# boolean. When condition is True, compilation continues and message is
# never even evaluated.
@ct.kernel
def kernel_assert_true(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.static_assert(x.shape[0] == 8, "expected shape 8")
    ct.store(c, (0,), x)
print(f"static_assert(True): {compile_bytes(kernel_assert_true, sig)} cubin bytes")

# When condition is False, message is evaluated with static_eval
# semantics -- including f-string interpolation of compile-time values
# -- and raises TileStaticAssertionError, a type this book has never
# triggered before.
@ct.kernel
def kernel_assert_false(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.static_assert(x.shape[0] == 16, f"expected shape 16, got {x.shape[0]}")
    ct.store(c, (0,), x)
try:
    n = compile_bytes(kernel_assert_false, sig)
    print(f"static_assert(False, message): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_assert(False, message): {type(e).__name__}: {e}")

# message is optional; the docstring says a missing message is replaced
# with an empty string rather than, say, "None".
@ct.kernel
def kernel_assert_no_message(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.static_assert(x.shape[0] == 16)
    ct.store(c, (0,), x)
try:
    n = compile_bytes(kernel_assert_no_message, sig)
    print(f"static_assert(False, no message): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_assert(False, no message): {type(e).__name__}: {e}")
```

Genuinely run:

```
static_assert(True): 20512 cubin bytes
static_assert(False, message): TileStaticAssertionError: Static assertion failed: expected shape 16, got 8
  "/tmp/ch25/01_static_assert_and_triggering_error.py", line 37, col 5-78, in kernel_assert_false:
        ct.static_assert(x.shape[0] == 16, f"expected shape 16, got {x.shape[0]}")
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

static_assert(False, no message): TileStaticAssertionError: Static assertion failed
  "/tmp/ch25/01_static_assert_and_triggering_error.py", line 50, col 5-38, in kernel_assert_no_message:
        ct.static_assert(x.shape[0] == 16)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

`TileStaticAssertionError` is confirmed, genuinely, for the first time in this book — one of eight exception types Chapter 24 flagged as never-triggered. The f-string message resolves its compile-time interpolation correctly, embedding the tile's actual shape (`8`) directly into the error text even though the assertion was checking against `16`. The no-message case confirms the docstring precisely: the message defaults to an empty string, which is why the error text is simply "Static assertion failed" with no trailing colon at all — not "Static assertion failed: " with an empty tail, and not "Static assertion failed: None". This book cannot directly observe from `export_kernel` alone whether `message`'s lazy evaluation genuinely never runs on the passing branch (there is no side effect visible in a static string to check), but the docstring's own claim is specific enough, and consistent enough with everything else `static_assert` does, that there is no reason this environment's results contradict it.

## 25.2 ct.static_eval: Two New Errors From One Function

### Intuition

`ct.static_eval(expr)` evaluates a Python expression using ordinary Python semantics at compile time, rather than the restricted subset tile code otherwise allows. Its docstring draws two lines around what "ordinary Python semantics" does not mean here: the expression must not perform any runtime operation on a dynamic value, and it must not assign to a local variable. Two structurally different kinds of misuse, worth triggering separately to see whether the compiler actually distinguishes them.

### Background

A first kernel establishes ordinary, valid use: reading a tile's compile-time `.shape` attribute, doing arithmetic on a module-level constant, and selecting between two dynamic tiles based on a compile-time condition — exactly the pattern the docstring's own example demonstrates. A second kernel performs a genuine runtime operation (`x + 1` on a dynamic tile) inside `static_eval`. A third assigns to a local variable via the walrus operator inside the expression.

### Worked Example 25.2.1 — valid static_eval, then its two documented misuses

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

N = 8

# ct.static_eval(expr) evaluates expr with ordinary Python semantics at
# compile time. It can read a tile's compile-time shape, do ordinary
# arithmetic on module-level constants, and select between two
# DIFFERENT dynamic tiles based on a compile-time condition.
@ct.kernel
def kernel_eval_basic(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    shape0 = ct.static_eval(x.shape[0])
    doubled = ct.static_eval(N * 2)
    y = ct.static_eval(x if N % 2 == 0 else x + 1)
    ct.store(c, (0,), y)
print(f"static_eval basic (shape/const/select): {compile_bytes(kernel_eval_basic, sig)} cubin bytes")

# static_eval's docstring says the expression must not perform any
# RUNTIME operation -- x + 1, where x is a genuinely dynamic tile, is
# exactly that. This raises TileStaticEvalError, never triggered before
# in this book.
@ct.kernel
def kernel_eval_runtime_op(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = ct.static_eval(x + 1)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_eval_runtime_op, sig)
    print(f"static_eval(x + 1) on dynamic tile: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_eval(x + 1) on dynamic tile: {type(e).__name__}: {e}")

# static_eval's docstring also says the expression must not assign to
# local variables via the walrus operator. This is a different failure
# than the one above -- a genuinely new exception type, TileSyntaxError,
# also never triggered before in this book, and reported with no
# traceback location at all.
@ct.kernel
def kernel_eval_walrus(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = ct.static_eval((z := x.shape[0]) + 1)
    ct.store(c, (0,), ct.full((1,), y, dtype=ct.int32))
try:
    n = compile_bytes(kernel_eval_walrus, sig)
    print(f"static_eval with walrus assignment: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_eval with walrus assignment: {type(e).__name__}: {e}")
```

Genuinely run:

```
static_eval basic (shape/const/select): 20512 cubin bytes
static_eval(x + 1) on dynamic tile: TileStaticEvalError: add: Tile functions cannot be called inside static_eval().
  "/tmp/ch25/02_static_eval_and_two_new_errors.py", line 42, col 9-29, in kernel_eval_runtime_op:
        y = ct.static_eval(x + 1)
            ^^^^^^^^^^^^^^^^^^^^^

static_eval with walrus assignment: TileSyntaxError: static_eval() expression attempted to modify a local variable 'z'
Unknown location
```

### Discussion

Two exception types, never triggered anywhere earlier in this book, both fall out of a single section. `TileStaticEvalError`'s message is worth reading literally: it says "add: Tile functions cannot be called inside static_eval()" — meaning the compiler is treating `x + 1` on a dynamic tile as a call to the tile addition operation, and rejecting that call specifically because of the `static_eval` context it appears in, not because addition itself is invalid. `TileSyntaxError`, by contrast, is reported with no source location at all ("Unknown location") — the first time this book has seen an exception with no `"file", line N` annotation whatsoever, suggesting the walrus check happens at a stage of compilation that has not yet attached (or has already lost) location tracking for the expression it is rejecting. The basic example confirms all three documented capabilities work together in one kernel: a compile-time shape query, ordinary constant arithmetic, and a genuine compile-time branch between two dynamic values — the mechanism the docstring's own example demonstrates for choosing between differently-shaped tiles based on a constant's parity.

## 25.3 ct.static_iter: What "Compile-Time" Actually Buys You Over range()

### Intuition

`for i in ct.static_iter(iterable):` is documented to unroll its loop body once per item, with the induction variable bound to a genuine compile-time constant in each copy. That description could describe an optimization — the compiler recognizing that `range(4)` is small and unrolling it automatically — or it could describe something a plain `for i in range(4):` loop genuinely cannot do at all. Distinguishing those two possibilities requires a test where only a *true* compile-time constant would work: indexing into a Python tuple whose elements have different types or shapes, which is something only a per-iteration concrete value of `i` (not a general integer) could resolve.

### Background

A basic `static_iter` unroll and a plain `for` loop over `range(4)` are compiled first, to confirm both compile without error on their own. Then the same heterogeneous-shape tuple — three tiles of shapes `(4,)`, `(8,)`, and `(16,)` — is indexed by the loop variable inside both a `static_iter` loop and a plain `for` loop, to see whether the difference this section is looking for actually shows up. Finally, `static_iter`'s own documented restrictions are tested directly: `break` and `return` inside its loop body, and an iterable exceeding its documented 1000-iteration cap — checking in each case whether a plain `for` loop shares the same restriction or not.

### Worked Example 25.3.1 — static_iter versus a plain for loop, and static_iter's own limits

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# for ... in ct.static_iter(range(4)) unrolls the loop body once per
# item, with the induction variable bound to a genuine compile-time
# constant at each copy.
@ct.kernel
def kernel_static_iter_basic(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(4)):
        total = total + x * i
    ct.store(c, (0,), total)
print(f"static_iter basic unroll: {compile_bytes(kernel_static_iter_basic, sig)} cubin bytes")

# A plain Python for-loop over range() -- with no ct.static_iter wrapper
# at all -- also compiles without error: range()'s bound is itself a
# compile-time constant, so the loop is still expressible.
@ct.kernel
def kernel_plain_for(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in range(4):
        total = total + x
    ct.store(c, (0,), total)
print(f"plain 'for i in range(4)' (no static_iter): {compile_bytes(kernel_plain_for, sig)} cubin bytes")

# The real difference shows up when the induction variable is used to
# index something whose elements have genuinely DIFFERENT types --
# here, a Python tuple of tiles with three different shapes. static_iter
# compiles a separate, concrete copy of the loop body per item, so `i`
# is a real compile-time constant inside each copy and can select a
# specific tuple element.
@ct.kernel
def kernel_static_iter_heterogeneous(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    tiles = (ct.full((4,), 1, dtype=ct.int32), ct.full((8,), 2, dtype=ct.int32), ct.full((16,), 3, dtype=ct.int32))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(3)):
        t = tiles[i]
        total = total + ct.sum(t, 0) * x
    ct.store(c, (0,), total)
print(f"static_iter indexing heterogeneous-shape tuple: {compile_bytes(kernel_static_iter_heterogeneous, sig)} cubin bytes")

# A plain for-loop's induction variable is NOT treated as a compile-time
# constant the same way, even though range(3) is itself compile-time
# known: the loop body is compiled once, generically, so indexing the
# identical heterogeneous tuple by `i` is rejected.
@ct.kernel
def kernel_plain_for_heterogeneous(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    tiles = (ct.full((4,), 1, dtype=ct.int32), ct.full((8,), 2, dtype=ct.int32), ct.full((16,), 3, dtype=ct.int32))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in range(3):
        t = tiles[i]
        total = total + ct.sum(t, 0) * x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_plain_for_heterogeneous, sig)
    print(f"plain for-loop indexing heterogeneous-shape tuple: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"plain for-loop indexing heterogeneous-shape tuple: {type(e).__name__}: {e}")

# static_iter's own validation: break and return are documented as
# disallowed inside its loop body -- but so are they inside an ordinary
# for-loop in tile code, as the second probe below confirms, so this is
# a general for-loop restriction, not something specific to static_iter.
@ct.kernel
def kernel_static_iter_break(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(4)):
        if i == 2:
            break
        total = total + x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_static_iter_break, sig)
    print(f"static_iter with break: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_iter with break: {type(e).__name__}: {e}")

@ct.kernel
def kernel_plain_for_break(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in range(4):
        if i == 2:
            break
        total = total + x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_plain_for_break, sig)
    print(f"plain for-loop with break: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"plain for-loop with break: {type(e).__name__}: {e}")

# static_iter also caps the iterable's length at a documented 1000
# iterations. Exceeding it raises TileStaticEvalError -- the same type
# Section 25.2 triggered from a completely different misuse of
# static_eval, since static_iter shares its underlying evaluation
# machinery.
@ct.kernel
def kernel_static_iter_too_long(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(2000)):
        total = total + x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_static_iter_too_long, sig)
    print(f"static_iter with 2000 iterations: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_iter with 2000 iterations: {type(e).__name__}: {e}")
```

Genuinely run:

```
static_iter basic unroll: 20784 cubin bytes
plain 'for i in range(4)' (no static_iter): 20512 cubin bytes
static_iter indexing heterogeneous-shape tuple: 21680 cubin bytes
plain for-loop indexing heterogeneous-shape tuple: TileTypeError: Tuple indices must be constant
  "/tmp/ch25/03_static_iter_vs_plain_for_loop.py", line 71, col 13-20, in kernel_plain_for_heterogeneous:
            t = tiles[i]
                ^^^^^^^^

static_iter with break: TileSyntaxError: Break in a for loop is not supported
  "/tmp/ch25/03_static_iter_vs_plain_for_loop.py", line 90, col 13-17, in kernel_static_iter_break:
                break
                ^^^^^

plain for-loop with break: TileSyntaxError: Break in a for loop is not supported
  "/tmp/ch25/03_static_iter_vs_plain_for_loop.py", line 105, col 13-17, in kernel_plain_for_break:
                break
                ^^^^^

static_iter with 2000 iterations: TileStaticEvalError: Maximum number of iterations (1000) has been reached while unpacking the static_iter() iterable
  "/tmp/ch25/03_static_iter_vs_plain_for_loop.py", line 123, col 29-39, in kernel_static_iter_too_long:
        for i in ct.static_iter(range(2000)):
                                ^^^^^^^^^^^
```

### Discussion

The heterogeneous-tuple test settles the question cleanly: a plain `for` loop's induction variable is rejected outright with "Tuple indices must be constant" the moment it is used to index a tuple of differently-shaped tiles, while the identical index expression under `ct.static_iter` compiles without incident. This confirms `static_iter` is not merely a hint that helps the compiler unroll a loop it could have handled some other way — it is the only mechanism in this book's toolkit that produces an index genuinely usable where a true compile-time constant is required, exactly as its docstring claims and a plain `range()`-based loop, despite also having a compile-time-known bound, cannot. The `break` and `return` restrictions, by contrast, turn out to be general properties of `for` loops in tile code — the identical error message appears whether or not `static_iter` is involved — so this book should not have assumed, from the docstring's own wording, that those restrictions were specific to `static_iter`'s own unrolling mechanism. The 1000-iteration cap raises `TileStaticEvalError`, the same type Section 25.2 triggered from an unrelated misuse (a runtime operation inside `static_eval`), which makes sense once `static_iter`'s own docstring is read closely: it says its contents are evaluated "using the same rules as `static_eval`," so the two functions evidently share a validation and evaluation layer, and an error native to one can surface from the other.

## 25.4 ct.function and TileRecursionError: How Deep Inlining Can Go

### Intuition

Every helper function this book has written and passed into tile code — the combining functions of Chapter 24's `reduce`/`scan`, the tie-break function of its capstone — has been an ordinary, undecorated Python function. `ct.function` exists to make a function's callable execution spaces explicit, but its docstring says an unannotated function called from tile code has tile added to its own execution space automatically, "recursive[ly]," with no explicit annotation required — which is presumably why none of this book's earlier helper functions ever needed the decorator. Tile functions are inlined at every call site, though, and inlining is not free: a function that calls itself has to bottom out somewhere, or inlining could never finish.

### Background

A tile function that recurses to a compile-time-bounded depth (terminating via a `ct.Constant[int]` parameter that counts down to zero) establishes that bounded recursion inlines to a normal, finite result. A second tile function that calls itself unconditionally, with no base case at all, tests what happens when inlining has no natural place to stop. Because that second kernel's resulting exception embeds one traceback frame per attempted inlining step — enough to make a verbatim embedding impractical — this probe reports only the exception's own message, plus a genuine count of how many repeated frames its text actually contains.

### Worked Example 25.4.1 — bounded recursion succeeds, unbounded recursion has a hard limit

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.function marks a callable as usable from tile code. This book has
# used ordinary (undecorated) helper functions throughout; @ct.function
# is required when a function needs to be explicitly callable from
# other execution spaces, but plain tile-only helpers work either way.
# Tile functions are inlined at every call site -- a bounded recursive
# one, terminating via a compile-time constant depth, inlines to a
# fixed, finite call chain without incident.
@ct.function
def recursive_fn(x, depth: ct.Constant[int]):
    if depth <= 0:
        return x
    return recursive_fn(x + 1, depth - 1)

@ct.kernel
def kernel_recursive(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = recursive_fn(x, 3)
    ct.store(c, (0,), y)
print(f"bounded recursive tile function (depth=3): {compile_bytes(kernel_recursive, sig)} cubin bytes")

# A tile function that calls itself with no base case at all: inlining
# it can never terminate on its own, so the compiler must impose some
# maximum inlining depth of its own. This raises TileRecursionError,
# never triggered before in this book. Its traceback repeats one frame
# per inlining attempt -- almost 1000 lines long -- so rather than
# embed it verbatim, this probe reports only the exception's own
# message and a genuine count of how many repeated frames it contains.
@ct.function
def infinite_recursive_fn(x):
    return infinite_recursive_fn(x + 1)

@ct.kernel
def kernel_infinite_recursive(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = infinite_recursive_fn(x)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_infinite_recursive, sig)
    print(f"unbounded recursive tile function: {n} cubin bytes (unexpected)")
except Exception as e:
    tb_text = str(e)
    lines = tb_text.splitlines()
    frame_count = tb_text.count("in infinite_recursive_fn")
    print(f"unbounded recursive tile function: {type(e).__name__}: {lines[0]}")
    print(f"traceback frame count for infinite_recursive_fn: {frame_count}")
    print(f"total lines in exception text: {len(lines)}")
```

Genuinely run:

```
bounded recursive tile function (depth=3): 20960 cubin bytes
unbounded recursive tile function: TileRecursionError: Maximum recursion depth (1000) reached while inlining a function call
traceback frame count for infinite_recursive_fn: 999
total lines in exception text: 3001
```

### Discussion

`TileRecursionError` is confirmed, and its message states the limit outright: 1000. That is the identical number `static_iter`'s own iteration cap uses (Section 25.3), which is a striking enough coincidence to name plainly without overclaiming a shared cause — this book's `export_kernel`-only environment has no way to inspect whether inlining depth and unroll-iteration count are implemented as literally the same internal counter, or whether the compiler's authors simply chose the same round number twice for two unrelated limits. The bounded-recursion case is a quiet but genuine confirmation of something this book has relied on implicitly since Chapter 2: every helper function passed to a tile operation gets inlined wherever it is called, so a "function call" in tile code is a compile-time textual substitution, not a runtime call with its own stack frame — which is exactly why an unbounded recursive definition cannot simply run out of stack at execution time the way it would in ordinary Python, and instead has to be caught during compilation, by counting inlining attempts directly.

## 25.5 TileValueError, ct.compiler_timeout, and a Confound in the Compiler Itself

### Intuition

Three more never-triggered exception types turn out to come from three genuinely different sources: an ordinary piece of Python syntax colliding with a dynamically-sized value, a deliberately tiny compiler timeout, and a compiler target string this book's own front-end cannot validate in advance. The middle one very nearly did not reproduce at all — not because `ct.compiler_timeout` doesn't work as documented, but because of something this environment does that has nothing to do with `compiler_timeout` specifically.

### Background

Chapter 24's `ct.reduce` tuple form returns a genuine Python tuple of two dynamic values. Unpacking that 2-tuple into three names, and separately into one name, tests whether ordinary Python tuple-unpacking syntax inside tile code is validated against the tuple's actual length. Separately, `ct.compiler_timeout(timeout_sec)` is a context manager whose own docstring pairs it with `ct.launch()` — a call this book's driver-free environment can never make — but `export_kernel` goes through the identical compiler pipeline, so an absurdly small timeout is tested there instead. Finally, an architecture string of the documented `"sm_<major><minor>"` shape, but one the installed compiler binary does not actually support, tests what happens when a well-formed request reaches the real compiler rather than being caught by this library's own argument validation.

### Worked Example 25.5.1 — a tuple-unpacking mismatch, a timeout, and the compiler's own voice

```python
# ct.compiler_timeout is demonstrated below by deliberately setting an
# absurdly small timeout. Making that demonstration reproducible turned
# up a genuine wrinkle: cuTile Python keeps a persistent, on-disk cache
# of previously-compiled cubins (an sqlite database, normally under
# ~/.cache/cutile-python), keyed by the compiled bytecode content and
# target architecture -- NOT by the timeout that was in effect. A cache
# HIT returns the stored cubin without ever invoking the compiler
# subprocess at all, which means it also never runs the timeout check.
# Once this exact trivial kernel had been compiled successfully for
# "sm_90" anywhere in this environment's history, a later attempt to
# recompile it under compiler_timeout(1e-9) would silently succeed from
# the cache instead of timing out -- a source of non-reproducibility
# this chapter's own verification discipline caught directly. The fix,
# read from the same source that documents the cache itself, is the
# CUDA_TILE_CACHE_DIR environment variable, which must be set before
# cuda.tile is imported.
import os
os.environ["CUDA_TILE_CACHE_DIR"] = "off"

import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code="sm_90"):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def sum_and_product(s1, p1, s2, p2):
    return s1 + s2, p1 * p2

# Chapter 24's ct.reduce tuple form returns a genuine 2-tuple. Ordinary
# Python tuple-unpacking that 2-tuple into the wrong number of names is
# a mismatch the compiler can only catch from the tuple's own known
# length -- not from any dtype or shape rule this book has met before.
# This raises TileValueError, never triggered before in this book.
@ct.kernel
def kernel_unpack_too_many_names(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total, product, extra = ct.reduce((x, x), 0, sum_and_product, (0, 1))
    ct.store(c, (0,), ct.full((1,), total, dtype=ct.int32))
try:
    n = compile_bytes(kernel_unpack_too_many_names, sig)
    print(f"unpack 2-tuple into 3 names: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack 2-tuple into 3 names: {type(e).__name__}: {e}")

@ct.kernel
def kernel_unpack_too_few_names(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    (total,) = ct.reduce((x, x), 0, sum_and_product, (0, 1))
    ct.store(c, (0,), ct.full((1,), total, dtype=ct.int32))
try:
    n = compile_bytes(kernel_unpack_too_few_names, sig)
    print(f"unpack 2-tuple into 1 name: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack 2-tuple into 1 name: {type(e).__name__}: {e}")

# ct.compiler_timeout(timeout_sec) is a context manager. Its own
# docstring pairs it with ct.launch(), which this book's export_kernel-
# only, driver-free environment can never call -- but export_kernel
# goes through the identical compiler pipeline, so an absurdly small
# timeout applies there too, deterministically triggering
# TileCompilerTimeoutError, now that the disk cache is disabled.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

with ct.compiler_timeout(1e-9):
    try:
        n = compile_bytes(kernel_fn, sig)
        print(f"compile under compiler_timeout(1e-9): {n} cubin bytes (unexpected, no timeout)")
    except Exception as e:
        timeout_error = e
        print(f"compile under compiler_timeout(1e-9): {type(e).__name__}: {e}")

# The context manager restores the previous timeout on exit: the
# identical kernel compiles normally right afterward.
n_after = compile_bytes(kernel_fn, sig)
print(f"compile with default timeout (after context exit): {n_after} cubin bytes")

# gpu_code accepts any string of the documented "sm_<major><minor>"
# shape, but this book's own front-end validation only checks that
# SHAPE -- it cannot know in advance which architectures the installed
# `tileiras` binary actually supports. An unsupported-but-well-formed
# architecture string is passed straight through to the real compiler
# binary, which rejects it on its own terms: TileCompilerExecutionError,
# carrying the backend compiler's own stderr text.
try:
    n = compile_bytes(kernel_fn, sig, gpu_code="sm_61")
    print(f"gpu_code='sm_61': {n} cubin bytes (unexpected)")
except Exception as e:
    execution_error = e
    print(f"gpu_code='sm_61': {type(e).__name__}: {e}")

# TileCompilerExecutionError and TileCompilerTimeoutError both inherit
# from TileInternalError (confirmed directly from the exception
# classes' own __mro__, not assumed) -- so this book has, in a strict
# isinstance sense, already triggered TileInternalError twice over,
# even though no code path in this environment raises a *bare*
# TileInternalError of its own (those guard internal invariants this
# book's own kernels never manage to violate).
print(f"timeout error isinstance TileInternalError: {isinstance(timeout_error, ct.TileInternalError)}")
print(f"execution error isinstance TileInternalError: {isinstance(execution_error, ct.TileInternalError)}")
```

Genuinely run:

```
unpack 2-tuple into 3 names: TileValueError: Too few values to unpack (expected 3, got 2)
  "/tmp/ch25/05_value_error_timeout_and_execution_error.py", line 50, col 5-25, in kernel_unpack_too_many_names:
        total, product, extra = ct.reduce((x, x), 0, sum_and_product, (0, 1))
        ^^^^^^^^^^^^^^^^^^^^^

unpack 2-tuple into 1 name: TileValueError: Too many values to unpack (expected 1, got 2)
  "/tmp/ch25/05_value_error_timeout_and_execution_error.py", line 61, col 5-12, in kernel_unpack_too_few_names:
        (total,) = ct.reduce((x, x), 0, sum_and_product, (0, 1))
        ^^^^^^^^

compile under compiler_timeout(1e-9): TileCompilerTimeoutError: `tileiras` compiler exceeded timeout 1e-09s. Using a smaller tile size may reduce compilation time.
Unknown location
compile with default timeout (after context exit): 20256 cubin bytes
gpu_code='sm_61': TileCompilerExecutionError: Return code 1
tileiras: for the --gpu-name option: Cannot find option named 'sm_61'!
Unknown location
timeout error isinstance TileInternalError: True
execution error isinstance TileInternalError: True
```

### Discussion

Every claim in this section's setup is now confirmed directly. `TileValueError`'s two messages are refreshingly literal — "Too few values to unpack (expected 3, got 2)" and "Too many values to unpack (expected 1, got 2)" — reporting both the number of names on the left and the tuple's actual length, exactly the information a `ValueError` from ordinary Python tuple-unpacking would report. `TileCompilerTimeoutError`'s message names the offending compiler binary by name (`tileiras`) and even offers practical advice ("Using a smaller tile size may reduce compilation time"), but reaching it reproducibly required discovering and then disabling a persistent, on-disk compiler cache — a genuinely new category of confound for this book. Chapter 10, Chapter 16, and Chapter 24 each found a way a byte count could shift for reasons unrelated to a kernel's own logic; this one is different in kind, since it does not change what gets compiled, but whether the compiler runs at all. A byte-count confound can be controlled for by holding names, file paths, or source lines constant; a disk-cache confound has to be controlled for by clearing or disabling state that outlives the Python process entirely, which is why this section's own worked example sets `CUDA_TILE_CACHE_DIR=off` before `cuda.tile` is even imported, rather than trying to guess what a cache initialized at import time will do. `TileCompilerExecutionError`'s text is, verbatim, output from the real `tileiras` binary rejecting an architecture string it does not recognize — the first time this book has embedded a compiler error that did not originate from cuTile Python's own Python-level code at all. And the closing `isinstance` check resolves the very last loose end from Chapter 24's tally: `TileInternalError` is not, strictly, one of this book's never-triggered exceptions after all — this environment has triggered two genuine instances of it, by way of its two named subclasses. What remains untriggered is only the *bare* form — the handful of internal-invariant guards this book's source-reading turned up (an "unexpected dtype," a "variable not found," an "unexpected memory effect") that appear to exist for defending against bugs in the compiler's own passes, not for any misuse a caller of the public API could construct on purpose.

## 25.6 Capstone: A Compile-Time-Validated, Unrolled Polynomial Evaluator

### Intuition

This chapter's three metaprogramming tools compose naturally: `ct.static_assert` can validate an invariant before any real work is unrolled, `ct.static_iter` can unroll that work into concrete, differently-indexed copies, and `ct.static_eval` can compute the compile-time index each copy needs. Evaluating a polynomial by Horner's method — repeatedly multiplying by `x` and adding the next coefficient — is a natural fit: the coefficients are known at compile time, the number of steps is fixed by the polynomial's degree, and each step needs a different, concrete coefficient.

### Background

A module-level tuple `COEFFS = (5, 3, 2, 1)` represents the cubic `5 + 3x + 2x^2 + x^3`. The kernel takes `degree` as a `ct.Constant[int]` parameter, uses `ct.static_assert` to check that `degree` actually matches `len(COEFFS) - 1` before doing anything else, then uses `ct.static_iter` to unroll `degree` multiply-and-add steps, with `ct.static_eval` computing each step's coefficient index from the two compile-time values `degree` and the loop's own `i`.

### Worked Example 25.6.1 — Horner's method, unrolled and validated at compile time

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

def sig_with_degree(degree):
    return ct.compilation.KernelSignature(
        [array_param(1), array_param(1), ct.compilation.ConstantConstraint(degree)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# Capstone: evaluate the cubic 5 + 3x + 2x^2 + x^3 via a compile-time-
# unrolled Horner's method, combining all three of this chapter's
# metaprogramming tools in one kernel. ct.static_assert checks that the
# caller-supplied "degree" constant actually matches the coefficient
# table below, before any of the polynomial arithmetic is built.
# ct.static_iter unrolls the multiply-and-add loop into `degree`
# concrete copies. ct.static_eval computes each copy's coefficient
# index from two compile-time values (the constant "degree" and the
# loop's own induction variable).
COEFFS = (5, 3, 2, 1)

@ct.kernel
def kernel_poly_horner(a, c, degree: ct.Constant[int]):
    ct.static_assert(degree == len(COEFFS) - 1,
                      f"degree {degree} does not match len(COEFFS)-1={len(COEFFS) - 1}")
    x = ct.load(a, (0,), (8,))
    result = ct.full((8,), COEFFS[degree], dtype=ct.int32)
    for i in ct.static_iter(range(degree)):
        idx = ct.static_eval(degree - 1 - i)
        result = result * x + COEFFS[idx]
    ct.store(c, (0,), result)
n = compile_bytes(kernel_poly_horner, sig_with_degree(3))
print(f"Horner-unrolled cubic (degree=3, static_assert+static_iter+static_eval): {n} cubin bytes")

# Calling the identical kernel with a "degree" constant that does not
# match COEFFS's own length trips the static_assert before a single
# multiply-and-add is ever unrolled.
try:
    n = compile_bytes(kernel_poly_horner, sig_with_degree(2))
    print(f"Horner with mismatched degree=2: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"Horner with mismatched degree=2: {type(e).__name__}: {e}")
```

Genuinely run:

```
Horner-unrolled cubic (degree=3, static_assert+static_iter+static_eval): 20640 cubin bytes
Horner with mismatched degree=2: TileStaticAssertionError: Static assertion failed: degree 2 does not match len(COEFFS)-1=3
  "/tmp/ch25/06_capstone_compile_time_horner_evaluation.py", lines 34--35, col 5-47, in kernel_poly_horner:
        ct.static_assert(degree == len(COEFFS) - 1,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Discussion

Both kernels compile — or fail to — exactly as designed. The `degree=3` kernel unrolls three multiply-and-add steps against the four-element coefficient table, confirming that `static_assert`, `static_iter`, and `static_eval` compose without friction inside a single kernel body: the assertion's own condition reads `len(COEFFS)`, a plain Python `len()` call on a plain Python tuple, evaluated at the same compile-time level as everything else in this chapter. The `degree=2` failure is caught at the very first line of the kernel body, before the loop, the index computation, or a single multiply-and-add is ever built — precisely the ordering `static_assert`'s own docstring promises, and precisely the value of putting a compile-time invariant check first: a mismatched constant is reported as a validation failure with a specific, actionable message, rather than surfacing later as a more confusing shape or index error once the unrolled loop actually runs into trouble. The traceback's own location line is worth a second look: "lines 34--35" rather than a single line number, the first time this book has seen a double-dashed line range in a traceback header — a direct, visible consequence of `static_assert`'s call in this kernel being written across two source lines, and the compiler's location-tracking machinery reporting the whole span rather than collapsing it to just where the call starts.

## Chapter Summary

This chapter opened `dir(ct)`'s compile-time metaprogramming tools — `ct.static_assert`, `ct.static_eval`, and `ct.static_iter` — and, in the process, triggered seven of the eight Tile exception types Chapter 24 named as never-seen in twenty-four chapters. `static_assert` checks a compile-time boolean and raises `TileStaticAssertionError` on failure, evaluating its optional message lazily and substituting an empty string when none is given. `static_eval` reads compile-time state with ordinary Python expression syntax, but forbids two things: performing a runtime operation on a dynamic value (`TileStaticEvalError`) and assigning to a local variable via the walrus operator (`TileSyntaxError`). `static_iter` was shown to do something a plain `for` loop over a compile-time-known `range()` genuinely cannot: bind its induction variable to a true compile-time constant at each unrolled copy, demonstrated by indexing a heterogeneous-shape tuple that a plain loop rejects outright — while sharing a 1000-iteration cap and a `break`/`return` restriction with ordinary tile-code loops. A bounded recursive tile function inlines cleanly; an unbounded one hits a hard, documented 1000-frame inlining limit, raising `TileRecursionError`. `TileValueError` surfaced from nothing more exotic than Python's own tuple-unpacking syntax meeting a tuple of the wrong length. `ct.compiler_timeout` genuinely triggers `TileCompilerTimeoutError` through `export_kernel`, once a previously-undiscovered persistent on-disk compiler cache is disabled — a new kind of confound distinct from anything found in Chapters 10, 16, or 24, since it affects whether the compiler runs at all rather than what it produces. An architecture string this book's own front-end cannot pre-validate surfaced `TileCompilerExecutionError`, carrying the real compiler binary's own stderr text. And a direct `isinstance` check against both compiler-related exceptions' `__mro__` confirmed that `TileInternalError` — the eighth and last name on Chapter 24's list — has, in fact, already been triggered twice over, closing out every concrete Tile exception type this book has ever named except the bare internal-invariant guards that appear reserved for the compiler's own bugs.

## Self-Check Questions

1. Section 25.2 found that `TileSyntaxError` (from the walrus-operator misuse) is reported with no source location at all, while `TileStaticEvalError` (from the runtime-operation misuse) is reported with a full, specific location. What does this difference suggest about which of the two checks happens earlier in the compiler's own pipeline, before location information has necessarily been attached to what it's examining?
2. Section 25.3 found that `break` and `return` are rejected identically whether or not `ct.static_iter` is involved, while indexing a heterogeneous-shape tuple by the loop variable is only accepted under `static_iter`. Why might a restriction like "no `break`" apply to for-loops in tile code generally, while "the induction variable is a true compile-time constant" is something only `static_iter` specifically provides?
3. Section 25.4 noted that `static_iter`'s 1000-iteration cap and `TileRecursionError`'s 1000-frame inlining cap use the identical number, and explicitly declined to conclude they share an implementation. What kind of evidence, unavailable to this book's `export_kernel`-only environment, would be needed to settle whether they are actually the same underlying counter?
4. Section 25.5 discovered that a persistent on-disk compiler cache could make `ct.compiler_timeout(1e-9)` silently succeed instead of raising `TileCompilerTimeoutError`, and fixed this by setting `CUDA_TILE_CACHE_DIR=off` before importing `cuda.tile`. Why does the environment variable need to be set before the import specifically, rather than at any point before the `compiler_timeout` context manager is entered?
5. Section 25.5's closing `isinstance` check showed that `TileCompilerTimeoutError` and `TileCompilerExecutionError` both satisfy `isinstance(e, TileInternalError)`. If this book's code had instead written `except TileInternalError:` around every compilation call across all twenty-five chapters, what would have changed about which of this book's own worked examples' error-handling blocks caught their exceptions, and which would not have?

## Where We Go Next

Between Chapter 1's `TileTypeError`, Chapter 21's `TileUnsupportedFeatureError`, and this chapter's seven new arrivals, every concrete Tile exception type this book has ever named has now been triggered at least once — a genuine milestone twenty-five chapters in the making. What remains in `dir(ct)` is a smaller, more specific residue: `ct.gather` and `ct.scatter`, a general index-tile-based memory access mechanism that looks related to Chapter 23's `load_advanced_indexing`/`store_advanced_indexing` but is built differently — every axis addressed by its own runtime integer tile, rather than Chapter 23's one-sparse-dimension-plus-compile-time-slices rule. The `ct.tune` module, glimpsed only by name so far, points toward autotuning: letting the compiler search over implementation choices rather than committing to one. And `ct.function(host=True)`, briefly read in this chapter's docstring research but never actually exercised, marks a boundary this book has not yet crossed at all: every kernel so far has run entirely on one side of the host/device line.

## Worked Solutions

**1.** The difference points to the two checks living at different stages of compilation. `TileStaticEvalError`'s message ("add: Tile functions cannot be called inside static_eval()") reads like a rejection that happens once the expression has already been translated into calls against real tile operations — `x + 1` has become a call to the same "add" implementation every other addition in this book goes through, and that implementation clearly carries the location metadata every other error in this book has shown. A walrus assignment, by contrast, is a purely syntactic property of the expression — the compiler can see "this expression contains an assignment target" by inspecting its shape alone, before deciding what any of its subexpressions mean or attaching them to specific tile operations with their own locations. The absence of a location on `TileSyntaxError` is consistent with a check that happens directly against the raw parsed expression, before whatever pass in the compiler is responsible for tagging things with source positions has had a chance to run — though this book's own `export_kernel`-only view can only observe the difference in the two messages, not the compiler's internal pass ordering directly.

**2.** Rejecting `break` and `return` looks like a property of how tile code translates a `for` loop into whatever control-flow representation the compiler builds internally — regardless of whether the loop's own bound came from `static_iter` or from a plain `range()` call, both loops still get lowered through the same "this is a for-loop" translation step, and that step evidently doesn't support early exit at all, for either kind of loop body. Making the induction variable a genuine compile-time constant, though, is not a property of loop *translation* — it's a property of how many separate copies of the body get built in the first place. `static_iter` doesn't compile one generic loop body and reason about `i` symbolically; it appears to produce a distinct, concrete copy of the body per item, with `i` replaced by that item's actual value in each copy. That is a fundamentally different thing from "translate a for-loop," which is presumably why only `static_iter` gets it: it's not a special case of the general for-loop rule, it's a different mechanism (unrolling) layered on top of one.

**3.** Settling this would require evidence this book's `export_kernel`-only, driver-free method cannot produce on its own: either a way to observe the compiler's internal state directly (for instance, seeing whether raising `static_iter`'s cap via some undocumented setting also raises `TileRecursionError`'s cap, which would suggest a shared constant, or seeing that it doesn't, which would suggest two independent hardcoded limits), or reading the compiler's own source closely enough to find whether both limits are defined by reference to the same named constant versus two separate literal `1000`s written in two different places. This book has, elsewhere, read installed package source to understand *behavior* worth testing (as it did to find `TileCompilerExecutionError`'s and `ct.compiler_timeout`'s disk-cache interaction this very chapter) — but confirming a shared-constant hypothesis specifically would mean reading implementation details this book has deliberately avoided reproducing or leaning on as its primary evidence, precisely because the whole point of this book's method has been to trust what a real compile run confirms over what source code merely suggests.

**4.** `cuda.tile`'s default compilation context is built from a compiled extension module, and Section 25.5's own research found that its cache directory setting is read once, by a function that runs when that context is first constructed — which happens as a side effect of importing `cuda.tile` itself, not lazily on each compile call. Once that one-time read has happened, the resulting configuration value is what every subsequent compile call consults, regardless of what the environment variable says afterward; changing the environment variable after import doesn't retroactively update a value that was already copied out of the environment and stored. This is exactly why the chapter's own worked example sets the environment variable as the very first lines of the file, before the `import cuda.tile as ct` line — setting it any later, even one line later, would be too late for this specific setting to take effect, even though it would still be perfectly valid Python to write it that way.

**5.** Almost none of this book's error-handling blocks would have caught anything. Across twenty-five chapters, this book has deliberately triggered `TileTypeError` (starting in Chapter 1), `TileUnsupportedFeatureError` (Chapter 21), and, in this chapter alone, `TileValueError`, `TileStaticEvalError`, `TileStaticAssertionError`, `TileSyntaxError`, and `TileRecursionError` — and none of those seven types inherit from `TileInternalError`, per the exception hierarchy this chapter confirmed directly from each class's own `__mro__`. An `except TileInternalError:` block would have let all of them propagate as uncaught exceptions, crashing the very scripts meant to demonstrate them. The only two exceptions in this entire book that such a handler would have caught are this chapter's own `TileCompilerTimeoutError` and `TileCompilerExecutionError` — a narrow, specific pair, not a stand-in for "any Tile-related failure." That broader catch-all role belongs to `TileError`, the common base every one of these types (including `TileInternalError` itself) ultimately inherits from, which is presumably why this book's own probes have used a bare `except Exception:` throughout rather than reaching for any more specific base class.

## Complete Runnable Code

### File: `01_static_assert_and_triggering_error.py`

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.static_assert(condition, message=None) asserts a compile-time
# boolean. When condition is True, compilation continues and message is
# never even evaluated.
@ct.kernel
def kernel_assert_true(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.static_assert(x.shape[0] == 8, "expected shape 8")
    ct.store(c, (0,), x)
print(f"static_assert(True): {compile_bytes(kernel_assert_true, sig)} cubin bytes")

# When condition is False, message is evaluated with static_eval
# semantics -- including f-string interpolation of compile-time values
# -- and raises TileStaticAssertionError, a type this book has never
# triggered before.
@ct.kernel
def kernel_assert_false(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.static_assert(x.shape[0] == 16, f"expected shape 16, got {x.shape[0]}")
    ct.store(c, (0,), x)
try:
    n = compile_bytes(kernel_assert_false, sig)
    print(f"static_assert(False, message): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_assert(False, message): {type(e).__name__}: {e}")

# message is optional; the docstring says a missing message is replaced
# with an empty string rather than, say, "None".
@ct.kernel
def kernel_assert_no_message(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.static_assert(x.shape[0] == 16)
    ct.store(c, (0,), x)
try:
    n = compile_bytes(kernel_assert_no_message, sig)
    print(f"static_assert(False, no message): {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_assert(False, no message): {type(e).__name__}: {e}")
```

### File: `02_static_eval_and_two_new_errors.py`

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

N = 8

# ct.static_eval(expr) evaluates expr with ordinary Python semantics at
# compile time. It can read a tile's compile-time shape, do ordinary
# arithmetic on module-level constants, and select between two
# DIFFERENT dynamic tiles based on a compile-time condition.
@ct.kernel
def kernel_eval_basic(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    shape0 = ct.static_eval(x.shape[0])
    doubled = ct.static_eval(N * 2)
    y = ct.static_eval(x if N % 2 == 0 else x + 1)
    ct.store(c, (0,), y)
print(f"static_eval basic (shape/const/select): {compile_bytes(kernel_eval_basic, sig)} cubin bytes")

# static_eval's docstring says the expression must not perform any
# RUNTIME operation -- x + 1, where x is a genuinely dynamic tile, is
# exactly that. This raises TileStaticEvalError, never triggered before
# in this book.
@ct.kernel
def kernel_eval_runtime_op(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = ct.static_eval(x + 1)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_eval_runtime_op, sig)
    print(f"static_eval(x + 1) on dynamic tile: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_eval(x + 1) on dynamic tile: {type(e).__name__}: {e}")

# static_eval's docstring also says the expression must not assign to
# local variables via the walrus operator. This is a different failure
# than the one above -- a genuinely new exception type, TileSyntaxError,
# also never triggered before in this book, and reported with no
# traceback location at all.
@ct.kernel
def kernel_eval_walrus(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = ct.static_eval((z := x.shape[0]) + 1)
    ct.store(c, (0,), ct.full((1,), y, dtype=ct.int32))
try:
    n = compile_bytes(kernel_eval_walrus, sig)
    print(f"static_eval with walrus assignment: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_eval with walrus assignment: {type(e).__name__}: {e}")
```

### File: `03_static_iter_vs_plain_for_loop.py`

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# for ... in ct.static_iter(range(4)) unrolls the loop body once per
# item, with the induction variable bound to a genuine compile-time
# constant at each copy.
@ct.kernel
def kernel_static_iter_basic(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(4)):
        total = total + x * i
    ct.store(c, (0,), total)
print(f"static_iter basic unroll: {compile_bytes(kernel_static_iter_basic, sig)} cubin bytes")

# A plain Python for-loop over range() -- with no ct.static_iter wrapper
# at all -- also compiles without error: range()'s bound is itself a
# compile-time constant, so the loop is still expressible.
@ct.kernel
def kernel_plain_for(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in range(4):
        total = total + x
    ct.store(c, (0,), total)
print(f"plain 'for i in range(4)' (no static_iter): {compile_bytes(kernel_plain_for, sig)} cubin bytes")

# The real difference shows up when the induction variable is used to
# index something whose elements have genuinely DIFFERENT types --
# here, a Python tuple of tiles with three different shapes. static_iter
# compiles a separate, concrete copy of the loop body per item, so `i`
# is a real compile-time constant inside each copy and can select a
# specific tuple element.
@ct.kernel
def kernel_static_iter_heterogeneous(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    tiles = (ct.full((4,), 1, dtype=ct.int32), ct.full((8,), 2, dtype=ct.int32), ct.full((16,), 3, dtype=ct.int32))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(3)):
        t = tiles[i]
        total = total + ct.sum(t, 0) * x
    ct.store(c, (0,), total)
print(f"static_iter indexing heterogeneous-shape tuple: {compile_bytes(kernel_static_iter_heterogeneous, sig)} cubin bytes")

# A plain for-loop's induction variable is NOT treated as a compile-time
# constant the same way, even though range(3) is itself compile-time
# known: the loop body is compiled once, generically, so indexing the
# identical heterogeneous tuple by `i` is rejected.
@ct.kernel
def kernel_plain_for_heterogeneous(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    tiles = (ct.full((4,), 1, dtype=ct.int32), ct.full((8,), 2, dtype=ct.int32), ct.full((16,), 3, dtype=ct.int32))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in range(3):
        t = tiles[i]
        total = total + ct.sum(t, 0) * x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_plain_for_heterogeneous, sig)
    print(f"plain for-loop indexing heterogeneous-shape tuple: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"plain for-loop indexing heterogeneous-shape tuple: {type(e).__name__}: {e}")

# static_iter's own validation: break and return are documented as
# disallowed inside its loop body -- but so are they inside an ordinary
# for-loop in tile code, as the second probe below confirms, so this is
# a general for-loop restriction, not something specific to static_iter.
@ct.kernel
def kernel_static_iter_break(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(4)):
        if i == 2:
            break
        total = total + x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_static_iter_break, sig)
    print(f"static_iter with break: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_iter with break: {type(e).__name__}: {e}")

@ct.kernel
def kernel_plain_for_break(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in range(4):
        if i == 2:
            break
        total = total + x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_plain_for_break, sig)
    print(f"plain for-loop with break: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"plain for-loop with break: {type(e).__name__}: {e}")

# static_iter also caps the iterable's length at a documented 1000
# iterations. Exceeding it raises TileStaticEvalError -- the same type
# Section 25.2 triggered from a completely different misuse of
# static_eval, since static_iter shares its underlying evaluation
# machinery.
@ct.kernel
def kernel_static_iter_too_long(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total = ct.zeros((8,), dtype=ct.int32)
    for i in ct.static_iter(range(2000)):
        total = total + x
    ct.store(c, (0,), total)
try:
    n = compile_bytes(kernel_static_iter_too_long, sig)
    print(f"static_iter with 2000 iterations: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"static_iter with 2000 iterations: {type(e).__name__}: {e}")
```

### File: `04_function_and_recursion_error.py`

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

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# ct.function marks a callable as usable from tile code. This book has
# used ordinary (undecorated) helper functions throughout; @ct.function
# is required when a function needs to be explicitly callable from
# other execution spaces, but plain tile-only helpers work either way.
# Tile functions are inlined at every call site -- a bounded recursive
# one, terminating via a compile-time constant depth, inlines to a
# fixed, finite call chain without incident.
@ct.function
def recursive_fn(x, depth: ct.Constant[int]):
    if depth <= 0:
        return x
    return recursive_fn(x + 1, depth - 1)

@ct.kernel
def kernel_recursive(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = recursive_fn(x, 3)
    ct.store(c, (0,), y)
print(f"bounded recursive tile function (depth=3): {compile_bytes(kernel_recursive, sig)} cubin bytes")

# A tile function that calls itself with no base case at all: inlining
# it can never terminate on its own, so the compiler must impose some
# maximum inlining depth of its own. This raises TileRecursionError,
# never triggered before in this book. Its traceback repeats one frame
# per inlining attempt -- almost 1000 lines long -- so rather than
# embed it verbatim, this probe reports only the exception's own
# message and a genuine count of how many repeated frames it contains.
@ct.function
def infinite_recursive_fn(x):
    return infinite_recursive_fn(x + 1)

@ct.kernel
def kernel_infinite_recursive(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    y = infinite_recursive_fn(x)
    ct.store(c, (0,), y)
try:
    n = compile_bytes(kernel_infinite_recursive, sig)
    print(f"unbounded recursive tile function: {n} cubin bytes (unexpected)")
except Exception as e:
    tb_text = str(e)
    lines = tb_text.splitlines()
    frame_count = tb_text.count("in infinite_recursive_fn")
    print(f"unbounded recursive tile function: {type(e).__name__}: {lines[0]}")
    print(f"traceback frame count for infinite_recursive_fn: {frame_count}")
    print(f"total lines in exception text: {len(lines)}")
```

### File: `05_value_error_timeout_and_execution_error.py`

```python
# ct.compiler_timeout is demonstrated below by deliberately setting an
# absurdly small timeout. Making that demonstration reproducible turned
# up a genuine wrinkle: cuTile Python keeps a persistent, on-disk cache
# of previously-compiled cubins (an sqlite database, normally under
# ~/.cache/cutile-python), keyed by the compiled bytecode content and
# target architecture -- NOT by the timeout that was in effect. A cache
# HIT returns the stored cubin without ever invoking the compiler
# subprocess at all, which means it also never runs the timeout check.
# Once this exact trivial kernel had been compiled successfully for
# "sm_90" anywhere in this environment's history, a later attempt to
# recompile it under compiler_timeout(1e-9) would silently succeed from
# the cache instead of timing out -- a source of non-reproducibility
# this chapter's own verification discipline caught directly. The fix,
# read from the same source that documents the cache itself, is the
# CUDA_TILE_CACHE_DIR environment variable, which must be set before
# cuda.tile is imported.
import os
os.environ["CUDA_TILE_CACHE_DIR"] = "off"

import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.int32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def compile_bytes(kernel, sig, gpu_code="sm_90"):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format="cubin")
    return len(buf.getvalue())

sig = ct.compilation.KernelSignature(
    [array_param(1), array_param(1), ct.compilation.ConstantConstraint(64)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

def sum_and_product(s1, p1, s2, p2):
    return s1 + s2, p1 * p2

# Chapter 24's ct.reduce tuple form returns a genuine 2-tuple. Ordinary
# Python tuple-unpacking that 2-tuple into the wrong number of names is
# a mismatch the compiler can only catch from the tuple's own known
# length -- not from any dtype or shape rule this book has met before.
# This raises TileValueError, never triggered before in this book.
@ct.kernel
def kernel_unpack_too_many_names(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    total, product, extra = ct.reduce((x, x), 0, sum_and_product, (0, 1))
    ct.store(c, (0,), ct.full((1,), total, dtype=ct.int32))
try:
    n = compile_bytes(kernel_unpack_too_many_names, sig)
    print(f"unpack 2-tuple into 3 names: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack 2-tuple into 3 names: {type(e).__name__}: {e}")

@ct.kernel
def kernel_unpack_too_few_names(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    (total,) = ct.reduce((x, x), 0, sum_and_product, (0, 1))
    ct.store(c, (0,), ct.full((1,), total, dtype=ct.int32))
try:
    n = compile_bytes(kernel_unpack_too_few_names, sig)
    print(f"unpack 2-tuple into 1 name: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"unpack 2-tuple into 1 name: {type(e).__name__}: {e}")

# ct.compiler_timeout(timeout_sec) is a context manager. Its own
# docstring pairs it with ct.launch(), which this book's export_kernel-
# only, driver-free environment can never call -- but export_kernel
# goes through the identical compiler pipeline, so an absurdly small
# timeout applies there too, deterministically triggering
# TileCompilerTimeoutError, now that the disk cache is disabled.
@ct.kernel
def kernel_fn(a, c, tile_size: ct.Constant[int]):
    x = ct.load(a, (0,), (8,))
    ct.store(c, (0,), x)

with ct.compiler_timeout(1e-9):
    try:
        n = compile_bytes(kernel_fn, sig)
        print(f"compile under compiler_timeout(1e-9): {n} cubin bytes (unexpected, no timeout)")
    except Exception as e:
        timeout_error = e
        print(f"compile under compiler_timeout(1e-9): {type(e).__name__}: {e}")

# The context manager restores the previous timeout on exit: the
# identical kernel compiles normally right afterward.
n_after = compile_bytes(kernel_fn, sig)
print(f"compile with default timeout (after context exit): {n_after} cubin bytes")

# gpu_code accepts any string of the documented "sm_<major><minor>"
# shape, but this book's own front-end validation only checks that
# SHAPE -- it cannot know in advance which architectures the installed
# `tileiras` binary actually supports. An unsupported-but-well-formed
# architecture string is passed straight through to the real compiler
# binary, which rejects it on its own terms: TileCompilerExecutionError,
# carrying the backend compiler's own stderr text.
try:
    n = compile_bytes(kernel_fn, sig, gpu_code="sm_61")
    print(f"gpu_code='sm_61': {n} cubin bytes (unexpected)")
except Exception as e:
    execution_error = e
    print(f"gpu_code='sm_61': {type(e).__name__}: {e}")

# TileCompilerExecutionError and TileCompilerTimeoutError both inherit
# from TileInternalError (confirmed directly from the exception
# classes' own __mro__, not assumed) -- so this book has, in a strict
# isinstance sense, already triggered TileInternalError twice over,
# even though no code path in this environment raises a *bare*
# TileInternalError of its own (those guard internal invariants this
# book's own kernels never manage to violate).
print(f"timeout error isinstance TileInternalError: {isinstance(timeout_error, ct.TileInternalError)}")
print(f"execution error isinstance TileInternalError: {isinstance(execution_error, ct.TileInternalError)}")
```

### File: `06_capstone_compile_time_horner_evaluation.py`

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

def sig_with_degree(degree):
    return ct.compilation.KernelSignature(
        [array_param(1), array_param(1), ct.compilation.ConstantConstraint(degree)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# Capstone: evaluate the cubic 5 + 3x + 2x^2 + x^3 via a compile-time-
# unrolled Horner's method, combining all three of this chapter's
# metaprogramming tools in one kernel. ct.static_assert checks that the
# caller-supplied "degree" constant actually matches the coefficient
# table below, before any of the polynomial arithmetic is built.
# ct.static_iter unrolls the multiply-and-add loop into `degree`
# concrete copies. ct.static_eval computes each copy's coefficient
# index from two compile-time values (the constant "degree" and the
# loop's own induction variable).
COEFFS = (5, 3, 2, 1)

@ct.kernel
def kernel_poly_horner(a, c, degree: ct.Constant[int]):
    ct.static_assert(degree == len(COEFFS) - 1,
                      f"degree {degree} does not match len(COEFFS)-1={len(COEFFS) - 1}")
    x = ct.load(a, (0,), (8,))
    result = ct.full((8,), COEFFS[degree], dtype=ct.int32)
    for i in ct.static_iter(range(degree)):
        idx = ct.static_eval(degree - 1 - i)
        result = result * x + COEFFS[idx]
    ct.store(c, (0,), result)
n = compile_bytes(kernel_poly_horner, sig_with_degree(3))
print(f"Horner-unrolled cubic (degree=3, static_assert+static_iter+static_eval): {n} cubin bytes")

# Calling the identical kernel with a "degree" constant that does not
# match COEFFS's own length trips the static_assert before a single
# multiply-and-add is ever unrolled.
try:
    n = compile_bytes(kernel_poly_horner, sig_with_degree(2))
    print(f"Horner with mismatched degree=2: {n} cubin bytes (unexpected)")
except Exception as e:
    print(f"Horner with mismatched degree=2: {type(e).__name__}: {e}")
```
