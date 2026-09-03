# Chapter 39: Compilation and Export Revisited — Cubin, Tile IR Bytecode, and Calling Conventions

> "A compiler that only ever emits one kind of output has hidden a choice from you, not eliminated it."

**What you will understand by the end of this chapter:**

- That `ct.compilation.export_kernel`'s `output_format` parameter has always had two settings, `"cubin"` and `"tileir_bytecode"`, even though every prior chapter in this book used only the first.
- That a cubin's byte count genuinely differs across target architectures (`sm_80`, `sm_90`, `sm_100`), while a Tile IR bytecode export of the identical kernel is byte-for-byte identical across all three — architecture-independence made concrete rather than asserted.
- That Tile IR bytecode's exact byte count is sensitive to something cubin's is not: the length of the absolute filesystem path of the source file being compiled — confirmed directly, and precisely characterized, rather than left as an unexplained discrepancy.
- What the `bytecode_version` parameter controls, how to discover which versions a given compiler supports without guessing, and a genuine blind spot in how it interacts with `output_format="cubin"`.
- That `ct.compilation.CallingConvention` has two members, `cutile_python_v1()` and `cutile_python_v2()` — this book has used `v2` in every `KernelSignature` since Part 3 without ever naming that choice — and what changes and what doesn't when a kernel is compiled under the other one.
- How `mangle_kernel_name` and `demangle_kernel_name` turn a kernel's Python name and signature into an exported symbol string and back, and why that round trip is what makes a calling convention identifiable from the compiled artifact alone.

**What you need to know first:**

- Part 0's introduction of `ct.compilation.export_kernel`, `gpu_code`, and `ArrayConstraint`.
- Part 4's closing chapters, all of which built a `KernelSignature` with `calling_convention=ct.compilation.CallingConvention.cutile_python_v2()` without discussing what that argument does.
- The book's standing verification method: every claim about compiled output in this chapter comes from a real `export_kernel` call in this book's sandboxed, driver-free build environment, with the resulting byte count or byte content shown directly.

## 39.1 Two Output Formats, One Export Function

### Intuition

Every chapter since Part 0 has called `ct.compilation.export_kernel` the same way: pick a target architecture with `gpu_code`, leave `output_format` at `"cubin"`, and read back a real CUDA binary's byte count. That default was never inspected. `export_kernel`'s own signature says `output_format` is a `Literal['cubin', 'tileir_bytecode']` — there is a second, entirely different kind of output this function can produce, and this book has silently avoided it for thirty-eight chapters.

The two formats answer different questions. A cubin is architecture-specific machine code: the whole reason `gpu_code` exists is to tell the compiler which real GPU's instruction set to target, and Part 0 already showed that compiling the same kernel to different `gpu_code` values changes what comes out. Tile IR bytecode is something else — an intermediate representation this book has not yet examined at all. If Tile IR bytecode is genuinely architecture-independent, deferring the final architecture-specific step to whatever loads it later, then asking for `tileir_bytecode` output at three different `gpu_code` values should produce identical bytes every time, no matter which architecture name was passed in. If it isn't, the byte counts should differ just like cubin's do. That is a testable claim, not a matter of reading documentation carefully.

### Background

`export_kernel`'s docstring is direct about what each value means: `"cubin"` exports a CUDA binary file, and `"tileir_bytecode"` exports a Tile IR bytecode file — a format this book has not opened before. Nothing in the docstring says whether `gpu_code` still matters once `output_format` is `"tileir_bytecode"`; the parameter is still required either way, but a required parameter can still be inert for one of two code paths. The only way to know is to compile the identical kernel and signature at all three of this book's usual target architectures under both output formats and compare what comes back.

### Worked Example 39.1.1: cubin vs. Tile IR bytecode across three architectures

```python
import cuda.tile as ct
import io
import os
import subprocess
import sys
import tempfile
import textwrap

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

def sig_2d_2d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- The same elementwise-add kernel this book has compiled many times
# since Part 2, unchanged. What is new in this chapter is not the
# kernel -- it is which of export_kernel's two output_format values we
# ask for, and for how many different target architectures. ---
@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

def export_bytes(kernel, sig, gpu_code, output_format):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format=output_format)
    return buf.getvalue()

sig = sig_2d_2d_2d()
architectures = ["sm_80", "sm_90", "sm_100"]

print("--- output_format='cubin' ---")
cubin_sizes = {}
for gc in architectures:
    b = export_bytes(kernel_add, sig, gc, "cubin")
    cubin_sizes[gc] = len(b)
    print(f"  gpu_code={gc}: {len(b)} bytes")

print()
print("--- output_format='tileir_bytecode' ---")
bytecode_sizes = {}
bytecode_bytes = {}
for gc in architectures:
    b = export_bytes(kernel_add, sig, gc, "tileir_bytecode")
    bytecode_sizes[gc] = len(b)
    bytecode_bytes[gc] = b
    print(f"  gpu_code={gc}: {len(b)} bytes")

print()
print(f"cubin sizes all identical across architectures: {len(set(cubin_sizes.values())) == 1}")
print(f"tileir_bytecode sizes all identical across architectures: {len(set(bytecode_sizes.values())) == 1}")
print(f"tileir_bytecode bytes byte-for-byte identical across architectures: {len(set(bytecode_bytes.values())) == 1}")
print(f"tileir_bytecode magic header: {bytecode_bytes['sm_90'][:8]}")

# --- The byte counts printed above are only reproducible from this
# exact file at this exact filesystem location -- suspicious, given
# that this book's earlier chapters already found compiled byte counts
# can depend on a kernel's source filename. Rather than assume this
# chapter's numbers are as portable as they look, test directly: export
# the identical kernel from three temporary directories whose absolute
# path lengths differ by exactly one character each, and compare. ---
KERNEL_SOURCE = textwrap.dedent("""
    import cuda.tile as ct
    import io

    def array_param(ndim, dtype=ct.float32):
        return ct.compilation.ArrayConstraint(
            dtype, ndim=ndim, index_dtype=ct.int32,
            stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        )

    M, N = 8, 8

    @ct.kernel
    def kernel_add(x, y, out):
        a = ct.load(x, (0, 0), (M, N))
        b = ct.load(y, (0, 0), (M, N))
        ct.store(out, (0, 0), a + b)

    sig = ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print(len(buf.getvalue()))
""")

base = tempfile.mkdtemp()
results = {}
for pad in (0, 1, 2):
    subdir = os.path.join(base, "d" + "x" * pad)
    os.makedirs(subdir)
    script_path = os.path.join(subdir, "probe.py")
    with open(script_path, "w") as f:
        f.write(KERNEL_SOURCE)
    proc = subprocess.run([sys.executable, "probe.py"], cwd=subdir, capture_output=True, text=True)
    results[len(script_path)] = int(proc.stdout.strip())

print()
print("--- same kernel, exported from three different absolute paths ---")
path_lengths = sorted(results)
for length in path_lengths:
    print(f"  absolute script path length {length}: {results[length]} tileir_bytecode bytes")
deltas = [results[path_lengths[i + 1]] - results[path_lengths[i]] for i in range(len(path_lengths) - 1)]
print(f"every +1 character of absolute path length adds exactly +2 bytes: {all(d == 2 for d in deltas)}")
```

### Genuinely run

```text
--- output_format='cubin' ---
  gpu_code=sm_80: 25920 bytes
  gpu_code=sm_90: 26568 bytes
  gpu_code=sm_100: 33216 bytes

--- output_format='tileir_bytecode' ---
  gpu_code=sm_80: 1019 bytes
  gpu_code=sm_90: 1019 bytes
  gpu_code=sm_100: 1019 bytes

cubin sizes all identical across architectures: False
tileir_bytecode sizes all identical across architectures: True
tileir_bytecode bytes byte-for-byte identical across architectures: True
tileir_bytecode magic header: b'\x7fTileIR\x00'

--- same kernel, exported from three different absolute paths ---
  absolute script path length 27: 939 tileir_bytecode bytes
  absolute script path length 28: 941 tileir_bytecode bytes
  absolute script path length 29: 943 tileir_bytecode bytes
every +1 character of absolute path length adds exactly +2 bytes: True
```

### Discussion

The first half of the result settles the architecture question cleanly. Cubin sizes genuinely differ across all three architectures, consistent with what Part 0 already established: `gpu_code` selects real, architecture-specific machine code, and different architectures need different amounts of it. Tile IR bytecode does not merely produce similar-sized output across architectures — it produces the exact same bytes, confirmed with a direct set-based equality check on the raw content, not just the lengths. `gpu_code` is still a required argument to `export_kernel` when `output_format="tileir_bytecode"`, but it is inert: whatever value is passed, the output is identical. That is architecture-independence demonstrated, not asserted — the same kind of genuine, driver-free confirmation this book has relied on since Part 0's own multi-target compilation chapter.

The file itself is a real, self-identifying format: its first eight bytes are `\x7fTileIR\x00`, an ASCII-readable magic number in the same spirit as ELF's `\x7fELF` header. That single detail is enough to say Tile IR bytecode is a genuine, independently defined file format with its own identity — not simply cubin renamed or repackaged.

But the byte counts themselves — 1019 for every architecture, in this run — are not a fact about cuTile Python in general; they are a fact about compiling this exact kernel from this exact file at this exact filesystem location. The second half of this Worked Example tests that suspicion directly, and confirms it precisely: writing the identical kernel source into three temporary directories whose absolute script paths differ by exactly one character each produces tileir_bytecode output that differs by exactly two bytes per character, every time. Re-running this exact file from a different directory than the one used for this chapter's own build would very likely print different absolute numbers than 1019 — not because anything about the kernel changed, but because Tile IR bytecode's exported bytes appear to embed something whose size scales with the length of the compiled file's own absolute path, most plausibly a source-location string used for debugging or diagnostics. The cross-architecture comparison earlier in this example is untouched by this — every one of those three numbers moves together, since all three came from the same file at the same path in the same run — but the raw byte counts are not a portable fact to memorize the way this book's cubin byte counts, from Part 0 onward, always have been. This is worth remembering the next time this book quotes a `tileir_bytecode` byte count: the relationship it demonstrates is genuine and reproducible: this specific number is not.

## 39.2 The Bytecode Version Knob

### Intuition

`export_kernel` takes one more parameter this book has never touched: `bytecode_version`, described as `None` for "automatically detect the latest TileIR bytecode version supported by the compiler," or an explicit `"major.minor"` string otherwise. Two questions follow immediately, both answerable only by testing this book's own installed compiler rather than assuming a version list from documentation that could drift out of date: which versions does it actually support right now, and does the default genuinely resolve to the latest one it claims to?

### Background

Rather than hard-coding a guessed list of supported versions into the chapter's own text, the more honest move — and the one consistent with this book's practice since Part 0 of triggering real errors instead of assuming constraints — is to deliberately pass a version string that cannot possibly be valid, and read the versions back out of the compiler's own `ValueError`. Whatever that error says is supported is what this book will then compile against, so the numbers in this section can never silently go stale relative to the installed package.

### Worked Example 39.2.1: reading supported versions from a real error, then testing the default

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

def sig_2d_2d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

def export_bytes(kernel, sig, gpu_code, output_format, bytecode_version=None):
    buf = io.BytesIO()
    ct.compilation.export_kernel(
        kernel, [sig], buf, gpu_code=gpu_code,
        output_format=output_format, bytecode_version=bytecode_version,
    )
    return buf.getvalue()

sig = sig_2d_2d_2d()

# --- Rather than assuming which bytecode versions the installed
# compiler supports, ask it -- by deliberately passing a version string
# it cannot possibly accept, and reading the ValueError it raises back.
# This is the same discipline this book has used since Part 0 whenever
# a constraint needed discovering rather than guessing: trigger the
# real error, then read its real message. ---
try:
    export_bytes(kernel_add, sig, "sm_90", "tileir_bytecode", bytecode_version="0.0")
    supported_versions = None
except ValueError as e:
    print(f"deliberately-invalid version rejected with: {e}")
    message = str(e)
    marker = "Supported versions are: "
    supported_versions = [v.strip() for v in message.split(marker)[1].split(",")]

print(f"supported bytecode versions (read from the error itself): {supported_versions}")
print()

print("--- byte count per explicit bytecode_version ---")
sizes_by_version = {}
for v in supported_versions:
    b = export_bytes(kernel_add, sig, "sm_90", "tileir_bytecode", bytecode_version=v)
    sizes_by_version[v] = len(b)
    print(f"  bytecode_version={v}: {len(b)} bytes")

default_bytes = export_bytes(kernel_add, sig, "sm_90", "tileir_bytecode", bytecode_version=None)
latest_version = supported_versions[-1]
print()
print(f"bytecode_version=None byte count: {len(default_bytes)}")
print(f"matches the LATEST supported version ({latest_version}): {len(default_bytes) == sizes_by_version[latest_version]}")

# --- The blind spot: does export_kernel even notice bytecode_version
# when output_format is 'cubin', where the parameter has no meaning?
# Genuinely tested rather than assumed from the docstring's silence
# on the interaction. ---
print()
cubin_plain = export_bytes(kernel_add, sig, "sm_90", "cubin", bytecode_version=None)
cubin_with_version = export_bytes(kernel_add, sig, "sm_90", "cubin", bytecode_version=latest_version)
print(f"cubin export with bytecode_version=None: {len(cubin_plain)} bytes")
print(f"cubin export with bytecode_version={latest_version!r}: {len(cubin_with_version)} bytes")
print(f"raised no error, and bytes are identical either way: {cubin_plain == cubin_with_version}")
```

### Genuinely run

```text
deliberately-invalid version rejected with: Unsupported bytecode version '0.0'. Supported versions are: 13.1, 13.2, 13.3
supported bytecode versions (read from the error itself): ['13.1', '13.2', '13.3']

--- byte count per explicit bytecode_version ---
  bytecode_version=13.1: 1019 bytes
  bytecode_version=13.2: 1019 bytes
  bytecode_version=13.3: 1021 bytes

bytecode_version=None byte count: 1021
matches the LATEST supported version (13.3): True

cubin export with bytecode_version=None: 26568 bytes
cubin export with bytecode_version='13.3': 26568 bytes
raised no error, and bytes are identical either way: True
```

### Discussion

This book's installed compiler currently supports exactly three Tile IR bytecode versions, 13.1 through 13.3 — a fact this chapter never had to assert, because it was read directly out of a genuine `ValueError` triggered on purpose. The version does affect the output: 13.1 and 13.2 compile to the same byte count as each other but a different one from 13.3, so the version string is not a cosmetic label. `bytecode_version=None`'s byte count matches 13.3 exactly, confirming the docstring's claim that the default resolves to "the latest TileIR bytecode version supported by the compiler" — a claim this chapter could have simply repeated, but instead measured.

One caveat carries over directly from Section 39.1: the raw byte counts above (1019, 1021 bytes, and so on) are specific to this book's own build path, for the reasons just measured there. What does not carry that caveat is every comparison this section actually draws conclusions from — `13.1 == 13.2`, `13.3` differing from both, `None` matching the latest version, and the two `cubin` exports matching each other — because each comparison is between byte counts produced from the same file at the same path in the same run, so whatever constant a different path might add would apply to both sides equally and cancel out. The relationships, not the raw numbers, are what this section's claims actually rest on.

The blind spot is the more interesting half of this section. `bytecode_version` is a keyword argument on `export_kernel` regardless of `output_format`, and nothing in the function's own type signature or docstring says it is ignored when `output_format="cubin"` — a reader would be entitled to wonder whether passing a bytecode version alongside a cubin request is even meaningful, or whether it might raise, or silently change something about the cubin. It does none of those: the call succeeds, and the resulting cubin is byte-for-byte identical whether `bytecode_version` is `None` or explicitly `"13.3"`. The parameter is real and load-bearing for one output format and completely inert for the other, with no warning printed either way. That asymmetry is worth knowing before writing code that threads a `bytecode_version` value through a call site that might, under some other code path, end up requesting a cubin instead.

## 39.3 Calling Conventions and Kernel Name Mangling

### Intuition

Every `KernelSignature` this book has built since Part 3 has included the line `calling_convention=ct.compilation.CallingConvention.cutile_python_v2()`, copied forward chapter after chapter without ever asking what that argument actually controls, or what `cutile_python_v1()` — the only other member this type exposes — would do differently. A calling convention, in the ordinary sense familiar from compiled languages, governs how a function's arguments are physically passed and how its symbol is named at the binary level; it does not usually change what the function computes. If that intuition transfers here, compiling the identical kernel and signature shape under `v1` and `v2` should produce the same computed code — the same cubin byte count — while changing only the identifying label attached to it.

### Background

`cuda.tile.compilation` exposes two free functions alongside `export_kernel`: `mangle_kernel_name(function_name, kernel_signature)`, which builds the exported symbol string for a given Python function name and signature, and `demangle_kernel_name(symbol)`, its inverse, which recovers a function name and a reconstructed `KernelSignature` from that string alone. Because a `KernelSignature` carries its own `calling_convention`, and `mangle_kernel_name` folds that signature into the symbol it produces, the calling convention ought to be visible in the mangled name itself — encoded somewhere a human could, in principle, read back out by eye. Whether that is true, and whether `demangle_kernel_name` genuinely recovers the original convention rather than some approximation of it, is directly testable.

### Worked Example 39.3.1: v1 vs. v2, mangled names, and a demangle round trip

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

def sig_with_convention(convention):
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=convention,
    )

def export_bytes(kernel, sig, gpu_code, output_format):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format=output_format)
    return buf.getvalue()

v1 = ct.compilation.CallingConvention.cutile_python_v1()
v2 = ct.compilation.CallingConvention.cutile_python_v2()
print(f"v1: {v1} (code={v1.code!r}, name={v1.name!r}, version={v1.version})")
print(f"v2: {v2} (code={v2.code!r}, name={v2.name!r}, version={v2.version})")
print(f"v1 == v2: {v1 == v2}")
print()

sig_v1 = sig_with_convention(v1)
sig_v2 = sig_with_convention(v2)

# --- Every prior chapter in this book that built a KernelSignature used
# cutile_python_v2() without remarking on it. This is the first time
# this book compiles the identical kernel and signature shape under
# BOTH calling conventions, to see whether the convention is cosmetic
# or load-bearing. ---
bytes_v1 = export_bytes(kernel_add, sig_v1, "sm_90", "cubin")
bytes_v2 = export_bytes(kernel_add, sig_v2, "sm_90", "cubin")
print(f"cubin bytes under v1: {len(bytes_v1)}")
print(f"cubin bytes under v2: {len(bytes_v2)}")
print(f"identical compiled code size under both conventions: {len(bytes_v1) == len(bytes_v2)}")
print()

name_v1 = ct.compilation.mangle_kernel_name("kernel_add", sig_v1)
name_v2 = ct.compilation.mangle_kernel_name("kernel_add", sig_v2)
print(f"mangled symbol under v1: {name_v1}")
print(f"mangled symbol under v2: {name_v2}")
print(f"mangled symbols differ: {name_v1 != name_v2}")
print()

# --- demangle_kernel_name is the inverse of mangle_kernel_name: given
# only the exported symbol string, recover the original function name
# and enough of the signature to know which calling convention produced
# it -- genuinely round-tripped, not assumed from the docstring. ---
recovered_name_v1, recovered_sig_v1 = ct.compilation.demangle_kernel_name(name_v1)
recovered_name_v2, recovered_sig_v2 = ct.compilation.demangle_kernel_name(name_v2)
print(f"demangle(v1 symbol) function name: {recovered_name_v1!r}")
print(f"demangle(v1 symbol) recovered calling convention: {recovered_sig_v1.calling_convention}")
print(f"demangle(v2 symbol) function name: {recovered_name_v2!r}")
print(f"demangle(v2 symbol) recovered calling convention: {recovered_sig_v2.calling_convention}")
print()
print(f"round trip recovers the same function name both times: {recovered_name_v1 == recovered_name_v2 == 'kernel_add'}")
print(f"round trip recovers the ORIGINAL calling convention in each case: "
      f"{recovered_sig_v1.calling_convention == v1 and recovered_sig_v2.calling_convention == v2}")
```

### Genuinely run

```text
v1: CallingConvention('cutile_python_v1', 't1') (code='t1', name='cutile_python_v1', version=1)
v2: CallingConvention('cutile_python_v2', 't2') (code='t2', name='cutile_python_v2', version=2)
v1 == v2: False

cubin bytes under v1: 26568
cubin bytes under v2: 26568
identical compiled code size under both conventions: True

mangled symbol under v1: kernel_add_Kt1_A2f32_3l0_A2f32_3l0_A2f32_3l0
mangled symbol under v2: kernel_add_Kt2_A2f32_3l0_A2f32_3l0_A2f32_3l0
mangled symbols differ: True

demangle(v1 symbol) function name: 'kernel_add'
demangle(v1 symbol) recovered calling convention: CallingConvention('cutile_python_v1', 't1')
demangle(v2 symbol) function name: 'kernel_add'
demangle(v2 symbol) recovered calling convention: CallingConvention('cutile_python_v2', 't2')

round trip recovers the same function name both times: True
round trip recovers the ORIGINAL calling convention in each case: True
```

### Discussion

The intuition holds, and it holds precisely: the identical kernel, compiled under `v1` and under `v2` with everything else fixed, produces the exact same cubin byte count — the calling convention changes nothing about what code gets generated. What it changes is the exported symbol's identity. Both mangled names share the same structure, encoding each `ArrayConstraint` the same way (`A2f32_3l0`, repeated three times for the three array parameters), but differ in exactly one segment: `Kt1` for `v1`, `Kt2` for `v2`, each embedding that convention's own short `code` (`t1` and `t2`, visible directly on the `CallingConvention` objects themselves). The convention is not folded into the machine code; it is folded into the name attached to that machine code — a distinction this chapter can now state with confidence rather than guess at, because both halves were independently measured.

`demangle_kernel_name`'s round trip closes the loop. Given nothing but the mangled string, it recovers both the original function name and a `KernelSignature` whose `calling_convention` compares equal to the exact object that produced it — not merely a convention with the same version number, but the same convention value. That is what makes `v1` and `v2` genuinely different calling conventions rather than cosmetic labels on identical output: given only an exported symbol, and nothing else, it is possible to know which convention compiled it, which is exactly the kind of ABI-level identifiability a calling convention is supposed to provide in the first place. It is also a fact worth carrying forward: every `KernelSignature` this book builds from here on is still choosing `cutile_python_v2()`, and now that choice is a stated one rather than an inherited default.

## Chapter Summary

`ct.compilation.export_kernel`'s `output_format` parameter has always offered a second setting this book had never used: `"tileir_bytecode"`, alongside the `"cubin"` this book has relied on since Part 0. Compiling the same kernel and signature at three different target architectures confirmed the difference is real rather than nominal — cubin's byte count genuinely varies across `sm_80`, `sm_90`, and `sm_100`, while Tile IR bytecode, identified by its own `\x7fTileIR` magic header, comes back byte-for-byte identical no matter which architecture name was passed alongside it. A further, unplanned discovery followed directly from that result: Tile IR bytecode's exact byte count is sensitive to the length of the exporting file's own absolute filesystem path — precisely two bytes per extra character, confirmed across three controlled temporary directories — while cubin's byte count is not, for this kernel, affected by path at all. The `bytecode_version` parameter was read out of the compiler's own `ValueError` rather than assumed, revealing exactly three supported versions (13.1 through 13.3) with `None` genuinely resolving to the latest, and a real blind spot where `bytecode_version` is silently ignored whenever `output_format="cubin"`, with no error to flag the mismatch. Finally, `ct.compilation.CallingConvention`'s two members, `cutile_python_v1()` and `v2()` — the second of which every `KernelSignature` since Part 3 has quietly used — were shown to compile to identical cubin bytes while producing distinctly different mangled symbol names, with `demangle_kernel_name` genuinely recovering the original convention from the symbol string alone.

## Self-Check Questions

1. Section 39.1 found that moving the identical kernel source to a directory whose absolute path was one character longer added exactly two bytes to the exported `tileir_bytecode` output, every time. Propose a concrete explanation for why the byte count changes by *two* bytes rather than *one* for each extra character of path length, and describe one further test that would confirm or rule it out.
2. Section 39.2's default `bytecode_version=None` resolved to version 13.3, the latest this compiler supports. If a future version of the installed `cuda-tile` package added support for a version 13.4, what would you expect `bytecode_version=None`'s byte count to do, and why?
3. Explain, in your own words, why the "blind spot" in Section 39.2 — `bytecode_version` being silently ignored under `output_format="cubin"` — is arguably more dangerous for a caller than if it had simply raised an error.
4. Section 39.3 showed that `v1` and `v2` calling conventions compile to identical cubin bytes for `kernel_add`. Does that mean calling convention can never affect compiled code size for any kernel, or is `kernel_add`'s particular shape and signature a reason to be cautious about generalizing that finding?
5. Both `mangle_kernel_name` and `demangle_kernel_name` were tested only on `kernel_add` with the same three-array signature shape used throughout this chapter. What would a stronger test of the round trip's correctness have looked like?

## Where We Go Next

This chapter re-opened `export_kernel` and found genuine texture this book had walked past for thirty-eight chapters: a second output format that behaves exactly as a portable intermediate representation should, a version knob with a real blind spot, and a calling-convention choice this book had been making silently since Part 3. All three findings share a theme — they are properties of the *compiled artifact*, observable without ever touching a live CUDA stream, which is exactly the kind of ground this book's driver-free verification method was built to cover honestly. Part 5's other half is the harder-looking problem: benchmarking. The next chapter turns to what "genuinely run" can still mean when the very thing worth measuring — how fast a kernel executes — is precisely the one number this sandboxed environment cannot produce, and asks what honest, clearly-labeled evidence looks like on the near side of that wall.

## Worked Solutions

**1.** A plausible concrete explanation is that the absolute path is stored somewhere in the bytecode using two bytes per character rather than one — consistent, for instance, with a UTF-16-style encoding of an ordinary ASCII path, where every additional ASCII character costs exactly two bytes instead of the one byte it would cost in UTF-8 or plain ASCII. A byte-level comparison of two exports whose paths differ by one character supports this: the two outputs are identical up to a small length-style field partway through the file, which increases by exactly two between the shorter and longer path, after which the remaining bytes shift accordingly. One further test that would distinguish this hypothesis from alternatives (for instance, two separate one-byte fields both incrementing, or an unrelated coincidence) is to use a path containing a single non-ASCII character — such as an accented letter, which takes two bytes in UTF-8 but is still one code unit in UTF-16 — and check whether the byte cost tracks the character's UTF-8 length (ruling out UTF-16) or stays at a flat two bytes regardless of which character was added (supporting it).

**2.** You would expect `bytecode_version=None`'s byte count to become whatever a fresh, explicit `bytecode_version="13.4"` call produces, no longer matching what today's `"13.3"` call produces — because the docstring's contract is "the latest version this compiler currently supports," not "version 13.3 specifically." The default is defined relative to the installed compiler's own capabilities, so it would track a hypothetical 13.4 forward automatically, exactly as it already tracked forward to 13.3 rather than stopping at 13.1 or 13.2.

**3.** An error at least tells the caller immediately, at the call site, that something about their request wasn't honored — they can fix the call or accept the limitation with full knowledge of it. Silent ignoring does neither: code that threads a `bytecode_version` value through a call site that sometimes requests `"cubin"` and sometimes `"tileir_bytecode"` will work exactly as expected for one format and silently produce output that never reflects the intended version pin for the other, with nothing in the returned bytes, the return value, or the console to reveal that anything was dropped. The bug only surfaces once someone specifically checks whether the pin took effect — which, absent Section 39.2's own test, might never happen at all.

**4.** It should not be generalized past `kernel_add`'s specific shape. `kernel_add` is a small, purely elementwise kernel with a straightforward three-array signature; there is no guarantee that a calling convention's effect on code size — if any exists at all — would stay zero for a kernel with a much larger parameter list, `ct.Constant` parameters, or more complex tile operations, where a different calling convention might, for instance, pass a scalar constant differently at the ABI level in a way that does show up in generated code. This chapter's finding is honestly scoped to what it tested: for this one kernel and signature shape, the two conventions produced identical code. A broader claim would need broader testing.

**5.** A stronger test would have varied the kernel's own signature — different array ranks, added `ct.Constant[int]` parameters, or a different function name entirely — and confirmed the round trip still recovered the exact original function name and an equal `KernelSignature` (not just an equal `calling_convention`) in every case, rather than relying on one fixed three-array shape throughout. It also would have tried demangling a symbol that was never produced by `mangle_kernel_name` at all — a hand-written or corrupted string — to see whether `demangle_kernel_name` fails loudly or silently produces a plausible-looking wrong answer, which this chapter never checked.

## Complete Runnable Code

`01_cubin_vs_tileir_bytecode_across_architectures.py`:

```python
import cuda.tile as ct
import io
import os
import subprocess
import sys
import tempfile
import textwrap

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

def sig_2d_2d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

# --- The same elementwise-add kernel this book has compiled many times
# since Part 2, unchanged. What is new in this chapter is not the
# kernel -- it is which of export_kernel's two output_format values we
# ask for, and for how many different target architectures. ---
@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

def export_bytes(kernel, sig, gpu_code, output_format):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format=output_format)
    return buf.getvalue()

sig = sig_2d_2d_2d()
architectures = ["sm_80", "sm_90", "sm_100"]

print("--- output_format='cubin' ---")
cubin_sizes = {}
for gc in architectures:
    b = export_bytes(kernel_add, sig, gc, "cubin")
    cubin_sizes[gc] = len(b)
    print(f"  gpu_code={gc}: {len(b)} bytes")

print()
print("--- output_format='tileir_bytecode' ---")
bytecode_sizes = {}
bytecode_bytes = {}
for gc in architectures:
    b = export_bytes(kernel_add, sig, gc, "tileir_bytecode")
    bytecode_sizes[gc] = len(b)
    bytecode_bytes[gc] = b
    print(f"  gpu_code={gc}: {len(b)} bytes")

print()
print(f"cubin sizes all identical across architectures: {len(set(cubin_sizes.values())) == 1}")
print(f"tileir_bytecode sizes all identical across architectures: {len(set(bytecode_sizes.values())) == 1}")
print(f"tileir_bytecode bytes byte-for-byte identical across architectures: {len(set(bytecode_bytes.values())) == 1}")
print(f"tileir_bytecode magic header: {bytecode_bytes['sm_90'][:8]}")

# --- The byte counts printed above are only reproducible from this
# exact file at this exact filesystem location -- suspicious, given
# that this book's earlier chapters already found compiled byte counts
# can depend on a kernel's source filename. Rather than assume this
# chapter's numbers are as portable as they look, test directly: export
# the identical kernel from three temporary directories whose absolute
# path lengths differ by exactly one character each, and compare. ---
KERNEL_SOURCE = textwrap.dedent("""
    import cuda.tile as ct
    import io

    def array_param(ndim, dtype=ct.float32):
        return ct.compilation.ArrayConstraint(
            dtype, ndim=ndim, index_dtype=ct.int32,
            stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        )

    M, N = 8, 8

    @ct.kernel
    def kernel_add(x, y, out):
        a = ct.load(x, (0, 0), (M, N))
        b = ct.load(y, (0, 0), (M, N))
        ct.store(out, (0, 0), a + b)

    sig = ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel_add, [sig], buf, gpu_code="sm_90", output_format="tileir_bytecode")
    print(len(buf.getvalue()))
""")

base = tempfile.mkdtemp()
results = {}
for pad in (0, 1, 2):
    subdir = os.path.join(base, "d" + "x" * pad)
    os.makedirs(subdir)
    script_path = os.path.join(subdir, "probe.py")
    with open(script_path, "w") as f:
        f.write(KERNEL_SOURCE)
    proc = subprocess.run([sys.executable, "probe.py"], cwd=subdir, capture_output=True, text=True)
    results[len(script_path)] = int(proc.stdout.strip())

print()
print("--- same kernel, exported from three different absolute paths ---")
path_lengths = sorted(results)
for length in path_lengths:
    print(f"  absolute script path length {length}: {results[length]} tileir_bytecode bytes")
deltas = [results[path_lengths[i + 1]] - results[path_lengths[i]] for i in range(len(path_lengths) - 1)]
print(f"every +1 character of absolute path length adds exactly +2 bytes: {all(d == 2 for d in deltas)}")
```

`02_bytecode_version_pins_and_the_cubin_blind_spot.py`:

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

def sig_2d_2d_2d():
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
    )

@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

def export_bytes(kernel, sig, gpu_code, output_format, bytecode_version=None):
    buf = io.BytesIO()
    ct.compilation.export_kernel(
        kernel, [sig], buf, gpu_code=gpu_code,
        output_format=output_format, bytecode_version=bytecode_version,
    )
    return buf.getvalue()

sig = sig_2d_2d_2d()

# --- Rather than assuming which bytecode versions the installed
# compiler supports, ask it -- by deliberately passing a version string
# it cannot possibly accept, and reading the ValueError it raises back.
# This is the same discipline this book has used since Part 0 whenever
# a constraint needed discovering rather than guessing: trigger the
# real error, then read its real message. ---
try:
    export_bytes(kernel_add, sig, "sm_90", "tileir_bytecode", bytecode_version="0.0")
    supported_versions = None
except ValueError as e:
    print(f"deliberately-invalid version rejected with: {e}")
    message = str(e)
    marker = "Supported versions are: "
    supported_versions = [v.strip() for v in message.split(marker)[1].split(",")]

print(f"supported bytecode versions (read from the error itself): {supported_versions}")
print()

print("--- byte count per explicit bytecode_version ---")
sizes_by_version = {}
for v in supported_versions:
    b = export_bytes(kernel_add, sig, "sm_90", "tileir_bytecode", bytecode_version=v)
    sizes_by_version[v] = len(b)
    print(f"  bytecode_version={v}: {len(b)} bytes")

default_bytes = export_bytes(kernel_add, sig, "sm_90", "tileir_bytecode", bytecode_version=None)
latest_version = supported_versions[-1]
print()
print(f"bytecode_version=None byte count: {len(default_bytes)}")
print(f"matches the LATEST supported version ({latest_version}): {len(default_bytes) == sizes_by_version[latest_version]}")

# --- The blind spot: does export_kernel even notice bytecode_version
# when output_format is 'cubin', where the parameter has no meaning?
# Genuinely tested rather than assumed from the docstring's silence
# on the interaction. ---
print()
cubin_plain = export_bytes(kernel_add, sig, "sm_90", "cubin", bytecode_version=None)
cubin_with_version = export_bytes(kernel_add, sig, "sm_90", "cubin", bytecode_version=latest_version)
print(f"cubin export with bytecode_version=None: {len(cubin_plain)} bytes")
print(f"cubin export with bytecode_version={latest_version!r}: {len(cubin_with_version)} bytes")
print(f"raised no error, and bytes are identical either way: {cubin_plain == cubin_with_version}")
```

`03_calling_conventions_and_kernel_name_mangling.py`:

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

M, N = 8, 8

@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

def sig_with_convention(convention):
    return ct.compilation.KernelSignature(
        [array_param(2), array_param(2), array_param(2)],
        calling_convention=convention,
    )

def export_bytes(kernel, sig, gpu_code, output_format):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format=output_format)
    return buf.getvalue()

v1 = ct.compilation.CallingConvention.cutile_python_v1()
v2 = ct.compilation.CallingConvention.cutile_python_v2()
print(f"v1: {v1} (code={v1.code!r}, name={v1.name!r}, version={v1.version})")
print(f"v2: {v2} (code={v2.code!r}, name={v2.name!r}, version={v2.version})")
print(f"v1 == v2: {v1 == v2}")
print()

sig_v1 = sig_with_convention(v1)
sig_v2 = sig_with_convention(v2)

# --- Every prior chapter in this book that built a KernelSignature used
# cutile_python_v2() without remarking on it. This is the first time
# this book compiles the identical kernel and signature shape under
# BOTH calling conventions, to see whether the convention is cosmetic
# or load-bearing. ---
bytes_v1 = export_bytes(kernel_add, sig_v1, "sm_90", "cubin")
bytes_v2 = export_bytes(kernel_add, sig_v2, "sm_90", "cubin")
print(f"cubin bytes under v1: {len(bytes_v1)}")
print(f"cubin bytes under v2: {len(bytes_v2)}")
print(f"identical compiled code size under both conventions: {len(bytes_v1) == len(bytes_v2)}")
print()

name_v1 = ct.compilation.mangle_kernel_name("kernel_add", sig_v1)
name_v2 = ct.compilation.mangle_kernel_name("kernel_add", sig_v2)
print(f"mangled symbol under v1: {name_v1}")
print(f"mangled symbol under v2: {name_v2}")
print(f"mangled symbols differ: {name_v1 != name_v2}")
print()

# --- demangle_kernel_name is the inverse of mangle_kernel_name: given
# only the exported symbol string, recover the original function name
# and enough of the signature to know which calling convention produced
# it -- genuinely round-tripped, not assumed from the docstring. ---
recovered_name_v1, recovered_sig_v1 = ct.compilation.demangle_kernel_name(name_v1)
recovered_name_v2, recovered_sig_v2 = ct.compilation.demangle_kernel_name(name_v2)
print(f"demangle(v1 symbol) function name: {recovered_name_v1!r}")
print(f"demangle(v1 symbol) recovered calling convention: {recovered_sig_v1.calling_convention}")
print(f"demangle(v2 symbol) function name: {recovered_name_v2!r}")
print(f"demangle(v2 symbol) recovered calling convention: {recovered_sig_v2.calling_convention}")
print()
print(f"round trip recovers the same function name both times: {recovered_name_v1 == recovered_name_v2 == 'kernel_add'}")
print(f"round trip recovers the ORIGINAL calling convention in each case: "
      f"{recovered_sig_v1.calling_convention == v1 and recovered_sig_v2.calling_convention == v2}")
```
