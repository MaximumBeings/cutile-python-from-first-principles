# Chapter 12: Symbol Names and Multi-Kernel Artifacts

> "Chapter 11 composed kernels by calling one Python function from inside another, all traced and inlined into a single compiled entry point. This chapter composes them a different way: putting more than one genuinely distinct, separately-callable compiled kernel into a single file. cuTile Python already does this quietly every time `export_kernel` is given more than one signature — Chapter 2 and Chapter 5 measured the different sizes those specializations produce, but neither asked what actually distinguishes them inside the file, or whether the same trick extends to combining genuinely different kernel functions, not just different specializations of one. This chapter answers both, cross-checking every claim against the real, compiled artifact with the same standard tools any GPU toolchain uses to inspect one."

**What you will understand by the end of this chapter:**

- That `ct.compilation.mangle_kernel_name` computes the exact compiled symbol name a given kernel name and signature will get, entirely without compiling anything — and that this prediction is confirmed to match the real symbol table of an actual compiled cubin, cross-checked with the standard `nm` tool
- That giving `export_kernel` multiple signatures in one call produces a single, genuinely coherent compiled artifact — one real ELF object, confirmed with `readelf`, holding every specialization's real, independently-findable symbol
- That calling `export_kernel` twice with the same string filename does not combine two different kernels into one file — it overwrites, and only the second kernel survives
- That calling `export_kernel` twice on the same open, unclosed file object does not overwrite, but also does not produce one coherent multi-kernel artifact either — it produces two complete, independently-valid compiled objects concatenated back to back, of which standard whole-file tools see only the first
- That `ct.compilation.demangle_kernel_name` genuinely rejects invalid input, and that its round-trip is stable at the level of a signature's real parameters and calling convention, even though a naive whole-object equality check on the result is misleading for a documented, specific reason

**What you need to know first:**

- Chapter 2's and Chapter 5's confirmation that one `export_kernel` call can compile multiple, genuinely different specializations of the same kernel from a list of signatures.
- Chapter 10's finding that a compiled cubin embeds its kernel's Python function name as a real symbol.
- Basic familiarity with what a symbol table is; this chapter uses the standard, pre-installed `nm` and `readelf` tools to inspect one directly, since a cubin is a real ELF file.

## 12.1 Predicting Symbol Names Before Compiling `[FOUNDATIONAL]`

### Intuition

Chapter 10 found that a kernel's Python function name ends up embedded as a real compiled symbol. A kernel compiled for several different signatures needs several different symbols to distinguish them — `ct.compilation.mangle_kernel_name` is documented as the function that computes exactly what those symbols are, without requiring a real compile at all.

### Background

`mangle_kernel_name(function_name: str, kernel_signature: KernelSignature) -> str` takes a kernel's name and one of its signatures and returns the symbol name that specific pairing would compile to. Its counterpart, `demangle_kernel_name(symbol: str) -> tuple[str, KernelSignature]`, is examined directly in Section 12.5.

### Worked Example 12.1.1 — different signatures, different predicted names

```python
import cuda.tile as ct

def array_param(dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig_f32 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f16 = ct.compilation.KernelSignature(
    [array_param(ct.float16), array_param(ct.float16), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f32_256 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(512)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# mangle_kernel_name(function_name, kernel_signature) computes the compiled
# symbol name a given (name, signature) pair would get -- entirely without
# compiling anything.
mangled_f32 = ct.compilation.mangle_kernel_name("my_kernel", sig_f32)
mangled_f16 = ct.compilation.mangle_kernel_name("my_kernel", sig_f16)
mangled_f32_256 = ct.compilation.mangle_kernel_name("my_kernel", sig_f32_256)

print(f"mangled (float32 dtype, tile_size=256): {mangled_f32}")
print(f"mangled (float16 dtype, tile_size=256): {mangled_f16}")
print(f"mangled (float32 dtype, tile_size=512): {mangled_f32_256}")
print(f"dtype change alone changes the mangled name: {mangled_f32 != mangled_f16}")
print(f"ConstantConstraint value alone changes the mangled name: {mangled_f32 != mangled_f32_256}")

# demangle_kernel_name is the inverse: given a real symbol, recover the
# original function name and an equivalent KernelSignature.
name, recovered_sig = ct.compilation.demangle_kernel_name(mangled_f32)
print(f"demangle_kernel_name({mangled_f32!r}) function name: {name}")
print(f"recovered parameters match original: {recovered_sig.parameters == sig_f32.parameters}")
```

The complete file is `01_mangle_kernel_name.py`, included in this chapter's Complete Runnable Code.

```bash
python3 01_mangle_kernel_name.py
```

Genuinely run:

```
mangled (float32 dtype, tile_size=256): my_kernel_Kt2_A1f32_1l0_A1f32_1l0_I256
mangled (float16 dtype, tile_size=256): my_kernel_Kt2_A1f16_1l0_A1f16_1l0_I256
mangled (float32 dtype, tile_size=512): my_kernel_Kt2_A1f32_1l0_A1f32_1l0_I512
dtype change alone changes the mangled name: True
ConstantConstraint value alone changes the mangled name: True
demangle_kernel_name('my_kernel_Kt2_A1f32_1l0_A1f32_1l0_I256') function name: my_kernel
recovered parameters match original: True
```

The mangled name genuinely encodes the discriminating parts of the signature: `f32` versus `f16` changes it, and `I256` versus `I512` (the `ConstantConstraint` value) changes it too, both confirmed directly rather than assumed from the function's name alone. This is exactly what makes multiple specializations of one kernel distinguishable as separate symbols in one file — the subject of Section 12.3.

## 12.2 Confirming Predicted Names Are the Real Ones `[FOUNDATIONAL]`

### Intuition

`mangle_kernel_name` computing *a* string is not the same claim as that string being the actual symbol a real compile produces. This section closes that gap by compiling a real kernel and inspecting its real, compiled symbol table with a tool that has no connection to cuTile Python at all.

### Background

A `.cubin` file is a real ELF (Executable and Linkable Format) object — the same object format Linux uses for ordinary executables and shared libraries. That means standard ELF tools, including `nm` (list symbols) and `readelf` (dump ELF structure), work on a cubin directly, giving this book a genuinely independent way to check a claim about compiled output: not by reading cuTile Python's own reported byte count, but by asking a completely unrelated tool what it actually finds inside the file.

### Worked Example 12.2.1 — predicted names against `nm`'s real symbol table

```python
import cuda.tile as ct
import subprocess

def array_param(dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def my_kernel(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

sig_f32 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f16 = ct.compilation.KernelSignature(
    [array_param(ct.float16), array_param(ct.float16), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

predicted_f32 = ct.compilation.mangle_kernel_name("my_kernel", sig_f32)
predicted_f16 = ct.compilation.mangle_kernel_name("my_kernel", sig_f16)
print(f"predicted symbol (f32): {predicted_f32}")
print(f"predicted symbol (f16): {predicted_f16}")

ct.compilation.export_kernel(my_kernel, [sig_f32, sig_f16], "02_out.cubin", gpu_code="sm_90", output_format="cubin")

# Cross-check the predicted names against the real symbol table of the
# actual compiled artifact, using the standard "nm" tool -- a cubin is a
# real ELF file, and export_kernel's own compiled output is the ground
# truth here, entirely independent of mangle_kernel_name's own prediction.
result = subprocess.run(["nm", "02_out.cubin"], capture_output=True, text=True, check=True)
real_symbols = set()
for line in result.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 3:
        real_symbols.add(parts[2])

print(f"real symbols found by nm: {sorted(real_symbols)}")
print(f"predicted f32 symbol is a real symbol in the compiled cubin: {predicted_f32 in real_symbols}")
print(f"predicted f16 symbol is a real symbol in the compiled cubin: {predicted_f16 in real_symbols}")
```

The complete file is `02_predicted_names_match_real_symbols.py`, included in this chapter's Complete Runnable Code.

```bash
python3 02_predicted_names_match_real_symbols.py
```

Genuinely run:

```
predicted symbol (f32): my_kernel_Kt2_A1f32_1l0_A1f32_1l0_I256
predicted symbol (f16): my_kernel_Kt2_A1f16_1l0_A1f16_1l0_I256
real symbols found by nm: ['___NV_TILE_LAUNCH_META_DATA___', '__nv_reservedSMEM_offset_0_alias', 'my_kernel_Kt2_A1f16_1l0_A1f16_1l0_I256', 'my_kernel_Kt2_A1f32_1l0_A1f32_1l0_I256']
predicted f32 symbol is a real symbol in the compiled cubin: True
predicted f16 symbol is a real symbol in the compiled cubin: True
```

Both predicted names are genuinely present in the real symbol table, confirmed by a tool with no dependency on cuTile Python whatsoever. `nm` also surfaces two other real symbols this book has not discussed — `___NV_TILE_LAUNCH_META_DATA___` and `__nv_reservedSMEM_offset_0_alias` — compiler-generated infrastructure alongside the kernel entry points themselves, a genuine reminder that a compiled cubin's symbol table holds more than just the kernels a reader wrote.

## 12.3 One Call, Multiple Signatures: A Genuinely Coherent Artifact `[FOUNDATIONAL]`

### Intuition

Chapter 2 and Chapter 5 already confirmed that one `export_kernel` call given several signatures compiles several genuinely different specializations. Section 12.2 confirmed each specialization gets its own real symbol. Neither fact, on its own, proves the *file* is one coherent artifact rather than several separate ones happening to share a name — this section checks the file's actual structure directly.

### Background

`readelf -h` prints an ELF file's top-level header — if a file contained several independent ELF objects rather than one coherent one, a tool reading it as a single ELF file would either fail, or report only the first object it finds. This section checks which of those two outcomes actually happens for a genuine multi-signature export.

### Worked Example 12.3.1 — three signatures, one file, checked from three angles

```python
import cuda.tile as ct
import subprocess

def array_param(dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def my_kernel(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

sig_f32 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f16 = ct.compilation.KernelSignature(
    [array_param(ct.float16), array_param(ct.float16), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f64 = ct.compilation.KernelSignature(
    [array_param(ct.float64), array_param(ct.float64), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# One export_kernel call, three signatures -- Chapter 2 and Chapter 5 already
# confirmed this produces three genuinely different compiled specializations.
# This checks something neither chapter checked: is the *file itself* one
# coherent artifact (a single ELF object), or three separate ones happening
# to sit in the same file?
ct.compilation.export_kernel(my_kernel, [sig_f32, sig_f16, sig_f64], "03_out.cubin", gpu_code="sm_90", output_format="cubin")

elf_magic_count = 0
data = open("03_out.cubin", "rb").read()
pos = 0
while True:
    idx = data.find(b"\x7fELF", pos)
    if idx == -1:
        break
    elf_magic_count += 1
    pos = idx + 1
print(f"raw ELF magic byte occurrences anywhere in the file: {elf_magic_count}")

readelf_result = subprocess.run(["readelf", "-h", "03_out.cubin"], capture_output=True, text=True, check=True)
header_lines = [l for l in readelf_result.stdout.splitlines() if "ELF Header" in l]
print(f"'ELF Header:' lines readelf reports for the whole file: {len(header_lines)}")

nm_result = subprocess.run(["nm", "03_out.cubin"], capture_output=True, text=True, check=True)
predicted = {
    ct.compilation.mangle_kernel_name("my_kernel", sig_f32),
    ct.compilation.mangle_kernel_name("my_kernel", sig_f16),
    ct.compilation.mangle_kernel_name("my_kernel", sig_f64),
}
real_symbols = {line.split()[2] for line in nm_result.stdout.splitlines() if len(line.split()) >= 3}
print(f"all three predicted symbols found by one nm call on the whole file: {predicted <= real_symbols}")
```

The complete file is `03_multi_signature_is_one_coherent_artifact.py`, included in this chapter's Complete Runnable Code.

```bash
python3 03_multi_signature_is_one_coherent_artifact.py
```

Genuinely run:

```
raw ELF magic byte occurrences anywhere in the file: 3
'ELF Header:' lines readelf reports for the whole file: 1
all three predicted symbols found by one nm call on the whole file: True
```

`readelf -h` reports exactly one top-level ELF header for the whole file, and one `nm` call finds all three specializations' real symbols in that single pass — genuine evidence this is one coherent artifact, not three files concatenated by accident. (The raw ELF magic bytes turning up three times inside the file is not evidence against this — a real compiled cubin can legitimately embed further ELF-shaped structure inside its own sections, and this book draws no further conclusion from that count beyond reporting it as measured.) This is the behavior Section 12.4 contrasts directly against what happens when the same file receives two calls instead of one.

## 12.4 Two Ways to (Not) Compose Different Kernels Into One File `[FOUNDATIONAL]`

### Intuition

Section 12.3 confirmed multiple signatures of *one* kernel compose cleanly into one file. Genuinely different kernel functions have no `signatures` list to share — the natural next attempt is calling `export_kernel` more than once, targeting the same file. This section checks the two most obvious ways to try that, and finds that neither one actually produces what Section 12.3 produced.

### Background

`export_kernel`'s `output_file` parameter accepts either a filename or an already-open, binary file-like object. Those are genuinely different things to hand it: a filename means `export_kernel` opens the file itself, while an object means the caller controls exactly when and how it was opened, and where its write position currently sits.

### Worked Example 12.4.1 — the same string filename, called twice

```python
import cuda.tile as ct
import os
import subprocess

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_add(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

@ct.kernel
def kernel_sub(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

# Calling export_kernel twice with the SAME filename -- does the second call
# overwrite the first, or accumulate into one file holding both kernels?
ct.compilation.export_kernel(kernel_add, [sig], "04_out.cubin", gpu_code="sm_90", output_format="cubin")
size_after_first = os.path.getsize("04_out.cubin")
print(f"size after exporting kernel_add alone: {size_after_first} bytes")

ct.compilation.export_kernel(kernel_sub, [sig], "04_out.cubin", gpu_code="sm_90", output_format="cubin")
size_after_second = os.path.getsize("04_out.cubin")
print(f"size after exporting kernel_sub to the same filename: {size_after_second} bytes")

nm_result = subprocess.run(["nm", "04_out.cubin"], capture_output=True, text=True, check=True)
real_symbols = {line.split()[2] for line in nm_result.stdout.splitlines() if len(line.split()) >= 3}
add_symbol = ct.compilation.mangle_kernel_name("kernel_add", sig)
sub_symbol = ct.compilation.mangle_kernel_name("kernel_sub", sig)
print(f"kernel_add's symbol still present: {add_symbol in real_symbols}")
print(f"kernel_sub's symbol present: {sub_symbol in real_symbols}")
```

The complete file is `04_same_filename_overwrites.py`, included in this chapter's Complete Runnable Code.

```bash
python3 04_same_filename_overwrites.py
```

Genuinely run:

```
size after exporting kernel_add alone: 21792 bytes
size after exporting kernel_sub to the same filename: 21792 bytes
kernel_add's symbol still present: False
kernel_sub's symbol present: True
```

> `[COMMON TRAP]` The file size never changes (21,792 bytes both times), and `kernel_add`'s real symbol is genuinely gone after the second call — passing a filename opens that file in a mode that overwrites whatever was there, exactly the way writing to any ordinary file by name would. This is not a way to accumulate two different kernels into one artifact; it silently discards the first one, with no error or warning of any kind at either call site.

### Worked Example 12.4.2 — the same open file object, called twice

```python
import cuda.tile as ct
import os
import subprocess

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_add(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

@ct.kernel
def kernel_sub(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

# Section 04 found that a repeated *filename* overwrites. This checks a
# different approach: keeping one file object open in "wb" mode across both
# calls, so export_kernel writes at wherever the file's current position is,
# rather than opening (and truncating) the file itself each time.
with open("05_out.cubin", "wb") as f:
    ct.compilation.export_kernel(kernel_add, [sig], f, gpu_code="sm_90", output_format="cubin")
    boundary = f.tell()
    print(f"file position after the first export_kernel call: {boundary}")
    ct.compilation.export_kernel(kernel_sub, [sig], f, gpu_code="sm_90", output_format="cubin")
    print(f"file position after the second export_kernel call: {f.tell()}")

print(f"final file size on disk: {os.path.getsize('05_out.cubin')} bytes")

add_symbol = ct.compilation.mangle_kernel_name("kernel_add", sig)
sub_symbol = ct.compilation.mangle_kernel_name("kernel_sub", sig)

# Reading the *whole* concatenated file with the standard tools:
nm_whole = subprocess.run(["nm", "05_out.cubin"], capture_output=True, text=True, check=True)
whole_symbols = {line.split()[2] for line in nm_whole.stdout.splitlines() if len(line.split()) >= 3}
print(f"kernel_add's symbol visible when nm reads the whole file: {add_symbol in whole_symbols}")
print(f"kernel_sub's symbol visible when nm reads the whole file: {sub_symbol in whole_symbols}")

# Slicing out exactly the bytes written by the second call, using the file
# position already recorded above as the real boundary between the two
# genuinely separate, complete compiled objects concatenated in this file.
with open("05_out.cubin", "rb") as f:
    f.seek(boundary)
    second_half = f.read()
with open("05_second_half.cubin", "wb") as f:
    f.write(second_half)

nm_second = subprocess.run(["nm", "05_second_half.cubin"], capture_output=True, text=True, check=True)
second_symbols = {line.split()[2] for line in nm_second.stdout.splitlines() if len(line.split()) >= 3}
print(f"kernel_sub's symbol visible once the second object is sliced out on its own: {sub_symbol in second_symbols}")
```

The complete file is `05_open_file_object_concatenates_but_only_first_is_visible.py`, included in this chapter's Complete Runnable Code.

```bash
python3 05_open_file_object_concatenates_but_only_first_is_visible.py
```

Genuinely run:

```
file position after the first export_kernel call: 21792
file position after the second export_kernel call: 43584
final file size on disk: 43584 bytes
kernel_add's symbol visible when nm reads the whole file: True
kernel_sub's symbol visible when nm reads the whole file: False
kernel_sub's symbol visible once the second object is sliced out on its own: True
```

> `[COMMON TRAP]` This result sits in between Section 12.3's success and Section 12.4.1's outright overwrite, and it is easy to misread as success. The file genuinely doubles in size (21,792 to 43,584 bytes), proving both calls wrote their real, complete output rather than one clobbering the other. But `kernel_sub`'s symbol is invisible when `nm` reads the *whole* file — only `kernel_add`'s is found. Slicing the file at exactly the byte offset recorded after the first call, and handing `nm` that second slice on its own, immediately reveals `kernel_sub`'s real, valid symbol. The two calls wrote two complete, independently valid compiled objects, back to back in one file — genuinely different from Section 12.3's single coherent artifact, where one `nm` call over the whole file found every specialization at once. An open file object stops `export_kernel` from overwriting, but it does not make repeated calls equivalent to one call with multiple signatures.

## 12.5 `demangle_kernel_name`: Real Errors, and a Round-Trip Subtlety `[FOUNDATIONAL]`

### Intuition

Section 12.1 already used `demangle_kernel_name` successfully. Two questions remain: what does it do with input that was never a real mangled name at all, and does the value it returns round-trip back to something genuinely equal to what `mangle_kernel_name` was given in the first place?

### Background

A `KernelSignature` carries a `symbol` field of its own, distinct from the parameters and calling convention that actually determine compiled behavior. A freshly constructed `KernelSignature`, like every one built in this book so far, leaves `symbol` at its default; `demangle_kernel_name` has a real reason to set it on the object it returns, since it was given that exact symbol to work backward from.

### Worked Example 12.5.1 — invalid input, and a round-trip checked field by field

```python
import cuda.tile as ct

# Genuinely invalid strings are rejected outright.
for bad in ["not_a_real_mangled_name", "", "my_kernel"]:
    try:
        result = ct.compilation.demangle_kernel_name(bad)
        print(f"demangle_kernel_name({bad!r}): {result}")
    except ValueError as e:
        print(f"demangle_kernel_name({bad!r}): ValueError: {e}")

# A real, complete signature and its mangled name.
sig = ct.compilation.KernelSignature(
    [ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                     stride_lower_bound_incl=None, alias_groups=[],
                                     may_alias_internally=False)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
mangled = ct.compilation.mangle_kernel_name("my_kernel", sig)
print(f"mangled: {mangled}")

name, recovered = ct.compilation.demangle_kernel_name(mangled)
print(f"demangled name: {name}")

# Naive whole-object equality: does it hold?
print(f"recovered == original (whole KernelSignature objects): {recovered == sig}")
print(f"original.symbol: {sig.symbol!r}")
print(f"recovered.symbol: {recovered.symbol!r}")

# The actual semantic content -- parameters and calling convention -- is
# what round-trips; only the symbol field (which demangle_kernel_name
# stamps onto the recovered object, and a freshly-built KernelSignature
# never carries by default) differs.
print(f"parameters match: {sig.parameters == recovered.parameters}")
print(f"calling_convention matches: {sig.calling_convention == recovered.calling_convention}")

# Re-mangling the recovered signature is stable at the string level too.
remangled = ct.compilation.mangle_kernel_name(name, recovered)
print(f"re-mangled from the recovered signature: {remangled}")
print(f"stable at the string level: {mangled == remangled}")
```

The complete file is `06_demangle_errors_and_roundtrip.py`, included in this chapter's Complete Runnable Code.

```bash
python3 06_demangle_errors_and_roundtrip.py
```

Genuinely run:

```
demangle_kernel_name('not_a_real_mangled_name'): ValueError: `not_a_real_mangled_name` is not a mangled kernel name
demangle_kernel_name(''): ValueError: `` is not a mangled kernel name
demangle_kernel_name('my_kernel'): ValueError: `my_kernel` is not a mangled kernel name
mangled: my_kernel_Kt2_A1f32
demangled name: my_kernel
recovered == original (whole KernelSignature objects): False
original.symbol: None
recovered.symbol: 'my_kernel_Kt2_A1f32'
parameters match: True
calling_convention matches: True
re-mangled from the recovered signature: my_kernel_Kt2_A1f32
stable at the string level: True
```

Three genuinely invalid strings are rejected with a plain `ValueError`, each naming the exact string that failed. Notably, an unadorned kernel name like `"my_kernel"`, with no mangling suffix at all, is also rejected — demangling requires the real mangled form, not just any string that happens to look like a plausible function name. The round-trip itself is where the more interesting, easy-to-misread result sits: `recovered == sig` is `False`, which could be misread as "the round-trip lost information." It did not — `parameters` and `calling_convention`, the fields that actually determine compiled behavior, match exactly, and re-mangling the recovered signature reproduces the identical string. The only field that differs is `symbol`: `demangle_kernel_name` stamps the mangled string it was given onto the signature it returns, while the original, freshly-built `sig` never had its own `symbol` set at all. A failed `==` check here is evidence about which field to inspect, not evidence that round-tripping actually failed.

## Complete Runnable Code

### File: `01_mangle_kernel_name.py`

```python
import cuda.tile as ct

def array_param(dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig_f32 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f16 = ct.compilation.KernelSignature(
    [array_param(ct.float16), array_param(ct.float16), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f32_256 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(512)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# mangle_kernel_name(function_name, kernel_signature) computes the compiled
# symbol name a given (name, signature) pair would get -- entirely without
# compiling anything.
mangled_f32 = ct.compilation.mangle_kernel_name("my_kernel", sig_f32)
mangled_f16 = ct.compilation.mangle_kernel_name("my_kernel", sig_f16)
mangled_f32_256 = ct.compilation.mangle_kernel_name("my_kernel", sig_f32_256)

print(f"mangled (float32 dtype, tile_size=256): {mangled_f32}")
print(f"mangled (float16 dtype, tile_size=256): {mangled_f16}")
print(f"mangled (float32 dtype, tile_size=512): {mangled_f32_256}")
print(f"dtype change alone changes the mangled name: {mangled_f32 != mangled_f16}")
print(f"ConstantConstraint value alone changes the mangled name: {mangled_f32 != mangled_f32_256}")

# demangle_kernel_name is the inverse: given a real symbol, recover the
# original function name and an equivalent KernelSignature.
name, recovered_sig = ct.compilation.demangle_kernel_name(mangled_f32)
print(f"demangle_kernel_name({mangled_f32!r}) function name: {name}")
print(f"recovered parameters match original: {recovered_sig.parameters == sig_f32.parameters}")
```

```bash
python3 01_mangle_kernel_name.py
```

### File: `02_predicted_names_match_real_symbols.py`

```python
import cuda.tile as ct
import subprocess

def array_param(dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def my_kernel(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

sig_f32 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f16 = ct.compilation.KernelSignature(
    [array_param(ct.float16), array_param(ct.float16), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

predicted_f32 = ct.compilation.mangle_kernel_name("my_kernel", sig_f32)
predicted_f16 = ct.compilation.mangle_kernel_name("my_kernel", sig_f16)
print(f"predicted symbol (f32): {predicted_f32}")
print(f"predicted symbol (f16): {predicted_f16}")

ct.compilation.export_kernel(my_kernel, [sig_f32, sig_f16], "02_out.cubin", gpu_code="sm_90", output_format="cubin")

# Cross-check the predicted names against the real symbol table of the
# actual compiled artifact, using the standard "nm" tool -- a cubin is a
# real ELF file, and export_kernel's own compiled output is the ground
# truth here, entirely independent of mangle_kernel_name's own prediction.
result = subprocess.run(["nm", "02_out.cubin"], capture_output=True, text=True, check=True)
real_symbols = set()
for line in result.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 3:
        real_symbols.add(parts[2])

print(f"real symbols found by nm: {sorted(real_symbols)}")
print(f"predicted f32 symbol is a real symbol in the compiled cubin: {predicted_f32 in real_symbols}")
print(f"predicted f16 symbol is a real symbol in the compiled cubin: {predicted_f16 in real_symbols}")
```

```bash
python3 02_predicted_names_match_real_symbols.py
```

### File: `03_multi_signature_is_one_coherent_artifact.py`

```python
import cuda.tile as ct
import subprocess

def array_param(dtype):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

@ct.kernel
def my_kernel(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

sig_f32 = ct.compilation.KernelSignature(
    [array_param(ct.float32), array_param(ct.float32), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f16 = ct.compilation.KernelSignature(
    [array_param(ct.float16), array_param(ct.float16), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
sig_f64 = ct.compilation.KernelSignature(
    [array_param(ct.float64), array_param(ct.float64), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

# One export_kernel call, three signatures -- Chapter 2 and Chapter 5 already
# confirmed this produces three genuinely different compiled specializations.
# This checks something neither chapter checked: is the *file itself* one
# coherent artifact (a single ELF object), or three separate ones happening
# to sit in the same file?
ct.compilation.export_kernel(my_kernel, [sig_f32, sig_f16, sig_f64], "03_out.cubin", gpu_code="sm_90", output_format="cubin")

elf_magic_count = 0
data = open("03_out.cubin", "rb").read()
pos = 0
while True:
    idx = data.find(b"\x7fELF", pos)
    if idx == -1:
        break
    elf_magic_count += 1
    pos = idx + 1
print(f"raw ELF magic byte occurrences anywhere in the file: {elf_magic_count}")

readelf_result = subprocess.run(["readelf", "-h", "03_out.cubin"], capture_output=True, text=True, check=True)
header_lines = [l for l in readelf_result.stdout.splitlines() if "ELF Header" in l]
print(f"'ELF Header:' lines readelf reports for the whole file: {len(header_lines)}")

nm_result = subprocess.run(["nm", "03_out.cubin"], capture_output=True, text=True, check=True)
predicted = {
    ct.compilation.mangle_kernel_name("my_kernel", sig_f32),
    ct.compilation.mangle_kernel_name("my_kernel", sig_f16),
    ct.compilation.mangle_kernel_name("my_kernel", sig_f64),
}
real_symbols = {line.split()[2] for line in nm_result.stdout.splitlines() if len(line.split()) >= 3}
print(f"all three predicted symbols found by one nm call on the whole file: {predicted <= real_symbols}")
```

```bash
python3 03_multi_signature_is_one_coherent_artifact.py
```

### File: `04_same_filename_overwrites.py`

```python
import cuda.tile as ct
import os
import subprocess

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_add(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

@ct.kernel
def kernel_sub(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

# Calling export_kernel twice with the SAME filename -- does the second call
# overwrite the first, or accumulate into one file holding both kernels?
ct.compilation.export_kernel(kernel_add, [sig], "04_out.cubin", gpu_code="sm_90", output_format="cubin")
size_after_first = os.path.getsize("04_out.cubin")
print(f"size after exporting kernel_add alone: {size_after_first} bytes")

ct.compilation.export_kernel(kernel_sub, [sig], "04_out.cubin", gpu_code="sm_90", output_format="cubin")
size_after_second = os.path.getsize("04_out.cubin")
print(f"size after exporting kernel_sub to the same filename: {size_after_second} bytes")

nm_result = subprocess.run(["nm", "04_out.cubin"], capture_output=True, text=True, check=True)
real_symbols = {line.split()[2] for line in nm_result.stdout.splitlines() if len(line.split()) >= 3}
add_symbol = ct.compilation.mangle_kernel_name("kernel_add", sig)
sub_symbol = ct.compilation.mangle_kernel_name("kernel_sub", sig)
print(f"kernel_add's symbol still present: {add_symbol in real_symbols}")
print(f"kernel_sub's symbol present: {sub_symbol in real_symbols}")
```

```bash
python3 04_same_filename_overwrites.py
```

### File: `05_open_file_object_concatenates_but_only_first_is_visible.py`

```python
import cuda.tile as ct
import os
import subprocess

def array_param():
    return ct.compilation.ArrayConstraint(
        ct.float32, ndim=1, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

sig = ct.compilation.KernelSignature(
    [array_param(), array_param(), ct.compilation.ConstantConstraint(256)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

@ct.kernel
def kernel_add(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

@ct.kernel
def kernel_sub(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(a, (pid,), (tile_size,))
    ct.store(b, (pid,), x)

# Section 04 found that a repeated *filename* overwrites. This checks a
# different approach: keeping one file object open in "wb" mode across both
# calls, so export_kernel writes at wherever the file's current position is,
# rather than opening (and truncating) the file itself each time.
with open("05_out.cubin", "wb") as f:
    ct.compilation.export_kernel(kernel_add, [sig], f, gpu_code="sm_90", output_format="cubin")
    boundary = f.tell()
    print(f"file position after the first export_kernel call: {boundary}")
    ct.compilation.export_kernel(kernel_sub, [sig], f, gpu_code="sm_90", output_format="cubin")
    print(f"file position after the second export_kernel call: {f.tell()}")

print(f"final file size on disk: {os.path.getsize('05_out.cubin')} bytes")

add_symbol = ct.compilation.mangle_kernel_name("kernel_add", sig)
sub_symbol = ct.compilation.mangle_kernel_name("kernel_sub", sig)

# Reading the *whole* concatenated file with the standard tools:
nm_whole = subprocess.run(["nm", "05_out.cubin"], capture_output=True, text=True, check=True)
whole_symbols = {line.split()[2] for line in nm_whole.stdout.splitlines() if len(line.split()) >= 3}
print(f"kernel_add's symbol visible when nm reads the whole file: {add_symbol in whole_symbols}")
print(f"kernel_sub's symbol visible when nm reads the whole file: {sub_symbol in whole_symbols}")

# Slicing out exactly the bytes written by the second call, using the file
# position already recorded above as the real boundary between the two
# genuinely separate, complete compiled objects concatenated in this file.
with open("05_out.cubin", "rb") as f:
    f.seek(boundary)
    second_half = f.read()
with open("05_second_half.cubin", "wb") as f:
    f.write(second_half)

nm_second = subprocess.run(["nm", "05_second_half.cubin"], capture_output=True, text=True, check=True)
second_symbols = {line.split()[2] for line in nm_second.stdout.splitlines() if len(line.split()) >= 3}
print(f"kernel_sub's symbol visible once the second object is sliced out on its own: {sub_symbol in second_symbols}")
```

```bash
python3 05_open_file_object_concatenates_but_only_first_is_visible.py
```

### File: `06_demangle_errors_and_roundtrip.py`

```python
import cuda.tile as ct

# Genuinely invalid strings are rejected outright.
for bad in ["not_a_real_mangled_name", "", "my_kernel"]:
    try:
        result = ct.compilation.demangle_kernel_name(bad)
        print(f"demangle_kernel_name({bad!r}): {result}")
    except ValueError as e:
        print(f"demangle_kernel_name({bad!r}): ValueError: {e}")

# A real, complete signature and its mangled name.
sig = ct.compilation.KernelSignature(
    [ct.compilation.ArrayConstraint(ct.float32, ndim=1, index_dtype=ct.int32,
                                     stride_lower_bound_incl=None, alias_groups=[],
                                     may_alias_internally=False)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)
mangled = ct.compilation.mangle_kernel_name("my_kernel", sig)
print(f"mangled: {mangled}")

name, recovered = ct.compilation.demangle_kernel_name(mangled)
print(f"demangled name: {name}")

# Naive whole-object equality: does it hold?
print(f"recovered == original (whole KernelSignature objects): {recovered == sig}")
print(f"original.symbol: {sig.symbol!r}")
print(f"recovered.symbol: {recovered.symbol!r}")

# The actual semantic content -- parameters and calling convention -- is
# what round-trips; only the symbol field (which demangle_kernel_name
# stamps onto the recovered object, and a freshly-built KernelSignature
# never carries by default) differs.
print(f"parameters match: {sig.parameters == recovered.parameters}")
print(f"calling_convention matches: {sig.calling_convention == recovered.calling_convention}")

# Re-mangling the recovered signature is stable at the string level too.
remangled = ct.compilation.mangle_kernel_name(name, recovered)
print(f"re-mangled from the recovered signature: {remangled}")
print(f"stable at the string level: {mangled == remangled}")
```

```bash
python3 06_demangle_errors_and_roundtrip.py
```

## Chapter Summary

`ct.compilation.mangle_kernel_name` computes a real compiled symbol name entirely without compiling anything, and this chapter confirmed that prediction against the actual symbol table of a genuinely compiled cubin using `nm`, a tool with no dependency on cuTile Python at all — the predicted names were exactly the real ones. Giving `export_kernel` several signatures in one call produces a single, genuinely coherent artifact: `readelf -h` reports one ELF header for the whole file, and one `nm` call finds every specialization's symbol at once. That coherence does not extend to calling `export_kernel` repeatedly at the same file: a repeated string filename silently overwrites, discarding the earlier kernel entirely, while a repeated call on the same open file object avoids overwriting but instead produces two complete, independently-valid compiled objects concatenated back to back — genuinely different from one coherent artifact, confirmed by `nm` finding only the first kernel when reading the whole file, and finding the second only once it was sliced out and inspected on its own. Finally, `demangle_kernel_name` genuinely rejects invalid input with a clear `ValueError`, and its round-trip is stable exactly where it matters — a recovered signature's parameters and calling convention, and the re-mangled string itself — even though a naive whole-object equality check fails for a specific, identified reason: `demangle_kernel_name` stamps its input string onto the `symbol` field of the signature it returns, a field a freshly-built signature never sets.

## Self-Check Questions

1. Section 12.2 found two extra real symbols, `___NV_TILE_LAUNCH_META_DATA___` and `__nv_reservedSMEM_offset_0_alias`, that neither `mangle_kernel_name` predicted nor this book has explained. What conclusion would be unsupported if this book claimed `mangle_kernel_name` predicts "every symbol in a compiled cubin"?
2. Section 12.3 and Section 12.4.2 both end up with more than one real, valid compiled kernel object physically present in one file. What is the concrete, testable difference between these two situations that makes one "one coherent artifact" and the other not?
3. Section 12.4.1's overwrite happened with no error or warning at either `export_kernel` call. If you were combining several kernels into a shared file in a real build script, what would you need to check yourself, given that cuTile Python will not check it for you?
4. Section 12.5 found `recovered == sig` to be `False` despite the round-trip being genuinely faithful. Name one other situation from an earlier chapter in this book where two objects failing an equality check did not mean the underlying data actually differed in the way that mattered.
5. Section 12.4.2 sliced the second compiled object out of the concatenated file using the exact byte offset recorded by `f.tell()` after the first `export_kernel` call. Why was that specific offset the right place to cut, rather than, say, guessing based on the file's total size divided by two?

## Where We Go Next

This chapter confirmed that cuTile Python's real mechanism for combining more than one compiled kernel into a single coherent artifact is `export_kernel`'s own `signatures` list — the same mechanism Chapter 2 and Chapter 5 already used for specializing one kernel, now confirmed structurally coherent at the file level, and shown to be genuinely different from the two more naive approaches this chapter also tested and found wanting. Chapter 13 turns to the other half of building a complete program out of confirmed pieces: composing the tile-shape and masking primitives from Chapters 3 and 8 into a single, multi-stage kernel that processes data in more than one pass, the same `export_kernel`-only rigor applied to a genuinely larger kernel body than any single primitive this book has tested so far.

## Worked Solutions

**1.** It would be unsupported to conclude `mangle_kernel_name` predicts every symbol a compiled cubin contains. It only predicts the symbol for a specific `(function_name, KernelSignature)` pairing — the compiler-generated infrastructure symbols found in Section 12.2 have no corresponding kernel name or signature to mangle in the first place, and nothing about `mangle_kernel_name`'s documented purpose claims coverage of those.

**2.** The concrete, testable difference is what a single `nm` (or `readelf`) call over the *whole* file reports. In Section 12.3, one `nm` call over the whole file found every specialization's real symbol at once, and `readelf -h` reported exactly one ELF header for the file. In Section 12.4.2, one `nm` call over the whole file found only the first kernel's symbol — the second was invisible until the file was manually sliced at the recorded byte boundary and inspected as its own, separate file. "One coherent artifact" means standard whole-file tooling sees everything in one pass; concatenation does not meet that bar even though the bytes are all genuinely present somewhere in the file.

**3.** You would need to check, before calling `export_kernel` again with the same output path, whether that path already holds compiled output you still need — cuTile Python raises no error and prints no warning when a filename-based export overwrites an existing file, so a build script combining several kernels one at a time into the same named file would need its own explicit tracking of what has already been written, or would need to export each kernel to its own file and combine them through some other verified mechanism, such as the shared-signature-list approach Section 12.3 confirmed actually works.

**4.** Chapter 9 found `tuned is base` to be `False` for `kernel.replace_hints(...)` in Chapter 10 (a new, independent kernel object was returned, not the same one mutated) — but the more directly parallel case is Chapter 10 itself: `kernel.replace_hints(num_ctas=2)` produced a genuinely different Python object from a kernel built directly with `num_ctas=2`, yet their *compiled* output was confirmed byte-identical. In both that case and this chapter's `demangle_kernel_name` result, an object-identity or field-level difference existed alongside content that was actually, verifiably equivalent by the measure that mattered.

**5.** The file's total size divided by two only happens to equal the right boundary here because both kernels in this example compiled to exactly the same size (a coincidence of this particular worked example, not a general rule). `f.tell()` after the first `export_kernel` call reports the actual, real number of bytes that call wrote, which is the only boundary guaranteed to be correct regardless of whether two kernels compile to the same size or not — using it instead of a guess is exactly the same discipline this book has applied since Chapter 1: measure the real, reported value directly rather than assuming a convenient number.
