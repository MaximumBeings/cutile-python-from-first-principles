# Getting Started

This book targets **cuTile Python 1.5.0+** (the `cuda-tile` package on PyPI), running under Python 3.10 through 3.14. Every listing's *definition* and *ahead-of-time compilation* is genuinely run against the real package in this book's build environment. Actually *launching* a kernel needs a real NVIDIA GPU and driver, which this environment does not have — this book flags that explicitly wherever it applies (see [How this book verifies its claims](#how-this-book-verifies-its-claims) below).

## Environment

Install cuTile Python from PyPI. The `[tileiras]` extra bundles the ahead-of-time compiler as a set of pure-Python wheels, so it installs cleanly with no CUDA Toolkit and no NVIDIA driver present at all:

```bash
pip install "cuda-tile[tileiras]"
```

If you already have CUDA Toolkit 13.1+ installed system-wide, the compiler is picked up from there instead and the plain package is enough:

```bash
pip install cuda-tile
```

Confirm the install and check the version:

```bash
python3 -c "import cuda.tile as ct; print(ct.__version__)"
```

This book was written and verified against:

```text
cuda-tile 1.5.0
Python 3.11.15
```

with `pip install "cuda-tile[tileiras]"` genuinely pulling in `cuda-toolkit 13.3.1`, `nvidia-cuda-nvcc 13.3.73`, `nvidia-cuda-tileiras 13.3.36`, `nvidia-cuda-crt 13.3.73`, `nvidia-cuda-runtime 13.3.29`, `nvidia-nvvm 13.3.73`, and `nvidia-nvjitlink 13.3.33` — no driver, no GPU, and no separate system CUDA Toolkit install required for any of it.

!!! warning "The Python import path is not what the package name suggests"
    The PyPI package is `cuda-tile`, but the module it installs is `cuda.tile`, not `cuda_tile`. `import cuda_tile as ct` genuinely fails with `ModuleNotFoundError` in this environment; `import cuda.tile as ct` is the real, working import used throughout this book.

## Verify the toolchain — genuinely run, no GPU required

```python
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

print(type(vector_add))
```

Genuinely run in this book's environment:

```text
<class 'cuda.tile._execution.kernel'>
```

Defining a `@ct.kernel` function costs nothing GPU-related — it just builds a Python-level kernel object. If you see the same class name, your install is working and you're ready for ahead-of-time compilation, the next real step this book relies on.

## Ahead-of-time compilation — the part that works with no GPU at all

cuTile Python resolves the real GPU driver only when a kernel is *launched*, not when it's compiled for a named target architecture. That makes `ct.compilation.export_kernel` genuinely runnable in a driver-free sandbox — this book's equivalent of the CUDA edition's plain `nvcc` compile, but able to target multiple architectures from the same environment:

```python
import io

sig = ct.compilation.KernelSignature(
    [
        ct.compilation.ArrayConstraint(
            ct.float32, ndim=1, index_dtype=ct.int32,
            stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        ),
        ct.compilation.ArrayConstraint(
            ct.float32, ndim=1, index_dtype=ct.int32,
            stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        ),
        ct.compilation.ArrayConstraint(
            ct.float32, ndim=1, index_dtype=ct.int32,
            stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
        ),
        ct.compilation.ConstantConstraint(256),
    ],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

for arch in ("sm_80", "sm_90", "sm_100"):
    buf = io.BytesIO()
    ct.compilation.export_kernel(vector_add, [sig], buf, gpu_code=arch, output_format="cubin")
    print(arch, "cubin bytes:", len(buf.getvalue()))
```

Genuinely run in this book's environment:

```text
sm_80 cubin bytes: 23728
sm_90 cubin bytes: 24736
sm_100 cubin bytes: 30760
```

Three real, differently-sized cubin binaries, one per target architecture, produced by the same Python function and the same `tileiras` compiler that ships with the `[tileiras]` extra — with no NVIDIA GPU, no driver, and no `nvidia-smi` anywhere in this environment. `output_format="tileir_bytecode"` genuinely works the same way if you want NVIDIA's intermediate representation instead of a finished binary.

## GPU toolchain (needed to actually launch a kernel)

Everything above — defining a kernel, building a `KernelSignature`, exporting real cubin or Tile IR bytecode for a named architecture — is genuine, driver-free evidence. Actually *launching* a kernel with `ct.launch(stream, grid, kernel, kernel_args)` needs a real CUDA stream, which in turn needs a real NVIDIA GPU, its driver, and typically a library like CuPy or PyTorch to construct that stream in the first place. This book's build environment has none of that, so no `ct.launch()` call anywhere in this book was actually attempted against a live device — that step is explicitly out of reach here, not attempted-and-hidden.

To run the kernels this book compiles, you'll need:

- An NVIDIA GPU with compute capability 8.x, 9.x, 10.x, 11.x, or 12.x — as of the `cuda-tile` 1.5.0 README, the bundled `tileiras` compiler itself only targets Ampere/Ada and Blackwell GPUs; Hopper support is described there as coming in a future release, so check the installed `nvidia-cuda-tileiras` package's own notes if you're targeting Hopper specifically
- NVIDIA Driver r580 or later
- CuPy or PyTorch with CUDA support, to allocate real device arrays and obtain a real `cudaStream_t`-backed stream to pass to `ct.launch`

```bash
nvidia-smi      # confirms a real driver + GPU are present
```

!!! note "A note on where this book's cuTile code was written"
    This book is authored and its ahead-of-time compilation steps are verified in a sandboxed environment with `pip install "cuda-tile[tileiras]"` genuinely succeeding and no NVIDIA GPU, driver, or `nvidia-smi` present at all. Every kernel definition and every `ct.compilation.export_kernel` call shown with real byte counts was genuinely run there. Any chapter or example that would require `ct.launch` against a live device is marked **UNVERIFIED — pending a real-GPU test pass** until independently run on real hardware and confirmed. If you're reading this after that pass, those tags should be gone and replaced with real captured output; if you still see one, that chapter hasn't been verified on hardware yet.

## How this book verifies its claims

Every "Worked Example" in this book that defines a kernel or compiles it ahead-of-time embeds real, unedited output from the actual `cuda-tile` package run against the exact listing shown — nothing is hand-typed or guessed. Where a chapter makes a numeric or behavioral claim that only a real GPU could produce, it's explicitly tagged unverified rather than invented, exactly as described above, until run on real hardware and folded back in.
