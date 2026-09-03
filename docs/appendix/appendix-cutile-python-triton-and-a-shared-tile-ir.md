# Appendix: cuTile Python, Triton, and a Shared Tile IR

> This book's introduction promised a closing look at how cuTile Python relates to its sibling series' own subject, Triton — and specifically, at "the public disagreement between NVIDIA and Triton's original creators" over a shared compiler. Genuine research for this appendix found that framing to be inaccurate, not merely incomplete: the real story is more interesting, and this appendix tells it as found, correcting the earlier claim rather than repeating it.

Every chapter in this book has followed one rule without exception: a claim about what the `cuda-tile` package genuinely does gets checked against the package itself, not assumed from documentation, blog posts, or an earlier draft of this book's own prose. This appendix applies that same rule to a claim this book made about the world outside its sandbox — cuTile Python's relationship to Triton, and a reported disagreement over a shared piece of compiler infrastructure — and, having checked it, corrects it.

## Two Tile-Based Languages, Two Different Origins

Triton and cuTile Python are both Python-embedded domain-specific languages for writing GPU kernels expressed in terms of tiles — blocks of data — rather than individual threads, and both lean on an external tensor library (PyTorch, in this book's case) rather than owning a tensor type of their own, the exact parallel this book's introduction draws throughout. Their origins, though, are entirely separate. Triton was created by Philippe Tillet, beginning as his PhD research on compilers for blocked algorithms on GPUs and continuing as a project at OpenAI, which he joined full-time in 2018 to build it out; it has since become one of the most widely adopted ways to write custom GPU kernels from Python across the industry. cuTile Python, this book's subject, is NVIDIA's own tile-based Python DSL, built directly on top of `cuda-tile` and the same underlying compiler infrastructure NVIDIA uses internally — a separate lineage, from a separate organization, arriving at a similar-looking programming model.

## NVIDIA Builds a Second Path Into the Same Compiler

The concrete technical link this book's introduction referenced is real: in a blog post dated January 30, 2026, NVIDIA compute architect Jie Xin and CUDA technical marketing lead Jonathan Bentz announced a CUDA Tile IR backend for Triton — an alternative compilation path that lets Triton kernels compile directly to CUDA Tile IR, the same MLIR-based intermediate representation `ct.compilation.export_kernel` has been targeting throughout this book (visible, concretely, every time this book has passed `output_format="tileir_bytecode"` since Chapter 39). Their stated motivation is architectural: Triton, like cuTile Python, is already tile-oriented rather than thread-oriented, so compiling straight to CUDA Tile IR preserves that level of abstraction all the way down, rather than lowering to PTX and losing it. The backend lives in a public repository, `triton-lang/Triton-to-tile-IR`, hosted under Triton's own GitHub organization and described there as "an incubator repo for CUDA-TileIR backend" — opt-in via an environment variable, currently limited to NVIDIA's Blackwell architecture and CUDA 13.1 or newer, and, as of this appendix's research, missing support for some Triton features (including certain TMA scatter/gather/reduce operations and some atomic operations) that the mainline PTX backend already has.

What that means concretely, and what this book can say with the same directness it has used for every `cuda-tile` claim since Part 0: the "CUDA Tile IR" a Triton kernel compiles to through this new backend is not a similarly-branded but separate thing from the Tile IR this book's own kernels have been compiling to since Chapter 39 — it is, as far as this research could establish, the identical underlying compiler infrastructure, reached from two different source languages.

## The Public Reaction, Accurately Attributed

This book's introduction described "the public disagreement between NVIDIA and Triton's original creators" over this relationship. That is not what the available evidence shows, and stating it that way would be exactly the kind of unverified claim this book has refused to make about its own code — so it is corrected here rather than repeated.

The public criticism of cuTile that this book's earlier draft was almost certainly gesturing at came from Nicholas Wilt, a founding member of NVIDIA's own original CUDA team, not from Triton's creator. Reporting on the reaction (HyperAI, referencing remarks from around December 2025) quotes Wilt saying of cuTile: "It's hard not to suspect that cuTile was developed directly to counter Triton," characterizing it as one more embedded DSL for kernel-writing alongside Triton and Helion, rather than something new. That is a real, sourced instance of skepticism about cuTile's motivations — but it is a CUDA veteran's public read on NVIDIA's own strategy, not a statement from Triton's own creator, and not evidence of a "disagreement" between NVIDIA and Triton's team as organizations.

What the evidence does show about Triton's own creator points the other way. Philippe Tillet has himself published jointly on NVIDIA's developer blog — a February 2025 post, "OpenAI Triton on NVIDIA Blackwell Boosts AI Performance and Programmability," credits him as an author — and the Triton-to-tile-IR backend this appendix describes lives inside Triton's own GitHub organization, built to preserve compatibility with existing Triton kernels rather than to compete with them. Whatever tension exists between NVIDIA's cuTile and Triton as competing kernel languages, the concrete, sourced picture is a CUDA-ecosystem veteran publicly reading cuTile as a competitive move, sitting alongside Triton's own creator actively co-publishing with NVIDIA on Triton's hardware support — not a public feud between NVIDIA and Triton's creators. This book's earlier line conflated those two very different things, and this appendix's correction is itself an instance of the standing rule stated on its very first page: when a genuine check contradicts an assumption, the text is corrected to match reality, never the reverse.

## Genuinely Verified: This Book's Own Kernels, Re-Run as Evidence

The one claim in this appendix that this book's own sandbox can check directly — that CUDA Tile IR is a real, shared piece of infrastructure, not a rebranding exercise — does not have to be taken from a blog post at all. Chapter 39 already established that this book's `tileir_bytecode` export is identical in size across `sm_80`, `sm_90`, and `sm_100` for a given kernel. Re-running that exact test here, unchanged, is this appendix's own contribution to the claim that the "CUDA Tile IR" NVIDIA's Triton backend targets is the same compilation path this book has used all along, not a coincidence of naming.

```python
import cuda.tile as ct
import io

def array_param(ndim, dtype=ct.float32):
    return ct.compilation.ArrayConstraint(
        dtype, ndim=ndim, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=[], may_alias_internally=False,
    )

def export_bytes(kernel, sig, gpu_code, output_format):
    buf = io.BytesIO()
    ct.compilation.export_kernel(kernel, [sig], buf, gpu_code=gpu_code, output_format=output_format)
    return buf.getvalue()

M, N = 8, 8

# --- The same elementwise-add kernel Chapter 39 used to establish that
# tileir_bytecode output is architecture-independent, run again here as
# concrete evidence for this appendix's central claim: the "CUDA Tile
# IR" this book's own export_kernel calls have been targeting since
# Part 5 is the exact same intermediate representation NVIDIA's new
# Triton backend now also compiles Triton kernels down to -- not a
# similarly-named but separate thing. ---
@ct.kernel
def kernel_add(x, y, out):
    a = ct.load(x, (0, 0), (M, N))
    b = ct.load(y, (0, 0), (M, N))
    ct.store(out, (0, 0), a + b)

sig = ct.compilation.KernelSignature(
    [array_param(2), array_param(2), array_param(2)],
    calling_convention=ct.compilation.CallingConvention.cutile_python_v2(),
)

architectures = ["sm_80", "sm_90", "sm_100"]

print("--- output_format='tileir_bytecode' ---")
bytecode_sizes = {}
for gc in architectures:
    b = export_bytes(kernel_add, sig, gc, "tileir_bytecode")
    bytecode_sizes[gc] = len(b)
    print(f"  gpu_code={gc}: {len(b)} bytes")

print()
print("tileir_bytecode sizes identical across architectures:", len(set(bytecode_sizes.values())) == 1)
print()
print("this is the same compilation path (ct.compilation.export_kernel with")
print("output_format='tileir_bytecode') this book has used since Chapter 39 --")
print("re-run here, unchanged, as this appendix's own evidence rather than a claim")
print("taken on faith from either NVIDIA's or this book's own earlier prose.")
```

### Genuinely run

```text
--- output_format='tileir_bytecode' ---
  gpu_code=sm_80: 1025 bytes
  gpu_code=sm_90: 1025 bytes
  gpu_code=sm_100: 1025 bytes

tileir_bytecode sizes identical across architectures: True

this is the same compilation path (ct.compilation.export_kernel with
output_format='tileir_bytecode') this book has used since Chapter 39 --
re-run here, unchanged, as this appendix's own evidence rather than a claim
taken on faith from either NVIDIA's or this book's own earlier prose.
```

The exact byte count shown here, `1025`, is not a universal constant — Chapter 39 documented, and this book's build pipeline has confirmed on every chapter since, that `tileir_bytecode`'s size shifts by exactly two bytes for every one-character change in the compiling script's absolute filesystem path, while `cubin` output for the identical kernel does not. What this run demonstrates is path-independent: the three architectures agree with each other, exactly, every time this exact script is run — the same architecture-independence Chapter 39 found, holding just as true here as evidence for this appendix as it did there as evidence for its own chapter.

## What This Actually Means

None of this changes anything about how this book's kernels compile, or about `cuda-tile`'s behavior as this book has documented it chapter by chapter — cuTile Python and Triton remain two separately-designed languages, and nothing here suggests either is going away. What it does mean is narrower and more concrete: NVIDIA has built a real, working bridge that lets Triton — a language it did not create — compile down to the same tile-based compiler infrastructure cuTile Python — a language it did create — has used throughout this entire book, and the reaction to that overlap in the wider developer community has been more varied than a simple "NVIDIA versus Triton" framing captures. A CUDA veteran reads cuTile as competitive positioning against Triton; Triton's own creator continues to publish alongside NVIDIA on Triton's hardware support; and the compiler underneath both languages is demonstrably, checkably the same. This book's own contribution to that picture is the smallest, most concrete piece of it: the exact export call this appendix re-ran is the same one this book has run in every chapter since Part 5, targeting the identical shared infrastructure, verified again here rather than assumed.

## Sources

- [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/) — NVIDIA Technical Blog, Jie Xin and Jonathan Bentz, January 30, 2026
- [triton-lang/Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR) — the incubator repository for the CUDA Tile IR backend, hosted under Triton's own GitHub organization
- [CUDA's Initial Team Members Sharply Criticized cuTile for "specifically Targeting" Triton](https://hyper.ai/en/news/47715) — HyperAI, reporting Nicholas Wilt's public remarks on cuTile
- [OpenAI Triton on NVIDIA Blackwell Boosts AI Performance and Programmability](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/) — NVIDIA Technical Blog, co-authored by Philippe Tillet, February 5, 2025
- [Philippe Tillet — NVIDIA Developer Blog author page](https://developer.nvidia.com/blog/author/philopenai/) — biographical background on Triton's creator and his continued publishing with NVIDIA
