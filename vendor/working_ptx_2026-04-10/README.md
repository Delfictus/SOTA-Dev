# Working PTX Bundle — 2026-04-10

Snapshot of the PTX files that produce the validated TWIN result on 4LPK APO
(SII-P at quality Rank 1, q≈0.318, lining residues 68/71/101/102/103 + P-loop
13/14/15).

## Source

Captured from `/home/diddy/Desktop/Prism4D-bio/target/ptx/` on 2026-04-10.
The SHAs in `SHA256SUMS` matched the source repo's working `target/ptx/` at
the time of capture.

## Why this exists

These PTX files are byte-different from what `cargo build` currently produces
in `target/release/build/prism-gpu-*/out/ptx/`, even from the same `.cu`
source. The freshly-built versions cause a runtime SIGSEGV on 4LPK APO with
`--multi-differential`; only the bundled versions here run cleanly. Origin of
the working bundle is unknown — possibly a manual `nvcc` invocation or a
build with different flags. Captured here so the validated state is
reproducible across clones until the build pipeline is fixed.

## Usage

To run a fresh clone end-to-end:

```bash
mkdir -p target/ptx
cp vendor/working_ptx_2026-04-10/*.ptx target/ptx/
```

The runtime `find_twin_ptx` searches `target/ptx/` first, so the bundled
files take precedence over anything `cargo build` produces.

## Outstanding investigation

- Why does `nvcc` produce a non-functional PTX from the committed `.cu` source?
- Where did the working bundle originally come from?
- Should `build.rs` be patched to copy these vendored files into `target/ptx/`
  on every build, until codegen is fixed?
