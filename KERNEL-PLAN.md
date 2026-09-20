# KERNEL-PLAN — SM120/SM121 DSV4 dual-cache sparse-MLA prefill at main topk 512

Branch: `sm121-dsv4-prefill-topk512` (based on `v0.6.18`, the version pinned by
the `vllm/vllm-openai:deepseekv4-flash-vision` day-0 image, vLLM
0.28.1rc1.dev137). Target hardware: DGX Spark GB10, sm_121a, TP=2.
Date: 2026-09-20. Phase K1 research record; phases K2–K4 append results.

## 1. The blocker

DeepSeek-V4-Flash-Vision-Exp widens every prefill sliding-window index row
from 128 to `128 + vision_max_n_token (384) = 512`
(vLLM `models/deepseek_v4/attention.py:209-214`, `sparse_swa.py:442`). The
flashinfer SM120 dual-cache prefill (compressed + SWA caches, DSV4-only) is
the default backend on SM12x, and in 0.6.18 it compiles only for main
topk = 128:

- `csrc/sparse_mla_sm120_prefill.cu:394` — `if (topk != 128) return false;`
  in `dispatch_dsv4_dual`'s CM (runtime-length) family, `TK` hardcoded 128.
- Same file, line 324 — the FULLTILE family additionally requires
  `topk == 128 && topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr`.
  The VL call passes `swa_topk_lens`, so it needs the CM family regardless.

Boot dies from `csrc/sparse_mla_sm120.cu:271`:
`Unsupported sparse-MLA prefill configuration: model=DSV4 num_heads=32
topk=512 page_block_size=64 topk_extra=512 extra_page_block_size=64`.
Text passes because `max_image_tokens == 0` keeps main topk at 128.
The alternate in-vLLM route (`FLASHMLA_SPARSE_DSV4`) dies one kernel later:
upstream FlashMLA `flash_mla_sparse_fwd` is SM90a/SM100f only
(`csrc/api/sparse_fwd.h:116` in vLLM), no SM121a cubins. Root-cause evidence:
dspark-fork `benchmarks/20260920-upstream-vision-phase4-vl-unblock.md`.

## 2. Upstream status: the fix already exists on main — this is a backport

flashinfer main resolved this on 2026-09-17 in
`eb5f05be refactor(sparse-mla): unify SM120 execution, calibration and DSv4.1
support (#5197)`, which rewrote the SM120 sparse-MLA subsystem around an
execution planner where **topk is a runtime kernel argument**:

- `include/flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_mg.cuh:44`
  — "topk (the main indices row width) is runtime via cold.topk"; the dual
  mainloop iterates `ceil(topk_len/64) + ceil(topk_len_extra/64)` tiles with
  per-lane length masking; sink handled once in the epilogue.
- `csrc/sparse_mla_sm120/attention_resolve.cu:185` — prefill requires only
  `topk % 64 == 0`; `:203-208` — dual MG requires DSV4 + dual and forces
  `NumericRoute::QkBF16PvFP8` (BF16 QK, FP8 PV) **for all dual shapes, any
  topk**. heads 32 dual MG is in the instantiated set
  (`execution/attention_plan.h:36-47`).

No release contains it: v0.6.9 and v0.7.0rc3 predate eb5f05be, and
`git tag --contains eb5f05be` is empty. The image pins 0.6.18 (2026-08-28),
which sits after `#4380` (single-cache prefill topk 192/256) but keeps the
dual CM family at TK=128. Backporting the planner rewrite is out of scope;
the surgical fix below reaches the same envelope on the 0.6.18 codebase.

## 3. Why TK=512 dual fits: the smem/tile math

Phase 4 inferred from the single-cache BF16→FP8 switch at topk ≥ 512 that a
BF16 dual tile at 512 likely does not fit SM120 smem. Reading the 0.6.18
kernel refutes that: in `prefill_mg_impl`
(`include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh:655`),
`TOPK` is a template parameter used only for

1. the runtime-length clamp (`topk_len = min(topk_len, TOPK)`, line 692-696),
2. the index-row stride (`idx_base = indices + s_i * TOPK`, line 739),
3. the compile-time `NI = TOPK/BI` used **only** by the FULLTILE variant's
   main/extra tile split (line 756); the CM variant's split is the runtime
   `main_ni = ceil(topk_len/BI)`.

The KV pipeline is a 2-slot double buffer of BI=64-entry tiles
(`buf = ti & 1`, `io_bulk_gather_tile`), the mainloop is `#pragma unroll 1`,
and registers are pinned by `setmaxnreg` (232 math / 32 IO). None of smem,
register pressure, or code size scales with TOPK — only the trip count does
(2 → 8 main tiles at 512; plus ≤ 8 extra tiles at extra_topk 512).

`SmemLayoutMG<DSV4, BF16>` (common/smem_layout.cuh) has no TOPK parameter.
Exact budget (HPB=16, BI=64, N_MATH_WARPS=8, N_HG=2, KV_SMEM_STRIDE=464,
SCALE_BYTES_PER_TOKEN=8, N_V_CHUNKS=7):

| buffer | bytes |
| --- | --- |
| Q nope BF16, 2 groups (16×520×2 B each) | 33,280 |
| KV double buffer, 2 × 64×464 | 59,392 |
| KV scale buffers, 2 × 64×8 | 1,024 |
| reduce (2×8×16×4) | 1,024 |
| m + l (2×16×4 each) | 256 |
| w_head_sc_all (2×7×16×4) | 896 |
| w_fp8, 2 parities × 2 groups × 16×80 (q_rope aliases this, needs 4,096) | 5,120 |
| mbarrier | 16 |
| **TOTAL** | **101,008 B = 98.64 KiB** |

The file's own `static_assert(TOTAL <= 101376)` passes; SM121 allows 227 KiB
opt-in dynamic smem per block — 2.25× headroom. A TK=512 CM instantiation is
the same kernel at the same 101,008 B; nothing in the layout moves.

Numerics choice: BF16 QK (the existing CM=BF16 dual mode), matching what
upstream main forces for every dual shape (`QkBF16PvFP8`) and matching the
production TK=128 dual path. The single-cache FP8 switch at 512 is a
throughput heuristic (prefill dispatch comment: "Small K-loop: BF16 QK skips
the FP8 Q-quantize prologue. Larger K amortises FP8's higher Tensor-Core
throughput"), not a capacity requirement, and does not apply to dual.

## 4. The change (K2 scope)

One file, dispatch-only; the kernel template is untouched:

`csrc/sparse_mla_sm120_prefill.cu`, `dispatch_dsv4_dual`:

- Replace `if (topk != 128) return false;` with a topk switch instantiating
  the CM family at TK ∈ {128, 192, 256, 512}, BF16, all NH ∈ {8,16,32,64,128},
  both extra page sizes (64, 2) — mirroring the single-cache BF16 set.
  Gate: `topk % 64 == 0` is implied by the switch; anything else still
  returns false and hits the existing loud error.
- FULLTILE stays TK=128 (VL passes runtime lengths; text is unaffected).
- Decode is untouched (VL decode rows stay 128-wide; only prefill widens).

Compile-cost note: dual CM instantiations grow 10 → 40 kernels. This module
is AOT-registered (`flashinfer/aot.py:852`), so the cost is one-time at
wheel/jit-cache build. If nvcc time becomes a problem, trim to {128, 512}.

No Python-side change: `flashinfer/mla/_sparse_mla_sm120.py` has no prefill
topk gate (decode tables only), and the orchestrator
(`csrc/sparse_mla_sm120.cu`) checks only `topk > 0`.

## 5. Correctness oracle: the vcruz305 slicing reference

`patch_dsv4_vl_sm120_wide_swa.py` (vcruz305,
DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe, Apache-2.0) solves the
same blocker in Python: slice each 512-wide row into four 128-wide slices,
run the stock TK=128 dual kernel per slice — compressed segment and attention
sink ride on slice 0 so they are counted once — and merge the per-slice
`(out_i, lse_i)` partials exactly:

```
m  = max(m, lse_i)
a  = exp(m_old - m)          (0 while m == -inf)
b  = exp(lse_i  - m)         (0 for empty slices: lse forced to -inf)
acc = acc·a + out_i·b        (out_i is the slice-normalized output)
den = den·a + b
out = acc / den              == one softmax over the full row + compressed + sink
```

Reported ~1% relative noise on widened rows vs torch reference on packed fp8
cache. This defines the acceptance ceiling and the fallback design: if the
kernel extension produced wrong numerics we could not fix, the same slicing
moves inside the dispatch (loop the kernel per 128-wide slice view, merge in
a second pass) rather than shipping anything approximate. It is also the
second oracle in validation: the extended kernel must agree with the slicing
path at least as tightly as the slicing path agrees with the reference.

## 6. Validation plan (K3, rig GPU window ≤ 1h)

Harness: extend `tests/attention/test_sparse_mla_sm120.py`
(`test_sparse_mla_sm120_prefill_dsv4_dual`), which already builds packed fp8
DSV4 caches (`quantize_kv_dsv4`), dequantizes, and checks against
`_ref_sparse_attn` (dense SDPA over gathered KV, supports `topk_length` and
sink) at atol=rtol=5e-2. New coverage:

- main topk=512 × extra_topk=512 × extra_pbs ∈ {64, 2}, heads ∈ {8, 16, 32,
  64, 128} — the VL shape at heads=32 is the must-pass.
- main topk=192/256 dual (new instantiations get coverage too).
- Runtime main `topk_length` truncation at topk=512 (VL semantics:
  `swa_topk_lens` ≤ 512), including len=0 rows (sink-only output) and
  non-tile-aligned lens.
- Sink on/off; report max abs/rel error, not only assert_close.

Acceptance: agreement with the torch reference well inside the slicing
reference's 1% relative bound (expected: same error class as the stock
TK=128 dual test, BF16-output rounding); and agreement with the slicing
oracle within its own noise. Any regression at TK=128 (text path) fails the
phase.

## 7. Build plan (K2)

- Fork `msuiche/flashinfer`, branch `sm121-dsv4-prefill-topk512` off
  `v0.6.18`. Dispatch edit + test edit only.
- Build on the rig head in a scratch dir (no production touch):
  `FLASHINFER_CUDA_ARCH_LIST="12.1a"` wheel build of `flashinfer-python`
  0.6.18+patch, plus the matching `flashinfer-jit-cache` AOT build so the
  patched module is precompiled (the image ships nvcc+ninja, so a JIT
  fallback would work but would compile at container start — avoid).
- Fail-closed, following the dspark-fork stage pattern: the image build
  stage verifies the patch is present in the installed tree (source hash /
  marker, like `scripts/check-patch3.sh`), and the launcher runs a GPU smoke
  before serving: dual prefill at topk=512 on sm_121a returns finite output
  and the TK=128 path still dispatches. Either failure aborts the boot
  loudly — no silent fallback to a broken VL.

## 8. PR outline

- Upstream main needs no fix (eb5f05be); the PR story is the 0.6.x backport
  for the day-0 DeepSeek-V4 VL image, filed from
  `msuiche/flashinfer:sm121-dsv4-prefill-topk512`. Title: "sm120: DSV4
  dual-cache prefill topk 192/256/512 (0.6.x backport)". Body: the blocker,
  the TOPK-is-trip-count evidence, the smem table from §3, the numerics
  decision (BF16 QK = main's `QkBF16PvFP8` dual route), validation numbers
  from K3/K4, and a pointer to eb5f05be as the main-branch resolution.
  Credit vcruz305 for the slicing reference used as validation oracle and
  fallback design (Apache-2.0).
- If upstream declines a 0.6.x patch release, the branch still ships as the
  dspark image's pinned flashinfer source.

## 9. Expected cost, stated plainly

A VL prefill row attends 512 main + 512 extra entries vs text's 128 + 512:
the dual kernel's main-cache tile count goes 2 → 8, so VL prefill rows cost
~1.6× the tile work of text rows through this kernel (16 tiles vs 10). That
is inherent to the widened attention, not overhead of the fix. No perf
regression at TK=128: same template instantiation as today.

## 10. Results

### K2 (2026-09-20, CPU only — no production touch)

Implementation: commit `e9ef4835` on `sm121-dsv4-prefill-topk512` — the
dispatch extension from §4 (one file) plus the §6 tests. Nothing else in the
tree changed; the kernel template is untouched.

Build (rig head, inside the day-0 image toolchain, CPU-only):

- `flashinfer_python-0.6.18-py3-none-any.whl` (18.4 MB, version string
  exactly `0.6.18`, so the flashinfer-jit-cache `startswith` version pin
  stays satisfied). Verified the wheel's installed
  `data/csrc/sparse_mla_sm120_prefill.cu` carries `DISPATCH_BY_TK_PBSX`.
- JIT-precompiled the `sparse_mla_sm120` module for `12.1a`
  (`FLASHINFER_CUDA_ARCH_LIST="12.1a"`, ninja 6/6). `strings` on the .so
  confirms all ten TK=512 dual instantiations: NH ∈ {8,16,32,64,128} ×
  extra-page ∈ {64,2}, DSV4/BF16 (`ModelType1E ComputeMode1E`), e.g.
  `sparse_mla_prefill_mg_dual_kernel<ModelType1, ComputeMode1, 32, 512, 64, 64, 2>`.

Integration trap found and handled: with flashinfer-jit-cache installed,
`JitSpecNvcc.try_load()` (`flashinfer/jit/core.py:396-410`) returns the AOT
`.so` **unconditionally** when it exists — patched sources would silently
load the stock TK=128-only module. The image stage therefore swaps the AOT
artifact
(`/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so`,
stock backed up as `sparse_mla_sm120.so.stock-k2`) and fails the build
unless version, dispatch marker, the TK=512 NH=32 kernel symbol, and the
post-swap sha256 all check out
(dspark-fork `recipe/upstream-vision/verify-k2-flashinfer-topk512.py`).

Artifacts:

- Test image `vllm-upstream-vision:k2-topk512`
  (`recipe/upstream-vision/Dockerfile.k2-flashinfer-topk512`), image id
  `d2c6f265d5a5` on head and worker (docker save | ssh docker load; ids
  compared — the repo's build-once-and-ship rule). Verifier prints
  `K2_FLASHINFER_TOPK512_PATCH_OK` on both nodes; AOT sha256
  `b504d7631fca…` on both.
- Scratch on head: `~/flashinfer-sm121/{src,out,fi-ws}` (fork checkout,
  wheel, prebuilt module). K3 script staged at
  `~/flashinfer-sm121/dsv4_dual_topk512_validation.py` (repo copy:
  dspark-fork `benchmarks/dsv4_dual_topk512_validation.py`).

K3/K4 readiness: everything below needs a GPU window (production parked on
both nodes, restored after). K3: pytest subset
(`-k "prefill_dsv4 or dsv4_public_api"` incl. the new wide-main dual tests)
+ the validation script (extended kernel vs slicing oracle vs torch
reference, TK=128 regression). K4: VL boot via
`scripts/start-upstream-vision-test.sh` with
`COMPOSE_FILE=recipe/upstream-vision/docker-compose.upstream-vision-vl-nospec.yml`,
`UPSTREAM_VISION_IMAGE=vllm-upstream-vision:k2-topk512`, Vision-Exp snapshot
`6821d6ad`, then `benchmarks/vision_smoke_probe.py`.

### K3 (2026-09-20 window, head GPU, sm_121a)

Window opened 21:09:49 UTC (production parked on both nodes). Two
gate-keeping artifacts, both in the `vllm-upstream-vision:k2-topk512` image:

**pytest** — `/fi-tests/attention/test_sparse_mla_sm120.py -k "prefill_dsv4
or dsv4_public_api"`: **165 passed, 0 failed** (39.35s). Covers the 20 new
wide-main dual cases (topk 192/256/512 x extra 512 x extra-pbs 64/2 x heads
{8,16,32,64,128}), the new 512 truncation cases (edge lens 0/1/63/64/65/128/
133/384/511/512, sink on/off), and the whole pre-existing dsv4 prefill +
dual suite (regression). One test-authoring fix during the window: the
kernel family reports LSE=-1e30 for a fully empty row
(`softmax_lse` in common/online_softmax.cuh; the pre-existing dsv3_2
zero-length test asserts the same sentinel), where the dense reference
produces -inf — the new truncation test now expects the kernel convention
(commit `22d670da`). Outputs already agreed (both 0).

**validation script** (`dsv4_dual_topk512_validation.py`, verbatim output):

```
== sink=on (topk=512, extra=512) ==
 A (extended kernel) vs C (torch reference):
  out   max_abs=0.001099 max_rel=0.105286 inf_agree=True worst(ref=-0.019531 got=-0.020630) elem(rel>1%&abs>2e-3)=0
  lse   max_abs=0.000002 max_rel=0.000000 inf_agree=True
 B (slicing oracle) vs C (torch reference):
  out   max_abs=0.001099 max_rel=0.105286 inf_agree=True elem(rel>1%&abs>2e-3)=0
  lse   max_abs=0.000002
 A vs B (kernel vs oracle):
  out   max_abs=0.000115 max_rel=0.008574 inf_agree=True elem(rel>1%&abs>2e-3)=0
  lse   max_abs=0.000002
== sink=off (topk=512, extra=512) ==
 A vs C:  out max_abs=0.001221 max_rel=0.112152 elem(rel>1%&abs>2e-3)=0 ; lse max_abs=0.000002
 B vs C:  out max_abs=0.001221 max_rel=0.112152 elem(rel>1%&abs>2e-3)=0 ; lse max_abs=0.000002
 A vs B:  out max_abs=0.000117 max_rel=0.008209 elem(rel>1%&abs>2e-3)=0 ; lse max_abs=0.000001
== regression: TK=128 dual (text shape) vs reference ==
  out   max_abs=0.001099 max_rel=0.103760 elem(rel>1%&abs>2e-3)=0 ; lse max_abs=0.000002
VERDICT: PASS
```

Reading: the extended kernel and the slicing oracle agree with each other
to **0.86% max relative** (inside the slicing reference's own 1% bound), and
each sits at max_abs ~1.1e-3 from the fp32 reference with LSE exact to 2e-6.
The ~0.10-0.11 "max_rel" against the reference is BF16 output rounding on
near-zero elements (worst element |ref|=0.019, err 1.1e-3) and is
**identical on the stock TK=128 text shape** (max_abs=0.001099, same
elements) — the extension adds no error over the kernel family's own
baseline error class. Zero elements fail both rel>1% and abs>2e-3 anywhere.

**K3 verdict: PASS.** The TK=512 dual kernel is numerically the TK=128
kernel at a longer trip count, as designed.
