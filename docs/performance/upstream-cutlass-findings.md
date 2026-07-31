# Draft findings for CUTLASS upstream — for Kohaku to review

**Nothing here has been filed.** These are drafts for review; filing an issue against
CUTLASS is outward-facing and is Kohaku's call, not ours.

Environment for both: CUTLASS **3.9.2**, CUDA **13.2** (also reproduced on 12.9),
RTX 5090 (sm_120), `nvcc -std=c++17 -arch=sm_120a --expt-relaxed-constexpr`.
`-Wno-deprecated-declarations` is required because CUDA 13.2 deprecated
`long4`/`ulong4`/`double4` in favour of the `*_16a`/`*_32a` spellings and 3.9.2
still uses the old names — warnings only, no errors.

---

## Finding 1 (high confidence, small, self-contained)

### `device_memory::copy` error reporter streams a device pointer as a C string

`tools/util/include/cutlass/util/device_memory.h`, in the `cudaMemcpy` failure path:

```cpp
os << "cutlass::device_memory::copy: cudaMemcpy() failed: "
   << "dst=" << dst << ", src=" << src        // <-- here
```

When `T` is a byte type, `dst`/`src` are `unsigned char*`, so
`std::ostream::operator<<` selects the **C-string** overload and calls `strlen` on a
*device* pointer. The process segfaults while formatting the message, so the
`cuda_exception` on the next line is never constructed and the underlying CUDA
error is never reported.

Observed as an opaque `SIGSEGV` from example
`79d_blackwell_geforce_nvfp4_grouped_gemm`, with this backtrace:

```
#0 __strlen_avx2
#1 std::operator<<(ostream&, unsigned char const*)
#2 cutlass::device_memory::copy<unsigned char>
#3 cutlass::device_memory::copy_to_host<unsigned char>
#4 HostTensor<float_e2m1_t, PackedVectorLayout>::sync_host()
#5 verify()
```

**Suggested fix:** cast to `void const*` before streaming. One line each for `dst`
and `src`.

**Why it is worth reporting:** it converts every `cudaMemcpy` failure on a byte-typed
tensor into a crash with no diagnostic. It cost roughly an hour of misattribution
here — the failure looks like a kernel bug and is not one.

### Related, and probably the actual first-order bug

The `cudaMemcpy` that fails in the trace above is copying a **4-bit**
(`float_e2m1_t`) `HostTensor`, and arrives at `copy<unsigned char>`.
`device_memory.h` computes `bytes = count * sizeof_bits<T>::value / 8` with
`T = unsigned char`, i.e. 8 bits, so a 4-bit element count yields **twice** the
allocated byte count and the copy runs off the end of the buffer.

We did not chase this to a definitive root cause because it does not affect our
use case: it keys on **sub-byte** element types, and MXFP8 has none. Confirmed by
two data points — 79d (4-bit `ElementD`) crashes; 79c (`ElementD = bfloat16_t`)
passes; and porting 79d to 8-bit `float_e4m3_t` output turns the segfault into a
properly reported `cuda_exception`.

---

## Finding 2 (now has a reproducible mechanism — see the addendum at the end)

### SM120 pointer-array (grouped) tags are absent from the MX block-scaled `SfVectorSize` whitelist

`include/cutlass/gemm/collective/builders/sm1xx_common.inl` static-asserts the
scale-factor vector size against a list of kernel schedule tags. In the
`MX_F4F6F8` branch the SM100 entries include the grouped one:

```cpp
|| (SfVectorSize == 32 && cute::is_base_of_v<KernelSchedulePtrArrayBlockScaledGemmSm100, BuilderScheduleTag>)
```

but the SM120 entries list only `KernelScheduleBlockScaledGemmSm120` and its sparse
counterpart — **there is no SM120 pointer-array entry.** Since
`KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120` derives from the
generic `KernelPtrArrayTmaWarpSpecializedCooperative` and *not* from
`KernelScheduleBlockScaledGemmSm120`, it satisfies no clause, and a grouped MXFP8
GEMM fails to compile with:

```
static assertion failed with "Incorrect SfVectorSize for MX_F4F6F8 is deduced."
```

There are three such sites (`SfVectorSizeA` once, `SfVectorSize` twice).

**What initially suggested an omission:** the SM120 block-scaled builder
(`sm120_blockscaled_mma_builder.inl`) has explicit `IsGroupedGemmKernel` handling
and emits exactly those PtrArray tags; `sm120_blockscaled_mma_array_tma.hpp` is a
full 1161-line mainloop with no "not implemented" guard; and NVFP4 grouped works
on the same builder.

**Why we are not proposing the patch as a fix.** Adding
`is_base_of_v<KernelPtrArrayTmaWarpSpecialized{Cooperative,Pingpong}>` clauses does
make a grouped MXFP8 GEMM compile, and it instantiates the whole intended stack —

- `MainloopSm120ArrayTmaWarpSpecializedBlockScaled`
- `SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>`
- `LinCombBlockScaleFactor<32, float_e4m3_t, float, float_ue8m0_t, RowMajor, ...>`

— but the kernel then **traps at runtime**. `can_implement()` and `initialize()`
both return success, shared memory is set to 82944 bytes (within this card's
~100 KB opt-in limit), and `gemm.run()` reports `unspecified launch failure`.
`compute-sanitizer --tool memcheck` reports **no** invalid accesses, only
`Trace/breakpoint trap` — i.e. a device-side `__trap()`/assert rather than memory
corruption.

So one of the following is true, and we cannot yet distinguish them:

1. the whitelist is an omission *and* there is a second, separate gap; or
2. the whitelist is an intentional guard over an unfinished path, and removing it
   is exactly the mistake it exists to prevent; or
3. our port is simply misconfigured somewhere we have not found.

**The question to ask upstream is therefore "is grouped MX block-scaled intended to
be supported on SM120 in 3.9.2, and if so is the whitelist omission a bug?"** —
not "please apply this patch". Repro details: tile shape is effectively pinned to
`128x128x128` because both `128x128x64` and `64x128x128` fail
`copy_traits_sm90_tma.hpp` with *"TMA requires CTA_Tile and SLayout top-level size
equivalence"* (the MX scale-factor atom constrains the tile), and forcing
`StageCount<2>` instead of `StageCountAutoCarveout` does not change the trap.

**Cheap next step before asking anyone:** check whether CUTLASS 4.x supports SM120
grouped MX. Our local checkout is 3.9.2 and predates CuTeDSL; a 4.x checkout would
also give the reference DSL kernels the pip wheel does not ship.

---

## Addendum to Finding 2 — the mechanism, found after the section above was written

The runtime trap is **not** evidence that grouped MX is unimplemented. Bisected on
an RTX 5090 (170 SMs):

| tiles | result |
|---|---|
| 168 | passes |
| 172 | `unspecified launch failure` |

**The boundary is exactly the SM count**, i.e. the kernel works for a single wave of
CTAs and fails as soon as the persistent scheduler must loop. It is not group count
and not problem size: 40 groups x 4 tiles passes, while 2 groups x 512 tiles fails.
The **cooperative and pingpong schedules break at the identical boundary**, so it is
not specific to one scheduler.

With `-lineinfo`, `compute-sanitizer` attributes the trap to
`sm90_epilogue_array_tma_warpspecialized.hpp:604` — the epilogue's C-matrix load
`consumer_wait`. Consistent with that, **`beta != 0` traps even inside one wave**,
and `beta = 0` is required for any configuration to run. `compute-sanitizer
--tool memcheck` reports no invalid accesses at all, only `Trace/breakpoint trap`.
Shared memory is not implicated (82944 bytes against a ~100 KB opt-in limit) and
`StageCount<2>` does not change the behaviour.

**Within one wave the kernel is correct**, including the fused block-scaled output:
`Disposition: Passed` at 2 groups of 512x1408x1280 (88 tiles) with
`LinCombBlockScaleFactor<32, float_e4m3_t, float, float_ue8m0_t>` verified against
CUTLASS's own host reference.

So the two questions for upstream sharpen to:

1. Is the missing SM120 pointer-array entry in the `MX_F4F6F8` `SfVectorSize`
   whitelist an omission? (Adding it compiles and the kernel is correct within one
   wave, which is evidence that it is.)
2. Is the pointer-array epilogue's load pipeline expected to work past a single
   wave on SM120, and is `beta != 0` expected to work at all on this path? Both
   schedules failing at exactly the SM count suggests a pipeline-state bug in the
   persistent tile loop rather than anything MX-specific.

---

## Addendum 2 — checked against CUTLASS `main` (4.x)

The 3.9.2 findings above are **not fixed upstream**, and what is present suggests the
gap is deliberate incompleteness rather than a typo:

1. The `MX_F4F6F8` `SfVectorSize` whitelist in `main` **still contains no SM120
   pointer-array clause.** SM100's `KernelSchedulePtrArrayBlockScaledGemmSm100` is
   there; SM120 has only `KernelScheduleBlockScaledGemmSm120` and its sparse
   counterpart, exactly as in 3.9.2.
2. `KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120` and its Pingpong
   sibling **do exist** in `main` — the builder's grouped branch needs them — and
   both still inherit from the *generic* `KernelPtrArrayTmaWarpSpecialized*` rather
   than from any block-scaled schedule family.
3. **There is no `KernelSchedulePtrArrayBlockScaledGemmSm120` at all**, in either
   version. So the schedule-family base that the whitelist tests against for SM100
   has no SM120 counterpart to test against — the omission is structural, not a
   missing line.
4. The CuTeDSL examples in `main` contain no SM120/GeForce grouped block-scaled
   reference kernel.

**Reading:** the grouped SM120 block-scaled path is *defined but never blessed* —
the tags exist because the collective builder emits them, the validation layer has
never been extended to admit them for MX, and no shipped example exercises it. Taken
with the runtime behaviour (correct within one wave, `__trap()` beyond it), the
best-supported of the three possibilities in Finding 2 is the **third**: an
incomplete path behind a guard, and our whitelist edit removed that guard.

This is why Finding 2 must stay a question about intent. "You omitted two clauses"
would be the wrong report; "is grouped MX block-scaled intended to be supported on
SM120, and if so what else is needed beyond the whitelist?" is the right one.
