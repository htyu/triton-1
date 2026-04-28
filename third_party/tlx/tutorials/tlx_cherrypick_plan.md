# TLX Cherry-Pick Plan: main → release/3.5.0.3

162 TLX-related commits in `main` not in `release/3.5.0.3`, sorted by committed date (oldest-first).

**Scope column:**
- *(empty)* — touches only `third_party/tlx/` (Python-only, no rebuild needed)
- **CROSS** — touches both `third_party/tlx/` and core compiler/backend code (C++ rebuild needed)
- **DEPS** — mentions TLX but does NOT touch `third_party/tlx/` (core compiler, backend, AutoWS, test infra)

## Commits

| # | Hash | Committed | Description | Scope |
|---|------|-----------|-------------|-------|
| 1 | ~~d5c8269cda~~ | 2026-02-04 | ~~[TLX] use single config when running third_party/tlx/tutorials/correctness_test.py (#834)~~ — SKIP: already present |  |
| 2 | ~~ce4d057ed0~~ | 2026-02-06 | ~~[TLX] Add TLX TableGen dependencies to TritonNvidiaGPUIR (#857)~~ — SKIP: already present | DEPS |
| 3 | ~~5898b9f857~~ | 2026-02-06 | ~~[Triton] Log IR dump directory for each autotune config (#853)~~ — DONE | DEPS |
| 4 | ~~ac9e9d9ffc~~ | 2026-02-06 | ~~[TLX] Unify correctness and perf tests for tutorial kernels (FA + matmul) (#854)~~ — SKIP: already present | CROSS |
| 5 | ~~eef144c047~~ | 2026-02-07 | ~~[TLX] Changes to support clustered-grid in Fixup (#855)~~ — SKIP: already present | CROSS |
| 6 | ~~1dd2dfd916~~ | 2026-02-07 | ~~[TLX] Add L2 cache policy support for TMA store (#858)~~ — SKIP: already present | CROSS |
| 7 | ~~1d33386666~~ | 2026-02-07 | ~~[BE] [TLX] Cleanup TMEM dummy layouts to avoid fp8 core dump (#832)~~ — SKIP: already present |  |
| 8 | ~~f7b1febc50~~ | 2026-02-09 | ~~[TLX] Fix global state race condition with thread-local storage (#865)~~ — SKIP: already present | CROSS |
| 9 | ~~888d1d5f5b~~ | 2026-02-09 | ~~[TLX] 3/N Define the IR interface for the reuse group. (#851)~~ — SKIP: already present | CROSS |
| 10 | ~~15484f4eb5~~ | 2026-02-10 | ~~Add distilled PTX ISA 9.1 knowledge base for Claude (#866)~~ — SKIP: already present | CROSS |
| 11 | ~~8afb3951b4~~ | 2026-02-10 | ~~[TLX] Add mxfp8 util file (#852)~~ — SKIP: already present | CROSS |
| 12 | ~~3bde255747~~ | 2026-02-10 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Towards a generic tcgen05.cp lowering (#8102)' (#878)~~ — SKIP: already present | DEPS |
| 13 | ~~030074a355~~ | 2026-02-11 | ~~Fix Clang-format lint issue (#888)~~ — SKIP: already present | DEPS |
| 14 | ~~69195fb9c4~~ | 2026-02-11 | ~~[TLX] 4/N define the IR interface for set_buffer_overlap (#876)~~ — SKIP: already present | CROSS |
| 15 | ~~517ce79ccf~~ | 2026-02-11 | ~~Add tlx.vote_ballot_sync op that lowers to NVVM::VoteSyncOp (#828)~~ — SKIP: already present | CROSS |
| 16 | ~~438cd73771~~ | 2026-02-11 | ~~[TLX] [FA] Add MXFP8 Support with HEAD_DIM=64 (#816)~~ — SKIP: already present | CROSS |
| 17 | ~~05ee2a7928~~ | 2026-02-11 | ~~[TLX] Add perf tracking for MXFP8 FA (#891)~~ — SKIP: already present |  |
| 18 | ~~fbfa8396f3~~ | 2026-02-12 | ~~[TLX] Support constexpr if-guards around async_task in async_tasks (#877)~~ — SKIP: already present | CROSS |
| 19 | ~~c8607e61fd~~ | 2026-02-12 | ~~[TLX] Enable multi-buffering scale values in TMEM (#884)~~ — SKIP: already present | CROSS |
| 20 | ~~dd5b2a2c79~~ | 2026-02-13 | ~~[TLX] Update MXFP8 to move P directly to TMEM (#894)~~ — SKIP: already present | CROSS |
| 21 | ~~75d115d737~~ | 2026-02-13 | ~~[TLX] Update slides for GPU Mode (#898)~~ — SKIP: already present | CROSS |
| 22 | ~~5bbb58956a~~ | 2026-02-13 | ~~[autoWS] generalize the PingPong pass to support warp_group_dot and exp (#668)~~ — REMOVED: AutoWS | DEPS |
| 23 | ~~7bc1d4ef86~~ | 2026-02-16 | ~~[TLX] Fix CLC barrier race for multi-CTA clusters (#908)~~ — SKIP: already present | CROSS |
| 24 | ~~1cc3b3ea9a~~ | 2026-02-17 | ~~[TLX] Apply cuBLAS block scaling layout swizzle to test_tlx scaled tests (#917)~~ — SKIP: already present | DEPS |
| 25 | 9e252c05d9 | 2026-02-18 | Add TRITON_AUTOTUNE_WARMUP_MS and TRITON_AUTOTUNE_REP_MS knobs (#918) | DEPS |
| 26 | 0365667ff7 | 2026-02-18 | [triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Add bitwidth to TMEM encoding (#8136)' (#913) | CROSS |
| 27 | ~~bc941c0839~~ | 2026-02-18 | ~~[TLX] 5/N Add lowering support for reuse groups (#889)~~ — SKIP: already present | CROSS |
| 28 | ~~3f8c9bff9b~~ | 2026-02-18 | ~~[TLX] Add bf16 support to Blackwell GEMM tutorial kernels (#920)~~ — SKIP: already present |  |
| 29 | ~~3496b8300c~~ | 2026-02-19 | ~~[TLX] Split-K, NUM_CTAS, autotune pruning for TLX GEMM (#929)~~ — SKIP: already present |  |
| 30 | ~~38657c2c02~~ | 2026-02-19 | ~~[TLX] 6/N Enable subtiling for reuse groups (#892)~~ — SKIP: already present | CROSS |
| 31 | ~~7268c6b327~~ | 2026-02-19 | ~~[autoWS] use CoarseSchedule to determine first arriving keyOp in PingPong (#823)~~ — REMOVED: AutoWS | DEPS |
| 32 | ~~80d851392a~~ | 2026-02-20 | ~~[TLX] Add DialectInlinerInterface to enable inlining of NVGPU ops (#939)~~ — SKIP: already present | DEPS |
| 33 | ~~d1e77f6f8f~~ | 2026-02-20 | ~~Narrow shared memory intervals through memdesc_index for precise hazard detection (#928)~~ — SKIP: already present | DEPS |
| 34 | ~~211a4ad9d2~~ | 2026-02-21 | ~~[TLX] 7/7 Add proper documentation for the StorageAliasSpec (#941)~~ — SKIP: already present | CROSS |
| 35 | ~~73495b5326~~ | 2026-02-21 | ~~[TLX] [MXFP8-FA] Add explicit SMEM -> TMEM scale transfer (#907)~~ — SKIP: already present |  |
| 36 | ~~74780c705f~~ | 2026-02-23 | ~~[TLX] Tuning GROUP_SIZE_M for Blackwell GEMM (#958)~~ — SKIP: already present |  |
| 37 | ~~913177102f~~ | 2026-02-24 | ~~[TLX] Add shape-dependent heuristic config selection for Blackwell GEMM (#960)~~ — SKIP: already present |  |
| 38 | ~~aef522e95d~~ | 2026-02-24 | ~~[BE] [TLX] Fix num_stages=0 (#964)~~ — SKIP: already present |  |
| 39 | ~~d5c02e84ba~~ | 2026-02-25 | ~~[TLX] Use async TMA store for epilogue in Blackwell GEMM (#982)~~ — SKIP: already present |  |
| 40 | ~~1d49e96664~~ | 2026-02-25 | ~~Update buffer reuse to use Columns as the TMEM unit instead of bytes (#943)~~ — DONE | CROSS |
| 41 | ~~b2f742a609~~ | 2026-02-25 | ~~[AutoWS] Fix memory and register handling for warp specialization (#860)~~ — REMOVED: AutoWS | DEPS |
| 42 | ~~44ce51850a~~ | 2026-02-26 | ~~[TLX][Tutorial] update FA/TLX with rescale opt (#921)~~ — SKIP: already present |  |
| 43 | ~~a4eafa3c6d~~ | 2026-02-26 | ~~[AutoWS] Fix thread ID and barrier handling for warp specialization (#861)~~ — REMOVED: AutoWS | DEPS |
| 44 | ~~1f22303e7a~~ | 2026-02-26 | ~~[AutoWS] Fix fused attention test (#859)~~ — REMOVED: AutoWS | DEPS |
| 45 | ~~c8383431b2~~ | 2026-02-26 | ~~[TLX] Relax requirement for memdesc_reinterpret op (#940)~~ — SKIP: already present | CROSS |
| 46 | ~~4867b62fd5~~ | 2026-02-28 | ~~[TLX] Improve PrintTTGIRToTLX output readability (#983)~~ — SKIP: already present |  |
| 47 | b9bd5b4b67 | 2026-02-28 | [TLX] Adds APIs to support async_bulk_copy (#977) | CROSS |
| 48 | ~~fb1b2582fe~~ | 2026-03-01 | ~~[TLX] Add store_reduce support to tlx.async_descriptor_store for TMA atomic reductions (#1006)~~ — SKIP: already present | CROSS |
| 49 | d664172c66 | 2026-03-02 | [triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Generic tcgen05.cp lowering (#8225)' (#1012) | DEPS |
| 50 | ~~aa77069d74~~ | 2026-03-02 | ~~[TLX] throw errors for M=64 2cta mma (#1004)~~ — SKIP: already present | DEPS |
| 51 | ~~17af1ee60a~~ | 2026-03-02 | ~~[TLX] Interleave TMA stores across MMA groups in Blackwell GEMM epilogue (#1003)~~ — SKIP: already present |  |
| 52 | cbfc4117b2 | 2026-03-03 | Add TMA prefetch Ops and expose to TLX (#1022) | CROSS |
| 53 | ~~56a02062d4~~ | 2026-03-03 | ~~[TLX] Add L2 cache hints to TMA async reduce operations (#1023)~~ — SKIP: already present | CROSS |
| 54 | ~~ce6394b284~~ | 2026-03-03 | ~~[TLX] Add tlx.threadfence() and tlx.threadfence_system() (#1011)~~ — SKIP: already present | CROSS |
| 55 | ~~3de877f04e~~ | 2026-03-03 | ~~[TLX] Add L2 cache hints to Blackwell GEMM TMA loads and stores (#1027)~~ — SKIP: already present |  |
| 56 | ~~4f2f0c0564~~ | 2026-03-04 | ~~[TLX] Multi-buffer epilogue TMA stores in Blackwell GEMM (#1028)~~ — SKIP: already present |  |
| 57 | ~~ae1e01f344~~ | 2026-03-04 | ~~[TLX] Improve Split-K autotuning for undersaturated GEMM shapes (#1032)~~ — SKIP: already present |  |
| 58 | ~~e3fc12a63f~~ | 2026-03-05 | ~~[TLX] Improve PrintTTGIRToTLX on Control Flow (#1005)~~ — SKIP: already present |  |
| 59 | ~~91e9544bd1~~ | 2026-03-05 | ~~[TLX] Fix atomic operations in clustered kernelsi (#1036)~~ — SKIP: already present | DEPS |
| 60 | ~~92469a5310~~ | 2026-03-05 | ~~[TLX] Add `alloc_warp_barrier` for multi-thread barrier arrival (#1031)~~ — SKIP: already present | CROSS |
| 61 | ~~1f7ae4a8c4~~ | 2026-03-05 | ~~[TLX] Use torch.empty instead of torch.zeros for hopper_gemm_ws (#1037)~~ — SKIP: already present |  |
| 62 | ~~4e10d95cea~~ | 2026-03-05 | ~~[TLX] Persistent WS kernel for Hopper GEMM (#1038)~~ — DONE |  |
| 63 | ~~256b6704f7~~ | 2026-03-06 | ~~[autoWS] support taskIDPropagation for tt.map_elementwise region (#1039)~~ — REMOVED: AutoWS | DEPS |
| 64 | ~~1110c2a265~~ | 2026-03-06 | ~~[TLX] Fix named barrier deadlock caused by LLVM jump threading (#1040)~~ — SKIP: already present | DEPS |
| 65 | ~~9b62ad525d~~ | 2026-03-09 | ~~[TLX] Add async_store to planCTA (#1055)~~ — SKIP: already present | DEPS |
| 66 | ~~d2ccc413d5~~ | 2026-03-10 | ~~[TLX] Tune GROUP_SIZE_M in hopper_gemm_ws using preprocess_configs (#1061)~~ — DONE |  |
| 67 | ~~59fc9adf5c~~ | 2026-03-10 | ~~[AutoWS] Define skill for AutoWS testing (#1063)~~ — REMOVED: AutoWS | DEPS |
| 68 | ~~13ed8ebe4c~~ | 2026-03-11 | ~~[TLX GEMM] Replace atomic reduction with separate reduction kernel for Split-K (#1067)~~ — DONE |  |
| 69 | ~~0b3ad5c1ed~~ | 2026-03-11 | ~~[TLX] Enable TMA multicast for hopper_gemm_ws (#1065)~~ — SKIP: already present |  |
| 70 | ~~5debd9849a~~ | 2026-03-11 | ~~[AutoWS] Use NameLoc for readable variable names and add source location comments in PrintTTGIRToTLX (#1068)~~ — REMOVED: AutoWS |  |
| 71 | ~~967cefe0b9~~ | 2026-03-12 | ~~[TLX] Fix incorrect 2-CTA results when GROUP_SIZE_M is not a multiple of NUM_CTAS (#1071)~~ — DONE |  |
| 72 | ~~1924579984~~ | 2026-03-12 | ~~FA Perf Test for multi-thread barrier (#1042)~~ — DONE |  |
| 73 | 0ebfe0b13a | 2026-03-12 | Add Gemm Perf Test for multi-thread barrier arrival" (#1070) |  |
| 74 | ~~12f2bec3d3~~ | 2026-03-13 | ~~[TLX] Blackwell "Preferred Cluster Size" (#1074)~~ — DONE | CROSS |
| 75 | ~~58c8f9250d~~ | 2026-03-13 | ~~[TLX] [Triton] Add an explicit skill for fence issues in TLX (#1080)~~ — DONE | DEPS |
| 76 | ~~66755c8300~~ | 2026-03-15 | ~~Support FA MXFP8 with HEAD-DIM=128 (#942)~~ — DONE |  |
| 77 | ~~6d9cf2350f~~ | 2026-03-15 | ~~[TLX][triton beta] Gate Blackwell "preferred cluster size" control with cuda version >=12.8 (#1092)~~ — DONE | DEPS |
| 78 | ~~3d754a509f~~ | 2026-03-16 | ~~[TLX] Extend Split-K range for undersaturated GEMM shapes (#1098)~~ — DONE |  |
| 79 | aaa648e73b | 2026-03-18 | [triton][beta] [Cherry-pick][RESOLVED] '[ConSan] ConSan env var should be cache invalidating (#8332)' | DEPS |
| 80 | ~~2c0c302e1d~~ | 2026-03-18 | ~~[TLX] Add two_cta support to async_descriptor_load for 2-CTA TMA barrier signaling (#1077) (#1077)~~ — DONE | CROSS |
| 81 | 694b0b9fb3 | 2026-03-19 | [triton][beta] [Cherry-pick][RESOLVED] '[AMD][NFC] Move LowerLoops into TritonAMDGPUPipeline (#8341)' | CROSS |
| 82 | ~~6292f205bd~~ | 2026-03-19 | ~~[TLX] Fix warp barrier arrive with remote_cta_rank (#1096)~~ — SKIP: already present | CROSS |
| 83 | ~~798274bbbc~~ | 2026-03-20 | ~~[TLX][CLC] Fix the final trailing remote bar arrive in the last consumer call (#1108)~~ — SKIP: already present | CROSS |
| 84 | ~~03c95cc3df~~ | 2026-03-20 | ~~Reformat with precommit (#1111)~~ — SKIP: already present | CROSS |
| 85 | c4310c9917 | 2026-03-20 | [TLX] More mm shapes for blackwell_gemm_ws in correctness tests (#1035) |  |
| 86 | ~~46e524ca5b~~ | 2026-03-20 | ~~[TLX] Fix split-K reduction in autotuner production path (#1117)~~ — DONE |  |
| 87 | ~~8ecf1c8688~~ | 2026-03-20 | ~~[TLX] Add BM=64 tile size for small unsaturated GEMM shapes (#1119)~~ — DONE |  |
| 88 | ~~daaca47640~~ | 2026-03-20 | ~~[TLX] Use smaller tiles for split-K reduction kernel (#1118)~~ — SKIP: already present |  |
| 89 | ~~ff56e1d360~~ | 2026-03-23 | ~~[TLX] Make heuristic config selection in blackwell_gemm_ws configurable via env var (#1115) (#1115)~~ — SKIP: already present |  |
| 90 | 9c8558ab4d | 2026-03-23 | [triton][beta] [Cherry-pick] '[AMD] Use PaddedLayout with AsyncCopy on gfx950 when pipelining (#8365)' | CROSS |
| 91 | ~~b2e68bd65e~~ | 2026-03-23 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[Triton Plugin] Reorder the configuration sequence of Triton, third-party libraries, and Triton plugins (#8397)'~~ — DONE | DEPS |
| 92 | ~~e7b01389f8~~ | 2026-03-23 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[LAYOUTS] Generate distributed layouts for `tcgen05.ld/st` generically (#8421)'~~ — SKIP: already present | CROSS |
| 93 | ~~e8677d1d93~~ | 2026-03-23 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[WS] Swap ordering of `OptimizePartitionWarps` and SWP (#8415)'~~ — DONE | DEPS |
| 94 | ~~60a79e8217~~ | 2026-03-24 | ~~[TLX] Add bwd correctness test for blackwell_fa_ws_pipelined_persistent (#1134)~~ — DONE |  |
| 95 | ~~88126ffd17~~ | 2026-03-24 | ~~[TLX] Fix planCTA for TLX (#1094) (#1094)~~ — SKIP: already present | DEPS |
| 96 | e4c43d957d | 2026-03-24 | [Triton] Support multi-CTA reduction in Triton (#1102) | CROSS |
| 97 | ~~9986f341b6~~ | 2026-03-24 | ~~[TLX] Add tlx.prefetch for pointer-based cache prefetch hints (#1132)~~ — SKIP: already present | CROSS |
| 98 | 8c2c6d8bc8 | 2026-03-24 | [TLX] Support column-major A and B inputs in blackwell_gemm_ws and hopper_gemm_ws |  |
| 99 | ~~648dca741b~~ | 2026-03-25 | ~~[TLX] Fix the TLX Blackwell GEMM kernel hang on empty split-K splits (#1141)~~ — SKIP: already present |  |
| 100 | ~~899394f04c~~ | 2026-03-25 | ~~[TLX GEMM] Filter out split-K values that create empty splits in autotuner (#1142)~~ — DONE |  |
| 101 | ~~9f7d1965ad~~ | 2026-03-26 | ~~[TLX] Update skills to reflect cluster usage (#1156)~~ — SKIP: already present | DEPS |
| 102 | 952795e7f6 | 2026-03-26 | Unify TLX backward attention kernel with triton-fb (#1148) |  |
| 103 | 1b938f0fae | 2026-03-26 | [TLX] Add tlxIsClustered API for cluster dimension checks (#1159) (#1159) (#1159) | CROSS |
| 104 | f3d70f5f6d | 2026-03-26 | [TLX] Support copy of local SMEM to remote SMEM (#1167) | CROSS |
| 105 | 512049b370 | 2026-03-27 | [triton][beta] [Cherry-pick][RESOLVED] 'Revert "[LAYOUTS] Generate distributed layouts for `tcgen05.ld/st` generically (#8421)" (#8469)' | CROSS |
| 106 | ~~7149073e06~~ | 2026-03-27 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Implement BF16x3 trick (#7592)'~~ — SKIP: already present | DEPS |
| 107 | 2386ecc768 | 2026-03-27 | [triton][beta] [Cherry-pick][RESOLVED] '[Backend][NFC] changed MLIR builder API for upcoming LLVM bump (#8572)' | DEPS |
| 108 | 5210641cd0 | 2026-03-27 | [triton][beta] [Cherry-pick][RESOLVED] '[RELAND][LAYOUTS] Generate distributed layouts for tcgen05.ld/st generically (#8421) (#8495)' | CROSS |
| 109 | 9970300d48 | 2026-03-27 | [TLX] Add a CLC variation of TLX FA (#1165) |  |
| 110 | ~~c048a7b267~~ | 2026-03-30 | ~~[lint-autofix] Fix pre-commit formatting issues (#1180)~~ — SKIP: already present | CROSS |
| 111 | ~~ef29285ade~~ | 2026-03-30 | ~~[TLX] Update doc for DSMEM store/copy (#1183)~~ — SKIP: already present |  |
| 112 | ~~4254c6f463~~ | 2026-03-31 | ~~[tlx] Add TMA-pipelined skinny GEMM kernel with split-K for hopper_gemm_ws (#1140)~~ — SKIP: already present |  |
| 113 | de8378b045 | 2026-03-31 | [TLX] Prefetch TMA descriptor object (#1182) | CROSS |
| 114 | 4a04b65cb5 | 2026-03-31 | [TLX] Use CLC for blackwell_fa_ws_pipelined_persistent kernel (#1187) |  |
| 115 | 01e5827036 | 2026-04-01 | [triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Set 2CTA mode as a global flag (#8653)' | DEPS |
| 116 | 12f8f27cce | 2026-04-01 | [triton][beta] [Cherry-pick][RESOLVED] '[NFC] Perform supportMMA check during IR verification (#8640)' | DEPS |
| 117 | c1b2704af9 | 2026-04-01 | [triton][beta] [Cherry-pick] '[BACKEND] Initial support for 2CTA mode in Gluon (#8644)' | CROSS |
| 118 | ~~d9d68520c7~~ | 2026-04-01 | ~~[triton][TLX] Add backward pass benchmarking to FA perf test (#1189)~~ — DONE |  |
| 119 | ~~d78845a03a~~ | 2026-04-01 | ~~[TLX] Slice bwd compute loop and pipeline dQ reduction in FA (#1191)~~ — SKIP: already present |  |
| 120 | d35b23667b | 2026-04-01 | [TLX] Load M and D via TMA in backward attention kernel (#1192) |  |
| 121 | f7d3c90829 | 2026-04-02 | [triton][beta] Fix WS lit test crashes, cluster launch dims via reqnctapercluster (#1193) | DEPS |
| 122 | ~~66a28edfb1~~ | 2026-04-03 | ~~[TLX] Another around of optimization for FA bwd (#1195)~~ — SKIP: already present |  |
| 123 | 180378840d | 2026-04-05 | [triton][beta] Fix nvgpu -> nvg dialect rename in TLX tests | DEPS |
| 124 | 23629ae966 | 2026-04-05 | [triton][beta] [Cherry-pick][RESOLVED] '[LAYOUTS] Make CTALayout an honest-to-goodness LinearLayout (#8770)' | CROSS |
| 125 | ~~f92c2e1b4f~~ | 2026-04-05 | ~~[triton][beta] Relax TCGen5MMAOp verification to support TLX two-CTA mode~~ — SKIP: already present | DEPS |
| 126 | ~~52f433907c~~ | 2026-04-07 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Add support for out of tree TTIR/TTGIR passes (#8401)'~~ — SKIP: already present | DEPS |
| 127 | ~~3c489efc15~~ | 2026-04-08 | ~~[Triton] [TLX] Fix Hopper GEMM Tests (#1206)~~ — SKIP: already present |  |
| 128 | 6c0b8e1284 | 2026-04-08 | [lint-autofix] Fix pre-commit formatting issues (#1210) | CROSS |
| 129 | ~~bb28230b65~~ | 2026-04-08 | ~~[triton-beta] Fix twoCTAs determination for TLX paired-CTA MMA kernels (#1209)~~ — SKIP: already present | DEPS |
| 130 | ~~884a149198~~ | 2026-04-08 | ~~[triton][beta] Fix CheckMatmulTwoCTA pass for scaled MMA and AMD launcher compilation errors (#1213)~~ — SKIP: already present | DEPS |
| 131 | ~~93d368248f~~ | 2026-04-08 | ~~[Triton] [TLX] Increase accuracy threshold for large K values (#1207)~~ — DONE |  |
| 132 | 39d733791f | 2026-04-09 | [TLX] [BE] Refactor test_tlx.py to allow better scaling (#1107) | CROSS |
| 133 | acee6eb847 | 2026-04-09 | [triton][beta] [Cherry-pick] Backport '[Gluon] Expose finer grained cluster fences (#9076)' (#1214) | DEPS |
| 134 | ~~37129aa5d1~~ | 2026-04-09 | ~~[TLX] disable the cluster sync ops verifier (#1232)~~ — SKIP: already present | DEPS |
| 135 | b0490dc4b0 | 2026-04-09 | [triton] Fix internal code for LLVM bump API changes (#1218) | CROSS |
| 136 | 1a7875973d | 2026-04-09 | [triton][beta] [Cherry-pick] '[NVIDIA] Verify encodings on TMA ops (#8886)' | DEPS |
| 137 | 2afad86417 | 2026-04-09 | [triton][beta] [Cherry-pick][RESOLVED] '[NFC] Rename CTAEncoding as CGAEncoding (#8850)' | CROSS |
| 138 | ~~355fab8da9~~ | 2026-04-09 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[WS] Nested loop support (#8687)'~~ — SKIP: already present | DEPS |
| 139 | 7701b912d8 | 2026-04-09 | [triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Gluon multicta + 2cta support (#8684)' | DEPS |
| 140 | daace4d6d2 | 2026-04-09 | [TLX] Unify cluster sync (#1198) | CROSS |
| 141 | ~~135c2d0f9b~~ | 2026-04-09 | ~~[TLX] Fix pytest import (#1234)~~ — SKIP: already present | DEPS |
| 142 | ~~c4bf51b13e~~ | 2026-04-10 | ~~[TLX] [Triton] FIx TLX internal test listing (#1236)~~ — SKIP: already present | DEPS |
| 143 | 44f8e836db | 2026-04-10 | [TLX] Implement implicit/explicit mbar init cluster fence (#1240) | CROSS |
| 144 | ~~664d65b8e2~~ | 2026-04-11 | ~~[TLX] Respect user-provided warpGroupStartIds in AllocateWarpGroups (#1237)~~ — REMOVED: AutoWS | DEPS |
| 145 | ~~5fd07d3347~~ | 2026-04-13 | ~~[ci] Add h100 runner (#1241)~~ — DONE | DEPS |
| 146 | ~~4b4e974455~~ | 2026-04-13 | ~~[TLX] remove redundant lit test checking num ctas for cluster sync ops (#1235)~~ — SKIP: already present | DEPS |
| 147 | ~~317a950eff~~ | 2026-04-13 | ~~[AutoWS] AutoWS Release Milestone (#1130) (#1130)~~ — REMOVED: AutoWS | CROSS |
| 148 | ~~45d0561185~~ | 2026-04-14 | ~~[TLX] Work around ptxas secondHalfOffset=0 bug for scalar TMEM tiles in FA (#1238)~~ — DONE |  |
| 149 | ~~a5bfb46845~~ | 2026-04-15 | ~~[TLX] Relax BM=64 tile gate for unsaturated GEMM shapes (#1256)~~ — DONE |  |
| 150 | ~~21f7009336~~ | 2026-04-15 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Fix Illegal Instruction in MMAv5 lowering (#8910)'~~ — SKIP: already present | DEPS |
| 151 | ~~b2b492ebcc~~ | 2026-04-15 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Throw an error instead of miscompiling very large tcgen05.mma along N (#8915)'~~ — SKIP: already present | DEPS |
| 152 | acc8cb0148 | 2026-04-15 | [TLX] Guard remote and local bars init together to simplify cluster sync op insertion (#1262) | DEPS |
| 153 | 0affca6798 | 2026-04-17 | [triton][beta] [Cherry-pick][RESOLVED] '[Gluon][Dialect] Tighten verifiers, add more helpful error messages (#8981)' | DEPS |
| 154 | ~~c0a476fd0a~~ | 2026-04-17 | ~~[autoWS] support Hopper + FA + SWP via annotation (#1249)~~ — REMOVED: AutoWS | DEPS |
| 155 | ~~b9c80a6f64~~ | 2026-04-20 | ~~[triton][beta] [Cherry-pick][RESOLVED] '[Backend] Add explicit semantics for async ops. (#8966)'~~ — SKIP: already present | DEPS |
| 156 | ff0be24bc3 | 2026-04-20 | [TLX] Outlaw register specification in the default partition (#1283) | CROSS |
| 157 | ~~053b50f75e~~ | 2026-04-20 | ~~[Triton] [Numerics] Add a prototype for specifying bitwise consistent reductions (#1100)~~ — SKIP: already present | CROSS |
| 158 | ~~67419a25e6~~ | 2026-04-21 | ~~[AutoWS] Avoid duplicate fence insertion with constant ops (#1291)~~ — REMOVED: AutoWS | DEPS |
| 159 | 44305c9ce3 | 2026-04-21 | [triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Remove synchronisation in 2CTA mma (#8986)' | DEPS |
| 160 | ~~df13782131~~ | 2026-04-21 | ~~[triton][tileir] Apply Meta-specific patches to TileIR backend in Triton Beta (#1301)~~ — SKIP: already present | CROSS |
| 161 | 251b85d2e0 | 2026-04-22 | [triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Fix uses of CGAEncodingAttr::getDefault (#9040)' | CROSS |
| 162 | f8c46f6352 | 2026-04-22 | [triton][beta] [Cherry-pick][RESOLVED] '[BACKEND] Add support for TMA with multicast (#9005)' | CROSS |

## Summary

- **Total:** 162 commits (2026-02-04 → 2026-04-22)
- **Already present (SKIP):** 83
- **Cherry-picked (DONE):** 24
- **AutoWS removed:** 12
- **Remaining to port:** 43
