# Port FA4 Single-CTA Backward Kernel Architecture to TLX

## Context

The TLX backward kernel (`_attn_bwd_ws` in `blackwell_fa_ws_pipelined_persistent.py`) differs
from the FA4 CuTe backward kernel (`FlashAttentionBackwardSm100` in `flash_bwd_sm100.py`)
in pipeline management, softmax computation, and epilogue handling. The goal is to bring the
TLX kernel's architecture in line with FA4's single-CTA (non-2CTA) backward pass.

**File to modify**: `/home/hoy/triton-fb/third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py`
**Reference**: `/home/hoy/flash-attention/flash_attn/cute/flash_bwd_sm100.py`

---

## Step 1: Warp and Register Structure — ALREADY DONE

`num_warps=8` in the TLX config specifies warps for the "default" (compute) task. Other
tasks specify their own warps. The total is already 8+4+1+1 = 14, matching FA4's 14 active
warps (8 compute + 4 reduce + 1 MMA + 1 load; FA4 also has 2 idle warps = 16 total).

Only register budgets could optionally be tuned to match FA4:

| Task      | TLX current | FA4  | Notes                                |
|-----------|-------------|------|--------------------------------------|
| reduce    | 88          | 152  | More regs for dQ staging loop        |
| compute   | 192         | 136  | FA4 uses packed f32x2 → less pressure|
| mma       | 24          | 88   | More regs for pipeline state         |
| load      | 48          | 88   | More regs for pipeline state         |

**Status**: No changes required for warp counts. Register tuning is optional/deferred.

---

## Step 2: Add SMEM Buffers for LSE and dPsum

**Current**: Compute task loads M (row-max) and D (delta) directly from GMEM via
`tl.load` + `tlx.prefetch` (lines 1103-1112 of `_bwd_compute_inner_loop`).

**FA4**: Load warp TMA-loads LSE and dPsum into SMEM (`sLSE`, `sdPsum`), compute
consumes via pipeline barriers. LSE has `Q_stage=2` (double-buffered), dPsum has
`dO_stage=1`.

**Changes** in `_attn_bwd_ws` (after existing SMEM allocations, ~line 1276):
```python
LSE_STAGE: tl.constexpr = NUM_BUFFERS_Q   # = 2
DPSUM_STAGE: tl.constexpr = NUM_BUFFERS_DO  # = 1
sLSE_tiles = tlx.local_alloc((BLOCK_M1,), tl.float32, LSE_STAGE)
sdPsum_tiles = tlx.local_alloc((BLOCK_M1,), tl.float32, DPSUM_STAGE)
lse_fulls = tlx.alloc_barriers(num_barriers=LSE_STAGE)
dpsum_fulls = tlx.alloc_barriers(num_barriers=DPSUM_STAGE)
```

**Also**: Add `desc_m` and `desc_delta` TMA descriptors. Create them in host `backward()`:
```python
desc_m = TensorDescriptor(M, shape=[BATCH*N_HEAD*N_CTX], strides=[1], block_shape=[BLOCK_M1])
desc_delta = TensorDescriptor(delta, shape=[BATCH*N_HEAD*N_CTX], strides=[1], block_shape=[BLOCK_M1])
```

---

## Step 3: Bundle K with Q Pipeline, V with dO Pipeline

**Current**: K, V, Q, dO loaded independently with 4 separate barrier pairs.

**FA4**: K committed to Q's barrier (first Q's `barrier_expect_bytes` includes K). V
committed to dO's barrier similarly. K/V loaded once per n_block; Q/dO per m_block.

**Load task prologue** (first iteration):
```python
# Expect bytes for BOTH K and Q on q_fulls
tlx.barrier_expect_bytes(q_fulls[q_buf_id],
    K_BYTES * BLOCK_N1 * HEAD_DIM + Q_BYTES * BLOCK_M1 * HEAD_DIM)
tlx.async_descriptor_load(desc_k, k_tiles, [...], q_fulls[q_buf_id])
tlx.async_descriptor_load(desc_q, q_tiles, [...], q_fulls[q_buf_id])
# Same pattern for V + dO on do_fulls
```

**Main loop**: Only Q and dO (K/V stay in SMEM).

**Remove**: Separate `k_fulls/k_empties` and `v_fulls` barriers.

**MMA task**: Waiting on `q_fulls` guarantees both K+Q ready; `do_fulls` guarantees V+dO.

---

## Step 4: Load LSE and dPsum in Load Task

After each Q load → TMA-load LSE:
```python
lse_buf_id = _get_bufidx_phase(blk_idx, LSE_STAGE)
tlx.barrier_expect_bytes(lse_fulls[lse_buf_id], 4 * BLOCK_M1)
tlx.async_descriptor_load(desc_m, sLSE_tiles[lse_buf_id],
    [(off_bh + curr_m).to(tl.int32)], lse_fulls[lse_buf_id])
```

After each dO load → TMA-load dPsum (same pattern with `dpsum_fulls`).

---

## Step 5: Compute Task — Consume LSE/dPsum from SMEM via Barriers

**Remove** from `_bwd_compute_inner_loop`:
```python
tlx.prefetch(M + offs_m_full, level="L1")
tlx.prefetch(D + offs_m_full, level="L1")
m_slices = _split_n(tl.load(M + offs_m_full).reshape([1, BLOCK_M1]), NUM_COMPUTE_SLICES)
Di_slices = _split_n(tl.load(D + offs_m_full).reshape([1, BLOCK_M1]), NUM_COMPUTE_SLICES)
```

**Replace with**:
```python
tlx.barrier_wait(lse_fulls[lse_buf_id], lse_phase)
m_full = tlx.local_load(sLSE_tiles[lse_buf_id])
tlx.barrier_wait(dpsum_fulls[dpsum_buf_id], dpsum_phase)
Di_full = tlx.local_load(sdPsum_tiles[dpsum_buf_id])
m_slices = _split_n(m_full.reshape([1, BLOCK_M1]), NUM_COMPUTE_SLICES)
Di_slices = _split_n(Di_full.reshape([1, BLOCK_M1]), NUM_COMPUTE_SLICES)
```

Remove `M`, `D` pointer params; add SMEM tiles + barriers as params.

---

## Step 6: Packed f32x2 Softmax Operations in Compute

FA4 uses `fma_packed_f32x2` / `sub_packed_f32x2` / `mul_packed_f32x2` for 2x throughput.
TLX already has `_fma_f32x2` and `_mul_f32x2` (lines 70-107). Add `_sub_f32x2`:

```python
@triton.jit
def _sub_f32x2(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            sub.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b], dtype=tl.float32, is_pure=True, pack=2,
    )
```

**Replace** in `_bwd_compute_inner_loop`:
```python
# Current:
pT = tl.math.exp2(qkT - m[None, :])
dsT = pT * (dpT - Di[None, :])

# Target:
qkT = _sub_f32x2(qkT, m[None, :])
pT = tl.math.exp2(qkT)
dpT = _sub_f32x2(dpT, Di[None, :])
dsT = _mul_f32x2(pT, dpT)
```

---

## Step 7: Compute Sync Barrier

S and P alias the same TMEM (`reuse=qk_tiles`). All compute warps must finish reading
S before any writes P back. FA4 uses `compute_sync_barrier.arrive_and_wait()`.

```python
compute_sync = tlx.alloc_warp_barrier(num_barriers=1, num_warps=8)
```

In compute loop, after all S slices read, before writing P:
```python
tlx.barrier_arrive(compute_sync[0])
tlx.barrier_wait(compute_sync[0], phase)
```

---

## Step 8: dKV Epilogue via SMEM Staging + TMA Store

**Current**: TMEM → regs → `desc_dv.store()` / `desc_dk.store()` (direct GMEM).

**FA4**: TMEM → regs → convert+scale → SMEM staging → TMA store.
Reuses `sQ` as `sdK`, `sdO` as `sdV` (free during epilogue).

**Changes**:
- Reuse `q_tiles` SMEM as dk staging, `do_tiles` as dv staging
- `EPILOGUE_SUBTILE` chunks of HEAD_DIM: read TMEM slice → store SMEM → TMA store
- `tlx.async_descriptor_store()` from SMEM for final write
- Add TMA store descriptors for dK/dV

---

## Step 9: dQ Reduce Parameter Adjustment

**Current**: `DQ_REDUCE_NCOL = 16`, `DQ_REDUCE_ITERS = 8`
**FA4**: `dQ_reduce_ncol = 32`, `dQ_reduce_iters = 4`

```python
DQ_REDUCE_NCOL = 32
DQ_REDUCE_ITERS = HEAD_DIM // DQ_REDUCE_NCOL  # 4
dq_store_buf = tlx.local_alloc((BLOCK_M1, 32), ...)  # was (BLOCK_M1, 16)
```

Update `desc_dq.block_shape` in host `_bwd_host_descriptor_pre_hook_tlx`.

---

## Step 10: Verify MMA Loop Ordering

TLX main loop already matches FA4's single-CTA ordering:

| Step | Operation           | Wait on           | Signal / Release    |
|------|---------------------|--------------------|---------------------|
| 1    | qkT = k @ qT       | q_fulls            | qk_fulls            |
| 2    | dq = dsT^T @ k      | ds_fulls, dq_empties| dq_fulls           |
| 3    | dk += dsT @ qT      | dsT_tmem_fulls     | q_empties           |
| 4    | dpT = v @ doT       | do_fulls, dp_empties| dp_fulls           |
| 5    | dv += ppT @ do      | p_fulls            | do_empties          |

Key: step 4 waits `dp_empties == dq_empties` (when `REUSE_DP_FOR_DQ=True`) — correct.

**No code changes needed** — verify only.

---

## Verification

```bash
cd /home/hoy/triton-fb
CUDA_VISIBLE_DEVICES=1 pytest third_party/tlx/tutorials/testing/test_correctness.py::test_blackwell_fa_ws_pipelined_persistent -x

# If hangs:
third_party/tlx/killgpu.sh
```
