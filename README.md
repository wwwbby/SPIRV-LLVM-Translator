"""Triton-ascend implementation of sparse_flash_attention (SFA).

Interface aligned with MindSpore ops.sparse_flash_attention (drop-in replacement
for the call site in mindformers .../transformer/dsa/dsa_attention.py).

Computes  softmax(Q @ K^T / sqrt(d)) @ V  over a sparsely gathered subset of KV,
in MLA-absorb mode (attention_mode=2):
  q_full = concat(query[..,:D], query_rope[..,:Dr])     (D in 128/256/512, Dr=64)
  k_full = concat(key[..,:D],   key_rope[..,:Dr])
  value  = key[..,:D]  (K and V share the compressed latent c_kv; the passed
           `value` tensor is IGNORED, matching CANN's attention_mode=2 contract)

MQA only (N2=1): the N1 query heads share one gathered KV per (b,s1).

Layout support:
  - layout_query: BSND / TND
  - layout_kv:    BSND / TND / PA_BSND (paged KV via block_table)
TND and PA_BSND are normalized to BSND on host (PyNative only); for GRAPH_MODE
the caller should pass BSND directly (mirrors lightning_indexer_triton).

sparse_mode 0 (full) and 3 (rightDownCausal) supported.
sparse_block_size 1 (token-wise) and 2^n in [1,128] (block-wise) supported.
"""
import triton
import triton.language as tl
import triton.backends.ascend.runtime

try:
    import triton_ascend.language.extra as tle
    _PREFILL_KERNEL_AVAILABLE = True
except ImportError:
    tle = None
    _PREFILL_KERNEL_AVAILABLE = False

import mindspore as ms
from mindspore import ops

INT64_MAX = 9223372036854775807

# CANN constraints. attention_mode=2 (MLA-absorb) fixes Dr=64; D (kv_lora_rank,
# the nope latent dim) is generalized here to 128/256/512 — CANN itself only
# does 512, so D!=512 has no CANN reference (verify against the numpy golden).
_VALID_D = (128, 256, 512)
_D_ROPE = 64
_VALID_N1 = (1, 2, 4, 8, 16, 32, 64, 128)


def _next_pow2(x):
    # Ascend 要求 kernel grid 每维都是 2 的幂, 否则分核映射出错 -> aicore trap。
    # padding 多出的 program 在 kernel 里靠 in_range 掩码空转。
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _patch_triton_ascend_mindspore_dtype_bytes():
    """修补 triton-ascend autotuner 的 dtype 字节数查询 (见 lightning_indexer_triton)。"""
    try:
        from triton.backends.ascend.runtime import utils as ascend_utils
        from triton.backends.ascend.runtime import autotuner as ascend_autotuner
    except ImportError:
        return

    origin_func = getattr(ascend_utils, "get_byte_per_numel", None)
    if origin_func is None:
        return
    if getattr(origin_func, "_mindspore_dtype_patched", False):
        if hasattr(ascend_autotuner, "get_byte_per_numel"):
            ascend_autotuner.get_byte_per_numel = origin_func
        return

    dtype_bytes = {}
    for byte_size, names in (
        (1, ("int8", "uint8", "bool_")),
        (2, ("float16", "bfloat16", "int16", "uint16")),
        (4, ("float32", "int32", "uint32")),
        (8, ("float64", "int64", "uint64")),
    ):
        for name in names:
            dt = getattr(ms, name, None)
            if dt is not None:
                dtype_bytes[dt] = byte_size

    def patched_get_byte_per_numel(dtype):
        try:
            if dtype in dtype_bytes:
                return dtype_bytes[dtype]
        except TypeError:
            pass
        return origin_func(dtype)

    patched_get_byte_per_numel._mindspore_dtype_patched = True
    ascend_utils.get_byte_per_numel = patched_get_byte_per_numel
    if hasattr(ascend_autotuner, "get_byte_per_numel"):
        ascend_autotuner.get_byte_per_numel = patched_get_byte_per_numel


_patch_triton_ascend_mindspore_dtype_bytes()


def _prune_configs(configs, named_args, **kwargs):
    """autotune config 过滤 (UB 上限 + grid pow2 + grid 总数上限)。

    Two-pass kernel: no [BLOCK_G,D] resident acc; pass-2 keeps acc[BLOCK_G,BLOCK_DV]
    + v[BLOCK_K,BLOCK_DV], so UB no longer scales with full D. The 2.0 factor below
    is a coarse guard for Ascend's auto-multi-buffer doubling (the on-device
    compiler is the final UB authority — observed it reject undersized estimates).
    """
    _UB_LIMIT_BYTES = 180 * 1024   # headroom under the 192KB hard limit
    _GRID_LIMIT = 131072  # 这个值持保留意见

    def _get(name):
        if name in named_args:
            return named_args[name]
        return kwargs.get(name, None)

    N1 = _get("N1")
    BS1 = _get("B_S1")

    def _estimate_ub_bytes(block_g, block_k, block_d, block_dv):
        if None in (block_g, block_k, block_d, block_dv):
            return 0
        acc = block_g * block_dv * 4        # acc[BLOCK_G, BLOCK_DV] fp32 (pass 2)
        v_tile = block_k * block_dv * 2     # v[BLOCK_K, BLOCK_DV] fp16
        q_tile = block_g * block_d * 2      # q/k QK tiles fp16
        k_tile = block_k * block_d * 2
        s_tile = block_g * block_k * 4      # scores[BLOCK_G, BLOCK_K] fp32
        p_tile = block_g * block_k * 2      # p cast fp16 for PV
        trans = block_k * block_d * 2       # tl.trans tmp
        total = acc + v_tile + q_tile + k_tile + s_tile + p_tile + trans
        return int(total * 2.0)             # multi-buffer doubling guard

    kept = []
    for c in configs:
        bg = c.kwargs.get("BLOCK_G")
        bk = c.kwargs.get("BLOCK_K")
        bd = c.kwargs.get("BLOCK_D")
        bdv = c.kwargs.get("BLOCK_DV")

        if _estimate_ub_bytes(bg, bk, bd, bdv) > _UB_LIMIT_BYTES:
            continue
        # NB: BLOCK_G may exceed N1; the kernel masks padded heads (g_valid),
        # so we do NOT prune on bg > N1 (would kill all configs for small N1).
        if None not in (BS1, N1) and bg:
            grid0 = _next_pow2(BS1)
            grid1 = _next_pow2((N1 + bg - 1) // bg)
            if grid0 * grid1 > _GRID_LIMIT:
                continue
        kept.append(c)

    if not kept:
        print('Warning: all autotune params pruned')
        kept = [min(configs, key=lambda c: _estimate_ub_bytes(
            c.kwargs.get("BLOCK_G"), c.kwargs.get("BLOCK_K"),
            c.kwargs.get("BLOCK_D"), c.kwargs.get("BLOCK_DV")))]
    return kept


@triton.jit
def _sfa_scores_block(
    q_ptr, q_base, qr_ptr, qr_base,
    k_ptr, k_base, kr_ptr, kr_base,
    tok_clamped, tok_valid, g_offs, g_valid,
    scale_value,
    D: tl.constexpr, D_ROPE: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """scores[BLOCK_G, BLOCK_K] = (q_nope·k_nope + q_rope·k_rope) * scale.

    Shared by both passes; recomputed (not cached) so the kernel keeps no large
    resident buffer. Invalid gathered tokens are masked to -inf.
    """
    scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_valid = d_offs < D
        q_tile = tl.load(
            q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
            mask=g_valid[:, None] & d_valid[None, :], other=0.0)
        k_tile = tl.load(
            k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
            mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
        scores += tl.dot(q_tile, tl.trans(k_tile))
    for d_start in range(0, D_ROPE, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_valid = d_offs < D_ROPE
        qr_tile = tl.load(
            qr_ptr + qr_base + g_offs[:, None] * D_ROPE + d_offs[None, :],
            mask=g_valid[:, None] & d_valid[None, :], other=0.0)
        kr_tile = tl.load(
            kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + d_offs[None, :],
            mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
        scores += tl.dot(qr_tile, tl.trans(kr_tile))
    scores = scores * scale_value
    return tl.where(tok_valid[None, :], scores, float('-inf'))


@triton.autotune(
    configs=[
        # Two-pass kernel: pass-2 keeps only acc[BLOCK_G, BLOCK_DV] (not [.,D]) +
        # v[BLOCK_K, BLOCK_DV], so UB no longer scales with full D. BLOCK_DV<=128.
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 64, "BLOCK_D": 128, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 32, "BLOCK_D": 128, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 128, "BLOCK_D": 64, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 32, "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 64}),
    ],
    key=["B_S1", "N1", "S2", "topK", "D", "D_ROPE"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sfa_kernel(
    q_ptr, qr_ptr,                       # query[B,S1,N1,D], query_rope[B,S1,N1,Dr]
    k_ptr, kr_ptr, v_ptr,                # key/key_rope/value, all [B,S2,1,*]
    sparse_ptr,                          # token indices [B,S1,1,topK] int32 (block-wise pre-expanded on host)
    out_ptr, sm_max_ptr, sm_sum_ptr,     # outputs
    act_q_ptr, act_k_ptr,
    B_S1, S1, S2, N1, topK,
    D: tl.constexpr, D_ROPE: tl.constexpr,
    scale_value,
    sparse_mode: tl.constexpr,
    return_lse: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """Flash attention over sparsely gathered KV (BSND, MQA / N2=1), two-pass.

    Grid: (_next_pow2(B*S1), _next_pow2(cdiv(N1, BLOCK_G))), both pow2-padded.
    Each program: one (b,s1) position, BLOCK_G query heads. Inline-gathers KV
    rows by sparse token indices.

    Pass 1: stream KV -> online-softmax stats m_i / l_i (only [BLOCK_G] vectors).
    Pass 2: tile output dim by BLOCK_DV; per tile re-stream KV, recompute scores,
            accumulate p @ v into acc[BLOCK_G, BLOCK_DV]. Keeps UB independent of D.

    sparse_ptr holds token positions directly; block-wise (sparse_block_size>1)
    is pre-expanded on host into per-token indices, so this kernel is token-wise.
    """
    pid_bs1 = tl.program_id(0)
    pid_g = tl.program_id(1)

    bs1_in_range = pid_bs1 < B_S1
    pid_bs1 = tl.where(bs1_in_range, pid_bs1, 0)

    b = pid_bs1 // S1
    s1 = pid_bs1 % S1

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_valid = g_offs < N1

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)

    # causal window upper bound (token threshold)
    if sparse_mode == 0:
        threshold = act_k
    else:
        threshold = act_k - act_q + s1 + 1

    # rightDownCausal: leading rows (query longer than key) are fully hidden.
    row_active = bs1_in_range & (s1 < act_q) & (threshold > 0)

    # base offsets (memory layout: q[B,S1,N1,D], k/v[B,S2,1,D], rope analogous)
    q_base = (b * S1 + s1) * N1 * D
    qr_base = (b * S1 + s1) * N1 * D_ROPE
    k_base = b * S2 * D
    kr_base = b * S2 * D_ROPE
    v_base = b * S2 * D
    sp_base = (b * S1 + s1) * topK

    # ---- Pass 1: online-softmax stats (m_i, l_i); no V, no [BLOCK_G,D] acc ----
    m_i = tl.full([BLOCK_G], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    for blk_start in range(0, topK, BLOCK_K):
        blk_offs = blk_start + tl.arange(0, BLOCK_K)
        blk_in_count = blk_offs < topK
        tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
        tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
        tok_clamped = tl.where(tok_valid, tok, 0)

        scores = _sfa_scores_block(
            q_ptr, q_base, qr_ptr, qr_base,
            k_ptr, k_base, kr_ptr, kr_base,
            tok_clamped, tok_valid, g_offs, g_valid,
            scale_value, D, D_ROPE, BLOCK_G, BLOCK_K, BLOCK_D)

        # guard all-masked block: max stays -inf -> safe 0 so exp(-inf)=0, not nan.
        m_blk = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_blk)
        m_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
        p = tl.exp(scores - m_safe[:, None])
        p = tl.where(tok_valid[None, :], p, 0.0)
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # final global max for pass 2 (empty rows: m_i==-inf -> use 0, p will be 0)
    m_final = tl.where(m_i == float('-inf'), 0.0, m_i)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)

    # ---- Pass 2: tile output dim, recompute scores, accumulate p @ v ----
    out_base = (b * S1 + s1) * N1 * D
    for dv_start in range(0, D, BLOCK_DV):
        dv_offs = dv_start + tl.arange(0, BLOCK_DV)
        dv_valid = dv_offs < D
        acc = tl.zeros([BLOCK_G, BLOCK_DV], dtype=tl.float32)
        for blk_start in range(0, topK, BLOCK_K):
            blk_offs = blk_start + tl.arange(0, BLOCK_K)
            blk_in_count = blk_offs < topK
            tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
            tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
            tok_clamped = tl.where(tok_valid, tok, 0)

            scores = _sfa_scores_block(
                q_ptr, q_base, qr_ptr, qr_base,
                k_ptr, k_base, kr_ptr, kr_base,
                tok_clamped, tok_valid, g_offs, g_valid,
                scale_value, D, D_ROPE, BLOCK_G, BLOCK_K, BLOCK_D)

            p = tl.exp(scores - m_final[:, None])
            p = tl.where(tok_valid[None, :], p, 0.0)
            v_tile = tl.load(
                v_ptr + v_base + tok_clamped[:, None] * D + dv_offs[None, :],
                mask=tok_valid[:, None] & dv_valid[None, :], other=0.0)
            acc += tl.dot(p.to(v_tile.dtype), v_tile)

        out_tile = acc / l_safe[:, None]
        tl.store(
            out_ptr + out_base + g_offs[:, None] * D + dv_offs[None, :],
            out_tile.to(out_ptr.dtype.element_ty),
            mask=g_valid[:, None] & dv_valid[None, :] & row_active)

    if return_lse:
        # softmax_max/sum layout (B, N2=1, S1, N1) -> flat (b*S1+s1)*N1 + g.
        # Empty/hidden rows (l_i==0) keep the pre-filled 0 to match the golden.
        sm_base = (b * S1 + s1) * N1
        store_mask = g_valid & row_active & (l_i > 0.0)
        tl.store(sm_max_ptr + sm_base + g_offs, m_i, mask=store_mask)
        tl.store(sm_sum_ptr + sm_base + g_offs, l_i, mask=store_mask)


# ---------------------------------------------------------------------------
# High-performance prefill kernel (from kkb experience)
# Requires: triton_ascend.language.extra (tle) with sub_vec_id/num and
#           index_select_simd. Falls back to _sfa_kernel if unavailable.
# ---------------------------------------------------------------------------

@triton.jit
def cube1(
    Q_ptr,
    Wksp_K_ptr,
    stride_qt1,
    stride_qn2,
    stride_qn1,
    stride_qd,
    stride_wkt1,
    stride_wkn2,
    stride_wksbs,
    stride_wkd,
    D_qk: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_SBS: tl.constexpr,
    offs_G,
    offs_sbs,
    q_base,
    wk_base,
    BLOCK_K: tl.constexpr,
):
    loop_times: tl.constexpr = (D_qk + BLOCK_K - 1) // BLOCK_K

    qk_acc = tl.zeros([BLOCK_G, BLOCK_SBS], dtype=tl.float32)

    for idx_D in range(0, loop_times):
        offs_D = idx_D * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_D = offs_D < D_qk

        l1_q = tl.load(
            Q_ptr
            + (q_base + offs_G[:, None] * stride_qn1 + offs_D[None, :] * stride_qd),
            mask=mask_D[None, :],
        )

        l1_k = tl.load(
            Wksp_K_ptr
            + (
                wk_base
                + offs_sbs[:, None] * stride_wksbs
                + offs_D[None, :] * stride_wkd
            ),
            mask=mask_D[None, :],
        )

        qk_acc += tl.dot(l1_q, l1_k.trans())

    return qk_acc


@triton.jit(do_not_specialize=["T1", "T2", "S1", "S2"])
def sparse_flash_attention_prefill_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Indices_ptr,
    O_ptr,
    Wksp_K_ptr,
    Wksp_K2_ptr,
    Wksp_qk_ptr,
    Wksp_score_ptr,
    Wksp_sv_ptr,
    stride_qt1,
    stride_qn2,
    stride_qn1,
    stride_qd,
    stride_it1,
    stride_in2,
    stride_ik,
    stride_ot1,
    stride_on1,
    stride_wod,
    stride_wkt1,
    stride_wkn2,
    stride_wksbs,
    stride_wkd,
    stride_wqkt1,
    stride_wqkn1,
    stride_wqks2,
    stride_wscoret1,
    stride_wscoren1,
    stride_wscores2,
    stride_wsvt1,
    stride_wsvn1,
    stride_wsvd,
    scale,
    T1,
    N1: tl.constexpr,
    T2,
    G: tl.constexpr,
    D_qk: tl.constexpr,
    D_v: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    S1,
    S2,
    bs1_per_core,
    BLOCK_G: tl.constexpr,
    BLOCK_SBS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    t1_start = pid * bs1_per_core

    t1_end = tl.minimum(T1, (pid + 1) * bs1_per_core)

    for idx_t1 in range(t1_start, t1_end):
        idx_b = idx_t1 // S1
        cur_s2 = S2
        beg_s2 = idx_b * S2

        topk_base = idx_t1 * stride_it1
        q_base = idx_t1 * stride_qt1

        wk_base = pid * stride_wkt1
        wdq_base = pid * stride_wqkt1
        wdqt_base = pid * stride_wscoret1

        wsv_base = pid * stride_wsvt1
        wsacc_base = idx_t1 * stride_ot1

        sub_vec_id = tle.sub_vec_id()
        sub_vec_num = tle.sub_vec_num()
        CACHE_BLOCK: tl.constexpr = (
            64
        )
        offs_sub_Dv = tl.arange(0, D_qk)
        sliced_row_num = BLOCK_SBS // sub_vec_num
        row_offset = sliced_row_num * sub_vec_id

        HEAD_LOOP_TIMES: tl.constexpr = triton.cdiv(G, BLOCK_G)
        BLOCK_SUB_G: tl.constexpr = BLOCK_G // 2
        sbs_loop_times = tl.cdiv(K, BLOCK_SBS)
        for idx_n1_blk in range(0, HEAD_LOOP_TIMES):
            offs_G = idx_n1_blk * BLOCK_G + tl.arange(0, BLOCK_G)
            arange_sub_G = sub_vec_id * BLOCK_SUB_G + tl.arange(0, BLOCK_SUB_G)
            offs_sub_G = idx_n1_blk * BLOCK_G + arange_sub_G

            lse = tl.zeros([BLOCK_SUB_G], dtype=tl.float32) - float("inf")

            for idx_topk_block in range(0, sliced_row_num, CACHE_BLOCK):
                idx_offset = row_offset + idx_topk_block + tl.arange(0, CACHE_BLOCK)
                sparse_ids = (
                    tl.load(Indices_ptr + topk_base + idx_offset * stride_ik) + beg_s2
                )
                result = tle.index_select_simd(
                    K_ptr,
                    dim=0,
                    index=sparse_ids,
                    src_shape=[cur_s2 * B, D_qk],
                    src_offset=[-1, 0],
                    read_shape=[-1, D_qk],
                )
                tl.store(
                    Wksp_K_ptr
                    + wk_base
                    + idx_offset[:, None] * stride_wksbs
                    + offs_sub_Dv[None, :] * stride_wkd,
                    result,
                )

            for idx_sub_sbs in range(0, sbs_loop_times):
                offs_sbs = idx_sub_sbs * BLOCK_SBS + tl.arange(0, BLOCK_SBS)

                qk_l1 = cube1(
                    Q_ptr,
                    Wksp_K_ptr,
                    stride_qt1,
                    stride_qn2,
                    stride_qn1,
                    stride_qd,
                    stride_wkt1,
                    stride_wkn2,
                    stride_wksbs,
                    stride_wkd,
                    D_qk,
                    BLOCK_G,
                    BLOCK_SBS,
                    offs_G,
                    offs_sbs,
                    q_base,
                    wk_base,
                    BLOCK_K,
                )
                tl.store(
                    Wksp_qk_ptr
                    + wdq_base
                    + tl.arange(0, BLOCK_G)[:, None] * stride_wqkn1
                    + tl.arange(0, BLOCK_SBS)[None, :] * stride_wqks2,
                    qk_l1,
                )

                qk = tl.load(
                    Wksp_qk_ptr
                    + wdq_base
                    + arange_sub_G[:, None] * stride_wqkn1
                    + tl.arange(0, BLOCK_SBS)[None, :] * stride_wqks2
                )
                qk *= scale

                mask = offs_sbs < K
                qk = tl.where(mask[None, :], qk, -float("inf"))
                local_max = tl.max(qk, 1)
                local_exp = tl.exp(qk - local_max[:, None])
                local_sum = tl.sum(local_exp, 1)
                local_lse = local_max + tl.log(local_sum)
                mean_lse = (lse + local_lse) / 2
                new_lse = mean_lse + tl.log(
                    tl.exp(lse - mean_lse) + tl.exp(local_lse - mean_lse)
                )
                new_lse = tl.where(new_lse != new_lse, local_lse, new_lse)
                current_weight = tl.exp(local_lse - new_lse)
                score = local_exp / local_sum[:, None]
                tl.store(
                    Wksp_score_ptr
                    + wdqt_base
                    + arange_sub_G[:, None] * stride_wscoren1
                    + tl.arange(0, BLOCK_SBS)[None, :] * stride_wscores2,
                    score.to(Q_ptr.dtype.element_ty),
                )

                blk_v_loop_time: tl.constexpr = triton.cdiv(D_v, BLOCK_V)

                start_row = (idx_sub_sbs + 1) * BLOCK_SBS
                end_row = tl.minimum(start_row + sliced_row_num, K)
                for idx_topk_block in range(start_row, end_row, CACHE_BLOCK):
                    idx_offset = row_offset + idx_topk_block + tl.arange(0, CACHE_BLOCK)
                    sparse_ids = (
                        tl.load(Indices_ptr + topk_base + idx_offset * stride_ik)
                        + beg_s2
                    )
                    result = tle.index_select_simd(
                        K_ptr,
                        dim=0,
                        index=sparse_ids,
                        src_shape=[cur_s2 * B, D_qk],
                        src_offset=[-1, 0],
                        read_shape=[-1, D_qk],
                    )
                    tl.store(
                        Wksp_K2_ptr
                        + wk_base
                        + idx_offset[:, None] * stride_wksbs
                        + offs_sub_Dv[None, :] * stride_wkd,
                        result,
                    )

                for idx_blk_v in range(0, blk_v_loop_time):
                    offset_V = idx_blk_v * BLOCK_V + tl.arange(0, BLOCK_V)
                    kv_cache = tl.load(
                        Wksp_K_ptr
                        + wk_base
                        + offs_sbs[:, None] * stride_wksbs
                        + offset_V[None, :] * stride_wkd
                    )
                    score_l1 = tl.load(
                        Wksp_score_ptr
                        + wdq_base
                        + tl.arange(0, BLOCK_G)[:, None] * stride_wqkn1
                        + tl.arange(0, BLOCK_SBS)[None, :] * stride_wqks2
                    )
                    sv = tl.dot(score_l1, kv_cache)
                    tl.store(
                        Wksp_sv_ptr
                        + wsv_base
                        + offs_G[:, None] * stride_wsvn1
                        + offset_V[None, :] * stride_wsvd,
                        sv,
                    )

                    acc = tl.load(
                        O_ptr
                        + wsacc_base
                        + offs_sub_G[:, None] * stride_on1
                        + offset_V[None, :] * stride_wod
                    )
                    sv_sub = tl.load(
                        Wksp_sv_ptr
                        + wsv_base
                        + offs_sub_G[:, None] * stride_wsvn1
                        + offset_V[None, :] * stride_wsvd
                    )
                    acc = acc * tl.exp(lse - new_lse)[:, None]

                    acc += sv_sub * current_weight[:, None]

                    tl.store(
                        O_ptr
                        + wsacc_base
                        + offs_sub_G[:, None] * stride_on1
                        + offset_V[None, :] * stride_wod,
                        acc,
                    )

                lse = new_lse


# ---------------------------------------------------------------------------
# host-side helpers
# ---------------------------------------------------------------------------
def _default_actual_seq_lens(actual_seq_lens, batch_size, seq_len):
    # None -> full seq; list/tuple -> int32 tensor; else pass-through.
    return ms.ops.fill(ms.int32, (batch_size,), seq_len) if actual_seq_lens is None else \
           ms.Tensor(list(actual_seq_lens), dtype=ms.int32) if isinstance(actual_seq_lens, (list, tuple)) else \
           actual_seq_lens


def _tnd_cumsum_to_per_batch(cumsum):
    # TND actual_seq_lengths are cumulative prefix sums; diff back to per-batch.
    return cumsum - ops.pad(cumsum[:-1], (1, 0))


def _tnd_to_bsnd(tensor, act_seq_per_batch):
    """[T, N, ...] -> [B, max_S, N, ...]; PyNative-only (data-dependent slicing)."""
    assert ms.get_context('mode') == ms.PYNATIVE_MODE, "TND path is PyNative-only."
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    max_seq = max(lengths) if lengths else 0
    out = ms.ops.zeros((B, max_seq, *tensor.shape[1:]), dtype=tensor.dtype)
    start = 0
    for b_idx in range(B):
        length = lengths[b_idx]
        if length > 0:
            out[b_idx, :length] = tensor[start:start + length]
            start += length
    return out


def _bsnd_to_tnd(tensor, act_seq_per_batch):
    """[B, S, N, ...] -> [T, N, ...]; inverse of _tnd_to_bsnd (PyNative-only)."""
    assert ms.get_context('mode') == ms.PYNATIVE_MODE, "TND path is PyNative-only."
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    total_t = sum(lengths)
    out = ms.ops.zeros((total_t, *tensor.shape[2:]), dtype=tensor.dtype)
    start = 0
    for b_idx in range(B):
        length = lengths[b_idx]
        if length > 0:
            out[start:start + length] = tensor[b_idx, :length]
            start += length
    return out


def _pa_to_bsnd(cache, block_table, act_k_per_batch, max_s2):
    """PageAttention cache [block_num, block_size, N, Dx] -> dense [B, max_s2, N, Dx].

    Reverses paging using block_table[B, max_blocks] (PyNative-only). -1 block ids
    are skipped. Mirrors the golden's tensor_to_pa inverse.
    """
    assert ms.get_context('mode') == ms.PYNATIVE_MODE, "PA_BSND path is PyNative-only."
    block_num, block_size, N, Dx = cache.shape
    B = block_table.shape[0]
    out = ms.ops.zeros((B, max_s2, N, Dx), dtype=cache.dtype)
    bt = block_table.asnumpy()
    for b in range(B):
        for blk_i in range(block_table.shape[1]):
            blk_id = int(bt[b, blk_i])
            if blk_id == -1:
                continue
            dst = blk_i * block_size
            if dst >= max_s2:
                break
            end = min(dst + block_size, max_s2)
            out[b, dst:end] = cache[blk_id, :end - dst]
    return out


def _expand_block_indices(sparse_indices, sparse_block_size):
    """Block ids [.., topK] -> token indices [.., topK*block_size].

    Each block id b expands to tokens [b*bs, b*bs+bs); -1 stays -1. The kernel
    then treats the result as plain token indices (token-wise path). Pure static
    tensor ops, so this works in GRAPH_MODE as well as PyNative.
    """
    if sparse_block_size == 1:
        return sparse_indices
    bs = sparse_block_size
    base = sparse_indices.astype(ms.int32)
    *lead, topK = base.shape
    base = base.reshape(*lead, topK, 1)
    offs = ms.ops.arange(0, bs, dtype=ms.int32).reshape(*([1] * len(lead)), 1, bs)
    tokens = ms.ops.where(base == -1, ms.Tensor(-1, ms.int32), base * bs + offs)
    return tokens.reshape(*lead, topK * bs)


# ---------------------------------------------------------------------------
# _ms_pyfunc core (launches the triton kernel). Type annotations are required by
# _ms_pyfunc shape/dtype inference and must match between infer_func and core.
# ---------------------------------------------------------------------------
def _infer_sfa(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    out_buf: ms.Tensor, sm_max_buf: ms.Tensor, sm_sum_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
    return_lse: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    return (ms.mint.empty_like(out_buf),
            ms.mint.empty_like(sm_max_buf),
            ms.mint.empty_like(sm_sum_buf))


@ms.ops._ms_pyfunc(infer_func=_infer_sfa)
def _sfa_core(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    out_buf: ms.Tensor, sm_max_buf: ms.Tensor, sm_sum_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
    return_lse: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    # grid both dims pow2-padded (Ascend traps on non-pow2 grid); out-of-range
    # programs idle via in_range masks. Padding must match _prune_configs.
    def grid_fn(meta): return (
        _next_pow2(B_S1),
        _next_pow2(triton.cdiv(N1, meta["BLOCK_G"])),
    )

    _sfa_kernel[grid_fn](
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        out_buf, sm_max_buf, sm_sum_buf,
        act_q, act_k,
        B_S1, S1, S2, N1, topK,
        D, D_ROPE,
        scale_value,
        sparse_mode=sparse_mode,
        return_lse=return_lse,
    )
    return out_buf, sm_max_buf, sm_sum_buf


def _launch_prefill_kernel(
    q_bsnd, qr_bsnd, k_bsnd, kr_bsnd, si_tok,
    B, S1, S2, N1, D, D_ROPE, topK, scale_value,
):
    D_qk = D + D_ROPE
    T1 = B * S1
    T2 = B * S2
    G = N1

    Q_full = ms.ops.cat([q_bsnd, qr_bsnd], axis=-1)
    Q_full = Q_full.reshape(T1, N1, D_qk).contiguous()

    K_full_bsnd = ms.ops.cat([k_bsnd, kr_bsnd], axis=-1)
    K_full = K_full_bsnd.reshape(T2, D_qk).contiguous()

    V_flat = k_bsnd.reshape(T2, D).contiguous()

    sparse_flat = si_tok.reshape(T1, topK).to(ms.int32).contiguous()

    BLOCK_G = min(N1, 64)
    if BLOCK_G < 2:
        raise ValueError(
            f"prefill kernel requires N1 >= 2 for BLOCK_SUB_G, got N1={N1}"
        )
    BLOCK_SBS = 128
    BLOCK_K = 128
    BLOCK_V = 128
    BLOCK_SUB_G = BLOCK_G // 2

    bs1_per_core = max(1, triton.cdiv(T1, 32))
    num_cores = _next_pow2(triton.cdiv(T1, bs1_per_core))
    grid = (num_cores,)

    Wksp_K = ms.mint.zeros((num_cores, topK, D_qk), dtype=q_bsnd.dtype).to('Ascend')
    Wksp_qk = ms.mint.zeros((num_cores, N1, BLOCK_SBS), dtype=ms.float32).to('Ascend')
    Wksp_score = ms.mint.zeros((num_cores, BLOCK_SUB_G, BLOCK_SBS), dtype=q_bsnd.dtype).to('Ascend')
    Wksp_sv = ms.mint.zeros((num_cores, N1, D), dtype=ms.float32).to('Ascend')

    out_buf = ms.mint.zeros((T1, N1, D), dtype=q_bsnd.dtype).to('Ascend')

    stride_qt1 = N1 * D_qk
    stride_qn2 = N1 * D_qk
    stride_qn1 = D_qk
    stride_qd = 1

    stride_it1 = topK
    stride_in2 = topK
    stride_ik = 1

    stride_ot1 = N1 * D
    stride_on1 = D
    stride_wod = 1

    stride_wkt1 = topK * D_qk
    stride_wkn2 = topK * D_qk
    stride_wksbs = D_qk
    stride_wkd = 1

    stride_wqkt1 = N1 * BLOCK_SBS
    stride_wqkn1 = BLOCK_SBS
    stride_wqks2 = 1

    stride_wscoret1 = BLOCK_SUB_G * BLOCK_SBS
    stride_wscoren1 = BLOCK_SBS
    stride_wscores2 = 1

    stride_wsvt1 = N1 * D
    stride_wsvn1 = D
    stride_wsvd = 1

    sparse_flash_attention_prefill_kernel[grid](
        Q_full, K_full, V_flat, sparse_flat, out_buf,
        Wksp_K, Wksp_K,
        Wksp_qk, Wksp_score, Wksp_sv,
        stride_qt1, stride_qn2, stride_qn1, stride_qd,
        stride_it1, stride_in2, stride_ik,
        stride_ot1, stride_on1, stride_wod,
        stride_wkt1, stride_wkn2, stride_wksbs, stride_wkd,
        stride_wqkt1, stride_wqkn1, stride_wqks2,
        stride_wscoret1, stride_wscoren1, stride_wscores2,
        stride_wsvt1, stride_wsvn1, stride_wsvd,
        scale_value,
        T1, N1, T2, G,
        D_qk, D, topK,
        B, S1, S2,
        bs1_per_core,
        BLOCK_G=BLOCK_G, BLOCK_SBS=BLOCK_SBS,
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    out_buf = out_buf.reshape(B, S1, N1, D)

    sm_max_buf = ms.mint.zeros((B, 1, S1, N1), dtype=ms.float32).to('Ascend')
    sm_sum_buf = ms.mint.zeros((B, 1, S1, N1), dtype=ms.float32).to('Ascend')

    return out_buf, sm_max_buf, sm_sum_buf


# ---------------------------------------------------------------------------
# public API — aligned with MindSpore ops.sparse_flash_attention signature
# ---------------------------------------------------------------------------
def sparse_flash_attention_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    sparse_indices: ms.Tensor,
    scale_value: float = 1.0,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 0,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
    attention_mode: int = 2,
    return_softmax_lse: bool = False,
    block_size: int = 0,
    use_prefill_kernel: bool = False,
):
    """Drop-in replacement for ops.sparse_flash_attention (MLA-absorb, MQA).

    Args:
        query: [B,S1,N1,D] BSND or [T1,N1,D] TND, fp16/bf16 (D in 128/256/512)
        key:   [B,S2,1,D] BSND / [T2,1,D] TND / [block_num,block_size,1,D] PA_BSND
        value: same layout/shape as key; IGNORED in MLA-absorb mode (value is
               taken as key[..,:D]). Kept for API parity with CANN.
        sparse_indices: [B,S1,1,sparse_count] (or TND) int32, block ids, -1 invalid
        scale_value: softmax scale (1/sqrt(d_k))
        block_table: [B, max_blocks] int32, required for PA_BSND
        actual_seq_lengths_query/kv: [B] int32 / list / None (cumulative for TND)
        query_rope/key_rope: [.,N,Dr] (Dr=64), required (no empty rope)
        sparse_block_size: 1 (token-wise) or 2^n in [1,128] (block-wise)
        layout_query: "BSND" / "TND"
        layout_kv: "BSND" / "TND" / "PA_BSND"
        sparse_mode: 0 (full) / 3 (rightDownCausal)
        pre_tokens/next_tokens: only default INT64_MAX supported
        attention_mode: only 2 (MLA-absorb) supported
        return_softmax_lse: also return softmax_max / softmax_sum
        block_size: PageAttention block token count (PA_BSND only)
        use_prefill_kernel: if True, use the high-performance prefill kernel
            (requires triton_ascend.language.extra; only supports sparse_mode=0
            and BSND layout; softmax_max/sum will be zeros). If False (default),
            use the original two-pass kernel.

    Returns:
        (attention_out, softmax_max, softmax_sum)
        softmax_max/sum: (B,1,S1,N1) BSND / (1,T1,N1) TND; zeros if not requested.
    """
    if attention_mode != 2:
        raise ValueError("Only attention_mode=2 (MLA-absorb) is supported")
    if pre_tokens != INT64_MAX or next_tokens != INT64_MAX:
        raise ValueError("pre_tokens/next_tokens only support default INT64_MAX")
    if sparse_mode not in (0, 3):
        raise ValueError("Only sparse_mode 0 (full) / 3 (rightDownCausal) supported")
    if query_rope is None or key_rope is None:
        raise ValueError("query_rope and key_rope are required (no empty rope)")
    if sparse_block_size < 1 or (sparse_block_size & (sparse_block_size - 1)) != 0 \
            or sparse_block_size > 128:
        raise ValueError("sparse_block_size must be a power of 2 in [1,128]")

    is_tnd_q = (layout_query == "TND")
    is_tnd_kv = (layout_kv == "TND")
    is_pa = (layout_kv == "PA_BSND")

    # Pre-init cross-branch vars: GRAPH_MODE's parser checks variable definition
    # across all branches (before dead-branch elimination), so anything used after
    # the merge must be defined in every path even when only BSND runs at runtime.
    act_q_pb = None
    act_k_pb = None

    # ---- normalize all layouts to dense BSND ----
    if is_tnd_q:
        act_q_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_query)
        q_bsnd = _tnd_to_bsnd(query, act_q_pb)
        qr_bsnd = _tnd_to_bsnd(query_rope, act_q_pb)
        si_bsnd = _tnd_to_bsnd(sparse_indices, act_q_pb)
    else:
        q_bsnd, qr_bsnd, si_bsnd = query, query_rope, sparse_indices

    if is_tnd_kv:
        act_k_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_kv)
        k_bsnd = _tnd_to_bsnd(key, act_k_pb)
        kr_bsnd = _tnd_to_bsnd(key_rope, act_k_pb)
    elif is_pa:
        if block_table is None:
            raise ValueError("PA_BSND requires block_table")
        act_k_list = list(actual_seq_lengths_kv)
        pa_block_size = block_size or key.shape[1]
        max_s2 = block_table.shape[1] * pa_block_size
        k_bsnd = _pa_to_bsnd(key, block_table, act_k_list, max_s2)
        kr_bsnd = _pa_to_bsnd(key_rope, block_table, act_k_list, max_s2)
    else:
        k_bsnd, kr_bsnd = key, key_rope

    B, S1, N1, D = q_bsnd.shape
    S2 = k_bsnd.shape[1]
    D_ROPE = qr_bsnd.shape[-1]

    if D not in _VALID_D:
        raise ValueError(f"D must be one of {_VALID_D}, got {D}")
    if D_ROPE != _D_ROPE:
        raise ValueError(f"rope dim must be {_D_ROPE}, got {D_ROPE}")
    if k_bsnd.shape[2] != 1:
        raise ValueError("Only N2=1 (MQA) is supported")
    if N1 not in _VALID_N1:
        raise ValueError(f"N1 must be one of {_VALID_N1}, got {N1}")

    act_q = _default_actual_seq_lens(
        None if is_tnd_q else actual_seq_lengths_query, B, S1)
    act_k = _default_actual_seq_lens(
        None if (is_tnd_kv or is_pa) else actual_seq_lengths_kv, B, S2)
    if is_tnd_q:
        act_q = act_q_pb.to(ms.int32)
    if is_tnd_kv:
        act_k = act_k_pb.to(ms.int32)
    elif is_pa:
        act_k = ms.Tensor(list(actual_seq_lengths_kv), dtype=ms.int32)

    # block-wise -> token-wise indices (kernel is purely token-wise)
    si_tok = _expand_block_indices(si_bsnd, sparse_block_size)
    topK = si_tok.shape[-1]

    # ---- dispatch: prefill kernel vs original two-pass kernel ----
    _can_prefill = (
        use_prefill_kernel
        and _PREFILL_KERNEL_AVAILABLE
        and sparse_mode == 0
        and not is_tnd_q
        and not is_tnd_kv
        and not is_pa
        and N1 >= 2
    )

    if _can_prefill:
        out_buf, sm_max_buf, sm_sum_buf = _launch_prefill_kernel(
            q_bsnd, qr_bsnd, k_bsnd, kr_bsnd, si_tok,
            B, S1, S2, N1, D, D_ROPE, topK, float(scale_value),
        )
    else:
        if use_prefill_kernel and not _can_prefill:
            import warnings
            warnings.warn(
                "use_prefill_kernel=True but prefill kernel cannot be used "
                f"(_PREFILL_KERNEL_AVAILABLE={_PREFILL_KERNEL_AVAILABLE}, "
                f"sparse_mode={sparse_mode}, is_tnd_q={is_tnd_q}, "
                f"is_tnd_kv={is_tnd_kv}, is_pa={is_pa}, N1={N1}). "
                "Falling back to original kernel.",
                stacklevel=2,
            )

        q_flat = q_bsnd.contiguous()
        qr_flat = qr_bsnd.contiguous()
        k_flat = k_bsnd.reshape(B * S2, D).contiguous()
        kr_flat = kr_bsnd.reshape(B * S2, D_ROPE).contiguous()
        v_flat = k_flat
        sparse_flat = si_tok.reshape(B * S1, topK).to(ms.int32).contiguous()

        out_buf = ms.mint.zeros((B, S1, N1, D), dtype=q_bsnd.dtype).to('Ascend')
        sm_max_buf = ms.mint.zeros((B, 1, S1, N1), dtype=ms.float32).to('Ascend')
        sm_sum_buf = ms.mint.zeros((B, 1, S1, N1), dtype=ms.float32).to('Ascend')

        out_buf, sm_max_buf, sm_sum_buf = _sfa_core(
            q_flat, qr_flat,
            k_flat, kr_flat, v_flat,
            sparse_flat,
            out_buf, sm_max_buf, sm_sum_buf,
            act_q.to('Ascend'), act_k.to('Ascend'),
            B * S1, S1, S2, N1, topK,
            D, D_ROPE,
            float(scale_value),
            sparse_mode,
            1 if return_softmax_lse else 0,
        )

    if is_tnd_q:
        attention_out = _bsnd_to_tnd(out_buf, act_q_pb)
        # softmax (B,1,S1,N1) -> (1,T1,N1): squeeze N2, tnd-pack S1, restore N2 axis
        sm_max = _bsnd_to_tnd(sm_max_buf.transpose(0, 2, 1, 3), act_q_pb).transpose(1, 0, 2)
        sm_sum = _bsnd_to_tnd(sm_sum_buf.transpose(0, 2, 1, 3), act_q_pb).transpose(1, 0, 2)
    else:
        attention_out = out_buf
        sm_max, sm_sum = sm_max_buf, sm_sum_buf

    return attention_out, sm_max, sm_sum


class SparseFlashAttentionTriton(ms.nn.Cell):
    """nn.Cell wrapper around sparse_flash_attention_triton; see that function."""

    def __init__(
        self,
        scale_value=1.0,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=False,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        block_size=0,
        use_prefill_kernel=False,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.sparse_block_size = sparse_block_size
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.sparse_mode = sparse_mode
        self.attention_mode = attention_mode
        self.return_softmax_lse = return_softmax_lse
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.block_size = block_size
        self.use_prefill_kernel = use_prefill_kernel

    def construct(
        self,
        query, key, value, sparse_indices,
        query_rope=None, key_rope=None,
        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
        block_table=None,
    ):
        return sparse_flash_attention_triton(
            query, key, value, sparse_indices,
            scale_value=self.scale_value,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope, key_rope=key_rope,
            sparse_block_size=self.sparse_block_size,
            layout_query=self.layout_query,
            layout_kv=self.layout_kv,
            sparse_mode=self.sparse_mode,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            attention_mode=self.attention_mode,
            return_softmax_lse=self.return_softmax_lse,
            block_size=self.block_size,
            use_prefill_kernel=self.use_prefill_kernel,
        )
