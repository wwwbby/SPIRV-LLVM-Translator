import torch
import triton
from triton._C.libtriton import ir, ascend_ir
import triton.language as tl
import triton.extension.ascend.language as al
import triton.extension.buffer.language as bl
import torch_npu


def create_mla_prefill_mask(
    bsz: int,
    n_heads: int,
    seqlen: int,
    topk_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    causal_mask = torch.tril(torch.ones((seqlen, seqlen), device=device, dtype=torch.bool))
    top_k = topk_indices.shape[-1]
    sparse_mask = torch.zeros((bsz, seqlen, seqlen), device=device, dtype=torch.bool)
    row_indices = torch.arange(seqlen, device=device).view(1, seqlen, 1).expand(bsz, -1, top_k)
    sparse_mask.scatter_(2, topk_indices, True)
    combined_mask = causal_mask.unsqueeze(0) | sparse_mask
    return combined_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)


def mla_prefill_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    combined_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    v_fp32 = v.to(torch.float32)
    scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * softmax_scale
    scores.masked_fill_(~combined_mask, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1).to(v_fp32.dtype)
    output = torch.matmul(attn_weights, v_fp32)
    return output.to(q.dtype)


@triton.jit
def mla_prefill_kernel(
    Q,
    K,
    V,
    Out,
    Combined_Mask,
    softmax_scale,
    stride_q_bsz,
    stride_q_h,
    stride_q_seq,
    stride_k_bsz,
    stride_k_h,
    stride_k_seq,
    stride_v_bsz,
    stride_v_h,
    stride_v_seq,
    stride_out_bsz,
    stride_out_h,
    stride_out_seq,
    stride_mask_bsz,
    stride_mask_h,
    stride_mask_q,
    stride_mask_k,
    N_HEADS: tl.constexpr,
    SEQLEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    BLOCK_M_SUB: tl.constexpr = BLOCK_M // 2
    start_m = tl.program_id(1) * BLOCK_M
    program_id_bh = tl.program_id(0)
    bsz_idx = program_id_bh // N_HEADS
    head_idx = program_id_bh % N_HEADS
    offs_m = start_m + tl.arange(0, BLOCK_M_SUB)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptr = Q + bsz_idx * stride_q_bsz + head_idx * stride_q_h + offs_m[:, None] * stride_q_seq + offs_d[None, :]
    k_ptr = K + bsz_idx * stride_k_bsz + head_idx * stride_k_h + offs_n[:, None] * stride_k_seq + offs_d[None, :]
    v_ptr = V + bsz_idx * stride_v_bsz + head_idx * stride_v_h + offs_n[:, None] * stride_v_seq + offs_d[None, :]
    mask_ptr_base = Combined_Mask + bsz_idx * stride_mask_bsz + head_idx * stride_mask_h
    sub_vec_id = al.sub_vec_id()
    acc = tl.zeros([BLOCK_M_SUB, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M_SUB], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M_SUB], dtype=tl.float32)
    mask_q = offs_m < SEQLEN
    q = tl.load(q_ptr, mask=mask_q[:, None], other=0.0).to(tl.float32)
    p_ij_l1 = bl.allocate_local_buffer(tl.float32, [BLOCK_M, BLOCK_N], al.address_space.L1)
    s_ij_sub = bl.allocate_local_buffer(tl.float32, [BLOCK_M_SUB, BLOCK_N], al.address_space.UB)
    out_sub = bl.allocate_local_buffer(tl.float32, [BLOCK_M_SUB, BLOCK_N], al.address_space.UB)

    with al.Scope(core_mode="cube"):
        for start_n in range(0, SEQLEN, BLOCK_N):
            mask_k_boundary = start_n + offs_n < SEQLEN
            # no wait
            k = tl.load(k_ptr, mask=mask_k_boundary[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=mask_k_boundary[:, None], other=0.0).to(tl.float32)
            s_ij = tl.dot(q, tl.trans(k))   # L0C
            al.fixpipe(s_ij, s_ij_sub, dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
            al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

            al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
            p_ij_tensor = bl.to_tensor(p_ij_l1)
            p_out = tl.dot(p_ij_tensor, v)
            al.fixpipe(p_out, out_sub, dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
            al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            k_ptr = k_ptr + BLOCK_N * stride_k_seq
            v_ptr = v_ptr + BLOCK_N * stride_v_seq

    with al.Scope(core_mode="vector"):
        for start_n in range(0, SEQLEN, BLOCK_N):
            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            mask_k_boundary = start_n + offs_n < SEQLEN
            s_ij = bl.to_tensor(s_ij_sub)
            s_ij *= softmax_scale
            mask_ptr = mask_ptr_base + (sub_vec_id * BLOCK_M_SUB + offs_m[:, None]) * stride_mask_q + offs_n[None, :] * stride_mask_k
            final_valid_pos_mask = tl.load(mask_ptr, mask=(mask_q[:, None] & mask_k_boundary[None, :]), other=False)
            s_ij = tl.where(final_valid_pos_mask, s_ij, -float("inf"))
            m_ij = tl.max(s_ij, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            p_ij = tl.exp(s_ij - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p_ij, axis=1)
            p_ij = p_ij.to(tl.float32)
            acc_scaled = acc * alpha[:, None]
            p_ij_ub = bl.to_buffer(p_ij, al.address_space.UB)
            p_ij_l1_half = bl.subview(p_ij_l1, [sub_vec_id * BLOCK_M_SUB, 0], [BLOCK_M_SUB, BLOCK_N], [1, 1])
            al.copy_from_ub_to_l1(p_ij_ub, p_ij_l1_half)
            al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)

            al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            out_tensor = bl.to_tensor(out_sub)
            acc = acc_scaled + out_tensor
            l_i = l_new
            m_i = m_new
        acc = acc / l_i[:, None]

        offs_m_sub = sub_vec_id * BLOCK_M_SUB + tl.arange(0, BLOCK_M_SUB)[:, None]
        out_ptr = Out + bsz_idx * stride_out_bsz + head_idx * stride_out_h + offs_m_sub * stride_out_seq + offs_d[None, :]
        tl.store(out_ptr, acc, mask=mask_q[:, None])


def mla_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    combined_mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    bsz, n_heads, seqlen, head_dim = q.shape
    output = torch.empty_like(q, dtype=v.dtype)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    combined_mask = combined_mask.contiguous()
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (bsz * n_heads, triton.cdiv(seqlen, BLOCK_M))
    mla_prefill_kernel[grid](
        q,
        k,
        v,
        output,
        combined_mask,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        combined_mask.stride(0),
        combined_mask.stride(1),
        combined_mask.stride(2),
        combined_mask.stride(3),
        N_HEADS=n_heads,
        SEQLEN=seqlen,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(1)
    BSZ, N_HEADS, SEQLEN_PREFILL, HEAD_DIM, TOP_K = 2, 8, 256, 64, 32
    device = "npu"
    dtype = torch.bfloat16
    q = torch.randn(BSZ, N_HEADS, SEQLEN_PREFILL, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(BSZ, N_HEADS, SEQLEN_PREFILL, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(BSZ, N_HEADS, SEQLEN_PREFILL, HEAD_DIM, device=device, dtype=dtype)
    topk_indices = torch.randint(0, SEQLEN_PREFILL, (BSZ, SEQLEN_PREFILL, TOP_K), device=device)
    softmax_scale = HEAD_DIM**-0.5
    combined_mask = create_mla_prefill_mask(
        bsz=BSZ,
        n_heads=N_HEADS,
        seqlen=SEQLEN_PREFILL,
        topk_indices=topk_indices,
        device=device,
    )
    output_pytorch = mla_prefill_pytorch(q, k, v, combined_mask, softmax_scale)
    output_triton = mla_prefill_triton(q, k, v, combined_mask, softmax_scale)
    max_abs_err = torch.max(torch.abs(output_pytorch - output_triton)).item()
    print(f"Max Absolute Error: {max_abs_err:.6f}")
