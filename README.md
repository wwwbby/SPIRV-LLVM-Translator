import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu
import math


# ============================================================
# Kernel 1: reduce-last (inner_size == 1)
# reduce 轴在最后维度，可直接连续访存
# ============================================================

@triton.jit
def sum_kernel_reduce_last(
    input_ptr, output_ptr,
    outer_size, reduce_size,
    BLOCK_RED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    tiles_per_pid = (outer_size + num_pids - 1) // num_pids
    tile_start = pid * tiles_per_pid
    tile_end = tl.minimum(tile_start + tiles_per_pid, outer_size)

    for outer_idx in range(tile_start, tile_end):
        acc = tl.full((), 0.0, tl.float32)
        base_offset = outer_idx * reduce_size

        for r_start in range(0, reduce_size, BLOCK_RED):
            r_end = tl.minimum(r_start + BLOCK_RED, reduce_size)
            r_offsets = r_start + tl.arange(0, BLOCK_RED)
            r_mask = r_offsets < r_end

            vals = tl.load(input_ptr + base_offset + r_offsets, mask=r_mask, other=0.0)
            vals = vals.to(tl.float32)

            acc += tl.sum(vals, axis=0)

        tl.store(output_ptr + outer_idx, acc.to(output_ptr.dtype.element_ty))


# ============================================================
# Kernel 2: reduce-non-last (inner_size > 1)
# reduce 轴不在最后维度，使用 2D tile 策略
# ============================================================

@triton.jit
def sum_kernel_reduce_non_last(
    input_ptr, output_ptr,
    outer_size, reduce_size, inner_size,
    BLOCK_RED: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    inner_tiles = (inner_size + BLOCK_INNER - 1) // BLOCK_INNER
    total_tiles = outer_size * inner_tiles

    tiles_per_pid = (total_tiles + num_pids - 1) // num_pids
    tile_start = pid * tiles_per_pid
    tile_end = tl.minimum(tile_start + tiles_per_pid, total_tiles)

    for tile_idx in range(tile_start, tile_end):
        outer_idx = tile_idx // inner_tiles
        inner_tile = tile_idx - (tile_idx // inner_tiles) * inner_tiles
        inner_start = inner_tile * BLOCK_INNER
        inner_end = tl.minimum(inner_start + BLOCK_INNER, inner_size)
        inner_len = inner_end - inner_start

        acc = tl.zeros((BLOCK_INNER,), dtype=tl.float32)

        base_offset = outer_idx * reduce_size * inner_size + inner_start

        for r_start in range(0, reduce_size, BLOCK_RED):
            r_end = tl.minimum(r_start + BLOCK_RED, reduce_size)

            r_offs = r_start + tl.arange(0, BLOCK_RED)[:, None]
            i_offs = tl.arange(0, BLOCK_INNER)[None, :]

            in_offsets = base_offset + r_offs * inner_size + i_offs
            mask_r = r_offs < r_end
            mask_i = i_offs < inner_len
            mask = mask_r & mask_i

            vals = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)
            vals = vals.to(tl.float32)

            acc += tl.sum(vals, axis=0)

        out_base = outer_idx * inner_size + inner_start
        out_offsets = out_base + tl.arange(0, BLOCK_INNER)
        out_mask = tl.arange(0, BLOCK_INNER) < inner_len

        tl.store(output_ptr + out_offsets, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


# ============================================================
# ModelNew: 带 dispatch 的统一入口
# ============================================================

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        original_dim = dim
        if dim is None:
            x = x.reshape(1, -1)
            dim = 1

        if dim < 0:
            dim = x.ndim + dim

        shape = x.shape
        outer_size = math.prod(shape[:dim]) if dim > 0 else 1
        reduce_size = shape[dim]
        inner_size = math.prod(shape[dim + 1:]) if dim + 1 < x.ndim else 1

        if keepdim:
            out_shape = list(shape)
            out_shape[dim] = 1
        else:
            out_shape = list(shape)
            del out_shape[dim]

        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        if inner_size == 1:
            # reduce-last: 直接沿 reduce 轴连续访存
            grid = (min(outer_size, self.VEC_CORE_NUM),)
            if outer_size > self.VEC_CORE_NUM:
                grid = (self.VEC_CORE_NUM,)
            sum_kernel_reduce_last[grid](
                x, output,
                outer_size, reduce_size,
                BLOCK_RED=1024,
            )
        else:
            # reduce-non-last: 2D tile 策略
            inner_tiles = (inner_size + 63) // 64
            total_tiles = outer_size * inner_tiles
            grid = (min(total_tiles, self.VEC_CORE_NUM),)
            if total_tiles > self.VEC_CORE_NUM:
                grid = (self.VEC_CORE_NUM,)
            sum_kernel_reduce_non_last[grid](
                x, output,
                outer_size, reduce_size, inner_size,
                BLOCK_RED=64,
                BLOCK_INNER=64,
            )

        if original_dim is None and not keepdim:
            output = output.squeeze()

        return output
