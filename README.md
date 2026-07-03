这个仓库的核心可以概括为：

cuda-optimized-skill 是一个面向 Claude 的 CUDA Kernel 优化 skill，用 Nsight Compute profiling、benchmark、roofline 分析、分支探索、消融实验和 SASS 验证，驱动 LLM 迭代优化 CUDA / CUTLASS / Triton kernel。

它不是一个单独点击运行的 optimizer，而是一个 skill package：Claude 读取 SKILL.md，负责“想优化方法 + 写代码”；仓库里的 Python 脚本负责环境检测、profile、benchmark、状态记录、消融、SASS 检查等确定性部分。

1. 输入是什么

用户需要给它：

baseline kernel: gemm.cu / gemm.py
reference: ref.py，用来校验正确性
dims: M/N/K 等参数
iterations: 默认 3
ncu_num: 每个轴提取 top-K 指标，默认 5
branches: 每轮生成多少个候选分支，默认 4

它支持 CUDA、CUTLASS 和 Triton kernel；正确性依赖 Python reference。

2. 整体流程

可以简化成：

baseline kernel
    ↓
环境检查 + correctness preflight
    ↓
benchmark baseline
    ↓
ncu profile 当前 best kernel
    ↓
提取 compute / memory / latency 三类瓶颈指标
    ↓
roofline.py 计算 Δc / Δm / Δl
    ↓
按瓶颈强度分配优化预算
    ↓
Claude 从 optimization catalog 里选择优化方法
    ↓
Claude 生成 K 个候选 kernel 分支
    ↓
编译 + benchmark 所有分支
    ↓
选择最快且正确的 champion
    ↓
对 champion 再做 ncu profile
    ↓
做 ablation，判断每个优化方法真实贡献
    ↓
做 SASS check，确认优化是否真的编进机器码
    ↓
更新 state.json，进入下一轮
    ↓
输出 summary.md

SKILL.md 里明确把循环写成：profile with ncu → compute roofline gaps → allocate axis budgets → pick methods → generate K branches → validate/benchmark → select champion → ablation attribution → SASS verification → update state。

3. 它的关键机制
A. Roofline-driven axis budget

它把性能问题分成三个轴：

compute axis
memory axis
latency axis

然后根据 profiling 结果计算：

Δc = compute utilization gap
Δm = memory bandwidth gap
Δl = latency / stall gap

再把每轮最多 3 个优化方法按比例分配给这三个轴；每个轴最多 2 个。如果三个 gap 都小于 0.15，就认为接近峰值，可以 early stop。

B. Branch-and-Select

每一轮不是只生成一个 kernel，而是生成 K 个候选分支。它们使用同一组优化方法，但改变 tile size、pipeline stages、warp count、implementation variant 等超参数。然后全部编译、测试、benchmark，选最快且正确的作为 champion。

C. Ablation attribution

选出 champion 后，它会把某个优化方法单独去掉，重新 benchmark，计算：

attribution(method) = runtime_without_method - runtime_champion

如果去掉某个方法后变慢，说明这个方法确实有贡献；如果变化很小或反而变快，就说明这个方法没用。

D. SASS verification

它不仅看源码里有没有写某个优化，还会用 cuobjdump --dump-sass 检查生成的机器码里是否出现预期指令模式。最终每个方法会被分到三类：

effective_methods
ineffective_methods
implementation_failed_methods

分类依据是：SASS 是否验证通过，以及 ablation 贡献是否超过噪声阈值。

4. profiling 是怎么被用起来的

它不是简单地把 ncu 输出塞给 LLM，而是把 profiling 结果变成一个结构化决策过程：

ncu metrics
    ↓
compute / memory / latency bottleneck
    ↓
axis budget
    ↓
method trigger
    ↓
candidate optimization
    ↓
benchmark + ablation + SASS verification

例如 ncu_metrics_guide.md 里把很多指标直接映射到优化动作：Tensor Core 利用率低可能触发 compute.tensor_core，L1 hit 低可能触发 memory.coalesced_access，Long Scoreboard stall 高可能触发 latency.async_pipeline，Barrier stall 高可能触发 latency.reduce_sync_count。

5. 输出是什么

每次运行会生成一个 run_YYYYMMDD_HHMMSS/ 目录，里面保存：

env.json
state.json
baseline/bench.json
iterv1/
  roofline.json
  methods.json
  analysis.md
  best_input.ncu-rep
  branches/
  kernel.cu / kernel.py
  kernel.ncu-rep
  ncu_top.json
  sass_check.json
  ablations/
  attribution.json
  bench.json
iterv2/
iterv3/
summary.md

也就是说，它很强调可复现、可追踪、可解释：每轮尝试了什么、为什么选这个方法、profile 结果是什么、哪种优化真的有效，都会落盘。

6. 和 KEET 的关系

如果和前面讨论的 KEET 对比，可以这样理解：

KEET:
    更偏“profiling → 解释报告 → 辅助 LLM 优化”

cuda-optimized-skill:
    更偏“profiling → 选择优化方法 → 生成多分支代码 → benchmark → ablation → SASS 验证 → 下一轮”

所以这个 skill 比 KEET 更像一个工程化的自动优化闭环。它不只是生成性能解释，还把 profiling 直接接到代码生成、分支搜索、因果归因和状态更新里。

7. 局限

README 也列了一些限制：如果 baseline 已经是 cuBLAS / cuDNN / cuBLASLt，想继续大幅提升通常需要更强的算法级变化；很短的 kernel 容易受 launch overhead 噪声影响；Triton autotune 在 ncu 下可能很慢；SASS signature 只是启发式 grep，不等于完整语义验证；每轮 correctness retry 最多 3 次。
