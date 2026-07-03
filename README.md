# cuda-optimized-skill 中 Profiling 的使用方式

## 1. 核心思想

`cuda-optimized-skill` 不是简单地把 Nsight Compute 的 profile dump 交给 LLM，而是把 profiling 结果转化为一个结构化的优化控制信号。

它的核心流程是：

```text
当前 best kernel
    ↓
Nsight Compute full profile
    ↓
抽取 compute / memory / latency 三类关键指标
    ↓
计算 roofline gap：Δc / Δm / Δl
    ↓
根据 gap 分配优化预算
    ↓
选择具体优化方法
    ↓
LLM 生成多个候选 kernel
    ↓
benchmark 选择 champion
    ↓
再次 profile champion
    ↓
ablation + SASS 验证
    ↓
更新 state，进入下一轮
```

一句话概括：

```text
profiling 不是结果展示工具，而是优化搜索的控制信号。
```

---

## 2. 每轮首先 profile 当前 best kernel

每一轮开始时，skill 会对当前最快的 kernel 运行 Nsight Compute。

示例命令：

```bash
python scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter 1 \
  --which best_input
```

该步骤会生成：

```text
iterv1/best_input.ncu-rep   # 完整 Nsight Compute 报告
iterv1/ncu_top.json         # 给 LLM 使用的摘要指标
```

这里的关键点是：
LLM 通常不会直接读取完整 `.ncu-rep`，而是读取压缩后的 `ncu_top.json`。

---

## 3. profile_ncu.py：把海量指标压缩成三类 top-K

`profile_ncu.py` 会执行：

```text
ncu --set full
    ↓
导出 raw CSV
    ↓
解析 metric
    ↓
按规则归类到 compute / memory / latency
    ↓
每类选出 top-K 严重指标
    ↓
写入 ncu_top.json
```

指标会被分成三类：

### 3.1 Compute 轴

关注计算资源是否被充分利用，例如：

```text
Tensor Core utilization
FP32 / FP64 pipe utilization
IPC
occupancy
SM throughput
```

### 3.2 Memory 轴

关注访存是否高效，例如：

```text
DRAM throughput
DRAM bytes
L1 / L2 hit rate
shared memory bank conflicts
L1 / L2 throughput
```

### 3.3 Latency 轴

关注 stall 和等待，例如：

```text
long scoreboard stall
short scoreboard stall
barrier stall
mio throttle
lg throttle
wait stall
```

最终 `ncu_top.json` 大致可以理解成：

```json
{
  "compute": [
    {
      "name": "sm__pipe_tensor_op_hmma_cycles_active...",
      "value": 12.3
    },
    {
      "name": "sm__warps_active...",
      "value": 18.0
    }
  ],
  "memory": [
    {
      "name": "dram__throughput...",
      "value": 35.2
    },
    {
      "name": "l1tex__t_sector_hit_rate.pct",
      "value": 22.1
    }
  ],
  "latency": [
    {
      "name": "smsp__warp_issue_stalled_long_scoreboard...",
      "value": 41.0
    }
  ]
}
```

也就是说，profiling 数据被转换成了 LLM 更容易使用的性能证据。

---

## 4. roofline.py：把 profiling 指标变成三个 gap

接下来，`roofline.py` 会读取：

```text
ncu_top.json
env.json
```

然后计算三个 gap：

```text
Δc = compute utilization gap
Δm = memory bandwidth utilization gap
Δl = latency / stall gap
```

可以粗略理解为：

```python
delta_compute = 1 - compute_utilization
delta_memory  = 1 - memory_bandwidth_utilization
delta_latency = max_stall_percentage
```

输出大致类似：

```json
{
  "delta_compute": 0.85,
  "delta_memory": 0.60,
  "delta_latency": 0.55,
  "bound": "compute",
  "near_peak": false,
  "axis_budget": {
    "compute": 1,
    "memory": 1,
    "latency": 1
  }
}
```

如果三个 gap 都很小，例如都小于阈值，就认为 kernel 已经接近硬件峰值，可以提前停止优化。

---

## 5. profiling 决定优化预算

这个 skill 每轮不会无限制地尝试优化方法，而是根据 profiling 结果分配优化预算。

例如：

```json
{
  "delta_compute": 0.20,
  "delta_memory": 0.75,
  "delta_latency": 0.55,
  "axis_budget": {
    "compute": 0,
    "memory": 2,
    "latency": 1
  }
}
```

这表示：

```text
当前主要问题在 memory 轴；
其次是 latency 轴；
compute 轴暂时不是主要优化方向。
```

因此下一轮会优先尝试：

```text
2 个 memory 优化方法
1 个 latency 优化方法
0 个 compute 优化方法
```

这一步的作用是减少搜索空间，避免 LLM 随机尝试不相关优化。

---

## 6. profiling 触发具体优化方法

有了 axis budget 后，LLM 会根据 profiling 指标选择具体优化方法。

它会参考：

```text
references/optimization_catalog.md
references/ncu_metrics_guide.md
```

这两个文件负责把指标映射到优化动作。

---

## 7. 指标到优化方法的映射示例

### 7.1 Long Scoreboard Stall 高

```text
smsp__warp_issue_stalled_long_scoreboard 高
        ↓
global memory latency 暴露
        ↓
可能触发 latency.async_pipeline / memory.async_copy
        ↓
尝试 cp.async、prefetch、double buffering、multi-stage pipeline
```

### 7.2 L1 Hit Rate 低

```text
l1tex__t_sector_hit_rate.pct 低
        ↓
访存局部性或合并访问较差
        ↓
可能触发 memory.coalesced_access
        ↓
调整 thread mapping、数据布局或访问顺序
```

### 7.3 Bank Conflict 高

```text
shared memory bank conflict 高
        ↓
shared memory 访问冲突
        ↓
触发 memory.bank_conflict
        ↓
调整 shared memory layout、padding 或访问模式
```

### 7.4 Barrier Stall 高

```text
barrier stall 高
        ↓
同步开销过大
        ↓
触发 latency.reduce_sync_count
        ↓
减少不必要的 __syncthreads()
```

### 7.5 Tensor Core 利用率低

```text
Tensor Core utilization 低
        ↓
计算资源没有充分使用
        ↓
触发 compute.tensor_core
        ↓
尝试 MMA / WMMA / CUTLASS / Tensor Core 路径
```

---

## 8. profiling 不直接生成代码，而是约束代码生成

LLM 不是直接根据所有 ncu 指标自由发挥，而是根据 profiling 结果选出的优化方法生成候选代码。

例如本轮选中的方法是：

```json
{
  "methods": [
    "memory.coalesced_access",
    "memory.vectorized_access",
    "latency.async_pipeline"
  ]
}
```

那么 LLM 生成的候选 kernel 就应该围绕这些方向展开。

每一轮会生成多个 branch：

```text
branch_0
branch_1
branch_2
branch_3
```

这些 branch 使用相同的优化方法，但可以改变实现细节，例如：

```text
tile size
num warps
pipeline stages
vector width
swizzle strategy
implementation variant
```

然后通过 benchmark 选择最快且正确的 champion。

所以这里 profiling 和 benchmark 的分工是：

```text
profiling 决定“应该尝试什么优化方向”
benchmark 决定“哪个候选实现真的最快”
```

---

## 9. 优化后再次 profile champion

选出 champion 后，skill 会再次运行 Nsight Compute：

```bash
python scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter 1 \
  --which kernel
```

生成：

```text
iterv1/kernel.ncu-rep
```

这一步用于观察优化后的瓶颈变化，例如：

```text
compute gap 是否下降？
memory gap 是否下降？
latency stall 是否下降？
瓶颈是否从 memory 转移到 compute？
是否出现新的 stall？
```

也就是说，profiling 不只用于第一轮分析，而是每轮都会更新优化方向。

---

## 10. profiling 历史进入 state

每轮结束后，profiling 相关结果会被写入 `state.json`。

其中会记录：

```text
roofline_history
selected_methods
effective_methods
ineffective_methods
implementation_failed_methods
frontier
```

例如：

```json
{
  "iter": 1,
  "delta_compute": 0.85,
  "delta_memory": 0.60,
  "delta_latency": 0.55,
  "bound": "compute",
  "axis_budget": {
    "compute": 1,
    "memory": 1,
    "latency": 1
  }
}
```

这样下一轮可以知道：

```text
哪些方法已经试过？
哪些方法有效？
哪些方法无效？
哪些方法实现失败？
当前瓶颈是否发生漂移？
```

例如：

```text
iter1: memory-bound
iter2: latency-bound
iter3: compute-bound
```

这就是所谓的 bottleneck drift。

---

## 11. Ablation：判断优化方法是否真的有效

选出 champion 后，skill 还会做 ablation。

做法是：
从 champion 中移除某个优化方法，然后重新 benchmark。

例如：

```text
champion
  - remove memory.coalesced_access
  - remove latency.async_pipeline
  - remove compute.launch_config
```

然后计算：

```text
attribution(method) = runtime_without_method - runtime_champion
```

如果：

```text
runtime_without_method > runtime_champion
```

说明去掉该方法后变慢，该方法有正贡献。

如果：

```text
runtime_without_method ≈ runtime_champion
```

说明该方法贡献很小。

如果：

```text
runtime_without_method < runtime_champion
```

说明该方法可能是负优化。

Ablation 的作用是防止 LLM 误判优化来源。

例如，LLM 可能声称性能提升来自 `async_pipeline`，但实际提升可能只是因为 tile size 变了。Ablation 可以把这些贡献拆开。

---

## 12. SASS 验证：确认优化是否真的编进机器码

除了 ablation，它还会做 SASS verification。

流程是：

```bash
cuobjdump --dump-sass
```

然后检查机器码中是否出现预期指令模式。

例如：

```text
compute.tensor_core
    → 期望看到 HMMA / WGMMA / UMMA

memory.vectorized_access
    → 期望看到向量化 load/store

warp specialization
    → 期望看到 SETMAXREG 等相关模式
```

这一步解决的问题是：

```text
源码里看起来写了某个优化，
但编译器可能没有真的生成对应机器码。
```

因此最终会根据：

```text
SASS 是否验证通过
Ablation 贡献是否超过噪声阈值
```

把方法分为三类：

```text
effective_methods
ineffective_methods
implementation_failed_methods
```

分类逻辑可以表示为：

```text
SASS verified + attribution > noise
    → effective_methods

SASS verified + attribution <= noise
    → ineffective_methods

SASS not verified
    → implementation_failed_methods
```

---

## 13. 程序化总结

整个 profiling-guided loop 可以写成：

```python
for iter in range(max_iters):
    # 1. Profile 当前 best kernel
    profile = ncu_profile(best_kernel)

    # 2. 抽取三类关键指标
    ncu_top = extract_top_metrics(
        profile,
        axes=["compute", "memory", "latency"]
    )

    # 3. 计算 roofline gap
    roofline = compute_roofline_gaps(
        ncu_top=ncu_top,
        env=env
    )

    # 4. 判断是否接近峰值
    if roofline.near_peak:
        break

    # 5. 根据 gap 分配优化预算
    axis_budget = allocate_budget(
        delta_compute=roofline.delta_compute,
        delta_memory=roofline.delta_memory,
        delta_latency=roofline.delta_latency
    )

    # 6. 根据 profiling 指标选择优化方法
    methods = select_methods(
        ncu_metrics=ncu_top,
        axis_budget=axis_budget,
        optimization_catalog=catalog,
        already_tried=state.selected_methods,
        ineffective=state.ineffective_methods,
        arch=env.sm_arch
    )

    # 7. LLM 生成多个候选 kernel
    branches = LLM_generate_kernels(
        best_kernel=best_kernel,
        methods=methods,
        variants=[
            "tile_size",
            "num_warps",
            "pipeline_stages",
            "implementation_variant"
        ]
    )

    # 8. 编译、测试、benchmark，选择 champion
    champion = benchmark_and_select_fastest_correct(branches)

    # 9. 再次 profile champion
    champion_profile = ncu_profile(champion)

    # 10. 对每个方法做 ablation
    attribution = ablate_each_method(
        champion=champion,
        methods=methods
    )

    # 11. 做 SASS 验证
    sass = verify_sass_signatures(
        champion=champion,
        methods=methods
    )

    # 12. 更新状态
    state.update(
        best_kernel=champion,
        roofline=roofline,
        methods=classify_methods(
            methods=methods,
            attribution=attribution,
            sass=sass
        )
    )
```

---

## 14. 最核心的数据流

可以进一步压缩为：

```text
ncu metrics
    ↓
compute / memory / latency bottleneck
    ↓
roofline gap
    ↓
axis budget
    ↓
method trigger
    ↓
LLM code generation
    ↓
benchmark selection
    ↓
champion profile
    ↓
ablation attribution
    ↓
SASS verification
    ↓
state update
```

---

## 15. 总结

`cuda-optimized-skill` 中 profiling 的作用可以总结为三点：

### 15.1 决定优化方向

通过 Nsight Compute 指标判断 kernel 当前主要受限于：

```text
compute
memory
latency
```

### 15.2 约束 LLM 搜索空间

根据 profiling 结果选择具体优化方法，而不是让 LLM 随机尝试。

例如：

```text
long scoreboard 高 → 尝试 async pipeline
bank conflict 高 → 优化 shared memory layout
Tensor Core 利用率低 → 尝试 Tensor Core 路径
barrier stall 高 → 减少同步
```

### 15.3 验证优化是否真实有效

通过：

```text
优化后重新 profile
ablation benchmark
SASS verification
```

判断优化是否真的降低瓶颈、提升速度，并且被编译进机器码。

最终，这个 skill 把 LLM CUDA 优化从：

```text
凭经验猜优化
```

变成：

```text
profiling 证据驱动的闭环搜索
```

# `ncu_top.json` 和 `roofline.json` 是怎么生成的

在 `cuda-optimized-skill` 中，下面两个文件都是由脚本自动生成的，而不是 LLM 手写出来的：

```text
iterv1/ncu_top.json
iterv1/roofline.json
```

它们的生成关系可以概括为：

```text
best_input.ncu-rep
    ↓ profile_ncu.py
ncu_top.json
    ↓ roofline.py
roofline.json
```

也就是说：

```text
.ncu-rep:
    原始完整 Nsight Compute profiling 数据

ncu_top.json:
    从 .ncu-rep 中提取出的关键指标摘要

roofline.json:
    根据 ncu_top.json 进一步计算出的优化决策信号
```

---

## 1. `ncu_top.json` 怎么生成

`ncu_top.json` 由脚本生成：

```bash
scripts/profile_ncu.py
```

典型运行方式是：

```bash
python scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter 1 \
  --which best_input
```

它的流程如下：

```text
读取 state.json
    ↓
找到当前 best kernel
    ↓
调用 ncu --set full 跑 Nsight Compute
    ↓
生成 best_input.ncu-rep
    ↓
使用 ncu --import 将 .ncu-rep 导出成 CSV
    ↓
解析 CSV 中的 metrics
    ↓
按照规则把 metrics 分成 compute / memory / latency 三类
    ↓
每类选择 top-K 最关键指标
    ↓
写出 ncu_top.json
```

---

## 2. `profile_ncu.py` 具体做了什么

`profile_ncu.py` 的核心作用是：

```text
完整 NCU profile
    → CSV metrics
    → 指标分类
    → 严重程度排序
    → top-K 指标摘要
```

它会把指标分成三类。

### 2.1 Compute 轴

关注计算资源是否被充分利用，例如：

```text
Tensor Core utilization
FP32 / FP64 pipe utilization
IPC
occupancy
SM throughput
```

### 2.2 Memory 轴

关注访存是否高效，例如：

```text
DRAM throughput
DRAM bytes
L1 / L2 hit rate
shared memory bank conflicts
L1 / L2 throughput
```

### 2.3 Latency 轴

关注 stall 和等待，例如：

```text
long scoreboard stall
short scoreboard stall
barrier stall
mio throttle
lg throttle
wait stall
```

最终生成的 `ncu_top.json` 可以理解成：

```json
{
  "compute": [
    {
      "name": "sm__pipe_tensor_op_hmma_cycles_active...",
      "value": 12.3
    },
    {
      "name": "sm__warps_active...",
      "value": 18.0
    }
  ],
  "memory": [
    {
      "name": "dram__throughput...",
      "value": 35.2
    },
    {
      "name": "l1tex__t_sector_hit_rate.pct",
      "value": 22.1
    }
  ],
  "latency": [
    {
      "name": "smsp__warp_issue_stalled_long_scoreboard...",
      "value": 41.0
    }
  ]
}
```

这里的 `ncu_top.json` 不是完整 profile，而是给 LLM 和后续脚本使用的 profiling 摘要。

---

## 3. `roofline.json` 怎么生成

`roofline.json` 由另一个脚本生成：

```bash
scripts/roofline.py
```

典型运行方式是：

```bash
python scripts/roofline.py \
  --state ./run_*/state.json \
  --iter 1
```

它读取：

```text
iterv1/ncu_top.json
env.json / state.json 中的硬件环境信息
```

然后计算三个 gap：

```text
Δc = delta_compute
Δm = delta_memory
Δl = delta_latency
```

---

## 4. `roofline.py` 的核心计算逻辑

`roofline.py` 会把 `ncu_top.json` 中的指标进一步压缩成三个优化方向的 gap。

### 4.1 Compute gap

粗略可以理解为：

```python
compute_util = max(
    tensor_core_utilization,
    fp32_pipe_utilization,
    sm_throughput
)

delta_compute = 1 - compute_util
```

含义是：

```text
compute utilization 越低，delta_compute 越大；
delta_compute 越大，说明计算资源利用越不充分。
```

---

### 4.2 Memory gap

粗略可以理解为：

```python
memory_util = dram_throughput_pct
delta_memory = 1 - memory_util
```

含义是：

```text
memory bandwidth utilization 越低，delta_memory 越大；
delta_memory 越大，说明访存带宽利用不足或存在访存瓶颈。
```

---

### 4.3 Latency gap

粗略可以理解为：

```python
delta_latency = max(stall_percentage_metrics)
```

它会关注一些 stall 指标，例如：

```text
long scoreboard stall
short scoreboard stall
barrier stall
mio throttle
lg throttle
wait stall
```

含义是：

```text
stall 百分比越高，delta_latency 越大；
delta_latency 越大，说明 kernel 中有较严重的等待或延迟问题。
```

---

## 5. `roofline.json` 中会包含什么

`roofline.json` 通常会包含：

```json
{
  "delta_compute": 0.85,
  "delta_memory": 0.60,
  "delta_latency": 0.55,
  "bound": "compute",
  "near_peak": false,
  "axis_budget": {
    "compute": 1,
    "memory": 1,
    "latency": 1
  }
}
```

这些字段的含义是：

| 字段              | 含义                     |
| --------------- | ---------------------- |
| `delta_compute` | 计算资源利用 gap             |
| `delta_memory`  | 访存带宽利用 gap             |
| `delta_latency` | stall / latency gap    |
| `bound`         | 当前主要瓶颈方向               |
| `near_peak`     | 是否已经接近硬件峰值             |
| `axis_budget`   | 下一轮每个优化轴可以选择多少个 method |

---

## 6. 如何判断主要瓶颈

可以简单理解为：

```text
如果 delta_compute 最大
    → compute-bound

如果 delta_memory 最大
    → bandwidth-bound / memory-bound

如果 delta_latency 最大
    → latency-bound

如果三个 gap 都很小
    → near_peak = true
```

例如：

```json
{
  "delta_compute": 0.92,
  "delta_memory": 0.57,
  "delta_latency": 0.61,
  "bound": "compute",
  "near_peak": false
}
```

表示：

```text
compute gap 最大；
当前 kernel 主要问题是计算资源利用率低；
下一轮应该优先尝试 compute 相关优化。
```

---

## 7. 如何分配 axis budget

`roofline.py` 会根据三个 gap 分配下一轮优化预算。

可以理解为：

```text
总 budget = 3
单个 axis 最多 = 2
gap 太小的 axis 分配 0
按照 delta_compute / delta_memory / delta_latency 的相对大小分配预算
```

例如：

```json
{
  "delta_compute": 0.20,
  "delta_memory": 0.75,
  "delta_latency": 0.55,
  "axis_budget": {
    "compute": 0,
    "memory": 2,
    "latency": 1
  }
}
```

表示：

```text
memory gap 最大，因此 memory 轴分配 2 个 method；
latency gap 次之，因此 latency 轴分配 1 个 method；
compute gap 较小，因此 compute 轴暂时不分配 method。
```

---

## 8. 两个文件的作用区别

| 文件                   | 来源               | 作用                |
| -------------------- | ---------------- | ----------------- |
| `best_input.ncu-rep` | Nsight Compute   | 保存完整 profiling 结果 |
| `ncu_top.json`       | `profile_ncu.py` | 从 NCU 结果中提取关键指标   |
| `roofline.json`      | `roofline.py`    | 根据关键指标计算优化方向和预算   |

可以进一步理解为：

```text
best_input.ncu-rep:
    原始数据层

ncu_top.json:
    profiling 摘要层

roofline.json:
    优化决策层
```

---

## 9. 程序化总结

整个过程可以写成：

```python
# 1. 跑 Nsight Compute，得到完整 profile
ncu_rep = run_nsight_compute(best_kernel)

# 2. 从完整 profile 中提取关键指标
ncu_top = extract_top_metrics(
    ncu_rep,
    axes=["compute", "memory", "latency"]
)

# 3. 根据关键指标计算 roofline gap
roofline = compute_roofline_gaps(
    ncu_top=ncu_top,
    env_info=env
)

# 4. 根据 gap 分配优化预算
axis_budget = allocate_method_budget(
    delta_compute=roofline["delta_compute"],
    delta_memory=roofline["delta_memory"],
    delta_latency=roofline["delta_latency"]
)
```

---

## 10. 最终总结

这两个文件的生成逻辑可以一句话概括：

```text
profile_ncu.py 负责把完整 NCU profile 压缩成 ncu_top.json；
roofline.py 负责把 ncu_top.json 转换成 roofline.json，用来指导下一轮优化。
```

也就是说：

```text
profiling 原始数据
    ↓
关键指标摘要
    ↓
compute / memory / latency gap
    ↓
axis budget
    ↓
method selection
    ↓
LLM code generation
```

