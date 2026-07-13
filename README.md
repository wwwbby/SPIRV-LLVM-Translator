1. 目标： 优化XXX目录下的算子 KDA 中的 XXXKernel triton ascend kernel。
2. 背景：算子 KDA 由多个 triton-kernel 拼接得到，其实现位于XXX文件的chunk_kda 中。你可以阅读kernel-generator下的reference以获取triton ascend开发的背景信息
3. 行动：
 3.1. 阅读整个chunk_kda的实现。
 3.2. 使用性能测试方法获取性能基线
 3.3. 调用 latency-optimizer skill 优化 XXXKernel

精度验证方法：
命令：在theta-flash-linear-attention目录下执行XXX命令，命令行中会打印用例通过情况，通过为PASSED。
执行方式：使用 itask-npu-remote skills完成，在执行之前请进入XXX目录同步代码。

性能获取方法：
命令：在theta-flash-linear-attention目录下执行XXX命令，命令行中会打印该算子的用时。
执行方式：使用 itask-npu-remote skills完成，在执行之前请进入XXX目录同步代码。

约束：
* 只能优化修改 XXXKernel 这一个 kernel，其他代码坚决不能修改。
* 如果修改后算子仍能通过精度验证方法，且性能有提升，则保留该修改，否则回退。
* 一次只能应用一个优化点；若优化后依次调用精度验证和性能获取，若无法通过精度校验或者性能变差，则回退该优化点。
