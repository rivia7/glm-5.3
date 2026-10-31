# 卡间 P2P 修复评估：动态 BAR1 P2P 方案

> 状态：评估完成，未实施。日期：2026-10-31。
>
> 适用环境：本仓库部署所用的两台 8 卡机器。nvidia-smi 显示为 RTX 5090 的 96GB 卡，驱动是 aikitoria 的 P2P 补丁版 595.71.05。
>
> 当前部署用 `GPU_P2P=0` 绕开 P2P（见 README「卡间 P2P 不可用时」），不依赖本方案。

## 结论

- **根因已确认**：BAR1 只有 32GB，装不下 96GB 显存，补丁依赖的静态 BAR1 建不起来，BAR1 P2P 用不上。驱动于是退回一条在这批卡上没有建成的 mailbox P2P 路径，却仍报告 P2P 可用，写入就这样静默丢失。
- **改配置修不好**：卡声明的 BAR1 最大就是 32GB；强制开静态 BAR1 会让 GPU 初始化失败。
- **动态 BAR1 P2P 技术上可行**：需要的接口和生命周期挂靠点都在开源代码里，没有找到硬性阻断。工作量约 1000–1500 行内核代码，以周计。风险集中在内核崩溃、错写数据和闭源组件。
- **收益有限**：P2P 修好后，decode 能用上的是 SGLang 自带的 FlashInfer PCIe-IPC all-reduce（不是自定义 all-reduce，它在超过 2 张卡的纯 PCIe 组里会自动关闭）。按它的实测数据估算，attention 按 TP8 切时单路 decode 约快一到两成，高并发和 prefill 基本不变（见第 7 节）。
- **建议**：先做第 8.1 节的零代码前置实验，确认这台平台上 P2P 硬件路径正确、小消息时延确实下降，再决定是否在独立测试机上开发。

## 1. 现象

- 驱动对所有卡对报告 P2P 读写可用（`nvidia-smi topo -p2p r`、`w` 全是 OK），`cuDeviceCanAccessPeer` 也返回可用。
- `run_gpu_test.sh`：56 个方向的 P2P 拷贝数据全错。测出的单向带宽 115–130 GB/s、16 字节时延 0.4 µs，超过 PCIe Gen5 x16 的物理上限（约 63 GB/s），说明数据根本没有经 PCIe 到达对端。
- PyTorch 同进程跨卡拷贝（`b.copy_(a)`）数据也错；经主机内存中转的拷贝正确。
- NCCL 不关 P2P 时，第一次集合通信就卡住，直到 watchdog 超时退出。
- 做 P2P 时 dmesg 出现两条驱动断言：

  ```
  NVRM: nvAssertFailedNoLog: Assertion failed: ((base & RM_PAGE_MASK) == 0) @ kern_bus.c:298
  NVRM: nvAssertFailedNoLog: Assertion failed: remoteWMBoxLocalAddr != ~0ULL @ kern_bus_gm200.c:89
  ```

## 2. 环境

| 项目 | 情况 |
|---|---|
| GPU | 每台 8 张；PCI ID `10de:2b85`，170 个 SM（RTX 5090 的配置；RTX PRO 6000 是 188 个），96GB 显存，512 位总线 |
| BAR1 | 32GB；卡声明的 Resizable BAR 可选 64MB–32GB，最大就是 32GB |
| VBIOS | 98.02.2E.00.90 |
| 驱动 | 开源内核模块 595.71.05 加 aikitoria P2P 补丁（补丁提交 `97fdda09`，对应上游版本提交 `51edebee`），在部署机上编译安装 |
| 平台 | 联想 WA5680 G3，AMI BIOS 1.17.35；2 颗 Xeon Gold 6530；每张卡直连一个 CPU 根端口，没有 PCIe 交换芯片，也没有 NVLink |
| IOMMU / ACS | 启动参数有 `iommu=pt`，但 VT-d 没有开启（没有 IOMMU 分组）；没有 PCIe 桥开启 ACS 重定向 |

## 3. 根因

### 3.1 补丁做了什么

补丁 `97fdda09` 的核心改动：

1. 把 Hopper 数据中心卡的 BAR1 P2P 实现（`*_GH100`）设为所有卡的默认实现（`g_kern_bus_nvoc.c`）。
2. 强制把 P2P 读写报告为可用（`p2pOverride = 0x11`），PCIe P2P 类型默认选 BAR1（`kernel_bif.c`）。
3. UVM 生成对端页表项时，BAR1 P2P 用 SYS 孔径，地址 = 对端 BAR1 的物理基址 + 显存物理偏移（`nv_gpu_ops.c`）。

### 3.2 失败链路

1. **BAR1 P2P 只有静态实现**：要求两张卡都开启「静态 BAR1」，即把整块显存线性映射进 BAR1（`kbusIsPcieBar1P2PMappingSupported_GH100`）。驱动里只有这一种 BAR1 P2P 实现，对端地址就是 BAR1 基址加偏移；`kbusGetBar1P2PDmaInfo_GH100` 直接取静态映射的描述符。
2. **静态 BAR1 要求 BAR1 装得下全部显存**（`kbusIsStaticBar1Supported_TU102`）：
   - 自动模式：BAR1 不小于显存地址空间，加上 USERD、控制台、mailbox 等预留，再加 512MB 对齐。
   - 用 `RMForceStaticBar1=1` 强制：仍要求 BAR1 不小于客户端可见显存，否则返回 `NV_ERR_INVALID_REGISTRY_KEY`，BAR1 初始化失败，GPU 起不来。

   32GB < 96GB，所以静态 BAR1 没有开启：运行时 BAR1 只用了 3 MiB。
3. **退回 mailbox**：BAR1 P2P 不可用，P2P 连接类型退回老的 PCIe mailbox（`kbusCreateP2PMapping_GP100`）。
4. **mailbox 没建成**：补丁把 PCIe P2P 类型设成了 BAR1，`_kbusRequiresP2PMailboxBar1_GM107` 因此返回假，初始化 BAR1 时没有给 mailbox 预留空间。建立 mailbox 时（`kbusSetupMailboxes_GM200` → `kbusSetupMailboxAccess` → `kbusSetupPeerBarAccess`）拿到的地址无效：
   - 先在 `kern_bus.c:298` 断言失败，返回 `~0`；
   - 再在 `kern_bus_gm200.c:89` 断言失败，函数中途返回，不向上报错。
5. **仍报告可用**：`p2pOverride` 让 P2P 继续报告为可用，CUDA、NCCL 照常走 P2P，写入到不了对端，也没有任何报错。

### 3.3 为什么 BAR1 调不大

- Resizable BAR 的可选大小由卡（VBIOS）声明，主板 BIOS 只能在其中选；这批卡最大 32GB。
- Blackwell 的 VBIOS 需要 NVIDIA 签名，不能自己改。
- 芯片是 5090 的配置（SM 数由熔丝决定），所以刷不了 RTX PRO 6000 的 VBIOS，也用不上官方驱动对 PRO 卡的 P2P 支持。

## 4. 修复路线对比

| 路线 | P2P 能否真正可用 | 代价和风险 | 结论 |
|---|---|---|---|
| 调主板 BIOS 或驱动参数 | 不能 | 卡最大只支持 32GB BAR1；强制静态 BAR1 会让 GPU 初始化失败 | 不可行 |
| 换 BAR1 支持 128GB 的 VBIOS | 能 | 只能由改装卡供应商提供 NVIDIA 签名的现成版本；64GB 不够；逐张刷卡有变砖风险 | 供应商有现成 VBIOS 才值得试 |
| 动态 BAR1 P2P（本文方案） | 能 | 新写内核代码，以周计；风险见第 6 节 | 先做前置实验再决定 |
| 换回官方开源驱动，或给补丁加保护（开不了静态 BAR1 就不报 P2P） | 不能，但会如实报告不支持 | 要停服务、重新加载驱动模块 | 推荐：消除对其它程序的隐患 |
| 维持 `GPU_P2P=0` | 不能 | 无 | 当前状态，部署已验证正确 |

补丁驱动现在对这两台上的所有程序都谎报 P2P 可用：

- PyTorch 同进程跨卡拷贝（`tensor.to("cuda:1")`、HF 的 `device_map="auto"`）会得到错误数据；
- NCCL 不设 `NCCL_P2P_DISABLE=1` 会卡死；
- SGLang 默认不实测 P2P：`--enable-p2p-check` 默认关，自定义 all-reduce 的 P2P 自检被替换成直接相信驱动。TP8 的自定义 all-reduce 在纯 PCIe 机器上会自动关闭，但 2 卡一组时（如 `ATTN_DP_SIZE=4` 的 attention 组）会用它；PCIe-IPC all-reduce（`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`）根本不检查 P2P。两条路径在这里都会悄悄算错，所以 `GPU_P2P=0` 时启动脚本加 `--disable-custom-all-reduce`，并强制关掉 PCIe-IPC。

## 5. 方案设计

### 5.1 思路

静态 BAR1 把整块显存一次性映射进 BAR1，对端地址 = BAR1 基址 + 物理偏移，所以 BAR1 必须不小于显存。

动态方案改成按需映射：某块显存要被对端访问时，才在它所在那张卡的 BAR1 动态区建一个「窗口」，把这块显存的物理页映射进去；对端页表项指向窗口的总线地址。这样 BAR1 只需装下正在被对端访问的显存。

### 5.2 改动落点：现有调用链

按 CUDA 在 Linux 上的实现，普通显存（`cudaMalloc`、cuMem 导入、`cudaIpcOpenMemHandle`）映射到对端，都经 UVM 的 external range。NCCL 的 P2P 传输、SGLang 的 PCIe-IPC all-reduce 和自定义 all-reduce（都用 `cudaIpc` 共享缓冲区）、`cudaMemcpyPeer` 都走这条链路：

```
UVM（kernel-open/nvidia-uvm/uvm_map_external.c）
  nvUvmInterfaceDupMemory(映射方 GPU, 用户的显存句柄)        复制句柄，同时给物理显存加引用
  uvm_va_space_peer_enabled(映射方, 所属方)                   对端互访检查
  nvUvmInterfaceGetExternalAllocPtes(复制的句柄, 偏移, 大小)   取页表项，可能分批调用
    → RM nvGpuOpsGetExternalAllocPtesOrPhysAddrs
      → nvGpuOpsBuildExternalAllocPtes                        补丁在这里按「BAR1 基址 + 物理偏移」填地址
  uvm_release_rm_handle（引用计数归零时）
    → nvUvmInterfaceFreeDupedHandle                           释放复制的句柄
```

### 5.3 RM 改动

1. **窗口管理（新增）**
   - 在所属 GPU 的 BAR1 动态区分配窗口，用 `kbusMapFbApertureSingle` 映射、`kbusUnmapFbApertureSingle` 拆除。静态 BAR1 本身就是用这个接口建的（`kbusEnableStaticBar1Mapping_TU102`），实现在开源的 CPU 侧代码里。
   - 按（显存描述符，区间）去重并做引用计数，所有对端共用同一个窗口。
   - 首次查询页表项时为整块分配建窗口，保证分批查询得到一致的地址。
   - 页表项 kind 与静态 BAR1 一致，用 generic memory。
   - BAR1 用满时返回错误，让这次映射失败，而不是错写。
2. **页表项**：`nvGpuOpsBuildExternalAllocPtes` 和 `nvGpuOpsBuildExternalAllocPhysAddrs` 的 BAR1 P2P 分支，把「BAR1 基址 + 物理偏移」改成「窗口总线地址 + 页在窗口内的偏移」，并把窗口登记到这次查询所用的复制句柄上。
3. **生命周期**：复制出的显存对象析构时，释放登记在它上面的窗口。
   - UVM 先清映射方的页表、刷 TLB，再释放句柄，顺序正确。
   - 物理显存在所有复制句柄释放前不会被回收，窗口不会指向已回收的显存。
4. **能力上报和 P2P 对象**：
   - 新增一个默认关闭的注册表开关（名字待定，如 `RMBar1P2PDynamic`）。
   - 开启时，`kbusIsPcieBar1P2PMappingSupported_GH100` 不再要求静态 BAR1；`kbusGetBar1P2PDmaInfo_GH100` 返回整个 BAR1 孔径（基址加 32GB）。
   - `kbusCreateP2PMappingForBar1P2P_GH100`：IOMMU 关闭时不需要额外映射；开启时要把整个 BAR1 地址范围映射进对端的 IOMMU 域。
   - 不再退回 mailbox。
5. **RM 自管地址空间里的对端映射**（`virtual_mem.c`）：同样改用窗口，或者在动态模式下明确拒绝。

### 5.4 UVM 改动

UVM 托管内存（`cudaMallocManaged`）的对端访问，在 `uvm_gpu_peer_phys_address` 里按「`bar1_p2p_dma_base_address` + 物理偏移」计算，同样假设静态映射；`uvm_va_space.c` 还会自动为 BAR1 对端开启互访。动态模式下需要：

- 对端互访只用于普通显存映射；
- 托管内存不直接访问对端，改走迁移，也不建对端恒等映射；
- 对动态对端，`uvm_gpu_peer_phys_address` 加断言保护，防止误用线性公式。

这些逻辑分散在 `uvm_gpu.c`、`uvm_va_space.c`、`uvm_va_block.c`、`uvm_mmu.c` 等文件里，是整个方案最难的部分。

### 5.5 BAR1 容量

每张卡 32GB BAR1，目前只用了 3 MiB。要放进窗口的是被对端访问的显存，包括：

- NCCL 的 P2P 缓冲区；
- PCIe-IPC all-reduce 的工作区：按 decode 的最大 batch × hidden 分配，很小；
- 自定义 all-reduce（只在 2 卡一组时用）的 IPC 缓冲区，以及它为 CUDA Graph 注册的缓冲区所在的整块分配（`cudaIpc` 共享的是整块 `cudaMalloc` 分配）。

估计合计几百 MB 到几 GB。窗口建在所属卡上、多个对端共用，应该装得下，需要在原型里实测确认。

### 5.6 工作量

| 位置 | 内容 | 估计 |
|---|---|---|
| RM 新增 | 窗口管理：分配、去重和引用计数、拆除、用满时报错 | 300–500 行 |
| `nv_gpu_ops.c` | 页表项和物理地址两处改用窗口地址，登记到复制句柄 | 100–200 行 |
| `kern_bus_gh100.c` 等 | 支持判断不再要求静态 BAR1；DMA 范围改为整个 BAR1；注册表开关 | 50–100 行 |
| `virtual_mem.c` | RM 自管地址空间的对端映射：改用窗口或拒绝 | 50–150 行 |
| 显存对象析构 | 释放挂在句柄上的窗口 | 约 50 行 |
| UVM（4–5 个文件） | 动态对端只用于普通显存；托管内存改走迁移；关掉对端恒等映射 | 200–500 行 |

合计约 1000–1500 行内核代码。熟悉 RM 代码的人做出原型约需 1–2 周，加上验证和压测，整体以周计。

## 6. 风险

1. **闭源组件**：CUDA 用户态驱动（libcuda）和 GSP 固件不开源。如果它们在别处默认 BAR1 是静态映射，只能靠测试发现，而且可能绕不过去。这是最大的未知数。
2. **错写的后果**：窗口地址算错时，数据会写进对端显存的错误位置，可能写坏别的进程或驱动自己的数据，比现在的「写丢」更危险。必须做严格的数据校验和长时间压测。
3. **开发环境**：每次迭代都要停两台的服务、卸载再重载驱动；内核崩溃或卡死会失去远程访问（这两台只能远程 SSH 登录）。必须在有现场人员或 BMC、同款卡的独立测试机上开发，不能在这两台上调试。
4. **维护**：改动叠在第三方补丁之上，每次驱动升级都要重新移植；libcuda 必须和内核模块同版本，CUDA 用户态也跟着锁定。
5. **收益不确定**：取决于这台双路 Intel 平台上 P2P 的实际时延和带宽，尤其是跨 CPU 的卡对。

## 7. 收益估计

P2P 修好后，SGLang 0.5.23 里能用上 P2P 的 all-reduce 是 FlashInfer PCIe-IPC，不是自定义 all-reduce：

- **自定义 all-reduce**：超过 2 张卡的纯 PCIe 组会自动关闭（`custom_all_reduce_utils.py`：`not supported on more than two PCIe-only GPUs`），TP8 用不上。
- **PCIe-IPC all-reduce**：用 `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` 打开，默认关，只用于 TP 组（2、4、8 卡）。它是给卡都直连 CPU、没有 PCIe 交换芯片的 8 卡 SM120 机器写的，和这两台结构一样：按 4+4 分组，尽量少走跨 CPU 的链路；只接管 decode 大小的消息，prefill 仍交给 NCCL。
- **源码注释里的实测**（同类机器，hidden 6144，BF16）：12 KiB 一次 5.3 µs，384 KiB 一次 34.6 µs；TP8 服务的 TPOT 从 21.1 ms 降到 13.6 ms。
- **那组数据的 NCCL 基线**：比这两台现在经主机内存还慢一个数量级（384 KiB：1169 µs 对 50 µs），所以 36% 的降幅在这里拿不到。这也说明这类平台上 NCCL 走 P2P 可能比走主机内存还慢：P2P 修好后，NCCL 的大消息要不要走 P2P（`GPU_P2P=1` 会设 `NCCL_P2P_LEVEL=SYS`）要重新实测。

按 attention TP8 切（`ATTN_DP_SIZE=1`）估算：

- **单路 decode**：每个 token 要做约 156 次 all-reduce（78 层 × 2）。现在经主机内存，384 KiB 实测 50 µs/次，单路的小消息估计 20–30 µs；换成 PCIe-IPC 后约 5–15 µs，每个 token 省 2–4 ms。按旧配置单路约 23 ms/token 算，快一到两成。
- **高并发**：32 路 batch 的消息（384 KiB）每次只省约 15 µs，整体约 5%。
- **prefill**：基本不变。经主机内存的 8 卡 all-reduce 已有约 40 GB/s 总线带宽；源码注释里把 prefill 也交给 PCIe-IPC，首 token 时间反而慢了 66%。

attention 分组时（`ATTN_DP_SIZE>1`，现在的默认），机内通信换成了 MoE 前后的 gather/scatter，这部分的收益没有评估。

## 8. 验证计划

### 8.1 前置实验：零代码，验证硬件路径和收益

原理：

- `OverrideFbSize` 把显存顶部标成保留区、不交给分配器，客户端可见显存随之缩小；
- 再用 `RMForceStaticBar1=1` 强制开静态 BAR1，补丁原有的 BAR1 P2P 就能真正生效；
- 补丁按「BAR1 基址 + 物理偏移」算地址，前提是静态区从 BAR1 偏移 0 开始。这两台不会预留 mailbox（见 3.2 第 4 步），控制台预留按理也是 0（这些卡不接显示器），但没有直接确认。

以上都是读代码得出的推断，要靠实验确认。

实验期间每张卡只剩 28GB，GLM 服务跑不了；两台的服务是一体的，都要先停。机内测试只需在一台上做。

```bash
# 0. 两台：在 tmux 会话 glm53 里 Ctrl-C，停掉 GLM 服务

# 1. 测试机：释放 GPU 设备，卸载驱动模块
sudo systemctl stop nvidia-persistenced
sudo fuser -v /dev/nvidia*              # 确认没有其它进程占用（如 nv-hostengine）
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia

# 2. 带参数重新加载
sudo modprobe nvidia NVreg_RegistryDwords="OverrideFbSize=28672;RMForceStaticBar1=1"
sudo modprobe nvidia_uvm && sudo modprobe nvidia_peermem && sudo modprobe nvidia_drm
nvidia-smi -q -d MEMORY | grep -A3 BAR1   # Used 接近 28 GB，说明静态 BAR1 建起来了
nvidia-smi topo -p2p r

# 3. 打开 P2P 测试。直接运行 Python，不经 run_gpu_test.sh（它会按 cluster.env 的 GPU_P2P=0 关掉 P2P）
cd <部署目录>
sudo env NCCL_P2P_LEVEL=SYS MODEL_PATH=<模型目录> /opt/glm53-venv/bin/python gpu_local_test.py

# 4. 恢复
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia_peermem nvidia
sudo modprobe nvidia && sudo modprobe nvidia_uvm && sudo modprobe nvidia_peermem && sudo modprobe nvidia_drm
sudo systemctl start nvidia-persistenced
nvidia-smi -q -d MEMORY | grep -A3 BAR1   # Used 回到几 MiB；nvidia-smi 里显存回到 96GB

# 5. 两台用相同选项重新运行 ./deploy.sh <0|1> serve --skip-nccl-test
```

判断：

| 观察 | 结论 |
|---|---|
| 第 2 步后 nvidia-smi 报错或看不到卡 | 强制静态 BAR1 没有生效（客户端可见显存没有缩小），实验做不成；按第 4 步恢复 |
| BAR1 Used 接近 28GB，P2P 数据全部正确 | 硬件路径可用，继续看时延和带宽 |
| P2P 数据仍然不对 | 先排除静态区起始偏移不为 0；排除后说明这台平台的 P2P 硬件路径本身不可用，动态方案没有意义 |
| 第 5 部分 NCCL all-reduce 的时延 | 只作参考：这类平台上 NCCL 走 P2P 可能比走主机内存还慢，P2P 下 decode 要用的是 PCIe-IPC |
| 开 PCIe-IPC 后单路 TPOT 明显下降（测法见下） | 收益成立，可以在测试机上做原型 |
| TPOT 没有明显下降 | 不值得做 |

PCIe-IPC 的测法：实验期间每张卡只剩 28GB，放不下整个模型。用 8 层的 config.json 加 `--load-format dummy`，在测试机上起一个单机 TP8 的 SGLang，分别带和不带 `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` 启动，对比单路 decode 的 TPOT。权重是随机的，输出没有意义，只看速度。

实验的风险：GPU 初始化失败时，去掉参数重新加载驱动即可；最坏情况要重启这一台。

### 8.2 原型测试清单

- **数据正确性**：`gpu_local_test.py`（56 个方向和 all-reduce 都校验）；`nccl_xnode_test.py` 的数据校验；cuda-samples 的 `simpleP2P` 和 `simpleIPC`；PyTorch 跨卡拷贝随机数据（多种大小、非对齐偏移、多流并发）。
- **性能**：`p2pBandwidthLatencyTest`，以及 `gpu_local_test.py` 第 3–5 部分。
- **端到端**：开、关 PCIe-IPC all-reduce（`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`），固定随机种子对比输出；跑 `smoke_test.py` 和 GSM8K（`python -m sglang.test.few_shot_gsm8k`）。
- **资源和清理**：
  - 长时间循环分配、IPC 打开关闭、释放后，`nvidia-smi -q -d MEMORY` 的 BAR1 Used 要回到基线；
  - 故意映射超过 32GB，应干净地报错；
  - 进程被 kill 后窗口要释放；
  - 驱动卸载重载后没有残留。
- **托管内存**：多卡访问 `cudaMallocManaged` 的内存不出错（走迁移）。

## 9. 不碰内核的备选

- **NCCL 小消息调参**：试 `NCCL_ALGO`、`NCCL_PROTO`，用 `gpu_local_test.py` 第 5 部分对比 decode 大小的 all-reduce 时延，几分钟就能出结果。
- **MTP**（`ENABLE_MTP=1`）：对单路 decode 的潜在提升可能更大，风险只在用户态。但 SGLang 在 PP 下的投机解码还是实验功能：要设 `SGLANG_ENABLE_PP_SPEC=1`（启动脚本目前没设），不支持 DP attention，只能配 `ATTN_DP_SIZE=1`，没有验证过。
- **用户态的小消息 all-reduce**：经主机内存一次性交换，不需要 P2P。要自己写 kernel；单路小消息时，时延可能比 NCCL 的环形算法低。

## 10. 参考

### 源码位置

aikitoria/open-gpu-kernel-modules，595.71.05 加 P2P 补丁；部署机上的源码 HEAD 为 `860df942`，补丁本体是 `97fdda09`。路径相对仓库根目录，按函数名搜索即可。

| 文件 | 函数或符号 | 说明 |
|---|---|---|
| `src/nvidia/src/kernel/gpu/bus/arch/hopper/kern_bus_gh100.c` | `kbusIsPcieBar1P2PMappingSupported_GH100` | BAR1 P2P 要求两张卡都开静态 BAR1 |
| 同上 | `kbusGetBar1P2PDmaInfo_GH100` | 对端 DMA 地址取自静态映射的描述符 |
| 同上 | `kbusCreateP2PMappingForBar1P2P_GH100` | 建 P2P 对象时的检查和 IOMMU 映射 |
| `src/nvidia/src/kernel/gpu/bus/arch/turing/kern_bus_tu102.c` | `kbusIsStaticBar1Supported_TU102` | 静态 BAR1 的自动、强制判断 |
| 同上 | `kbusEnableStaticBar1Mapping_TU102` | 建静态 BAR1（调用 `kbusMapFbApertureSingle`） |
| `src/nvidia/src/kernel/gpu/bus/arch/maxwell/kern_bus_gm107.c` | `_kbusRequiresP2PMailboxBar1_GM107`、`kbusInitBar1_GM107` | mailbox 预留条件；BAR1 初始化和静态 BAR1 决策 |
| `src/nvidia/src/kernel/gpu/bus/arch/maxwell/kern_bus_gm200.c` | `kbusSetupMailboxes_GM200` | 第 89 行的断言 |
| `src/nvidia/src/kernel/gpu/bus/kern_bus.c` | `kbusSetupPeerBarAccess_IMPL`、`kbusMapFbApertureSingle_IMPL` | 第 298 行的断言；BAR1 映射接口 |
| `src/nvidia/src/kernel/gpu/bus/arch/pascal/kern_bus_gp100.c` | `kbusCreateP2PMapping_GP100` | 按连接类型分派到 BAR1 或 mailbox |
| `src/nvidia/src/kernel/gpu/bif/kernel_bif.c` | `_kbifInitRegistryOverrides` | 补丁改的 `p2pOverride`、`pcieP2PType` 默认值 |
| `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c` | `nvGpuOpsBuildExternalAllocPtes`、`nvGpuOpsBuildExternalAllocPhysAddrs`、`nvGpuOpsGetExternalAllocPtesOrPhysAddrs` | 对端页表项的生成 |
| `src/nvidia/interface/nvrm_registry.h` | `OverrideFbSize`、`RMForceStaticBar1` | 前置实验用的注册表项 |
| `src/nvidia/src/kernel/gpu/mem_mgr/arch/pascal/mem_mgr_gp100.c` | `memmgrHandleSizeOverrides_GP100` | `OverrideFbSize` 插入保留区 |
| `src/nvidia/src/kernel/gpu/mem_mgr/phys_mem_allocator/phys_mem_allocator.c` | `pmaGetClientAddrSpaceSize` | 客户端可见显存的大小 |
| `kernel-open/nvidia-uvm/uvm_map_external.c` | `nvUvmInterfaceDupMemory`、`nvUvmInterfaceGetExternalAllocPtes` 的调用处，`uvm_release_rm_handle` | 普通显存映射的建立和释放 |
| `kernel-open/nvidia-uvm/uvm_gpu.c` | `uvm_gpu_peer_phys_address` | 托管内存的对端地址（线性公式） |
| `kernel-open/nvidia-uvm/uvm_va_space.c` | `uvm_parent_gpus_are_bar1_peers` 的调用处 | BAR1 对端在地址空间里自动开启互访 |

### 取证命令

```bash
sudo lspci -vvv -s <GPU 的 PCI 地址> | grep -A2 "Physical Resizable BAR"   # BAR1 可选大小
nvidia-smi -q -d MEMORY | grep -A3 BAR1                                  # BAR1 用量，看静态 BAR1 是否建立
nvidia-smi topo -p2p r                                                    # 驱动报告的 P2P 读能力
sudo dmesg | grep NVRM                                                    # mailbox 断言

# SM 数：170 是 RTX 5090 的配置，188 是 RTX PRO 6000
python3 - <<'EOF'
import ctypes
cu = ctypes.CDLL("libcuda.so.1"); cu.cuInit(0); v = ctypes.c_int()
cu.cuDeviceGetAttribute(ctypes.byref(v), 16, 0)   # 16 = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
print("SM:", v.value)
EOF
```
