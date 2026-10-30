# GLM-5.3 FP8 · 两台 8×RTX PRO 6000 · PP2×TP8 部署脚本

适用环境：两台机器各有 8 张 RTX PRO 6000 Blackwell（96GB，SM120），机器之间是单路 400G RDMA（IB 或 RoCE 都可以），模型是 `zai-org/GLM-5.3`（默认就是 FP8）。

> 截至 2026 年 10 月，我没有找到「FP8 + SM120 + 跨机 PP」这个组合跑通的公开案例。所以这套脚本是按**逐步验证**来设计的：每一步都先确认通过，再进行下一步。

## 文件

| 文件 | 作用 |
|---|---|
| `cluster.env` | 唯一需要修改的配置文件，两台机器要保持一致 |
| `deploy.sh` | 一键部署（裸机或容器）：依次做宿主机检查、环境准备、自检、跨机测试、启动，节点0 就绪后自动跑冒烟测试 |
| `preflight.sh` | 上线前自检：GPU、RDMA 端口、RoCE GID、GPU 与网卡的 PCIe 距离、memlock、SGLang 参数支持情况、权重完整性、KV 容量估算 |
| `run_nccl_test.sh` + `nccl_xnode_test.py` | 按 PP2×TP8 的实际通信方式测跨机带宽和时延，并检查 NCCL 是否走了 RDMA、是否用上了 GPUDirect |
| `launch_glm53_pp2.sh` | 启动脚本，分 `bringup` 和 `serve` 两种模式 |
| `smoke_test.py` | 服务冒烟测试：检查乱码、思考内容分离、工具调用、长上下文、并发吞吐 |
| `install_env.sh` | 裸机：用 uv 建 Python 虚拟环境，并生成 `requirements.lock` 让两台完全一致 |
| `host_setup.sh` | 裸机：宿主机准备。默认只检查，`--apply` 才修改 |
| `rdma_perftest.sh` | 裸机：不经过 NCCL 的纯 RDMA 带宽测试，先排除网卡和交换机的问题 |

## 部署结构

```
                单路 400G RDMA（只传两级流水之间的 hidden states）
  节点0 ─────────────────────────────────────────────── 节点1
  GPU0-7：TP8，第 0-38 层 + embedding              GPU0-7：TP8，第 39-77 层 + lm_head
  HTTP :30000（对外提供服务）                       不对外提供服务
```

## 一键部署

`deploy.sh` 把下面「步骤」里的流程串起来：宿主机检查 → Python 环境 → 自检 → 跨机测试 → 启动 → 节点0 冒烟测试，任何一步失败就停下。两台用相同的选项各运行一次：

```bash
# 裸机，用已有的 SGLang 环境（也可以写进 cluster.env 的 PY_ENV）
./deploy.sh 0 bringup --py-env /home/test/sglang_env/bin/activate    # 节点0
./deploy.sh 1 bringup --py-env /home/test/sglang_env/bin/activate    # 节点1

# 容器：自动创建或复用容器 glm53（镜像取 cluster.env 的 CONTAINER_IMAGE），后续步骤在容器里执行
./deploy.sh 0 bringup --container    # 节点0
./deploy.sh 1 bringup --container    # 节点1
```

第一次用 `bringup`，跑通后去掉它（默认是 `serve`）再运行一次。服务在前台运行，Ctrl-C 停止；要在断开 ssh 后继续运行，请放在 tmux 或 screen 里。其它选项：

| 选项 | 作用 |
|---|---|
| `--host-apply` | 先执行 `host_setup.sh --apply`（非 root 时用 sudo） |
| `--rdma-test` | 加跑纯 RDMA 带宽测试 |
| `--skip-nccl-test` | 跳过跨机 NCCL 测试（测过一次之后可以加上） |
| `--no-smoke` | 节点0不自动跑冒烟测试 |
| `--dry-run` | 检查和测试照做，只打印启动命令，不启动 |

`PY_ENV` 指向的环境不存在时，`deploy.sh` 会调用 `install_env.sh` 创建，然后停下：先在节点0运行，把生成的 `requirements.lock` 同步到节点1 再运行节点1，两台环境都就绪后再同时运行。

BIOS 设置和系统软件（驱动、RDMA 库、CUDA Toolkit）仍要按 0b 手动准备。

## 步骤

### 0. 准备（两台机器都要做）

- **模型**：下载到两台机器的同一路径。约 750GB，建议放本地 NVMe；放 NFS 也能跑，只是加载慢。
- **配置**：编辑 `cluster.env`，然后把整个目录同步到另一台机器。
- **运行环境**：容器和裸机二选一，后面的步骤 1–6 完全相同。

#### 0a. 用容器

用 `deploy.sh --container` 时，下面的 `docker run` 会自动执行，不用手动做。

手动进容器运行脚本时，把 `cluster.env` 里的 `PY_ENV` 留空。宿主机上执行 `modprobe nvidia-peermem`，并在两台之间放通 TCP。

```bash
docker run -d --name glm53 --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband --cap-add IPC_LOCK \
  -v /home/test/ext_module:/home/test/ext_module -v "$PWD":/work -w /work \
  lmsysorg/sglang:latest sleep infinity
docker exec -it glm53 bash
```

RDMA 在容器里要正常工作，必须带上 `--device /dev/infiniband`、`--cap-add IPC_LOCK` 和 `--ulimit memlock=-1` 这三项。

#### 0b. 裸机

裸机少了容器这一层，RDMA 设备不需要透传，排查问题也更直接。代价是镜像里已经准备好的东西要自己装齐，而且两台机器的环境要自己保证一致。

1. **系统软件**（发行版自带的包管理器安装）：
   - **NVIDIA 驱动**：必须是开源内核模块版（nvidia-open），Blackwell 只支持这一版。
   - **RDMA 用户态库**：`rdma-core`、`ibverbs-utils`、`perftest`；用 NVIDIA DOCA-OFED 也可以。
   - **CUDA Toolkit**：版本与 torch 的 CUDA 主版本一致，用于 flashinfer 的 JIT 编译。装好后在 `cluster.env` 里设置 `CUDA_HOME`。
   - **uv**：`pip install uv`。
2. **BIOS**（两台都要做）：
   - 关闭 PCIe ACS。不关的话，GPU 之间、GPU 与网卡之间的 P2P 会被绕到 CPU。
   - IOMMU 关闭，或在内核参数里加 `iommu=pt`。
   - 开启 Above 4G Decoding 和 Resizable BAR。
3. **宿主机调优**：先只检查，确认后再实际修改。

   ```bash
   ./host_setup.sh
   sudo ./host_setup.sh --apply
   ```

   会处理以下几项。memlock 改完需要重新登录才生效。
   - GPU persistence 模式
   - nvidia-peermem（加载并设为开机加载）
   - memlock 不限
   - `numa_balancing=0`
   - CPU 调频设为 performance
   - 防火墙放行对端 IP
4. **Python 环境**（先装一台，再同步到另一台）：

   ```bash
   ./install_env.sh          # 第一台：安装 SGLang，生成 requirements.lock
   # 把整个目录（含 requirements.lock）同步到另一台
   ./install_env.sh          # 第二台：按 lock 文件安装
   ```

   两台打印的「环境指纹」必须一致。其它脚本会自动激活 `PY_ENV` 指定的虚拟环境。

   已有 SGLang 环境的话可以跳过这一步：把 `PY_ENV` 指向它（写目录或 `bin/activate` 都行，如 `/home/test/sglang_env/bin/activate`），或者运行 `deploy.sh` 时加 `--py-env`。
5. **先测纯 RDMA**（可选，但推荐在 NCCL 测试之前做）：

   ```bash
   ./rdma_perftest.sh server    # 节点0
   ./rdma_perftest.sh client    # 节点1
   ```

   400G 网卡的 BW average 应达到 360 Gb/s 以上。如果 perftest 编译时带了 CUDA，可以在两条命令后面都加上离网卡最近的 GPU 序号，测 GPUDirect 的带宽。

完成后照常进行步骤 1–6。

### 1. 自检（两台都跑）

```bash
./preflight.sh
```

所有 FAIL 项必须先解决。如果是 RoCE 网络，自检会列出所有 RoCE v2 GID，把标了「IPv4」的那个 index 填到 `IB_GID_INDEX`。

### 2. 跨机通信测试

```bash
./run_nccl_test.sh 0    # 节点0
./run_nccl_test.sh 1    # 节点1
```

结果在节点0输出，重点看三项：

- **8 对并发的合计带宽**：400G 的线速约 50 GB/s，达到 **35 GB/s 以上**算正常。
- **日志分析**：应该出现 `NET/IB : Using`，并且有 GDRDMA 通道。
- **PP 边界传输估算**：脚本会按实测带宽算出 prefill 每块、decode 每步的传输耗时。

测试通过后再进行下一步。结果表里有 FAIL 项时，两台的脚本都会以非 0 退出。

### 3. 首次启动（bringup 模式）

先只打印最终命令，不实际启动，检查一遍：

```bash
DRY_RUN=1 ./launch_glm53_pp2.sh 0 bringup
```

确认无误后，两台分别启动：

```bash
./launch_glm53_pp2.sh 0 bringup    # 节点0
./launch_glm53_pp2.sh 1 bringup    # 节点1
```

bringup 模式会关闭 CUDA Graph，上下文限制为 32K，最多 8 个并发，不开 MTP，NCCL 打印 INFO 日志。目的是先确认所有 kernel 和跨机通信都能跑通。加载约 750GB 权重，视磁盘速度需要十几分钟到半小时。

启动后在日志里确认以下几点：

1. **注意力后端**：确认选中的注意力后端是 flashinfer，而不是 nsa/dsa 后端（原因见下文「为什么默认用稠密注意力」）。
2. **层划分**：两级流水各自加载了一半的层。
3. **KV 容量**：`max_total_num_tokens` 是 KV 池的实际容量，可以和 preflight 的估算对照。

### 4. 冒烟测试（在节点0上，或任何能访问节点0的机器上）

```bash
python3 smoke_test.py --long 20000
```

### 5. 正式服务

两台机器都改用 `serve` 模式重新启动，参数取自 `cluster.env`。启动后再跑一次 `smoke_test.py --long 60000 --bench 8`。

服务在前台运行，要在断开 ssh 后继续运行，请放在 tmux 或 screen 里。两台机器是一个整体：任一台退出，另一台也会因为通信中断而退出。如果某一台没有退出、一直卡着，就两台都停掉，再一起重新启动。

### 6. 调优（每次只改一项，改完重跑冒烟测试）

| 想要 | 改什么 |
|---|---|
| 更快的单路速度 | `ENABLE_MTP=1`（在 PP 下没有验证过；如果报错或接受长度很低就关掉） |
| 两级负载更均衡 | `PP_LAYER_PARTITION=40,38`（第一级含 3 层稠密层，计算较轻） |
| 更多并发或更长上下文 | 调高 `MAX_RUNNING` / `CONTEXT_LENGTH`，总量受 KV 池容量限制 |
| 启动时 OOM | `MEM_FRACTION=0.80`、`CUDA_GRAPH_MAX_BS=16` |

## 关键取舍

**为什么用 PP2×TP8，而不是 TP16 或 EP**

TP 每一层都要做 all-reduce。如果做成 TP16 跨机，每一步 decode 要经过网络做上百次 all-reduce，prefill 每块要通过单路 400G 传几十 GB 数据。PP 只在两级流水的交界处传一次 hidden states，每个 token 约 24KB。EP 用的 DeepEP 要求机器内部有 NVLink，RTX PRO 6000 没有 NVLink。

**单路 400G 够不够用**

够用。以 16K token 的 prefill 块为例，传输量约 0.4GB，最坏情况（8 个 TP rank 各发一份完整副本）约 3GB，按 45GB/s 计算需要 10–70ms，而这块数据本身的计算要几百毫秒。decode 一步只有几 MB。实际数字以 `run_nccl_test.sh` 的输出为准。

**为什么用 BF16 KV**

社区实测在 SM120 上用 FP8 KV cache 跑 GLM-5，会输出乱码，或者只生成一个 token 就停止。

**为什么默认用稠密注意力（`ATTN_MODE=dense`）**

SM120 上的 DSA 稀疏注意力目前还跑不通：

- TileLang 实现的 DSA kernel 需要的共享内存超过了 SM120 每个 SM 的约 100KB（sglang issue #37105）。
- SM120 的 FP8 DSA 支持 PR（#29586）还没有合入主线。

社区在 SM120 上跑 GLM-5 的做法是用 `--attention-backend flashinfer` 绕开 DSA，按稠密 MLA 计算。这样做的代价有两点：

- **结果差异**：上下文在 2048 token 以内时，结果和稀疏注意力完全一样；超过之后每个 token 会看全部历史，和训练时的稀疏模式不同。
- **计算量**：decode 的计算量随上下文长度线性增长，prefill 按平方增长。所以默认把上下文设在 128K。

如果日志显示仍然选了 nsa/dsa 后端，并且报共享内存不足或架构不支持，说明你的 SGLang 版本没有这条绕行路径。这时需要换成社区的 SM120 补丁镜像（例如 `voipmonitor/llm-pytorch-blackwell:nightly`），这是第三方构建，使用前请自行审查。

**为什么关闭 DeepGEMM，FP8 GEMM 和 MoE 都走 Triton**

DeepGEMM 依赖 SM100 才有的 TMEM/tcgen05 指令，SM120 没有这些指令。

## 排错

| 现象 | 可能原因 | 处理 |
|---|---|---|
| NCCL 报 `Failed to initialize network`（脚本设置了 `NCCL_NET=IB`，不会悄悄退回 TCP） | 容器没挂载 `/dev/infiniband`、memlock 受限、`IB_HCA` 名称写错 | 按第 0 步补齐 docker 参数，用 `ibv_devices` 核对设备名 |
| RoCE 下连接卡住或超时 | GID index 不对，或两端不在同一子网、交换机没配 PFC/ECN | 用 preflight 列出的 RoCE v2 GID；检查交换机配置 |
| 合计带宽低于 15 GB/s，且没有 GDRDMA | nvidia-peermem 没加载、PCIe ACS 开着、IOMMU 拦截 | `modprobe nvidia-peermem`，BIOS 里关闭 ACS（或用 setpci 关），IOMMU 设为 passthrough |
| 节点1 一直等待，或初始化超时 | 端口没放通，或 `NET_IFNAME` / `NODE0_ADDR` 写错 | 两台之间放通 TCP；在节点1上 `ping -I <网卡> <节点0 IP>` |
| 输出乱码，或只生成一个 token | 用了 FP8 KV | 确认启动命令里是 `--kv-cache-dtype bf16` |
| 报 DeepGEMM 相关的 import 错误或 NameError | 走到了 DeepGEMM 代码路径 | 确认环境变量没被覆盖，必要时换 SM120 镜像 |
| 报 tilelang 或 shared memory 相关错误 | 走到了 DSA 后端 | 设置 `ATTN_MODE=dense`，或换 SM120 镜像 |
| `pcie_allreduce.cu` 处崩溃 | 设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | 启动脚本已经 unset 这个变量，检查是否有别的地方重新设置了 |
| 工具调用出现在 content 里，而不是 tool_calls | 解析器不对 | 必须用 `--tool-call-parser glm47` |
| 裸机：ssh 登录后 memlock 仍不是 unlimited | limits.d 只对新的登录会话生效 | 重新登录。`deploy.sh` 也会尝试把当前会话调到 unlimited，但不能超过硬限制 |
| `deploy.sh`：一台停在 FAIL，另一台卡在测试或启动 | 另一台在等它会合；或两台的选项不一致（比如只有一台加了 `--skip-nccl-test`） | 在卡住的那台 Ctrl-C，解决问题后两台用相同的选项一起重新运行 |
| 裸机：两台的环境指纹不一致 | 两台分别从网上安装，版本漂移了 | 删掉第二台的 `PY_ENV`，用第一台生成的 `requirements.lock` 重新运行 `install_env.sh` |
| 裸机：首次启动长时间卡在编译 / 报找不到 nvcc | flashinfer 在 SM120 上做 JIT 编译 | 装 CUDA Toolkit 并设置 `CUDA_HOME`；编译缓存在 `~/.cache/flashinfer`，第二次启动就快了 |
| 裸机：perftest 带宽正常，NCCL 带宽很低 | 没用上 GPUDirect（ACS/IOMMU 拦截，或 peermem/DMA-BUF 不可用） | 运行 `sudo ./host_setup.sh` 查看 ACS 和 IOMMU 状态 |
