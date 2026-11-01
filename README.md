# GLM-5.3 FP8 · 两台 8×RTX PRO 6000 · PP2×TP8 部署脚本

适用环境：两台机器各有 8 张 RTX PRO 6000 Blackwell（96GB，SM120），机器之间是单路 400G RDMA（IB 或 RoCE 都可以），模型是 `zai-org/GLM-5.3`（默认就是 FP8）。nvidia-smi 显示为 RTX 5090 的 96GB 卡也已跑通，要在 `cluster.env` 设 `GPU_P2P=0`（见下文「卡间 P2P 不可用时」）。

> 截至 2026 年 10 月，我没有找到「FP8 + SM120 + 跨机 PP」这个组合跑通的公开案例。所以这套脚本是按**逐步验证**来设计的：每一步都先确认通过，再进行下一步。

## 文件

| 文件 | 作用 |
|---|---|
| `cluster.env` | 唯一需要修改的配置文件，两台机器要保持一致 |
| `deploy.sh` | 一键部署（裸机或容器）：依次做宿主机检查、环境准备、自检、跨机测试、启动，节点0 就绪后自动跑冒烟测试 |
| `preflight.sh` | 上线前自检：GPU、RDMA 端口、RoCE GID、GPU 与网卡的 PCIe 距离、memlock、SGLang 参数支持情况、注意力配置组合、权重完整性、KV 容量估算、HiCache 内存用量估算 |
| `run_gpu_test.sh` + `gpu_local_test.py` | 本机 8 卡测试报告：单卡带宽、卡与卡之间（按 PIX/PXB/PHB/NODE/SYS 分类）的带宽和时延、8 卡 NCCL all-reduce |
| `run_nccl_test.sh` + `nccl_xnode_test.py` | 按 PP2×TP8 的实际通信方式测跨机带宽和时延，并检查 NCCL 是否走了 RDMA、是否用上了 GPUDirect |
| `launch_glm53_pp2.sh` | 启动脚本，分 `bringup` 和 `serve` 两种模式；`ROUTER=dynamo` 时节点0 同时起 Dynamo 前端和 Dynamo worker |
| `dynamo_worker.py` | 节点0 的 Dynamo SGLang worker 入口：修正 Dynamo 在 PP 下对 attention 组归属的判断（订阅全部 8 组的 KV 事件），KV 事件只走本机回环地址 |
| `smoke_test.py` | 服务冒烟测试：检查乱码、思考内容分离、工具调用、长上下文、并发吞吐 |
| `install_env.sh` | 裸机：用 uv 建 Python 虚拟环境，并生成 `requirements.lock` 让两台完全一致 |
| `install_dynamo.sh` | 节点0：装 NVIDIA Dynamo（`ROUTER=dynamo` 时用，`deploy.sh` 发现没装会自动运行）。装在叠加于 `PY_ENV` 的虚拟环境里，共用其中的 SGLang，`PY_ENV` 不变 |
| `host_setup.sh` | 裸机：宿主机准备。默认只检查，`--apply` 才修改 |
| `rdma_perftest.sh` | 裸机：不经过 NCCL 的纯 RDMA 带宽测试，先排除网卡和交换机的问题 |
| `frpc.tar` | frp 客户端镜像 `snowdreamtech/frpc:0.61`（`docker save` 导出），用 `docker load -i frpc.tar` 导入。镜像自带的 `frpc.toml` 只是上游示例，运行时挂载自己的配置；真实配置里有服务器地址和 token，不要提交 |
| `docs/dynamic_bar1_p2p.md` | 卡间 P2P 写错数据的根因分析、各修复路线对比，以及「动态 BAR1 P2P」驱动方案的可行性评估和验证计划 |

## 部署结构

```
                单路 400G RDMA（只传两级流水之间的 hidden states）
  节点0 ─────────────────────────────────────────────── 节点1
  GPU0-7：第 0-41 层 + embedding                   GPU0-7：第 42-77 层 + lm_head
  HTTP :30000 Dynamo 前端（对外提供服务）            原生 SGLang，不对外提供服务
    └ 按会话把请求固定到 8 组之一 → SGLang（跑在 Dynamo worker 里）
```

每台机器内部：MoE 按 TP8 切；attention 默认分 8 组（`ATTN_DP_SIZE=8`），每张卡一份完整的 attention，只处理分到自己这组的请求。节点0 前面是 Dynamo 前端（`ROUTER=dynamo`），把同一会话的请求固定到同一组，长会话的前缀缓存才命中得了。注意力用 DSA 稀疏注意力，KV cache 用 FP8（原因见「关键取舍」）。

## 一键部署

`deploy.sh` 把下面「步骤」里的流程串起来：配置检查 → 宿主机检查 → Python 环境（`ROUTER=dynamo` 时节点0 还会用 `install_dynamo.sh` 装 Dynamo）→ 自检 → 跨机测试 → 启动 → 节点0 冒烟测试，任何一步失败就停下。

配置检查会先打印 `cluster.env` 的必填项：两台的 IP、`MODEL_PATH`、`API_KEY`，裸机还有 `PY_ENV`。它会确认本机的 IP 和节点编号对得上；还会核对 `NET_IFNAME`、`IB_HCA`/`IB_PORT`、`IB_GID_INDEX`、`LINE_RATE_GBPS`、`CUDA_HOME` 和本机的实际情况是否一致，不一致时给出建议值。

两台用相同的选项各运行一次：

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
| `--gpu-test` | 加跑本机 GPU 测试（见步骤 1） |
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
   - 关闭 PCIe ACS。不关的话，GPU 之间、GPU 与网卡之间的 P2P 会被绕到 CPU。用 `sudo lspci -vvv | grep ACSCtl` 查看：`ReqRedir`、`CmpltRedir`、`EgressCtrl` 有一项是 `+` 就是开着；只有 `SrcValid+` 不影响 P2P。
   - IOMMU 关闭，或在内核参数里加 `iommu=pt`。IOMMU 开着时（包括 `iommu=pt`），Linux 启动时会自己给 PCIe 桥打开 ACS，BIOS 里关 ACS 不管用：要么在 BIOS 里关掉 IOMMU（Intel 叫 VT-d，AMD 叫 AMD-Vi），要么每次开机后、起服务前运行：

     ```bash
     for b in $(sudo lspci -vvv | awk '/^[0-9a-f]/ {dev = $1} /ACSCtl:.*(ReqRedir|CmpltRedir|EgressCtrl)[+]/ {print dev}'); do
       sudo setpci -s "$b" ECAP_ACS+0x6.w=0000
     done
     ```
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

自检还会做两项容易漏掉、出了问题又很难查的检查：

- **卡间 P2P**（`GPU_P2P=1` 时）：实际拷一次数据比对。驱动报告能 P2P、拷贝也不报错，数据却可能是错的，NCCL 和 SGLang 都察觉不到，表现为通信卡住或输出乱码。数据不对就设 `GPU_P2P=0`。
- **层划分**：第二级必须从 DSA indexer 为 `full` 的层开始（原因见「关键取舍」），不满足时给出可选的 `PP_LAYER_PARTITION`。启动脚本启动前也会再查一次。
- **注意力配置**：`ATTN_MODE`、`KV_CACHE_DTYPE`、`ATTN_DP_SIZE` 的组合（稠密注意力不能配 FP8 KV，attention 分组时不能开 MTP），并按这三项估算 KV 池容量。启动脚本也会拦下不支持的组合。`HICACHE_RATIO>0` 时还会估算每台要锁的主机内存，超过可用内存时报 FAIL。

然后测本机 8 卡。两台各跑一次，不需要同时运行，约 1 分钟；`deploy.sh` 加 `--gpu-test` 时也会跑。服务在跑时不要测。

```bash
./run_gpu_test.sh
```

报告同时写入 `logs/gpu_test_<时间>.log`，有 FAIL 项时以非 0 退出。报告分五部分：

1. **单卡**：主机内存 ↔ GPU 的拷贝带宽（主机内存取自卡所在的 NUMA 节点）、显存拷贝带宽，以及 PCIe 链路有没有跑到应有的代数和宽度。PCIe 5.0 x16 一般能到 50 GB/s 以上。
2. **连接类型**：`nvidia-smi topo -m` 里每对卡之间的连接（PIX/PXB/PHB/NODE/SYS），以及哪些卡对不能 P2P。
3. **卡与卡之间**：每对卡的单向、双向带宽和 16 字节拷贝时延，与 CUDA 示例 `p2pBandwidthLatencyTest` 同口径；拷完会校验数据。数据不对时报 FAIL；已设 `GPU_P2P=0` 时只报 WARN，因为部署不会再用 P2P。
4. **按连接类型汇总**：每类卡对的带宽和时延，并列出明显比同类慢的卡对。
5. **8 卡 all-reduce**：用部署时的 NCCL 设置，按 decode 一步、prefill 一块的消息大小测，也就是 TP8 实际用的通信，并校验结果。总线带宽 20 GB/s 以上算正常。

跨 CPU（SYS）的卡对明显比其它类型慢时，可以运行 `NCCL_P2P_LEVEL=NODE ./run_gpu_test.sh`，让这些卡对改走主机内存，再对比第 5 部分的结果。

### 2. 跨机通信测试

```bash
./run_nccl_test.sh 0    # 节点0
./run_nccl_test.sh 1    # 节点1
```

结果在节点0输出，重点看四项：

- **数据校验**：机内 all-reduce 的结果、跨机两个方向 send/recv 收到的内容必须正确。P2P 或 GPUDirect RDMA 写错数据时 NCCL 不会报错，只能这样查出来。
- **8 对并发的合计带宽**：400G 的线速约 50 GB/s，达到 **35 GB/s 以上**算正常。达不到时先跑 `rdma_perftest.sh` 看裸 RDMA 带宽，再查网卡的 PCIe 链路（`sudo lspci -vv -s <网卡 PCI 地址> | grep LnkSta`）：链路只有 Gen5 x8 时单向最多约 31 GB/s，裸 RDMA 和 NCCL 都只能跑到 30 GB/s 左右（会报 WARN）。只跑 PP 的话这个带宽也够用。
- **日志分析**：应该出现 `NET/IB : Using`。GPU 和网卡在同一个 PCIe 交换芯片下时还应有 GDRDMA 通道；GPU 与网卡之间隔着 CPU（topo 里是 NODE/SYS），或 CUDA 报告 GPU 不支持 GDR（GeForce 卡，preflight 会显示）时，NCCL 本来就不用。
- **PP 边界传输估算**：脚本会按实测带宽算出 prefill 每块、decode 每步的传输耗时。

通信超过 5 分钟没完成（`NCCL_TEST_TIMEOUT` 可调）会报错退出，常见原因是机内 P2P 不通。

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

1. **注意力后端**：默认配置（`ATTN_MODE=dsa` + FP8 KV）下，日志里应有 `Set DSA backends for GLM FP8 KV Cache on SM120/SM121: prefill=flashinfer_sparse_mla, decode=flashinfer_sparse_mla`。`ATTN_MODE=dense` 时确认 `server_args` 里 `attention_backend` 是 flashinfer；这时日志里的 `Set DSA backends for bfloat16 KV Cache` 只是 SGLang 预先选好的 DSA 后端，用不到。
2. **层划分**：两级流水按 `PP_LAYER_PARTITION` 加载（默认 42/36）。
3. **KV 容量**：`max_total_num_tokens` 是 KV 池的实际容量，可以和 preflight 的估算对照。两级会取同一个值；attention 分组时这是每组的容量。

### 4. 冒烟测试（在节点0上，或任何能访问节点0的机器上）

```bash
python3 smoke_test.py --long 20000
```

### 5. 正式服务

两台机器都改用 `serve` 模式重新启动，参数取自 `cluster.env`。启动后再跑一次 `smoke_test.py --long 60000 --bench 8`。

服务在前台运行，要在断开 ssh 后继续运行，请放在 tmux 或 screen 里。两台机器是一个整体：任一台退出，另一台也会因为通信中断而退出。如果某一台没有退出、一直卡着，就两台都停掉，再一起重新启动。

参考数据（2026-10-31，两台 8×96GB SM120，nvidia-smi 显示为 RTX 5090，`GPU_P2P=0`；跨机无 GPUDirect RDMA，网卡 PCIe 只有 Gen5 x8，裸 RDMA 约 30 GB/s；SGLang 0.5.23，serve 模式参数：DSA + FP8 KV + attention 8 组 + 512 并发，入口是 SGLang 自带的 HTTP 服务（`ROUTER=sglang`）；2026-11-01 改成 Dynamo 入口后复测的项目在表里注明）：

| 场景 | 结果 |
|---|---|
| 启动 | 第一次约 2.5 分钟就绪（权重从 NVMe 读，节点0 约 95 秒，每台只读自己那一级）；之后权重走页缓存，约 1.5 分钟就绪。`ROUTER=dynamo` 时前端几秒就起来，SGLang 加载完、worker 注册后才列出模型，约 2 分钟（`deploy.sh` 的配置检查和自检另需约 1 分钟） |
| KV 池 | 每组 `max_total_num_tokens` 约 95 万，8 组合计约 761 万 |
| 单路，1K 上下文 | 首 token 0.6 s，decode 约 25 tok/s |
| 单路，30K 上下文 | 首 token 约 11 s，decode 约 25 tok/s |
| 512 并发，每路输入 1K、输出 2K | 总输出约 3700 tok/s，单路约 8 tok/s。经 Dynamo（`--backend sglang-oai`）是 2800–2930 tok/s，原因见表下 |
| 冒烟测试（`--long 60000 --bench 8`） | 5 项全部通过（`ROUTER=dynamo` 下同样通过） |

两路以上并发时单路速度约减半：PP 把请求分成两个微批，两级交替处理。

经 Dynamo 打 512 并发时，逐 token 速度和直连一样（TPOT 中位 121–123 ms），少的那两成多全在分组：

- SGLang 轮询正好每组 64 个；Dynamo 按负载分，分成每组 63–65 个。
- 每组最多同时跑 64 个（`MAX_RUNNING/ATTN_DP_SIZE`），多出来的几个要等前一整批跑完才开始。这次有 4 个请求的首 token 等了 100 秒以上，独自把总时长拉长了约 80 秒。
- 这种合成负载所有请求一样长、同时开始、同时结束，才会把这个问题放大；真实负载里请求长短不一，空位很快就会腾出来。

另外 random-ids 经 OpenAI 接口发送时是先解码成文本、再由前端重新分词的，每个请求的实际输入是 1066–1111 个 token，比直连多 4%–8%。

几种配置的对比如下（都是 `ROUTER=sglang`）。压测用 SGLang 自带的 `python -m sglang.benchmark.serving --dataset-name random-ids`（随机 token，请求持续到达，节点上没有外网时加 `HF_HUB_OFFLINE=1`），表里是总输出 tok/s。GSM8K 用 `python -m sglang.test.few_shot_gsm8k`，1319 题、5-shot。

| 场景 | 稠密 + BF16 KV，attention TP8，32 并发上限（旧默认） | DSA + FP8 KV，attention TP8 | DSA + FP8 KV，attention 8 组（默认的注意力配置） |
|---|---|---|---|
| KV 池 | 72 万 token | 123 万 | 每组 95 万 × 8 组 |
| GSM8K | 92.9% | 92.6% | 92.0% |
| 输入 1K、输出 256，32 并发 | 336 | 508 | — |
| 输入 1K、输出 256，128 并发 | — | 1175 | 1013 |
| 输入 1K、输出 256，256 并发 | — | 1401 | 1441 |
| 输入 8K、输出 512 | 190（32 并发） | 399（64 并发） | 421（64 并发） |
| 输入 1K、输出 4K，256 并发 | — | 2437（KV 池不够，有请求被换出重算） | 2417 |
| 输入 1K、输出 2K，512 并发 | — | 没测（要约 157 万 token，超过 KV 池） | 3722 |

GSM8K 三者相差不到 1 个百分点，在 1319 题的误差范围内（约 ±0.7%）。

上面的 random-ids 每个请求互不相关，测不出 prefix cache 的作用。agent 类负载另用 Claude Code 的真实会话回放测过一轮（2026-10-31）。

- **数据**：SemiAnalysis 公开的 Claude Code 抓包 trace `semianalysisai/cc-traces-weka-062126-256k`，393 个会话、6.8 万个请求，含子代理。trace 里只有每个请求的长度、前缀共享结构（hash_id）和时间。主会话的请求输入中位 13.5 万 token，其中新增的只有中位 1600 token，其余都能命中前缀缓存；输出中位 483 token；相邻两轮间隔中位 2 秒，长尾到几分钟。
- **回放**：aiperf 0.13.0 的 AgentX 规则（`--scenario inferencex-agentx-mvp`）。按 hash_id 合成同样长度、同样共享结构的提示词，每棵会话树按录制的间隔一轮轮发；开始前先把每条会话预热到 trace 中随机的一点。每档回放 10 分钟（AgentX 要求 15 分钟，所以加了 `--unsafe-override`），`ignore_eos`，种子固定，同一并发下各配置回放的是同一批会话、同一组起点。
- **配置**：都是 DSA + FP8 KV。TP8 用 `MAX_RUNNING=256`、`CUDA_GRAPH_MAX_BS=128`，8 组用默认的 512、64。会话的峰值上下文多数超过 128K（393 个里只有 83 个在 131072 以内），10-31 这几轮把 `CONTEXT_LENGTH` 临时调到 300000。
- **Dynamo 入口（2026-11-01 加测）**：就是现在的默认配置（`ROUTER=dynamo`、`CONTEXT_LENGTH=262144`）。aiperf 设 `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`，给每个会话带上 `X-Dynamo-Session-ID`（子代理另带父会话的 ID）；另测了不带会话头的，看只按打分选组时的效果，分别用 Dynamo 自己默认的重算权重 1 和现在默认的 8（`ROUTER_PREFILL_WEIGHT`，见「为什么入口用 Dynamo」）。权重 8 那档是在同一套服务上另起一个只改了这个参数的前端测的。「测试代理」是 10-31 临时写的：按 aiperf 写在第一条 user 消息里的会话树标记固定分组，新会话分给活跃会话最少的组。

「并发」是同时进行的会话树数，不是同时在跑的请求数：会话大部分时间在等工具或用户，16 棵树平均只有 2.5 个请求在跑。表里的输出 tok/s 是 10 分钟内的总输出，TTFT 取中位数，ITL 是平均值，命中率是输入 token 里不用重算的占比，显存命中和从主机内存拷回的都算（理论上限约 94%）。

| 配置 | 并发 8 | 并发 16 | 并发 32 | 并发 64 |
|---|---|---|---|---|
| attention TP8（`ATTN_DP_SIZE=1`） | 40 tok/s · TTFT 0.4 s · ITL 25 ms · 命中 92% | 120 · 0.4 s · 41 ms · 93% | 33 · 59 s · 178 ms · 34%（KV 池装不下，大量重算） | 没测（预计更差） |
| TP8 + HiCache（`HICACHE_RATIO=2`） | — | — | 145 · 5.2 s · 63 ms · 91%（命中的约一半是从内存拷回的） | 51 · 257 s · 150 ms · 50%（内存池也装不下） |
| attention 8 组 + `ROUTER=sglang`（SGLang 轮询分组） | 12 · 13 s · 88 ms · 30% | 31 · 7.1 s · 157 ms · 48% | 31 · 4.0 s · 370 ms · 41% | — |
| attention 8 组 + 测试代理按会话固定分组 | — | 62 · 0.6 s · 68 ms · 94% | 122 · 0.7 s · 91 ms · 94% | 184 · 1.1 s · 160 ms · 92% |
| **attention 8 组 + Dynamo（默认），带会话头** | — | 64 · 0.6 s · 61 ms · 95% | 124 · 0.6 s · 86 ms · 95% | 204 · 1.0 s · 155 ms · 93% |
| attention 8 组 + Dynamo，不带会话头，`ROUTER_PREFILL_WEIGHT=8`（默认） | — | — | — | 176 · 1.2 s · 177 ms · 91% |
| attention 8 组 + Dynamo，不带会话头，`ROUTER_PREFILL_WEIGHT=1`（Dynamo 自己的默认） | — | — | — | 83 · 4.6 s · 490 ms · 69% |

- **SGLang 轮询分组最不适合 agent 负载**（`ROUTER=sglang`，10-31 之前的默认）：同一会话的下一轮多半落到别的组，要把十几万 token 的历史重算一遍。16 并发时输出只有 TP8 的四分之一，ITL 也被别组的重算拖到 157 ms。
- **TP8 撑到约 16 个会话**：首 token 0.4 秒，单路 28–40 tok/s。32 个会话的上下文合计超过 123 万 token 的 KV 池，命中率掉到 34%，吞吐反而下降。
- **HiCache 把 TP8 的上限推到约 32 个会话**：显存放不下的前缀存到主机内存，回访时拷回。代价是首 token 变慢（中位 5.2 s，约 7.7 万 token 拷回要 2 秒多），每台锁约 575 GB 内存。64 个会话时内存池也装不下，同样崩掉。
- **按会话固定分组后，8 组撑到 64 个会话**：命中率回到理论值，KV 合计 761 万 token，吞吐随并发一路上升。Dynamo 和测试代理在 16、32 个会话时持平，64 个会话时多 11%（204 对 184 tok/s）。对照前端日志和 aiperf 的逐请求记录：同一会话相邻两轮一次也没有换过组。代价是单路慢（64 并发时约 7 tok/s）；各组忙闲不均：会话的活跃程度差别很大，16 个会话时一组承担了六成的输入，几轮累计下来输入最多的一组约占三成。
- **最好带会话头，不带时靠重算权重**：不带头时 Dynamo 按「重算权重 × 要重算的 KV 块 + 在跑请求占的 KV 块」给各组打分。
  - 权重取 Dynamo 自己默认的 1 时，64 个会话里同一会话相邻两轮有 58% 换了组，命中率掉到 69%，吞吐只有带头时的四成。一个十万 token 的会话在 decode 时占约 1500 个 KV 块，两组之间差一两个长会话，负载差就超过重算整段前缀的代价，路由器就把会话挪走。
  - 权重 8（`ROUTER_PREFILL_WEIGHT` 的默认值）时换组降到 12%。32 次换组里有 25 次换去的组已经缓存了六成以上的前缀（子代理和父会话共用的部分），从头重算的只有 7 次，换组造成的重算从 304 万 token 降到 42 万。吞吐 176 tok/s，是权重 1 的 2.1 倍、带头时的 86%。带会话头时权重只影响新会话第一次选组：改成 8 后带头复测 64 个会话，202 tok/s、命中 92%、换组 0 次，和权重 1 时持平。
  - Claude Code 自带 `x-claude-code-session-id`，不用改；其它客户端最好在请求里加 `X-Session-ID`。

### 6. 调优（每次只改一项，改完重跑冒烟测试）

| 想要 | 改什么 |
|---|---|
| 单路更快（交互式、并发不高） | `ATTN_DP_SIZE=1`、`MAX_RUNNING=256`、`CUDA_GRAPH_MAX_BS=256`；再快可以试 `ENABLE_MTP=1`（只能配 `ATTN_DP_SIZE=1`；PP 下没有验证过，报错或接受长度很低就关掉） |
| 多轮对话、agent（每轮都带着很长的历史） | 默认配置就是按这类负载定的：attention 8 组 + Dynamo 按会话固定分组，`CONTEXT_LENGTH=262144`（Claude Code 的会话常超过 128K）。会话不多（约 16 个以内）、更看重单路速度时，可以改成 `ATTN_DP_SIZE=1`、`MAX_RUNNING=256`、`CUDA_GRAPH_MAX_BS=128`，所有请求共用一份 prefix cache，单路 28–40 tok/s；到约 32 个会话再加 `HICACHE_RATIO=2`（每台锁约 575 GB 内存）。实测见「正式服务」的 Claude Code 回放，原因见「关键取舍」 |
| 短请求、彼此没有共同前缀 | 入口用哪个都行，`ROUTER=sglang` 少一跳转发；要鉴权（`API_KEY`）就只能用它 |
| 两级负载更均衡 | `PP_LAYER_PARTITION` 在 `42,36` 和 `38,40` 之间换（第二级只能从 full 层开始，preflight 会列出可选值；第一级含 3 层稠密层，计算较轻） |
| 更多并发或更长上下文 | 调高 `MAX_RUNNING`（`CUDA_GRAPH_MAX_BS` 不小于 `MAX_RUNNING/ATTN_DP_SIZE`）/ `CONTEXT_LENGTH`；总量受 KV 池容量限制，attention 分组时单个请求只能用所在组的那份 |
| 启动时 OOM | `MEM_FRACTION=0.80`，`CUDA_GRAPH_MAX_BS` 调小 |

## 关键取舍

**为什么用 PP2×TP8，而不是 TP16 或 EP**

TP 每一层都要做 all-reduce。如果做成 TP16 跨机，每一步 decode 要经过网络做上百次 all-reduce，prefill 每块要通过单路 400G 传几十 GB 数据。PP 只在两级流水的交界处传一次 hidden states，每个 token 约 24KB。EP 用的 DeepEP 要求机器内部有 NVLink，RTX PRO 6000 没有 NVLink。

**单路 400G 够不够用**

够用。以 16K token 的 prefill 块为例，传输量约 0.4GB，最坏情况（8 个 TP rank 各发一份完整副本）约 3GB，按 45GB/s 计算需要 10–70ms，而这块数据本身的计算要几百毫秒。decode 一步只有几 MB。实际数字以 `run_nccl_test.sh` 的输出为准。

**为什么默认用 DSA 稀疏注意力 + FP8 KV（`ATTN_MODE=dsa`、`KV_CACHE_DTYPE=fp8_e4m3`）**

SGLang 0.5.23 给 GLM 系列在 SM120 上加了专用路径：DSA 配 FP8 KV 时，prefill 和 decode 都用 FlashInfer 的 SM120 稀疏 MLA kernel（`flashinfer_sparse_mla`）。

- **容量**：每层每 token 从 1152 字节（BF16）降到 656 字节（FP8 latent、每 128 维一个 FP32 scale、BF16 rope），同样的显存多放约 72%。
- **精度**：GSM8K 和稠密 + BF16 KV 相差不到 1 个百分点，在误差范围内（见上面的对比表）。
- **和训练一致**：DSA 每个 token 只看 indexer 选出的 2048 个历史 token，和训练时一样；decode 读 KV 的量不随上下文增长，30K 上下文的单路 decode 速度和 1K 一样。

保留了 `ATTN_MODE=dense`（`--attention-backend flashinfer`，按稠密 MLA 计算）给没有这条路径的 SGLang 版本。旧版本在 SM120 上的情况：

- FP8 KV 跑 GLM-5 会输出乱码，或者只生成一个 token 就停止。所以稠密注意力只能配 BF16 KV，启动脚本和 preflight 会拦下稠密 + FP8 的组合。
- DSA 跑不通：TileLang 实现的 DSA kernel 需要的共享内存超过了 SM120 每个 SM 的约 100KB（sglang issue #37105）。

稠密注意力的代价有两点：上下文超过 2048 token 后每个 token 会看全部历史，和训练时的稀疏模式不同；decode 的计算量随上下文长度线性增长，prefill 按平方增长。旧版本跑不通时，也可以换社区的 SM120 补丁镜像（例如 `voipmonitor/llm-pytorch-blackwell:nightly`），这是第三方构建，使用前请自行审查。

**为什么 attention 默认分 8 组（`ATTN_DP_SIZE=8`）**

GLM-5.3 的 attention 是 MLA：64 个头共用一份压缩 KV。按 TP8 切时，每张卡分到 8 个头，却要存全部请求的 KV，8 张卡存的是同一份（每卡 123 万 token 就是整个 KV 池）。分成 8 组后，每张卡放一份完整的 attention 权重（每卡多占约 7.6 GB），只处理分到自己这组的请求，KV 池变成每组 95 万 token、合计 761 万。MoE 仍按 TP8 切。

实测（见上面的对比表）：

- 128 并发时 TP8 快 16%：分组后每张卡每一步都要读完整的 attention 权重。
- 256 并发时两者持平。
- 再往上 TP8 的 KV 池放不下；分组后 512 并发比 256 并发多出 53% 的吞吐。
- 单路 decode 约 25 tok/s，比旧默认（稠密 + BF16 + TP8）的 43 tok/s 慢：一个请求的 attention 只在一张卡上算。

代价：

- **prefix cache 按组分开**：每组各有一份。SGLang 自己按轮询或负载把请求分到各组，不看前缀，多轮对话的下一轮多半落到另一组，要重新 prefill。Claude Code 回放里，前缀命中率从理论上的约 94% 掉到 30%–48%。所以默认在前面加 Dynamo，把同一会话固定到同一组（`ROUTER=dynamo`，见下一节），命中率回到理论值。用 `ROUTER=sglang` 时，多轮对话、agent 场景改用 `ATTN_DP_SIZE=1`。
- **单个请求只能用所在组的 KV**：每组 95 万 token，远大于 `CONTEXT_LENGTH`，一般不受影响。
- **不能开 MTP**：SGLang 在 PP 下的投机解码不支持 DP attention。
- **chunked prefill 平分到各组**：每组每块 16384/8 = 2048 token。

**为什么入口用 Dynamo（`ROUTER=dynamo`，默认）**

节点0 的 HTTP_PORT 上跑 NVIDIA Dynamo 1.6.1 的前端，它决定每个请求进哪一组：

- **怎么选组**：
  - 请求带 `X-Session-ID`、`X-Dynamo-Session-ID` 头，或 Claude Code 自带的 `x-claude-code-session-id` 头时，同一会话固定在它第一次选中的组。这是硬约束：即使内容换成别的会话的前缀也不换组。会话空闲 `SESSION_TTL` 秒（默认 1800）后解除。
  - 新会话和不带会话头的请求按打分选组（`--router-mode kv`）：每组的分数是「`ROUTER_PREFILL_WEIGHT` × 要重算的 KV 块数（扣掉该组已经缓存的前缀）+ 该组在跑请求占的 KV 块数」，选分数最低的。Dynamo 自己的默认权重是 1，长会话的 decode 占用很容易盖过重算整段前缀的代价，会话被挪到别的组；这里默认 8，不带头的多轮对话多数也会回到缓存所在的组（回放数据见「正式服务」）。
- **结构**：
  - 前端（`python -m dynamo.frontend`）负责 HTTP、chat template、分词，以及解析思考内容和工具调用（`--dyn-reasoning-parser glm45 --dyn-tool-call-parser glm47`）。它把 token 和选好的组发给节点0 的 Dynamo worker。
  - worker 进程里跑的就是 SGLang 引擎，请求按选好的组下发。节点1 不变，仍是原生 SGLang。
  - 前端和 worker 用本地目录（`logs/dynamo_discovery`）做服务发现，请求和响应走本机回环地址，不需要 etcd 或 NATS。
- **KV 事件**：
  - SGLang 加了 `--kv-events-config`，第一级流水的 8 个调度进程把各组缓存了哪些块，发到本机 127.0.0.1 的 `HTTP_PORT+100` 起的 8 个端口。事件里带 token，所以只绑回环地址。
  - Dynamo 1.6.1 默认认为多机时 attention 组按机器切分（8 组、2 台，每台 4 组）。PP2 下每台机器都有全部 8 组，不改的话节点0 只订阅 4 组。`dynamo_worker.py` 把这个判断改成按每级流水有几台机器来算，节点0 订阅全部 8 组。
- **API**：
  - OpenAI 的 `/v1/chat/completions`、`/v1/completions`。
  - Anthropic 的 `/v1/messages`（`--enable-anthropic-api`，Dynamo 标为实验性）。thinking、tool_use、流式输出都测过。
- **端口**：
  - `HTTP_PORT`（30000）是前端。
  - `HTTP_PORT+1`（30001）是 worker：`/metrics` 里有 SGLang 的 `sglang:` 指标，另有 `/health`，`POST /engine/flush_cache` 清前缀缓存。
  - Dynamo 的事件平面还会在 0.0.0.0 上开十来个随机端口，传的是块哈希和负载，不含 token。
- **限制**：
  - **不支持 `API_KEY`**：Dynamo 前端不做鉴权，`ROUTER=dynamo` 配 `API_KEY` 时启动脚本和 preflight 会拦下。要鉴权就在前面另加一层网关，或者用 `ROUTER=sglang`。
  - **版本搭配没有官方背书**：ai-dynamo 1.6.1 声明配 SGLang 0.5.21。`install_dynamo.sh` 用 `--no-deps` 只装 Dynamo，跑在 `PY_ENV` 的 SGLang 0.5.23 上，冒烟测试和下面的回放都通过了。升级任何一边都要重测；`--router-prefill-load-scale` 在 Dynamo 1.7 会移除（改在 `--router-policy-config` 里设），preflight 会检查前端认不认启动脚本用到的参数。
  - **前端绑核**：多 NUMA 的机器上，前端会绑到网卡所在的 NUMA 节点。Dynamo 对跨 NUMA 运行会告警，说会严重变慢。
  - **前端退出即停服**：前端一退出，启动脚本会停掉整个服务，和一台退出另一台也退出一样。

**主机内存二级前缀缓存（`HICACHE_RATIO`，默认关）**

`HICACHE_RATIO>0` 时启动脚本加 `--enable-hierarchical-cache --hicache-ratio <倍数>`（SGLang 的 HiCache）。每张卡在主机内存里开一个「显存 KV 池 × 倍数」的锁页池。默认是 write_through：算出来的 KV 同时写一份到内存，显存里的被挤掉后，会话回访时从内存拷回，不用重算。SGLang 0.5.23 给 DSA 单独做了 indexer K 的主机池，换入换出会连它一起拷。

- **容量**：TP8 下 8 张卡存的是同样的内容，内存也是 8 份。倍数 2 时每卡 246 万 token（67.9 GB）加 indexer 3.9 GB，每台约 575 GB；前缀缓存总容量约 247 万 token，是显存池的 2 倍。分组时每张卡存不同的会话，同样倍数下容量是 TP8 的约 6 倍（没测）。preflight 会按 `MemAvailable` 估算每台的用量，超出时报 FAIL。
- **速度**：SGLang 的 JIT 拷贝 kernel 不支持 FP8 MLA 每 token 656 字节的布局（启动日志有 `Unsupported element_size = 656 for JIT HiCache kernel`），退回 sgl-kernel 的旧 kernel。实测约 7.7 万 token 从内存拷回要 2 秒多，比重算（约 10 秒）快得多，但比显存命中慢。
- **正确性**：用 6 万 token 的检索题测过：先问一次，用 150 万 token 的别的文档把它挤出显存，再问。前缀全部从内存拷回，答案正确。
- **代价**：锁页内存会把页缓存里的权重挤出去，下次重启要重新读盘。不适合内存紧张、或和别的服务共用内存的机器。

**为什么第二级要从 DSA indexer 为 full 的层开始**

GLM-5.3 的 DSA indexer 只在 `config.json` 里 `indexer_types` 为 `full` 的层（第 0、1、2 层和之后每 4 层一次：第 6、10、14…层）计算 top-k，`shared` 层直接复用上一层的结果。即使用稠密注意力，SGLang 也照样构建 indexer。第二级从 `shared` 层开始时，SGLang 要求第一级把 top-k 随 hidden states 一起传过去；稠密注意力下第一级不算 top-k，处理第一个请求时就会报 `PP stage ending at layer N must forward DSA topk_indices`。DSA 下跨级传 top-k 没有验证过，所以两种模式都按从 full 层开始划分。

78 层的默认均分是 39/39，第 39 层正好是 `shared`，所以 `cluster.env` 默认 `PP_LAYER_PARTITION=42,36`；`38,40` 也可以。preflight 和启动脚本都会检查。

**卡间 P2P 不可用时（`GPU_P2P=0`）**

有的机器上驱动报告能 P2P，拷贝也不报错，数据却是错的。这次遇到的情况：消费级卡装了开源驱动的 P2P 补丁（P2P 走 PCIe BAR1），BAR1 只有 32GB、显存却有 96GB（很可能就是写不对的原因）；`run_gpu_test.sh` 测出的卡间带宽超过 PCIe 链路上限、数据全部校验失败，`dmesg` 里有 `kern_bus.c` 相关的 NVRM 断言。这时 NCCL 走 P2P 会在第一次通信时卡住，SGLang 的自定义 all-reduce 会悄悄算错（SGLang 默认不检查 P2P 能不能用）。

设 `GPU_P2P=0` 后：NCCL 设 `NCCL_P2P_DISABLE=1`，机内改走主机共享内存，8 卡 all-reduce 实测总线带宽约 40 GB/s；SGLang 加 `--disable-custom-all-reduce`，并强制关掉 PCIe-IPC all-reduce（`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=0`），all-reduce 全走 NCCL。这两条路径都靠 P2P，SGLang 默认都不先验证 P2P 的数据对不对，所以 P2P 修好之前不要手动打开 PCIe-IPC。

根因的完整证据链、几条修复路线的对比，以及给驱动加「动态 BAR1 P2P」的方案评估，见 [docs/dynamic_bar1_p2p.md](docs/dynamic_bar1_p2p.md)。

**为什么关闭 DeepGEMM，FP8 GEMM 和 MoE 都走 Triton**

DeepGEMM 依赖 SM100 才有的 TMEM/tcgen05 指令，SM120 没有这些指令。

## 排错

| 现象 | 可能原因 | 处理 |
|---|---|---|
| NCCL 报 `Failed to initialize network`（脚本设置了 `NCCL_NET=IB`，不会悄悄退回 TCP） | 容器没挂载 `/dev/infiniband`、memlock 受限、`IB_HCA` 名称写错 | 按第 0 步补齐 docker 参数，用 `ibv_devices` 核对设备名 |
| RoCE 下连接卡住或超时 | GID index 不对，或两端不在同一子网、交换机没配 PFC/ECN | 用 preflight 列出的 RoCE v2 GID；检查交换机配置 |
| 合计带宽低于 15 GB/s，且没有 GDRDMA | nvidia-peermem 没加载、PCIe ACS 开着、IOMMU 拦截 | `modprobe nvidia-peermem`；按 0b 第 2 步关闭 ACS、IOMMU 设为 passthrough |
| 合计带宽和裸 RDMA（`rdma_perftest.sh`）都只有约 30 GB/s（240 Gb/s） | 网卡 PCIe 只协商到 Gen5 x8：`lspci -vv` 的 LnkSta 显示 `Width x8 (downgraded)` | 把网卡换到 Gen5 x16 插槽，或检查 BIOS 里这个插槽的通道拆分设置；只跑 PP 的话不改也够用 |
| 日志里没有 GDRDMA，带宽正常 | GPU 与网卡之间隔着 CPU（NODE/SYS），或 GPU 不支持 GDR（GeForce 卡），数据经主机内存中转 | 正常 |
| NCCL 测试或服务在第一次通信时卡住，几分钟后 watchdog 超时退出 | 机内 P2P 不通：驱动说能 P2P，数据却到不了对端 | `./run_gpu_test.sh` 看第 3 部分的数据校验，不对就设 `GPU_P2P=0`（见「卡间 P2P 不可用时」） |
| 本机 GPU 测试：卡间带宽超过 PCIe 链路上限，数据全部校验失败 | 消费级卡的 P2P 补丁驱动，BAR1 比显存小 | 设 `GPU_P2P=0` |
| 第一个请求时报 `PP stage ending at layer N must forward DSA topk_indices` | 第二级从 DSA indexer 为 shared 的层开始（如 39/39 均分） | 按 preflight 的提示改 `PP_LAYER_PARTITION`（如 `42,36`） |
| 节点1 一直等待，或初始化超时 | 端口没放通，或 `NET_IFNAME` / `NODE0_ADDR` 写错 | 两台之间放通 TCP；在节点1上 `ping -I <网卡> <节点0 IP>` |
| 输出乱码，或只生成一个 token | FP8 KV 没走 SM120 的 GLM 专用路径：日志里没有 `Set DSA backends for GLM FP8 KV Cache on SM120/SM121`（SGLang 版本太旧） | 升级 SGLang；或改成 `ATTN_MODE=dense` + `KV_CACHE_DTYPE=bf16` |
| 报 DeepGEMM 相关的 import 错误或 NameError | 走到了 DeepGEMM 代码路径 | 确认环境变量没被覆盖，必要时换 SM120 镜像 |
| 报 tilelang 或 shared memory 相关错误 | DSA 选到了 SM120 跑不了的 kernel（SGLang 版本太旧，或 DSA 配了 BF16 KV） | 用 `KV_CACHE_DTYPE=fp8_e4m3`；还不行就 `ATTN_MODE=dense` + `KV_CACHE_DTYPE=bf16`，或换 SM120 镜像 |
| 多轮对话的后续轮次首 token 很慢 | `ROUTER=sglang` 且 `ATTN_DP_SIZE>1`：每组各有一份 prefix cache，SGLang 不按会话分组，多半要重新 prefill；`ATTN_DP_SIZE=1` 时则是会话太多、KV 池装不下。看 `sglang:prefill_effective_tokens_total`：`device_hit` 加 `host_hit` 的占比明显低于平时就是这个原因（`input`、`device_hit`、`host_hit` 三项互不重叠，加起来是全部输入；`ROUTER=dynamo` 时这个指标在 `HTTP_PORT+1` 的 `/metrics`） | 分组时用默认的 `ROUTER=dynamo`；`ATTN_DP_SIZE=1` 会话不多时 `MAX_RUNNING` 降到 256 以内，会话多时加 `HICACHE_RATIO=2` |
| `ROUTER=dynamo` 时，同一会话的后续轮次仍然要重算 | 客户端没带会话头，新一轮又被负载挤到了别的组；或会话空闲超过 `SESSION_TTL`，固定已经解除 | 请求带上 `X-Session-ID`（或 `X-Dynamo-Session-ID`）头，Claude Code 不用改；没法带头时确认 `ROUTER_PREFILL_WEIGHT` 没有调小（默认 8）；会话间隔长就调大 `SESSION_TTL` |
| `ROUTER=dynamo`：前端已经起来，`/v1/models` 却一直是空的 | 节点0 的 SGLang 还在加载，或者启动失败了（前端只列出已经注册的模型） | 看 `logs/glm53_node0_*.log`；加载完成、worker 注册之后模型就会出现 |
| `ROUTER=dynamo`：服务自己停了，终端里有「Dynamo 前端退出了」 | 前端进程退出了，比如端口被占用 | 看提示里的前端日志 `logs/dynamo_frontend_*.log` |
| `ROUTER=dynamo`：找不到 SGLang 的 `/metrics`、`/flush_cache` | 节点0 没有 SGLang 自己的 HTTP 服务 | 指标在 `http://<节点0>:<HTTP_PORT+1>/metrics`；清前缀缓存用 `POST http://<节点0>:<HTTP_PORT+1>/engine/flush_cache` |
| 日志里频繁出现 retract，吞吐上不去 | KV 池满了，SGLang 把请求换出、之后重算 | 降低 `MAX_RUNNING`；`ATTN_DP_SIZE=1` 时可以改成 8（KV 池约为原来的 6 倍） |
| `pcie_allreduce.cu` 处崩溃 | 设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | 启动脚本已经 unset 这个变量，检查是否有别的地方重新设置了 |
| 工具调用出现在 content 里，而不是 tool_calls | 解析器不对 | `ROUTER=sglang` 要用 `--tool-call-parser glm47`，`ROUTER=dynamo` 要用前端的 `--dyn-tool-call-parser glm47`（启动脚本都已设置） |
| 裸机：ssh 登录后 memlock 仍不是 unlimited | limits.d 只对新的登录会话生效 | 重新登录。`deploy.sh` 也会尝试把当前会话调到 unlimited，但不能超过硬限制 |
| 本机 GPU 测试：某张卡主机 ↔ GPU 带宽偏低 | PCIe 链路降速（插槽、转接线、BIOS 里的速率设置），或 IOMMU 不是 passthrough | 报告里「实测链路」低于「PCIe 应为」时查硬件和 BIOS；链路正常就运行 `sudo ./host_setup.sh` 看 IOMMU |
| 本机 GPU 测试：卡对不能 P2P、带宽明显偏低，或数据校验出错 | PCIe ACS 开着、IOMMU 拦截，或平台不支持跨 CPU 的 P2P | 运行 `sudo ./host_setup.sh` 查看 ACS 和 IOMMU，按 0b 第 2 步改 BIOS |
| `deploy.sh`：一台停在 FAIL，另一台卡在测试或启动 | 另一台在等它会合；或两台的选项不一致（比如只有一台加了 `--skip-nccl-test`） | 在卡住的那台 Ctrl-C，解决问题后两台用相同的选项一起重新运行 |
| 裸机：两台的环境指纹不一致 | 两台分别从网上安装，版本漂移了 | 删掉第二台的 `PY_ENV`，用第一台生成的 `requirements.lock` 重新运行 `install_env.sh` |
| 裸机：首次启动长时间卡在编译 / 报找不到 nvcc | flashinfer 在 SM120 上做 JIT 编译 | 装 CUDA Toolkit 并设置 `CUDA_HOME`；编译缓存在 `~/.cache/flashinfer`，第二次启动就快了 |
| 裸机：perftest 带宽正常，NCCL 带宽很低 | 没用上 GPUDirect（ACS/IOMMU 拦截，或 peermem/DMA-BUF 不可用） | 运行 `sudo ./host_setup.sh` 查看 ACS 和 IOMMU 状态 |
