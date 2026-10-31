#!/usr/bin/env bash
# =====================================================================
# 上线前自检。两台机器各跑一次：./preflight.sh
# 只读检查，不改系统。FAIL 必须先解决，WARN 视情况处理。
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

NFAIL=0
NWARN=0
bad()     { fail "$*"; NFAIL=$((NFAIL + 1)); }
meh()     { warn "$*"; NWARN=$((NWARN + 1)); }
section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------
section "GPU"
GPU_MEM_MIB=0
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,name,compute_cap,memory.total \
                                 --format=csv,noheader,nounits)
  if [[ ${#GPUS[@]} -eq 8 ]]; then ok "GPU 数量 8"; else bad "GPU 数量 ${#GPUS[@]}（需要 8）"; fi
  for g in "${GPUS[@]}"; do
    IFS=',' read -r idx name cap mem <<<"$g"
    name="$(xargs <<<"$name")"; cap="$(xargs <<<"$cap")"; mem="$(xargs <<<"$mem")"
    line="GPU$idx $name  SM ${cap/./}  ${mem} MiB"
    if [[ $cap != 12.0 ]]; then
      meh "$line —— 不是 SM120，这套脚本里的 SM120 绕行设置可能不需要"
    elif [[ $name != *"RTX PRO 6000"* ]]; then
      meh "$line —— 型号不是 RTX PRO 6000"
    elif ((mem < 90000)); then
      meh "$line —— 显存不足 96GB（如 6000D 的 84GB），KV 预算会明显变小"
    else
      ok "$line"
    fi
    if ((GPU_MEM_MIB == 0 || mem < GPU_MEM_MIB)); then GPU_MEM_MIB=$mem; fi
  done
  info "驱动版本：$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  if ((busy > 0)); then meh "有 $busy 个进程正在占用 GPU（nvidia-smi 查看），启动前请停掉"; fi
  if nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | grep -qv Enabled; then
    meh "部分 GPU 没开 persistence 模式（裸机运行 host_setup.sh 处理）"
  fi
  if grep -q "Open Kernel Module" /proc/driver/nvidia/version 2>/dev/null; then
    OPEN_KM=1
    ok "使用开源内核模块"
  fi
else
  bad "nvidia-smi 不可用或看不到 GPU（容器需要 --gpus all）"
fi
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  meh "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES，启动前请 unset"
fi

# ---------------------------------------------------------------------
section "卡间 P2P（GPU_P2P=${GPU_P2P:-1}）"
if [[ ${GPU_P2P:-1} == 0 ]]; then
  info "不用 P2P：NCCL 机内走主机共享内存，SGLang 关掉自定义 all-reduce"
else
  # 驱动说能 P2P、拷贝也不报错，数据却可能是错的（如 P2P 补丁驱动 + BAR1 小于显存的消费级卡）。
  # NCCL 和 SGLang 都不会察觉，表现为通信卡住或输出乱码，所以这里实际拷一次比对
  p2p_out="$(timeout 120 python3 - 2>&1 <<'PY'
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    import torch
except Exception as e:
    print(f"SKIP:导入 torch 失败（{type(e).__name__}），没有检查")
    raise SystemExit
n = torch.cuda.device_count()
if n < 2:
    print("SKIP:GPU 少于 2 张，没有检查")
    raise SystemExit
bad, nop2p = [], []
for s, d in dict.fromkeys([(0, 1), (0, n // 2), (n - 1, 0)]):  # 同一 CPU 下、跨 CPU、反方向
    if not torch.cuda.can_device_access_peer(s, d):
        nop2p.append(f"GPU{s}→GPU{d}")
        continue
    a = torch.randint(-2**31, 2**31 - 1, (16 << 20,), dtype=torch.int32, device=f"cuda:{s}")
    b = torch.empty_like(a, device=f"cuda:{d}")
    b.copy_(a)  # 能 P2P 时 torch 直接走 P2P
    torch.cuda.synchronize(d)
    if not torch.equal(a.cpu(), b.cpu()):
        bad.append(f"GPU{s}→GPU{d}")
if bad:
    print("FAIL:" + "、".join(bad) + " 的 P2P 拷贝数据不对：在 cluster.env 设 GPU_P2P=0（./run_gpu_test.sh 可以看全部卡对）")
elif nop2p:
    print("WARN:" + "、".join(nop2p) + " 不能 P2P（NCCL 会自动改走主机内存）；检查 ACS / IOMMU，或设 GPU_P2P=0")
else:
    print("OK:抽查的卡对 P2P 拷贝数据正确")
PY
)"
  p2p_rc=$?
  case $p2p_out in
    *$'\n'*) p2p_last="${p2p_out##*$'\n'}" ;;
    *) p2p_last="$p2p_out" ;;
  esac
  case $p2p_last in
    FAIL:*) bad "${p2p_last#FAIL:}" ;;
    WARN:*) meh "${p2p_last#WARN:}" ;;
    OK:*)   ok "${p2p_last#OK:}" ;;
    SKIP:*) info "${p2p_last#SKIP:}" ;;
    *)
      if ((p2p_rc == 124)); then
        bad "P2P 拷贝 120 秒没完成（P2P 卡死）：在 cluster.env 设 GPU_P2P=0"
      else
        meh "P2P 检查没有跑完：$(tail -3 <<<"$p2p_out" | tr '\n' ' ')"
      fi
      ;;
  esac
fi

# ---------------------------------------------------------------------
section "RDMA（${IB_HCA} 端口 ${IB_PORT:-1}）"
P="$(ib_sys)"
LL="$(rdma_link_layer)"
if [[ ! -d $P ]]; then
  devs=""
  for d in /sys/class/infiniband/*; do [[ -e $d ]] && devs+="$(basename "$d") "; done
  bad "找不到 $P。本机 RDMA 设备：${devs:-无}（容器需要 --device /dev/infiniband）"
else
  state="$(cat "$P/state" 2>/dev/null)"
  phys="$(cat "$P/phys_state" 2>/dev/null)"
  rate="$(cat "$P/rate" 2>/dev/null)"
  if [[ $state == *ACTIVE* ]]; then ok "端口状态 $state"; else bad "端口状态 ${state:-未知}（需要 ACTIVE）"; fi
  if [[ $phys == *LinkUp* ]]; then ok "物理链路 $phys"; else bad "物理链路 ${phys:-未知}（需要 LinkUp）"; fi
  rate_num="$(grep -oE '^[0-9]+' <<<"$rate" || echo 0)"
  if ((rate_num >= 400)); then ok "速率 $rate"; else meh "速率 ${rate:-未知}（预期 400 Gb/sec）"; fi
  info "链路层：$LL"

  if [[ $LL == Ethernet ]]; then
    want="${IB_GID_INDEX:-3}"
    v2=()
    for t in "$P"/gid_attrs/types/*; do
      i="$(basename "$t")"
      typ="$(cat "$t" 2>/dev/null)" || continue
      gid="$(cat "$P/gids/$i" 2>/dev/null)" || continue
      [[ $typ == *"RoCE v2"* ]] || continue
      [[ $gid == 0000:0000:0000:0000:0000:0000:0000:0000 ]] && continue
      nd="$(cat "$P/gid_attrs/ndevs/$i" 2>/dev/null)"
      mark=""
      [[ $gid == 0000:0000:0000:0000:0000:ffff:* ]] && mark="  ← IPv4，一般选这个"
      info "  RoCE v2 GID index $i  $gid  ${nd}${mark}"
      v2+=("$i")
    done
    if ((${#v2[@]} == 0)); then
      bad "没有可用的 RoCE v2 GID（网卡没配 IP？）"
    elif [[ " ${v2[*]} " == *" $want "* ]]; then
      ok "NCCL_IB_GID_INDEX=$want 是 RoCE v2"
    else
      bad "NCCL_IB_GID_INDEX=$want 不在上面的列表里，请把 cluster.env 的 IB_GID_INDEX 改成标了 IPv4 的那个"
    fi
  fi

  nds=""
  for d in "/sys/class/infiniband/$IB_HCA/device/net"/*; do [[ -e $d ]] && nds+="$(basename "$d") "; done
  info "该 RDMA 设备对应的网卡：${nds:-未知}"
  if [[ -n $nds && " $nds " != *" $NET_IFNAME "* ]]; then
    meh "NET_IFNAME=$NET_IFNAME 不是这块 RDMA 卡的网口（只用于建连，可以不同，但建议一致）"
  fi
fi
if [[ ! -d /dev/infiniband ]]; then
  bad "看不到 /dev/infiniband（容器需要 --device /dev/infiniband --cap-add IPC_LOCK）"
fi

# ---------------------------------------------------------------------
section "网络接口与连通性（$NET_IFNAME）"
if have ip && ip -4 -o addr show dev "$NET_IFNAME" >/dev/null 2>&1; then
  myip="$(ip -4 -o addr show dev "$NET_IFNAME" | awk '{print $4}' | cut -d/ -f1 | head -1)"
  if [[ -z $myip ]]; then
    bad "$NET_IFNAME 没有 IPv4 地址"
  else
    if [[ $myip == "$NODE0_ADDR" ]]; then
      ok "$NET_IFNAME = $myip，本机是节点0"
      peer="$NODE1_ADDR"
    elif [[ $myip == "$NODE1_ADDR" ]]; then
      ok "$NET_IFNAME = $myip，本机是节点1"
      peer="$NODE0_ADDR"
    else
      meh "$NET_IFNAME = $myip，既不是 NODE0_ADDR 也不是 NODE1_ADDR，检查 cluster.env"
      peer="$NODE0_ADDR"
    fi
    if have ping; then
      if ping -c 2 -W 1 -I "$NET_IFNAME" "$peer" >/dev/null 2>&1; then
        ok "能 ping 通对端（$peer）"
      else
        bad "ping 不通对端（$peer）"
      fi
    fi
  fi
  mtu="$(cat "/sys/class/net/$NET_IFNAME/mtu" 2>/dev/null)"
  if [[ $LL == Ethernet && -n $mtu ]] && ((mtu < 4200)); then
    meh "RoCE 网口 MTU=$mtu，建议 9000（交换机同步调整）"
  else
    info "MTU=${mtu:-未知}"
  fi
else
  bad "找不到网卡 $NET_IFNAME 或 ip 命令不可用"
fi

# ---------------------------------------------------------------------
section "驱动模块与系统限制"
kmaj="$(uname -r | cut -d. -f1)"; kmin="$(uname -r | cut -d. -f2)"
if [[ -d /sys/module/nvidia_peermem || -d /sys/module/nv_peer_mem ]]; then
  ok "nvidia-peermem 已加载（GPUDirect RDMA）"
elif [[ ${OPEN_KM:-0} == 1 ]] && ((kmaj > 5 || (kmaj == 5 && kmin >= 12))); then
  info "未加载 nvidia-peermem；开源内核模块 + 内核 $(uname -r) 下 NCCL 可以走 DMA-BUF，以 NCCL 测试里是否出现 GDRDMA 为准"
else
  meh "未加载 nvidia-peermem（宿主机执行 modprobe nvidia-peermem，或运行 host_setup.sh）"
fi
# 116 = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED；为 0 时 NCCL 不会用 GDR，peermem 加载了也一样
gdr="$(python3 -c '
import ctypes
cu = ctypes.CDLL("libcuda.so.1"); cu.cuInit(0); v = ctypes.c_int()
print(v.value if cu.cuDeviceGetAttribute(ctypes.byref(v), 116, 0) == 0 else "?")' 2>/dev/null || true)"
if [[ $gdr == 0 ]]; then
  info "CUDA 报告 GPU 不支持 GPUDirect RDMA（GeForce 卡就是这样）：NCCL 跨机数据经主机内存中转，带宽低一些，PP 够用"
fi
ml="$(ulimit -l)"
if [[ $ml == unlimited ]]; then
  ok "memlock = unlimited"
else
  bad "memlock = $ml，RDMA 注册内存会失败（容器加 --ulimit memlock=-1；裸机运行 sudo ./host_setup.sh --apply 后重新登录）"
fi
if [[ ! -f /.dockerenv ]]; then
  info "裸机部署：宿主机调优（IOMMU、ACS、NUMA、调频、防火墙）请运行 ./host_setup.sh 查看"
fi

# ---------------------------------------------------------------------
section "GPU 与网卡的 PCIe 距离"
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  topo="$(nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')"
  printf '%s\n' "$topo"
  nic="$(awk -v h="$IB_HCA" '$1 ~ /^NIC[0-9]+:$/ && $2 == h { sub(":", "", $1); print $1 }' <<<"$topo")"
  if [[ -n $nic ]]; then
    rel="$(awk -F'\t' -v n="$nic" '
      { k = $1; gsub(/ /, "", k) }
      k == n { for (i = 2; i <= 9; i++) { v = $i; gsub(/ /, "", v); printf "GPU%d:%s ", i - 2, v }; print "" }' <<<"$topo")"
    info "$IB_HCA（$nic）到各 GPU：$rel"
    near="$(grep -oE 'GPU[0-9]:(PIX|PXB)' <<<"$rel" | wc -l)"
    if ((near == 0)); then
      meh "网卡与所有 GPU 之间都要经过 CPU（NODE/SYS），NCCL 可能不启用 GPUDirect RDMA。PP 只需要几十 GB/s，通常仍够用，以 NCCL 测试结果为准"
    else
      ok "$near 张 GPU 与网卡在同一 PCIe 交换下（PIX/PXB），可走 GPUDirect RDMA"
    fi
  else
    meh "topo 图里没找到 $IB_HCA"
  fi
else
  info "跳过（没有 GPU）"
fi

# ---------------------------------------------------------------------
section "Python / SGLang 环境"
python3 - <<'PY'
import importlib
for m in ("torch", "sglang", "transformers", "flashinfer", "triton"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
    except Exception as e:
        v = f"未安装（{type(e).__name__}）"
    print(f"    {m:13s} {v}")
try:
    import torch
    if torch.cuda.is_available():
        nv = torch.cuda.nccl.version()
        nv = ".".join(map(str, nv)) if isinstance(nv, tuple) else str(nv)
        print(f"    CUDA {torch.version.cuda} · NCCL {nv} · capability {torch.cuda.get_device_capability(0)}")
    else:
        print("    torch 看不到 GPU")
except Exception as e:
    print(f"    torch 检查失败：{e}")
PY
info "Python：$(command -v python3)"
info "环境指纹：$(env_fingerprint) —— 两台机器必须一致"
if command -v nvcc >/dev/null; then
  info "nvcc：$(nvcc --version | grep -oE 'release [0-9.]+' | head -1)（应与上面 torch 的 CUDA 主版本一致）"
else
  meh "找不到 nvcc：flashinfer 在 SM120 上首次运行可能要 JIT 编译。装对应版本的 CUDA Toolkit，并在 cluster.env 设置 CUDA_HOME"
fi
if HELP="$(python3 -m sglang.launch_server --help 2>/dev/null)"; then
  for f in --pp-size --nnodes --dist-init-addr --kv-cache-dtype --attention-backend \
           --moe-runner-backend --disable-cuda-graph --disable-custom-all-reduce \
           --fp8-gemm-backend --enable-pcie-oneshot-allreduce --speculative-algorithm; do
    if flag_in "$f" "$HELP"; then
      ok "支持 $f"
    else
      case $f in
        # 启动脚本对这几个参数做了降级处理，其余都是直接传入
        --fp8-gemm-backend) meh "不支持 $f（启动脚本会跳过）" ;;
        --enable-pcie-oneshot-allreduce)
          if [[ ${GPU_P2P:-1} == 0 ]]; then
            info "不支持 $f（GPU_P2P=0 时本来就不用）"
          else
            meh "不支持 $f（启动脚本会换成 --disable-custom-all-reduce）"
          fi
          ;;
        --disable-custom-all-reduce)  # GPU_P2P=0 或没有 PCIe oneshot all-reduce 时才传
          if [[ ${GPU_P2P:-1} == 0 ]] || ! flag_in --enable-pcie-oneshot-allreduce "$HELP"; then
            bad "不支持 $f，请升级 SGLang"
          else
            info "不支持 $f（启动脚本用 --enable-pcie-oneshot-allreduce，用不到它）"
          fi
          ;;
        --speculative-algorithm)
          if [[ ${ENABLE_MTP:-0} == 1 ]]; then
            bad "不支持 $f，ENABLE_MTP=1 时启动会失败"
          else
            meh "不支持 $f（只在 ENABLE_MTP=1 时需要）"
          fi
          ;;
        *) bad "不支持 $f，请升级 SGLang" ;;
      esac
    fi
  done
  if flag_in --cuda-graph-max-bs-decode "$HELP"; then
    ok "支持 --cuda-graph-max-bs-decode"
  elif flag_in --cuda-graph-max-bs "$HELP"; then
    ok "支持 --cuda-graph-max-bs"
  else
    meh "不支持 --cuda-graph-max-bs-decode / --cuda-graph-max-bs（serve 模式下 CUDA_GRAPH_MAX_BS 不生效）"
  fi
  if [[ ${ATTN_DP_SIZE:-1} =~ ^[248]$ ]]; then  # 启动脚本优先用 --attn-dp-size，没有时降级成旧写法
    if flag_in --attn-dp-size "$HELP"; then
      ok "支持 --attn-dp-size"
    elif flag_in --enable-dp-attention "$HELP"; then
      meh "不支持 --attn-dp-size（启动脚本会改用旧写法 --dp-size --enable-dp-attention）"
    else
      bad "不支持 DP attention（--attn-dp-size / --enable-dp-attention），把 ATTN_DP_SIZE 设成 1 或升级 SGLang"
    fi
  fi
else
  bad "python3 -m sglang.launch_server --help 运行失败"
fi

# ---------------------------------------------------------------------
section "serve 参数（cluster.env）"
attn_mode="${ATTN_MODE:-dsa}" kv_dtype="${KV_CACHE_DTYPE:-fp8_e4m3}" attn_dp="${ATTN_DP_SIZE:-1}"
max_run="${MAX_RUNNING:-}" graph_bs="${CUDA_GRAPH_MAX_BS:-}"
[[ $max_run =~ ^[1-9][0-9]*$ ]] || { bad "MAX_RUNNING 要是正整数，现在是 ${max_run:-空}"; max_run=0; }
case $attn_mode in
  dsa | dense) ok "ATTN_MODE=$attn_mode" ;;
  *) bad "ATTN_MODE 只能是 dsa 或 dense，现在是 $attn_mode" ;;
esac
case $kv_dtype in
  fp8_e4m3 | bf16) ok "KV_CACHE_DTYPE=$kv_dtype" ;;
  *) bad "KV_CACHE_DTYPE 只能是 fp8_e4m3 或 bf16，现在是 $kv_dtype" ;;
esac
if [[ $attn_mode == dense && $kv_dtype == fp8_e4m3 ]]; then
  bad "ATTN_MODE=dense 只能配 KV_CACHE_DTYPE=bf16（SM120 上稠密注意力配 FP8 KV 会输出乱码）；要用 FP8 KV 请设 ATTN_MODE=dsa"
fi
case $attn_dp in
  1) ok "ATTN_DP_SIZE=1：attention 按 TP8 切，所有请求共用一份 prefix cache" ;;
  2 | 4 | 8)
    ok "ATTN_DP_SIZE=${attn_dp}：attention 分 ${attn_dp} 组，每组最多 $((max_run / attn_dp)) 个请求，各有一份 prefix cache"
    if ((max_run % attn_dp)); then
      meh "MAX_RUNNING=${max_run} 不是 ${attn_dp} 的倍数，每组按 $((max_run / attn_dp)) 个算"
    fi
    if [[ ${ENABLE_MTP:-0} == 1 ]]; then
      bad "ENABLE_MTP=1 不能和 ATTN_DP_SIZE>1 同时用（SGLang 的 PP 投机解码不支持 DP attention）"
    fi
    ;;
  *) bad "ATTN_DP_SIZE 只能是 1、2、4、8（要能整除 TP8），现在是 ${attn_dp}" ;;
esac
if [[ $attn_dp =~ ^[1248]$ && $graph_bs =~ ^[0-9]+$ ]] && ((graph_bs < max_run / attn_dp)); then
  meh "CUDA_GRAPH_MAX_BS=${graph_bs} 小于每组的请求上限 $((max_run / attn_dp))：batch 超过它时 decode 不走 CUDA Graph，会变慢"
fi

# ---------------------------------------------------------------------
section "模型（$MODEL_PATH）"
if [[ -f $MODEL_PATH/config.json ]]; then
  ok "config.json 存在"
  while IFS= read -r line; do
    case $line in
      FAIL:*) bad "${line#FAIL:}" ;;
      WARN:*) meh "${line#WARN:}" ;;
      OK:*)   ok "${line#OK:}" ;;
      *)      info "$line" ;;
    esac
  done < <(
    MODEL_PATH="$MODEL_PATH" GPU_MEM_MIB="$GPU_MEM_MIB" MEM_FRACTION="$MEM_FRACTION" \
    PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-}" KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}" \
    ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}" python3 - <<'PY'
import glob, json, os
mp = os.environ["MODEL_PATH"]
cfg = json.load(open(os.path.join(mp, "config.json")))
tc = cfg.get("text_config", cfg)
arch = ",".join(cfg.get("architectures") or ["?"])
L = int(tc.get("num_hidden_layers") or 0)
H = int(tc.get("hidden_size") or 0)
kv_dim = int(tc.get("kv_lora_rank") or 0) + int(tc.get("qk_rope_head_dim") or 0)
q = cfg.get("quantization_config") or tc.get("quantization_config") or {}
qm = q.get("quant_method") or q.get("quant_algo") or "无"
print(f"架构 {arch} · {L} 层 · hidden {H} · MLA KV 维度 {kv_dim} · 量化 {qm} · MTP 层 {tc.get('num_nextn_predict_layers', '?')}")
print(("OK:" if "GlmMoeDsa" in arch else "WARN:") + f"架构 {arch}")
print(("OK:" if str(qm).lower().startswith("fp8") else "WARN:") + f"量化方式 {qm}（本方案按 FP8 写）")

idx_path = os.path.join(mp, "model.safetensors.index.json")
files = sorted(glob.glob(os.path.join(mp, "*.safetensors")))
if os.path.exists(idx_path):
    need = set(json.load(open(idx_path))["weight_map"].values())
    missing = [f for f in need if not os.path.exists(os.path.join(mp, f))]
    if missing:
        print(f"FAIL:缺 {len(missing)} 个权重分片，例如 {missing[0]}（下载未完成？）")
    else:
        print(f"OK:权重分片齐全（{len(need)} 个）")
else:
    print("WARN:没有 model.safetensors.index.json")
total = sum(os.path.getsize(f) for f in files)
print(f"权重总大小 {total / 2**30:.0f} GiB")

if L and kv_dim and total:
    part = os.environ.get("PP_LAYER_PARTITION", "").strip()
    try:  # 写错时下面的「层划分」会报 FAIL，这里按均分估算
        stages = [int(x) for x in part.split(",")] if part else []
    except ValueError:
        stages = []
    if len(stages) != 2 or sum(stages) != L:
        stages = [L // 2, L - L // 2]
    mem_mib = float(os.environ.get("GPU_MEM_MIB") or 0) or 97887.0  # 96GB 卡的典型值
    frac = float(os.environ.get("MEM_FRACTION") or 0.85)
    fp8 = os.environ.get("KV_CACHE_DTYPE") == "fp8_e4m3"
    dp = os.environ.get("ATTN_DP_SIZE", "1")
    dp = int(dp) if dp in ("1", "2", "4", "8") else 1
    kv_lora = int(tc.get("kv_lora_rank") or 0)
    rope = int(tc.get("qk_rope_head_dim") or 0)
    # 每层每 token：BF16 存 latent + rope；FP8 存 FP8 latent、每 128 维一个 FP32 scale、BF16 rope（SGLang DSA 的布局）
    per_layer = kv_lora + kv_lora // 128 * 4 + rope * 2 if fp8 else kv_dim * 2
    # DSA indexer 的 K cache 只在 full 层分配：FP8 K + 一个 FP32 scale
    types = tc.get("indexer_types") or []
    idx_bytes = int(tc.get("index_head_dim") or 0) + 4
    bounds = [(0, stages[0]), (stages[0], L)]
    per_tok = [(hi - lo) * per_layer + sum(t == "full" for t in types[lo:hi]) * idx_bytes
               for lo, hi in bounds]
    # attention 每组 8/dp 张卡：attention 权重（FP8）和词表（BF16）只在组内切分，每卡比 TP8 多放一些
    nh = int(tc.get("num_attention_heads") or 0)
    q_lora = int(tc.get("q_lora_rank") or 0)
    nope = int(tc.get("qk_nope_head_dim") or 0)
    vd = int(tc.get("v_head_dim") or 0)
    attn_layer = (H * q_lora + q_lora * nh * (nope + rope) + H * (kv_lora + rope)
                  + kv_lora * nh * (nope + vd) + nh * vd * H)
    share = dp / 8 - 1 / 8  # 每卡多放的比例：组内 8/dp 张卡分，TP8 下是 8 张
    extra = [(n * attn_layer + int(tc.get("vocab_size") or 0) * H * 2) * share for n in stages]
    w_gpu = [total * n / L / 8 for n in stages]  # 每级按层数分到 8 张卡（MTP 层不加载，算多了一点）
    budget = [mem_mib * 2**20 * frac - w - e for w, e in zip(w_gpu, extra)]
    toks = min(b / t for b, t in zip(budget, per_tok))  # 两级的 token 数取小的那个
    print(f"层划分 {stages[0]}/{stages[1]} · KV {'FP8' if fp8 else 'BF16'} · 每卡权重约 "
          f"{max(w + e for w, e in zip(w_gpu, extra)) / 2**30:.0f} GiB · KV 预算约 {min(budget) / 2**30:.0f} GiB · "
          f"每 token KV {max(per_tok) / 1024:.1f} KiB")
    if min(budget) <= 0:
        print("FAIL:按当前 MEM_FRACTION 放不下权重")
    elif dp == 1:
        print(f"粗估 KV 池约 {toks / 1e4:.0f} 万 token（所有请求共享；"
              f"以启动日志的 max_total_num_tokens 为准，激活和 CUDA Graph 还会再占一些）")
    else:
        print(f"粗估 KV 池每组约 {toks / 1e4:.0f} 万 token × {dp} 组 = {toks * dp / 1e4:.0f} 万 token"
              f"（单个请求只能用所在那组的；attention 权重每卡多占约 {max(extra) / 2**30:.1f} GiB；"
              f"以启动日志为准）")
if H:
    print(f"PP 边界每 token 约 {H * 2 * 2 / 1024:.0f} KiB（BF16 hidden + residual）")
PY
  )
  split="$(pp_split_check)"
  case $split in
    OK:*)   ok "层划分：${split#OK:}" ;;
    WARN:*) meh "层划分：${split#WARN:}" ;;
    *)      bad "层划分：${split#FAIL:}" ;;
  esac
  if have df; then info "磁盘：$(df -h "$MODEL_PATH" | awk 'NR==2{print $1" 总 "$2"，可用 "$4}')"; fi
else
  bad "找不到 $MODEL_PATH/config.json（两台机器要放在同一路径）"
fi

# ---------------------------------------------------------------------
section "主机"
if have free; then info "内存：$(free -g | awk '/^Mem:/{print $2" GB 总量，"$7" GB 可用"}')"; fi
if have lscpu; then info "$(lscpu | grep -E '^NUMA node\(s\)' | xargs)"; fi
info "CPU 核数：$(nproc)"

# ---------------------------------------------------------------------
section "结果"
if ((NFAIL)); then
  fail "$NFAIL 项 FAIL，$NWARN 项 WARN —— 先解决 FAIL"
  exit 1
elif ((NWARN)); then
  warn "0 项 FAIL，$NWARN 项 WARN"
else
  ok "全部通过"
fi
